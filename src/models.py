# -*- coding: utf-8 -*-
from __future__ import division
from functools import partial
import warnings

import qinfer as qi

import numpy as np
import scipy.io as sio
from scipy.linalg import expm
from scipy.special import xlogy, gammaln
from scipy.interpolate import CubicSpline
from vexpm import vexpm
from vexpm import matmul
from fractions import gcd as fgcd

from glob import glob

## CONSTANTS ##################################################################

# Construct Relevant operators

Si = np.eye(3)
Sx = np.array([[0,1,0],[1,0,1],[0,1,0]])/np.sqrt(2)
Syp = np.array([[0,-1j,0],[1j,0,1j],[0,-1j,0]])/np.sqrt(2)
Sz = np.array([[1,0,0],[0,0,0],[0,0,-1]])
Sz2 = np.dot(Sz, Sz)

two_pi = np.pi * 2
two_pii = two_pi*1j
two_pii_s2 = two_pii/np.sqrt(2)
four_pi = two_pi * 2

P0 = np.zeros((3, 3), dtype=complex)
P0[1, 1] = 1
P0vec = P0.flatten(order='F')

HX = 2 * np.pi * Sx
GX = 1j *(np.kron(HX.T, np.eye(3)) - np.kron(np.eye(3), HX))
HY = 2 * np.pi * Syp
GY = 1j *(np.kron(HY.T, np.eye(3)) - np.kron(np.eye(3), HY))
HZ = 2 * np.pi * Sz
GZ = 1j *(np.kron(HZ.T, np.eye(3)) - np.kron(np.eye(3), HZ))
HZ2 = 2 * np.pi * Sz2
GZ2 = 1j *(np.kron(HZ2.T, np.eye(3)) - np.kron(np.eye(3), HZ2))
Gi = np.eye(9)

class RepeatedOperator(object):
    def __init__(self, op):
        self.op = op
        self._memo = {}
        
    def _construct_op(self, size):
        return np.repeat(self.op[np.newaxis, ...], size, axis=0)
    
    def __call__(self, size=1):
        if size not in self._memo:
            self._memo[size] = self._construct_op(size)
        return self._memo[size]
            
vecGx = RepeatedOperator(GX)
vecGy = RepeatedOperator(GY)
vecGz = RepeatedOperator(GZ)
vecGz2 = RepeatedOperator(GZ2)   
vecSx = RepeatedOperator(-1j * 2 * np.pi * Sx)
vecSyp = RepeatedOperator(-1j * 2 * np.pi * Syp)
vecSz = RepeatedOperator(-1j * 2 * np.pi * Sz)
vecSz2 = RepeatedOperator(-1j * 2 * np.pi * Sz2)   
vecLz = RepeatedOperator((np.kron(Sz, Sz) - (np.kron(Sz2, Si) + np.kron(Si, Sz2)) / 2))
vecPhasePlus = RepeatedOperator(np.array([0,1,0,0,0,0,0,1,0]))
vecPhaseMinus = RepeatedOperator(np.array([0,0,0,1,0,1,0,0,0]))
vecPhaseNone = RepeatedOperator(np.array([1,0,1,0,1,0,1,0,1]))

## FUNCTIONS ###################################################################

def poisson_pdf(k, mu):
    """
    Probability of k events in a poisson process of expectation mu
    """
    return np.exp(xlogy(k, mu) - gammaln(k + 1) - mu)

ufgcd = np.frompyfunc(fgcd, 2, 1)
def gcd(m):
    """Return the greatest common divisor of the given integers"""
    return np.ufunc.reduce(ufgcd, np.round(m).astype(np.int))

## SIMULATORS ##################################################################

class CachedPropagator(object):
    
    def __init__(self, generator, base_timestep=1, max_expected_time=4):
        self.base_timestep = base_timestep
        self.generator = generator
        self.shape = self.generator.shape
        
        # we are careful to choose a pade_order and scale_power that matches the
        # the expected max size of the norm of generator, rather than just 
        # at base_timestep
        norm_multiplier = max_expected_time / base_timestep
        self._propagator = {(1,): vexpm(base_timestep * generator, norm_multiplier=norm_multiplier)}
    
    def propagator(self, bin_expansion):
        """
        Returns expm(base_timestep * generator) ** m where 
        the binary expansion of m is a tuple given by bin_expansion.
        """
        # leading zeros do not contribute to the multiplier
        if bin_expansion[0] == 0:
            return self.propagator(bin_expansion[1:])
        
        # otherwise look to see if we have already computed this multiplier    
        if not self._propagator.has_key(bin_expansion):
            if not any(bin_expansion[1:]):
                # in this case we have a power of 2, ex, bin_expansion = (1,0,0)
                self._propagator[bin_expansion] = matmul(
                        self.propagator(bin_expansion[:-1]), 
                        self.propagator(bin_expansion[:-1])
                    )
            else:
                # recursion step
                self._propagator[bin_expansion] = matmul(
                        self.propagator((1,) + (0,)*(len(bin_expansion)-1)), 
                        self.propagator(bin_expansion[1:])
                    )
            
        return self._propagator[bin_expansion]
        
    def __call__(self, time):
        """
        Returns an approximation of expm(time * generator)
        """
        mult = int(np.round(time / self.base_timestep))
        if mult == 0:
            return np.repeat(np.eye(self.shape[-1])[np.newaxis,...],self.shape[0], axis=0)
        base2_expansion = map(lambda x: 0 if x == '0' else 1, np.base_repr(mult))
        return self.propagator(tuple(base2_expansion))
    

def rabi_cached(tp, tau, phi, wr, we, dwc, an, T2inv):
    n_models = wr.shape[0]
    base_timestep = gcd(tp * 1000) / 1000
    # supergenerator without nitrogen
    G1 = (wr[:, np.newaxis, np.newaxis] * vecGx(n_models) \
        + we[:, np.newaxis, np.newaxis] * vecGz(n_models) \
        + dwc[:, np.newaxis, np.newaxis] * vecGz2(n_models) \
        + T2inv[:, np.newaxis, np.newaxis] * vecLz(n_models))
    # supergenerator for just nitrogen
    GA = an[:, np.newaxis, np.newaxis] * vecGz(n_models)

    Sm = CachedPropagator(G1 - GA, base_timestep=base_timestep, max_expected_time=.2)
    S0 = CachedPropagator(G1, base_timestep=base_timestep, max_expected_time=.2)
    Sp = CachedPropagator(G1 + GA, base_timestep=base_timestep, max_expected_time=.2)

    pr0 = np.empty((tp.size, wr.size))
    for idx_t, t in enumerate(tp):
        pr0[idx_t, :]  = np.real(Sm(t)[:, 4, 4] + S0(t)[:, 4, 4] + Sp(t)[:, 4, 4]) / 3

    return pr0
    
def ramsey_cached(tp, tau, phi, wr, we, dwc, an, T2inv):
    """
    Return signal due to Ramsey experiment with
    given parameters
    """
    n_models = wr.shape[0]
    base_timestep_tp = gcd(tp * 1000) / 1000
    base_timestep_tau = gcd(tau * 1000) / 1000
    # hamiltonian without nitrogen during rabi
    H1 = (wr[:, np.newaxis, np.newaxis] * vecSx(n_models) \
        + we[:, np.newaxis, np.newaxis] * vecSz(n_models) \
        + dwc[:, np.newaxis, np.newaxis] * vecSz2(n_models))
    # hamiltonian for just nitrogen
    HA = an[:, np.newaxis, np.newaxis] * vecSz(n_models)
    # supergenerator during wait
    G1 = (we[:, np.newaxis, np.newaxis] * vecGz(n_models) \
        + dwc[:, np.newaxis, np.newaxis] * vecGz2(n_models) \
        + T2inv[:, np.newaxis, np.newaxis] * vecLz(n_models))
    # supergenerator for just nitrogen
    GA = an[:, np.newaxis, np.newaxis] * vecGz(n_models)
    
    Sm = CachedPropagator(G1 - GA, base_timestep=base_timestep_tau, max_expected_time=4)
    S0 = CachedPropagator(G1, base_timestep=base_timestep_tau, max_expected_time=4)
    Sp = CachedPropagator(G1 + GA, base_timestep=base_timestep_tau, max_expected_time=4)
    Um = CachedPropagator(H1 - HA, base_timestep=base_timestep_tp, max_expected_time=0.1)
    U0 = CachedPropagator(H1, base_timestep=base_timestep_tp, max_expected_time=0.1)
    Up = CachedPropagator(H1 + HA, base_timestep=base_timestep_tp, max_expected_time=0.1)
    
    pr0 = np.empty((tp.size, wr.size))
    for idx_t in range(tp.size):
        t1, t2 = tp[idx_t], tau[idx_t]

        # states after square pulses
        sm = Um(t1)[...,1]
        s0 = U0(t1)[...,1]
        sp = Up(t1)[...,1]
        
        # convert to vectorized density matrix
        sm = np.repeat(sm.conj(), 3, axis=-1) * np.reshape(np.tile(sm, 3), (-1, 9))
        sm = sm[...,np.newaxis]
        s0 = np.repeat(s0.conj(), 3, axis=-1) * np.reshape(np.tile(s0, 3), (-1, 9))
        s0 = s0[...,np.newaxis]
        sp = np.repeat(sp.conj(), 3, axis=-1) * np.reshape(np.tile(sp, 3), (-1, 9))
        sp = sp[...,np.newaxis]
        
        # the euler angle qutrit decomposition 
        # exp(a*(sin(phi)Sx + cos(phi)Syp)) = exp(-i*phi*Sz2).exp(a*Sx).exp(i*phi*Sz2)
        # lets us avoid computing the matrix exp of both end pulses; we just need 
        # to put the right phases on the state after the wait period
        #exp_phi_p = np.exp(1j * phi[idx_t])
        #exp_phi_m = exp_phi_p.conj()
        #phases = np.array([[1,exp_phi_p,1,exp_phi_m,1,exp_phi_m,1,exp_phi_p,1]]).T

        # sandwich wait between square pulses
        #p = matmul(sm.swapaxes(-2,-1), phases * matmul(Sm(t2), sm))[...,0,0]
        #p = p + matmul(s0.swapaxes(-2,-1), phases * matmul(S0(t2), s0))[...,0,0]
        #p = p + matmul(sp.swapaxes(-2,-1), phases * matmul(Sp(t2), sp))[...,0,0]
        p = matmul(sm.swapaxes(-2,-1), matmul(Sm(t2), sm))[...,0,0]
        p = p + matmul(s0.swapaxes(-2,-1), matmul(S0(t2), s0))[...,0,0]
        p = p + matmul(sp.swapaxes(-2,-1), matmul(Sp(t2), sp))[...,0,0]
        
        pr0[idx_t, :] = np.real(p) / 3
    return pr0
    
def two_pulse_cached(tp, tp2, tau, phi, wr, we, dwc, an, T2inv):
    """
    Return signal due to Ramsey experiment with
    given parameters. Pulse sequence is tp)_x --- tau --- tp2)_x
    """
    n_models = wr.shape[0]
    base_timestep_tp = gcd(np.concatenate([tp, tp2]) * 1000) / 1000
    base_timestep_tau = gcd(tau * 1000) / 1000
    # hamiltonian without nitrogen during rabi
    H1 = (wr[:, np.newaxis, np.newaxis] * vecSx(n_models) \
        + we[:, np.newaxis, np.newaxis] * vecSz(n_models) \
        + dwc[:, np.newaxis, np.newaxis] * vecSz2(n_models))
    # hamiltonian for just nitrogen
    HA = an[:, np.newaxis, np.newaxis] * vecSz(n_models)
    # supergenerator during wait
    G1 = (we[:, np.newaxis, np.newaxis] * vecGz(n_models) \
        + dwc[:, np.newaxis, np.newaxis] * vecGz2(n_models) \
        + T2inv[:, np.newaxis, np.newaxis] * vecLz(n_models))
    # supergenerator for just nitrogen
    GA = an[:, np.newaxis, np.newaxis] * vecGz(n_models)
    
    Sm = CachedPropagator(G1 - GA, base_timestep=base_timestep_tau, max_expected_time=4)
    S0 = CachedPropagator(G1, base_timestep=base_timestep_tau, max_expected_time=4)
    Sp = CachedPropagator(G1 + GA, base_timestep=base_timestep_tau, max_expected_time=4)
    Um = CachedPropagator(H1 - HA, base_timestep=base_timestep_tp, max_expected_time=0.1)
    U0 = CachedPropagator(H1, base_timestep=base_timestep_tp, max_expected_time=0.1)
    Up = CachedPropagator(H1 + HA, base_timestep=base_timestep_tp, max_expected_time=0.1)
    
    pr0 = np.empty((tp.size, wr.size))
    for idx_t in range(tp.size):
        t1, t2, t3 = tp[idx_t], tau[idx_t], tp2[idx_t]

        # states after square pulses
        sm = Um(t1)[...,1]
        s0 = U0(t1)[...,1]
        sp = Up(t1)[...,1]
        
        # convert to vectorized density matrix
        sm = np.repeat(sm.conj(), 3, axis=-1) * np.reshape(np.tile(sm, 3), (-1, 9))
        sm = sm[...,np.newaxis]
        s0 = np.repeat(s0.conj(), 3, axis=-1) * np.reshape(np.tile(s0, 3), (-1, 9))
        s0 = s0[...,np.newaxis]
        sp = np.repeat(sp.conj(), 3, axis=-1) * np.reshape(np.tile(sp, 3), (-1, 9))
        sp = sp[...,np.newaxis]
        
        # states after second square pulses
        sm2 = Um(t3)[...,1]
        s02 = U0(t3)[...,1]
        sp2 = Up(t3)[...,1]
        
        # convert to vectorized density matrix
        sm2 = np.repeat(sm2, 3, axis=-1) * np.reshape(np.tile(sm2.conj(), 3), (-1, 9))
        sm2 = sm2[...,np.newaxis]
        s02 = np.repeat(s02, 3, axis=-1) * np.reshape(np.tile(s02.conj(), 3), (-1, 9))
        s02 = s02[...,np.newaxis]
        sp2 = np.repeat(sp2, 3, axis=-1) * np.reshape(np.tile(sp2, 3).conj(), (-1, 9))
        sp2 = sp2[...,np.newaxis]
        
        # sandwich wait between square pulses
        p = matmul(sm2.swapaxes(-2,-1), matmul(Sm(t2), sm))[...,0,0]
        p = p + matmul(s02.swapaxes(-2,-1), matmul(S0(t2), s0))[...,0,0]
        p = p + matmul(sp2.swapaxes(-2,-1), matmul(Sp(t2), sp))[...,0,0]
        
        pr0[idx_t, :] = np.real(p) / 3
    return pr0
    
def rabi(tp, tau, phi, wr, we, dwc, an, T2inv):
    n_models = wr.shape[0]
    pr0 = np.empty((tp.size, wr.size))
    # supergenerator without nitrogen
    G1 = (wr[:, np.newaxis, np.newaxis] * vecGx(n_models) \
        + we[:, np.newaxis, np.newaxis] * vecGz(n_models) \
        + dwc[:, np.newaxis, np.newaxis] * vecGz2(n_models) \
        + T2inv[:, np.newaxis, np.newaxis] * vecLz(n_models))
    # supergenerator for just nitrogen
    GA = an[:, np.newaxis, np.newaxis] * vecGz(n_models)
    for idx_t, t in enumerate(tp):
        # simulate three nitrogen cases and average results
        # note that preparing P0 and measuring P0 is equivalent to getting central element (4,4)
        pr0[idx_t, :]  = np.real(
            vexpm(t * (G1 - GA))[:, 4, 4] + 
            vexpm(t * G1)[:, 4, 4] + 
            vexpm(t * (G1 + GA))[:, 4, 4]
        ) / 3
    return pr0
    
def ramsey(tp, tau, phi, wr, we, dwc, an, T2inv):
    """
    Return signal due to Ramsey experiment with
    given parameters
    """
    n_models = wr.shape[0]
    # hamiltonian without nitrogen during rabi
    H1 = (wr[:, np.newaxis, np.newaxis] * vecSx(n_models) \
        + we[:, np.newaxis, np.newaxis] * vecSz(n_models) \
        + dwc[:, np.newaxis, np.newaxis] * vecSz2(n_models))
    # hamiltonian for just nitrogen
    HA = an[:, np.newaxis, np.newaxis] * vecSz(n_models)
    # supergenerator during wait
    G1 = (we[:, np.newaxis, np.newaxis] * vecGz(n_models) \
        + dwc[:, np.newaxis, np.newaxis] * vecGz2(n_models) \
        + T2inv[:, np.newaxis, np.newaxis] * vecLz(n_models))
    # supergenerator for just nitrogen
    GA = an[:, np.newaxis, np.newaxis] * vecGz(n_models)
    
    pr0 = np.empty((tp.size, wr.size))
    for idx_t in range(tp.size):
        t1, t2 = tp[idx_t], tau[idx_t]

        # states after square pulses
        sm = vexpm(t1 * (H1 - HA))[...,1]
        s0 = vexpm(t1 * H1)[...,1]
        sp = vexpm(t1 * (H1 + HA))[...,1]
        
        # convert to vectorized density matrix
        sm = np.repeat(sm.conj(), 3, axis=-1) * np.reshape(np.tile(sm, 3), (-1, 9))
        sm = sm[...,np.newaxis]
        s0 = np.repeat(s0.conj(), 3, axis=-1) * np.reshape(np.tile(s0, 3), (-1, 9))
        s0 = s0[...,np.newaxis]
        sp = np.repeat(sp.conj(), 3, axis=-1) * np.reshape(np.tile(sp, 3), (-1, 9))
        sp = sp[...,np.newaxis]
        
        # sandwich wait between square pulses
        p = matmul(sm.swapaxes(-2,-1), matmul(vexpm(t2 * (G1 - GA)), sm))[...,0,0]
        p = p + matmul(s0.swapaxes(-2,-1), matmul(vexpm(t2 * G1), s0))[...,0,0]
        p = p + matmul(sp.swapaxes(-2,-1), matmul(vexpm(t2 * (G1 + GA)), sp))[...,0,0]
        
        pr0[idx_t, :] = np.real(p) / 3
    return pr0
    

## CLASSES ##################################################################

class MemoizeLikelihood(object):
    """
    Simple depth-1 memoizer for the likelihood function -- useful because 
    particles only move with resample.
    """
    def __init__(self, lhood):
        self._lhood = lhood
        self._latest_hash = None
        self._cached_lhood = None
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self._lhood
        return partial(self, obj)
    def __call__(self, obj, outcomes, modelparams, expparams):
        # this is a naive but fast hash; not too worried because the 
        # the memory is only depth 1
        h = hash(str(outcomes) + str(modelparams) + str(expparams))
        if h != self._latest_hash:
            self._latest_hash = h
            self._cached_lhood = self._lhood(obj, outcomes, modelparams, expparams)
        return self._cached_lhood

class RabiRamseyModel(qi.FiniteOutcomeModel):
    r"""
    Model of a single shot in a Rabi flopping experiment.
    
    Model parameters:
    0: :math:`\Omega`, Rabi strength (MHz); coefficient of Sx
    1: :math:`\omega_e`, Zeeman frequency (MHz); coefficient of Sz
    2: :math:`\Delta \omega_c`, ZFS detuning (MHz); coefficient of Sz^2
    3: :math:`\A_N`, Nitrogen hyperfine splitting (MHz); modeled as incoherent average  
    4: :math:`T_2^-1`, inverse of electron T2* (MHz)
    5: :math:`\Omega_\text{Ramsey}`, the Rabi strength (MHz) while doing a Ramsey experiment

    Experiment parameters:
    mode: Specifies whether a reference or signal count is being performed.
    t:   Pulse width
    tau: Ramsey wait time (only relevent if mode is `RabiRamseyModel.RAMSEY`)
    phi: Ramsey phase between pulses (")
    """
    
    RABI = 0
    RAMSEY = 1

    (
        IDX_OMEGA, IDX_ZEEMAN,
        IDX_DCENTER,
        IDX_A_N, IDX_T2_INV,
        IDX_ALPHA, IDX_BETA
    ) = range(7)

    def __init__(self):
        super(RabiRamseyModel, self).__init__()

        self.simulator = {
            self.RABI:   rabi_cached,
            self.RAMSEY: ramsey_cached
        }

        self._domain = qi.IntegerDomain(min=0, max=1)

    @property
    def n_modelparams(self):
        return len(self.modelparam_names)

    @property
    def modelparam_names(self):
        return [
            r'\Omega',
            r'\omega_e',
            r'\Delta\omega_c',
            r'A_N',
            r'T_2^{-1}'
        ]
        
    @property
    def expparams_dtype(self):
        return [('t', 'float'), ('tau', 'float'), ('phi', 'float'), ('wo','float'), ('emode', 'int')]
        
    @property
    def is_n_outcomes_constant(self):
        return True
        
    @staticmethod
    def are_models_valid(modelparams):
        return np.all(
            [
                # Require that some frequencies be positive.
                modelparams[:, RabiRamseyModel.IDX_OMEGA] >= 0,
                modelparams[:, RabiRamseyModel.IDX_ZEEMAN] >= 0,
                modelparams[:, RabiRamseyModel.IDX_A_N] >= 0,

                # Require that T₂ is positive.
                modelparams[:, RabiRamseyModel.IDX_T2_INV] >= 0
            ],
            axis=0
        )
        
    def domain(self, expparams):
        return self._domain if expparams is None else [self._domain]*expparams.shape[0]

    def n_outcomes(self, expparams):
        return 2
        
    #@MemoizeLikelihood
    def likelihood(self, outcomes, modelparams, expparams):
        """
        Returns the likelihood of measuring |0> under a projective
        measurement.
        """
        
        # get model details for all particles
        wr    = modelparams[:, self.IDX_OMEGA]
        we    = modelparams[:, self.IDX_ZEEMAN]
        dwc   = modelparams[:, self.IDX_DCENTER]
        an    = modelparams[:, self.IDX_A_N]
        T2inv = modelparams[:, self.IDX_T2_INV]
        
        # get expparam details
        mode = expparams['emode']
        t = expparams['t'] 
        tau = expparams['tau'] 
        phi = expparams['phi']
        
        # note that all wo have to be the same, this considerably improves simulator efficiency
        wo = expparams['wo'][0] 
        if not np.allclose(expparams['wo'], wo):
            warnings.warn('In a given likelihood call, all carrier offsets must be identical')        
        
        # figure out which experements are rabi and ramsey
        rabi_mask, ramsey_mask = mode == self.RABI, mode == self.RAMSEY
        
        # run both simulations
        pr0 = np.empty((expparams.shape[0], modelparams.shape[0]))
        if rabi_mask.sum() > 0:
            pr0[rabi_mask] = self.simulator[self.RABI](
                    t[rabi_mask], 0, 0, wr, we, dwc-wo, an, T2inv
                )
        if ramsey_mask.sum() > 0:
            pr0[ramsey_mask] = self.simulator[self.RAMSEY](
                    t[ramsey_mask], tau[ramsey_mask], phi[ramsey_mask], wr, we, dwc-wo, an, T2inv
                )

        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0.T)
        
class RabiRamseyExtendedModel(qi.FiniteOutcomeModel):
    r"""
    Generalizes RabiRamseyModel slightly to include:
        1) different rabi frequencies at different offsets
        2) different pulse lengths on the ramsey hard pulses
    
    Model parameters:
    0: :math:`\Omega`, Rabi strength (MHz); coefficient of Sx
    1: :math:`\omega_e`, Zeeman frequency (MHz); coefficient of Sz
    2: :math:`\Delta \omega_c`, ZFS detuning (MHz); coefficient of Sz^2
    3: :math:`\A_N`, Nitrogen hyperfine splitting (MHz); modeled as incoherent average  
    4: :math:`T_2^-1`, inverse of electron T2* (MHz)
    5: relative amplitudes of rabi frequencies

    Experiment parameters:
    mode: Specifies whether a reference or signal count is being performed.
    t:   Pulse width
    tp2: Pulse width of second ramsey pulse.
    tau: Ramsey wait time (only relevent if mode is `RabiRamseyModel.RAMSEY`)
    phi: Ramsey phase between pulses (")
    
    :param float max_offset: The maximum offset in MHz from 2.87GHz that an 
        experiment will be run at.
    :param int n_offset: The number of points on one side of 2.87GHz 
        to use in the interpolation function that describes the amplitude
        transfer function. There will be a total of `2*n_offset` unitless model
        parameters used. The amplitude at 0MHz offset is 1, with frequency 
        corresponding to the `\omega_e` parameter.
    """
    
    RABI = 0
    RAMSEY = 1

    (
        IDX_OMEGA, IDX_ZEEMAN,
        IDX_DCENTER,
        IDX_A_N, IDX_T2_INV
    ) = range(5)

    def __init__(self, max_offset, n_offset):
        self.max_offset = max_offset
        self.n_offset = n_offset
        self.neg_offsets = np.linspace(-max_offset, 0, n_offset+1)[:-1]
        self.pos_offsets = np.linspace(0, max_offset, n_offset+1)[1:]
        self.all_offsets = np.linspace(-max_offset, max_offset, 2*n_offset+1)

        super(RabiRamseyExtendedModel, self).__init__()
        
        self.simulator = {
            self.RABI:   rabi_cached,
            self.RAMSEY: two_pulse_cached
        }

        self._domain = qi.IntegerDomain(min=0, max=1)

    @property
    def n_modelparams(self):
        return len(self.modelparam_names)
        
    def transfer_function(self, modelparams):
        wr = modelparams[:,0]
        neg = modelparams[:,5:5+self.n_offset]
        pos = modelparams[:,5+self.n_offset:5+2*self.n_offset]
        return CubicSpline(
            self.all_offsets,
            wr[:,np.newaxis] * np.concatenate(
                [neg, np.ones((modelparams.shape[0],1)), pos], 
                axis=1
            ),
            axis=1
        )
        
    def rabi_frequency(self, offset, modelparams):
        return self.transfer_function(modelparams)(offset)

    @property
    def modelparam_names(self):
        return [
            r'\Omega',
            r'\omega_e',
            r'\Delta\omega_c',
            r'A_N',
            r'T_2^{-1}'
        ] + [
            'a_{{{}MHz}}'.format(offset) for offset in self.neg_offsets
        ] + [
            'a_{{{}MHz}}'.format(offset) for offset in self.pos_offsets
        ]
        
    @property
    def expparams_dtype(self):
        return [('t', 'float'), ('tp2', 'float'), ('tau', 'float'), ('phi', 'float'), ('wo','float'), ('emode', 'int')]
        
    @property
    def is_n_outcomes_constant(self):
        return True
        
    @staticmethod
    def are_models_valid(modelparams):
        return np.all(
            [
                # Require that some frequencies be positive.
                modelparams[:, RabiRamseyModel.IDX_OMEGA] >= 0,
                modelparams[:, RabiRamseyModel.IDX_ZEEMAN] >= 0,
                modelparams[:, RabiRamseyModel.IDX_A_N] >= 0,

                # Require that T₂ is positive.
                modelparams[:, RabiRamseyModel.IDX_T2_INV] >= 0
            ],
            axis=0
        )
        
    def domain(self, expparams):
        return self._domain if expparams is None else [self._domain]*expparams.shape[0]

    def n_outcomes(self, expparams):
        return 2
        
    #@MemoizeLikelihood
    def likelihood(self, outcomes, modelparams, expparams):
        """
        Returns the likelihood of measuring |0> under a projective
        measurement.
        """
        
        # note that all wo have to be the same, this considerably improves simulator efficiency
        wo = expparams['wo'][0] 
        if not np.allclose(expparams['wo'], wo):
            warnings.warn('In a given likelihood call, all carrier offsets must be identical') 
        if np.abs(wo) > self.max_offset:
            warnings.warn('Offset value {} is too big.'.format(wo)) 
        
        # get model details for all particles
        wr    = self.rabi_frequency(wo, modelparams)
        we    = modelparams[:, self.IDX_ZEEMAN]
        dwc   = modelparams[:, self.IDX_DCENTER]
        an    = modelparams[:, self.IDX_A_N]
        T2inv = modelparams[:, self.IDX_T2_INV]
        
        # get expparam details
        mode = expparams['emode']
        t = expparams['t'] 
        tp2 = expparams['tp2']
        tau = expparams['tau'] 
        phi = expparams['phi']
   
        
        # figure out which experements are rabi and ramsey
        rabi_mask, ramsey_mask = mode == self.RABI, mode == self.RAMSEY
        
        # run both simulations
        pr0 = np.empty((expparams.shape[0], modelparams.shape[0]))
        if rabi_mask.sum() > 0:
            pr0[rabi_mask] = self.simulator[self.RABI](
                    t[rabi_mask], 0, 0, wr, we, dwc-wo, an, T2inv
                )
        if ramsey_mask.sum() > 0:
            pr0[ramsey_mask] = self.simulator[self.RAMSEY](
                    t[ramsey_mask], tp2[ramsey_mask], tau[ramsey_mask], phi[ramsey_mask], wr, we, dwc-wo, an, T2inv
                )

        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0.T)

class ReferencedPoissonModel(qi.DerivedModel):
    """
    Model whose "true" underlying model is a coin flip, but where the coin is
    only accessible by drawing three poisson random variates, the rate
    of the third being the convex combination of the rates of the first two,
    and the linear parameter being the weight of the coin.
    By drawing from all three poisson random variables, information about the
    coin can be extracted, and the rates are thought of as nuisance parameters.

    This model is in many ways similar to the :class:`BinomialModel`, in
    particular, it requires the underlying model to have exactly two outcomes.

    :param Model underlying_model: The "true" model hidden by poisson random
        variables which set upper and lower references.

    Note that new ``modelparam`` fields alpha and beta are
    added by this model. They refer to, respectively, the higher poisson
    rate (corresponding to underlying probability 1)
    and the lower poisson rate (corresponding to underlying probability 0).

    Additionally, an exparam field ``mode`` is added.
    This field indicates whether just the signal has been measured (0), the
    bright reference (1), or the dark reference (2).
    To ensure the correct operation of this model, it is important that the
    decorated model does not also admit a field with the name ``mode``.
    """

    SIGNAL = 0
    BRIGHT = 1
    DARK = 2

    def __init__(self, underlying_model):
        super(ReferencedPoissonModel, self).__init__(underlying_model)

        if not (underlying_model.is_n_outcomes_constant and underlying_model.domain(None).n_members == 2):
            raise ValueError("Decorated model must be a two-outcome model.")

        if isinstance(underlying_model.expparams_dtype, str):
            # We default to calling the original experiment parameters "p".
            self._expparams_scalar = True
            self._expparams_dtype = [('p', underlying_model.expparams_dtype), ('mode', 'int'), ('n_meas','float')]
        else:
            self._expparams_scalar = False
            self._expparams_dtype = underlying_model.expparams_dtype + [('mode', 'int'), ('n_meas','float')]

        # The domain for any mode of an experiment is all of the non-negative integers
        self._domain = qi.IntegerDomain(min=0, max=1e6)
        
        

    ## PROPERTIES ##

    @property
    def n_modelparams(self):
        return super(ReferencedPoissonModel, self).n_modelparams + 2

    @property
    def modelparam_names(self):
        underlying_names = super(ReferencedPoissonModel, self).modelparam_names
        return underlying_names + [r'\alpha', r'\beta']

    @property
    def expparams_dtype(self):
        return self._expparams_dtype

    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.

        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        return True

    ## METHODS ##

    def are_models_valid(self, modelparams):
        u_valid = self.underlying_model.are_models_valid(modelparams[:,:-2])
        s_valid = np.logical_and(modelparams[:,-1] <= modelparams[:,-2], modelparams[:,-2] >= 0)
        return np.logical_and(u_valid, s_valid)

    def canonicalize(self, modelparams):
        u_model = self.underlying_model.canonicalize(modelparams[:,:-2])
        mask = modelparams[:,-2] <= modelparams[:,-1]
        avg = (modelparams[mask,-1] + modelparams[mask,-2]) / 2 
        modelparams[mask,-2] = avg
        modelparams[mask,-1] = avg
        return modelparams

    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.

        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.

        Note: This is incorrect as there are an infinite number of outcomes.
        We arbitrarily pick a number.
        """
        return 1000

    def domain(self, expparams):
        """
        Returns a list of ``Domain``s, one for each input expparam.

        :param numpy.ndarray expparams:  Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.

        :rtype: list of ``Domain``
        """
        return self._domain if expparams is None else [self._domain for ep in expparams]

    def likelihood(self, outcomes, modelparams, expparams):
        
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(ReferencedPoissonModel, self).likelihood(outcomes, modelparams, expparams)

        L = np.empty((outcomes.shape[0], modelparams.shape[0], expparams.shape[0]))
        ot = np.tile(outcomes, (modelparams.shape[0],1)).T

        for idx_ep, expparam in enumerate(expparams):

            if expparam['mode'] == self.SIGNAL:

                # Get the probability of outcome 1 for the underlying model.
                pr0 = self.underlying_model.likelihood(
                    np.array([0], dtype='uint'),
                    modelparams[:,:-2],
                    np.array([expparam['p']]) if self._expparams_scalar else np.array([expparam]))[0,:,0]
                pr0 = np.tile(pr0, (outcomes.shape[0], 1))

                # Reference Rate
                alpha = expparam['n_meas'] * np.tile(modelparams[:, -2], (outcomes.shape[0], 1))
                beta = expparam['n_meas'] * np.tile(modelparams[:, -1], (outcomes.shape[0], 1))

                # For each model parameter, turn this into an expected poisson rate
                gamma = pr0 * alpha + (1 - pr0) * beta

                # The likelihood of getting each of the outcomes for each of the modelparams
                L[:,:,idx_ep] = poisson_pdf(ot, gamma)

            elif expparam['mode'] == self.BRIGHT:

                # Reference Rate
                alpha = expparam['n_meas'] * np.tile(modelparams[:, -2], (outcomes.shape[0], 1))

                # The likelihood of getting each of the outcomes for each of the modelparams
                L[:,:,idx_ep] = poisson_pdf(ot, alpha)

            elif expparam['mode'] == self.DARK:

                # Reference Rate
                beta = expparam['n_meas'] * np.tile(modelparams[:, -1], (outcomes.shape[0], 1))

                # The likelihood of getting each of the outcomes for each of the modelparams
                L[:,:,idx_ep] = poisson_pdf(ot, beta)
            else:
                raise(ValueError('Unknown mode detected in ReferencedPoissonModel.'))

        assert not np.any(np.isnan(L))
        return L

    def simulate_experiment(self, modelparams, expparams, repeat=1):
        super(ReferencedPoissonModel, self).simulate_experiment(modelparams, expparams, repeat)

        n_mps = modelparams.shape[0]
        n_eps = expparams.shape[0]
        outcomes = np.empty(shape=(repeat, n_mps, n_eps))

        for idx_ep, expparam in enumerate(expparams):
            if expparam['mode'] == self.SIGNAL:
                # Get the probability of outcome 1 for the underlying model.
 
                ep = np.array([expparam['p']]) if self._expparams_scalar else np.array([expparam])
                pr0 = self.underlying_model.likelihood(
                    np.array([0], dtype='uint'),
                    modelparams[:,:-2],
                    ep)[0,:,0]

                # Reference Rate
                alpha = expparam['n_meas'] * modelparams[:, -2]
                beta = expparam['n_meas'] * modelparams[:, -1]

                outcomes[:,:,idx_ep] = np.random.poisson(pr0 * alpha + (1 - pr0) * beta, size=(repeat, n_mps))
            elif expparam['mode'] == self.BRIGHT:
                alpha = expparam['n_meas'] * modelparams[:, -2]
                outcomes[:,:,idx_ep] = np.random.poisson(alpha, size=(repeat, n_mps))
            elif expparam['mode'] == self.DARK:
                beta = expparam['n_meas'] * modelparams[:, -1]
                outcomes[:,:,idx_ep] = np.random.poisson(beta, size=(repeat, n_mps))
            else:
                raise(ValueError('Unknown mode detected in ReferencedPoissonModel.'))

        return outcomes[0,0,0] if outcomes.size == 1 else outcomes

    def update_timestep(self, modelparams, expparams):
        mps = self.underlying_model.update_timestep(modelparams[:,:-2],
            np.array([expparams['p']]) if self._expparams_scalar else expparams)
        return np.concatenate([
                mps, 
                np.repeat(modelparams[:,-2:,np.newaxis], expparams.shape[0], axis=2)
            ], axis=1)
            
            
class BridgedRPMUpdater(qi.SMCUpdater):
    """
    We make a few changes to the SMC updater specific to the referenced 
    poisson model. We can make the updater more numerically stable by detecting
    those updates which reduce the number of effective particles drastically,
    and compensating by reducing the update into smaller artificial updates.
    """
    
    def __init__(self, model, n_particles, prior, branch_size=2, max_recursion=10, zero_weight_policy='ignore', n_ess_thresh= 1000, **kwargs):
        super(BridgedRPMUpdater, self).__init__(model, n_particles, prior, **kwargs)
        self.n_ess_thresh = n_ess_thresh
        self.branch_size = branch_size
        self.max_recursion = max_recursion
        self._zero_weight_policy = zero_weight_policy

    def update(self, outcome, expparams, check_for_resample=True, data_divider=1):
        """
        Here we modify the usual update step slightly by refusing to update 
        the particles if the anticipated n_ess is lower than self.n_ess_thresh.
        In this case we recursively call update, having divided the data 
        counts 
        
        We can bypass this behaviour by setting force_update to True
        """

        # calculate what would happen with an update of these data at the
        # current value of data_divider
        # note that we may not be giving it an integer outcome, but poisson_pdf can handle it
        divided_expparams = expparams.copy()
        divided_expparams['n_meas'] = divided_expparams['n_meas'] / data_divider
        weights, norm = self.hypothetical_update(outcome / data_divider, divided_expparams, return_normalization=True)
        weight_sum = np.sum(weights[0,0,:]**2)
        if weight_sum > 0:
            n_ess = 1 / weight_sum
        else:
            n_ess = 0

        # Check for negative weights           
        if not np.all(weights >= 0):
            warnings.warn("Negative weights occured in particle approximation. Smallest weight observed == {}. Clipping weights.".format(np.min(weights)), ApproximationWarning)
            np.clip(weights, 0, 1, out=weights)

        # Next, check if we have caused the weights to go to zero, as can
        # happen if the likelihood is identically zero for all particles,
        # or if the previous clip step choked on a NaN.
        if np.sum(weights) <= self._zero_weight_thresh:
            if self._zero_weight_policy == 'ignore':
                n_ess = 0
            elif self._zero_weight_policy == 'skip':
                return
            elif self._zero_weight_policy == 'warn':
                warnings.warn("All particle weights are zero. This will very likely fail quite badly.", ApproximationWarning)
            elif self._zero_weight_policy == 'error':
                raise RuntimeError("All particle weights are zero.")
            elif self._zero_weight_policy == 'reset':
                warnings.warn("All particle weights are zero. Resetting from initial prior.", ApproximationWarning)
                self.reset()
            else:
                raise ValueError("Invalid zero-weight policy {} encountered.".format(self._zero_weight_policy))

        # we can use the current data_divider value to figure out 
        # how deep into the recursion we are
        depth = np.log(data_divider) / np.log(self.branch_size)

        # allow another recurse only if we are not too deep
        if depth < self.max_recursion:
            # if effective particle count from this update is too low, 
            # bridge it into branch_size number of steps, with a resample
            # allowed after each one
            if n_ess < self.n_ess_thresh:
                n_ess = [
                    self.update(
                        outcome, expparams, 
                        check_for_resample=check_for_resample, 
                        data_divider = self.branch_size*data_divider
                    )
                    for idx_branch in range(self.branch_size)
                ]
                
                return n_ess
        
        # if we have made it here, we have hit a base case, and 
        # are actually going to update
        self._data_record.append(outcome / data_divider)
        self._just_resampled = False
        self._normalization_record.append(norm[0][0])

        self.particle_weights[:] = weights[0,0,:]
        self.particle_locations = self.model.update_timestep(
            self.particle_locations, expparams
        )[:, :, 0]

        # Check if we need to update our min_n_ess attribute.
        if self.n_ess <= self._min_n_ess:
            self._min_n_ess = self.n_ess
        
        # Resample if needed.
        if check_for_resample:
            self._maybe_resample()
            
        return int(n_ess)
