# -*- coding: utf-8 -*-
from __future__ import division
from functools import partial
import warnings

import qinfer as qi

import numpy as np
import scipy.io as sio
from scipy.linalg import expm
from scipy.special import xlogy, gammaln
import scipy.stats as st
from scipy import interpolate
from vexpm import vexpm
from vexpm import matmul
from fractions import gcd as fgcd

from glob import glob

try:
    import ipyparallel as ipp
    interactive = ipp.interactive
except ImportError:
    try:
        import IPython.parallel as ipp
        interactive = ipp.interactive
    except (ImportError, AttributeError):
        import warnings
        warnings.warn(
            "Could not import IPython parallel. "
            "Parallelization support will be disabled."
        )
        ipp = None
        interactive = lambda fn: fn

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
LZ = (np.kron(Sz, Sz) - (np.kron(Sz2, Si) + np.kron(Si, Sz2)) / 2)

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
vecLz = RepeatedOperator(LZ)
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

    Sm = CachedPropagator(G1 - GA, base_timestep=base_timestep, max_expected_time=np.amax(tp))
    S0 = CachedPropagator(G1, base_timestep=base_timestep, max_expected_time=np.amax(tp))
    Sp = CachedPropagator(G1 + GA, base_timestep=base_timestep, max_expected_time=np.amax(tp))

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
    
    Sm = CachedPropagator(G1 - GA, base_timestep=base_timestep_tau, max_expected_time=np.amax(tau))
    S0 = CachedPropagator(G1, base_timestep=base_timestep_tau, max_expected_time=np.amax(tau))
    Sp = CachedPropagator(G1 + GA, base_timestep=base_timestep_tau, max_expected_time=np.amax(tau))
    Um = CachedPropagator(H1 - HA, base_timestep=base_timestep_tp, max_expected_time=np.amax(tp))
    U0 = CachedPropagator(H1, base_timestep=base_timestep_tp, max_expected_time=np.amax(tp))
    Up = CachedPropagator(H1 + HA, base_timestep=base_timestep_tp, max_expected_time=np.amax(tp))
    
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

class RabiRamseyModel(qi.FiniteOutcomeModel, qi.DifferentiableModel):
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
    
    def _simulated_probs(self, modelparams, expparams):
        # get model details for all particles
        wr    = modelparams[:, self.IDX_OMEGA]
        we    = modelparams[:, self.IDX_ZEEMAN]
        dwc   = modelparams[:, self.IDX_DCENTER]
        an    = modelparams[:, self.IDX_A_N]
        T2inv = modelparams[:, self.IDX_T2_INV]
        
        # get expparam details
        mode = expparams['emode']
        t, tau, phi =  expparams['t'], expparams['tau'], expparams['phi']
        
        # note that all wo have to be the same, this considerably improves simulator efficiency
        wo = expparams['wo'][0] 
        if not np.allclose(expparams['wo'], wo):
            warnings.warn('In a given likelihood call, all carrier offsets must be identical')        
        
        # figure out which experements are rabi and ramsey
        rabi_mask, ramsey_mask = mode == self.RABI, mode == self.RAMSEY
        
        # finally run all simulations
        pr0 = np.empty((expparams.shape[0], modelparams.shape[0]))
        if rabi_mask.sum() > 0:
            pr0[rabi_mask,:] = self.simulator[self.RABI](
                    t[rabi_mask], 0, 0, wr, we, dwc-wo, an, T2inv
                )
        if ramsey_mask.sum() > 0:
            pr0[ramsey_mask,:] = self.simulator[self.RAMSEY](
                    t[ramsey_mask], tau[ramsey_mask], phi[ramsey_mask], wr, we, dwc-wo, an, T2inv
                )
        return pr0.T
            
    
    #@MemoizeLikelihood
    def likelihood(self, outcomes, modelparams, expparams):
        """
        Returns the likelihood of measuring |0> under a projective
        measurement.
        """
        pr0 = self._simulated_probs(modelparams, expparams)
        return qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)
        
    def grad(self, outcomes, modelparams, expparams, return_L=False, d_mps=None, steps=None):
        """
        Computes the gradient of the likelihood function with respect to
        model parameters.
        
        :param bool return_L: If true, returns `(grad, L)` where L is the
            likelihood at the input paramaters.
        :param np.ndarray d_mps: A list of step sizes to use for each modeparam.
        :param dict steps: The finite difference scheme, for example, 
            `{0:-1, 1:1}` for forward difference, or `{-1:-0.5, 1:0.5}` 
            for central difference.
            See https://en.wikipedia.org/wiki/Finite_difference_coefficient
        """
        
        d_mps = 1e-6 * np.ones(self.n_modelparams) if d_mps is None else d_mps
        steps = {0:-1,1:1} if steps is None else steps
        
        # compute likelihood at 0 difference, if needed
        if return_L or 0 in steps:
            pr0 = self._simulated_probs(modelparams, expparams)
            
        # add each term of the finite difference one at a time
        grad0 = np.zeros((self.n_modelparams, modelparams.shape[0], expparams.shape[0]))
        all_directions = np.diag(d_mps)[:,np.newaxis,:]
        for idx_step, step in enumerate(steps):
            if step == 0:
                grad0 += steps[step] * pr0[np.newaxis, ...]
            else:
                new_mps = modelparams[np.newaxis,:,:] + step * all_directions
                new_mps = new_mps.reshape(-1, self.n_modelparams)
                grad0 += steps[step] * self._simulated_probs(new_mps, expparams).reshape(grad0.shape)
        grad0 /= d_mps[:, np.newaxis, np.newaxis]
        
        # deal with the dumb outcomes; pr1 = 1-pr0, so just flip sign
        zero_mask = outcomes == 0
        grad = np.empty((outcomes.shape[0], self.n_modelparams, modelparams.shape[0], expparams.shape[0]))
        grad[zero_mask,...] = grad0
        one_mask = np.logical_not(zero_mask)
        grad[one_mask,...] = -grad0
        grad = grad.transpose(1,0,2,3)
        
        if return_L:
            return grad, qi.FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0) 
        else:
            return grad
        
    def score(self, outcomes, modelparams, expparams, return_L=False):
        grad, prs = self.grad(outcomes, modelparams, expparams, return_L=True)
        score = grad / prs[np.newaxis,:,:,:]
        
        if return_L:
            return score, prs 
        else:
            return score

class ParallelModel(qi.DirectViewParallelizedModel):
    def grad(self, outcomes, modelparams, expparams, return_L=False, d_mps=None, steps=None):
       
        @interactive
        def serial_grad(mps, sm, os, eps, return_L, d_mps, steps):
            return sm.grad(os, mps, eps, return_L=return_L, d_mps=d_mps, steps=steps)

        results = self._dv.map_sync(
            serial_grad,
            np.array_split(modelparams, self.n_engines, axis=0),
            [self.underlying_model] * self.n_engines,
            [outcomes] * self.n_engines,
            [expparams] * self.n_engines,
            [return_L] * self.n_engines,
            [d_mps] * self.n_engines,
            [steps] * self.n_engines
        )

        if return_L:
            g = np.concatenate([r[0] for r in results], axis=2)
            L = np.concatenate([r[1] for r in results], axis=1)
            return g, L
        else:
            return np.concatenate(results, axis=2)
        
class RabiRamseyExtendedModel(qi.FiniteOutcomeModel):
    r"""
    Generalizes RabiRamseyModel slightly to include:
        1) different rabi frequencies at different offsets
        2) different pulse lengths on the two ramsey hard pulses
    
    Note that we keep the zeeman frequency at 0MHz offset as parameter 0 
    for the semblance of backwards-compatibility with RabiRamseyModel, which
    this class supercedes.
    
    Model parameters:
    0: :math:`\Omega`, Rabi strength (MHz); coefficient of Sx
    1: :math:`\omega_e`, Zeeman frequency (MHz) at 0MHz offset; coefficient of Sz
    2: :math:`\Delta \omega_c`, ZFS detuning (MHz); coefficient of Sz^2
    3: :math:`\A_N`, Nitrogen hyperfine splitting (MHz); modeled as incoherent average  
    4: :math:`T_2^-1`, inverse of electron T2* (MHz)
    5...: Zeeman frequencies at offsets -max_offset...not(0)...+max_offset

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
        transfer function. There will be a total of `2*n_offset` rabi
        frequencies stored apart from the one at 0 offset. 
        Interpolation is linear between these stored values.
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
        wr = modelparams[:,0,np.newaxis]
        neg = modelparams[:,5:5+self.n_offset]
        pos = modelparams[:,5+self.n_offset:5+2*self.n_offset]
        return interpolate.interp1d(
            self.all_offsets,
            np.concatenate(
                [neg, wr, pos], 
                axis=1
            ),
            axis=1,
            assume_sorted=True,
            kind='linear'
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
            '\Omega_{{{}MHz}}'.format(offset) for offset in self.neg_offsets
        ] + [
            '\Omega_{{{}MHz}}'.format(offset) for offset in self.pos_offsets
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
            ] +
            [
                modelparams[:,idx] >= 0 for idx in range(5, modelparams.shape[1])
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

################################################################################
# DISTRIBUTIONS AND UPDATERS
################################################################################

class InverseUniform(qi.Distribution):
    def __init__(self, bounds):
        super(InverseUniform, self).__init__()
        self._dist = qi.UniformDistribution(bounds)
    @property
    def n_rvs(self):
        return self._dist.n_rvs
    def sample(self, n):
        samples = self._dist.sample(n)
        return 1 / samples
        
class InverseWishartDistribution(qi.Distribution):
    """
    Inverse Wishart distribution over positive matrices.
    
    :param float nu: Number of prior observations of a normal distibution.
    :param np.ndarray sigma: Square positive matrix.
    :param bool sigma_is_mean: If `False`, `sigma` is a positive matrix with
        the usual definition as found on wikipedia. If `True`, `sigma`
        is scaled so that it is equal to the mean value of this distribution.        
    """
    def __init__(self, nu, sigma, sigma_is_mean=True):
        self.dim = sigma.shape[0]
        _sigma = (nu - self.dim - 1) * sigma if sigma_is_mean else sigma
        self._dist = st.invwishart(df=nu, scale=_sigma)
        
    @property
    def n_rvs(self):
        return self.dim
    
    def sample(self, n=1):
        samples = self._dist.rvs(size=n)
        if samples.ndim == 2:
            samples = samples[np.newaxis, :, :]
        return samples
    
class InverseWishartForCholeskyDistribution(InverseWishartDistribution):
    """
    Applies a parameter change to the samples of `InverseWishartDistribution`.
    Namely, samples from this distribution are the flattened lower
    cholesky decomposition of `InverseWishartDistribution`. Flattening order
    is determined by `np.tril_indices`.
    
    :param float nu: Number of prior observations of a normal distibution.
    :param np.ndarray sigma: Square positive matrix.
    :param bool sigma_is_mean: If `False`, `sigma` is a positive matrix with
        the usual definition as found on wikipedia. If `True`, `sigma`
        is scaled so that it is equal to the mean value of this distribution.
    """
    @property
    def n_rvs(self):
        return int(self.dim * (self.dim + 1) / 2)
    def sample(self, n=1):
        samples = super(InverseWishartForCholeskyDistribution, self).sample(n)
        samples = np.linalg.cholesky(samples)
        idx = (np.s_[:],) + np.tril_indices(self.dim)
        return samples[idx]
    
class DriftDistribution(qi.Distribution):
    """
    A convenience wrappen for `InverseWishartForCholeskyDistribution` describing
    the diffusion rate of the drift parameters.
    
    :param float alpha_mean_drift: The mean dispersion of the alpha parameter
        per second of time.
    :param beta_mean_drift: The mean dispersion of the beta parameter per 
        second of time.
    :param float r: The mean correlation between the two dispersions.
    :param float nu: The inverse wishart distribution degrees of freedom.
        Larger values make a tighter prior. Must be greater than 2.
    """
    def __init__(self, alpha_mean_drift, beta_mean_drift=None, r=0.7, nu=30):
        if beta_mean_drift is None:
            beta_mean_drift = alpha_mean_drift
        a, b = alpha_mean_drift, beta_mean_drift
        sigma = np.array([[a**2, r*a*b], [r*a*b, b**2]])
        self._iw = InverseWishartForCholeskyDistribution(nu, sigma, sigma_is_mean=True)
        
    @property
    def n_rvs(self):
        return 3
    
    def sample(self, n=1):
        return self._iw.sample(n)

class ExtendedPrior(qi.Distribution):
    """
    Takes a prior for RabiRamseyModel possibly wrapped in ReferencedPoissonModel
    and turns it into a prior for RabiRamseyExtended model.
    
    :param qinfer.Distribution base_prior: A prior for the base distribution.
    :param int n_offsets: See RabiRamseyExtendedModel.
    :param float max_offset: See RabiRamseyExtendedModel.
    :param float unit_std: Std per MHz of the transfer function. 
        For example, 0.1 says something like 0.1MHz change
        in frequency allowed per MHz of distance away from 0MHz offset.
    """
    def __init__(self, base_prior, n_offsets, max_offset, unit_std):
        assert base_prior.n_rvs in [5,7]
        self.has_ref = base_prior.n_rvs == 7
        
        self.base_prior = base_prior
        self.n_offsets = n_offsets
        self.max_offset = max_offset
        # it is common for variance to increase linearly when 
        # moving away from a known value. See a Wiener process, eg.
        self.tf_prior = qi.ProductDistribution(*([
                qi.NormalDistribution(mean=0, var=unit_std**2 * (max_offset / n_offsets))
            ] * n_offsets))
                
        self._n_rvs = self.base_prior.n_rvs + self.tf_prior.n_rvs
    
    @property
    def n_rvs(self):
        return self._n_rvs
        
    def sample(self, n=1):
        base_sample = self.base_prior.sample(n)
        tf_sample = np.concatenate([
                np.cumsum(self.tf_prior.sample(n), axis=1)[:,::-1],
                np.cumsum(self.tf_prior.sample(n), axis=1)
            ], axis=1)
        tf_sample = np.abs(base_sample[:,RabiRamseyModel.IDX_OMEGA,np.newaxis] + tf_sample)
            
        if self.has_ref:
            ref_sample = base_sample[:,5:]
            base_sample = base_sample[:,:5]
        else:
            ref_sample = np.zeros((n, 0))

        return np.concatenate([
                base_sample,
                tf_sample,
                ref_sample
            ], axis=1)

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


    def __init__(self, underlying_model, n_outcomes=1000,dview=None):

        super(ReferencedPoissonModel, self).__init__(underlying_model)
        self.dview = dview
        self.allow_identical_outcomes = True
        self._n_outcomes = n_outcomes
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
        
        self._Q = np.concatenate([self.underlying_model.Q, [0,0]])
        self.dview = dview
        
        

    ## PROPERTIES ##

    @property
    def n_modelparams(self):
        return super(ReferencedPoissonModel, self).n_modelparams + 2
        
    @property
    def Q(self):
        # derived model usually returns underlying_model._Q
        return self._Q

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
        
        # allow expparams to mix different types
        mask_signal = expparams['mode'] == self.SIGNAL
        mask_bright = expparams['mode'] == self.BRIGHT
        mask_dark = expparams['mode'] == self.DARK
        assert np.sum(mask_signal) + np.sum(mask_bright) + np.sum(mask_dark) == expparams.size
        
        # compute prob that state is |0> for all modelparams and eps
        pr0 = np.empty((modelparams.shape[0], expparams.shape[0]))
        if np.sum(mask_signal) > 0:
            pr0[:, mask_signal] = self.underlying_model.likelihood(
                np.array([0], dtype='uint'),
                modelparams[:,:-2],
                expparams[mask_signal]['p'] if self._expparams_scalar else expparams[mask_signal]
            )[0,:,:]
        pr0[:,mask_bright] = 1
        pr0[:,mask_dark] = 0
        
        # now work out poisson rates for all combos of mps and eps
        n_meas = expparams['n_meas'][np.newaxis, :]
        alpha = n_meas * modelparams[:, -2, np.newaxis]
        beta = n_meas * modelparams[:, -1, np.newaxis]
        gamma = pr0 * alpha + (1 - pr0) * beta

        # work out likelihood of these rates for each outcome and return
        outcome_idxs = (np.s_[:],np.newaxis,np.newaxis) if outcomes.ndim == 1 else (np.s_[:],np.newaxis,np.s_[:])
        if self.dview is None:
            L = poisson_pdf(outcomes[outcome_idxs], gamma[np.newaxis,:,:])
        else:
            n_engines = len(self.dview)
            # we are careful to avoid copying arrays, passing only views.

            L = np.concatenate(
                self.dview.map_sync(
                    poisson_pdf,
                    np.array_split(outcomes[outcome_idxs], n_engines, axis=2),
                    np.array_split(gamma[np.newaxis,:,:], n_engines, axis=2)
                ),
                axis=2
            )


        assert not np.any(np.isnan(L))

        return L

    def simulate_experiment(self, modelparams, expparams, repeat=1):
        super(ReferencedPoissonModel, self).simulate_experiment(modelparams, expparams, repeat)

        n_mps = modelparams.shape[0]
        n_eps = expparams.shape[0]
        
        # allow expparams to mix different types
        mask_signal = expparams['mode'] == self.SIGNAL
        mask_bright = expparams['mode'] == self.BRIGHT
        mask_dark = expparams['mode'] == self.DARK
        assert np.sum(mask_signal) + np.sum(mask_bright) + np.sum(mask_dark) == expparams.size
        
        # compute prob that state is |0> for all modelparams and eps
        pr0 = np.empty((n_mps, n_eps))
        if np.sum(mask_signal) > 0:
            pr0[:, mask_signal] = self.underlying_model.likelihood(
                np.array([0], dtype='uint'),
                modelparams[:,:-2],
                expparams[mask_signal]['p'] if self._expparams_scalar else expparams[mask_signal]
            )[0,:,:]
        pr0[:,mask_bright] = 1
        pr0[:,mask_dark] = 0
        
        # now work out poisson rates for all combos of mps and eps
        n_meas = expparams['n_meas'][np.newaxis, :]
        alpha = n_meas * modelparams[:, -2, np.newaxis]
        beta = n_meas * modelparams[:, -1, np.newaxis]
        gamma = pr0 * alpha + (1 - pr0) * beta
        
        if self.dview is None:
            outcomes = np.random.poisson(gamma, size=(repeat, n_mps, n_eps))
        else:
            n_engines = len(self.dview)
            split_gamma = np.array_split(gamma, n_engines, axis=0)
            split_size = [ (repeat,s.shape[0],s.shape[1]) for s in split_gamma]
            outcomes = np.concatenate(
                self.dview.map_sync(
                    np.random.poisson,
                    split_gamma,
                    split_size
                ),
                axis=1
            )
        
        return outcomes[0,0,0] if outcomes.size == 1 else outcomes

    def update_timestep(self, modelparams, expparams):
        mps = self.underlying_model.update_timestep(modelparams[:,:-2],
            np.array([expparams['p']]) if self._expparams_scalar else expparams)
        return np.concatenate([
                mps, 
                np.repeat(modelparams[:,-2:,np.newaxis], expparams.shape[0], axis=2)
            ], axis=1)

    def representative_outcomes(self, weights, modelparams, expparams,likelihood_modelparams=None,
                                likelihood_weights=None):
        return super(qi.DerivedModel,self).representative_outcomes(weights, modelparams, expparams,
                                likelihood_modelparams=likelihood_modelparams, likelihood_weights=likelihood_weights)

    def n_outcomes(self,expparams):
        return self._n_outcomes
           
    def score(self, outcomes, modelparams, expparams, return_L):
        raise NotImplemented('We can compute fisher info directly, so skip the score.')
        
    def fisher_information(self, modelparams, expparams, **kwargs):
        
        n_meas = expparams['n_meas'][np.newaxis, :]
        alpha = n_meas * modelparams[:, -2, np.newaxis]
        beta = n_meas * modelparams[:, -1, np.newaxis]
        
        outcomes = np.array([0])
        kwargs['return_L'] = True
        grad, L = self.underlying_model.grad(
                outcomes, 
                modelparams[:,:-2], 
                expparams, 
                **kwargs
            )
        
        factor = (alpha - beta)**2 / (beta + L * (alpha - beta))
        fi = grad[:,np.newaxis,:,:] * grad[np.newaxis,:,:]
        fi *= factor[np.newaxis, np.newaxis,:,:]
        return fi[:,:,0,:,:]
            
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
        
    def update_timestep(self, eps):
        # usually this is done by the update method, but this way is less 
        # confusing
        assert eps.size == 1
        self.particle_locations = self.model.update_timestep(
            self.particle_locations, eps
        )[:, :, 0]

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
        # easiest not to deal with timesteps in this recursive formula,
        # wo don't want to end up calling too many times
        #self.particle_locations = self.model.update_timestep(
        #    self.particle_locations, expparams
        #)[:, :, 0]

        # Check if we need to update our min_n_ess attribute.
        if self.n_ess <= self._min_n_ess:
            self._min_n_ess = self.n_ess
        
        # Resample if needed.
        if check_for_resample:
            self._maybe_resample()
            
        return int(n_ess)
        
        
class RAIUpdaters(qi.SMCUpdater):
    """
    Redundant array of identical updaters. Maintains a number of independent
    updaters which all have the same prior and update with the same data.
    If it is detected that the bayes factor for a given updater is inconsistent,
    then that updater is resampled according to the remaining good ones.
    
    :param updater_class: A `qinfer.SMCUpdater`-like class (not instance).
    :param int n_updaters: The number of independent updaters to use.
    :param model: A QInfer model.
    :param int n_particles_per_updater: The number of particles to use in 
        each updater.
    :param qinfer.Distirbution prior: The prior to use for each updater.
    """
    
    BAYES_FACTOR_THRESHOLD = 0.01
    
    def __init__(self, updater_class, n_updaters, model, n_particles_per_updater, prior, **kwargs):
        
        self.n_updaters = n_updaters
        self.rai_updaters = [
            updater_class(model, n_particles_per_updater, prior, **kwargs)
        ]
        
        self._normalization_record = []
        self._log_updater_weights = np.log(np.ones(n_updaters) / n_updaters)
        self.redistribution_count = 0
        
        n_tot_particles = n_updaters * n_particles
        super(RAIUpdaters, self).__init__(model, n_tot_particles, prior)

    @property
    def particle_locations(self):
        return np.concatenate(
            [updater.particle_locations for updater in self.rai_updaters],
            axis=0
        )
    @particle_locations.setter
    def particle_locations(self, value):
        locs_list = np.array_split(value, self.n_updaters, axis=0)
        for updater, locs in zip(self.rai_updaters, locs_list):
            updater.particle_locations = locs
            
    @property
    def particle_weights(self):
        return np.concatenate(
            [updater.particle_weights for updater in self.rai_updaters],
            axis=0
        ) / self.n_updaters
    @particle_weights.setter
    def particle_weights(self, value):
        w_list = np.array_split(value, self.n_updaters, axis=0)
        for updater, w in zip(self.rai_updaters, w_list):
            updater.particle_weights = w / w.sum()
    
    @property
    def log_updater_weights(self):
        return self._log_updater_weights
        
    def redistribute_particles(self, idxs_replace, idxs_keep):
        #collect the good particles and weights
        keep_particles = np.concatenate(
            [self.rai_updaters[idx].particle_locations for idx in idxs_keep],
            axis=0
        )
        keep_weights = np.concatenate(
            [self.rai_updaters[idx].particle_weights for idx in idxs_keep],
            axis=0
        ) / len(idxs_keep)
        
        #resample the bad updaters from the good ones
        for idx in idxs_replace:
            updater = self.rai_updaters[idx]
            choices = np.random.choice(updater.n_particles, p=keep_weights)
            updater.particle_locations = keep_particles[choices, :]
            updater.particle_weights = keep_weights[choices]
            
        self.redistribution_count += len(idxs_replace)
        
    @property
    def normalized_bayes_factors(self):
        return np.exp(self.log_updater_weights - np.amax(self.log_updater_weights))
            
    def update(self, outcome, expparams, check_for_resample=True):
        
        latest_data_probs = np.empty(self.n_updaters)
        for idx_u, updater in enumerate(self.rai_updaters):
            updater.update(outcome, expparams, check_for_resample=check_for_resample)
            latest_data_probs[idx_u] = np.log(updater.normalization_record[-1])
            
        self._normalization_record.append(latest_data_probs)
        self._log_updater_weights = self.log_updater_weights + latest_data_probs
            
        bad_updaters = self.normalized_bayes_factors < RAIUpdaters.BAYES_FACTOR_THRESHOLD
        idxs_replace = list(np.range(self.n_updaters)[bad_updaters])
        if len(idxs_replace > 0):
            idxs_keep = list(set(range(self.n_particles)) - set(idxs_replace))
            self.redistribute_particles(idxs_replace, idxs_keep)
            good_avg = np.log(np.exp(self.log_updater_weights).mean())
            self._log_updater_weights[idxs_replace] = good_avg    
        
