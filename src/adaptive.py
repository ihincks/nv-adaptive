# -*- coding: utf-8 -*-
from __future__ import division
from future.utils import with_metaclass
import warnings
import abc

import qinfer as qi
import numpy as np
import models as m
import datetime
import dateutil


#-------------------------------------------------------------------------------
# CONSTANTS
#-------------------------------------------------------------------------------

SOME_PRIOR = qi.UniformDistribution(np.array([
            [0,10],
            [0,10],
            [-5,5],
            [1.5,3.5],
            [100**-1,1**-1]
        ]))

#-------------------------------------------------------------------------------
# SWEEP EXPPARAMS
#-------------------------------------------------------------------------------

def rabi_sweep(min_t=None, max_t=0.3, n=50, n_bin=None, wo=0):
    ham_model = m.RabiRamseyModel()
    if min_t is None:
        min_t = max_t / n
    vals = [
        np.linspace(min_t, max_t, n),
        np.zeros(n),
        np.zeros(n),
        np.ones(n) * wo,
        np.ones(n) * ham_model.RABI
    ]
    dtype = ham_model.expparams_dtype
    if n_bin is not None:
        vals = vals + [(n_bin * np.ones(n)).astype(np.int)]
        dtype = dtype + [('n_meas','int')]
    rabi_eps = np.array(vals).T
    rabi_eps = np.array(list(zip(*rabi_eps.T)), dtype=dtype)
    return rabi_eps

def ramsey_sweep(min_tau=None, max_tau=2, tp=0.01, phi=0, n=50, n_bin=None, wo=0):
    ham_model = m.RabiRamseyModel()
    if min_tau is None:
        min_tau = max_tau / n
    vals = [
        tp * np.ones(n),
        np.linspace(min_tau, max_tau, n),
        phi * np.ones(n),
        np.ones(n) * wo,
        np.ones(n) * ham_model.RAMSEY
    ]
    dtype = ham_model.expparams_dtype
    if n_bin is not None:
        vals = vals + [(n_bin * np.ones(n)).astype(np.int)]
        dtype = dtype + [('n_meas','int')]
    ramsey_eps = np.array(vals).T
    ramsey_eps = np.array(list(zip(*ramsey_eps.T)), dtype=dtype)
    return ramsey_eps

#-------------------------------------------------------------------------------
# RUNNERS and TIME STEPPERS
#-------------------------------------------------------------------------------

class StochasticStepper(object):
    def __init__(self):
        self._x = None
        self._history = []
        self.reset()
    def _set_value(self, value):
        self._x = value
        self._history.append(self._x)
    @property
    def value(self):
        return self._x
    @property
    def history(self):
        return np.array(self._history)
    @abc.abstractmethod
    def step(self):
        pass
    @abc.abstractmethod
    def reset(self):
        pass

class OrnsteinUhlenbeckStepper(StochasticStepper):
    def __init__(self, mu, sigma, theta, x0, sigma_x0):
        self.x0 = x0
        self.sigma_x0 = sigma_x0
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        super(OrnsteinUhlenbeckStepper, self).__init__()
        
    def step(self, time):
        eps = np.sqrt(time) * np.random.randn()
        new_x = self._x + self.theta * (self.mu - self._x) * time + self.sigma * eps
        self._set_value(new_x)
        
    def reset(self):
        self._set_value(self.x0 + self.sigma_x0 * np.random.randn(1))
        
class NVDriftStepper(StochasticStepper):
    def __init__(self, 
                mu_alpha=0.016, sigma_alpha=0.0005, 
                sigma_nu=5e-5, theta_nu=0.005, 
                mu_kappa=0.33, sigma_kappa=0.01, theta_kappa=0.01,
                background=0.01
            ):
        self._background = background
        self._nu = OrnsteinUhlenbeckStepper(0, sigma_nu, theta_nu, mu_alpha - background, sigma_alpha)
        self._kappa = OrnsteinUhlenbeckStepper(mu_kappa, sigma_kappa, theta_kappa, 0, 0)
        super(NVDriftStepper, self).__init__()
    
    def _set_value(self):
        super(NVDriftStepper, self)._set_value(
            np.array([
                self._background + self._nu.value, 
                self._background + self._kappa.value * self._nu.value
            ]).flatten()
        )
        
    def step(self, time):
        self._kappa.step(time)
        self._nu.step(time)
        self._set_value()
        
    def reset(self):
        self._kappa.reset()
        self._nu.reset()
        self._set_value()

class ExperimentResult(object):
    def __init__(self, bright, dark, signal, timestamp=None):
        self._bright = bright
        self._dark = dark
        self._signal = signal
        self._timestamp = datetime.datetime.now() if timestamp is None else timestamp
        
    @property
    def bright(self):
        return self._bright
    @property
    def dark(self):
        return self._dark
    @property
    def signal(self):
        return self._signal
    @property
    def timestamp(self):
        return self._timestamp
    @property
    def triplet(self):
        return np.array([self.bright, self.dark, self.signal])
        
class AbstractExperimentJob(with_metaclass(abc.ABCMeta, object)):
    @abc.abstractproperty
    def is_complete(self):
        pass
        
    @abc.abstractmethod
    def get_result(self):
        if not self.is_complete:
            raise RuntimeError('Job is not complete.')
            
class OfflineExperimentJob(AbstractExperimentJob):
    def __init__(self, result):
        self._result = result
    
    @property    
    def is_complete(self):
        return True
        
    def get_result(self):
        super(OfflineExperimentJob, self).get_result()
        return self._result
        
class TopChefExperimentJob(AbstractExperimentJob):
    def __init__(self, topchef_job):
        self._job = topchef_job
    
    @property    
    def is_complete(self):
        return self._job.is_complete
        
    def get_result(self):
        super(TopChefExperimentJob, self).get_result()
        result = self._job.result
        return ExperimentResult(
            result['light_count'],
            result['dark_count'],
            result['result_count'],
            dateutil.parser.parse(result['time_completed'])
        )

class AbstractRabiRamseyExperimentRunner(object):
    
    def __init__(self):
        self._experiment_history = None
        
    def append_experiment_to_history(self, expparam):
        eps = np.atleast_1d(expparam)
        if self._experiment_history is None:
            self._experiment_history = eps
        else:
            self._experiment_history = np.concatenate([self._experiment_history, eps])
    
    @property
    def experiment_history(self):
        return self._experiment_history
    
    @abc.abstractmethod
    def run_experiment(self, expparam):
        """
        Runs the experiment with the given expparams 
        and returns an instance of `AbstractExperimentJob`
        """
        self.append_experiment_to_history(expparam)
        
    @abc.abstractmethod
    def run_tracking(self):
        pass
    
def experiment_time(expparam):
    extra_time_per_shot = 20e-6
    extra_time_per_experiment = 0.6
    if expparam['emode'] == m.RabiRamseyModel.RABI:
        t_experiment = expparam['t']
    else:
        t_experiment = expparam['tau'] + 2 * exparam['t']
    return expparam['mode'] * (t_experiment + extra_time_per_shot) \
            + extra_time_per_experiment
        
class SimulatedRabiRamseyExperimentRunner(AbstractRabiRamseyExperimentRunner):
    def __init__(self, hamparam, drift=None):
        super(SimulatedRabiRamseyExperimentRunner, self).__init__()
        
        self._ham_param = hamparam
        self._ham_model = m.RabiRamseyModel()
        self._model = m.ReferencedPoissonModel(self._ham_model)
        #qi.DirectViewParallelizedModel(
        #    m.ReferencedPoissonModel(self._ham_model), 
        #    dview, 
        #    serial_threshold=1
        #)

        self.drift = NVDriftStepper() if drift is None else drift

    @property
    def modelparam(self):
        return np.atleast_2d(np.concatenate([self._ham_param, self.drift.value]))
    
    def update_timestep(self, expparam):
        self.drift.step(experiment_time(expparam))
        
    def run_tracking(self):
        self.drift.reset()
        
    def run_experiment(self, expparam):
        super(SimulatedRabiRamseyExperimentRunner, self).run_experiment(expparam)
        self.update_timestep(expparam)
        expparam['mode'] = m.ReferencedPoissonModel.BRIGHT
        bright = self._model.simulate_experiment(self.modelparam, expparam)
        expparam['mode'] = m.ReferencedPoissonModel.DARK
        dark = self._model.simulate_experiment(self.modelparam, expparam)
        expparam['mode'] = m.ReferencedPoissonModel.SIGNAL
        signal = self._model.simulate_experiment(self.modelparam, expparam)
        
        return OfflineExperimentJob(
            ExperimentResult(bright, dark, signal)
        )
        
class RealRabiRamseyExperimentRunner(AbstractRabiRamseyExperimentRunner):
    def __init__(self, topchef_service):
        super(RealRabiRamseyExperimentRunner, self).__init__()
        
        self._service = topchef_service
        
        self._n_shots = 100000
        self._meas_time = 800e-9
        self._center_freq = 2.87e9
        self._intermediate_freq = 0
        self._adiabatic_power = -12

    def run_tracking(self):
        raise NotImplemented()
        
    @property
    def _bare_eps(self):
        return {
           'number_of_repetitions': int(self._n_shots),
           'meas_time': self._meas_time,
           'center_freq': self._center_freq,
           'intermediate_freq': self._intermediate_freq,
           'delay_time': 0,
           'pulse1_time': 0,
           'pulse1_phase': 0,
           'pulse1_power': 0,
           'pulse1_offset_freq': 0,
           'pulse1_modulation_freq': 0,
           'pulse1_modulation_phase': 0,
           'pulse2_time': 0,
           'pulse2_phase': 0,
           'pulse2_power': 0,
           'pulse2_offset_freq': 0,
           'pulse2_modulation_freq': 0,
           'pulse2_modulation_phase': 0,
           'adiabatic_power': self._adiabatic_power
       }
        
    def run_experiment(self, expparam):
        super(RealRabiRamseyExperimentRunner, self).run_experiment(expparam)
        
        eps = self._bare_eps
        
        if expparam['emode'] == m.RabiRamseyModel.RABI:
            eps['pulse1_time'] = 1e-6 * expparam['t']
        elif expparam['emode'] == m.RabiRamseyModel.RABI:
            eps['delay_time'] = 1e-6 * expparam['tau']
            eps['pulse1_time'] = 1e-6 * expparam['t']
            eps['pulse2_time'] = 1e-6 * expparam['t']
            eps['pulse2_phase'] = expparam['phi']
        eps['center_freq'] = eps['center_freq'] - 1e6 * expparam['wo']
        
        # the JSON parser doesn't like getting np arrays
        for key in eps.keys():
            try:
                eps[key] = np.asscalar(eps[key])
            except AttributeError:
                pass
            
        topchef_job = self._service.new_job(eps)

        return TopChefExperimentJob(topchef_job)


#-------------------------------------------------------------------------------
# HEURISTICS
#-------------------------------------------------------------------------------

class RiskHeuristic(qi.Heuristic):
    def __init__(self, updater, Q, rabi_eps, ramsey_eps, name=None, dview=None):
        self.updater = updater
        if dview is None:
            self._ham_model = m.RabiRamseyModel()
        else:
            self._ham_model = qi.DirectViewParallelizedModel(m.RabiRamseyModel(), dview, serial_threshold=1)
        self._ham_model._Q = Q
        self._risk_taker = qi.SMCUpdater(self._ham_model, updater.n_particles, SOME_PRIOR)
        self._update_risk_particles()
        self._rabi_eps = rabi_eps
        self._ramsey_eps = ramsey_eps
        self.name = "Bayes Risk, Q={}".format(Q) if name is None else name
        self.risk_history = []
        
    def _update_risk_particles(self):
        self._risk_taker.particle_locations = self.updater.particle_locations
        self._risk_taker.particle_weights = self.updater.particle_weights
        
    def __call__(self, tp):
        ramsey_eps = self._ramsey_eps
        ramsey_eps['t'] = tp
        all_eps = np.concatenate([self._rabi_eps, ramsey_eps])
        
        self._update_risk_particles()
        
        risk = self._risk_taker.bayes_risk(all_eps)
        self.risk_history += [risk]
        best_idx = np.argmin(risk, axis=0)
        eps = np.array([all_eps[best_idx]])
        return eps
    
class InfoGainHeuristic(qi.Heuristic):
    def __init__(self, updater, rabi_eps, ramsey_eps, name=None, dview=None):
        self.updater = updater
        if dview is None:
            self._ham_model = m.RabiRamseyModel()
        else:
            self._ham_model = qi.DirectViewParallelizedModel(m.RabiRamseyModel(), dview, serial_threshold=1)
        self._risk_taker = qi.SMCUpdater(self._ham_model, updater.n_particles, SOME_PRIOR)
        self._update_risk_particles()
        self._rabi_eps = rabi_eps
        self._ramsey_eps = ramsey_eps
        self.name = "Expected Information Gain" if name is None else name
        self.risk_history = []
        
    def _update_risk_particles(self):
        self._risk_taker.particle_locations = self.updater.particle_locations
        self._risk_taker.particle_weights = self.updater.particle_weights
        
    def __call__(self, tp):
        ramsey_eps = self._ramsey_eps
        ramsey_eps['t'] = tp
        all_eps = np.concatenate([self._rabi_eps, ramsey_eps])
        
        self._update_risk_particles()
        
        eig = self._risk_taker.expected_information_gain(all_eps)
        self.risk_history += [eig]
        best_idx = np.argmax(eig, axis=0)
        eps = np.array([all_eps[best_idx]])
        return eps
    
class ExponentialHeuristic(qi.Heuristic):
    def __init__(self, updater, max_t=0.3, max_tau=2, base=11/10, n=50, n_bin=100, name=None):
        self.updater = updater
        self._rabi_eps = rabi_sweep(max_t=1, n=n, n_bin=n_bin)
        self._ramsey_eps = ramsey_sweep(max_tau=1, n=n, n_bin=n_bin)
        
        self._rabi_eps['t'] = max_t * (base ** np.arange(n)) / (base ** (n-1))
        self._ramsey_eps['tau'] = max_tau * (base ** np.arange(n)) / (base ** (n-1))
        
        self._rabi_eps['t'] = np.round(self._rabi_eps['t'] / 0.002) * 0.002
        self._ramsey_eps['tau'] = np.round(self._ramsey_eps['tau'] / 0.002) * 0.002
        
        self._idx = 0
        self.name = "Exponentially Sparse Heur" if name is None else name
        
    def __call__(self, tp):
        ramsey_eps = self._ramsey_eps
        ramsey_eps['t'] = tp
        all_eps = np.vstack([self._rabi_eps, ramsey_eps]).reshape((-1,), order='F')
        
        eps =  np.array([all_eps[self._idx]])
        self._idx += 1
        return eps
    
class LinearHeuristic(qi.Heuristic):
    def __init__(self, updater, max_t=0.3, max_tau=2, n=50, n_bin=100, name=None):
        self.updater = updater
        self._rabi_eps = rabi_sweep(max_t=1, n=n, n_bin=n_bin)
        self._ramsey_eps = ramsey_sweep(max_tau=1, n=n, n_bin=n_bin)
        
        self._rabi_eps['t'] = np.round(self._rabi_eps['t'] / 0.002) * 0.002
        self._ramsey_eps['tau'] = np.round(self._ramsey_eps['tau'] / 0.002) * 0.002
        
        self._idx = 0
        self.name = "Standard Linear Heuristic" if name is None else name
        
    def __call__(self, tp):
        ramsey_eps = self._ramsey_eps
        ramsey_eps['t'] = tp
        all_eps = np.vstack([self._rabi_eps, ramsey_eps]).reshape((-1,), order='F')
        
        eps =  np.array([all_eps[self._idx]])
        self._idx += 1
        return eps
