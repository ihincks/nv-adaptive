# -*- coding: utf-8 -*-
from __future__ import division
from future.utils import with_metaclass
import time
import warnings
import abc

import qinfer as qi
import numpy as np
import models as m
import dateutil
import socket
import datetime
from pandas import DataFrame, Panel, Timestamp, Timedelta, read_pickle

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
# FUNCTIONS
#-------------------------------------------------------------------------------

def asscalar(a):
    try:
        return np.asscalar(a)
    except AttributeError:
        return a

def compute_run_time(expparam):
    """
    Computes the amount of time it takes to run all repetitions
    of the given experiment back to back.
    """
    # this is the amount of time in AdaptiveTwoPulse.pp not
    # spent on the experiment's pulse sequence, in microseconds
    other_time = 27.65
    
    if expparam['emode'] == m.RabiRamseyModel.RABI:
        pulse_time = expparam['t']
    else:
        pulse_time = 2 * expparam['t'] + expparam['tau']
        
    return float(1e-6 * expparam['n_meas'] * (pulse_time + other_time))
    
def compute_single_eff_num_bits(alpha, beta, var_alpha=None, var_beta=None):
    """
    Helper function for compute_eff_num_bits. Returns effective number
    of strong measurements for given alpha and beta. If they are
    arrays, returns array of the same shape.
    """
    if var_alpha is None:
        var_alpha = alpha
    if var_beta is None:
        var_beta = beta
    return (alpha - beta)**2 / (3 * (alpha + beta) + 2 * (var_alpha + var_beta))


def compute_eff_num_bits(n_meas, updater):
    """
    Computes the number of effective number of strong measurements
    given the current value of the references. Taken as the mean over the
    distribution in updater.
    
    :param int n_meas: Number of repetitions of the experiment.
    :param ReferencedPoissonModel updater: This will have a distribution over 
         the paramters of alpha and beta describing the number of bright and
         dark photons expected at n_meas=1.
    """
    # hopefully hardcoding these indices doesn't come back to 
    # haunt me
    mu_alpha, mu_beta = updater.est_mean()[5:7]
    var_alpha, var_beta = np.diag(updater.est_covariance_mtx())[5:7]
    return asscalar(n_meas * compute_single_eff_num_bits(
        mu_alpha, mu_beta, 
        var_alpha=var_alpha, var_beta=var_beta
    ))

def get_now():
    return Timestamp(datetime.datetime.now())


def asscalar(a):
    try:
        return np.asscalar(a)
    except AttributeError:
        return a
        
def perform_update(heuristic, expparam, result, preceded_by_tracking, drift_tracking=False):
    updater = heuristic.updater
    n_meas = int(expparam['n_meas'])
    if drift_tracking:
        if preceded_by_tracking:
            heuristic.reset_reference_prior()
        updater.update_timestep(expparam)
        expparam['mode'] = m.ReferencedPoissonModel.BRIGHT
        bright = updater.update(result.bright, expparam)
        expparam['mode'] = m.ReferencedPoissonModel.DARK
        dark = updater.update(result.dark, expparam)
    else:
        dist = qi.ProductDistribution(
                qi.GammaDistribution(
                    mean=result.bright / n_meas, 
                    var=result.bright / n_meas**2
                ),
                qi.GammaDistribution(
                    mean= result.dark / n_meas,
                    var= result.dark / n_meas**2
                )
            )
        n_mps = updater.model.base_model.n_modelparams
        updater.particle_locations[:,n_mps:n_mps+2] = dist.sample(updater.n_particles)
    expparam['mode'] = m.ReferencedPoissonModel.SIGNAL
    updater.update(result.signal, expparam)

def rabi_sweep(min_t=None, max_t=0.3, n=50, n_meas=None, wo=0, mode=None, include_tp2=False):
    if include_tp2:
        ham_model = m.RabiRamseyExtendedModel(1,1)
        if min_t is None:
            min_t = max_t / n
        vals = [
            np.linspace(min_t, max_t, n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.ones(n) * wo,
            np.ones(n) * ham_model.RABI
        ]
    else:
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
    if n_meas is not None:
        mode = m.ReferencedPoissonModel.SIGNAL if mode is None else mode
        vals = vals + [(mode * np.ones(n)).astype(np.int)]
        vals = vals + [(n_meas * np.ones(n)).astype(np.int)]
        dtype = dtype + [('mode', 'int'),('n_meas','int')]
    rabi_eps = np.array(vals).T
    rabi_eps = np.array(list(zip(*rabi_eps.T)), dtype=dtype)
    return rabi_eps

def ramsey_sweep(min_tau=None, max_tau=2, tp=0.01, tp2=None, phi=0, n=50, n_meas=None, wo=0, mode=None, include_tp2=False):
    if include_tp2 or tp2 is not None:
        ham_model = m.RabiRamseyExtendedModel(1,1)
        tp2 = tp if tp2 is None else tp2
        if min_tau is None:
            min_tau = max_tau / n
        vals = [
            tp * np.ones(n),
            tp2 * np.ones(n),
            np.linspace(min_tau, max_tau, n),
            phi * np.ones(n),
            np.ones(n) * wo,
            np.ones(n) * ham_model.RAMSEY
        ]
    else:
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
    dtype = ham_model.expparams_dtype
    if n_meas is not None:
        mode = m.ReferencedPoissonModel.SIGNAL if mode is None else mode
        vals = vals + [(mode * np.ones(n)).astype(np.int)]
        vals = vals + [(n_meas * np.ones(n)).astype(np.int)]
        dtype = dtype + [('mode', 'int'),('n_meas','int')]
    ramsey_eps = np.array(vals).T
    ramsey_eps = np.array(list(zip(*ramsey_eps.T)), dtype=dtype)
    return ramsey_eps
    
#-------------------------------------------------------------------------------
# TIME STEPPERS
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
        self._kappa = OrnsteinUhlenbeckStepper(1, sigma_kappa, theta_kappa, mu_kappa, 0)
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

#-------------------------------------------------------------------------------
# COMMUNICATION
#-------------------------------------------------------------------------------

class TCPClient(object):
    """
    Wraps socket.socket to make it a bit more suitable for receiving and
    transmitting terminated strings.
    """
        
    TERMINATOR = '\n'
    INPUT_TIMEOUT = 10

    def __init__(self, server_ip, server_port):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_ip = server_ip
        self._server_port = server_port
        # we implement our own buffer just to make bytes_available
        # a thing, since socket.socket doesn't have it.
        self._buffer = ''

    def connect(self):
        self._socket.connect((self._server_ip, self._server_port))
        self._socket.setblocking(0)

    def close(self):
        self._socket.close()

    def _fill_buffer(self):
        is_complete = False
        while not is_complete:
            try:
                self._buffer += self._socket.recv(8192)
                time.sleep(0.01)
            except:
                is_complete = True


    @property
    def bytes_available(self):
        self._fill_buffer()
        return len(self._buffer)

    def flush_input(self):
        self._fill_buffer()
        self._buffer =  ''

    def write_string(self, string):
        self._socket.send(string + TCPClient.TERMINATOR)

    def read_string(self):
        self._fill_buffer()
        message, sep, end = self._buffer.partition(TCPClient.TERMINATOR)
        self._buffer = end
        return message + sep

    def read_string_until_terminator(self, timeout=None):
        timeout = self.INPUT_TIMEOUT if timeout is None else timeout
        message = self.read_string()
        t = time.time()
        while len(message) == 0 or ord(message[-1]) != ord(TCPClient.TERMINATOR):
            message += self.read_string()
            time.sleep(0.01)
            if timeout > 0 and time.time() - t > timeout:
                raise IOError('TCP input stream timeout waiting for terminator. So far we have: \n"{}"'.format(message))
        return message

#-------------------------------------------------------------------------------
# JOBS and RESULTS
#-------------------------------------------------------------------------------


class ExperimentResult(object):
    def __init__(self, bright, dark, signal, timestamp=None, preceded_by_tracking=False):
        self._bright = bright
        self._dark = dark
        self._signal = signal
        self._preceded_by_tracking = preceded_by_tracking
        self._timestamp = get_now() if timestamp is None else timestamp
        
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
    def preceded_by_tracking(self):
        return self._preceded_by_tracking
    @property
    def timestamp(self):
        return self._timestamp
    @property
    def triplet(self):
        return np.array([self.bright, self.dark, self.signal])
        
class ExperimentJob(with_metaclass(abc.ABCMeta, object)):
    def __init__(self):
        self.submitted_ts = get_now()
        self.returned_ts = None
        
    @abc.abstractproperty
    def is_complete(self):
        pass
        
    @abc.abstractmethod
    def get_result(self):
        if not self.is_complete:
            raise RuntimeError('Job is not complete.')
        self.returned_ts = get_now()
            
class OfflineExperimentJob(ExperimentJob):
    def __init__(self):
        super(OfflineExperimentJob, self).__init__()
        self._is_complete = False
    
    @property    
    def is_complete(self):
        return self._is_complete
        
    def push_result(self, bright, dark, signal):
        self._result = ExperimentResult(bright, dark, signal)
        self._is_complete = True
        
    def get_result(self):
        super(OfflineExperimentJob, self).get_result()
        return self._result
        
class TopChefExperimentJob(ExperimentJob):
    def __init__(self, topchef_job):
        super(TopChefExperimentJob, self).__init__()
        self._job = topchef_job
    
    @property    
    def is_complete(self):
        return self._job.is_complete
        
    def get_result(self):
        super(TopChefExperimentJob, self).get_result()
        result = self._job.result
        ts = dateutil.parser.parse(result['time_completed'])
        ts = Timestamp(ts.replace(tzinfo=None))
        
        preceded_by_tracking = False
        bright = result['light_count']
        dark = result['dark_count']
        signal = result['result_count']
        # our ghetto way of encoding tracks in the JSON result
        if bright > 1e6:
            bright, dark, signal = bright - 1e6, dark - 1e6, signal - 1e6
            preceded_by_tracking = True
            
        return ExperimentResult(
            bright, dark, signal, 
            preceded_by_tracking=preceded_by_tracking, 
            timestamp=ts
        )
        
class TCPExperimentJob(ExperimentJob):
    def __init__(self, job_id, tcp_client):
        super(TCPExperimentJob, self).__init__()
        self.job_id = job_id
        self._tcp_client = tcp_client
        self._cached_result = None
    
    @property    
    def is_complete(self):
        if self._cached_result is not None:
            return True
        else:
            return self._tcp_client.bytes_available > 0
        
    def get_result(self):
        super(TCPExperimentJob, self).get_result()

        if self._cached_result is None:
            message = self._tcp_client.read_string_until_terminator()

            job_id, bright, dark, signal, pbt, ts = message.strip().split(',')

            job_id = int(job_id)
            assert job_id == self.job_id

            bright, dark, signal = int(bright), int(dark), int(signal)

            preceded_by_tracking = bool(pbt)
            ts = dateutil.parser.parse(ts)
            ts = Timestamp(ts.replace(tzinfo=None))

            self._cached_result = ExperimentResult(
                bright, dark, signal, 
                preceded_by_tracking=preceded_by_tracking, 
                timestamp=ts
            )

        return self._cached_result

#-------------------------------------------------------------------------------
# RUNNERS
#-------------------------------------------------------------------------------

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
    def run_experiment(self, expparam, precede_by_tracking=False):
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
        t_experiment = expparam['tau'] + 2 * expparam['t']
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
        
    def run_experiment(self, expparam, precede_by_tracking=False):
        super(SimulatedRabiRamseyExperimentRunner, self).run_experiment(expparam)
        job = OfflineExperimentJob()
        self.update_timestep(expparam)
        expparam['mode'] = m.ReferencedPoissonModel.BRIGHT
        bright = self._model.simulate_experiment(self.modelparam, expparam)
        expparam['mode'] = m.ReferencedPoissonModel.DARK
        dark = self._model.simulate_experiment(self.modelparam, expparam)
        expparam['mode'] = m.ReferencedPoissonModel.SIGNAL
        signal = self._model.simulate_experiment(self.modelparam, expparam)
        job.push_result(bright, dark, signal)
        return job
        
class TopChefRabiRamseyExperimentRunner(AbstractRabiRamseyExperimentRunner):
    def __init__(self, topchef_service):
        super(TopChefRabiRamseyExperimentRunner, self).__init__()
        
        self._service = topchef_service
        
        self._n_shots = 100000
        self._meas_time = 800e-9
        self._center_freq = 2.87e9
        self._intermediate_freq = 0
        self._adiabatic_power = -2

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
           'adiabatic_power': self._adiabatic_power,
           'precede_by_tracking': False
       }
        
    def run_experiment(self, expparam, precede_by_tracking=False):
        super(TopChefRabiRamseyExperimentRunner, self).run_experiment(expparam)
        
        eps = self._bare_eps
        
        if expparam['emode'] == m.RabiRamseyModel.RABI:
            eps['pulse1_time'] = 1e-6 * expparam['t']
            eps['pulse1_offset_freq'] = expparam['wo']
        elif expparam['emode'] == m.RabiRamseyModel.RAMSEY:
            eps['delay_time'] = 1e-6 * expparam['tau']
            eps['pulse1_time'] = 1e-6 * expparam['t']
            eps['pulse2_time'] = 1e-6 * expparam['t']
            eps['pulse2_phase'] = expparam['phi']
            eps['pulse1_offset_freq'] = expparam['wo']
            eps['pulse2_offset_freq'] = expparam['wo']
        eps['center_freq'] = eps['center_freq']
        eps['number_of_repetitions'] = expparam['n_meas']
        
        if precede_by_tracking:
            eps['precede_by_tracking'] = True
        
        # the JSON parser doesn't like getting np arrays
        for key in eps.keys():
            eps[key] = asscalar(eps[key])
            
        topchef_job = self._service.new_job(eps)

        return TopChefExperimentJob(topchef_job)
        
        

class TCPRabiRamseyExperimentRunner(AbstractRabiRamseyExperimentRunner):
    def __init__(self, job_client, results_client):
        super(TCPRabiRamseyExperimentRunner, self).__init__()
        
        self._job_client = job_client
        self._results_client = results_client

    def run_tracking(self):
        raise NotImplemented()
    
    @staticmethod    
    def make_job_string(n_meas=100000, meas_time=800e-9, center_freq=2.87e9, 
            intermediate_freq=50e6, adiabatic_power=-18, delay_time=0,
            pulse1_time=0, pulse1_phase=0, pulse1_power=0, pulse1_offset_freq=0,
            pulse1_modulation_freq=0, pulse1_modulation_phase=0,
            pulse2_time=0, pulse2_phase=0, pulse2_power=0, pulse2_offset_freq=0,
            pulse2_modulation_freq=0, pulse2_modulation_phase=0,
            precede_by_tracking=False, job_id=0
            ):
        job_string = '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}'
        job_string = job_string.format(
            int(precede_by_tracking), int(n_meas), 
            meas_time, center_freq, intermediate_freq,
            adiabatic_power,
            delay_time,
            pulse1_time, pulse1_phase, pulse1_power, pulse1_offset_freq,
            pulse1_modulation_freq, pulse1_modulation_phase,
            pulse2_time, pulse2_phase, pulse2_power, pulse2_offset_freq,
            pulse2_modulation_freq, pulse2_modulation_phase,
            int(job_id)
        )
        return job_string
        
        
    def run_experiment(self, expparam, precede_by_tracking=False):
        super(TCPRabiRamseyExperimentRunner, self).run_experiment(expparam)
        
        job_id = np.random.randint(1e6)
        
        if expparam['emode'] == m.RabiRamseyModel.RABI:
            job_string = self.make_job_string(
                job_id=job_id,
                pulse1_time = asscalar(1e-6 * expparam['t']), 
                pulse1_offset_freq = asscalar(1e6 * expparam['wo']), 
                n_meas=asscalar(expparam['n_meas']),
                precede_by_tracking=precede_by_tracking
            )
        elif expparam['emode'] == m.RabiRamseyModel.RAMSEY:
            job_string = self.make_job_string(
                job_id=job_id,
                delay_time = asscalar(1e-6 * expparam['tau']),
                pulse1_time = asscalar(1e-6 * expparam['t']),
                pulse2_time = asscalar(1e-6 * expparam['t']),
                pulse1_offset_freq = asscalar(1e6 * expparam['wo']), 
                pulse2_offset_freq = asscalar(1e6 * expparam['wo']), 
                pulse2_phase = asscalar(expparam['phi']),
                n_meas=asscalar(expparam['n_meas']),
                precede_by_tracking=precede_by_tracking
            )
        else:
            raise AttributeError('Unknown experiment.')

        self._job_client.flush_input()
        self._job_client.write_string(job_string)

        return TCPExperimentJob(job_id, self._results_client)


#-------------------------------------------------------------------------------
# HEURISTICS
#-------------------------------------------------------------------------------

class RiskHeuristic(qi.Heuristic):
    def __init__(self, updater, Q, rabi_eps, ramsey_eps, name=None, dview=None, subsample_particles=None):
        self.updater = updater
        if dview is None:
            self._ham_model = m.RabiRamseyModel()
        else:
            self._ham_model = qi.DirectViewParallelizedModel(m.RabiRamseyModel(), dview, serial_threshold=1)
        self._ham_model._Q = Q
        self.n_particles = updater.n_particles if subsample_particles is None else subsample_particles
        self._risk_taker = qi.SMCUpdater(self._ham_model, self.n_particles, SOME_PRIOR)
        self._update_risk_particles()
        self._rabi_eps = rabi_eps
        self._ramsey_eps = ramsey_eps
        self.name = "Bayes Risk, Q={}".format(Q) if name is None else name
        self.risk_history = []
        
    def _update_risk_particles(self):
        n_mps = self._risk_taker.model.base_model.n_modelparams
        if self.n_particles == self.updater.n_particles:
            locs = self.updater.particle_locations[:,:n_mps]
            weights = self.updater.particle_weights
        else:
            locs = self.updater.sample(n=self.n_particles)[:,:n_mps]
            weights = np.ones(self.n_particles) / self.n_particles
        self._risk_taker.particle_locations = locs
        self._risk_taker.particle_weights = weights
        
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

class MCRiskHeuristic(qi.Heuristic):
    def __init__(self, updater, rabi_eps, ramsey_eps, name=None, dview=None,n_particle_subset=1000):
        self.updater = updater
        self._update_risk_particles()
        self._rabi_eps = rabi_eps
        self._ramsey_eps = ramsey_eps
        self.name = "MC Integrated Bayes Risk, Q={}".format(Q) if name is None else name
        self.risk_history = []
        
    def _update_risk_particles(self):
       # self._risk_taker.particle_locations = self.updater.particle_locations
       # self._risk_taker.particle_weights = self.updater.particle_weights
       pass  
    def __call__(self, tp):
        ramsey_eps = self._ramsey_eps
        ramsey_eps['t'] = tp
        all_eps = np.concatenate([self._rabi_eps, ramsey_eps])
        
        self._update_risk_particles()
        
        risk = self.updater.bayes_risk(all_eps,n_particle_subset)
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
        n_mps = self._risk_taker.model.base_model.n_modelparams
        self._risk_taker.particle_locations = self.updater.particle_locations[:,:n_mps]
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
    def __init__(self, updater, max_t=0.3, max_tau=2, base=11/10, n=50, name=None):
        self.updater = updater
        # n_meas should be overwritten by the TrackingHeuristic wrapper
        self._rabi_eps = rabi_sweep(max_t=1, n=n, n_meas=1000)
        self._ramsey_eps = ramsey_sweep(max_tau=1, n=n, n_meas=1000)
        
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
    def __init__(self, updater, max_t=0.3, max_tau=2, n=50, name=None):
        self.updater = updater
        # n_meas should be overwritten by the TrackingHeuristic wrapper
        self._rabi_eps = rabi_sweep(max_t=max_t, n=n, n_meas=1000)
        self._ramsey_eps = ramsey_sweep(max_tau=max_tau, n=n, n_meas=1000)
        
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
        
class PredeterminedSingleAdaptHeuristic(qi.Heuristic):
    def __init__(self, updater, rabi_eps, ramsey_eps, name=None):
        self.updater = updater
        self.n_rabi = rabi_eps.size
        self._eps = np.concatenate([rabi_eps, ramsey_eps])
        
        self._idx = 0
        self._tp = 0
        self.name = "Predetermined with single adaptation Heuristic" if name is None else name
        
    def __call__(self, tp):
        if self._idx == self.n_rabi:
            self._tp = tp
        eps = self._eps[self._idx, np.newaxis]
        if self._idx >= self.n_rabi:
            eps['t'] = self._tp

        self._idx += 1
        return eps
        
class DataFrameHeuristic(qi.Heuristic):
    def __init__(self, updater, df):
        self.df = df
        self._idx = 0
        self.name = 'DataFrameHeuristic({})'.format(df.heuristic[0])
        self.updater = updater
        
        self.std_mult = 3
        n_meas = df.expparam[1]['n_meas']
        self._initial_bright_mean = df.bright[1] / n_meas
        self._initial_dark_mean = df.dark[1] / n_meas
        self._initial_bright_std = np.sqrt(df.bright[1]) / n_meas
        self._initial_dark_std = np.sqrt(df.dark[1]) / n_meas
        
    def __call__(self, tp):
        # we purposely start from idx=1, since there is no expparam 
        # at idx=0
        self._idx += 1
        return self.df.expparam[self._idx]
        
    def reset_reference_prior(self):
        dist = qi.ProductDistribution(
            qi.GammaDistribution(
                mean=self._initial_bright_mean, 
                var=(self.std_mult * self._initial_bright_std)**2
            ),
            qi.GammaDistribution(
                mean=self._initial_dark_mean, 
                var=(self.std_mult * self._initial_dark_std)**2
            )
        )
        samples = dist.sample(self.updater.n_particles)
        n_mps = self.updater.model.base_model.n_modelparams
        self.updater.particle_locations[:,n_mps:n_mps+2] = samples
        
class TrackingHeuristic(qi.Heuristic):
    """
    Wraps an existing heuristic, so that calling it returns a tuple 
    (eps, precede_by_tracking) where eps is the experiment that the underlying
    heuristic wants.
    
    Also introduces the idea of an initial reference experiment, which
    sets an empirical prior on the reference coordinates of the distribution.
    
    :param qinfer.Heuristic heuristic: The heuristic to wrap.
    :param float std_tracking: How many stds below the initial bright reference
        we must go before demanding a tracking operation.
    :param bool track_on_initial_reference: Whether to precede the initial 
        reference taking by a tracking operation.
    :param float std_mult: How much bigger than 1 standard deviation we
        should set the reference prior width.
    """
    
    # the hardware will have trouble doing too many shots because of 
    # certain timeout counters in the the nv-command-center code
    MAX_N_MEAS = 500000
    
    def __init__(self, heuristic, std_tracking=5, track_on_initial_reference=True, std_mult=3, n_meas=None, eff_num_bits=10):
        self.underlying_heuristic = heuristic
        self.std_tracking = std_tracking
        self.has_initial_reference = False
        self.track_on_initial_reference = track_on_initial_reference
        self.std_mult = std_mult
        
        self._initial_bright_mean = None
        self._initial_dark_mean = None
        self._initial_bright_std = None
        self._initial_dark_std = None
        
        self._n_meas = n_meas
        if n_meas is None and eff_num_bits is None:
            warnings.warn('Neither n_meas nor eff_num_bits was specified; defaulting to 200000.')
            self._n_meas = 200000
        self.eff_num_bits = eff_num_bits
    
    @property
    def n_meas(self):
        """
        Returns the number of shots to do on the next experiment. This might
        depend on the reference counts.
        """
        if self.eff_num_bits is not None:
            # solve for the number of shots we need to reach eff_num_bits
            single_shot = compute_eff_num_bits(1, self.updater)
            self._n_meas = self.eff_num_bits / single_shot
        return min(self._n_meas, TrackingHeuristic.MAX_N_MEAS)
        
    @property
    def updater(self):
        return self.underlying_heuristic.updater
        
    @property
    def name(self):
        return self.underlying_heuristic.name + ' (with tracking)'
        
    def reset_reference_prior(self):
        """
        Resets the alpha and beta coordinates of the updater to an empirical
        gamma distribution based on the initial reference taking.
        """
        if self.has_initial_reference:
            dist = qi.ProductDistribution(
                qi.GammaDistribution(
                    mean=self._initial_bright_mean, 
                    var=(self.std_mult * self._initial_bright_std)**2
                ),
                qi.GammaDistribution(
                    mean=self._initial_dark_mean, 
                    var=(self.std_mult * self._initial_dark_std)**2
                )
            )
            samples = dist.sample(self.updater.n_particles)
            n_mps = self.updater.model.base_model.n_modelparams
            self.updater.particle_locations[:,n_mps:n_mps+2] = samples
        else:
            warnings.warn('Reference experiment has not been made yet; call `take_initial_reference`')
        
    def take_initial_reference(self, experiment, n_meas=300000, n_repetitions=1):
        """
        Runs a job to set the initial distribution on the reference prior, 
        and to 
        """
        eps = rabi_sweep(0, n=1, n_meas=n_meas)
        counts = np.empty((n_repetitions, 3))
        for idx_repetition in range(n_repetitions):
            if idx_repetition == 0:
                precede_by_tracking = self.track_on_initial_reference 
            else:
                precede_by_tracking = False
            job = experiment.run_experiment(eps, precede_by_tracking)
            while not job.is_complete:
                time.sleep(0.4)
            counts[idx_repetition, :] = job.get_result().triplet
            
        bright, dark, signal = np.sum(counts, axis=0)
        # bright and signal are the same experiment since there is no
        # pulsing, so we might as well use the data (bright+signal)
        bright = bright + signal
        self._initial_bright_mean = bright / (2 * n_meas * n_repetitions)
        self._initial_dark_mean = dark / (n_meas * n_repetitions)
        self._initial_bright_std = np.sqrt(bright) / (2 * n_meas * n_repetitions)
        self._initial_dark_std = np.sqrt(dark) / (n_meas * n_repetitions)
        
        self.has_initial_reference = True
        
        self.reset_reference_prior()
        
        
    def _decide_on_tracking(self):
        n_mps = self.updater.model.base_model.n_modelparams
        bright_est = self.updater.est_mean()[n_mps]
        return bright_est < self._initial_bright_mean - self.std_tracking * self._initial_bright_std
        
        
    def __call__(self, tp):
        if not self.has_initial_reference:
            raise RuntimeError('take_initial_reference must be called before an experiment can be suggested.')
        n_meas = self.n_meas
        precede_by_tracking = self._decide_on_tracking()
        if hasattr(self.underlying_heuristic, '_rabi_eps') and hasattr(self.underlying_heuristic, '_ramsey_eps'):
            # we change this in case the underlying heuristic uses it
            self.underlying_heuristic._rabi_eps['n_meas'] = n_meas
            self.underlying_heuristic._ramsey_eps['n_meas'] = n_meas
        eps = self.underlying_heuristic(tp)
        eps['n_meas'] = n_meas
        return eps, precede_by_tracking


class QPriorRiskHeuristic(RiskHeuristic):
    def __init__(self, updater, Q, rabi_eps, ramsey_eps, name=None, dview=None, subsample_particles=None):
        

        super(QPriorRiskHeuristic, self).__init__(updater, Q, rabi_eps, ramsey_eps, name=name, dview=dview, 
                                                subsample_particles=subsample_particles)
        
        self._prior_covariance = updater.est_covariance_mtx()
        self._ham_model._Q = Q / np.diag(self.prior_covariance)[:self._ham_model.n_modelparams]
        self.name = "QPrior Bayes Risk, Q={}".format(Q) if name is None else name
        

    @property
    def prior_covariance(self):
        return self._prior_covariance



class FullRiskHeuristic(qi.Heuristic):
    def __init__(self, updater, Q, rabi_eps, ramsey_eps, name=None, dview=None, subsample_particles=None, n_outcome_samples = 250,
                    var_fun='simplified',batch=None):
        self.updater = updater
        self.updater.model._Q = Q

        if n_outcome_samples is not None:
            self.updater.model._n_outcomes = n_outcome_samples
        
        self.n_particles = updater.n_particles if subsample_particles is None else subsample_particles
        self._rabi_eps = rabi_eps
        self._ramsey_eps = ramsey_eps
        self.name = "Full Bayes Risk, Q={}".format(Q) if name is None else name
        self.risk_history = []
        self.var_fun = var_fun
        self.batch = batch 
        
    def __call__(self, tp):
        ramsey_eps = self._ramsey_eps
        ramsey_eps['t'] = tp
        all_eps = np.concatenate([self._rabi_eps, ramsey_eps])

        risk = self.updater.bayes_risk(all_eps,use_cached_samples=True,cache_samples=True,
                n_particle_subset=self.n_particles,var_fun=self.var_fun,batch=self.batch)
        self.risk_history += [risk]
        best_idx = np.argmin(risk, axis=0)
        eps = np.array([all_eps[best_idx]])
        return eps

class TrackingFullRiskHeuristic(qi.Heuristic):
    def __init__(self, updater, Q, rabi_eps, ramsey_eps, name=None, dview=None, subsample_particles=None,
                 n_outcome_samples = 250,var_fun='simplified',batch=None):
        self.updater = updater
        
        if dview is None:
            self._ref_model = m.ReferencedPoissonModel(m.RabiRamseyModel())
        else:
            self._ref_model = m.ReferencedPoissonModel(
                    qi.DirectViewParallelizedModel(
                        m.RabiRamseyModel(),
                        dview,
                        serial_threshold=1
                    ),
                    dview=dview
                )
        self._ref_model._Q = Q
        self.subsample_particles = subsample_particles
        self.n_particles = updater.n_particles
        if n_outcome_samples is not None:
            self._ref_model._n_outcomes = n_outcome_samples

        self.name = "Tracking Full Bayes Risk, Q={}".format(Q) if name is None else name
        prior = qi.ProductDistribution(SOME_PRIOR, qi.UniformDistribution([[0,1],[0,1]]))
        self._risk_taker = qi.SMCUpdater(self._ref_model, self.n_particles, prior, dview=dview)
        self._risk_taker.particle_locations = self.updater.particle_locations[:,self._risk_taker.model.n_modelparams:]
        self._risk_taker.particle_weights = self.updater.particle_weights

        self._update_risk_particles()
        self._rabi_eps = rabi_eps
        self._ramsey_eps = ramsey_eps
        
        self.var_fun = var_fun
        self.batch = batch 
        self.risk_history = []
        
    def _update_risk_particles(self):
        n_mps = self._ref_model.n_modelparams
       
        locs = self.updater.particle_locations[:,:n_mps]
        weights = self.updater.particle_weights

        self._risk_taker.particle_locations = locs
        self._risk_taker.particle_weights = weights
        
    def __call__(self, tp):
        ramsey_eps = self._ramsey_eps
        ramsey_eps['t'] = tp
        all_eps = np.concatenate([self._rabi_eps, ramsey_eps])
        
        self._update_risk_particles()
        
        risk = self._risk_taker.bayes_risk(all_eps,use_cached_samples=False,cache_samples=True,
                n_particle_subset=self.subsample_particles,var_fun=self.var_fun,batch=self.batch)
        
        self.risk_history += [risk]
        best_idx = np.argmin(risk, axis=0)
        eps = np.array([all_eps[best_idx]])
        return eps
