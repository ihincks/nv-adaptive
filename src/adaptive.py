# -*- coding: utf-8 -*-
from __future__ import division
from future.utils import with_metaclass
import warnings
import abc
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from time import sleep

import qinfer as qi
import numpy as np
import models as m
import datetime
import dateutil
from pandas import DataFrame, Panel, Timestamp, Timedelta, read_pickle
import wquantiles


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
        
def get_now():
    return Timestamp(datetime.datetime.now())
    
#-------------------------------------------------------------------------------
# FUNCTIONS
#-------------------------------------------------------------------------------

def est_std(p, alpha, beta):
    return np.sqrt(p*(p+1)*alpha + (p-1)*(p-2)*beta)/(alpha-beta)
    
def add_counts_by_unique_expparams(df):
    # need to hash the expparams for pandas to group them
    groups = df[1:].groupby(df[1:].expparam.apply(lambda x: '{}'.format(x)))
    
    bright = np.array(list(groups['bright'].sum()))
    dark = np.array(list(groups['dark'].sum()))
    signal = np.array(list(groups['signal'].sum()))
    
    eps = np.array(list(groups['expparam'].first())).flatten()
    
    return eps, bright, dark, signal

def normalized_signal_by_unique_expparams(df):
    eps, bright, dark, signal = add_counts_by_unique_expparams(df)
    p = (signal - dark).astype(float) / (bright - dark)
    p_stds = est_std(p, bright, dark)
    return eps, p, p_stds

def normalized_and_separated_signal(df):
    eps, p, p_stds = normalized_signal_by_unique_expparams(df)
    rabi_idx = eps['emode'] == m.RabiRamseyModel.RABI
    rabi_eps = eps[rabi_idx]
    rabi_p = p[rabi_idx]
    rabi_p_stds = p_stds[rabi_idx]
    ramsey_idx = np.logical_not(rabi_idx)
    ramsey_eps = eps[ramsey_idx]
    ramsey_p = p[ramsey_idx]
    ramsey_p_stds = p_stds[ramsey_idx]
    return rabi_eps, rabi_p, rabi_p_stds, ramsey_eps, ramsey_p, ramsey_p_stds  
    
#-------------------------------------------------------------------------------
# DATA STORAGE
#-------------------------------------------------------------------------------

def new_experiment_dataframe(heuristic):
    """
    Returns an empty `pandas.DataFrame` to store data from 
    a single run of a single heuristic. Rows of this data 
    structure correspond to single experimental configurations
    on an NV center, summed over repetitions (n_meas). Columns
    are as follows:
    
    `expparam`:         `expparam` object used by the updater's simulator, of 
                        shape `(1,)`, and dtype `updater.model.expparam_dtype`.
    `n_meas`:           The number of repetitions performed.
    `bright`:           Number of bright reference photons measured, integer.
    `dark`:             Number of dark reference photons measured, integer.
    `signal`:           Number of signal reference photons measured, integer.
    `completed_ts`:     Timestamp of when experiment was completed on the 
                        physical hardware.
    `returned_ts`:      Timestamp of when experiment was returned to python.
    `submitted_ts`:     Timestamp of when experiment job was submitted.
    `diff_time`:        Time between this `completed_ts` and the previous
                        `completed_ts`.
    `run_time`:         Actual experiment time, without server/hardware 
                        overhead, in seconds
    `wall_time`:        Time between job being requested, and result returned,
                        in seconds.
    `overhead_time`:    `e_wall_time` minus `e_run_time`.
    `cum_wall_time`:    Cumulative time of this dataframe as measured by a 
                        wall clock, marked by the `returned_ts` timestamps.
    `cum_run_time`:     Cumulative time of actual experiment running.
    `preceded_by_tracking`:     Whether this experiment was directly preceded by a 
                        tracking operation, bool.
    `eff_num_bits`:     Number of effective bits (strong measurments) 
                        collected in this experiment.
    `cum_eff_num_bits`: Cumulative number of effective bits collected in this
                        dataframe.
    `heuristic`:        Description of this heuristic.
    `heuristic_value`:  Some kind of performance metric, as calculated prior to
                        this experiment.
    `smc_mean`:         Mean value of updater.
    `smc_cov`:          Covariance matrix of updater.
    `smc_n_eff_particles`:    Number of effective particles in updater.
    `smc_upper_quantile`:     Upper 0.95 quantile of the paramaters.
    `smc_lower_quantile`:     Lower 0.95 quantile of the parameters.
    `smc_resample_count`:     Number of resamples done by updater so far.
    
    :param qinfer.Heuristic updater: By convention, it is nice to store 
        what's going on before any data have arrived.
    """
    df = DataFrame(
        columns=[
            'expparam',
            'bright','dark','signal',
            'completed_ts', 'returned_ts', 'submitted_ts',
            'diff_time', 'run_time', 'wall_time', 'overhead_time', 
            'cum_wall_time', 'cum_run_time',
            'preceded_by_tracking',
            'eff_num_bits', 'cum_eff_num_bits',
            'heuristic', 'heuristic_value',
            'smc_mean', 'smc_cov', 'smc_n_eff_particles', 'smc_resample_count',
            'smc_upper_quantile', 'smc_lower_quantile'
        ])
    df = append_experiment_data(df, heuristic=heuristic, preceded_by_tracking=True)
    
    return df
    
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
    
def compute_eff_num_bits(n_meas, updater):
    """
    Computes the number of effective number of strong measurements
    given the current value of the references.
    """
    # hopefully hardcoding these indices doesn't come back to 
    # haunt me
    alpha = updater.particle_locations[:,5]
    beta = updater.particle_locations[:,6]
    n_eff = (alpha - beta)**2 / (5 * (alpha + beta))
    return float(n_meas * np.dot(updater.particle_weights, n_eff))
    

def append_experiment_data(
        dataframe, 
        expparam=None, heuristic=None, 
        result=None, job=None, 
        preceded_by_tracking=False, heuristic_value=None
    ):
    """
    Appends the results from a single experiment to a DataFrame, returning 
    the new object.
    
    :param DataFrame dataframe: See `new_experiment_dataframe`.
    :param np.ndarray expparam: Of dtype `heuristic.updater.model.expparam_dtype`
    :param ExperimentResult result: Result of the experiment. This is 
        overwritted by job.get_result() if it exists. Normally parameter 
        should not be input, and instead `job.get_result()` should be used.
    :param AbstractExperimentJob job: Job that was run.
    :param qi.Heuristic heuristic: Heuristic, containing the updater.
    :param heuristic_value: Whatever we want to store here, depends on 
        heuristic.
    """
    n_rows = dataframe.shape[0]
    first_row = n_rows == 0
    data = {'heuristic_value': heuristic_value}
    
    if heuristic is not None:
        updater = heuristic.updater
        data['heuristic'] = heuristic.name
    else:
        updater = None
        
    if updater is not None:
        data['smc_mean'] = updater.est_mean()
        data['smc_cov'] = updater.est_covariance_mtx()
        data['smc_lower_quantile'] = wquantiles.quantile(
            heuristic.updater.particle_locations.T,
            heuristic.updater.particle_weights,
            0.05
        )
        data['smc_upper_quantile'] = wquantiles.quantile(
            heuristic.updater.particle_locations.T,
            heuristic.updater.particle_weights,
            0.95
        )
        data['smc_n_eff_particles'] = updater.n_ess
        data['smc_resample_count'] = updater.resample_count
        if first_row:
            # this if statement will usually be invoked by the thing 
            # initializing the data frame
            data['eff_num_bits'] = 0
            data['cum_eff_num_bits'] = 0
            
            data['run_time'] = Timedelta(0, 's')
            data['cum_run_time'] = Timedelta(0, 's')
            
            data['wall_time'] = Timedelta(0, 's')
            data['cum_wall_time'] = Timedelta(0, 's')
            # these timestamps are a lie, but they remove a 
            # allow the first row of the dataframe to be pre-data
            now =  get_now()
            data['submitted_ts'] = now
            data['returned_ts'] = now
            data['completed_ts'] = now
            data['diff_time'] = Timedelta(0, 's')
            data['overhead_time'] = Timedelta(0, 's')
        else:
            if expparam is not None:
                data['eff_num_bits'] = compute_eff_num_bits(
                    expparam['n_meas'], updater
                )
            prev_cenb = dataframe.loc[n_rows-1, 'cum_eff_num_bits']
            data['cum_eff_num_bits'] = prev_cenb + data['eff_num_bits']
    
    if expparam is not None:
        data['expparam'] = expparam
        data['n_meas'] = expparam['n_meas']
        data['run_time'] = Timedelta(compute_run_time(expparam), 's')
        if first_row:
            data['cum_run_time'] = data['run_time']
        else:
            prev_cum_run_time = dataframe.loc[n_rows-1, 'cum_run_time']
            data['cum_run_time'] = prev_cum_run_time + data['run_time']
    
    if job is not None:
        result = job.get_result()
        data['submitted_ts'] = job.submitted_ts
        data['returned_ts'] = job.returned_ts
        data['wall_time'] = job.returned_ts - job.submitted_ts
        data['overhead_time'] = data['wall_time'] - data['run_time']
        if first_row:
            data['cum_wall_time'] = data['wall_time']
        else:
            prev_return = dataframe.loc[n_rows-1, 'returned_ts']
            prev_cum_wall_time = dataframe.loc[n_rows-1, 'cum_wall_time']
            data['cum_wall_time'] = prev_cum_wall_time \
                + job.returned_ts - prev_return

    if result is not None:
        data['bright'] = result.bright
        data['dark'] = result.dark
        data['signal'] = result.signal
        data['preceded_by_tracking'] = result.preceded_by_tracking
        data['completed_ts'] = result.timestamp
        if first_row:
            data['diff_time'] = Timedelta(0, 's')
        else:
            prev_timestamp = dataframe.loc[n_rows-1, 'completed_ts']
            data['diff_time'] = result.timestamp - prev_timestamp
        
    return dataframe.append(data, ignore_index=True)
        
class HeuristicData(object):
    """
    Stores the data from trials of a single heuristic into a pandas.Panel
    of DataFrames, as constructed with new_experiment_dataframe and 
    append_experiment_data.
    """
    def __init__(self, filename):
        self._filename = filename
        try:
            panel = read_pickle(filename)
            self._df_dict = dict(panel)
            print 'Imported existing Panel with {} DataFrames from {}'.format(
                self.n_dataframes, filename
            )
        except:
            print 'Created empty Panel.'
            self._df_dict = {}
        
    def append(self, dataframe):
        self._df_dict[self.n_dataframes] = dataframe
    
    @property 
    def n_dataframes(self):
        return len(self._df_dict)
    
    @property
    def filename(self):
        return self._filename
    
    @property    
    def panel(self):
        return Panel(self._df_dict)
        
    def save(self):
        self.panel.to_pickle(self.filename)
        
class DataFrameLiveView(object):
    def __init__(self, dataframe):
        self.df = dataframe
        self.ham_model = m.RabiRamseyModel()
        self.ref_model = m.ReferencedPoissonModel(self.ham_model)
        
        self.draw_from_scratch()
        
    @staticmethod
    def update_fill_between(collection, x_vals, lower, upper):
        path_x = np.concatenate([x_vals[0,np.newaxis], x_vals, x_vals[::-1]])
        path_y = np.concatenate([lower[0,np.newaxis], upper, lower[::-1]])
        collection.set_paths([np.vstack([path_x,path_y]).T])
        
    @staticmethod
    def update_line(line, x_vals, y_vals):
        line.set_xdata(x_vals)
        line.set_ydata(y_vals)
        
    @staticmethod
    def update_errorbar(lines, x_vals, y_vals, y_errs):
        DataFrameLiveView.update_line(lines[0], x_vals, y_vals)
        DataFrameLiveView.update_line(lines[1], x_vals, y_vals - y_errs)
        DataFrameLiveView.update_line(lines[2], x_vals, y_vals + y_errs)
        
    def update(self):
	if self.df.shape[0] > 1:
            self.update_ref(self.axes[0])
            self.update_rabi(self.axes[1])
            self.update_ramsey(self.axes[2])
            self.update_simulations(self.axes[3], self.axes[4])
            self.fig.canvas.draw()
        
    def update_ref(self, axis):
        df = self.df
        
        # estimates
        bright_means, dark_means = (np.array(list(df['smc_mean']))[:,5:7]).T
        x_vals = np.arange(bright_means.size)
        DataFrameLiveView.update_line(axis.lines[0], x_vals, bright_means)
        DataFrameLiveView.update_line(axis.lines[1], x_vals, dark_means)
            
        # credible regions
        bright_upper, dark_upper = (np.array(list(df['smc_upper_quantile']))[:,5:7]).T
        bright_lower, dark_lower = (np.array(list(df['smc_lower_quantile']))[:,5:7]).T
        DataFrameLiveView.update_fill_between(axis.collections[0], x_vals, bright_lower, bright_upper)
        DataFrameLiveView.update_fill_between(axis.collections[1], x_vals, dark_lower, dark_upper)
        
        # data
        x_vals = x_vals[1:]
        for idx, label in enumerate(['bright', 'dark']):
            data = np.array(df[label][1:]).astype(float) / np.array(df['n_meas'][1:]).astype(float)
            DataFrameLiveView.update_line(axis.lines[idx+2], x_vals, data)

	axis.set_ylim([0.9 * np.amin(dark_lower), 1.1 * np.amax(bright_upper)])
            
    def update_rabi(self, axis):
        idx_param = self.ham_model.IDX_OMEGA
        means = np.array(list(self.df['smc_mean']))[:,idx_param]
        x_vals = np.arange(means.size)
        DataFrameLiveView.update_line(axis.lines[0], x_vals, means)
        
        upper = np.array(list(self.df['smc_upper_quantile']))[:,idx_param]
        lower = np.array(list(self.df['smc_lower_quantile']))[:,idx_param]
        DataFrameLiveView.update_fill_between(axis.collections[0], x_vals, lower, upper)
	axis.set_ylim([0.9 * np.amin(lower), 1.1 * np.amax(upper)])
        axis.set_xlim([0, max(200, x_vals[-1])])
        
    def update_ramsey(self, axis):
        idx_param = self.ham_model.IDX_ZEEMAN
        means = np.array(list(self.df['smc_mean']))[:,idx_param]
        x_vals = np.arange(means.size)
        DataFrameLiveView.update_line(axis.lines[0], x_vals, means)
        
        upper = np.array(list(self.df['smc_upper_quantile']))[:,idx_param]
        lower = np.array(list(self.df['smc_lower_quantile']))[:,idx_param]
        DataFrameLiveView.update_fill_between(axis.collections[0], x_vals, lower, upper)
	axis.set_ylim([0.9 * np.amin(lower), 1.1 * np.amax(upper)])
        
    def update_simulations(self, axis_rabi, axis_ramsey):
        eps_rabi, rabi_p, rabi_p_stds, eps_ramsey, ramsey_p, ramsey_p_stds = normalized_and_separated_signal(self.df)
        current_mean = self.df['smc_mean'][self.df.shape[0]-1][:5]
        
        ts = eps_rabi['t']
        max_t = 0 if ts.size == 0 else np.amax(ts)
        sim_ts = np.linspace(0, max(max_t,0.2), 100)
        sim_eps = rabi_sweep(1, n=100)
        sim_eps['t'] = sim_ts
        simulation =  self.ham_model.likelihood(
            0, current_mean[np.newaxis, :], sim_eps
        ).flatten()
        
        DataFrameLiveView.update_line(axis_rabi.lines[0], sim_ts, simulation)
        DataFrameLiveView.update_errorbar(axis_rabi.lines[1:], ts, rabi_p, rabi_p_stds)
        axis_rabi.set_xlim([0, np.amax(sim_ts)])
                                      
        ts = eps_ramsey['tau']
        max_t = 0 if ts.size == 0 else np.amax(ts)
        sim_ts = np.linspace(0, max(max_t,2), 100)
        tp_est = np.round(1 / current_mean[0] / 4 / 0.002) * 0.002
        sim_eps = ramsey_sweep(1, n=100, tp=tp_est)
        sim_eps['tau'] = sim_ts
        simulation = self.ham_model.likelihood(
            0, current_mean[np.newaxis, :], sim_eps
        ).flatten()
        
        DataFrameLiveView.update_line(axis_ramsey.lines[0], sim_ts, simulation)
        DataFrameLiveView.update_errorbar(axis_ramsey.lines[1:], ts, ramsey_p, ramsey_p_stds)
        axis_ramsey.set_xlim([0, np.amax(sim_ts)])
        
        
    def draw_from_scratch(self):
        df = self.df
        
        fig = plt.figure(figsize=(10,7))
        
        gs = gridspec.GridSpec(3,1,left=0,bottom=0,right=0.4,top=1)
        ax_ramsey = fig.add_subplot(gs[2,0])
        ax_ref = fig.add_subplot(gs[0,0], sharex=ax_ramsey)
        ax_rabi = fig.add_subplot(gs[1,0], sharex=ax_ramsey)
        plt.setp(ax_ref.get_xticklabels(), visible=False)
        plt.setp(ax_rabi.get_xticklabels(), visible=False)

        gs2 = gridspec.GridSpec(2,1,left=0.4,right=1,bottom=0,top=1)
        ax_ramsey_sim = fig.add_subplot(gs2[1,0])
        ax_rabi_sim = fig.add_subplot(gs2[0,0])

        #----------------------------------------------------------------
        # draw references
        #----------------------------------------------------------------
        plt.sca(ax_ref)
        plt.plot([],[])                           # bright mean
        plt.plot([],[])                           # dark mean
        plt.plot([],[],'.')                       # bright data
        plt.plot([],[],'.')                       # dark data
        plt.fill_between([],[],[],alpha=0.3)      # bright 90%
        plt.fill_between([],[],[],alpha=0.3)      # dark 90%
            
        plt.ylabel('References\nPhotons per Shot') 

        #----------------------------------------------------------------
        # draw rabi learning
        #----------------------------------------------------------------
        plt.sca(ax_rabi)
        plt.plot([],[])
        plt.fill_between([],[],[], alpha=0.3) 
        idx_param = m.RabiRamseyModel.IDX_OMEGA
        plt.ylabel('${}$ (MHz)'.format(self.ham_model.modelparam_names[idx_param]))

        #----------------------------------------------------------------
        # draw ramsey learning
        #----------------------------------------------------------------
        plt.sca(ax_ramsey)
        plt.plot([],[])
        plt.fill_between([],[],[], alpha=0.3) 
        idx_param = m.RabiRamseyModel.IDX_ZEEMAN
        plt.ylabel('${}$ (MHz)'.format(self.ham_model.modelparam_names[idx_param]))
        plt.xlabel('Number of Experiments')
        plt.xlim([0,100])

        #----------------------------------------------------------------
        # draw rabi simulation
        #----------------------------------------------------------------
        plt.sca(ax_rabi_sim)
        plt.plot([],[])
        plt.errorbar([0],[0],fmt='.',yerr=[0],capthick=1,capsize=3)

        plt.ylim([-0.05,1.05])
        plt.title('Rabi Experiment and Best Simulation')
        plt.xlabel('$t_p$ ($\mu$s)')
        plt.ylabel(r'Tr$(\rho|0\rangle\langle 0|)$')

        #----------------------------------------------------------------
        # draw ramsey simulation
        #----------------------------------------------------------------
        plt.sca(ax_ramsey_sim)
        plt.plot([],[])
        plt.errorbar([0],[0],fmt='.',yerr=[0],capthick=1,capsize=3)
            
        plt.ylim([-0.05,1.05])
        plt.title('Ramsey Experiment and Best Simulation')
        plt.xlabel(r'$\tau$ ($\mu$s)')
        plt.ylabel(r'Tr$(\rho|0\rangle\langle 0|)$')

        gs.tight_layout(fig, h_pad=0.1, rect=(0,0,0.5,1))
        gs2.tight_layout(fig, rect=(0.5,0,1,1))
        
        self.fig = fig
        self.axes = [ax_ref, ax_rabi, ax_ramsey, ax_rabi_sim, ax_ramsey_sim]
        
        self.update()

#-------------------------------------------------------------------------------
# SWEEP EXPPARAMS
#-------------------------------------------------------------------------------

def rabi_sweep(min_t=None, max_t=0.3, n=50, n_meas=None, wo=0, mode=None):
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

def ramsey_sweep(min_tau=None, max_tau=2, tp=0.01, phi=0, n=50, n_meas=None, wo=0, mode=None):
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
    if n_meas is not None:
        mode = m.ReferencedPoissonModel.SIGNAL if mode is None else mode
        vals = vals + [(mode * np.ones(n)).astype(np.int)]
        vals = vals + [(n_meas * np.ones(n)).astype(np.int)]
        dtype = dtype + [('mode', 'int'),('n_meas','int')]
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
    def __init__(self, tcp_client):
        super(TCPExperimentJob, self).__init__()
        self._tcp_client = tcp_client
        self._buffer = ''
    
    @property    
    def is_complete(self):
        try:
            self._buffer += self._tcp_client.recv(1024)
            return True
        except:
            return False
        
    def get_result(self):
        super(TCPExperimentJob, self).get_result()
        assert self.is_complete
        
        read_tries = 0
        while '\n' not in self._buffer and read_tries < 100:
            try:
                read_tries += 1
                self._buffer += self._tcp_client.recv(1024)
            except:
                raise RuntimeError('Failed to get experiment response.')
            
        message = self._buffer.split('\n')[0]
        
        bright, dark, signal, pbt, ts = message.split(',')

        bright, dark, signal = int(bright), int(dark), int(signal)
        preceded_by_tracking = bool(pbt)
        ts = dateutil.parser.parse(ts)
        ts = Timestamp(ts.replace(tzinfo=None))
                    
        return ExperimentResult(
            bright, dark, signal, 
            preceded_by_tracking=preceded_by_tracking, 
            timestamp=ts
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
        elif expparam['emode'] == m.RabiRamseyModel.RAMSEY:
            eps['delay_time'] = 1e-6 * expparam['tau']
            eps['pulse1_time'] = 1e-6 * expparam['t']
            eps['pulse2_time'] = 1e-6 * expparam['t']
            eps['pulse2_phase'] = expparam['phi']
        eps['center_freq'] = eps['center_freq'] - 1e6 * expparam['wo']
        eps['number_of_repetitions'] = expparam['n_meas']
        
        if precede_by_tracking:
            eps['precede_by_tracking'] = True
        
        # the JSON parser doesn't like getting np arrays
        for key in eps.keys():
            try:
                eps[key] = np.asscalar(eps[key])
            except AttributeError:
                pass
            
        topchef_job = self._service.new_job(eps)

        return TopChefExperimentJob(topchef_job)
        
        

class TCPRabiRamseyExperimentRunner(AbstractRabiRamseyExperimentRunner):
    def __init__(self, tcp_client):
        super(TCPRabiRamseyExperimentRunner, self).__init__()
        
        self._tcp_client = tcp_client

    def run_tracking(self):
        raise NotImplemented()
    
    @staticmethod    
    def make_job_string(n_meas=100000, meas_time=800e-9, center_freq=2.87e9, 
            intermediate_freq=0, adiabatic_power=0, delay_time=0,
            pulse1_time=0, pulse1_phase=0, pulse1_power=0, pulse1_offset_freq=0,
            pulse1_modulation_freq=0, pulse1_modulation_phase=0,
            pulse2_time=0, pulse2_phase=0, pulse2_power=0, pulse2_offset_freq=0,
            pulse2_modulation_freq=0, pulse2_modulation_phase=0,
            precede_by_tracking=False
            ):
        job_string = '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'
        job_string = job_string.format(
            int(precede_by_tracking), int(n_meas), 
            meas_time, center_freq, intermediate_freq,
            adiabatic_power,
            delay_time,
            pulse1_time, pulse1_phase, pulse1_power, pulse1_offset_freq,
            pulse1_modulation_freq, pulse1_modulation_phase,
            pulse2_time, pulse2_phase, pulse2_power, pulse2_offset_freq,
            pulse2_modulation_freq, pulse2_modulation_phase
        )
        return job_string
        
        
    def run_experiment(self, expparam, precede_by_tracking=False):
        super(TCPRabiRamseyExperimentRunner, self).run_experiment(expparam)
        
        center_freq = 2.87e9 - 1e6 * expparam['wo']
        
        if expparam['emode'] == m.RabiRamseyModel.RABI:
            job_string = self.make_job_string(
                pulse1_time = 1e-6 * expparam['t'], 
                center_freq=center_freq,
                n_meas=expparam['n_meas']
            )
        elif expparam['emode'] == m.RabiRamseyModel.RAMSEY:
            job_string = self.make_job_string(
                delay_time = 1e-6 * expparam['tau'],
                pulse1_time = 1e-6 * expparam['t'],
                pulse2_time = 1e-6 * expparam['t'],
                pulse2_phase = expparam['phi'],
                center_freq=center_freq,
                n_meas=expparam['n_meas']
            )
        else:
            raise RuntimeError('Unknown experiment.')
            
        try:
            # try to clear the read buffer just in case the old job
            # was not collected
            self._tcp_client.recv(8192)
        except:
            pass
        self._tcp_client.send(job_string)

        return TCPExperimentJob(self._tcp_client)


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
        self._risk_taker.particle_locations = self.updater.particle_locations[:,:5]
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
        self._risk_taker.particle_locations = self.updater.particle_locations[:,:5]
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
    def __init__(self, updater, max_t=0.3, max_tau=2, base=11/10, n=50, n_meas=100, name=None):
        self.updater = updater
        self._rabi_eps = rabi_sweep(max_t=1, n=n, n_meas=n_meas)
        self._ramsey_eps = ramsey_sweep(max_tau=1, n=n, n_meas=n_meas)
        
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
    def __init__(self, updater, max_t=0.3, max_tau=2, n=50, n_meas=100, name=None):
        self.updater = updater
        self._rabi_eps = rabi_sweep(max_t=max_t, n=n, n_meas=n_meas)
        self._ramsey_eps = ramsey_sweep(max_tau=max_tau, n=n, n_meas=n_meas)
        
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
        
class TrackingHeuristic(qi.Heuristic):
    """
    Wraps an existing heuristic, so that calling it returns a tuple 
    (eps, precede_by_tracking) where eps is the experiment that the underlying
    heuristic wants.
    
    Also introduces the idea of an initial reference experiment, which
    sets an empirical prior on the reference coordinates of the distribution.
    
    :param qinfer.Heuristic heuristic: The heuristic to wrap.
    :param float cutoff: How far below the initial bright reference we can 
        drop before demanding a tracking operation.
    :param bool track_on_initial_reference: Whether to precede the initial 
        reference taking by a tracking operation.
    :param float std_mult: How much bigger than 1 standard deviation we
        should set the reference prior to.
    """
    def __init__(self, heuristic, cutoff=0.85, track_on_initial_reference=True, std_mult=3):
        self.underlying_heuristic = heuristic
        self.cutoff = cutoff
        self.has_initial_reference = False
        self.track_on_initial_reference = track_on_initial_reference
        self.std_mult = std_mult
        
        self._initial_bright_mean = None
        self._initial_dark_mean = None
        self._initial_bright_std = None
        self._initial_dark_std = None
        
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
            self.updater.particle_locations[:,5:7] = samples
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
                sleep(0.4)
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
        bright_est = self.updater.est_mean()[5]
        return bright_est < self._initial_bright_mean - 2 * self.std_mult * self._initial_bright_std
        
        
    def __call__(self, tp):
        if not self.has_initial_reference:
            raise RuntimeError('take_initial_reference must be called before an experiment can be suggested.')
        eps = self.underlying_heuristic(tp)
        precede_by_tracking = self._decide_on_tracking()
        return eps, precede_by_tracking
