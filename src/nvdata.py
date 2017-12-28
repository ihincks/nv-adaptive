# -*- coding: utf-8 -*-
from __future__ import division
from future.utils import with_metaclass
import warnings
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import numpy as np
import models as m
import datetime
from pandas import DataFrame, Panel, Timestamp, Timedelta, read_pickle
import wquantiles
from scipy.interpolate import interp1d

#-------------------------------------------------------------------------------
# CONSTANTS
#-------------------------------------------------------------------------------

        
#-------------------------------------------------------------------------------
# FUNCTIONS
#-------------------------------------------------------------------------------

def get_now():
    return Timestamp(datetime.datetime.now())

def est_std(p, alpha, beta):
    return np.sqrt(p*(p+1)*alpha + (p-1)*(p-2)*beta)/(alpha-beta)
    
def asscalar(a):
    try:
        return np.asscalar(a)
    except AttributeError:
        return a


def add_counts_by_unique_expparams(df):
    """
    Groups the given DataFrame by unique expparams,
    and performs a sum of bright, dark, and signal
    over those groups.
    
    :param DataFrame df: As constructed by `new_experiment_dataframe`
    
    :returns: Tuple `(eps, bright, dark, signal)`, each an
    `np.ndarray` of the same length.
    """
    # need to hash the expparams for pandas to group them
    groups = df[1:].groupby(df[1:].expparam.apply(lambda x: '{}'.format(x)))
    
    bright = np.array(list(groups['bright'].sum()))
    dark = np.array(list(groups['dark'].sum()))
    signal = np.array(list(groups['signal'].sum()))
    
    eps = np.array(list(groups['expparam'].first())).flatten()
    
    return eps, bright, dark, signal

def normalized_signal_by_unique_expparams(df):
    """
    Uses the output of `add_counts_by_unique_expparams` to
    normalize the signal by the references, returning also
    an estimate of the std of the normalized value based
    on the Fisher information of the referenced poisson model.
    
    :param DataFrame df: As constructed by `new_experiment_dataframe`
    
    :returns: Tuple `(eps, p, p_stds)`, each an
    `np.ndarray` of the same length.
    """
    eps, bright, dark, signal = add_counts_by_unique_expparams(df)
    p = (signal - dark).astype(float) / (bright - dark)
    p_stds = est_std(p, bright, dark)
    return eps, p, p_stds

def normalized_and_separated_signal(df):
    """
    Separates the output of `normalized_signal_by_unique_expparams`
    into ramsey and rabi experiments.
    
    :param DataFrame df: As constructed by `new_experiment_dataframe`
    
    :returns: Tuple `(rabi_eps, rabi_p, rabi_p_stds, ramsey_eps, ramsey_p, ramsey_p_stds)`
    """
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

def extract_panel_data(panel, y_column, idxs=None, x_column=None, skip_first=False):
    """
    Extracts data from a HeuristicData object across data frames. It is assumed
    that all dataframes have the same number of rows.
    
    :param str y_column: Which column of the dataframes to extract.
    :param idxs: A numpy slice object (ex. np.s_[3,3]) to apply to the data
    entries, or None, to take everything.
    :param str x_column: Which column to use as the x-axis.
    
    :return: A tuple `(x_vals, y_vals)` where `y_vals is an
    `np.ndarray` of shape `(n_dataframes, n_rows, ...)` where the
    ellipsis is determined by the `idxs` pararameter.
    """
    n_df = panel.n_dataframes
    
    idxs = Ellipsis if idxs is None else idxs
    sample_y = panel.panel[0][y_column]
    if skip_first:
        row_idxs = np.s_[1:]
        n_rows = len(sample_y) - 1
        sample_y = sample_y[1][idxs]
    else:
        row_idxs = np.s_[:]
        n_rows = len(sample_y)
        sample_y = sample_y[0][idxs]
    
    y_data = np.empty((n_df, n_rows,) + sample_y.shape, dtype=sample_y.dtype)
    for idx_df in range(n_df):
        all_rows = np.array(list(panel.panel[idx_df][y_column][row_idxs]))
        try:
            y_data[idx_df, ...] = all_rows[(np.s_[:],) + idxs]
        except TypeError:
            y_data[idx_df, ...] = all_rows[(np.s_[:],) + (idxs,)]
    
    if x_column is None:
        x_data = np.arange(y_data.shape[1])
    else:
        sample_x = panel.panel[0][x_column]
        try:
            sample_x = np.array([sample_x[0]])
        except:
            sample_x = np.array([sample_x])
        assert sample_x.size == 1
            
        x_data_all = np.empty((n_df, n_rows), dtype=float)
        for idx_df in range(n_df):
            x_data = panel.panel[idx_df][x_column][row_idxs]
            try:
                x_data_all[idx_df, :] = x_data / Timedelta(seconds=1)
            except TypeError:
                x_data_all[idx_df, :] = list(x_data)
        
        x_data = np.linspace(
                np.amax(x_data_all[:,0]),
                np.amin(x_data_all[:,-1]),
                x_data_all.shape[1]
            )
        for idx_df in range(n_df):
            fun = interp1d(x_data_all[idx_df, :], y_data[idx_df, ...], axis=0)
            y_data[idx_df, ...] = fun(x_data)
        
        
    return x_data, y_data
    
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
