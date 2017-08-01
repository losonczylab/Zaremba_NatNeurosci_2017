import numpy as np
import pandas as pd
from scipy.integrate import trapz
from itertools import count, izip
import cPickle as pkl

# import imaging_analysis as ia
from ..classes.interval import Interval, ImagingInterval


def calc_activity(
        experiment, method, interval=None, dF='from_file', channel='Ch2',
        label=None, roi_filter=None, demixed=False, running_kwargs=None,
        trans_threshold=95):

    """Calculate various population statistics on each ROI

    Takes an BehavioralAnalysis.Experiment object and calculates various
    statistics on the imaging data for each ROI, returning a population vector
    of the desired activity measure. Each cycle is analyzed individually and an
    interval can be passed in to select which frames to include.

    Parameters
    ----------
    experiment : behavior_analysis.Experiment
        Experiment object to analyze
    method : string
        Calculation to perform on each ROI
    interval : boolean array or start/stop frames, optional
        Boolean array of imaging frames to include in analysis, defaults to all
        frames
        Can have a unique interval for each ROI or cycle, automatically
        expanded if a single interval is passed in
    df : string, optional
        dF/F algorithm to run on imaging data, passed to
        behavior_analysis.Experiment.imagingData as 'dFOverF' argument
    average_trials : bool, optional
        If True, average metric across trials

    Returns
    -------
    a : ndarray
        Returns a ndarray of shape (nROIS, nCycles)
    """
    # im_shape = experiment.imaging_shape(
    #    channel=channel, label=label, roi_filter=roi_filter)
    # if im_shape[0] == 0:
    #     return np.empty((0, im_shape[2]))
    data = None

    if interval is None:
        # If no interval passed in, look at the entire imaging sequence
        data = experiment.imagingData(
            dFOverF=dF, roi_filter=roi_filter, channel=channel, label=label,
            demixed=demixed)
        interval = np.ones(data.shape, 'bool')
    elif interval == 'running':
        if running_kwargs:
            interval = np.array(experiment.runningIntervals(
                returnBoolList=True, **running_kwargs))
        else:
            interval = np.array(experiment.runningIntervals(
                returnBoolList=True))
    elif interval == 'non-running':
        if running_kwargs:
            interval = ~np.array(experiment.runningIntervals(
                returnBoolList=True, **running_kwargs))
        else:
            interval = ~np.array(experiment.runningIntervals(
                returnBoolList=True))
    elif isinstance(interval, Interval):
        num_rois, num_frames, num_cycles = experiment.imaging_shape(
            channel=channel, label=label, roi_filter=roi_filter)
        if not isinstance(interval, ImagingInterval):
            sampling_interval = experiment.frame_period()
            interval = ImagingInterval(
                interval, sampling_interval=sampling_interval,
                num_rois=num_rois)
            if interval.shape[1] > num_frames:
                interval = interval[:, :num_frames, :]
        interval = interval.mask
        assert interval.shape == (num_rois, num_frames, num_cycles)
    elif interval.dtype is not np.dtype('bool'):
        # If interval is not boolean, assume start/stop times and convert
        # Must pass in a tuple/list/array of exactly 2 elements
        data = experiment.imagingData(
            dFOverF=dF, roi_filter=roi_filter, channel=channel, label=label,
            demixed=demixed)
        inter = np.zeros((data.shape[1], 1), 'bool')
        inter[interval[0]:interval[1] + 1] = True
        interval = np.tile(inter, (data.shape[0], 1, data.shape[2]))

    # If input interval is smaller than shape of data, expand it
    if interval.ndim == 1:
        data = experiment.imagingData(
            dFOverF=dF, roi_filter=roi_filter, channel=channel, label=label,
            demixed=demixed)
        interval = np.reshape(interval, (-1, 1))
        interval = np.tile(interval, (data.shape[0], 1, data.shape[2]))
    elif interval.ndim == 2 and \
            (interval.shape[0] == 1 or interval.shape[1] == 1):
        data = experiment.imagingData(
            dFOverF=dF, roi_filter=roi_filter, channel=channel, label=label,
            demixed=demixed)
        interval = np.reshape(interval, (-1, 1))
        interval = np.tile(interval, (data.shape[0], 1, data.shape[2]))
    elif interval.ndim == 2:
        data = experiment.imagingData(
            dFOverF=dF, roi_filter=roi_filter, channel=channel, label=label,
            demixed=demixed)
        interval = interval[:, :, np.newaxis]
        interval = np.tile(interval, (1, 1, data.shape[2]))

    #
    # Begin calculations
    #

    if method == 'mean':
        # Mean value of signal during interval
        if data is None:
            data = experiment.imagingData(
                dFOverF=dF, roi_filter=roi_filter, channel=channel,
                label=label, demixed=demixed)

        metric = np.zeros((data.shape[0], data.shape[2]))
        for roi_idx, roi_data, roi_int in izip(count(), data, interval):
            for cycle_idx, cycle_data, cycle_int in izip(
                    count(), roi_data.T, roi_int.T):
                metric[roi_idx, cycle_idx] = np.nanmean(
                    cycle_data[cycle_int])

    elif method == 'auc':
        # Area under curve of signal during interval
        if data is None:
            data = experiment.imagingData(
                dFOverF=dF, roi_filter=roi_filter, channel=channel,
                label=label, demixed=demixed)
        period = experiment.frame_period()

        metric = np.zeros((data.shape[0], data.shape[2]))
        for roi_idx, roi_data, roi_int in izip(count(), data, interval):
            for cycle_idx, cycle_data, cycle_int in izip(
                    count(), roi_data.T, roi_int.T):
                metric[roi_idx, cycle_idx] = nantrapz_1d(
                    cycle_data[cycle_int], dx=period)

    elif method == 'amplitude':
        # Average amplitude of transients that peak within interval
        if data is None:
            data = experiment.imagingData(
                dFOverF=dF, roi_filter=roi_filter, channel=channel,
                label=label, demixed=demixed)
        trans = experiment.transientsData(
            threshold=trans_threshold, roi_filter=roi_filter, channel=channel,
            label=label, demixed=demixed)
        inc_trans = includedTransients(trans, interval)

        metric = np.zeros((data.shape[0], data.shape[2]))
        for roi_idx, roi_data, roi_trans, inc in izip(
                count(), data, trans, inc_trans):
            for cycle_idx, cycle_data, cycle_trans, cycle_inc in izip(
                    count(), roi_data.T, roi_trans, inc):
                if len(cycle_inc) > 0:
                    metric[roi_idx, cycle_idx] = np.mean(
                        cycle_data[cycle_trans['max_indices'][cycle_inc]])
                else:
                    metric[roi_idx, cycle_idx] = np.nan

    elif method == 'duration':
        # Average duration of transients that peak within interval (in seconds)
        trans = experiment.transientsData(
            threshold=trans_threshold, roi_filter=roi_filter, channel=channel,
            label=label, demixed=demixed)
        inc_trans = includedTransients(trans, interval)

        metric = np.zeros((len(trans), len(trans[0])))
        for roi_idx, roi_trans, inc in izip(count(), trans, inc_trans):
            for cycle_idx, cycle_trans, cycle_inc in izip(
                    count(), roi_trans, inc):
                if np.sum(cycle_inc) > 0:
                    metric[roi_idx, cycle_idx] = np.mean(
                        cycle_trans['durations_sec'][cycle_inc])
                else:
                    metric[roi_idx, cycle_idx] = np.nan

    elif method == 'responseMagnitude':
        # Average area under curve of transients  that peak within interval
        # (in s*dF)
        trans_auc = calc_activity(
            experiment, 'transient auc2', interval=interval, dF=dF,
            channel=channel, label=label, demixed=demixed,
            roi_filter=roi_filter)
        n_trans = calc_activity(
            experiment, 'n transients', interval=interval, dF=dF,
            channel=channel, label=label, demixed=demixed,
            roi_filter=roi_filter)

        metric = trans_auc / n_trans

    elif method == 'transient auc':
        # Total area under curve of transients during interval
        if data is None:
            data = experiment.imagingData(
                dFOverF=dF, roi_filter=roi_filter, channel=channel,
                label=label, demixed=demixed)
        data *= interval
        period = experiment.frame_period()

        trans = experiment.transientsData(
            threshold=trans_threshold, roi_filter=roi_filter, channel=channel, label=label,
            demixed=demixed)

        metric = np.zeros((data.shape[0], data.shape[2]))
        for roi_idx, roi_trans in enumerate(trans):
            for cycle_idx, cycle_trans in enumerate(roi_trans):
                if len(cycle_trans['start_indices']):
                    for start_idx, stop_idx in zip(
                            cycle_trans['start_indices'],
                            cycle_trans['end_indices']):
                        if np.isnan(start_idx):
                            start_idx = 0
                        if np.isnan(stop_idx):
                            stop_idx = data.shape[1]
                        metric[roi_idx, cycle_idx] += nantrapz_1d(
                            data[roi_idx, start_idx:stop_idx + 1, cycle_idx],
                            dx=period)
                else:
                    # If there were no transients in the given interval...
                    # return NaN, not 0
                    metric[roi_idx, cycle_idx] = np.nan

    elif method == 'norm transient auc':
        # Total area under curve of transients during interval normalized to
        # length of interval
        auc = calc_activity(
            experiment, 'transient auc', interval=interval, dF=dF,
            channel=channel, label=label, demixed=demixed,
            roi_filter=roi_filter)
        period = experiment.frame_period()

        metric = auc / (np.sum(interval, axis=1) * period)

    elif method == 'transient auc2':
        # Total area under curve of transients that peak during interval
        if data is None:
            data = experiment.imagingData(
                dFOverF=dF, roi_filter=roi_filter, channel=channel,
                label=label, demixed=demixed)
        period = experiment.frame_period()

        trans = experiment.transientsData(
            threshold=trans_threshold, roi_filter=roi_filter, channel=channel, label=label,
            demixed=demixed)
        inc_trans = includedTransients(trans, interval)

        metric = np.zeros((data.shape[0], data.shape[2]))
        for roi_idx, roi_trans, inc in izip(count(), trans, inc_trans):
            for cycle_idx, cycle_trans, cycle_inc in izip(
                    count(), roi_trans, inc):
                if np.sum(cycle_inc) > 0:
                    for start_idx, stop_idx in zip(
                            cycle_trans['start_indices'][cycle_inc],
                            cycle_trans['end_indices'][cycle_inc]):
                        if np.isnan(start_idx):
                            start_idx = 0
                        if np.isnan(stop_idx):
                            stop_idx = data.shape[1]
                        metric[roi_idx, cycle_idx] += nantrapz_1d(
                            data[roi_idx, start_idx:stop_idx + 1, cycle_idx],
                            dx=period)
                else:
                    metric[roi_idx, cycle_idx] = 0

    elif method == 'norm transient auc2':
        # Total area under curve of transients that peak during interval
        # normalized to length of interval
        auc = calc_activity(
            experiment, 'transient auc2', interval=interval, dF=dF,
            channel=channel, label=label, demixed=demixed,
            roi_filter=roi_filter)
        period = experiment.frame_period()

        metric = auc / (np.sum(interval, axis=1) * period)

    elif method == 'time active':
        # Percentage of the interval the cell is active
        active = ia.isActive(
            experiment, conf_level=trans_threshold, roi_filter=roi_filter, channel=channel,
            label=label, demixed=demixed)

        metric = np.sum(active & interval, axis=1) / \
            np.sum(interval, axis=1).astype('float')

    elif method == 'frequency':
        # Frequency of transients that peak during interval (in Hz)
        period = experiment.frame_period()

        n_trans = calc_activity(
            experiment, 'n transients', interval=interval, dF=dF,
            channel=channel, label=label, demixed=demixed,
            roi_filter=roi_filter)

        metric = n_trans / (np.sum(interval, axis=1) * period)

    elif method == 'n transients':
        # Number of transients that peak during interval
        trans = experiment.transientsData(
            threshold=trans_threshold, roi_filter=roi_filter, channel=channel, label=label,
            demixed=demixed)
        inc_trans = includedTransients(trans, interval)

        n_trans = np.zeros((len(trans), interval.shape[2]))
        for roi_idx, inc in enumerate(inc_trans):
            for cycle_idx, cycle_inc in enumerate(inc):
                if np.sum(cycle_inc) > 0:
                    n_trans[roi_idx, cycle_idx] = np.sum(cycle_inc)

        metric = n_trans.astype('int')

    elif method == 'is place cell':
        with open(experiment.placeFieldsFilePath(channel=channel), 'rb') as f:
            pfs = pkl.load(
                f)[label]['demixed' if demixed else 'undemixed']['pfs']
        inds = experiment._filter_indices(
            roi_filter, channel=channel, label=label)

        pfs = np.array(pfs)[np.array(inds)]

        pc = []
        for roi in pfs:
            if len(roi):
                pc.append(1)
            else:
                pc.append(0)
        metric = np.array(pc).astype('int')[:, np.newaxis]

    elif method == 'time to max peak':
        if data is None:
            data = experiment.imagingData(dFOverF=dF, roi_filter=roi_filter,
                                          channel=channel, label=label,
                                          demixed=demixed)
        period = experiment.frame_period()

        trans = experiment.transientsData(threshold=trans_threshold, roi_filter=roi_filter,
                                          channel=channel, label=label,
                                          demixed=demixed)
        inc_trans = includedTransients(trans, interval)

        auc = np.zeros((data.shape[0], data.shape[2]))
        for roi_idx, roi_trans, inc in izip(count(), trans, inc_trans):
            for cycle_idx, cycle_trans, cycle_inc in izip(count(), roi_trans,
                                                          inc):
                if np.sum(cycle_inc) > 0:
                    idx = cycle_trans['max_amplitudes'][cycle_inc].argmax()
                    auc[roi_idx, cycle_idx] = \
                        cycle_trans['start_indices'][cycle_inc][idx]
        return auc

    else:
        raise ValueError('Unrecognized method: ' + str(method))

    return metric


def includedTransients(transients, interval):
    # Returns an array of logical arrays indicating which transient starts
    # within the given interval
    # Returned array will be nROIs x nCycles, with each element an nTransients
    # length boolean array for logical indexing
    #
    # Interval must be correct format, not checking at the moment

    try:
        transients[0]
    except IndexError:
        return np.empty((0, interval.shape[2]), 'object')
    else:
        inc = np.empty((len(transients), len(transients[0])), 'object')
        for roi_idx, roi_trans, roi_int in izip(count(), transients, interval):
            for cycle_idx, cycle_trans, cycle_int in izip(count(), roi_trans, roi_int.T):
                inc[roi_idx, cycle_idx] = np.zeros(len(cycle_trans['start_indices']), 'bool')
                for trans_idx, idx_max in enumerate(cycle_trans['start_indices']):
                    inc[roi_idx, cycle_idx][trans_idx] = (idx_max in np.nonzero(cycle_int)[0] )

        return inc


def nantrapz_1d(y, x=None, dx=1.0):
    if x is None:
        x_vals = np.arange(len(y)) * dx
    else:
        x_vals = x

    nans = np.isnan(y)

    assert len(nans) == len(x_vals)
    assert len(nans) == len(y)

    return trapz(y[~nans], x=x_vals[~nans])


def included_transient(x):
    """Return whether the start time is range of tuple pair of interval."""
    return (x.int_start <= x.start_frame <= x.int_end)


def included_transients(data, interval):
    """Return filtered dataframe of transients that occur during interval."""
    data = pd.merge(data, interval, how='outer')
    inc_tran = data.groupby(["roi",
                             "trial",
                             "trans_idx"]).apply(included_transient)
    return data.loc[inc_tran]


def roi_trials(expt_grp, channel='Ch2', label=None, roi_filter=None):
    rois = expt_grp.rois(channel=channel, label=label, roi_filter=roi_filter)
    df_list = []
    for expt in expt_grp:
        for trial in expt.findall('trial'):
            for roi in rois[expt]:
                df_list.append({'trial': trial, 'roi': roi})
    return pd.DataFrame(df_list)


def calc_n_transients(expt_grp, interval=None, channel='Ch2',
                      label=None, roi_filter=None, demixed=False,
                      behaviorSync=False):
    """Return the number of transients per ROI."""
    data_list = [expt.transientsData(threshold=95,
                                     channel=channel,
                                     label=label,
                                     roi_filter=roi_filter,
                                     behaviorSync=behaviorSync,
                                     dataframe=True) for expt in expt_grp]
    trans_data = pd.concat(data_list)
    if interval is not None:
        trans_data = included_transients(trans_data, interval)
    n_trans = trans_data.groupby(["roi", "trial"]).count()
    n_trans = n_trans.reset_index()[['roi', 'trial', 'trans_idx']]
    n_trans.rename(columns={"trans_idx": "n_trans"}, inplace=True)
    return n_trans


def transient_auc(x):
    """Return the AUC for the transient."""
    period = x.trial.parent.frame_period()
    y = np.zeros(x.im_data.shape)
    y[x.start_frame:x.stop_frame] = 1
    y *= x.im_data
    return nantrapz_1d(y, dx=period)


def calc_transient_auc(expt_grp, interval=None, channel='Ch2',
                       label=None, roi_filter=None, demixed=False,
                       behaviorSync=False):
    """Return the AUC for each transient."""
    trim_to_behavior = behaviorSync
    data_list = [expt.transientsData(threshold=95,
                                     channel=channel,
                                     label=label,
                                     roi_filter=roi_filter,
                                     behaviorSync=behaviorSync,
                                     dataframe=True) for expt in expt_grp]
    im_data = [expt.imagingData(dFOverF='from_file',
                                demixed=demixed,
                                roi_filter=roi_filter,
                                removeNanBoutons=False,
                                trim_to_behavior=trim_to_behavior,
                                channel=channel,
                                label=label,
                                dataframe=True) for expt in expt_grp]
    data_list, im_data = pd.concat(data_list), pd.concat(im_data)
    if interval is not None:
        data_list = included_transients(data_list, interval)
    data = pd.merge(data_list, im_data)
    data["AUC"] = data.apply(transient_auc, axis=1)
    return data[["roi", "trial", "trans_idx", "AUC"]]


def calc_sum_transient_auc(expt_grp, interval=None, channel='Ch2',
                           label=None, roi_filter=None, demixed=False,
                           behaviorSync=False):
    """Return the total AUC for each unique ROI trace."""
    data = calc_transient_auc(expt_grp=expt_grp, interval=interval,
                              channel=channel, label=label,
                              roi_filter=roi_filter, behaviorSync=behaviorSync,
                              demixed=demixed)
    data = data.groupby(["roi", "trial"]).agg({"AUC": np.nansum})
    data.reset_index(level=["roi", "trial"], inplace=True)
    return data


def calc_average_duration(expt_grp, interval=None, channel='Ch2',
                          label=None, roi_filter=None, demixed=False,
                          behaviorSync=False):
    """
    Return the average transient duration for each unique ROI trace.

    The duration returned is in seconds."""
    data_list = [expt.transientsData(threshold=95,
                                     channel=channel,
                                     label=label,
                                     roi_filter=roi_filter,
                                     behaviorSync=behaviorSync,
                                     dataframe=True) for expt in expt_grp]
    data_list = pd.concat(data_list)
    if interval is not None:
        data_list = included_transients(data_list, interval)
    data = data_list.groupby(["roi", "trial"]).agg({"duration": np.nanmean})
    data.reset_index(level=["roi", "trial"], inplace=True)
    return data


def calc_average_amplitude(expt_grp, interval=None, channel='Ch2',
                           label=None, roi_filter=None, demixed=False,
                           behaviorSync=False):
    """Return the average transient amplitude for each unique ROI trace."""
    data_list = [expt.transientsData(threshold=95,
                                     channel=channel,
                                     label=label,
                                     roi_filter=roi_filter,
                                     behaviorSync=behaviorSync,
                                     dataframe=True) for expt in expt_grp]
    data_list = pd.concat(data_list)
    if interval is not None:
        data_list = included_transients(data_list, interval)
    data = data_list.groupby(["roi",
                              "trial"]).agg({"max_amplitude": np.nanmean})
    data.reset_index(level=["roi", "trial"], inplace=True)
    return data


def calc_time_active(expt_grp, interval=None, channel='Ch2',
                     label=None, roi_filter=None, demixed=False,
                     behaviorSync=False):
    """Return the percentage of time that unique transient is active."""
    data_list = [expt.transientsData(threshold=95,
                                     channel=channel,
                                     label=label,
                                     roi_filter=roi_filter,
                                     behaviorSync=behaviorSync,
                                     dataframe=True) for expt in expt_grp]
    trans_data = pd.concat(data_list)
    if interval is not None:
        data_list = included_transients(data_list, interval)
        interval["int_len"] = interval["int_start"] - interval["int_end"]
    # If no interval is passed, the entire imaging lenght should be
    # assumed.
    # else:
    #     interval = ImagingInterval()
    summed_len = interval.groupby(["roi", "trial"]).agg({"int_len": np.sum})
    summed_len.reset_index(level=["roi", "trial"], inplace=True)
    trans_data = pd.merge(trans_data, summed_len)
    trans_data["time_active"] = trans_data["duration"] / trans_data["int_len"]
    return trans_data[["roi", "trial", "trans_idx", "time_active"]]


def calc_summed_time_active(expt_grp, interval=None, channel='Ch2',
                            label=None, roi_filter=None, demixed=False,
                            behaviorSync=False):
    """Return the percentage of time that unique ROI trace is active."""
    roi_trial_object = roi_trials(expt_grp, channel=channel,
                                  label=label, roi_filter=roi_filter)
    trans_data = calc_time_active(expt_grp=expt_grp, interval=interval,
                                  channel=channel, label=label,
                                  roi_filter=roi_filter, demixed=demixed,
                                  behaviorSync=behaviorSync)
    summed_trans_data = trans_data.groupby(["roi",
                                            "trial"
                                            ]).agg({"time_active": np.sum})
    summed_trans_data.reset_index(level=["roi", "trial"], inplace=True)
    summed_trans_data = pd.merge(roi_trial_object,
                                 summed_trans_data, how='outer')
    return summed_trans_data


def calc_response_magnitude(expt_grp, interval=None, channel='Ch2',
                            label=None, roi_filter=None, demixed=False,
                            behaviorSync=False):
    """Return the summed average AUC divided by the number of transients."""
    auc = calc_sum_transient_auc(expt_grp=expt_grp, interval=interval,
                                 channel=channel, label=label,
                                 roi_filter=roi_filter,
                                 behaviorSync=behaviorSync,
                                 demixed=demixed)
    n_trans = calc_n_transients(expt_grp=expt_grp, interval=interval,
                                channel=channel, label=label,
                                roi_filter=roi_filter,
                                behaviorSync=behaviorSync,
                                demixed=demixed)
    data = pd.merge(auc, n_trans)
    data["response_magnitude"] = data["AUC"] / data["n_trans"]
    return data[["roi", "trial", "trans_idx", "AUC"]]
