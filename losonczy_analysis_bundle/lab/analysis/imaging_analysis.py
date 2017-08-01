"""Analysis of calcium imaging time series data"""

import os
import warnings
import numpy as np
import scipy
from scipy.fftpack import fft, fftfreq
from scipy import corrcoef
from scipy.cluster import hierarchy
from scipy.stats import mode, chisquare, zscore
from scipy.spatial.distance import squareform
try:
    from bottleneck import nanmean, nanstd
except ImportError:
    warnings.warn("Install bottleneck to speed up some numpy functions")
    from numpy import nanmean, nanstd
from matplotlib.mlab import demean
import pandas as pd
import seaborn.apionly as sns
import itertools as it
import cPickle as pickle
import shapely

try:
    from bottleneck import nanmean
except ImportError:
    warnings.warn("Install bottleneck to speed up some numpy functions")
    from numpy import nanmean

import sima

import matplotlib.pyplot as plt

import behavior_analysis as ba
import filters
# from ..analysis import boutonAnalysis as bouton
import calc_activity as ca
from ..classes import exceptions as exc
# from ..classes.experiment import SalienceExperiment, FearConditioningExperiment
from ..misc.misc import timestamp
from ..misc import stats, memoize
from ..plotting import plotting as plotting
import filters
"""
Free functions
"""


def _responseIntegrals(timeSeries, stimIdx, maxIdx, dt):
    baseline = timeSeries[:, :stimIdx, :].mean(axis=2).mean(axis=1)
    for i in range(timeSeries.shape[0]):
        timeSeries[i, :, :] -= baseline[i]
        timeSeries[i, :, :] /= baseline[i]
    return timeSeries[:, stimIdx:maxIdx, :].sum(axis=1) * dt


def _responseZScores(timeSeries, stimIdx, maxIdx, dt):
    respIntegrals = _responseIntegrals(timeSeries, stimIdx, maxIdx, dt)
    return respIntegrals.mean(axis=1) / np.sqrt(
        np.var(respIntegrals, axis=1)) * np.sqrt(respIntegrals.shape[1])

"""
Helper functions
"""


def offsetPCA(data):
    """Perform PCA, but rather than diagonalizing the cross-correlation matrix at zero
    delay, diagonalize the symmetrized cross-correlation matrix at one timestep delays"""
    meanSubtractedData = data - np.dot(
        np.ones([data.shape[0], 1]), np.mean(data, axis=0).reshape([1, -1]))
    corrOffset = np.dot(meanSubtractedData[0:-1, :].T, meanSubtractedData[1:, :])
    corrOffset = 0.5 * (corrOffset + corrOffset.T)
    eiVals, eiVects = np.linalg.eigh(corrOffset)
    # sort the eigenvectors and eigenvalues
    idx = np.argsort(-eiVals)
    eiVals = eiVals[idx]
    eiVects = eiVects[:, idx]
    return eiVals, eiVects


def offsetCorrelation(data, pcaClean=True):
    if pcaClean:
        data = pcaCleanup(data, 1.)
    data = demean(data, axis=1)
    for i in range(data.shape[0]):
        data[i, :] /= np.sqrt((data[i, 1:] * data[i, :-1]).mean())
    data[np.logical_not(np.isfinite(data))] = np.nan
    corrMatrix = np.dot(data[:, 1:], data[:, :-1].T) / (data.shape[1] - 1)
    return 0.5 * (corrMatrix + corrMatrix.T)


def pcaCleanupMatrix(data, retainedVariance=1.):
    variances, pcs = offsetPCA(data.T)
    variances = variances / sum(variances)
    coefficientTimeSeries = np.dot(pcs.T, data)
    numPCs = 0
    capturedVariance = 0.
    while numPCs < len(variances) and capturedVariance < retainedVariance:
        capturedVariance += variances[numPCs]
        numPCs += 1
    for i in range(numPCs, len(pcs)):
        pcs[:, i] = 0.
    cleanup = np.dot(pcs, pcs.T)
    return cleanup


def pcaCleanup(data, retainedVariance=.99):
    variances, pcs = offsetPCA(data.T)
    variances = variances / sum(variances)
    coefficientTimeSeries = np.dot(pcs.T, data)
    numPCs = 0
    capturedVariance = 0.
    while numPCs < len(variances) and capturedVariance < retainedVariance:
        capturedVariance += variances[numPCs]
        numPCs += 1
    cleanData = np.dot(pcs[:, :numPCs], coefficientTimeSeries[:numPCs, :])
    return cleanData


def principalAngles(A, B):
    _, s, _ = scipy.linalg.svd(np.dot(A.T, B))
    return np.arccos(s)

"""
Experiment analysis
"""


def isActive(expt, conf_level=95, channel='Ch2', label=None,
             demixed=False, roi_filter=None):
    # return a boolean np array #rois x #frames corresponding to whether
    # the cell is in a significant transient

    activity = np.zeros(expt.imaging_shape(channel=channel, label=label,
                                           roi_filter=roi_filter),
                        dtype='bool')
    transients = expt.transientsData(
        threshold=conf_level, channel=channel, demixed=demixed,
        label=label, roi_filter=roi_filter)

    if activity.ndim == 2:
        activity = activity[::, np.newaxis]

    for cell_index, cell in enumerate(transients):
        for cycle_index, cycle in enumerate(cell):
            starts = cycle['start_indices']
            ends = cycle['end_indices']
            for start, end in zip(starts, ends):
                if np.isnan(start):
                    start = 0
                if np.isnan(end):
                    end = activity.shape[1] - 1
                activity[cell_index, start:end + 1, cycle_index] = True
    return activity


def oPCA(expt, channel='Ch2', num_pcs=75):
    """Perform offset principal component analysis on the dataset

    If the number of requests PCs have already been calculated and saved,
    just returns those. Otherwise re-run oPCA and save desired number of
    PCs

    Returns
    -------
    oPC_vars : array
        The offset variance accounted for by each oPC. Shape: num_pcs.
    oPCs : array
        The spatial representations of the oPCs.
        Shape: (num_rows, num_columns, num_pcs).
    oPC_signals : array
        The temporal representations of the oPCs.
        Shape: (num_times, num_pcs).

    """
    dataset = expt.imaging_dataset()
    channel_idx = dataset._resolve_channel(channel)
    path = os.path.join(
        dataset.savedir, 'opca_' + str(channel_idx) + '.npz')
    return sima.segment.oPCA.dataset_opca(
        dataset, ch=channel_idx, num_pcs=num_pcs, path=path)


def powerSpectra(
        expt, dFOverF='None', demixed=False, linearTransform=None,
        window_width=100, dFOverF_percentile=0.25, removeNanBoutons=False,
        channel='Ch2', label=None, roi_filter=None):
    """Give the power spectrum of each signal and the corresponding frequencies.
    See imagingData for parameters.

    """

    imData = expt.imagingData(
        dFOverF=dFOverF, demixed=demixed,
        linearTransform=linearTransform, window_width=window_width,
        dFOverF_percentile=dFOverF_percentile,
        removeNanBoutons=removeNanBoutons, channel=channel, label=label,
        roi_filter=roi_filter)

    ft = fft(imData, axis=1)
    power = np.real(2 * ft * ft.conjugate())
    # average across trials
    spectra = power[:, :(ft.shape[1] / 2), :].mean(axis=2)
    frequencies = fftfreq(imData[0, :, 0].size,
                          expt.frame_period())[:(ft.shape[1] / 2)]
    return spectra, frequencies


def _psth(expt, stimulus, ax=None, pre_frames=10, post_frames=20,
          exclude=None, data=None, transients_conf_level=99,
          dFOverF='from_file', smoothing=None, window_length=5,
          plot_mean=True, shade_ste=None, channel='Ch2', label=None,
          roi_filter=None, return_full=False, return_starts=False, shuffle=False,
          plot='heatmap', deduplicate=False, duplicate_window=None,
          z_score=False):
    """Calculates a psth based on trigger times

    Keyword arguments:
    stimulus -- either a behaviorData key or frames to trigger the PSTH,
        nTrials length list of trigger times for each trial
    pre_frames, post_frames -- frames preceding and following the stim
        frame to include
    exclude -- 'running' or nTrials x nFrames boolean array used to mask
        data
    data -- None, 'trans', imaging data
    smoothing -- window function to use, should be 'flat' for a moving
        average or np.'smoothing' (hamming, hanning, bartlett, etc.)
    window_length -- length of smoothing window function, should be odd
    shuffle -- if True, shuffle stim start times within each trial

    Returns:
    nRois x (pre_frames + 1 + post_frames) array of the average response
        to the stimulus

    Note:
    If stimulus is empty, returns a NaN array of the same size.

    """

    # see if stimulus is a behaviorData key
    if((not isinstance(stimulus, list)) or (not isinstance(stimulus, np.ndarray))):
        if isinstance(stimulus, basestring):
            stimulus = ba.stimStarts(expt, stimulus,
                                     deduplicate=deduplicate,
                                     duplicate_window=duplicate_window)
        elif isinstance(stimulus, int):
            frame = expt.imagingIndex(stimulus)
            stimulus = []
            stimulus.append(np.array([frame]))

    if pre_frames is None:
        frame_period = expt.frame_period()
        pre_frames = int(expt.stimulusTime() / frame_period)
    if post_frames is None:
        frame_period = expt.frame_period()
        n_frames = expt.imaging_shape()[1]
        post_frames = n_frames - int(expt.stimulusTime() / frame_period) \
            - 1

    if data is None:
        imData = expt.imagingData(dFOverF=dFOverF, channel=channel,
                                  label=label, roi_filter=roi_filter)
    elif data == 'trans':
        imData = isActive(
            expt, conf_level=transients_conf_level, channel=channel,
            label=label, roi_filter=roi_filter)
    elif data == 'raw':
        imData = expt.imagingData(channel=channel, label=label,
                                  roi_filter=roi_filter)
    else:
        imData = data

    imData = np.rollaxis(imData, 2, 0)
    (nTrials, nROIs, nFrames) = imData.shape

    if z_score:
        imData = zscore(imData, axis=2)

    nTriggers = int(np.sum([len(trial_stims) for trial_stims in stimulus]))
    if shuffle:
        stimulus = [np.random.randint(
            0, nFrames, len(trial_stims)) for trial_stims in stimulus]

    if exclude == 'running':
        exclude = expt.runningIntervals(
            imageSync=True, returnBoolList=True, end_padding=2.0)
        exclude = np.rollaxis(np.tile(exclude, (nROIs, 1, 1)), 1, 0)

    if exclude is not None:
        if exclude.shape[2] < nFrames:
            exclude = np.dstack(
                (exclude, np.zeros(
                    (nTrials, nROIs, nFrames - exclude.shape[2]),
                    dtype='bool')))
        elif exclude.shape[2] > nFrames:
            exclude = exclude[:, :, :nFrames]

        imData[exclude] = np.nan

    psth_data = np.empty([nROIs, pre_frames + post_frames + 1, nTriggers])
    psth_data.fill(np.nan)

    trig_idx = 0
    for triggers, data in it.izip(stimulus, imData):
        for trig in triggers:
            # Check for running off the ends
            if trig - pre_frames >= 0:
                window_start = trig - pre_frames
                data_start = 0
            else:
                window_start = 0
                data_start = pre_frames - trig
            if trig + post_frames < data.shape[1]:
                window_end = trig + post_frames + 1
                data_end = psth_data.shape[1]
            else:
                window_end = data.shape[1]
                data_end = data.shape[1] - trig - post_frames - 1

            psth_data[:, data_start:data_end, trig_idx] = \
                data[:, window_start:window_end]

            trig_idx += 1

    if return_full:
        if(return_starts):
            return psth_data, stimulus
        return psth_data

    # Taking the nanmean over an all NaN axis raises a warning, but it
    # will still return NaN there which is the intended behavior
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = nanmean(psth_data, axis=2)

    if smoothing is not None:
        if smoothing == 'flat':  # moving average
            w = np.ones(window_length, 'd')
        else:
            # If 'smoothing' is not a valid method this will throw an AttributeError
            w = eval('np.' + smoothing + '(window_length)')
        for idx, row in enumerate(result):
            s = np.r_[row[window_length - 1:0:-1], row, row[-1:-window_length:-1]]
            tmp = np.convolve(w / w.sum(), s, mode='valid')
            # Trim away extra frames
            result[idx] = tmp[window_length / 2 - 1:-window_length / 2]

    if ax:
        framePeriod = expt.frame_period()
        xAxis = np.linspace(
            -pre_frames, post_frames, result.shape[1]) * framePeriod

        if shade_ste:
            if shade_ste == 'sem':
                ste_psth = np.std(result, axis=0) / np.sqrt(nROIs)
            elif shade_ste == 'std':
                ste_psth = np.std(result, axis=0)
            else:
                raise ValueError(
                    'Unrecognized error shading argument: {}'.format(
                        shade_ste))

            mean_psth = np.mean(result, axis=0)

            if plot == 'heatmap':
                ax.imshow(mean_psth, interpolation='none', aspect='auto')
            elif plot == 'line':
                ax.plot(xAxis, mean_psth, color='b', lw=1)
                ax.fill_between(
                    xAxis, mean_psth - ste_psth, mean_psth + ste_psth,
                    facecolor='r', alpha=0.4)
        else:
            for roi in result:
                ax.plot(xAxis, roi)
            if plot_mean:
                mean_psth = np.mean(result, axis=0)
                ax.plot(xAxis, mean_psth, color='k', lw=2)

        ax.axvline(0, linestyle='dashed', color='k')
        ax.set_xlim((-pre_frames * framePeriod, post_frames * framePeriod))

        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'Mean $\Delta$F/F')

    return result


"""
ExperimentGroup analysis functions
"""


@memoize
def population_activity_new(
        expt_grp, stat, channel='Ch2', label=None, roi_filter=None,
        interval=None, **imaging_kwargs):
    """Calculates various activity metrics on each cell.

    Parameters
    ----------
    stat : str
        Metric to calculate. See lab.analysis.calc_activity for details.
    interval : dict of lab.classes.interval.Interval
        Dictionary by experiment of Interval objects corresponding to frames
        to include.
    **imaging_kwargs
        Additional arguments are passed to expt.imagingData.

    Returns
    -------
    pd.DataFrame
        Returns DataFrame with 'trial', 'roi', and 'value' as columns.

    """

    activity = []
    for expt in expt_grp:
        if interval is None:
            expt_interval = None
        else:
            expt_interval = interval[expt]

        if label is None:
            try:
                label = expt_grp.args['imaging_label']
            except (KeyError, AttributeError):
                pass

        expt_activity = ca.calc_activity(
            expt, channel=channel, label=label, roi_filter=roi_filter,
            method=stat, interval=expt_interval, **imaging_kwargs)

        expt_rois = expt.rois(
            channel=channel, label=label, roi_filter=roi_filter)

        assert expt_activity.shape[0] == len(expt_rois)
        assert expt_activity.shape[1] == len(expt.findall('trial'))

        for trial_idx, trial in enumerate(expt.findall('trial')):
            for roi_activity, roi in it.izip(expt_activity, expt_rois):
                activity.append(
                    {'trial': trial, 'roi': roi,
                     'value': roi_activity[trial_idx]})

    return pd.DataFrame(activity, columns=['trial', 'roi', 'value'])


def population_activity(
        exptGrp, stat, channel='Ch2', label=None, roi_filter=None,
        interval=None, dF='from_file', running_only=False,
        non_running_only=False, running_kwargs=None):
    """Calculates the activity of a population of cells with various methods

    Keyword arguments:
    exptGrp -- pcExperimentGroup to analyze
    stat -- activity statistic to calculate, see calc_activity for details
    interval -- [(start1, stop1), ...] times to include, or boolean mask
    running_only/non_running_only -- If True, only include running/non-running
        intervals
    dF -- dF method to use on imaging data
    running_kwargs -- Optional, arguments for running intervals

    """

    if running_kwargs is None:
        running_kwargs = {}

    if label is None:
        try:
            label = expt_grp.args['imaging_label']
        except (KeyError, AttributeError):
            pass

    activity = []
    for expt in exptGrp:
        (nROIs, nFrames, nTrials) = expt.imaging_shape(
            channel=channel, label=label, roi_filter=roi_filter)

        if interval is not None:
            if interval.shape == (nROIs, nFrames, nTrials):
                # Hackily allow passing in pre-calced intervals
                # No reason nROIs will be the same for all expts, so need a
                # better solution here
                expt_interval = interval
            else:
                frame_period = expt.frame_period()
                expt_interval = np.zeros(
                    (nROIs, nFrames, nTrials), dtype='bool')
                for inter in interval:
                    expt_interval[:,
                                  int(inter[0] / frame_period):
                                  int(inter[1] / frame_period) + 1, :] = True
        else:
            expt_interval = None

        if running_only:
            running_intervals = np.array(expt.runningIntervals(
                returnBoolList=True, imageSync=True, **running_kwargs))
            running_intervals = np.tile(running_intervals.T, (nROIs, 1, 1))
            assert running_intervals.shape == (nROIs, nFrames, nTrials)
            if expt_interval is None:
                expt_interval = running_intervals
            else:
                expt_interval = np.logical_and(
                    expt_interval, running_intervals)
        elif non_running_only:
            non_running_intervals = ~np.array(expt.runningIntervals(
                returnBoolList=True, imageSync=True, **running_kwargs))
            non_running_intervals = np.tile(
                non_running_intervals.T, (nROIs, 1, 1))
            assert non_running_intervals.shape == (nROIs, nFrames, nTrials)
            if expt_interval is None:
                expt_interval = non_running_intervals
            else:
                expt_interval = np.logical_and(
                    expt_interval, non_running_intervals)

        expt_activity = ca.calc_activity(
            expt, channel=channel, label=label, roi_filter=roi_filter,
            method=stat, interval=expt_interval, dF=dF)

        expt_rois = expt.rois(
            channel=channel, label=label, roi_filter=roi_filter)

        assert expt_activity.shape[0] == len(expt_rois)
        assert expt_activity.shape[1] == len(expt.findall('trial'))

        for trial_idx, trial in enumerate(expt.findall('trial')):
            for roi_activity, roi in it.izip(expt_activity, expt_rois):
                activity.append(
                    {'trial': trial, 'roi': roi,
                     'value': roi_activity[trial_idx]})

    return pd.DataFrame(activity, columns=['trial', 'roi', 'value'])


def trace_sigma(
        expt_grp, channel='Ch2', label=None, roi_filter=None, **trans_kwargs):
    """Calculates the standard deviation of the calcium trace for each ROI.

    Parameters
    ----------
    **trans_kwargs
        All additional keyword arguments are passed to expt.transientsData

    Returns
    -------
    DataFrame
        Returns a DataFrame with trial, roi and value as columns

    """

    rois = expt_grp.rois(channel=channel, label=label, roi_filter=roi_filter)

    data_list = []

    for expt in expt_grp:
        if not expt.hasTransientsFile(channel=channel):
            continue
        sigmas = expt.transientsData(
            channel=channel, label=label, roi_filter=roi_filter,
            **trans_kwargs)['sigma']
        trials = expt.findall('trial')
        assert len(sigmas) == len(rois[expt])
        for roi, sigma in zip(rois[expt], sigmas):
            assert len(sigma) == len(trials)
            for trial, s in zip(trials, sigma):
                data_list.append(
                    {'trial': trial, 'roi': roi, 'value': float(s)})

    return pd.DataFrame(data_list, columns=['trial', 'roi', 'value'])


@memoize
def mean_fluorescence(
        expt_grp, channel='Ch2', label=None, roi_filter=None):
    """Calculates the mane raw fluorescence (not dF/F) for each ROI.

    Parameters
    ----------

    Returns
    -------
    DataFrame
        Returns a DataFrame with expt, roi, and value as columns.

    """

    rois = expt_grp.rois(channel=channel, label=label, roi_filter=roi_filter)

    data_list = []

    for expt in expt_grp:
        dset = expt.imaging_dataset()
        ch_idx = sima.misc.resolve_channels(channel, dset.channel_names)
        time_average = dset.time_averages[..., ch_idx]

        for roi in rois[expt]:

            mask = roi.mask

            assert len(mask) == time_average.shape[0]

            vals = []
            for plane, plane_mask in zip(time_average, mask):
                assert plane_mask.shape == plane.shape

                vals.append(plane[np.array(plane_mask.todense())])

            roi_mean = np.concatenate(vals).mean()

            data_list.append({'expt': expt, 'roi': roi, 'value': roi_mean})

    return pd.DataFrame(data_list, columns=['expt', 'roi', 'value'])


def running_modulation(
        exptGrp, roi_filter, stat, population_activity_kwargs=None,
        mode='difference'):
    """Calculates for each ROI a population metric in running and non-running
    conditions.  Returns either the difference or ratio of the metric between
    conditions, set by mode='difference' or mode='ratio'

    Returns a pandas df

    """

    if not population_activity_kwargs:
        population_activity_kwargs = {}

    running_df = population_activity(
        exptGrp, stat, roi_filter=roi_filter, running_only=True,
        **population_activity_kwargs)

    nonrunning_df = population_activity(
        exptGrp, stat, roi_filter=roi_filter, non_running_only=True,
        **population_activity_kwargs)

    df = pd.merge(
        running_df, nonrunning_df, how='inner', on='roi',
        suffixes=['_running', '_nonrunning'])

    if mode == 'difference':
        df['value'] = df['value_running'] - df['value_nonrunning']

    if mode == 'ratio':
        df['value'] = df['value_running'] / df['value_nonrunning']

    df = df.dropna()
    df['trial'] = df['trial_running']

    df.drop(['value_running', 'value_nonrunning', 'trial_running',
             'trial_nonrunning'], axis=1, inplace=True)

    return df


def baselineActivityCorrelations(
        exptGrp, ax=None, cbarAx=None, includeStimResponse=False,
        offset=False, cluster=None, reordering=None, colorcode=None,
        channel='Ch2', label=None, roi_filter=None):
    """
    Return the correlation matrix for the activity across ROIs before the
    stimuli
    """

    ROIs = exptGrp.sharedROIs(
        channel=channel, label=label, roi_filter=roi_filter)
    shared_filter = lambda x: x.id in ROIs
    corrs = np.zeros([len(ROIs), len(ROIs)])
    totalFrames = 0.
    for expt in exptGrp:
        imData = expt.imagingData(channel=channel,
                                  label=label, roi_filter=shared_filter)
        if not includeStimResponse and isinstance(expt, SalienceExperiment):
            stimIdx = expt.imagingIndex(expt.stimulusTime())
            imData = imData[:, 0:(stimIdx - 1), :]
        imData = imData.reshape([imData.shape[0], -1], order='F')
        # collapse trial dimension of imaging data and take the correlation
        # matrix
        if offset:
            tmpCorr = offsetCorrelation(imData, pcaClean=True)
            # corrs = np.minimum(corrs, 1.)
        else:
            tmpCorr = corrcoef(imData)
        corrs += tmpCorr * imData.shape[1]
        totalFrames += imData.shape[1]
    corrs /= totalFrames
    if ax is not None:
        if cluster == 'complete':
            distances = 1 - corrs
            condensedDistances = squareform(distances, checks=False)
            linkageMatrix = hierarchy.complete(condensedDistances)
        elif cluster is not None:
            linkageMatrix = hierarchy.ward(corrs)
        if cluster is not None:
            ordering = hierarchy.leaves_list(linkageMatrix)
            if reordering is not None:
                ordering = ordering[reordering]
            corrs = corrs[ordering, :]
            corrs = corrs[:, ordering]
            # plt.figure(); hierarchy.dendrogram(linkageMatrix, labels=ROIs)
            ROIs = [ROIs[i] for i in ordering]

        im = ax.imshow(corrs, interpolation='nearest', aspect=1.0,
                       vmin=-1.0, vmax=1.0, cmap='bwr')
        if cbarAx:
            cbar = cbarAx.figure.colorbar(im, ax=cbarAx, ticks=[-1, 1],
                                          fraction=0.05)
            cbar.set_label('correlation', labelpad=-5)
        ax.set_xticks([])
        ax.tick_params(axis='y', color='white', labelcolor='k')
        try:
            roiGroups, roiGroupNames = bouton.BoutonSet(ROIs).boutonGroups()
        except:
            ax.set_yticks(range(len(ROIs)))
            ax.set_yticklabels(ROIs)
        else:
            if colorcode == "postSynaptic":
                for k, group in enumerate(roiGroups):
                    for roi in group:
                        #    if roiGroupNames[k] != 'other':
                        ax.add_patch(plt.Rectangle(
                            (-2, ROIs.index(roi.id) - 0.5), 1, 1,
                            color=bouton.groupPointStyle(roiGroupNames[k])[0],
                            lw=0))
                ax.set_xlim(left=-2)
            elif colorcode == "axon":
                colors = ['b', 'r', 'c', 'm', 'y', 'g', 'b', 'r']
                for k, group in enumerate(bouton.BoutonSet(ROIs).axonGroups()):
                    for roi in group:
                        ax.add_patch(plt.Rectangle(
                            (-2, ROIs.index(roi.id) - 0.5), 1, 1,
                            color=colors[k], lw=0))
                ax.set_xlim(left=-2)
            ax.set_xlim(right=ax.get_xlim()[1])
            ax.set_axis_off()
    return corrs, ROIs


def autocorrelation(exptGrp, channel='Ch2', label=None, roi_filter=None):
    ROIs = exptGrp.sharedROIs(
        channel=channel, label=label, roi_filter=roi_filter)
    shared_filter = lambda x: x.id in ROIs
    ac = []
    for roiIdx in xrange(len(ROIs)):
        ac.append(np.zeros([]))
    for expt in exptGrp:
        imData = expt.imagingData(dFOverF='mean',
                                  channel=channel, label=label,
                                  roi_filter=shared_filter)
        for roiIdx in xrange(len(ROIs)):
            for i in xrange(imData.shape[2]):
                try:
                    ac[roiIdx] = ac[roiIdx] + np.correlate(
                        imData[roiIdx, :, i] - imData[roiIdx, :, i].mean(),
                        imData[roiIdx, :, i] - imData[roiIdx, :, i].mean(),
                        'full')
                except:
                    tmpCorr = np.correlate(
                        imData[roiIdx, :, i] - imData[roiIdx, :, i].mean(),
                        imData[roiIdx, :, i] - imData[roiIdx, :, i].mean(),
                        'full')
                    ac[roiIdx] = ac[roiIdx][:tmpCorr.size] + \
                        tmpCorr[:ac[roiIdx].size]
    for roiIdx in range(len(ROIs)):
        ac[roiIdx] = ac[roiIdx][ac[roiIdx].size / 2:]
        ac[roiIdx] = ac[roiIdx] / ac[roiIdx][0]
    return np.array(ac, dtype=float)


def trialAverages(exptGrp, stimulus, duration=None, excludeRunning=False,
                  power=None, removeNanBoutons=False, **imaging_kwargs):
    timeSeries = []
    for expt in exptGrp:
        timeSeries.append(expt.imagingData(**imaging_kwargs))
        if expt.get('experimentType') == 'contextualFearConditioning':
            trialIndices = [0]
        else:
            trialIndices = expt.trialIndices(
                stimulus, duration, power=power,
                excludeRunning=excludeRunning)
        timeSeries[-1] = timeSeries[-1][:, :, trialIndices]
    timeSeries = np.concatenate(timeSeries, axis=2)
    if removeNanBoutons:
        timeSeries = timeSeries[np.nonzero(np.all(np.isfinite(
            timeSeries.reshape([timeSeries.shape[0], -1], order='F')),
            axis=1))[0], :, :]
    return np.mean(timeSeries, axis=2)


def peakAverageResponses(exptGrp, stimulus, postStimDuration=1.5,
                         duration=None, excludeRunning=False, power=None,
                         removeNanBoutons=False, **kwargs):
    t = trialAverages(exptGrp, stimulus, duration=duration,
                      excludeRunning=excludeRunning, power=power,
                      removeNanBoutons=removeNanBoutons, **kwargs)
    if exptGrp[0].get('experimentType') == 'contextualFearConditioning' and \
            stimulus == 'context':
        stimIdx = exptGrp[0].imagingIndex(exptGrp.contextInterval()[0])
    else:
        # TODO: make robust to different stim times and imaging parameters
        stimIdx = exptGrp[0].imagingIndex(exptGrp[0].stimulusTime())
    maxIdx = stimIdx + exptGrp[0].imagingIndex(postStimDuration)
    baseline = nanmean(t[:, :stimIdx], axis=1)
    peaks = (np.nanmax(t[:, stimIdx:maxIdx], axis=1) - baseline) / baseline
    return peaks


def averageResponseIntegrals(
        exptGrp, stimulus, duration=None, ROIs=None, power=None,
        excludeRunning=False, demixed=False, dFOverF=None,
        postStimDuration=1.5, sharedBaseline=True, linearTransform=None,
        channel='Ch2', label=None, roi_filter=None):
    """
    Return the integral of the trial-averaged response of each roi to the
    stimulus
    """

    return responseIntegrals(
        exptGrp, stimulus, postStimDuration=postStimDuration,
        duration=duration, excludeRunning=excludeRunning, demixed=demixed,
        dFOverF=dFOverF, sharedBaseline=sharedBaseline,
        linearTransform=linearTransform, channe=channel, label=label,
        roi_filter=roi_filter).mean(axis=1)


def responseIntegrals(exptGrp, stimuli, postStimDuration=1.5, duration=None,
                      power=None, excludeRunning=False, demixed=False,
                      dFOverF=None, sharedBaseline=False,
                      linearTransform=None, channel='Ch2', label=None,
                      roi_filter=None):
    ROIs = exptGrp.sharedROIs(
        channel=channel, label=label, roi_filter=roi_filter)
    shared_filter = lambda x: x.id in ROIs
    if not isinstance(stimuli, list):
        stimuli = [stimuli]
    integrals = []
    for experiment in exptGrp:
        if isinstance(experiment, SalienceExperiment) or isinstance(
                experiment, FearConditioningExperiment):
            for stimulus in stimuli:
                if isinstance(experiment, FearConditioningExperiment) or \
                        stimulus in experiment.stimuli():
                    integrals.append(experiment.responseIntegrals(
                        stimulus, postStimDuration=postStimDuration,
                        linearTransform=linearTransform,
                        excludeRunning=excludeRunning, demixed=demixed,
                        duration=duration, power=power,
                        dFOverF=dFOverF,
                        sharedBaseline=sharedBaseline, channel=channel,
                        label=label, roi_filter=shared_filter))
    if len(integrals):
        return np.concatenate(integrals, axis=1)
    else:
        return np.zeros([len(ROIs), 0])


# TODO: Figure out what's up w/ this function, looks broken
def responseZScores(
        exptGrp, stimuli, postStimDuration=1.5, duration=None, power=None,
        excludeRunning=False, demixed=False, dFOverF=None,
        sharedBaseline=False, linearTransform=None, channel='Ch2',
        label=None, roi_filter=None):
    if stimuli == 'running':
        modulations = runningModulation(
            exptGrp, channel=channel, label=label, roi_filter=roi_filter)
    else:
        if not isinstance(stimuli, list):
            stimuli = [stimuli]
        response_integrals = responseIntegrals(
            exptGrp, stimuli, duration=duration, power=power,
            excludeRunning=excludeRunning, demixed=demixed,
            dFOverF=dFOverF, postStimDuration=postStimDuration,
            sharedBaseline=sharedBaseline, linearTransform=linearTransform,
            channel=channel, label=label, roi_filter=roi_filter)
        return response_integrals.mean(axis=1) / np.sqrt(
            np.var(response_integrals, axis=1)) * np.sqrt(
                response_integrals.shape[1])


# This function also looks broken
def runningModulation(
        exptGrp, voidRange=None, returnFull=False, minRunningFrames=3,
        padding=0.5, pcaCleanup=False, linearTransform=None,
        signal='imaging_data', LFP_freq_range=(4, 8), channel='Ch2',
        label=None, roi_filter=None):
    """
    Evaluate how signals are modulated by running.

    Inputs:
      signal: {'imaging_data', 'LFP'} -- determines whether to analyze imaging
      data or LFP void range: pre-set range to exclude

    Return:

    """
    if signal == 'LFP':
        modulations = []
        runDurations = []
        for experiment in exptGrp:
            blankData = []
            runData = []
            experimentModulations = []
            if experiment.get('experimentType') in ['salience', 'doubleStimulus', 'intensityRanges']:
                stimTime = experiment.stimulusTime()
                postStimTime = experiment.stimulusTime() + 5.
            for trial in experiment.findall('trial'):
                LFP_power, LFP_times = trial.freqBandPower(LFP_freq_range[0], LFP_freq_range[1])
                dt = LFP_times[1] - LFP_times[0]
                if voidRange is None:
                    if experiment.get('experimentType') in ['salience', 'doubleStimulus', 'intensityRanges']:
                        tmpVoidRanges = [[stimTime, postStimTime]]
                    elif experiment.get('experimentType') == 'contextualFearConditioning':
                        tmpVoidRanges = [[experiment.contextInterval()[0], np.Inf]]
                    else:
                        tmpVoidRanges = []
                else:
                    tmpVoidRanges = [voidRange]
                tmpVoidRanges = linePartition.LinePartition(tmpVoidRanges)
                runIntervals = linePartition.LinePartition([list(x) for x in
                        ba.runningIntervals(trial, stationary_tolerance=2.2, imageSync=False, end_padding=padding)])
                validRunIntervals = linePartition.intersection(runIntervals,
                        linePartition.complement(tmpVoidRanges))
                nonRunningIntervals = linePartition.complement(linePartition.union(tmpVoidRanges, runIntervals))

                for interval in nonRunningIntervals:
                    startIdx = 0 if interval[0] == -np.Inf else int(interval[0] / dt)
                    endIdx = len(LFP_times) if interval[1] == np.Inf else int(interval[1] / dt)
                    blankData.append(LFP_power[startIdx:endIdx])

                for interval in validRunIntervals:
                    startIdx = 0 if interval[0] == -np.Inf else int(interval[0] / dt)
                    endIdx = len(LFP_times) if interval[1] == np.Inf else int(interval[1] / dt)
                    runData.append(LFP_power[startIdx:endIdx])
                    runDurations.append(interval[1] - interval[0])

            baseline = np.concatenate(blankData).mean()
            for runEpoch in runData:
                runEpoch /= baseline
                runEpoch -= 1
            modulations.extend(runData)
        if len(modulations):
            modulations = np.concatenate(modulations, axis=1)
        runDurations = np.array(runDurations)
        if returnFull:
            return modulations, runDurations
        else:
            raise Exception('Code incomplete')
            meanModulations = np.zeros(len(ROIs))
            for i, d in enumerate(runDurations):
                meanModulations += modulations[:, i] * d
            meanModulations /= runDurations.sum()
            return meanModulations

    if signal == 'imaging_data':
        # find shared ROIs and pca cleanup matrix
        ROIs = exptGrp.sharedROIs(
            channel=channel, label=label, roi_filter=roi_filter)
        shared_filter = lambda x: x.id in ROIs
        if not len(ROIs):
            raise exc.NoSharedROIs
        if pcaCleanup:
            assert linearTransform is None
            linearTransform = exptGrp.pcaCleanupMatrix(ROIs)
        modulations = []
        runDurations = []
        for experiment in exptGrp:
            blankData = []
            runData = []
            if signal == 'imaging_data':
                if not len(experiment.findall('trial')):
                    raise exc.MissingTrial
                imData = experiment.imagingData(
                    dFOverF='mean', demixed=False,
                    linearTransform=linearTransform, channel=channel,
                    label=label, roi_filter=shared_filter)
                imData += 1.
                imageSync = True
            elif signal == 'LFP':
                imageSync = False
            runInts = experiment.runningIntervals(stationary_tolerance=2.2, imageSync=imageSync)
            if experiment.get('experimentType') in ['salience', 'doubleStimulus', 'intensityRanges']:
                stimIdx = experiment.imagingIndex(experiment.stimulusTime())
                maxIdx = experiment.imagingIndex(experiment.stimulusTime() + 5.)
            padIndices = experiment.imagingIndex(padding)
            for cycleIdx in range(min(imData.shape[2], len(experiment.findall('trial')))):
                if voidRange is None:
                    if experiment.get('experimentType') in ['salience', 'doubleStimulus', 'intensityRanges']:
                        if experiment.findall('trial')[cycleIdx].get('stimulus') == 'light':
                            tmpVoidRange = []
                        else:
                            tmpVoidRange = range(stimIdx, maxIdx)
                    elif experiment.get('experimentType') == 'contextualFearConditioning':
                        onsetIdx = experiment.imagingIndex(experiment.contextInterval()[0])
                        tmpVoidRange = range(onsetIdx, imData.shape[1])
                    else:
                        tmpVoidRange = []
                else:
                    tmpVoidRange = voidRange
                runningIndices = []
                for interval in runInts[cycleIdx]:
                    if interval[0] not in tmpVoidRange:
                        runningIndices.extend(
                            [x for x in range(interval[0] - padIndices,
                                              interval[1] + padIndices + 1)
                             if (x not in tmpVoidRange)
                             and x < imData.shape[1] and x >= 0])
                nonRunningIndices = [x for x in range(imData.shape[1])
                                     if x not in runningIndices
                                     and x not in tmpVoidRange]
                blankData.append(imData[:, nonRunningIndices, cycleIdx])
                if len(runningIndices) >= minRunningFrames:
                    assert exptGrp[0].parent.get('mouseID') != 'al1'
                    runData.append(imData[:, runningIndices, cycleIdx].mean(axis=1).reshape([-1, 1]))
                    runDurations.append(len(runningIndices))
            baseline = np.concatenate(blankData, axis=1).mean(axis=1).reshape([-1, 1])
            for runEpoch in runData:
                runEpoch /= baseline
                runEpoch -= 1
            modulations.extend(runData)
        if len(modulations):
            modulations = np.concatenate(modulations, axis=1)
        runDurations = np.array(runDurations)
        if returnFull:
            return modulations, runDurations
        else:
            meanModulations = np.zeros(len(ROIs))
            for i, d in enumerate(runDurations):
                meanModulations += modulations[:, i] * d
            meanModulations /= runDurations.sum()
            return meanModulations


def lickingModulation(exptGrp, voidRange=None, returnFull=False,
                      minLickingFrames=3, padding=0., pcaCleanup=False,
                      linearTransform=None, excludeRunning=True,
                      channel='Ch2', label=None, roi_filter=None):
    """For each ROI, return the (L-B)/B, where L is the activity during licking intervals,
    and B is the baseline activity, i.e during non-licking intervals

    """

    ROIs = exptGrp.sharedROIs(
        channel=channel, label=label, roi_filter=roi_filter)
    shared_filter = lambda x: x.id in ROIs
    if not len(ROIs):
        raise exc.NoSharedROIs
    if pcaCleanup:
        assert linearTransform is None
        linearTransform = exptGrp.pcaCleanupMatrix(
            channel=channel, label=label, roi_filter=shared_filter)
    modulations = []
    lickDurations = []
    for experiment in exptGrp:
        blankData = []
        lickData = []
        if not len(experiment.findall('trial')):
            raise exc.MissingTrial
        imData = experiment.imagingData(
            dFOverF='mean', demixed=False,
            linearTransform=linearTransform, channel=channel, label=label,
            shared_filter=shared_filter)
        imData += 1.
        lickInts = experiment.lickingIntervals(imageSync=True, threshold=20 * experiment.frame_period())
        if experiment.get('experimentType') in ['salience', 'doubleStimulus', 'intensityRanges']:
            stimIdx = experiment.imagingIndex(experiment.stimulusTime())
            maxIdx = experiment.imagingIndex(experiment.stimulusTime() + 5.)
        padIndices = experiment.imagingIndex(padding)
        for cycleIdx in range(min(imData.shape[2],
                              len(experiment.findall('trial')))):
            # determine range of data to be ignored
            if voidRange is None:
                if experiment.get('experimentType') in [
                        'salience', 'doubleStimulus', 'intensityRanges']:
                    if experiment.findall('trial')[cycleIdx].get(
                            'stimulus') == 'light':
                        tmpVoidRange = []
                    else:
                        tmpVoidRange = range(stimIdx, maxIdx)
                elif experiment.get('experimentType') \
                        == 'contextualFearConditioning':
                    onsetIdx = experiment.imagingIndex(experiment.contextInterval()[0])
                    tmpVoidRange = range(onsetIdx, imData.shape[1])
                else:
                    tmpVoidRange = []
            else:
                tmpVoidRange = voidRange
            lickingIndices = []
            for interval in lickInts[cycleIdx]:
                if interval[0] not in tmpVoidRange:
                    lickingIndices.extend([x for x in range(interval[0] - padIndices,
                        interval[1] + padIndices + 1)
                        if (x not in tmpVoidRange) and x < imData.shape[1] and x >= 0])
            nonLickingIndices = [x for x in range(imData.shape[1]) if (x not in lickingIndices)
                    and x not in tmpVoidRange]
            if excludeRunning:
                runInts = experiment.runningIntervals(stationary_tolerance=2.2)[0]
                nonLickingIndices = [i for i in nonLickingIndices
                        if not any([i <= M and i >= m for m, M in runInts])]
            blankData.append(imData[:, nonLickingIndices, cycleIdx])
            if len(lickingIndices) >= minLickingFrames:
                lickData.append(imData[:, lickingIndices, cycleIdx].mean(axis=1).reshape([-1, 1]))
                lickDurations.append(len(lickingIndices))
        baseline = np.concatenate(blankData, axis=1).mean(axis=1).reshape([-1, 1])
        for lickEpoch in lickData:
            lickEpoch /= baseline
            lickEpoch -= 1
        modulations.extend(lickData)
    if len(modulations):
        modulations = np.concatenate(modulations, axis=1)
    lickDurations = np.array(lickDurations)
    if returnFull:
        return modulations, lickDurations
    else:
        meanModulations = np.zeros(len(ROIs))
        for i, d in enumerate(lickDurations):
            meanModulations += modulations[:, i] * d
        meanModulations /= lickDurations.sum()
        return meanModulations


def runTriggeredTraces(exptGrp, linearTransform=None, prePad=15, postPad=35,
                       voidRange=None, postRunningTime=3., channel='Ch2',
                       label=None, roi_filter=None):
    ROIs = exptGrp.sharedROIs(
        channel=channel, label=label, roi_filter=roi_filter)
    shared_filter = lambda x: x.id in ROIs
    if not len(ROIs):
        raise exc.NoSharedROIs
    runningTriggeredData = []
    for experiment in exptGrp:
        postRunIndices = experiment.imagingIndex(postRunningTime)
        imData = experiment.imagingData(
            dFOverF='mean', demixed=False,
            linearTransform=linearTransform, channel=channel, label=label,
            roi_filter=shared_filter)
        runInts = experiment.runningIntervals(stationary_tolerance=2.2)
        if experiment.get('experimentType') in ['salience',
                                                'doubleStimulus',
                                                'intensityRanges']:
            stimIdx = experiment.imagingIndex(experiment.stimulusTime())
            maxIdx = experiment.imagingIndex(
                experiment.stimulusTime() + 5.)
        for cycleIdx in range(imData.shape[2]):
            if voidRange is None:
                if experiment.get('experimentType') in ['salience',
                                                        'doubleStimulus',
                                                        'intensityRanges']:
                    if experiment.findall('trial')[
                            cycleIdx].get('stimulus') == 'light':
                        tmpVoidRange = []
                    else:
                        tmpVoidRange = range(stimIdx, maxIdx)
                elif experiment.get('experimentType') == \
                        'contextualFearConditioning':
                    onsetIdx = experiment.imagingIndex(
                        experiment.contextInterval()[0])
                    tmpVoidRange = range(onsetIdx, imData.shape[1])
                else:
                    tmpVoidRange = []
            else:
                tmpVoidRange = voidRange
            for interval in runInts[cycleIdx]:
                if interval[0] not in tmpVoidRange and \
                        interval[0] < imData.shape[1]:
                    indices = np.arange(interval[0] - prePad,
                                        min(interval[1] + postRunIndices,
                                            imData.shape[1]))
                    m = min(prePad + postPad, len(indices))
                    indices = indices[:m]
                    assert interval[0] in list(indices)
                    if indices.size:
                        preClip = -min(indices.min(), 0)
                        runData = np.nan * np.ones([imData.shape[0],
                                                   prePad + postPad])
                        runData[:, preClip:len(indices)] = \
                            imData[:, indices[preClip:], cycleIdx]
                        for i in range(preClip, prePad):
                            if indices[i] in tmpVoidRange or any(
                                    [indices[i] in I for I in runInts[cycleIdx]
                                     if all(I != interval)]):
                                runData[:, :(i + 1)] = np.nan
                        for i in range(prePad, len(indices)):
                            if indices[i] in tmpVoidRange:
                                runData[:, i:] = np.nan
                        assert np.isfinite(runData[0, list(indices).index(
                            interval[0])])
                        runningTriggeredData.append(runData.reshape(
                            [runData.shape[0], runData.shape[1], 1]))
    if runningTriggeredData:
        return np.concatenate(runningTriggeredData, axis=2)
    return np.nan * np.ones([len(ROIs), imData.shape[1], 1])


def getSalienceTraces(expGroup):
    """
    method to get the salience data and return a pandas dataframe

    Arguments:
    expGroup -- a lab.ExperimentGroup instance that contains only salience
        experiments

    return a pandas dataframe instance that has the columns ROI, expt, trialNum,
        stimulus, dFOF, time, and stimTime.
    """
    data = dict()
    data["ROI"] = []
    data["expt"] = []
    data["trialNum"] = []
    data["stimulus"] = []
    data["dFOF"] = []
    data["time"] = []
    data["stimTime"] = []

    for exp in expGroup:
        iData = np.transpose(exp.imagingData(dFOverF="from_file"), (2, 0, 1))
        stimuli = map(lambda trial: trial.get("stimulus"), exp.findall("trial"))
        mouse = exp.parent;
        rois = map(lambda roi: (mouse, roi[1], roi[2]), exp.roi_tuples())
        numROIs = exp.num_rois()
        imagingTimes = [exp.imagingTimes() - exp.stimulusTime()] * numROIs
        stimTime = [exp.stimulusTime()]*numROIs

        for i in np.r_[0:len(stimuli)]:
            data["ROI"].extend(rois)
            data["expt"].extend([exp] * numROIs)
            data["trialNum"].extend([i] * numROIs)
            data["stimulus"].extend([stimuli[i]] * numROIs)
            data["dFOF"].extend(list(iData[i, :, :]))
            data["time"].extend(imagingTimes)
            data["stimTime"].extend(stimTime)

    return pd.DataFrame(data)



def PSTH(exptGrp, stimulus, ax=None, pre_time=5, post_time=5,
         channel='Ch2', label=None, roi_filter=None, plot_mean=True,
         shade_ste=None, return_full=False, return_df=False, exclude=None, data=None,
         gray_traces=False, color=None, **kwargs):
    """PSTH method for combining data within an ExperimentGroup.
    See expt.psth for details.

    stimulus, exclude, and data can also be a dictionary with experiments
    as keys that will be passed in to each call of Experiment.psth

    Returns an array n_total_rois x n_frames as well as an n_total_rois
    length list of (Mouse, location, roi_id) tuples that uniquely identify
    each ROI and an x_range for each ROI.

    Also doesn't assume the same sampling_rate

    """

    if not isinstance(stimulus, dict):
        stimulus = {expt.totuple(): stimulus for expt in exptGrp}

    if not isinstance(exclude, dict):
        exclude = {expt.totuple(): exclude for expt in exptGrp}

    if not isinstance(data, dict):
        data = {expt.totuple(): data for expt in exptGrp}

    if pre_time is None:
        pre_time = 0
        for expt in exptGrp:
            pre_time = np.amax([pre_time, expt.stimulusTime()])
    if post_time is None:
        post_time = 0
        for expt in exptGrp:
            post_time = np.amax(
                [post_time, expt.imagingTimes()[-1] - expt.stimulusTime()])

    psth_data = {}
    stimStarts = {}
    x_range = {}

    df = dict()
    if(return_df):
        df["expt"] = []
        df["stimulus"] = []
        df["stimStart"] = []
        df["ROI"] = []
        df["activity"] = []
        df["time"] = []

    for expt in exptGrp:
        frame_period = expt.frame_period()
        pre_frames = int(pre_time / frame_period)
        post_frames = int(post_time / frame_period)
        psth_data[expt.totuple()], stimStarts[expt] = _psth(
            expt, stimulus[expt.totuple()], pre_frames=pre_frames,
            post_frames=post_frames, channel=channel, label=label,
            roi_filter=roi_filter, return_full=True, return_starts=True,
            exclude=exclude[expt.totuple()], data=data[expt.totuple()],
            duplicate_window=pre_time+post_time, **kwargs)
        x_range[expt.totuple()] = np.linspace(
            -pre_frames, post_frames, psth_data[expt.totuple()].shape[1]) \
            * frame_period
        assert 0 in x_range[expt.totuple()]

    all_rois = exptGrp.allROIs(
        channel=channel, label=label, roi_filter=roi_filter)
    result = []
    rois = []
    x_ranges = []
    if return_full:
        all_stacked_data = []
    for roi in all_rois:
        roi_data = []
        for roi_expt, roi_idx in all_rois[roi]:
            roi_data.append(psth_data[roi_expt.totuple()][roi_idx])
            if(return_df):
                psh = psth_data[roi_expt.totuple()][roi_idx]
                numPoints, numTraces = psh.shape
                df["stimulus"].extend([stimulus[roi_expt.totuple()]] * numTraces)
                df["stimStart"].extend([val for sublist in stimStarts[roi_expt] for val in sublist])
                df["ROI"].extend([roi] * numTraces)
                df["expt"].extend([roi_expt] * numTraces)
                df["activity"].extend(list(psth_data[roi_expt.totuple()][roi_idx].T))
                df["time"].extend([x_range[roi_expt.totuple()]] * numTraces)

        # If the experiments were all sampled at the same rate, just stack
        # the data, otherwise keep the most common frame rate, drop the rest
        try:
            stacked_data = np.hstack(roi_data)
        except ValueError:
            lengths = [r.shape[0] for r in roi_data]
            most_common = mode(lengths)[0][0]
            new_data = [r for r, length in zip(roi_data, lengths)
                        if length == most_common]
            stacked_data = np.hstack(new_data)
            warnings.warn(
                'ROI imaged at multiple frame rates, only keeping most ' +
                'common frame rate: mouse {}, loc {}, id {}'.format(
                    roi[0].get('mouseID'), roi[1], roi[2]))
            # Find an experiment that was sampled at the slowest rate
            roi_expt, _ = all_rois[roi][lengths.index(most_common)]

        if return_full:
            all_stacked_data.append(stacked_data)
        result.append(nanmean(stacked_data, axis=1))
        rois.append(roi)
        x_ranges.append(x_range[roi_expt.totuple()])

    if (ax or return_full == 'norm') and len(result):
        min_x_range_idx = np.argmin([len(x) for x in x_ranges])
        min_x_range = x_ranges[min_x_range_idx]
        normalized_psths = []
        for x_range, psth in it.izip(x_ranges, result):
            normalized_psths.append(np.interp(min_x_range, x_range, psth))
        normalized_psths = np.array(normalized_psths)

    if ax and len(result):
        if color is not None:
            light_color = sns.light_palette(color)[1]
        if shade_ste:
            if shade_ste == 'sem':
                ste_psth = nanstd(normalized_psths, axis=0) / np.sqrt(
                    normalized_psths.shape[0])
            elif shade_ste == 'std':
                ste_psth = nanstd(normalized_psths, axis=0)
            else:
                raise ValueError(
                    'Unrecognized error shading argument: {}'.format(
                        shade_ste))

            mean_psth = nanmean(normalized_psths, axis=0)

            ax.plot(min_x_range, mean_psth,
                    color='b' if color is None else color, lw=1)
            ax.fill_between(
                min_x_range, mean_psth - ste_psth, mean_psth + ste_psth,
                facecolor='r' if color is None else color, alpha=0.4)
        else:
            for x_range, psth in it.izip(x_ranges, result):
                if gray_traces or color is not None:
                    ax.plot(x_range, psth,
                            color='0.8' if color is None else light_color)
                else:
                    ax.plot(x_range, psth)
            if plot_mean:
                mean_psth = nanmean(normalized_psths, axis=0)
                ax.plot(min_x_range, mean_psth,
                        color='k' if color is None else color, lw=2)

        ax.axvline(0, linestyle='dashed', color='k')
        ax.set_xlim(min_x_range[0], min_x_range[-1])

        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'Mean $\Delta$F/F')

    if(return_df):
        return pd.DataFrame(df)

    if return_full == 'norm':
        if not len(result):
            return np.array([]), np.array([])
        return normalized_psths, min_x_range

    if return_full:
        return all_stacked_data, rois, x_ranges
    return result, rois, x_ranges


def response_magnitudes(
        exptGrp, stimulus, method='responsiveness', return_full=False,
        z_score=False, return_df=False, **kwargs):
    """Determine response magnitudes for all ROIs across all experiments.

    stimulus : stim to calculate responsiveness
    method : method used to calculate responsiveness.
        'responsiveness': difference in mean activity after and before stim
        'peak': difference in peak activity after and before stim
    return_df : If True, returns a pandas DataFrame
    **kwargs : additional keyword arguments to pass to psth method

    """
    psths, rois, x_ranges = PSTH(exptGrp, stimulus, return_full=True, **kwargs)

    responses = np.empty(len(psths))
    response_stds = np.empty(len(psths))
    for idx, psth, x_range in it.izip(it.count(), psths, x_ranges):
        if method == 'responsiveness':
            roi_responses = nanmean(psth[x_range > 0], axis=0) \
                - nanmean(psth[x_range < 0], axis=0)
        elif method == 'peak':
            roi_responses = np.nanmax(psth[x_range > 0], axis=0) \
                - np.nanmax(psth[x_range < 0], axis=0)
        else:
            raise ValueError('Unrecognized method: {}'.format(method))
        responses[idx] = nanmean(roi_responses)
        if return_full:
            response_stds[idx] = nanstd(roi_responses)

    # ROIs responding to 'off' stimuli, decrease their activity at the
    # stim times, so flip the sign of the response
    if 'off' in stimulus:
        responses *= -1

    if z_score:
        responses -= nanmean(responses)
        responses /= nanstd(responses)

        # This might be possible to scale, but for now this is incomplete
        response_stds = None

    if return_df:
        assert len(responses) == len(rois)
        if not len(rois):
            rois = np.empty((0, 3))
        df = pd.DataFrame(
            rois, columns=['mouse', 'uniqueLocationKey', 'roi_id'])
        df['value'] = responses
        return df
    if return_full:
        return responses, response_stds, psths, rois, x_ranges
    return responses


def response_matrix(
        exptGrp, stimuli, z_score=True, return_full=False, **response_kwargs):
    """Returns a matrix that is n_rois x n_stimuli of the responsiveness
    of each ROI to each stim"""

    responses = {}
    rois = {}
    for stim in stimuli:
        responses[stim], _, _, rois[stim], _ = response_magnitudes(
            exptGrp, stim, z_score=z_score, return_full=True,
            **response_kwargs)

    all_rois = list(set(it.chain(*rois.itervalues())))
    data = np.empty((len(all_rois), len(stimuli)))
    for roi_idx, roi in it.izip(it.count(), all_rois):
        for stim_idx, stim in enumerate(stimuli):
            try:
                data[roi_idx, stim_idx] = responses[stim][rois[stim].index(
                    roi)]
            except ValueError:
                data[roi_idx, stim_idx] = np.nan

    if return_full:
        return data, all_rois
    return data


def _expt_grp_response_magnitudes_shuffler(inputs):
    stimulus, method, pre_time, post_time, exclude, data, channel, \
        label, roi_filter = inputs
    global expt_grp
    return response_magnitudes(
        expt_grp, stimulus, method=method, pre_time=pre_time,
        post_time=post_time, exclude=exclude, data=data, channel=channel,
        label=label, roi_filter=roi_filter, shuffle=True)


# TODO: SHOULD THIS BE MOVED TO FILTERS?
def identify_stim_responsive_cells(
        exptGrp, stimulus, ax=None, method='responsiveness', data=None,
        pre_time=None, post_time=None, conf_level=95, sig_tail='upper',
        transients_conf_level=99, exclude=None, shuffle_exclude='exclude',
        plot_mean=True, shade_ste=None, channel='Ch2', label=None,
        roi_filter=None, n_bootstraps=10000, dFOverF='from_file',
        save_to_expt=True, ignore_saved=False, n_processes=1,
        return_roi_tuple_filter=False, return_set=False):
    """Identifies cells that significantly respond to a given stimulus

    See Experiment.psth for most of the arguments.

    Parameters
    ----------
    method : str
        'responsiveness', 'peak'. Method to determine responsive rois
    data : str
        None, 'trans' for normal imaging or isActive methods respectively
    sig_tail : str
        'upper', 'lower', 'both'
        Choose if you want to look for responses in the upper tail, lower
        tail or either tail of the bootstrap distribution.
    exclude : str or bool array
        nTrials x nFrames array or 'running'
        Frames to exclude from the response calculation
    shuffle_exclude : str
        Frames to exclude from the shuffles, 'running' to
        exclude running intervals, 'all' to exclude both stims and running,
        or 'exclude' to match the 'exclude' argument
    transients_conf_level : int
        95 or 99, only used if 'data' is 'trans'
    save_to_expt: bool
        If True, saves responsive rois for the given set of
        parameters for each experiment
    ignore_saved : bool
        If True, don't check for saved responsive rois
    n_processes : int
        If > 1, farms out the shuffling over multiple cores.
    return_roi_tuple_filter : bool
        The default return will filter on ROI objects present in the current
        ExperimentGroup. Alternatively, return a filter on ROI
        (mouse, location, id) tuples that will also filter ROIs not in the
        current ExperimentGroup.

    Returns
    -------
    responsive_filter: roi_filter
         An roi_filter for the responsive ROIs.

    """

    if roi_filter and n_processes > 1:
        warnings.warn('Currently unable to use an roi_filter with pools.' +
                      ' This will run with 1 core, re-run with no filter' +
                      ' to re-enable pools.')
        n_processes = 1

    STIM_INTERVAL = 10.

    if stimulus == 'air':
        stimulus = 'airpuff'

    if shuffle_exclude == 'exclude':
        shuffle_exclude = exclude

    if pre_time is None:
        pre_time = 0
        for expt in exptGrp:
            pre_time = np.amax([pre_time, expt.stimulusTime()])
    if post_time is None:
        post_time = 0
        for expt in exptGrp:
            post_time = np.amax(
                [post_time, expt.imagingTimes()[-1] - expt.stimulusTime()])

    pre_filtered_expts = []
    responsive_rois = set()
    if not ignore_saved:
        # See if all the experiments already have a saved filter
        for expt in exptGrp:
            frame_period = expt.frame_period()
            pre_frames = int(pre_time / frame_period)
            post_frames = int(post_time / frame_period)
            trials = []
            for ee in exptGrp:
                if ee.parent.get('mouseID') == expt.parent.get('mouseID') \
                        and ee.get('uniqueLocationKey') == \
                        expt.get('uniqueLocationKey'):
                    trials.extend(ee.findall('trial'))
            try:
                expt_filter = filters.responsive_roi_filter(
                    expt, stimulus, trials, method=method,
                    pre_frames=pre_frames, post_frames=post_frames,
                    channel=channel, label=label, roi_filter=roi_filter,
                    data=data, conf_level=conf_level, sig_tail=sig_tail,
                    transients_conf_level=transients_conf_level,
                    exclude=exclude, shuffle_exclude=shuffle_exclude,
                    n_bootstraps=n_bootstraps, dFOverF=dFOverF)
            except ValueError:
                break
            else:
                pre_filtered_expts.append(expt)
                responsive_rois = responsive_rois.union(expt.rois(
                    channel=channel, label=label, roi_filter=expt_filter))

    # Find experiments that still need to be processed
    not_done_expts = set(exptGrp).difference(pre_filtered_expts)
    fields_to_run = set([expt.field_tuple() for expt in not_done_expts])
    expts_to_run = [expt for expt in exptGrp
                    if expt.field_tuple() in fields_to_run]
    new_grp = exptGrp.subGroup(expts_to_run)

    if isinstance(stimulus, basestring):
        stimulus_arg = {
            expt.totuple(): ba.stimStarts(expt, stimulus) for expt in new_grp}
    else:
        stimulus_arg = stimulus

    if data is None:
        data_arg = {expt.totuple(): expt.imagingData(
            dFOverF=dFOverF, channel=channel, label=label,
            roi_filter=roi_filter) for expt in new_grp}
    elif data == 'trans':
        data_arg = {expt.totuple(): isActive(
            expt, conf_level=transients_conf_level, channel=channel,
            label=label, roi_filter=roi_filter) for expt in new_grp}
    else:
        data_arg = data

    if shuffle_exclude == 'running':
        shuffle_exclude_arg = {}
        for expt in new_grp:
            nROIs = expt.imaging_shape(
                channel=channel, label=label, roi_filter=roi_filter)[0]
            expt_exclude = expt.runningIntervals(
                imageSync=True, returnBoolList=True, end_padding=2.0)
            expt_exclude = np.rollaxis(
                np.tile(expt_exclude, (nROIs, 1, 1)), 1, 0)
            shuffle_exclude_arg[expt.totuple()] = expt_exclude
    elif shuffle_exclude == 'all':
        shuffle_exclude_arg = {}
        for expt in new_grp:
            nROIs, nFrames, _ = expt.imaging_shape(
                channel=channel, label=label, roi_filter=roi_filter)
            expt_exclude = expt.runningIntervals(
                imageSync=True, returnBoolList=True, end_padding=2.0)
            expt_exclude = np.rollaxis(
                np.tile(expt_exclude, (nROIs, 1, 1)), 1, 0)
            frame_period = expt.frame_period()
            stim_starts = ba.stimStarts(expt, 'all', imageSync=True)
            for trial_idx, trial in enumerate(stim_starts):
                trial_mask = np.zeros(nFrames, dtype='bool')
                for stim in trial:
                    trial_mask[stim:
                               stim + STIM_INTERVAL / frame_period] = True
                expt_exclude[trial_idx, ...] = np.logical_or(
                    expt_exclude[trial_idx, ...], trial_mask)
            shuffle_exclude_arg[expt.totuple()] = expt_exclude
    else:
        shuffle_exclude_arg = shuffle_exclude

    inputs = (stimulus_arg, method, pre_time, post_time,
              shuffle_exclude_arg, data_arg, channel, label, roi_filter)

    global expt_grp
    expt_grp = new_grp

    # Run through one shuffle to pre-load a few things
    _expt_grp_response_magnitudes_shuffler(inputs)

    if n_processes > 1:

        def init_grp(grp):
            global expt_grp
            expt_grp = grp

        from multiprocessing import Pool
        pool = Pool(
            processes=n_processes, initializer=init_grp, initargs=[new_grp])

        # import multiprocessing.util as util
        # util.log_to_stderr(util.SUBDEBUG)

        bootstrap_results = pool.map(
            _expt_grp_response_magnitudes_shuffler,
            it.repeat(inputs, n_bootstraps))

        pool.close()
        pool.join()
    else:

        bootstrap_results = map(
            _expt_grp_response_magnitudes_shuffler,
            it.repeat(inputs, n_bootstraps))

    bootstrap_results = np.vstack(bootstrap_results)

    trueDiff, _, _, true_rois, _ = response_magnitudes(
        new_grp, stimulus, method=method, pre_time=pre_time,
        post_time=post_time, data=data_arg, exclude=exclude, channel=channel,
        label=label, roi_filter=roi_filter, return_full=True)

    if sig_tail == 'upper':
        upper_threshold = np.percentile(
            bootstrap_results, conf_level, axis=0)
        responsiveCells = trueDiff > upper_threshold
    elif sig_tail == 'lower':
        lower_threshold = np.percentile(
            bootstrap_results, 100 - conf_level, axis=0)
        responsiveCells = trueDiff < lower_threshold
    elif sig_tail == 'both':
        half_conf_level = (100 - conf_level) / 2.
        upper_threshold = np.percentile(
            bootstrap_results, 100 - half_conf_level, axis=0)
        lower_threshold = np.percentile(
            bootstrap_results, half_conf_level, axis=0)
        responsiveCells = np.bitwise_or(
            trueDiff > upper_threshold, trueDiff < lower_threshold)
    else:
        raise ValueError('Unrecognized sig_tail value')

    for roi_mouse, roi_location, roi_id in np.array(
            true_rois)[responsiveCells]:
        for expt in roi_mouse.findall('experiment'):
            if expt in new_grp \
                    and expt.get('uniqueLocationKey') == roi_location:
                responsive_rois = responsive_rois.union(expt.rois(
                    channel=channel, label=label,
                    roi_filter=lambda x: x.id == roi_id))

    if return_roi_tuple_filter:
        responsive_roi_tuples = set(
            [(roi.expt.parent.get('mouseID'),
             roi.expt.get('uniqueLocationKey'),
             roi.id) for roi in responsive_rois])

        def responsive_roi_tuple_filter(roi):
            return (roi.expt.parent.get('mouseID'),
                    roi.expt.get('uniqueLocationKey'),
                    roi.id) in responsive_roi_tuples

    else:
        def responsive_filter(roi):
            return roi in responsive_rois

    if roi_filter is None and save_to_expt:
        for expt in new_grp:
            trials = []
            for ee in new_grp:
                if ee.parent.get('mouseID') == expt.parent.get('mouseID') \
                        and ee.get('uniqueLocationKey') == \
                        expt.get('uniqueLocationKey'):
                    trials.extend([trial.get('time') for trial
                                   in ee.findall('trial')])
            responsive_rois_path = os.path.join(
                expt.sima_path(), 'responsive_rois.pkl')
            frame_period = expt.frame_period()
            pre_frames = int(pre_time / frame_period)
            post_frames = int(post_time / frame_period)
            if label is None:
                expt_label = expt.most_recent_key(channel=channel)
            else:
                expt_label = label
            key_tuple = (''.join(sorted(trials)), method, pre_frames,
                         post_frames, channel, expt_label, data,
                         conf_level, sig_tail, transients_conf_level,
                         exclude, shuffle_exclude, n_bootstraps, dFOverF)
            roi_ids = expt.roi_ids(
                channel=channel, label=label, roi_filter=responsive_filter)
            try:
                with open(responsive_rois_path, 'r') as f:
                    responsive_dict = pickle.load(f)
            except (IOError, pickle.UnpicklingError):
                responsive_dict = {}

            if stimulus not in responsive_dict:
                responsive_dict[stimulus] = {}

            responsive_dict[stimulus][key_tuple] = {}
            responsive_dict[stimulus][key_tuple]['roi_ids'] = roi_ids
            responsive_dict[stimulus][key_tuple]['timestamp'] = timestamp()

            with open(responsive_rois_path, 'w') as f:
                pickle.dump(responsive_dict, f, pickle.HIGHEST_PROTOCOL)

    if ax:
        PSTH(exptGrp, stimulus, ax=ax, pre_time=pre_time, post_time=post_time,
             exclude=exclude, data=data, shade_ste=shade_ste,
             plot_mean=plot_mean, channel=channel, label=label,
             roi_filter=responsive_filter)

    if return_roi_tuple_filter:
        return responsive_roi_tuple_filter

    return responsive_filter

expt_grp = None


# TODO: REMOVE THIS?  IT'S NEVER USED IN THE REPO
def identify_stim_responsive_trials(
        exptGrp, stimulus, pre_time=None, post_time=None, conf_level=95,
        data=None, sig_tail='upper', exclude=None, channel='Ch2',
        shuffle_exclude='exclude', label=None, roi_filter=None,
        n_bootstraps=10000, transients_conf_level=99, dFOverF='from_file',
        save_to_expt=True):
    """Identifies trials that are significantly responsive"""

    if stimulus == 'air':
        stimulus = 'airpuff'

    if stimulus == 'running_stop_off':
        sig_tail = 'lower'

    if isinstance(stimulus, basestring):
        stimulus_arg = {
            expt: ba.stimStarts(expt, stimulus) for expt in exptGrp}
    else:
        stimulus_arg = stimulus

    if pre_time is None:
        pre_time = 0
        for expt in exptGrp:
            pre_time = np.amax([pre_time, expt.stimulusTime()])
    if post_time is None:
        post_time = 0
        for expt in exptGrp:
            post_time = np.amax(
                [post_time, expt.imagingTimes()[-1] - expt.stimulusTime()])

    if data is None:
        data_arg = {expt.totuple(): expt.imagingData(
            dFOverF=dFOverF, channel=channel, label=label,
            roi_filter=roi_filter) for expt in exptGrp}
    elif data == 'trans':
        data_arg = {expt.totuple(): isActive(
            expt, conf_level=transients_conf_level, channel=channel,
            label=label, roi_filter=roi_filter) for expt in exptGrp}
    else:
        data_arg = data

    if exclude == 'running':
        exclude_arg = {}
        for expt in exptGrp:
            nROIs = expt.imaging_shape(
                channel=channel, label=label, roi_filter=roi_filter)[0]
            expt_exclude = expt.runningIntervals(
                imageSync=True, returnBoolList=True, end_padding=2.0)
            expt_exclude = np.rollaxis(
                np.tile(expt_exclude, (nROIs, 1, 1)), 1, 0)
            exclude_arg[expt.totuple()] = expt_exclude
    elif exclude == 'stim':
        exclude_arg = {}
        for expt in exptGrp:
            pass
    else:
        exclude_arg = exclude

    max_frames = max([expt.imaging_shape()[1] for expt in exptGrp])

    for bootstrap_idx in xrange(max_frames):
        bootstrap_stim_arg = {}
        for expt in exptGrp:
            expt_frames = expt.imaging_shape()[1]
            bootstrap_stim_arg[expt.totuple()] = [
                np.array([bootstrap_idx]) if len(trial_stim)
                and bootstrap_idx < expt_frames
                else np.array([]) for trial_stim
                in stimulus_arg[expt.totuple()]]

        psths, _, x_ranges = PSTH(
            exptGrp, bootstrap_stim_arg, pre_time=pre_time,
            post_time=post_time, exclude=exclude_arg, data=data_arg,
            channel=channel, label=label, roi_filter=roi_filter,
            return_full=True, shuffle=False)

        # Initialize output array
        if bootstrap_idx == 0:
            bootstrap_results = [[] for _ in x_ranges]

        for psth_idx, psth, x_range in it.izip(
                it.count(), psths, x_ranges):
            responses = nanmean(psth[x_range > 0, :], axis=0) - \
                nanmean(psth[x_range < 0, :], axis=0)
            responses = [response for response in responses
                         if np.isfinite(response)]
            bootstrap_results[psth_idx].extend(responses)

    true_psths, true_rois, true_x_ranges = PSTH(
        exptGrp, stimulus, pre_time=pre_time, post_time=post_time,
        data=data_arg, exclude=exclude, channel=channel, label=label,
        roi_filter=roi_filter, return_full=True)

    trueDiffs = []
    for psth, x_range in it.izip(true_psths, true_x_ranges):
        trueDiffs.append(
            nanmean(psth[x_range > 0, :], axis=0) -
            nanmean(psth[x_range < 0, :], axis=0))

    responsive_trials = []
    if sig_tail == 'upper':
        for responses, roi_bootstrap_results in it.izip(
                trueDiffs, bootstrap_results):
            upper_threshold = np.percentile(
                roi_bootstrap_results, conf_level)
            responsive_trials.append(responses > upper_threshold)
    elif sig_tail == 'lower':
        for responses, roi_bootstrap_results in it.izip(
                trueDiffs, bootstrap_results):
            lower_threshold = np.percentile(
                roi_bootstrap_results, 100 - conf_level)
            responsive_trials.append(responses < lower_threshold)
    elif sig_tail == 'both':
        half_conf_level = (100 - conf_level) / 2.
        for responses, roi_bootstrap_results in it.izip(
                trueDiffs, bootstrap_results):
            upper_threshold = np.percentile(
                roi_bootstrap_results, 100 - half_conf_level)
            lower_threshold = np.percentile(
                roi_bootstrap_results, half_conf_level)
            responsive_trials.append(np.bitwise_or(
                responses > upper_threshold, responses < lower_threshold))
    else:
        raise ValueError('Unrecognized sig_tail value')

    return responsive_trials


def compare_run_response_by_running_duration(
        exptGrp, ax, run_intervals='running', response_method='responsiveness',
        plot_method='scatter', pre_time=None, post_time=None,
        channel='Ch2', label=None, roi_filter=None,
        responsive_method=None, **psth_kwargs):
    """Compare the running response of ROIs by scattering single trial
    responses against running interval duration

    run_intervals -- any argument to stimStarts that returns a subset
        of running intervals
    reponse_method -- how to calculate the running response on a per-trial
        basis. 'responsiveness' is the same metric used to determine
        responsive rois, 'mean' is the mean during the running bout
    pre/post_time -- time to include in psth before and after stim
    responsive_method -- None, to include all cells, or a method used to
        identify stim responsive cells
    psth_kwargs -- any other arguments will be passed to expt.psth

    """

    if pre_time is None:
        pre_time = 0
        for expt in exptGrp:
            pre_time = max([pre_time, expt.stimulusTime()])
    if post_time is None:
        post_time = 0
        for expt in exptGrp:
            post_time = max(
                [post_time, expt.imagingTimes()[-1] - expt.stimulusTime()])

    if responsive_method:
        roi_filter = identify_stim_responsive_cells(
            exptGrp, stimulus=run_intervals, method=responsive_method,
            pre_time=pre_time, post_time=post_time, channel=channel,
            label=label, roi_filter=roi_filter, **psth_kwargs)

    durations = []
    responses = []
    for expt in exptGrp:

        if response_method == 'responsiveness':
            frame_period = expt.frame_period()
            pre_frames = int(pre_time / frame_period)
            post_frames = int(post_time / frame_period)
            # psths is nROIs x nFrames x nIntervals
            psths = _psth(
                expt, run_intervals, pre_frames=pre_frames,
                post_frames=post_frames, return_full=True, channel=channel,
                label=label, roi_filter=roi_filter, **psth_kwargs)
            post = nanmean(psths[:, -post_frames:, :], axis=1)
            pre = nanmean(psths[:, :pre_frames, :], axis=1)
            # responses is now nROIs x nIntervals
            expt_responses = post - pre

            # Need to determine which running intervals were included in
            # psths, match filtered stim starts with all running intervals
            starts = ba.stimStarts(expt, run_intervals, imageSync=True)
            starts = list(it.chain(*starts))
            running_intervals = expt.runningIntervals(imageSync=True)
            response_idx = 0
            for interval in it.chain(*running_intervals):
                if response_idx >= expt_responses.shape[1]:
                    break
                if 'stop' in run_intervals:
                    if interval[1] != starts[response_idx]:
                        continue
                else:
                    if interval[0] != starts[response_idx]:
                        continue
                interval_responses = expt_responses[:, response_idx]
                interval_duration = \
                    (interval[1] - interval[0] + 1) * frame_period
                responses.extend(interval_responses.tolist())
                durations.extend(
                    [interval_duration] * len(interval_responses))
                response_idx += 1

        elif response_method == 'mean':
            frame_period = expt.frame_period()
            imaging_data = expt.imagingData(
                channel=channel, label=label, roi_filter=roi_filter,
                dFOverF='from_file')
            starts = ba.stimStarts(expt, run_intervals, imageSync=True)
            running_intervals = expt.runningIntervals(imageSync=True)
            for trial_idx, trial_starts, trial_intervals in it.izip(
                    it.count(), starts, running_intervals):
                for start in trial_starts:
                    if 'stop' in run_intervals:
                        if start not in trial_intervals[:, 1]:
                            continue
                        interval_idx = np.nonzero(
                            trial_intervals == start)[0][0]
                        run_start = trial_intervals[interval_idx, 0]
                        run_stop = start + 1
                    else:
                        if start not in trial_intervals[:, 0]:
                            continue
                        interval_idx = np.nonzero(
                            trial_intervals == start)[0][0]
                        run_start = start
                        run_stop = trial_intervals[interval_idx, 1] + 1
                    running_imaging = imaging_data[:, run_start:run_stop,
                                                   trial_idx]
                    mean_responses = nanmean(
                        running_imaging, axis=1).tolist()
                    interval_duration = \
                        (run_stop - run_start) * frame_period
                    responses.extend(mean_responses)
                    durations.extend(
                        [interval_duration] * len(mean_responses))

        else:
            raise NotImplementedError

    if not len(durations):
        warnings.warn('No running intervals found')
        return

    if plot_method == 'scatter':
        plotting.scatterPlot(
            ax, np.vstack([durations, responses]),
            ['Running duration (s)', response_method],
            plotEqualLine=False, print_stats=True, s=1)
        ax.set_title(run_intervals)
    else:
        raise NotImplementedError


def plot_number_of_stims_responsive(
        exptGrp, ax, stimuli, method='responsiveness', pre_time=None,
        post_time=None, exclude=None, channel='Ch2', label=None,
        roi_filter=None, plot_mean=True, n_processes=1,
        n_bootstraps=10000):
    """Plot a histogram of the number of stims each ROI is responsive to
    and compare to the distribution if all ROIs were equally likely to
    respond to each stim and all the stims were independent.

    """

    # Only include ROIs that were actually exposed to all stims
    stims_to_check = set(filter(
        lambda stim: 'running' not in stim and 'licking' not in stim,
        stimuli))

    responsive_cells = {}
    for stim in stimuli:
        responsive_cells[stim] = identify_stim_responsive_cells(
            exptGrp, stimulus=stim, method=method, pre_time=pre_time,
            post_time=post_time, sig_tail='upper', exclude=exclude,
            channel=channel, label=label, roi_filter=roi_filter,
            n_bootstraps=n_bootstraps, save_to_expt=True,
            n_processes=n_processes)

    all_rois = exptGrp.allROIs(
        channel=channel, label=label, roi_filter=roi_filter)
    roi_counts = []

    stim_counts = {stimulus: 0 for stimulus in stimuli}
    for roi in all_rois:
        # Skip ROIs that were not exposed to all stims
        roi_stims = set(it.chain(
            *[expt.stimuli() for expt, _ in all_rois[roi]]))
        if len(stims_to_check.difference(roi_stims)) > 0:
            continue
        roi_id = roi[2]
        roi_expt = all_rois[roi][0][0]
        first_roi = roi_expt.rois(
            channel=channel, label=label,
            roi_filter=lambda r: r.id == roi_id)[0]
        roi_count = np.sum([responsive_cells[stimulus](first_roi)
                            for stimulus in stimuli])
        roi_counts.append(roi_count)
        for stimulus in stimuli:
            stim_counts[stimulus] += int(
                responsive_cells[stimulus](first_roi))

    counts, edges, _ = plotting.histogram(
        ax, roi_counts, range(len(stimuli) + 2), color='m',
        plot_mean=plot_mean)

    probs = np.array([stim_counts[stimulus] for stimulus in stim_counts]) \
        / float(len(roi_counts))
    dist = stats.poisson_binomial_distribution(probs)
    dist_counts = np.array([val * sum(counts) for val in dist])
    ax.step(range(len(dist_counts)), dist_counts, where='post', color='c')

    # Not sure that ddof should be 0, but 0 is the most conservative
    _, p_val = chisquare(counts[dist_counts > 0],
                         dist_counts[dist_counts > 0], ddof=0)

    ax.set_xticks((edges[:-1] + edges[1:]) / 2.0)
    ax.set_xticklabels(range(0, len(stimuli) + 1))
    ax.set_xlabel('Number of stims responsive')
    ax.set_ylabel('Number of ROIs')
    plotting.stackedText(
        ax, ['actual', 'expected', 'p={:.5f}'.format(p_val)],
        colors=['m', 'c', 'k'])

    return counts, dist_counts


@memoize
def roi_area(expt_grp, channel='Ch2', label=None, roi_filter=None):
    """Calculate the area of each ROI (in um^2).

    Parameters
    ----------
    channel : string
    label : string or None
    roi_filter : filter_function or None

    Returns
    -------
    pd.DataFrame

    """
    rois = expt_grp.rois(channel=channel, label=label, roi_filter=roi_filter)

    data_list = []
    for expt in expt_grp:
        params = expt.imagingParameters()
        x_um = params['micronsPerPixel']['XAxis']
        y_um = params['micronsPerPixel']['YAxis']
        for roi in rois[expt]:
            new_polys = []
            for poly in roi.polygons:
                coords = np.array(poly.exterior)
                coords[:, 0] *= x_um
                coords[:, 1] *= y_um
                new_polys.append(shapely.geometry.Polygon(coords))
            tmp_multi_poly = shapely.geometry.MultiPolygon(new_polys)
            data_list.append(
                {'expt': expt, 'roi': roi, 'value': tmp_multi_poly.area})

    return pd.DataFrame(data_list, columns=['expt', 'roi', 'value'])


@memoize
def transients(
        expt_grp, key=None, interval=None, invert_interval=False, **transient_kwargs):
    """Return a DataFrame of all transients.

    Parameters
    ----------
    key : None or str
        If not None, drop all columns other than trial, roi, and 'key'.
        Also renames column 'key' to 'value'.
    interval : lab.classes.new_interval.Interval
        An interval object used to filter out events. Filters on the
        'start_frame' of the transient.
    **transient_kwargs : dict
        All other keyword arguments are passed to the per-experiment
        transients data method.

    Returns
    -------
    pd.DataFrame

    """
    trans_list = [expt.transientsData(dataframe=True, **transient_kwargs)
                  for expt in expt_grp]

    trans = pd.concat(trans_list, ignore_index=True)

    if interval is not None:
        int_sec = interval.resample()
        trans['_start_time'] = trans[['trial', 'start_frame']].apply(
            lambda inputs: inputs[0].parent.frame_period() * inputs[1],
            axis=1)
        trans = int_sec.filter_events(trans, key='_start_time', invert=invert_interval)

        del trans['_start_time']

    if key is not None:
        trans = trans[['trial', 'roi', key]]
        trans.rename(columns={key: 'value'}, inplace=True)

    return trans
