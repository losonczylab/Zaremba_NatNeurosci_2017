"""Analysis-specific plotting methods"""

import warnings
import numpy as np
import scipy as sp
import itertools as it
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import datetime

import lab
from ..classes.classes import ExperimentGroup
import plotting as plotting
import plotting_helpers as plotting_helpers
from lab.misc import signalsmooth
from ..analysis import behavior_analysis as ba
from ..analysis import place_cell_analysis as place
from ..analysis import imaging_analysis as ia
from ..analysis import calc_activity as calc_activity

from ..classes import exceptions as exc


def activityPlot(
        trial, ax, dFOverF='median', demixed=False,
        yOffsets=None, linearTransform=None, window_width=100,
        dFOverF_percentile=8, timeInterval=None, removeNanBoutons=False,
        colorbarAx=None, smoothSize=0, resampling=None, style='color',
        colorcode=None, markerDuration=5, colorRange=[-0.2, 1],
        label_x_axis=False, channel='Ch2', label=None, roi_filter=None):
    """Plot the activity of all boutons at each time as a heatmap"""
    times = trial.parent.imagingTimes()
    imData = trial.parent.imagingData(
        dFOverF=dFOverF, demixed=demixed,
        linearTransform=linearTransform, window_width=window_width,
        dFOverF_percentile=dFOverF_percentile,
        removeNanBoutons=removeNanBoutons, channel=channel, label=label,
        roi_filter=roi_filter)[:, :, trial.trialNum()]
    if timeInterval is not None:
        imData = imData[:, trial.parent.imagingIndex(
            timeInterval[0]):trial.parent.imagingIndex(timeInterval[1])]
        times = np.array(times)[trial.parent.imagingIndex(timeInterval[0]):
                                trial.parent.imagingIndex(timeInterval[1])]
    if smoothSize:
        for roiIdx in range(imData.shape[0]):
            imData[roiIdx] = signalsmooth.smooth(
                imData[roiIdx], window_len=smoothSize, window='hanning')
            # imData = imData[:,int(smoothSize/2):-int(smoothSize/2)]
            # times = times[:-(2*int(smoothSize/2))]
    if resampling is not None:
        imData = sp.signal.decimate(imData, resampling, axis=1)
        times = times[::resampling]

    if style == 'color':
        roiNums = np.arange(0, imData.shape[0] + 1) + 0.5
        TIMES, ROI_NUMS = np.meshgrid(times, roiNums)
        im = ax.pcolor(TIMES, ROI_NUMS, imData, vmin=colorRange[0],
                       vmax=colorRange[1], rasterized=True)
        if colorbarAx is not None:
            ticks = colorRange
            if 0 > ticks[0] and 0 < ticks[1]:
                ticks.append(0)
            if not colorbarAx == ax:
                cbar = colorbarAx.figure.colorbar(
                    im, ax=colorbarAx, ticks=ticks, fraction=1)
            else:
                cbar = colorbarAx.figure.colorbar(
                    im, ax=colorbarAx, ticks=ticks)
            cbar.set_label(r'$\Delta$F/F', labelpad=-10)

        """ Label the ROIs """
        ROIs = [roi.id for roi in trial.rois(channel=channel, label=label)
                if roi_filter(roi)]
        try:
            roiGroups, roiGroupNames = bouton.BoutonSet(ROIs).boutonGroups()
        except:
            ax.set_yticks(range(len(ROIs)))
            ax.set_yticklabels(ROIs)
        else:
            if colorcode == 'postSynaptic':
                for k, group in enumerate(roiGroups):
                    for roi in group:
                        #    if roiGroupNames[k] != 'other':
                        ax.add_patch(plt.Rectangle(
                            (-2, ROIs.index(roi.name) + 0.5), 1, 1,
                            color=bouton.groupPointStyle(roiGroupNames[k])[0],
                            lw=0))

        """ Plot the behavior data beneath the plot """
        framePeriod = trial.parent.frame_period()
        for interval in ba.runningIntervals(trial) * framePeriod:
            ax.add_patch(plt.Rectangle(
                (interval[0], -1), interval[1] - interval[0], 1.3,
                color='g', lw=0))
        height = -1
        for key, color in [('air', 'r'), ('airpuff', 'r'),
                           ('licking', 'b'), ('odorA', 'c'),
                           ('odorB', 'm')]:
            try:
                intervals = trial.behaviorData()[key]
            except KeyError:
                pass
            else:
                height -= 1
                for interval in intervals:
                    ax.add_patch(Rectangle(
                        (interval[0], height), interval[1] - interval[0],
                        1, facecolor=color, lw=0))

        ax.set_xlim([-2, times[-1]])
        ax.spines['left'].set_bounds(1, len(roiNums) - 1)
        ax.spines['left'].set_position(('outward', 2))
        for side in ['right', 'top', 'bottom']:
            ax.spines[side].set_color('none')
        ax.set_yticks([1, len(roiNums) - 1])
        if label_x_axis:
            ax.set_xlabel('time (s)')
        else:
            ax.set_xticks([])
        ax.set_ylabel('ROI #', labelpad=-9)
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', direction='out')
        ax.set_ylim([height, len(roiNums) - 0.5])
    elif style == 'traces':
        data = [imData.reshape([imData.shape[0], imData.shape[1], 1])]
        plotting.tracePlot(
            ax, data, times, ROIs, stimulusDurations=None, shading=None,
            yOffsets=yOffsets, markerDuration=markerDuration)
        framePeriod = trial.parent.frame_period()
        yMin = ax.get_ylim()[0]
        for interval in ba.runningIntervals(trial) * framePeriod:
            ax.add_patch(plt.Rectangle(
                (interval[0], yMin - 1), interval[1] - interval[0], 1,
                color='g', lw=0))
        # ax.set_xlim([-2, times[-1]])
        ax.set_ylim(bottom=yMin - 1)


# ADDED SUPPORT FOR LASER PLOT
def behaviorPlot(
        trial, ax, keys=['velocity', 'running', 'position', 'licking', 'tone', 'light',
                         'water', 'reward', 'airpuff', 'motion', 'laser'],
        colors=None, include_empty=False, y_start=-1):
    """Plot behavior data over time

    Keyword arguments:
    ax -- axis to plot on
    keys -- behavior data to plot, id data is missing it is skipped
    colors -- colors list to use, will be iterated over
    include_empty -- if True, plot data that has no intervals, if false,
        exclude those rows
    y_start -- bottom of first plot, decreases by one for each successive plot

    """

    try:
        bd = trial.behaviorData()
    except exc.MissingBehaviorData:
        return
    if colors is None:
        colors = lab.plotting.color_cycle()
    else:
        colors = iter(colors)
    next_y = y_start
    labels = []
    label_colors = []
    for key in keys:
        if key == 'velocity':
            try:
                velocity = ba.velocity(
                    trial, imageSync=False, sampling_interval='actual',
                    smoothing='hanning', window_length=71)
                bd['recordingDuration']
            except (exc.MissingBehaviorData, KeyError):
                bd['recordingDuration']
            except (exc.MissingBehaviorData, KeyError):
                pass
            else:
                labels.append('velocity')
                next_color = colors.next()
                label_colors.append(next_color)
                velocity -= np.amin(velocity)
                velocity /= np.amax(velocity) / 0.9
                velocity += next_y + 0.05
                ax.plot(np.linspace(0, bd['recordingDuration'],
                                    len(velocity)), velocity,
                        color=next_color)
                next_y -= 1
        elif key == 'position':
            try:
                position = bd['treadmillPosition']
                bd['recordingDuration']
                bd['samplingInterval']
            except KeyError:
                pass
            else:
                labels.append('position')
                next_color = colors.next()
                label_colors.append(next_color)
                full_position = np.empty(int(np.ceil(
                    bd['recordingDuration'] / bd['samplingInterval'])))
                for t, pos in position:
                    full_position[int(t / bd['samplingInterval']):] = pos
                full_position *= 0.9
                full_position += next_y + 0.05
                ax.plot(np.linspace(0, bd['recordingDuration'],
                                    len(full_position)), full_position,
                        color=next_color)
                next_y -= 1
        else:
            try:
                if key == 'running':
                    data = ba.runningIntervals(trial, imageSync=False) *\
                        bd['samplingInterval']
                else:
                    data = bd[key]
            except KeyError:
                pass
            else:
                if include_empty or len(data) > 0:
                    labels.append(key)
                    next_color = colors.next()
                    label_colors.append(next_color)
                    for interval in data:
                        ax.add_patch(Rectangle(
                            (interval[0], next_y),
                            interval[1] - interval[0], 1,
                            facecolor=next_color, lw=0))
                    next_y -= 1
    if next_y == y_start:
        return
    ax.set_yticks(np.arange(-0.5, next_y + 0.5, -1))
    ax.set_yticklabels(labels)
    for tick, c in zip(ax.get_yticklabels(), label_colors):
        tick.set_color(c)
    try:
        ax.set_xlim([0, int(bd['recordingDuration'])])
    except KeyError:
        pass
    ax.set_ylim([next_y + 1, 0])
    ax.set_xlabel('Time (s)')
    ax.set_title('{0}:{1}'.format(trial.parent.parent.get('mouseID'),
                 trial.get('time')))


def plot_imaging_and_behavior(
        trial, ax, start_time=0, stop_time=None, channel='Ch2', label=None,
        roi_filter=None, label_rois=False,
        keys=['running', 'licking', 'water', 'airpuff', 'tone', 'light'],
        colors=None, include_empty=False, dFOverF='from_file'):
    """Plot imaging data for all ROIs with behavior data underneath"""

    imaging_data = trial.parent.imagingData(
        channel=channel, label=label, roi_filter=roi_filter,
        dFOverF=dFOverF)[..., trial.parent.findall('trial').index(trial)]
    if not imaging_data.shape[0]:
        return

    frame_period = trial.parent.frame_period()
    start_frame = int(start_time / frame_period)
    if stop_time is None:
        stop_frame = imaging_data.shape[1]
    else:
        stop_frame = int(stop_time / frame_period)

    if stop_time is None:
        stop_time = trial.parent.imagingTimes(channel=channel)[-1]

    imaging_data = imaging_data[:, start_frame:stop_frame]
    t_range = np.linspace(start_time, stop_time, imaging_data.shape[1])

    max_F = np.nanmax(imaging_data)

    # Normalize and re-scale so they can all be plotted on top of eachother
    imaging_data /= max_F
    imaging_data += np.arange(imaging_data.shape[0]).reshape((-1, 1)) + 0.5

    ax.plot(t_range, imaging_data.T)

    behaviorPlot(
        trial, ax, keys=keys, colors=colors, include_empty=include_empty)

    if label_rois:
        roi_ids = trial.parent.roi_ids(
            channel=channel, label=label, roi_filter=roi_filter)
        x_range = ax.get_xlim()[1]
        for idx, roi_id in enumerate(roi_ids):
            ax.text(x_range * -0.01, idx + 0.5, roi_id, ha='right')

    ax.set_ylim(top=imaging_data.shape[0] + 0.5)

    plotting_helpers.add_scalebar(
        ax, matchx=False, matchy=False, hidex=False, hidey=False,
        sizey=0.5 / max_F, labely='0.5', bar_thickness=0, loc=1,
        borderpad=0.5)


def responsePairPlot(exptGrp, ax, stim1, stim2, stimuliLabels=None,
                     excludeRunning=True, boutonGroupLabeling=False,
                     linearTransform=None, axesCenter=True, channel='Ch2',
                     label=None, roi_filter=None):
    if not isinstance(stim1, list):
        stim1 = [stim1]
    if not isinstance(stim2, list):
        stim2 = [stim2]
    if stimuliLabels is None:
        stimuliLabels = [stim1[0], stim2[0]]
    ROIs = exptGrp.sharedROIs(
        roiType='GABAergic', channel=channel, label=label,
        roi_filter=roi_filter)
    shared_filter = lambda x: x.id in ROIs
    rIntegrals = []
    for stim in [stim1, stim2]:
        if stim == ['running']:
            rIntegrals.append(ia.runningModulation(
                exptGrp, linearTransform=linearTransform, channel=channel,
                label=label, roi_filter=shared_filter).reshape([-1, 1]))
        elif stim == ['licking']:
            rIntegrals.append(ia.lickingModulation(
                exptGrp, linearTransform=linearTransform, channel=channel,
                label=label, roi_filter=shared_filter).reshape([-1, 1]))
        else:
            rIntegrals.append(ia.responseIntegrals(
                exptGrp, stim, excludeRunning=excludeRunning,
                sharedBaseline=True, linearTransform=linearTransform,
                dFOverF='mean', channel=channel, label=label,
                roi_filter=shared_filter))
    if not boutonGroupLabeling:
        ROIs = None
    plotting.ellipsePlot(
        ax, rIntegrals[0].mean(axis=1), rIntegrals[1].mean(axis=1),
        2 * np.sqrt(rIntegrals[0].var(axis=1) / rIntegrals[0].shape[1]),
        2 * np.sqrt(rIntegrals[1].var(axis=1) / rIntegrals[1].shape[1]),
        boutonGroupLabeling=ROIs, color='k', axesCenter=axesCenter)
    ax.set_xlabel(stimuliLabels[0], labelpad=1)
    ax.set_ylabel(stimuliLabels[1], labelpad=1)


# TODO: LOOKS BROKEN IF YOU PASS IN AX
def plotLickRate(exptGrp, ax=None, minTrialDuration=0):
    """Generate a figure showing the lick rate for each trial in this
        ExperimentGroup.

    Keyword arguments:
    ax -- axis to plot on, created if 'None'
    minTrialLength -- minimum length of trial (in seconds) to be included in
        analysis

    """
    lickrates = []
    dates = []
    for experiment in exptGrp:
        for trial in experiment.findall('trial'):
            try:
                bd = trial.behaviorData()
                if 'licking' in bd.keys() and \
                        'recordingDuration' in bd.keys() and \
                        bd['recordingDuration'] >= minTrialDuration:
                    lickrates.append(bd['licking'].shape[0] /
                                     bd['recordingDuration'])
                    dates.append(trial.attrib['time'])
            except exc.MissingBehaviorData:
                pass

    if len(lickrates) > 0:
        if ax is None:
            fig = plt.figure(figsize=(11, 8))
            ax = fig.add_subplot(111)
        ax.bar(np.arange(len(lickrates)), lickrates, 0.5)
        ax.set_ylabel('Lick rate (Hz)')
        ax.set_title('lick rate per trial')
        ax.set_xticks(np.arange(len(lickrates)) + 0.25)
        ax.set_xticklabels(
            dates, ha='right', rotation_mode='anchor', rotation=30)

        return fig


# SAME, LOOKS BROKEN IF YOU PASS IN AX
def plotLapRate(exptGrp, ax=None, minTrialDuration=0):
    """Generates a figure showing the number of laps completed per minute.

    Keyword arguments:
    ax -- axis to plot on, created if 'None'
    minTrialLength -- minimum length of trial (in seconds) to be included
        in analysis

    """

    laprates = []
    dates = []
    for experiment in exptGrp:
        for trial in experiment.findall('trial'):
            try:
                bd = trial.behaviorData()
                if 'lapCounter' in bd.keys() and \
                        len(bd['lapCounter']) > 0 and \
                        'recordingDuration' in bd.keys() and \
                        bd['recordingDuration'] >= minTrialDuration:
                    laprates.append(sum(bd['lapCounter'][:, 1] == 1) /
                                    bd['recordingDuration'] * 60.0)
                    dates.append(trial.attrib['time'])
            except exc.MissingBehaviorData:
                pass

    if len(laprates) > 0:
        if ax is None:
            fig = plt.figure(figsize=(11, 8))
            ax = fig.add_subplot(111)
        ax.bar(np.arange(len(laprates)), laprates, 0.5)
        ax.set_ylabel('Lap rate (laps/minute)')
        ax.set_title('lap rate per trial')
        ax.set_xticks(np.arange(len(laprates)) + 0.25)
        ax.set_xticklabels(
            dates, ha='right', rotation_mode='anchor', rotation=15)

        return fig


def plotLapRateByDays(exptGrp, ax=None, color=None):
    """Plots lap rate by days of training"""

    if ax is None:
        ax = plt.axes()

    if color is None:
        color = lab.plotting.color_cycle().next()

    training_days = exptGrp.priorDaysOfExposure(ignoreBelt=True)

    lap_rates = {}
    for expt in exptGrp:
        for trial in expt.findall('trial'):
            try:
                bd = trial.behaviorData()
            except exc.MissingBehaviorData:
                continue
            else:
                if len(bd.get('lapCounter', [])) > 0 \
                        and 'recordingDuration' in bd:
                    if training_days[expt] not in lap_rates:
                        lap_rates[training_days[expt]] = []
                    lap_rates[training_days[expt]].append(
                        np.sum(bd['lapCounter'][:, 1] == 1) /
                        bd['recordingDuration'] * 60.0)

    if len(lap_rates) > 0:
        days = lap_rates.keys()
        days.sort()
        day_means = []
        for day in days:
            # Jitter x position
            x = (np.random.rand(len(lap_rates[day])) * 0.2) - 0.1 + day
            ax.plot(x, lap_rates[day], '.', color=color, markersize=7)
            day_means.append(np.mean(lap_rates[day]))
        ax.plot(days, day_means, '-', label=exptGrp.label(), color=color)
        ax.set_ylabel('Lap rate (laps/minute)')
        ax.set_xlabel('Days of training')
        ax.set_title('Average running by days of belt exposure')


def activityComparisonPlot(exptGrp, method, ax=None, mask1=None, mask2=None,
                           label1=None, label2=None, roiNamesToLabel=None,
                           normalize=False, rasterized=False,
                           dF='from_file', channel='Ch2', label=None,
                           roi_filter=None, demixed=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    if len(exptGrp) != 2:
        warnings.warn(
            'activityComparisonPlot requires an experimentGroup of 2 ' +
            'experiments.  Using the first two elements of {}'.format(exptGrp))
        grp = exptGrp[:2]
    else:
        grp = exptGrp

    ROIs = grp.sharedROIs(channel=channel, label=label,
                          roi_filter=roi_filter)
    shared_filter = lambda x: x.id in ROIs

    exp1ROIs = grp[0].roi_ids(channel=channel, label=label,
                              roi_filter=shared_filter)
    exp2ROIs = grp[1].roi_ids(channel=channel, label=label,
                              roi_filter=shared_filter)

    order1 = np.array([exp1ROIs.index(x) for x in ROIs])
    order2 = np.array([exp2ROIs.index(x) for x in ROIs])
    # inds of the roiNamesToLabel (in terms of exp1 indices)
    if roiNamesToLabel:
        order3 = np.array(
            [exp1ROIs.index(x) for x in roiNamesToLabel if x in ROIs])

    activity1 = calc_activity(
        grp[0], method=method, interval=mask1, dF=dF, channel=channel,
        label=label, roi_filter=roi_filter, demixed=demixed)
    activity2 = calc_activity(
        grp[1], method=method, interval=mask2, dF=dF, channel=channel,
        label=label, roi_filter=roi_filter, demixed=demixed)

    # ordering corresponds to sharedROIs() ordering
    activity1 = np.array([activity1[x] for x in order1]).flatten()
    activity2 = np.array([activity2[x] for x in order2]).flatten()

    if normalize:
        activity1 = activity1 / float(np.amax(activity1))
        activity2 = activity2 / float(np.amax(activity2))

    # -1 flips sort so it's actually high to low and also puts NaNs at the end
    order = np.argsort(-1 * activity2)

    bar_lefts = np.arange(len(ROIs))
    width = 1

    if not label1:
        label1 = grp[0].get('startTime')
    if not label2:
        label2 = grp[1].get('startTime')

    ax.bar(np.array(bar_lefts), activity1[order], width, color='b',
           alpha=0.5, label=label1, rasterized=rasterized)
    ax.bar(np.array(bar_lefts), np.negative(activity2)[order], width,
           color='r', alpha=0.5, label=label2, rasterized=rasterized)

    max_y = np.amax(np.abs(ax.get_ylim()))
    ax.set_ylim(-max_y, max_y)
    ax.set_xlim(right=len(ROIs))

    # roiIndsToIndicate = [np.argwhere(order1[order]==roi)[0][0] for roi in exp1RoisToIndicate if roi in order1[order]]
    if roiNamesToLabel:
        # ylim = ax.get_ylim()
        roiIndsToIndicate = [
            np.argwhere(order1[order] == x)[0][0] for x in order3]
        for idx in roiIndsToIndicate:
            ax.axvline(
                idx + 0.5, linestyle='dashed', color='k',
                rasterized=rasterized)
        # ax.vlines(np.array(roiIndsToIndicate)+0.5, ylim[0], ylim[1], linestyles='dashed', color='k')
        # ax.set_ylim(ylim)

    # make all y-axis labels positive
    ax.set_yticklabels(np.abs(ax.get_yticks()))

    ax.set_xlabel('ROI index')
    ax.set_ylabel('Activity = {}'.format(method))
    ax.legend()

    return fig


def activityByExposure(exptGrp, ax=None, stat='mean',
                       combineTimes=datetime.timedelta(hours=12),
                       ignoreContext=False, **kwargs):
    """Plots cdf of activity of ROIs by days of context exposure

    Keyword arguments:
    stat -- statistic to plot, see calc_activity.py for details
    combineTimes -- experiments within this timedelta of each other are
        considered the same day for determining exposure
    ignoreContext -- if True, ignores context for determining exposure
    **kwargs -- any additional arguments will be passed in to
        place.calc_activity_statistic

    """

    if ax is None:
        _, ax = plt.subplots()

    exptsByExposure = ExperimentGroup.dictByExposure(
        exptGrp, combineTimes=combineTimes, ignoreBelt=ignoreContext,
        ignoreContext=ignoreContext)
    colors = lab.plotting.color_cycle()

    for exposure in sorted(exptsByExposure):
        exgrp = ExperimentGroup(
            exptsByExposure[exposure],
            label='1 day' if exposure == 0 else str(exposure + 1) +
            ' days')
        place.calc_activity_statistic(
            exgrp, ax=ax, stat=stat, plot_method='cdf',
            label=exgrp.label(), c=colors.next(), **kwargs)

    ax.legend(loc='lower right')
    ax.set_title('{} by exposure - {}'.format(
        stat,
        'ignoring context' if ignoreContext else 'including context'))


def compare_bouton_responses(
        exptGrp, ax, stimuli, comp_method='angle', plot_method='cdf',
        channel='Ch2', label=None, roi_filter=None, **response_kwargs):
    """Compare various pairs of boutons, based on several conventions:

    'bouton' in label of bouton ROIs
    boutons targeting a cell soma are tagged with the cell number they are
        targeting, i.e. 'cell1', 'cell2', etc.
    boutons on an axons are tagged with the fiber number they are on,
        i.e. 'fiber1', 'fiber2', etc.
    boutons with no tags have no information about their axon or target

    """

    response_matrix, rois = ia.response_matrix(
        exptGrp, stimuli, channel=channel, label=label, roi_filter=roi_filter,
        return_full=True, **response_kwargs)

    data = {}
    data['mouse'] = [roi[0] for roi in rois]
    data['loc'] = [roi[1] for roi in rois]
    data['label'] = [roi[2] for roi in rois]

    tags = []
    for mouse, loc, name in it.izip(
            data['mouse'], data['loc'], data['label']):
        roi_tags = set()
        for expt in exptGrp:
            if expt.parent == mouse \
                    and expt.get('uniqueLocationKey') == loc:
                for roi in expt.rois(
                        channel=channel, label=label,
                        roi_filter=roi_filter):
                    if roi.label == name:
                        # NOTE: Taking the union of all tags,
                        #  so mis-matched tags will just be combined
                        roi_tags = roi_tags.union(roi.tags)
        tags.append(roi_tags)
    data['tags'] = tags

    data['responses'] = [response for response in response_matrix]

    df = pd.DataFrame(data)

    if comp_method == 'angle':
        ax.set_xlabel('Response similarity (angle)')

        def compare(roi1, roi2):
            return np.dot(roi1, roi2) / np.linalg.norm(roi1) \
                / np.linalg.norm(roi2)

    elif comp_method == 'abs angle':
        ax.set_xlabel('Response similarity (abs angle)')

        def compare(roi1, roi2):
            return np.abs(np.dot(roi1, roi2) / np.linalg.norm(roi1)
                          / np.linalg.norm(roi2))

    elif comp_method == 'corr':
        ax.set_xlabel('Response similarity (corr)')

        def compare(roi1, roi2):
            return np.corrcoef(roi1, roi2)[0, 1]

    elif comp_method == 'abs corr':
        ax.set_xlabel('Response similarity (abs corr)')

        def compare(roi1, roi2):
            return np.abs(np.corrcoef(roi1, roi2)[0, 1])

    elif comp_method == 'mean diff':
        ax.set_xlabel('Response similarity (mean diff)')

        def compare(roi1, roi2):
            return np.abs(roi1 - roi2).mean()

    else:
        raise ValueError('Unrecognized compare method argument')

    same_fiber = []
    fiber_with_not = []
    same_soma = []
    soma_with_not = []
    bouton_with_fiber = []
    diff_all = []
    for name, group in df.groupby(['mouse', 'loc']):
        for roi1, roi2 in it.combinations(group.iterrows(), 2):

            r1_responses = roi1[1]['responses']
            r2_responses = roi2[1]['responses']

            non_nan = np.isfinite(r1_responses) & np.isfinite(r2_responses)

            comp = compare(r1_responses[non_nan], r2_responses[non_nan])

            if np.isnan(comp):
                continue

            fiber1 = set(
                [tag for tag in roi1[1]['tags'] if 'fiber' in tag])
            fiber2 = set(
                [tag for tag in roi2[1]['tags'] if 'fiber' in tag])
            cell1 = set([tag for tag in roi1[1]['tags'] if 'cell' in tag])
            cell2 = set([tag for tag in roi2[1]['tags'] if 'cell' in tag])

            if len(fiber1.intersection(fiber2)):
                same_fiber.append(comp)
            elif len(fiber1) or len(fiber2):
                fiber_with_not.append(comp)

            if len(cell1.intersection(cell2)):
                same_soma.append(comp)
            elif len(cell1) or len(cell2):
                soma_with_not.append(comp)

            if len(fiber1) and roi2[1]['label'] in fiber1 \
                    or len(fiber2) and roi1[1]['label'] in fiber2:
                bouton_with_fiber.append(comp)
            elif not len(fiber1.intersection(fiber2)) \
                    and not len(cell1.intersection(cell2)):
                diff_all.append(comp)

    if plot_method == 'cdf':
        plotting.cdf(
            ax, same_fiber, bins='exact', label='same fiber')
        plotting.cdf(
            ax, same_soma, bins='exact', label='same soma')
        plotting.cdf(
            ax, bouton_with_fiber, bins='exact', label='bouton with fiber')
        plotting.cdf(
            ax, fiber_with_not, bins='exact', label='fiber with not')
        plotting.cdf(
            ax, soma_with_not, bins='exact', label='soma with not')
        plotting.cdf(
            ax, diff_all, bins='exact', label='diff all')

    elif plot_method == 'hist':
        colors = lab.plotting.color_cycle()
        plotting.histogram(
            ax, same_fiber, bins=50, color=colors.next(), normed=True,
            label='same fiber')
        plotting.histogram(
            ax, same_soma, bins=50, color=colors.next(), normed=True,
            label='same soma')
        plotting.histogram(
            ax, bouton_with_fiber, bins=50, color=colors.next(),
            normed=True, label='bouton with fiber')
        plotting.histogram(
            ax, fiber_with_not, bins=50, color=colors.next(), normed=True,
            label='fiber with not')
        plotting.histogram(
            ax, soma_with_not, bins=50, color=colors.next(), normed=True,
            label='soma with not')
        plotting.histogram(
            ax, diff_all, bins=50, color=colors.next(), normed=True,
            label='diff all')

    # ax.legend()

    return {'same fiber': same_fiber, 'same soma': same_soma,
            'bouton_with_fiber': bouton_with_fiber,
            'fiber_with_not': fiber_with_not,
            'soma_with_not': soma_with_not, 'diff all': diff_all}


def stim_response_heatmap(
        exptGrp, ax, stims, sort_by=None, method='responsiveness',
        z_score=True, aspect_ratio=0.25, **response_kwargs):
    """Plot a heatmap of stim responses per ROI."""

    data = ia.response_matrix(
        exptGrp, stims, method=method, z_score=z_score, **response_kwargs)

    if sort_by is not None:
        if isinstance(sort_by, list):
            # If we get a list of stims, sort by the mean of them
            indices = [stims.index(stim) for stim in sort_by]
            to_sort = data[:, indices].mean(1)
            # Remove rows that have a nan in any of the sort by cols
            non_nan_rows = np.isfinite(to_sort)
            data = data[non_nan_rows]
            order = to_sort[non_nan_rows].argsort()[::-1]
            data = data[order]
        else:
            # If we get a single stim, sort by the response to that stim
            sort_column = stims.index(sort_by)
            # Remove rows that have NaN's in the sort_by column
            non_nan_rows = np.isfinite(data[:, sort_column])
            data = data[non_nan_rows, :]
            order = data[:, sort_column].argsort()[::-1]
            data = data[order]

    ax.imshow(data, interpolation='none', aspect=aspect_ratio)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(len(stims)))
    ax.set_xticklabels(stims)
    ax.tick_params(labelbottom=False, bottom=False, left=False, top=False,
                   right=False)
