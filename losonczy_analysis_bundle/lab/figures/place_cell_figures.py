"""Figure generating functions to accompany placeCellAnalysis, used heavily by
analysis scripts
All functions should return either a figure or list of figures.

"""
import matplotlib.pyplot as plt
import numpy as np
import datetime
from copy import copy
import pandas as pd
import seaborn.apionly as sns
import itertools as it
from warnings import warn

import lab
import lab.analysis.place_cell_analysis as place
from ..analysis import behavior_analysis as ba
from ..analysis import imaging_analysis as ia
from ..analysis import reward_analysis as ra
from ..analysis import filters
from lab.analysis.place_cell_analysis import get_activity_label as gal
from lab.classes.place_cell_classes import pcExperimentGroup
from lab import plotting
from lab.plotting import plot_metric, plot_paired_metrics
from lab.figures import analysisFigures as af
import lab.misc as misc


def plotRoiAndPosition(
        exptGrp, roi_filter=None, trial=0, placeCells=True,
        shadePlaceFields=True, dFOverF='from_file', max_per_fig=None,
        rasterized=False):

    """Plot imaging data of select ROIs and position during the trial

    Keyword arguments:
    exptGrp -- a single experiment experiment group
    trial -- index of trial for which traces will be drawn
    placeCells -- plot place cells only.  If roi_filter is None, plot all place
        cells.  Else, intersect filter with place cell list
    shadePlaceFields -- if True, shades the time intervals in a place field
    dFOverF -- df/f method to use on imaging data
    max_per_fig -- maximum number of traces per page
    rasterized -- rasterize the figures

    """

    if isinstance(exptGrp, lab.Experiment):
        exptGrp = place.pcExperimentGroup([exptGrp])

    bd = exptGrp[0].findall('trial')[trial].behaviorData(imageSync=True)

    if placeCells:
        roi_filter = exptGrp.pcs_filter(roi_filter=roi_filter)
    else:
        roi_filter = roi_filter

    nROIs = exptGrp[0].imaging_shape(
        roi_filter=roi_filter, channel=exptGrp.args['channel'],
        label=exptGrp.args['imaging_label'])[0]

    if nROIs == 0:
        return []

    # sort place cells
    centroids = place.calcCentroids(
        exptGrp.data(roi_filter=roi_filter)[exptGrp[0]],
        exptGrp.pfs(roi_filter=roi_filter)[exptGrp[0]])

    centroids = [centroid[0] if centroid != [] else -1 for
                 centroid in centroids]
    nPCs = np.sum(np.array(centroids) > -1)
    # Sort by centroids, pull out the last ones (the actual PCs)
    centroids_sorted = np.argsort(centroids)[-nPCs:]
    pcs = centroids_sorted.tolist()
    nPCs = len(pcs)

    ax_to_label = []

    if max_per_fig is None:
        place_ax = []
        fig, axs = plt.subplots(nROIs + 2, 1, sharex=True, figsize=(15, 8))
        place_ax.append(plt.subplot2grid((nROIs + 2, 1), (nROIs, 0), rowspan=2,
                                         sharex=axs[0]))
        ax_to_label.append(axs[0])
        max_per_fig = nROIs

    else:
        nFigs = int(np.ceil(nROIs / float(max_per_fig)))
        fig = [[] for x in range(nFigs)]
        axs = []
        place_ax = []
        for f in range(nFigs):
            fig[f], ax = plt.subplots(max_per_fig + 2, 1, sharex=True,
                                      figsize=(15, 8))
            place_ax.append(plt.subplot2grid((max_per_fig + 2, 1),
                                             (max_per_fig, 0), rowspan=2,
                                             sharex=ax[0]))
            axs = np.hstack([axs, ax[:-2]])
            ax_to_label.append(ax[0])
        n_extras = (max_per_fig * nFigs) - nROIs
        if n_extras > 0:
            for a in ax[-n_extras - 2:]:
                a.set_visible(False)

    imagingData = exptGrp[0].imagingData(dFOverF=dFOverF,
                                         channel=exptGrp.args['channel'],
                                         label=exptGrp.args['imaging_label'],
                                         demixed=exptGrp.args['demixed'],
                                         roi_filter=roi_filter)

    if shadePlaceFields:
        # colors = [plt.cm.Set1(i) for i in np.linspace(0, .9, nPCs)]
        rel_centroids = np.array(centroids)[pcs] / \
            float(exptGrp.args['nPositionBins'])
        colors = [plt.cm.hsv(i) for i in rel_centroids]
        position = bd['treadmillPosition']

    if exptGrp[0].hasTransientsFile():
        transients = exptGrp[0].transientsData(
            threshold=95, channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'],
            demixed=exptGrp.args['demixed'], roi_filter=roi_filter)

    else:
        transients = [[None] * len(exptGrp[0].findall('trial'))] * \
            imagingData.shape[0]

    pc_idx = 0
    for roi_index, roi_id, roi_tSeries, roi_transients in it.izip(
            it.count(), exptGrp[0].roi_ids(roi_filter=roi_filter), imagingData,
            transients):

        if roi_index in pcs and shadePlaceFields:
            place.plotImagingData(
                roi_tSeries=roi_tSeries, ax=axs[roi_index],
                roi_transients=roi_transients[trial], position=position,
                placeField=exptGrp.pfs_n(
                    roi_filter=roi_filter)[exptGrp[0]][roi_index],
                imaging_interval=exptGrp[0].frame_period(),
                xlabel_visible=False, ylabel_visible=(roi_index == 0),
                right_label=True, placeFieldColor=colors[pc_idx],
                title=roi_id, rasterized=False)
            pc_idx += 1

        else:
            place.plotImagingData(
                roi_tSeries=roi_tSeries, ax=axs[roi_index],
                roi_transients=roi_transients[trial], position=None,
                imaging_interval=exptGrp[0].frame_period(),
                placeField=None, xlabel_visible=False,
                ylabel_visible=(roi_index == 0), right_label=True,
                placeFieldColor=None, title=roi_id, rasterized=False)

    if shadePlaceFields:
        # shade the place cell intervals
        intervals = exptGrp.pfs_n(roi_filter=roi_filter)[exptGrp[0]]

        for p_idx, p_ax in enumerate(place_ax):
            place.plotPosition(
                exptGrp[0].findall('trial')[trial], ax=p_ax,
                placeFields=intervals[
                    max_per_fig * p_idx:max_per_fig * (p_idx + 1)],
                placeFieldColors=colors[
                    max_per_fig * p_idx:max_per_fig * (p_idx + 1)],
                rasterized=rasterized, behaviorData=bd)
    else:
        for p_ax in place_ax:
            place.plotPosition(exptGrp[0].findall('trial')[trial], ax=p_ax,
                               rasterized=rasterized, behaviorData=bd)

    for a in ax_to_label:
        a.set_title('{0}: experiment {1}'.format(
            exptGrp[0].parent.get('mouseID'), exptGrp[0].exptNum()))

    return fig


def plotTuningCurves(
        exptGrp, roi_filter=None, rois='pcs', sort='centroid', polar=False,
        plot_err=True, plot_pf=True, subplot_dim=None, rasterized=False):
    """Generate a subplot array of tuning curves for all place cells sorted by
    centroid of first place field

    Keyword arguments:
    exptGrp -- pcExperimentGroup to analyze
    rois -- ROIs to plot. 'all' plots all ROIs, 'pcs' only plots place cells,
        list if there is only 1 experiment in exptGrp is the list of rois
        to include by number, dict can be a dictionary of lists with
        experiments in exptGrp as keys each list as in the list option
    sort -- determines if place cells should be sorted, options are: 'centroid'
        sorts by average centroid, 'shift' sorts by centroid shift,
        None for no sorting
        At the moment, 'shift' is not implemented
    polar -- if True, plots tuning curve on a polar axis
    plot_err -- if True, plots standard deviation as a shaded region around
        the tuning curve
    plot_pf -- if True, plots shades the place fields
    subplot_dim -- a tuple of the form (nRows, nCols) that determines the
        layout of the tuning curves

    """

    # TODO: Add plot showing belt coverage

    # roi_dict is the same format as the dictionary option for 'rois',
    # a dictionary of lists containing the idx of ROIs to include
    # roi_list is a list with each element a list of (expt, roi_idx) tuples
    # containing all the common ROIs across experiments for the given cell
    roi_dict = None
    roi_list = None
    if rois == 'pcs':
        roi_dict = {}
        for expt in exptGrp:
            placeFields = exptGrp.pfs(roi_filter=roi_filter)[expt]
            pcs = [x for x in range(len(placeFields)) if placeFields[x] != []]
            roi_dict[expt] = pcs
    elif rois == 'all':
        # roi_dict = {expt: range(shape(exptGrp.data(
        #     roi_filter=roi_filter)[expt])[0]) for expt in exptGrp}
        roi_dict = {expt: range(expt.imaging_shape(
            channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'],
            roi_filter=roi_filter)[0]) for expt in exptGrp}

    # See if we got a list or ROI numbers
    elif len(exptGrp) == 1:
        try:
            for idx in rois:
                exptGrp.pfs(roi_filter=roi_filter)[exptGrp[0]][idx]
        except:
            pass
        else:
            roi_dict = {exptGrp[0]: rois}
            roi_list = [(exptGrp[0], idx) for idx in rois]
    # Otherwise assume we got the dictionary format
    else:
        try:
            for expt in exptGrp:
                for idx in rois[expt]:
                    exptGrp.pfs(roi_filter=roi_filter)[expt][idx]
        except TypeError:
            pass
        else:
            roi_dict = rois

    if roi_dict is None:
        raise Exception(
            "Invalid argument, see doc for options for 'rois' keyword")
    # If roi_list isn't set yet, pull out all the desired ROIs from roi_dict
    if roi_list is None:
        roi_list = []
        all_ROIs = exptGrp.allROIs(channel=exptGrp.args['channel'],
                                   label=exptGrp.args['imaging_label'],
                                   roi_filter=roi_filter)

        for roi in all_ROIs:
            include = False
            for (expt, idx) in all_ROIs[roi]:
                if idx in roi_dict[expt]:
                    include = True
            if include:
                roi_list.append(all_ROIs[roi])

    nROIs = len(roi_list)

    if sort == 'centroid':
        # Calculate place field centroids
        centroids = {expt: place.calcCentroids(
            exptGrp.data(roi_filter=roi_filter)[expt],
            exptGrp.pfs(roi_filter=roi_filter)[expt]) for expt in exptGrp}

        # Average across observations of the ROI
        mean_centroid = []
        for roi in roi_list:
            c = [centroids[expt][idx] for (expt, idx) in roi
                 if centroids[expt][idx] != []]
            mean_centroid.append(np.mean(c) if len(c) > 0 else np.inf)
        order = np.argsort(mean_centroid)
        # TODO: HOW DID THIS WORK BEFORE???
        roi_list = [roi_list[x] for x in order]
        # roi_list = np.array(roi_list)[order]
    elif sort == 'shift':
        raise Exception('Not implemented')

    # Setup axis to plot on
    if subplot_dim is None:
        rows = 4
        cols = int(np.ceil((nROIs + 1.0) / 4.0))
    else:
        rows, cols = subplot_dim

    fig, axs, ax_to_label = plotting.layout_subplots(
        nROIs, rows, cols, polar=polar, figsize=(15, 8), rasterized=rasterized)

    if plot_pf:
        # colors = [plt.cm.Set1(i) for i in np.linspace(0, .9, nROIs)]
        rel_centroids = np.array(mean_centroid)[order] / \
            float(exptGrp.args['nPositionBins'])
        colors = [plt.cm.hsv(i) for i in rel_centroids]
    else:
        colors = [None] * nROIs

    # Plotting
    for ax, roi, color in zip(axs, roi_list, colors):
        label = ax in ax_to_label

        for expt, idx in roi:
            if plot_pf:
                placeField = exptGrp.pfs_n(roi_filter=roi_filter)[expt][idx]
            else:
                placeField = None
            if plot_err:
                error_bars = exptGrp.std(roi_filter=roi_filter)[expt]
            else:
                error_bars = None

            place.plotTuningCurve(
                exptGrp.data(roi_filter=roi_filter)[expt], idx, ax=ax,
                polar=polar, placeField=placeField, placeFieldColor=color,
                xlabel_visible=label, ylabel_visible=(label and not polar),
                error_bars=error_bars, axis_title=expt.roi_ids(
                    channel=exptGrp.args['channel'],
                    label=exptGrp.args['imaging_label'],
                    roi_filter=roi_filter)[idx],
                rasterized=rasterized)

    return fig


def plot_lap_transients(
        expt_grp, channel='Ch2', label=None, roi_filter=None,
        running_only=False, n_rows=6, n_cols=10, sizeby=None, shapeby=None,
        colorby='trial', rasterized=False, plot_pfs=False, plot_rewards=False):

    MIN_MS = 1.
    MAX_MS = 12.
    DEFAULT_MS = 10.

    all_positions = ba.total_absolute_position(expt_grp, imageSync=True)

    def trial_frame_total_pos(inputs):
        trial, frame = inputs
        return all_positions[trial][frame]

    all_trans = pd.concat(ee.transientsData(
        channel=channel, label=label, roi_filter=roi_filter, dataframe=True)
        for ee in expt_grp)
    all_trans['full_position'] = all_trans[['trial', 'start_frame']].apply(
        trial_frame_total_pos, axis=1)
    all_trans['lap'] = all_trans['full_position'].astype('int')
    all_trans['position'] = all_trans['full_position'].apply(
        lambda x: np.mod(x, 1))

    if running_only:
        all_running = {
            trial: ba.runningIntervals(
                trial, imageSync=True, returnBoolList=True)
            for expt in expt_grp for trial in expt.findall('trial')}

        def is_running(inputs):
            trial, frame = inputs
            return all_running[trial][frame]

        all_trans['running'] = all_trans[['trial', 'start_frame']].apply(
            is_running, axis=1)

        all_trans = all_trans[all_trans['running']]

    for col in (shapeby, colorby, sizeby):

        if col is None or col in all_trans.columns:
            continue

        # See if we can infer it
        try:
            plotting.prepare_dataframe(all_trans, [col])
        except lab.classes.exceptions.InvalidDataFrame:
            pass
        else:
            continue

        # Else assume it is a per-lap behavior metric
        stimmed_laps = lab.classes.ExperimentGroup.stims_per_lap(
            expt_grp, col, trim_incomplete=False)

        stimmed_laps[col] = stimmed_laps['value'] > 0
        stimmed_laps.rename(columns={'lap': 'trial_lap'}, inplace=True)
        del stimmed_laps['value']

        trial_positions = {
            trial: ba.absolutePosition(trial, imageSync=True)
            for expt in expt_grp for trial in expt.findall('trial')}

        def trial_frame_pos(inputs):
            trial, frame = inputs
            return trial_positions[trial][frame].astype('int')

        all_trans['trial_lap'] = all_trans[['trial', 'start_frame']].apply(
            trial_frame_pos, axis=1)

        all_trans = pd.merge(
            all_trans, stimmed_laps, how='left', on=['trial', 'trial_lap'])

    if shapeby:
        all_trans['marker_shape'] = all_trans[shapeby].apply(
            lambda mark: '*' if mark else 's')
    else:
        all_trans['marker_shape'] = '*'

    if plot_pfs:
        pfs = expt_grp.pfs_n(roi_filter=roi_filter)

    plotting.prepare_dataframe(
        all_trans, ['mouseID', 'uniqueLocationKey', 'roi_id'])

    trans_by_roi = all_trans.groupby(
        ['mouseID', 'uniqueLocationKey', 'roi_id'])

    n_rois = len(trans_by_roi)

    figs, axs, axs_to_label = plotting.layout_subplots(
        n_rois, n_rows, n_cols, sharex=True, sharey=False)

    for ax, (roi_tuple, roi_trans) in zip(axs, trans_by_roi):

        if sizeby is None:
            roi_trans['sizes'] = DEFAULT_MS
        elif sizeby in all_trans.columns:
            roi_trans['sizes'] = np.interp(
                roi_trans[sizeby],
                [roi_trans[sizeby].min(), roi_trans[sizeby].max()],
                [MIN_MS, MAX_MS])
        else:
            raise ValueError('Unrecognized sizeby argument.')

        if colorby is not None:
            colorby_keys = sorted(set(roi_trans[colorby]))
            colors = {
                key: c for key, c in zip(
                    colorby_keys, lab.plotting.color_cycle())}
            roi_trans['color'] = roi_trans[colorby].apply(
                lambda key: colors[key])
        else:
            roi_trans['color'] = 'r'

        if plot_pfs:
            for trial in set(roi_trans['trial']):
                idx = trial.parent.roi_ids(
                    channel=channel, label=label,
                    roi_filter=roi_filter).index(roi_tuple[2])
                roi_pfs = pfs[trial.parent][idx]
                if len(roi_pfs):
                    y_start = int(all_positions[trial].min()) - 0.5
                    y_stop = int(all_positions[trial].max()) + 0.5
                for pf in roi_pfs:
                    if pf[0] < pf[1]:
                        ax.fill_between(
                            pf, [y_start, y_start], [y_stop, y_stop],
                            color='c', alpha=0.5)
                    else:
                        ax.fill_between(
                            [0, pf[1]], [y_start, y_start], [y_stop, y_stop],
                            color='c', alpha=0.5)
                        ax.fill_between(
                            [pf[0], 1], [y_start, y_start], [y_stop, y_stop],
                            color='c', alpha=0.5)

        # Shade un-experienced regions
        for trial in set(roi_trans['trial']):
            first_lap = int(all_positions[trial].min())
            first_pos = np.mod(all_positions[trial].min(), 1)
            last_lap = int(all_positions[trial].max())
            last_pos = np.mod(all_positions[trial].max(), 1)
            ax.fill_between(
                [0, first_pos], [first_lap - 0.5, first_lap - 0.5],
                [first_lap + 0.5, first_lap + 0.5], color='0.9')
            ax.fill_between(
                [last_pos, 1], [last_lap - 0.5, last_lap - 0.5],
                [last_lap + 0.5, last_lap + 0.5], color='0.9')

        if plot_rewards:
            for trial in set(roi_trans['trial']):
                reward_positions = trial.parent.rewardPositions(
                    units='normalized')
                if len(reward_positions):
                    y_start = int(all_positions[trial].min()) - 0.5
                    y_stop = int(all_positions[trial].max()) + 0.5
                for reward_pos in reward_positions:
                    ax.plot([reward_pos, reward_pos], [y_start, y_stop],
                            ls='--', color='g')

        # Plot transients
        for marker, marker_trans in roi_trans.groupby('marker_shape'):
            ax.scatter(
                marker_trans['position'], marker_trans['lap'], marker=marker,
                color=marker_trans['color'], s=marker_trans['sizes'])

        ax.set_ylim(-0.5, roi_trans['lap'].max() + 0.5)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xlim(0, 1)
        ax.set_title("{}, {}, {}".format(*roi_tuple))

    for ax in (ax for ax in axs if ax not in axs_to_label):
        ax.tick_params(labelleft=False, labelbottom=False, length=1)

    for ax in axs_to_label:
        ax.set_xlabel('position')
        ax.set_ylabel('lap number')

    for fig in figs:
        sns.despine(fig)

    return figs


def plotPositionAndTransients(
        exptGrp, roi_filter=None, rois='pcs', sort='centroid', plot_pf=True,
        subplot_dim=None, rasterized=False, running_trans_only=False):
    """Generate a subplot array of tuning curves for all place cells sorted by
    centroid of first place field

    Keyword arguments:
    exptGrp -- pcExperimentGroup to analyze
    rois -- ROIs to plot. 'all' plots all ROIs, 'pcs' only plots place cells,
        list if there is only 1 experiment in exptGrp is the list of rois
        to include by number, dict can be a dictionary of lists with
        experiments in exptGrp as keys each list as in the list option
    sort -- determines if place cells should be sorted, options are: 'centroid'
        sorts by average centroid, 'shift' sorts by centroid shift, None for no
        sorting. At the moment, 'shift' is not implemented
    plot_pf -- if True, plots shades the place fields
    subplot_dim -- a tuple of the form (nRows, nCols) that determines the
        layout of the tuning curves

    """
    bd = {}
    for expt in exptGrp:
        if len(expt.findall('trial')) > 1:
            warn('Multiple trials found, using first')
        bd[expt] = expt.find('trial').behaviorData(imageSync=True)

    # roi_dict is the same format as the dictionary option for 'rois',
    # a dictionary of lists containing the idx of ROIs to include
    # roi_list is a list with each element a list of (expt, roi_idx) tuples
    # containing all the common ROIs across experiments for the given cell
    roi_dict = None
    roi_list = None
    if rois == 'pcs':
        roi_dict = {}
        for expt in exptGrp:
            placeFields = exptGrp.pfs(roi_filter=roi_filter)[expt]
            pcs = [x for x in range(len(placeFields)) if placeFields[x] != []]
            roi_dict[expt] = pcs
    elif rois == 'all':
        roi_dict = {expt: range(expt.imaging_shape(
            channel=exptGrp.args['channel'],
            label=exptGrp.args.get('imaging_label'),
            roi_filter=roi_filter)[0]) for expt in exptGrp}

    # See if we got a list or ROI numbers
    elif len(exptGrp) == 1:
        try:
            for idx in rois:
                exptGrp.pfs(roi_filter=roi_filter)[exptGrp[0]][idx]
        except:
            pass
        else:
            roi_dict = {exptGrp[0]: rois}
            roi_list = [(exptGrp[0], idx) for idx in rois]
    # Otherwise assume we got the dictionary format
    else:
        try:
            for expt in exptGrp:
                for idx in rois[expt]:
                    exptGrp.pfs(roi_filter=roi_filter)[expt][idx]
        except TypeError:
            pass
        else:
            roi_dict = rois

    if roi_dict is None:
        raise Exception(
            "Invalid argument, see doc for options for 'rois' keyword")
    # If roi_list isn't set yet, pull out all the desired ROIs from roi_dict
    if roi_list is None:
        roi_list = []
        all_ROIs = exptGrp.allROIs(roi_filter=roi_filter,
                                   label=exptGrp.args.get('imaging_label'))
        for roi in all_ROIs:
            include = False
            for (expt, idx) in all_ROIs[roi]:
                if idx in roi_dict[expt]:
                    include = True
            if include:
                roi_list.append(all_ROIs[roi])

    nROIs = len(roi_list)

    if sort == 'centroid':
        # Calculate place field centroids
        centroids = {expt: place.calcCentroids(
            exptGrp.data(roi_filter=roi_filter)[expt], exptGrp.pfs(
                roi_filter=roi_filter)[expt]) for expt in exptGrp}

        # Average across observations of the ROI
        mean_centroid = []
        for roi in roi_list:
            c = [centroids[expt][idx] for (expt, idx) in roi
                 if centroids[expt][idx] != []]
            mean_centroid.append(np.mean(c) if len(c) > 0 else np.inf)
        order = np.argsort(mean_centroid)
        roi_list = np.array(roi_list)[order]
    elif sort == 'shift':
        raise Exception('Not implemented')

    # Setup axis to plot on
    if subplot_dim is None:
        # Subplots will be laid out in a 4xN array
        nCols = int(np.ceil((nROIs + 1.0) / 4.0))

        fig, axs = plt.subplots(4, nCols, subplot_kw={'polar': True},
                                figsize=(15, 8), squeeze=False)

        ax_to_label = [axs[-1, 0]]

        axs = axs.flatten()

        n_extras = (4 * nCols) - nROIs
        if n_extras > 0:
            for a in axs[-n_extras:]:
                a.set_visible(False)

    else:
        nRows = subplot_dim[0]
        nCols = subplot_dim[1]
        nFigs = int(np.ceil(nROIs / float(nRows * nCols)))
        fig = [[] for x in range(nFigs)]
        axs = []
        ax_to_label = []
        for f in range(nFigs):

            fig[f], ax = plt.subplots(nRows, nCols, subplot_kw={'polar': True},
                                      figsize=(15, 8), squeeze=False)

            ax_to_label.append(ax[-1, 0])

            axs = np.hstack([axs, ax.flatten()])

        n_extras = (nFigs * nRows * nCols) - nROIs
        if n_extras > 0:
            for a in ax.flatten()[-n_extras:]:
                a.set_visible(False)

    if plot_pf:
        # colors = [plt.cm.Set1(i) for i in np.linspace(0, .9, nROIs)]
        rel_centroids = np.array(mean_centroid)[order] / \
            float(exptGrp.args['nPositionBins'])
        colors = [plt.cm.hsv(i) for i in rel_centroids]
    else:
        colors = [None] * nROIs

    # Plotting
    for ax, roi, color in zip(axs, roi_list, colors):
        for expt, idx in roi:
            if plot_pf:
                placeField = [exptGrp.pfs_n(roi_filter=roi_filter)[expt][idx]]
            else:
                placeField = None

            trans_roi_filter = lambda x: x.id == expt.rois(
                roi_filter=roi_filter,
                label=exptGrp.args.get('imaging_label'))[idx].id

            place.plotPosition(
                expt.find('trial'), ax=ax, placeFields=placeField,
                placeFieldColors=[color], polar=True,
                trans_roi_filter=trans_roi_filter,
                rasterized=False, running_trans_only=running_trans_only,
                demixed=exptGrp.args['demixed'],
                behaviorData=bd[expt],
                label=exptGrp.args.get('imaging_label'))

            title = expt.roi_ids(roi_filter=roi_filter,
                                 label=exptGrp.args.get('imaging_label'))[idx]
            for tag in expt.rois(roi_filter=roi_filter,
                                 label=exptGrp.args.get('imaging_label')
                                 )[idx].tags:
                title += '_' + tag

            ax.set_title(title)
            # ax.set_title(expt.roi_ids(roi_filter=roi_filter)[idx])

        if ax not in ax_to_label:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_xlabel('')
            ax.set_ylabel('')

    return fig


def plotHeatmaps(exptGrps, roi_filters=None, valsToPlot=None, sortByVal=None,
                 dict_fn=lab.ExperimentGroup.dictByExposure,
                 pcs_only=True, cmaps=None, show_belt=True,
                 start_bin=0, stop_bin=-1):
    """Plot a heatmap of ROI activity combined across all PCs/ROIs in an
    pcExperimentGroup.

    Plots are laid out one row per exptGrp, one column per value.

    """

    if isinstance(exptGrps[0], lab.Experiment):
        exptGrps = [exptGrps]
    if roi_filters is None:
        roi_filters = it.repeat(None)

    grp_value_dicts = []
    for grp in exptGrps:
        grp_value_dicts.append(dict_fn(grp))

    if valsToPlot is None:
        valsToPlot = []
        for grp_value_dict in grp_value_dicts:
            for key in grp_value_dict.keys():
                if key not in valsToPlot:
                    valsToPlot.append(key)
    valsToPlot = np.sort(valsToPlot)
    if sortByVal is None:
        sortByVal = valsToPlot[0]

    nGroups = len(exptGrps)
    nVals = len(valsToPlot)
    fig, axs = plt.subplots(
        nGroups, nVals, figsize=(15, 8), squeeze=False)

    if cmaps is None:
        cmaps = [None] * nGroups

    for grpIdx, (grp, roi_filter, cmap) in enumerate(zip(exptGrps, roi_filters, cmaps)):
        mice_and_locations = list(set(
            [(e.parent, e.get('uniqueLocationKey', '')) for e in grp]))

        if pcs_only:
            pcs_filter = grp.pcs_filter(roi_filter=roi_filter)
        else:
            pcs_filter = roi_filter

        rois_by_val = {}
        expts_by_val = {}
        for val in valsToPlot:
            rois_by_val[val] = {}
            expts_by_val[val] = []

        for (mouse, location) in mice_and_locations[::-1]:
            # collect all experiments from this mouse at this location
            exptsAtLocation = mouse.exptsAtLocation(location)
            for e in exptsAtLocation[::-1]:
                if e not in grp:
                    exptsAtLocation.remove(e)
            mouse_loc_grp = grp.subGroup(exptsAtLocation)

            # now find the experiments for each column
            expts_to_track_rois = []
            value_dict = dict_fn(
                mouse_loc_grp)

            missing_data = False

            for val in valsToPlot:
                if val not in value_dict.keys():
                    mice_and_locations.remove((mouse, location))
                    missing_data = True
                    break
                for e in value_dict[val][::-1]:
                    if e not in mouse_loc_grp:
                        value_dict[val].remove(e)
                expts_to_track_rois.append(value_dict[val][0])

            if missing_data:
                continue
                # This ensures each value exists as a key in expts_to_track

            expts_to_track_rois = mouse_loc_grp.subGroup(
                expts_to_track_rois)

            # shared_pcs = expts_to_track_rois.sharedROIs(
            #     channel=grp.args['channel'],
            #     label=grp.args['imaging_label'],
            #     roi_filter=expts_to_track_rois.pcs_filter(
            #         roi_filter=roi_filter))
            shared_pcs = expts_to_track_rois.sharedROIs(
                channel=grp.args['channel'],
                label=grp.args['imaging_label'],
                roi_filter=pcs_filter)

            if not len(shared_pcs):
                continue

            dict_by_value = dict_fn(
                expts_to_track_rois)

            for val in valsToPlot:
                expt = dict_by_value[val][0]
                if expt not in expts_to_track_rois:
                    continue
                expts_by_val[val].append(expt)
                for roi_id in shared_pcs:
                    rois_by_val[val][(
                        mouse, location, roi_id)] = mouse_loc_grp.data(
                        roi_filter=lambda x: x.id == roi_id)[expt][0]

        # need to sort the ROIs of sortByVal according to peak
        order = np.argsort([np.argmax(x) for x in
                            rois_by_val[sortByVal].itervalues()])

        roi_tuples = rois_by_val[sortByVal].keys()
        rois_order = [roi_tuples[x] for x in order]

        for valIdx, val in enumerate(valsToPlot):
            val_grp = pcExperimentGroup(
                expts_by_val[val],
                channel=grp.args['channel'],
                imaging_label=grp.args['imaging_label'],
                demixed=grp.args['demixed'])

            if not len(val_grp):
                warn('No experiments found: Group: {}, Val: {}'.format(
                    grp.label(), valIdx))
                continue

            if valIdx == len(valsToPlot) - 1:
                cbar_visible = True
            else:
                cbar_visible = False

            # roi_filter is None because you already filtered above at the
            # level of sharedROIs
            place.plotPositionHeatmap(val_grp, roi_filter=None,
                                      ax=axs[grpIdx, valIdx],
                                      title='Val {}'.format(val),
                                      plotting_order=rois_order,
                                      cbar_visible=cbar_visible,
                                      norm=None,
                                      rasterized=False,
                                      cmap=cmap, show_belt=show_belt,
                                      start_bin=start_bin, stop_bin=stop_bin)
        plotting.right_label(axs[grpIdx, -1], grp.label())

    return fig


def plotRoisOverlay(exptGrp, roi_filter=None, rasterized=False):
    """Generate a figure of the imaging location with all ROIs overlaid"""

    assert len(exptGrp) == 1

    return af.plotRoisOverlay(
        exptGrp[0], channel=exptGrp.args['channel'],
        label=exptGrp.args['imaging_label'], roi_filter=roi_filter,
        rasterized=rasterized)


def plotSpatialTuningOverlay(exptGrp, roi_filter=None, labels_visible=True,
                             alpha=0.2, rasterized=False):
    """Takes single experiment experiment group and overlays spatial tuning
    information on prototype image

    """

    background_figure = exptGrp[0].returnFinalPrototype(
        channel=exptGrp.args['channel'])

    figs = []
    for plane in xrange(background_figure.shape[0]):
        fig = plt.figure()
        ax = plt.subplot2grid((1, 20), (0, 0), colspan=19,
                              rasterized=rasterized)
        cax = plt.subplot2grid((1, 20), (0, 19), rasterized=True)

        place.plot_spatial_tuning_overlay(
            ax, exptGrp, plane=plane, roi_filter=roi_filter,
            labels_visible=labels_visible, cax=cax, alpha=alpha)

        figs.append(fig)
    return figs


def plotPlaceFieldStatSummary(
        exptGrps, roi_filters=None, plot_method='hist', groupby=None,
        plotby=None, label_every_n=1, save_data=False, rasterized=False,
        **plot_kwargs):
    """Generate a summary plot of various statistics on the place fields.
    Can accept an expt, an exptGrp, or a list of exptGrps

    """

    data_to_save = {}

    fig, axs = plt.subplots(
        2, 3, figsize=(15, 8), subplot_kw={'rasterized': rasterized})
    fig.subplots_adjust(hspace=0.3)

    data_to_save['sensitivity'] = plot_metric(
        axs[0, 0], exptGrps, metric_fn=place.sensitivity,
        groupby=groupby, plotby=plotby, plot_method=plot_method,
        roi_filters=roi_filters, activity_kwargs=None,
        activity_label=gal(place.sensitivity), label_every_n=label_every_n,
        **plot_kwargs)

    data_to_save['specificity'] = plot_metric(
        axs[1, 0], exptGrps, metric_fn=place.specificity,
        groupby=groupby, plotby=plotby, plot_method=plot_method,
        roi_filters=roi_filters, activity_kwargs=None,
        activity_label=gal(place.specificity), label_every_n=label_every_n,
        **plot_kwargs)

    data_to_save['width'] = plot_metric(
        axs[0, 1], exptGrps, metric_fn=place.place_field_width,
        groupby=groupby, plotby=plotby, plot_method=plot_method,
        roi_filters=roi_filters, activity_kwargs=None,
        activity_label=gal(place.place_field_width),
        label_every_n=label_every_n, **plot_kwargs)

    if len(exptGrps) > 1:
        if groupby is None:
            percentage_groupby = None
        else:
            percentage_groupby = [filter(lambda x: x not in ['roi_id'], g)
                                  for g in groupby]
        data_to_save['pc_percentage'] = plot_metric(
            axs[1, 1], exptGrps, metric_fn=place.place_cell_percentage,
            groupby=percentage_groupby, plotby=plotby, plot_method=plot_method,
            roi_filters=roi_filters, activity_kwargs=None,
            activity_label=gal(place.place_cell_percentage),
            label_every_n=label_every_n, **plot_kwargs)

    data_to_save['circular_variance'] = plot_metric(
        axs[0, 2], exptGrps, metric_fn=place.circular_variance,
        groupby=groupby, plotby=plotby, plot_method=plot_method,
        roi_filters=roi_filters, activity_kwargs=None,
        activity_label=gal(place.circular_variance),
        label_every_n=label_every_n, **plot_kwargs)

    data_to_save['sparsity'] = plot_metric(
        axs[1, 2], exptGrps, metric_fn=place.sparsity,
        groupby=groupby, plotby=plotby, plot_method=plot_method,
        roi_filters=roi_filters, activity_kwargs=None,
        activity_label=gal(place.sparsity), label_every_n=label_every_n,
        **plot_kwargs)

    if save_data:
        misc.save_data(data_to_save, fig=fig, label='place_field_stat_summary',
                       method=save_data)

    return fig


def plotPlaceFieldSummary(
        exptGrps, roi_filters=None, save_data=False, rasterized=False,
        colors=None, **plot_kwargs):

    data_to_save = {}

    fig, axs = plt.subplots(
        2, 3, figsize=(15, 8), subplot_kw={'rasterized': rasterized})
    fig.subplots_adjust(hspace=0.3)

    if roi_filters is None:
        roi_filters = [None] * len(exptGrps)

    if len(exptGrps) == 1 and len(exptGrps[0]) == 1:
        pass  # if single experiment group, don't do nSessions imaged
    else:
        data_to_save['n_sessions_imaged'] = plot_metric(
            axs[0, 0], exptGrps, metric_fn=place.n_sessions_imaged,
            groupby=[['roi_id', 'location', 'mouseID']], plotby=None,
            plot_method='hist', roi_filters=roi_filters, activity_kwargs=None,
            uniform_bins=True, unit_width=True, normed=True, plot_mean=True,
            activity_label=gal(place.n_sessions_imaged), colors=colors,
            **plot_kwargs)

    if len(exptGrps) == 1:
        data_to_save['n_sessions_pc'] = place.n_sessions_place_cell(
            exptGrps[0], roi_filter=roi_filters[0], ax=axs[1, 0],
            title_visible=True, minimum_observations=3, plotShuffle=True,
            color=None if colors is None else colors[0])
    else:
        data_to_save['n_sessions_pc'] = {}
        if colors is not None:
            n_ses_colors = colors
        else:
            color_cycle = lab.plotting.color_cycle()
            n_ses_colors = [color_cycle.next() for _ in exptGrps]

        for grp, roi_filter, color in zip(exptGrps, roi_filters, n_ses_colors):
            data_to_save['n_sessions_pc'][grp.label()] = \
                place.n_sessions_place_cell(
                    grp, roi_filter=roi_filter, ax=axs[1, 0],
                    title_visible=True, minimum_observations=0,
                    plotShuffle=False, color=color)

    data_to_save['n_place_fields'] = plot_metric(
        axs[0, 1], exptGrps, metric_fn=place.n_place_fields,
        groupby=[['roi_id', 'expt']], plotby=None, plot_method='hist',
        roi_filters=roi_filters, activity_kwargs=None, uniform_bins=True,
        unit_width=True, normed=True, plot_mean=True, colors=colors,
        activity_label=gal(place.n_place_fields), **plot_kwargs)

    data_to_save['is_ever_pc'] = place.is_ever_place_cell(
        exptGrps, roi_filters=roi_filters, ax=axs[1, 1],
        groupby=[['mouseID', 'session_number']], colors=colors, **plot_kwargs)

    if colors is not None:
        occupancy_colors = colors
    else:
        occupancy_colors = [c for c, _ in zip(
            lab.plotting.color_cycle(), exptGrps)]
    if all([grp.sameBelt() for grp in exptGrps]) and \
            len(set([grp[0].belt().get('beltID') for grp in exptGrps])) == 1:
        if len(exptGrps) == 1:
            data_to_save['occupancy'] = ba.positionOccupancy(
                exptGrps[0], ax=axs[0, 2], nBins=exptGrps[0].args['nPositionBins'],
                label=exptGrps[0].label(), color=occupancy_colors[0])
            data_to_save['distribution'] = place.place_field_distribution(
                exptGrps[0], roi_filter=roi_filters[0], ax=axs[1, 2],
                normed=True, showBelt=True, color=occupancy_colors[0],
                nBins=exptGrps[0].args['nPositionBins'],
                label=exptGrps[0].label())
        else:
            data_to_save['occupancy'] = {}
            data_to_save['distribution'] = {}
            for idx, (grp, roi_filter, color) in enumerate(
                    zip(exptGrps, roi_filters, occupancy_colors)):
                show_belt = idx == 1
                data_to_save['occupancy'][grp.label()] = ba.positionOccupancy(
                    grp, ax=axs[0, 2], nBins=grp.args['nPositionBins'],
                    showBelt=show_belt, label=grp.label(), color=color)
                data_to_save['distribution'][grp.label()] = \
                    place.place_field_distribution(
                        grp, roi_filter=roi_filter, ax=axs[1, 2], normed=True,
                        showBelt=show_belt, nBins=grp.args['nPositionBins'],
                        label=grp.label(), color=color)
    else:
        pc_filters = [expt_grp.pcs_filter(roi_filter=roi_filter) for
                      expt_grp, roi_filter in zip(exptGrps, roi_filters)]
        data_to_save['fraction_sessions_pc'] = plot_metric(
            axs[1, 2], exptGrps, metric_fn=lab.ExperimentGroup.filtered_rois,
            groupby=(('mouseID', 'uniqueLocationKey', 'roi_id'),), plotby=None,
            colorby=None, plot_method='cdf', roi_filters=pc_filters,
            activity_kwargs=[
                {'include_roi_filter': roi_filter} for roi_filter in roi_filters],
            colors=colors, activity_label='Fraction of sessions a place cell',
            rotate_labels=False)

    if save_data:
        misc.save_data(data_to_save, fig=fig, label='place_field_summary',
                       method=save_data)

    return fig


def plotStabilitySummary(
        exptGrps, roi_filters=None, plot_method='line', save_data=False,
        rasterized=False, groupby=None, plotby=None, shuffle_plotby=True,
        pool_shuffle=False, plot_shuffle=True, fig_title='', **plot_kwargs):

    data_to_save = {}

    fig, axs = plt.subplots(
        2, 3, figsize=(15, 8), subplot_kw={'rasterized': rasterized})
    fig.subplots_adjust(hspace=0.3)

    if groupby is None:
        recurrence_groupby = None
    else:
        recurrence_groupby = [filter(lambda x: 'roi' not in x, g)
                              for g in groupby]
    data_to_save['recurrence'] = plot_metric(
        axs[0, 0], exptGrps, metric_fn=place.recurrence_probability,
        groupby=recurrence_groupby, plotby=plotby, plot_method=plot_method,
        roi_filters=roi_filters, activity_kwargs=None,
        plot_shuffle=plot_shuffle, shuffle_plotby=shuffle_plotby,
        pool_shuffle=pool_shuffle, activity_label=gal(
            place.recurrence_probability), **plot_kwargs)
    axs[0, 0].set_title(gal(place.recurrence_probability))

    activity_kwargs = {'activity_filter': 'pc_both', 'shuffle': plot_shuffle}
    data_to_save['pf_corr'] = plot_metric(
        axs[1, 0], exptGrps, metric_fn=place.place_field_correlation,
        groupby=groupby, plotby=plotby, plot_method=plot_method,
        roi_filters=roi_filters,
        activity_kwargs=activity_kwargs, plot_shuffle=plot_shuffle,
        shuffle_plotby=shuffle_plotby, pool_shuffle=pool_shuffle,
        activity_label=gal(place.place_field_correlation, activity_kwargs),
        **plot_kwargs)
    axs[1, 0].set_title(gal(place.place_field_correlation, activity_kwargs))

    if groupby is None:
        pop_vector_groupby = None
    else:
        pop_vector_groupby = [[
            grouping if 'roi_id' not in grouping else 'position_bin_index'
            for grouping in group] for group in groupby]

    activity_kwargs = {
        'method': 'corr', 'activity_filter': 'pc_both',
        'shuffle': plot_shuffle}
    data_to_save['pv_corr_c'] = plot_metric(
        axs[0, 1], exptGrps, metric_fn=place.population_vector_correlation,
        groupby=pop_vector_groupby, plotby=plotby, plot_method=plot_method,
        roi_filters=roi_filters, activity_kwargs=activity_kwargs,
        plot_shuffle=plot_shuffle, shuffle_plotby=shuffle_plotby,
        pool_shuffle=pool_shuffle, activity_label=gal(
            place.population_vector_correlation, activity_kwargs),
        **plot_kwargs)
    axs[0, 1].set_title(gal(
        place.population_vector_correlation, activity_kwargs))

    activity_kwargs = {
        'method': 'angle', 'activity_filter': 'pc_both',
        'shuffle': plot_shuffle}
    data_to_save['pv_corr_a'] = plot_metric(
        axs[1, 1], exptGrps, metric_fn=place.population_vector_correlation,
        groupby=pop_vector_groupby, plotby=plotby, plot_method=plot_method,
        roi_filters=roi_filters, activity_kwargs=activity_kwargs,
        plot_shuffle=plot_shuffle, shuffle_plotby=shuffle_plotby,
        pool_shuffle=pool_shuffle, activity_label=gal(
            place.population_vector_correlation, activity_kwargs),
        **plot_kwargs)
    axs[1, 1].set_title(gal(
        place.population_vector_correlation, activity_kwargs))

    activity_kwargs = {'activity_method': 'frequency', 'shuffle': plot_shuffle}
    data_to_save['overlap_f'] = plot_metric(
        axs[0, 2], exptGrps, metric_fn=place.overlap,
        groupby=groupby, plotby=plotby, plot_method=plot_method,
        roi_filters=roi_filters, activity_kwargs=activity_kwargs,
        plot_shuffle=plot_shuffle, shuffle_plotby=shuffle_plotby,
        pool_shuffle=False, activity_label=gal(
            place.overlap, activity_kwargs), **plot_kwargs)
    axs[0, 2].set_title(gal(place.overlap, activity_kwargs))

    activity_kwargs = {'activity_method': 'amplitude', 'shuffle': plot_shuffle}
    data_to_save['overlap_a'] = plot_metric(
        axs[1, 2], exptGrps, metric_fn=place.overlap,
        groupby=groupby, plotby=plotby, plot_method=plot_method,
        roi_filters=roi_filters, activity_kwargs=activity_kwargs,
        plot_shuffle=plot_shuffle, shuffle_plotby=shuffle_plotby,
        pool_shuffle=False, activity_label=gal(
            place.overlap, activity_kwargs), **plot_kwargs)
    axs[1, 2].set_title(gal(place.overlap, activity_kwargs))

    for ax in list(axs.flat)[1:]:
        ax.get_legend().set_visible(False)

    if fig_title:
        title_text = fig_title + '\ngroupby={}'.format(groupby)
    else:
        title_text = 'groupby={}'.format(groupby)
    fig.suptitle(title_text)

    if save_data:
        misc.save_data(data_to_save, fig=fig, label='stability_summary')

    return fig


def plotCentroidShifts(
        exptGrps, roi_filters=None, save_data=False, rasterized=False,
        groupby=None, plotby=None, fig_title='', **plot_kwargs):

    data_to_save = {}

    fig, axs = plt.subplots(
        2, 3, figsize=(15, 8), subplot_kw={'rasterized': rasterized})
    fig.subplots_adjust(hspace=0.3)

    plot_kwargs['hist_kwargs'] = {'normed': True, 'bins': 5, 'range': (-1, 1)}
    data_to_save['centroid_line-o-gram'] = plot_metric(
        axs[0, 0], exptGrps, metric_fn=place.centroid_shift, groupby=groupby,
        plotby=plotby, plot_method='line-o-gram',
        roi_filters=roi_filters, activity_kwargs=None, plot_shuffle=True,
        shuffle_plotby=False, pool_shuffle=True,
        activity_label=gal(place.centroid_shift), **plot_kwargs)
    axs[0, 0].set_title(gal(place.centroid_shift))

    plot_kwargs.pop('hist_kwargs')
    data_to_save['centroid_abs_cdf'] = plot_metric(
        axs[0, 1], exptGrps, metric_fn=place.centroid_shift, groupby=groupby,
        plotby=plotby, plot_method='cdf',
        roi_filters=roi_filters, activity_kwargs=None, plot_shuffle=True,
        shuffle_plotby=False, pool_shuffle=True, plot_abs=True,
        activity_label=gal(place.centroid_shift), **plot_kwargs)
    axs[0, 1].set_title(gal(place.centroid_shift))

    data_to_save['centroid_line'] = plot_metric(
        axs[0, 2], exptGrps, metric_fn=place.centroid_shift,
        groupby=groupby, plotby=plotby, plot_method='line',
        roi_filters=roi_filters, activity_kwargs=None, plot_shuffle=True,
        shuffle_plotby=True, pool_shuffle=True, plot_abs=True,
        activity_label=gal(place.centroid_shift), **plot_kwargs)
    axs[0, 2].set_title(gal(place.centroid_shift))

    plot_kwargs['hist_kwargs'] = {'normed': True, 'bins': 5, 'range': (-1, 1)}
    data_to_save['activity_centroid_line-o-gram'] = plot_metric(
        axs[1, 0], exptGrps, metric_fn=place.activity_centroid_shift,
        groupby=groupby, plotby=plotby, plot_method='line-o-gram',
        roi_filters=roi_filters, activity_kwargs=None, plot_shuffle=True,
        shuffle_plotby=False, pool_shuffle=True,
        activity_label=gal(place.activity_centroid_shift), **plot_kwargs)
    axs[1, 0].set_title(gal(place.activity_centroid_shift))

    plot_kwargs.pop('hist_kwargs')
    data_to_save['activity_centroid_abs_cdf'] = plot_metric(
        axs[1, 1], exptGrps, metric_fn=place.activity_centroid_shift,
        groupby=groupby, plotby=plotby, plot_method='cdf',
        roi_filters=roi_filters, activity_kwargs=None, plot_shuffle=True,
        shuffle_plotby=False, pool_shuffle=True, plot_abs=True,
        activity_label=gal(place.activity_centroid_shift), **plot_kwargs)
    axs[1, 1].set_title(gal(place.activity_centroid_shift))

    data_to_save['activity_centroid_line'] = plot_metric(
        axs[1, 2], exptGrps, metric_fn=place.activity_centroid_shift,
        groupby=groupby, plotby=plotby, plot_method='line',
        roi_filters=roi_filters, activity_kwargs=None, plot_shuffle=True,
        shuffle_plotby=True, pool_shuffle=True, plot_abs=True,
        activity_label=gal(place.activity_centroid_shift), **plot_kwargs)
    axs[1, 2].set_title(gal(place.activity_centroid_shift))

    if fig_title:
        title_text = fig_title + '\ngroupby={}'.format(groupby)
    else:
        title_text = 'groupby={}'.format(groupby)
    fig.suptitle(title_text)

    if save_data:
        misc.save_data(data_to_save, fig=fig, label='centroid_shift_summary')

    return fig


def plotPlaceFieldDistributionByDay(
        exptGrps, roi_filters=None, normed=True, plotMean=True):
    """
    Accepts either exptGrp or list of exptGrps
    Creates a figure nGroups x nBelts and draws a placeFieldDistribution curve
    for each day of exposure
    Experiments in the same group with the same days of exposure are combined
    """

    if isinstance(exptGrps, place.pcExperimentGroup):
        exptGrps = [exptGrps]
    if roi_filters is None:
        roi_filters = [None] * len(exptGrps)
    if callable(roi_filters):
        roi_filters = [roi_filters]

    nGroups = len(exptGrps)
    beltsByGrp = []
    for grp in exptGrps:
        beltsInGrp = set()
        for expt in grp:
            beltsInGrp.add(expt.get('belt'))
        if '' in beltsInGrp:
            beltsInGrp.remove('')
        if None in beltsInGrp:
            beltsInGrp.remove(None)
        beltsByGrp.append(beltsInGrp)

    allBelts = set()
    for beltsInGrp in beltsByGrp:
        allBelts = allBelts | beltsInGrp
    nBelts = len(allBelts)

    fig, axs = plt.subplots(nGroups, nBelts, figsize=(15, 8), squeeze=False)

    fig.suptitle('Place field distribution by days of exposure')

    for grpIdx, grp, roi_filter in it.izip(it.count(), exptGrps, roi_filters):
        for beltIdx, belt in enumerate(allBelts):
            ax = axs[grpIdx, beltIdx]

            # find all the experiments in the group run on this belt
            exptsOnBelt = [expt for expt in grp if expt.get('belt') == belt]
            exptsOnBelt = grp.subGroup(exptsOnBelt)

            # Skip if there are no expts with this belt for this exptGrp
            if len(exptsOnBelt) == 0:
                continue

            b = exptsOnBelt[0].belt()
            b.addToAxis(ax)

            # on what days of exposure was the animal imaged on this belt?
            exptsByExposure = lab.ExperimentGroup.dictByExposure(exptsOnBelt)

            colors = lab.plotting.color_cycle()
            color = []
            labels = []
            all_data = np.empty((0, exptGrps[0].args['nPositionBins']))
            # for val in allDaysOfExposure:
            for exposure, expts in exptsByExposure.iteritems():
                exptsAtDay = exptsOnBelt.subGroup(expts)

                c = colors.next()
                data = place.place_field_distribution(
                    exptsAtDay, ax=ax, roi_filter=roi_filter, normed=normed,
                    color=c)

                if plotMean:
                    all_data = np.vstack([all_data, data])
                color.append(c)
                labels.append('{} days'.format(exposure))

            if plotMean:
                ax.plot(np.linspace(0, 1, exptGrps[0].args['nPositionBins']),
                        np.mean(all_data, axis=0), 'k', lw=5)
                color.append('k')
                labels.append('Mean')

            plotting.stackedText(ax, textList=labels, colors=color, loc=1)
            if grpIdx == 0:
                ax.set_title(belt)

            if beltIdx == nBelts - 1:
                plotting.right_label(ax, grp.label())

    return fig


def activity_stability_correlation_figure(
        exptGrp, activity_metric='responseMagnitude',
        activity_combine_method='mean',
        min_pair_delta=datetime.timedelta(hours=12),
        max_pair_delta=datetime.timedelta(hours=36), roi_filter=None,
        rasterized=False, save_data=False):

    paired_group = exptGrp.pair(
        min_pair_delta=min_pair_delta, max_pair_delta=max_pair_delta)

    fig, axs = plt.subplots(
        2, 4, figsize=(15, 8), subplot_kw={'rasterized': rasterized})

    data_to_save = {}

    activity, stability = place.plot_activity_stability_correlation(
        axs[0, 0], paired_group, activity_metric,
        place.place_field_correlation,
        stability_kwargs={'activity_filter': 'pc_first'},
        activity_combine_method=activity_combine_method, roi_filter=roi_filter,
        z_score=True)
    data_to_save['stability'] = {'activity': activity, 'stability': stability}

    activity, stability = place.plot_activity_stability_correlation(
        axs[1, 0], paired_group, activity_metric, place.centroid_shift,
        activity_combine_method=activity_combine_method, roi_filter=roi_filter,
        z_score=True)
    data_to_save['centroid'] = {'activity': activity, 'stability': stability}

    activity, stability = place.plot_activity_stability_correlation(
        axs[0, 1], paired_group, activity_metric, place.overlap,
        stability_kwargs={'activity_method': 'mean', 'activity_filter': None},
        activity_combine_method=activity_combine_method, roi_filter=roi_filter,
        z_score=True)
    data_to_save['overlap_m'] = {'activity': activity, 'stability': stability}

    activity, stability = place.plot_activity_stability_correlation(
        axs[1, 1], paired_group, activity_metric, place.overlap,
        stability_kwargs={'activity_method': 'amplitude',
                          'activity_filter': None},
        activity_combine_method=activity_combine_method, roi_filter=roi_filter,
        z_score=True)
    data_to_save['overlap_amp'] = {
        'activity': activity, 'stability': stability}

    activity, stability = place.plot_activity_stability_correlation(
        axs[0, 2], paired_group, activity_metric, place.overlap,
        stability_kwargs={'activity_method': 'norm transient auc2',
                          'activity_filter': None},
        activity_combine_method=activity_combine_method, roi_filter=roi_filter,
        z_score=True)
    data_to_save['overlap_non_transient_area'] = {
        'activity': activity, 'stability': stability}

    activity, stability = place.plot_activity_stability_correlation(
        axs[1, 2], paired_group, activity_metric, place.overlap,
        stability_kwargs={'activity_method': 'is place cell',
                          'activity_filter': 'pc_either'},
        activity_combine_method=activity_combine_method, roi_filter=roi_filter,
        z_score=True)
    data_to_save['overlap_is_place_cell_either'] = {
        'activity': activity, 'stability': stability}

    activity, stability = place.plot_activity_stability_correlation(
        axs[0, 3], paired_group, activity_metric, place.overlap,
        stability_kwargs={'activity_method': 'is place cell',
                          'activity_filter': 'pc_first'},
        activity_combine_method=activity_combine_method, roi_filter=roi_filter,
        z_score=True)
    data_to_save['overlap_is_place_cell_first'] = {
        'activity': activity, 'stability': stability}

    activity, stability = place.plot_activity_stability_correlation(
        axs[1, 3], paired_group, activity_metric, place.overlap,
        stability_kwargs={'activity_method': 'is place cell',
                          'activity_filter': 'pc_second'},
        activity_combine_method=activity_combine_method, roi_filter=roi_filter,
        z_score=True)
    data_to_save['overlap_is_place_cell_second'] = {
        'activity': activity, 'stability': stability}

    fig.suptitle(
        'Activity-stability correlation, paired from '
        + '{} to {}, combined method: {}'.format(
            min_pair_delta, max_pair_delta, activity_combine_method))

    if save_data:
        misc.save_data(
            data_to_save, fig=fig, label='activity_stability_correlation')

    return fig


def activity_vs_place_coding_figure(
        exptGrp, min_pair_delta=datetime.timedelta(hours=12),
        max_pair_delta=datetime.timedelta(hours=36), roi_filter=None,
        rasterized=False, z_score=True, save_data=False):

    paired_group = exptGrp.pair(
        min_pair_delta=min_pair_delta, max_pair_delta=max_pair_delta)

    fig, axs = plt.subplots(
        2, 3, figsize=(15, 8), squeeze=False,
        subplot_kw={'rasterized': rasterized})

    data_to_save = {}

    values = place.plot_activity_versus_place_coding(
        axs[0, 0], paired_group, place.population_activity,
        fn_kwargs={'stat': 'amplitude'}, roi_filter=roi_filter,
        z_score=z_score)
    data_to_save['amplitude'] = values
    axs[0, 0].set_ylabel('Amplitude')

    values = place.plot_activity_versus_place_coding(
        axs[0, 1], paired_group, place.population_activity,
        fn_kwargs={'stat': 'frequency'}, roi_filter=roi_filter,
        z_score=z_score)
    data_to_save['frequency'] = values
    axs[0, 1].set_ylabel('Frequency')

    values = place.plot_activity_versus_place_coding(
        axs[0, 2], paired_group, place.population_activity,
        fn_kwargs={'stat': 'norm transient auc'}, roi_filter=roi_filter,
        z_score=z_score)
    data_to_save['norm transient auc'] = values
    axs[0, 2].set_ylabel('Norm Transient AUC')

    values = place.plot_activity_versus_place_coding(
        axs[1, 0], paired_group, place.population_activity,
        fn_kwargs={'stat': 'responseMagnitude'}, roi_filter=roi_filter,
        z_score=z_score)
    data_to_save['responseMagnitude'] = values
    axs[1, 0].set_ylabel('Response Magnitude')

    values = place.plot_activity_versus_place_coding(
        axs[1, 1], paired_group, place.population_activity,
        fn_kwargs={'stat': 'is place cell'}, roi_filter=roi_filter,
        z_score=z_score)
    data_to_save['is place cell'] = values
    axs[1, 1].set_ylabel('Is Place Cell')

    fig.suptitle(
        'Of the cells that are/are not place cells today, \n' +
        'what was/is/will be their activity yesterday/today/tomorrow?')

    if save_data:
        misc.save_data(
            data_to_save, fig=fig, label='activity_vs_place_coding')

    return fig


def acute_remapping_figure(
        exptGrps_list, metric_fn, roi_filters_list=None, group_labels=None,
        groupby=None, plotby=None, save_data=False, rasterized=False,
        activity_kwargs=None, **plot_kwargs):

    data_to_save = {}

    fig, axs = plt.subplots(
        1, len(exptGrps_list[0]), figsize=(15, 8), squeeze=False,
        subplot_kw={'rasterized': rasterized})

    if roi_filters_list is None:
        roi_filters_list = [
            [None] * len(conditions) for conditions in exptGrps_list]

    if group_labels is None:
        group_labels = [
            'Group ' + str(idx) for idx in range(len(exptGrps_list))]

    condition_labels = [
        exptGrp.label() if exptGrp.label() is not None
        else 'Condition ' + str(idx)
        for idx, exptGrp in it.izip(it.count(), exptGrps_list[0])]

    for condition_idx, condition_label, condition_exptGrps, \
            condition_roi_filters in it.izip(
            it.count(), condition_labels, it.izip(*exptGrps_list),
            it.izip(*roi_filters_list)):

        # Change the exptGrp label to be the overall group label, not the
        # condition label. Save the original label to restore after plotting
        assert len(condition_exptGrps) == len(group_labels)
        original_labels = {}
        for label, exptGrp in it.izip(group_labels, condition_exptGrps):
            original_labels[exptGrp] = exptGrp.label()
            exptGrp.label(label)

        data_to_save[condition_label] = plot_metric(
            axs[0, condition_idx], condition_exptGrps, metric_fn=metric_fn,
            groupby=groupby, plotby=plotby, plot_method='cdf',
            roi_filters=condition_roi_filters, activity_kwargs=activity_kwargs,
            plot_shuffle=True, shuffle_plotby=False, pool_shuffle=True,
            activity_label=gal(metric_fn, activity_kwargs), **plot_kwargs)

        for exptGrp, label in original_labels.iteritems():
            exptGrp.label(label)

        ax_title = axs[0, condition_idx].get_title()
        axs[0, condition_idx].set_title(condition_label + ': ' + ax_title)

    if save_data:
        misc.save_data(data_to_save, fig=fig, label='acute_remap')

    return fig


def acute_remapping_summary_figure(
        exptGrps_list, roi_filters_list=None, group_labels=None, groupby=None,
        save_data=False, rasterized=False):

    fig, axs = plt.subplots(
        2, 7, figsize=(15, 8), subplot_kw={'rasterized': rasterized})
    fig.subplots_adjust(hspace=0.3)

    data_to_save = {}

    data_to_save['recurrence'] = place.plot_acute_remapping_metric(
        axs[0, 0], exptGrps_list, metric_fn=place.recurrence_probability,
        group_labels=group_labels, groupby=groupby, plotby=['condition'],
        plot_method='grouped_bar', roi_filters_list=roi_filters_list,
        activity_kwargs=None, plot_shuffle=True, shuffle_plotby=True,
        pool_shuffle=False)

    data_to_save['pf_corr'] = place.plot_acute_remapping_metric(
        axs[0, 1], exptGrps_list, metric_fn=place.place_field_correlation,
        group_labels=group_labels, groupby=groupby, plotby=['condition'],
        plot_method='grouped_bar', roi_filters_list=roi_filters_list,
        activity_kwargs=None, plot_shuffle=True, shuffle_plotby=True,
        pool_shuffle=False)

    data_to_save['pv_corr'] = place.plot_acute_remapping_metric(
        axs[0, 2], exptGrps_list,
        metric_fn=place.population_vector_correlation,
        group_labels=group_labels, groupby=groupby, plotby=['condition'],
        plot_method='grouped_bar', roi_filters_list=roi_filters_list,
        activity_kwargs={'method': 'corr'}, plot_shuffle=True,
        shuffle_plotby=True, pool_shuffle=False)

    data_to_save['centroid'] = place.plot_acute_remapping_metric(
        axs[0, 3], exptGrps_list, metric_fn=place.centroid_shift,
        group_labels=group_labels, groupby=groupby, plotby=['condition'],
        plot_method='grouped_bar', roi_filters_list=roi_filters_list,
        activity_kwargs=None, plot_abs=True, plot_shuffle=True,
        shuffle_plotby=True, pool_shuffle=False)

    data_to_save['activity_centroid'] = place.plot_acute_remapping_metric(
        axs[0, 4], exptGrps_list, metric_fn=place.activity_centroid_shift,
        group_labels=group_labels, groupby=groupby, plotby=['condition'],
        plot_method='grouped_bar', roi_filters_list=roi_filters_list,
        activity_kwargs=None, plot_shuffle=True,
        shuffle_plotby=True, pool_shuffle=False)

    data_to_save['overlap_f'] = place.plot_acute_remapping_metric(
        axs[0, 5], exptGrps_list, metric_fn=place.overlap,
        group_labels=group_labels, groupby=groupby, plotby=['condition'],
        plot_method='grouped_bar', roi_filters_list=roi_filters_list,
        activity_kwargs={'activity_method': 'frequency'}, plot_shuffle=True,
        shuffle_plotby=True, pool_shuffle=False)

    data_to_save['overlap_m'] = place.plot_acute_remapping_metric(
        axs[0, 6], exptGrps_list, metric_fn=place.overlap,
        group_labels=group_labels, groupby=groupby, plotby=['condition'],
        plot_method='grouped_bar', roi_filters_list=roi_filters_list,
        activity_kwargs={'activity_method': 'is place cell'},
        plot_shuffle=True, shuffle_plotby=True, pool_shuffle=False)

    place.plot_acute_remapping_metric(
        axs[1, 0], exptGrps_list, metric_fn=place.recurrence_probability,
        group_labels=group_labels, groupby=groupby, plotby=['condition'],
        plot_method='cdf', roi_filters_list=roi_filters_list,
        activity_kwargs=None, plot_shuffle=True, shuffle_plotby=False,
        pool_shuffle=False)

    place.plot_acute_remapping_metric(
        axs[1, 1], exptGrps_list, metric_fn=place.place_field_correlation,
        group_labels=group_labels, groupby=groupby, plotby=['condition'],
        plot_method='cdf', roi_filters_list=roi_filters_list,
        activity_kwargs=None, plot_shuffle=True, shuffle_plotby=False,
        pool_shuffle=False)

    place.plot_acute_remapping_metric(
        axs[1, 2], exptGrps_list,
        metric_fn=place.population_vector_correlation,
        group_labels=group_labels, groupby=groupby, plotby=['condition'],
        plot_method='cdf', roi_filters_list=roi_filters_list,
        activity_kwargs={'method': 'corr'}, plot_shuffle=True,
        shuffle_plotby=False, pool_shuffle=False)

    place.plot_acute_remapping_metric(
        axs[1, 3], exptGrps_list, metric_fn=place.centroid_shift,
        group_labels=group_labels, groupby=groupby, plotby=['condition'],
        plot_method='cdf', roi_filters_list=roi_filters_list,
        activity_kwargs=None, plot_abs=True, plot_shuffle=True,
        shuffle_plotby=False, pool_shuffle=False)

    place.plot_acute_remapping_metric(
        axs[1, 4], exptGrps_list, metric_fn=place.activity_centroid_shift,
        group_labels=group_labels, groupby=groupby, plotby=['condition'],
        plot_method='cdf', roi_filters_list=roi_filters_list,
        activity_kwargs=None, plot_shuffle=True,
        shuffle_plotby=False, pool_shuffle=False)

    place.plot_acute_remapping_metric(
        axs[1, 5], exptGrps_list, metric_fn=place.overlap,
        group_labels=group_labels, groupby=groupby, plotby=['condition'],
        plot_method='cdf', roi_filters_list=roi_filters_list,
        activity_kwargs={'activity_method': 'frequency'}, plot_shuffle=True,
        shuffle_plotby=False, pool_shuffle=False)

    place.plot_acute_remapping_metric(
        axs[1, 6], exptGrps_list, metric_fn=place.overlap,
        group_labels=group_labels, groupby=groupby, plotby=['condition'],
        plot_method='cdf', roi_filters_list=roi_filters_list,
        activity_kwargs={'activity_method': 'is place cell'},
        plot_shuffle=True, shuffle_plotby=False, pool_shuffle=False)

    # Clean up labels
    fig.suptitle('groupby = {}'.format(groupby))
    for ax in axs[:, 1:].flat:
        ax.get_legend().set_visible(False)
    for ax in axs[1, 1:]:
        ax.set_ylabel('')
    for ax in axs[1, :]:
        ax.set_title('')
    for ax in axs[0, :]:
        ax.set_ylabel('')

    if save_data:
        misc.save_data(data_to_save, fig=fig, label='acute_remap_summary',
                       method=save_data)

    return fig


def place_field_reward_remapping_figure(
        exptGrps, roi_filters=None, groupby=None, plotby=None, orderby=None,
        label_every_n=1, save_data=False, rasterized=False, **plot_kwargs):

    if roi_filters is None:
        roi_filters = [None] * len(exptGrps)

    conditions_list = []
    for exptGrp in exptGrps:
        conditions = set()
        for condition in set(exptGrp.dataframe(
                exptGrp, include_columns=['condition'])['condition'].values):
            conditions.add(condition)
        conditions_list.append(conditions)

    # Take the intersection of conditions across the groups
    conditions = conditions_list[0]
    for cc in conditions_list:
        conditions = conditions.intersection(cc)
    conditions = list(conditions)
    conditions.sort()

    new_exptGrps = []
    new_roiFilters = []
    activity_kwargs = []
    new_plot_kwargs = copy(plot_kwargs)
    if 'markers' in new_plot_kwargs:
        new_plot_kwargs['markers'] = [
            marker for marker in new_plot_kwargs['markers']
            for p in conditions]
    if 'colors' in new_plot_kwargs:
        import seaborn.apionly as sns
        colors = new_plot_kwargs['colors']
        new_colors = [sns.light_palette(
            color, len(conditions) + 1, reverse=True)[:-1]
            for color in colors]
        new_colors = list(it.chain(*new_colors))
        new_plot_kwargs['colors'] = new_colors

    for exptGrp, roi_filter in zip(exptGrps, roi_filters):

        for condition in conditions:
            new_exptGrp = copy(exptGrp)
            new_exptGrp.label(
                exptGrp.label() + '_condition_{}'.format(condition))
            new_exptGrps.append(new_exptGrp)
            new_roiFilters.append(roi_filter)
            activity_kwargs.append({'positions': condition})

    fig, axs = plt.subplots(
        2, 3, figsize=(15, 8), subplot_kw={'rasterized': rasterized},
        squeeze=False)
    fig.subplots_adjust(hspace=0.3)

    plot_metric(
        axs[0, 0], new_exptGrps, metric_fn=place.centroid_to_position_distance,
        groupby=groupby, plotby=plotby, orderby=orderby, plot_abs=True,
        plot_method='line', roi_filters=new_roiFilters,
        activity_kwargs=activity_kwargs,
        activity_label='pf centroid distance to position (norm units)',
        label_every_n=label_every_n, **new_plot_kwargs)

    plot_metric(
        axs[1, 0], exptGrps, metric_fn=place.centroid_to_position_distance,
        groupby=groupby, plotby=plotby, orderby=orderby, plot_abs=True,
        plot_method='cdf', roi_filters=roi_filters,
        activity_kwargs={'positions': 'reward'},
        activity_label='pf centroid distance to reward (norm units)',
        label_every_n=label_every_n, **plot_kwargs)

    plot_metric(
        axs[0, 1], new_exptGrps,
        metric_fn=place.mean_resultant_vector_to_position_angle,
        groupby=groupby, plotby=plotby, orderby=orderby, plot_abs=True,
        plot_method='line', roi_filters=new_roiFilters,
        activity_kwargs=activity_kwargs,
        activity_label='resultant vector to position (rad, all ROIs)',
        label_every_n=label_every_n, **new_plot_kwargs)

    plot_metric(
        axs[1, 1], exptGrps,
        metric_fn=place.mean_resultant_vector_to_position_angle,
        groupby=groupby, plotby=plotby, orderby=orderby, plot_abs=True,
        plot_method='cdf', roi_filters=roi_filters,
        activity_kwargs={'positions': 'reward'},
        activity_label='resultant vector to position (rad, all ROIs)',
        label_every_n=label_every_n, **plot_kwargs)

    for kwargs in activity_kwargs:
        kwargs.update({'pcs_only': True})
    plot_metric(
        axs[0, 2], new_exptGrps,
        metric_fn=place.mean_resultant_vector_to_position_angle,
        groupby=groupby, plotby=plotby, orderby=orderby, plot_abs=True,
        plot_method='line', roi_filters=new_roiFilters,
        activity_kwargs=activity_kwargs,
        activity_label='resultant vector to position (rad, circ var PCs)',
        label_every_n=label_every_n, **new_plot_kwargs)

    plot_metric(
        axs[1, 2], exptGrps,
        metric_fn=place.mean_resultant_vector_to_position_angle,
        groupby=groupby, plotby=plotby, orderby=orderby, plot_abs=True,
        plot_method='cdf', roi_filters=roi_filters,
        activity_kwargs={'positions': 'reward', 'pcs_only': True},
        activity_label='resultant vector to position (rad, circ var PCs)',
        label_every_n=label_every_n, **plot_kwargs)

    for ax in axs[:, 1:].flat:
        ax.get_legend().set_visible(False)

    fig.suptitle('groupby={}'.format(groupby))

    return fig


def place_to_reward_fraction_figure(
        exptGrps, roi_filters=None, groupby=None, plotby=None, orderby=None,
        method='centroid', agg_fn=np.mean, label_every_n=1, save_data=False,
        rasterized=False, **plot_kwargs):

    if method == 'centroid':
        thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
        titles = ['PCs with PF within {} of reward'.format(threshold)
                  for threshold in thresholds]
        activity_kwargs = {'method': 'centroid', 'positions': 'reward'}
        expected_value = lambda threshold: 2 * threshold
    elif method == 'resultant_vector':
        thresholds = [np.pi / 32, np.pi / 16, np.pi / 8, np.pi / 6, np.pi / 4,
                      np.pi / 2]
        titles = [r'Fraction of all cells with tuning within $\frac{\pi}{32}$ of reward',
                  r'Fraction of all cells with tuning within $\frac{\pi}{16}$ of reward',
                  r'Fraction of all cells with tuning within $\frac{\pi}{8}$ of reward',
                  r'Fraction of all cells with tuning within $\frac{\pi}{6}$ of reward',
                  r'Fraction of all cells with tuning within $\frac{\pi}{4}$ of reward',
                  r'Fraction of all cells with tuning within $\frac{\pi}{2}$ of reward']
        activity_kwargs = {'method': 'resultant_vector', 'positions': 'reward',
                           'pcs_only': False}
        expected_value = lambda threshold: 2 * (threshold / (2 * np.pi))

    if roi_filters is None:
        roi_filters = [None] * len(exptGrps)

    fig, axs = plt.subplots(
        2, 3, figsize=(15, 8), subplot_kw={'rasterized': rasterized},
        squeeze=False)
    fig.subplots_adjust(hspace=0.3)

    for ax, threshold, title in zip(axs.flat, thresholds, titles):
        activity_kwargs['threshold'] = threshold
        plot_metric(
            ax, exptGrps, metric_fn=place.centroid_to_position_threshold,
            roi_filters=roi_filters, groupby=groupby, plotby=plotby,
            orderby=orderby, agg_fn=agg_fn, plot_abs=False, plot_method='line',
            activity_kwargs=activity_kwargs,
            activity_label='fraction of cells', label_every_n=label_every_n,
            **plot_kwargs)
        ax.set_title(title)
        ax.axhline(expected_value(threshold), linestyle='--', color='k')

    fig.suptitle('groupby={}'.format(groupby))
    for ax in axs[0, :]:
        ax.set_xlabel('')
    for ax in axs[:, 1:].flat:
        ax.set_ylabel('')

    return fig


def hidden_activity_behavior_compare_figure(
        expt_grps, roi_filters=None, groupby=(('expt',),),
        colorby=('expt_grp',), metrics='position', rasterized=False,
        fig_title='', save_data=False, **plot_kwargs):
    """Scatters data from an activity metric against a behavior metric,
    pairing on the remaining columns after applying all groupbys, so
    effectively the last groupby

    """

    if roi_filters is None:
        roi_filters = [None] * len(expt_grps)

    # All metrics are (function, activity_kwargs, label) tuples
    # Make sure that all labels are unique!
    behavior_metrics = [
        (ra.fraction_of_laps_rewarded,
         {}, 'Fraction of laps rewarded'),
        (ra.fraction_licks_in_reward_zone,
         {}, 'Fraction of licks in reward zone'),
        # (ra.lick_to_reward_distance,
        #  {}, 'Lick to reward distance'),
        (ra.fraction_licks_near_rewards,
         {'pre_window_cm': 5, 'exclude_reward': True}, 'Anticipatory licking')
    ]

    if metrics == 'paired':
        activity_metrics = [
            (place.recurrence_probability, {}, 'Fraction of PCs recur'),
            (place.place_field_correlation,
             {'activity_filter': 'pc_both', 'shuffle': False},
             'Place field correlation'),
            (place.population_vector_correlation,
             {'method': 'corr', 'activity_filter': 'pc_both',
              'shuffle': False}, 'Population vector correlation'),
            (place.population_vector_correlation,
             {'method': 'angle', 'activity_filter': 'pc_both',
              'shuffle': False}, 'Population vector angle'),
            (place.centroid_shift, {'return_abs': True, 'shuffle': False},
             'Centroid shift'),
            (place.activity_centroid_shift, {'shuffle': False},
             'Activity centroid shift (all ROIs)'),
            (place.activity_centroid_shift,
             {'activity_filter': 'pc_both', 'circ_var_pcs': True,
              'shuffle': False}, 'Activity centroid shift (PCs only)'),
            (place.overlap, {'activity_method': 'frequency', 'shuffle': False},
             'Frequency overlap'),
            (place.overlap, {'activity_method': 'norm transient auc',
             'activity_filter': None, 'shuffle': False},
             'Activity overlap (norm transient AUC)'),
        ]
    elif metrics == 'position':
        activity_metrics = [
            (place.centroid_to_position_distance,
             {'positions': 'reward', 'return_abs': True},
             'Centroid to reward distance'),
            (place.mean_resultant_vector_to_position_angle,
             {'positions': 'reward'},
             'Activity centroid to reward angle (all ROIs)'),
            (place.mean_resultant_vector_to_position_angle,
             {'positions': 'reward', 'pcs_only': True},
             'Activity centroid to reward angle (PCs only)'),
            (place.centroid_to_position_threshold,
             {'positions': 'reward', 'threshold': 0.05},
             'Fraction PCs near reward, 0.05 threshold'),
            (place.centroid_to_position_threshold,
             {'positions': 'reward', 'threshold': 0.1},
             'Fraction PCs near reward, 0.1 threshold'),
            (place.centroid_to_position_threshold,
             {'positions': 'reward', 'threshold': np.pi / 8,
              'method': 'resultant_vector', 'pcs_only': False},
             r'Fraction of all cells with tuning within $\frac{\pi}{8}$ of reward'),
        ]
    elif metrics == 'place field':
        activity_metrics = [
            (place.place_cell_percentage, {}, 'Place cell fraction'),
            (place.sensitivity, {}, 'Place field sensitivity'),
            (place.specificity, {}, 'Place field specificity'),
            (place.place_field_width, {}, 'Place field width'),
            (place.circular_variance, {}, 'Circular variance (all ROIs)'),
            (place.sparsity, {}, 'Single-cell sparsity'),
        ]
    elif metrics == 'transients':
        channels = set(expt_grp.args['channel'] for expt_grp in expt_grps)
        assert len(channels) == 1
        channel = list(channels)[0]
        labels = set(expt_grp.args['imaging_label'] for expt_grp in expt_grps)
        assert len(labels) == 1
        label = list(labels)[0]
        activity_metrics = [
            (ia.population_activity_new,
             {'channel': channel, 'label': label, 'stat': 'amplitude'},
             'Transient amplitude'),
            (ia.population_activity_new,
             {'channel': channel, 'label': label, 'stat': 'duration'},
             'Transient duration'),
            (ia.population_activity_new,
             {'channel': channel, 'label': label, 'stat': 'responseMagnitude'},
             'Transient responseMagnitude'),
            (ia.population_activity_new,
             {'channel': channel, 'label': label, 'stat': 'norm transient auc2'},
             'Transient norm AUC2'),
            (ia.population_activity_new,
             {'channel': channel, 'label': label, 'stat': 'frequency'},
             'Transient frequency'),
        ]
    else:
        raise ValueError("Unrecognized 'metrics' argument")

    figs, axs, _ = plotting.layout_subplots(
        len(behavior_metrics) * len(activity_metrics), 2, 3,
        sharex=False, rasterized=rasterized)

    if fig_title:
        title_text = fig_title + '\ngroupby={}, colorby={}'.format(
            groupby, colorby)
    else:
        title_text = 'groupby={}, colorby={}'.format(groupby, colorby)

    for fig in figs:
        fig.suptitle(title_text)

    data_to_save = {}
    for ax_idx, ax, (activity, behavior) in it.izip(
            it.count(), axs, it.product(activity_metrics, behavior_metrics)):
        df = plot_paired_metrics(
            expt_grps, roi_filters=roi_filters, ax=ax,
            first_metric_fn=activity[0], second_metric_fn=behavior[0],
            first_metric_kwargs=activity[1], second_metric_kwargs=behavior[1],
            first_metric_label=activity[2], second_metric_label=behavior[2],
            groupby=groupby, colorby=colorby, **plot_kwargs)
        data_to_save[activity[2] + ' vs. ' + behavior[2]] = df

        # HACK: Some of the stats can be hard to read, add some room at the top
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        ax.set_ylim(top=ax.get_ylim()[1] + 0.2 * y_range)

        if save_data:
            # Every 6 axs save data if needed
            if ax_idx % 6 == 5:
                fig = figs[ax_idx / 6]
                misc.save_data(
                    data_to_save, fig=fig, label='hidden_correlation',
                    method=save_data)
                data_to_save = {}

    # If the last page wasn't full, save now
    if save_data and ax_idx % 6 != 5:
        fig = figs[-1]
        misc.save_data(
            data_to_save, fig=fig, label='hidden_correlation',
            method=save_data)

    return figs


def distance_to_reward_stability_figure(
        expt_grps, roi_filters=None, groupby=(('first_roi',),),
        colorby=('expt_grp',), rasterized=False, x_bins=3, z_score=False,
        shuffle_colors=False, fig_title='', save_data=False, scatter_kwargs={},
        bar_kwargs={}):

    if roi_filters is None:
        roi_filters = [None] * len(expt_grps)

    # All metrics are (function, activity_kwargs, plot_kwargs, label) tuples
    # Make sure that all labels are unique!
    rewardness_metrics = [
        (place.centroid_to_position_distance,
         {'positions': 'reward', 'return_abs': True}, {},
         'Centroid to reward distance'),
        (place.mean_resultant_vector_to_position_angle,
         {'positions': 'reward'}, {},
         'Activity centroid to reward angle (all ROIs)'),
        (place.mean_resultant_vector_to_position_angle,
         {'positions': 'reward', 'pcs_only': True}, {},
         'Activity centroid to reward angle (PCs only)')
    ]

    stability_metrics = [
        (place.place_field_correlation,
         {'activity_filter': 'pc_both', 'shuffle': False}, {},
         'Place field correlation (PC both)'),
        (place.centroid_shift, {'return_abs': True, 'shuffle': False}, {},
         'Centroid shift'),
        (place.activity_centroid_shift,
         {'activity_filter': 'pc_both', 'circ_var_pcs': True,
          'shuffle': False}, {}, 'Activity centroid shift (PC both)'),
        (place.overlap,
         {'activity_method': 'frequency', 'activity_filter': 'active_both',
          'shuffle': False}, {'s': 5}, 'Frequency overlap (Active both)'),
        (place.overlap,
         {'activity_method': 'is place cell', 'activity_filter': 'pc_either',
          'shuffle': False}, {'s': 5}, 'Place cell overlap (PC either)'),
        (place.overlap,
         {'activity_method': 'norm transient auc', 'activity_filter': None,
          'shuffle': False}, {'s': 5},
         'Place cell overlap (norm transient auc)')
    ]

    axs_per_fig = (len(expt_grps) + 1) * len(rewardness_metrics)
    figs, axs, _ = plotting.layout_subplots(
        n_plots=len(stability_metrics) * axs_per_fig,
        rows=len(expt_grps) + 1, cols=len(rewardness_metrics),
        sharex=False, rasterized=rasterized)

    if fig_title:
        title_text = fig_title + '\ngroupby={}, colorby={}'.format(
            groupby, colorby)
    else:
        title_text = 'groupby={}, colorby={}'.format(groupby, colorby)

    for fig in figs:
        fig.subplots_adjust(hspace=0.3)
        fig.suptitle(title_text)

    data_to_save = {}
    ax_idx = 0
    fig_idx = 0
    for stability in stability_metrics:
        for expt_grp, roi_filter in zip(expt_grps, roi_filters):
            data_to_save[expt_grp.label()] = {}
            for rewardness in rewardness_metrics:
                current_plot_kwargs = copy(scatter_kwargs)
                current_plot_kwargs.update(rewardness[2])
                current_plot_kwargs.update(stability[2])
                df = plot_paired_metrics(
                    [expt_grp], roi_filters=[roi_filter], ax=axs[ax_idx],
                    first_metric_fn=rewardness[0],
                    second_metric_fn=stability[0],
                    first_metric_kwargs=rewardness[1],
                    second_metric_kwargs=stability[1],
                    first_metric_label=rewardness[3],
                    second_metric_label=stability[3],
                    groupby=groupby, colorby=colorby,
                    z_score=z_score, shuffle_colors=shuffle_colors,
                    **current_plot_kwargs)
                ax_idx += 1
                data_to_save[expt_grp.label()][rewardness[3] + ' vs. ' + stability[3]] = df
            plotting.right_label(axs[ax_idx - 1], expt_grp.label())

        for rewardness in rewardness_metrics:
            current_plot_kwargs = copy(bar_kwargs)
            plot_paired_metrics(
                expt_grps, roi_filters=roi_filters, ax=axs[ax_idx],
                first_metric_fn=rewardness[0], second_metric_fn=stability[0],
                first_metric_kwargs=rewardness[1],
                second_metric_kwargs=stability[1],
                first_metric_label=rewardness[3],
                second_metric_label=stability[3],
                plot_method='grouped_bar', groupby=groupby, colorby=colorby,
                z_score=z_score, shuffle_colors=shuffle_colors,
                post_pair_plotby=[rewardness[3]], x_bins=x_bins, **current_plot_kwargs)
            ax_idx += 1

        if save_data:
            misc.save_data(data_to_save, fig=figs[fig_idx],
                         label='hidden_reward_pc_stability', method=save_data)
            data_to_save = {}
        fig_idx += 1

    return figs


def distance_to_reward_recurrence_figure(
        expt_grps, roi_filters=None, rasterized=False, save_data=False,
        **plot_kwargs):

    if roi_filters is None:
        roi_filters = [None] * len(expt_grps)

    figs = []

    fig, axs = plt.subplots(
        1, len(expt_grps), subplot_kw={'rasterized': rasterized},
        squeeze=False)

    figs.append(fig)

    for ax, expt_grp, roi_filter in it.izip(axs[0], expt_grps, roi_filters):
        place.plot_activity_versus_place_coding(
            ax, expt_grp, place.mean_resultant_vector_to_position_angle,
            fn_kwargs={'positions': 'reward'}, roi_filter=roi_filter,
            z_score=False, **plot_kwargs)
        ax.set_title(expt_grp.label())

    axs[0, 0].set_ylabel('Activity centroid to reward angle')

    fig.suptitle(
        'Of the cells that are/are not place cells today, \n' +
        'what was/is/will be their rewardness yesterday/today/tomorrow?')

    fig, axs = plt.subplots(
        1, len(expt_grps), subplot_kw={'rasterized': rasterized},
        squeeze=False)

    figs.append(fig)

    for ax, expt_grp, roi_filter in it.izip(axs[0], expt_grps, roi_filters):
        place.plot_recur_vs_non_recur_activity(
            ax, expt_grp, place.mean_resultant_vector_to_position_angle,
            fn_kwargs={'positions': 'reward'}, roi_filter=roi_filter,
            z_score=False, **plot_kwargs)
        ax.set_title(expt_grp.label())

    axs[0, 0].set_ylabel('Activity centroid to reward angle')

    fig.suptitle(
        'Of all the place cells today, what is the rewardness of the ones ' +
        '\nthat were/were not also place cells yesterday/tomorrow?')

    return figs


def rewardness_shift_figure(
        expt_grps, roi_filters=None, rasterized=False, save_data=False,
        **plot_kwargs):
    """Do place cells shift towards or away from the reward?"""

    if roi_filters is None:
        roi_filters = [None] * len(expt_grps)

    fig, axs = plt.subplots(
        1, 2, subplot_kw={'rasterized': rasterized}, squeeze=False)

    # (ax, kwargs, label)
    metrics = [
        (axs[0, 0],
         {'method': 'resultant', 'positions': 'reward', 'pcs_only': True},
         'Resultant vector to reward shift (circ var PCs only)'),
        (axs[0, 1],
         {'method': 'centroid', 'positions': 'reward', 'return_abs': True},
         'Centroid to reward shift')]

    for ax, kwargs, label in metrics:
            plot_metric(
                ax, expt_grps, place.distance_to_position_shift,
                roi_filters=roi_filters, activity_kwargs=kwargs,
                activity_label=label, plot_method='hist', uniform_bins=True,
                normed=True, plot_mean=True, bins=20, **plot_kwargs)

    fig.suptitle('Change in distance to reward (negative is closer)')

    return fig


def population_vector_stability_by_bin_and_condition(
        expt_grps, roi_filters=None, groupby=None, rasterized=False,
        save_data=False, fig_title='', activity_filter='pc_both',
        **plot_kwargs):

    conditions = pd.concat(
        [grp.dataframe(
            grp, include_columns=['second_condition'])['second_condition']
            for grp in expt_grps])
    conditions = sorted(set(conditions))

    fig, axs = plt.subplots(
        2, len(conditions), subplot_kw={'rasterized': rasterized})

    metric_kwargs = [
        {'method': 'corr', 'activity_filter': activity_filter,
         'shuffle': True, 'reward_at_zero': True},
        {'method': 'angle', 'activity_filter': activity_filter,
         'reward_at_zero': True}]

    for ax_row, kwargs in zip(axs, metric_kwargs):
        for ax, condition in zip(ax_row, conditions):
            plot_metric(
                ax, expt_grps, metric_fn=place.population_vector_correlation,
                groupby=groupby, plotby=['position_bin_index'],
                plot_method='line', roi_filters=roi_filters,
                activity_kwargs=kwargs, plot_shuffle=False,
                activity_label='PV correlation',
                filter_fn=lambda df: df['second_condition'] == condition,
                filter_columns=['second_condition'], label_every_n=20,
                **plot_kwargs)

    for ax, condition in zip(axs[0, :], conditions):
        ax.set_title('Condition {}'.format(condition))
    for ax in axs[1, :]:
        ax.set_title('')
    for ax in it.chain(axs[0, 1:], axs[1, 1:]):
        ax.set_ylabel('')
    for ax in axs[0, :]:
        ax.set_xlabel('')
    for ax in list(axs.flat)[1:]:
        ax.get_legend().set_visible(False)
    axs[0, 0].set_ylabel('PV correlation (corr)')
    axs[1, 0].set_ylabel('PV correlation (angle)')

    y_lim = [np.inf, -np.inf]
    for ax in axs[0, :]:
        y_lim[0] = min(ax.get_ylim()[0], y_lim[0])
        y_lim[1] = max(ax.get_ylim()[1], y_lim[1])
    for ax in axs[0, :]:
        ax.set_ylim(y_lim)

    y_lim = [np.inf, -np.inf]
    for ax in axs[1, :]:
        y_lim[0] = min(ax.get_ylim()[0], y_lim[0])
        y_lim[1] = max(ax.get_ylim()[1], y_lim[1])
    for ax in axs[1, :]:
        ax.set_ylim(y_lim)

    sns.despine(fig)
    fig.suptitle(fig_title)

    return fig


def hidden_rewards_place_field_distribution_density_figure(
        expt_grps, **plot_kwargs):

    grp_dfs = [lab.ExperimentGroup.dataframe(
        expt_grp, include_columns=[
            'condition_day_session', 'condition', 'exposure'])
        for expt_grp in expt_grps]

    conditions = sorted(set(
        condition for grp_df in grp_dfs for condition in grp_df['condition']))
    days = sorted(set(
        condition for grp_df in grp_dfs for condition in grp_df['exposure']))

    fig, axs = plt.subplots(
        len(conditions), len(days), sharex=True, sharey=True,
        figsize=(15, 8), squeeze=False)

    for condition, axs_row in zip(conditions, axs):
        for day, ax in zip(days, axs_row):
            new_expt_grps = []
            for expt_grp, grp_df in zip(expt_grps, grp_dfs):
                expt_list = list(grp_df[(grp_df['condition'] == condition) &
                                        (grp_df['exposure'] == day)]['expt'])
                new_expt_grps.append(expt_grp.subGroup(expt_list))
            plotting.analysis_plotting.plot_metric(
                ax, new_expt_grps, plot_method='kde',
                metric_fn=place.mean_resultant_vector_to_position_angle,
                activity_kwargs={'positions': 'reward', 'pcs_only': True,
                                 'method': 'angle_difference'}, **plot_kwargs)

    ax.set_xlim(-np.pi, np.pi)

    for ax in axs[:, 1:].flat:
        ax.set_ylabel('')
    for ax in axs[:-1].flat:
        ax.set_xlabel('')
    for ax in axs[-1]:
        ax.set_xlabel('Distance from reward')
    for ax in axs[1:].flat:
        ax.set_title('')

    for ax, day in zip(axs[0], days):
        ax.set_title('Day {}'.format(day))
    for ax, condition in zip(axs[:, -1].flat, conditions):
        plotting.right_label(ax, str(condition))

    for ax in list(axs.flat)[1:]:
        ax.legend().remove()

    fig.suptitle('Place field distribution by condition/day')

    return fig


def distance_thresholded_stability_vs_performance(
        paired_exptGrps, roi_filters=None, title='',
        **plot_kwargs):

    behavior_metrics = [
        (ra.fraction_of_laps_rewarded,
         {}, 'Fraction of laps rewarded'),
        (ra.fraction_licks_in_reward_zone,
         {}, 'Fraction of licks in reward zone'),
        (ra.lick_to_reward_distance,
         {}, 'Lick to reward distance')]

    stability_metrics = [
        (place.place_field_correlation,
         {'activity_filter': 'pc_both', 'shuffle': False},
         'Place field correlation'),
        (place.centroid_shift, {'return_abs': True, 'shuffle': False},
         'Centroid shift'),
        (place.activity_centroid_shift, {'shuffle': False},
         'Activity centroid shift (all ROIs)'),
        (place.activity_centroid_shift,
         {'activity_filter': 'pc_both', 'circ_var_pcs': True,
          'shuffle': False}, 'Activity centroid shift (PCs only)'),
        (place.overlap, {'activity_method': 'frequency', 'shuffle': False},
         'Frequency overlap'),
        (place.overlap, {'activity_method': 'norm transient auc',
         'activity_filter': None, 'shuffle': False},
         'Activity overlap (norm transient AUC)')]

    thresholds = [np.pi / 32., np.pi / 16, np.pi / 8, np.pi / 4., np.pi / 2.]

    figs = []

    for behavior_metric in behavior_metrics:
        for stability_metric in stability_metrics:
            figs.append(af.thresholded_metric_vs_metric_figure(
                paired_exptGrps, stability_metric[0], behavior_metric[0],
                filter_metric=place.mean_resultant_vector_to_position_angle,
                thresholds=thresholds, roi_filters=roi_filters,
                x_metric_kwargs=stability_metric[1],
                y_metric_kwargs=behavior_metric[1],
                filter_metric_kwargs={'positions': 'reward'},
                xlabel=stability_metric[2], ylabel=behavior_metric[2],
                plot_method='scatter', groupby=[['second_expt']],
                colorby=['expt_grp'], filter_on=('first_roi',),
                title=title, print_stats=True, stats_by_color=True,
                plotEqualLine=False, **plot_kwargs))

    return figs


def reward_cell_properties(
        expt_grps, roi_filters=None, colors=None, reward_threshold=0.1,
        **plot_kwargs):
    """Compare properties of reward cells and non-reward cells between groups.

    Parameters
    ----------
    reward_threshold : float on [0, 0.5)
        Distance (in normalized units) a place cell's centroid must be from the
        reward to be considered a 'reward cell'.
    plot_kwargs : optional
        Additional arguments are passed to the plotting function.

    """
    if roi_filters is None:
        roi_filters = [None] * len(expt_grps)

    if colors is None:
        colors = plotting.color_cycle()

    fig, axs = plt.subplots(2, 4, figsize=(15, 8), gridspec_kw={'wspace': 0.3})
    sns.despine(fig)

    title_text = ''
    for key, value in plot_kwargs.iteritems():
        title_text += '{}: {}, '.format(key, value)
    title_text = title_text.rstrip(', ')
    fig.suptitle('Reward cell properties; threshold: {}\n{}'.format(
        reward_threshold, title_text))

    orig_roi_filters = []
    new_grps, new_roi_filters, new_colors = [], [], []
    for expt_grp, roi_filter, color in zip(expt_grps, roi_filters, colors):
        orig_roi_filters.extend([roi_filter] * 3)

        place_cell_filter = expt_grp.pcs_filter(roi_filter=None)
        reward_cell_filter = filters.reward_cell_filter(
            expt_grp, roi_filter=None, threshold=reward_threshold)

        reward_cell_grp = copy(expt_grp)
        reward_cell_grp.label('{}: reward'.format(expt_grp.label()))
        new_grps.append(reward_cell_grp)
        new_roi_filters.append(lab.misc.filter_intersection(
            [reward_cell_filter, roi_filter]))

        non_reward_cell_grp = copy(expt_grp)
        non_reward_cell_grp.label('{}: non-reward'.format(expt_grp.label()))
        new_grps.append(non_reward_cell_grp)
        new_roi_filters.append(lab.misc.filter_intersection(
            [place_cell_filter, lab.misc.invert_filter(reward_cell_filter),
             roi_filter]))

        non_place_cell_grp = copy(expt_grp)
        non_place_cell_grp.label('{}: non-spatial'.format(expt_grp.label()))
        new_grps.append(non_place_cell_grp)
        new_roi_filters.append(lab.misc.filter_intersection(
            [lab.misc.invert_filter(place_cell_filter), roi_filter]))

        new_colors.extend(sns.light_palette(color, 4, reverse=True)[:3])

    plot_metric(
        axs[0, 0], new_grps,
        lab.analysis.imaging_analysis.population_activity_new,
        roi_filters=new_roi_filters, colors=new_colors,
        activity_kwargs={'stat': 'norm transient auc2'},
        activity_label='transient auc rate', **plot_kwargs)
    axs[0, 0].set_xlabel('')

    plot_metric(
        axs[0, 1], new_grps,
        lab.analysis.imaging_analysis.transients,
        roi_filters=new_roi_filters, colors=new_colors,
        activity_kwargs={'key': 'duration'},
        activity_label='transient duration', **plot_kwargs)
    axs[0, 1].set_xlabel('')

    plot_metric(
        axs[0, 2], new_grps,
        lab.analysis.imaging_analysis.population_activity_new,
        roi_filters=new_roi_filters, colors=new_colors,
        activity_kwargs={'stat': 'frequency'},
        activity_label='transient frequency', **plot_kwargs)
    axs[0, 2].set_xlabel('')

    plot_metric(
        axs[0, 3], new_grps,
        lab.analysis.imaging_analysis.transients,
        roi_filters=new_roi_filters, colors=new_colors,
        activity_kwargs={'key': 'max_amplitude'},
        activity_label='transient amplitude', **plot_kwargs)
    axs[0, 3].set_xlabel('')

    plot_metric(
        axs[1, 0], new_grps,
        lab.analysis.place_cell_analysis.place_field_gain,
        roi_filters=new_roi_filters, colors=new_colors,
        activity_kwargs={},
        activity_label='place field gain', **plot_kwargs)
    axs[1, 0].set_xlabel('')

    plot_metric(
        axs[1, 1], new_grps,
        lab.analysis.place_cell_analysis.circular_variance,
        roi_filters=new_roi_filters, colors=new_colors,
        activity_kwargs={},
        activity_label='circular variance', **plot_kwargs)
    axs[1, 1].set_xlabel('')

    plot_metric(
        axs[1, 2], new_grps,
        lab.analysis.place_cell_analysis.place_field_width,
        roi_filters=new_roi_filters, colors=new_colors,
        activity_kwargs={},
        activity_label='place field width', **plot_kwargs)
    axs[1, 2].set_xlabel('')

    plot_metric(
        axs[1, 3], new_grps,
        lab.classes.classes.ExperimentGroup.filtered_rois,
        roi_filters=new_roi_filters, colors=new_colors,
        activity_kwargs=[
            {'include_roi_filter': roi_filter}
            for roi_filter in orig_roi_filters],
        activity_label='ROI fraction', **plot_kwargs)
    axs[1, 3].set_xlabel('')

    for ax in list(axs.flat)[1:]:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_visible(False)

    return fig
