"""Analysis functions for place field data."""

import numpy as np
import math
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
try:
    from bottleneck import nanmean
except ImportError:
    from numpy import nanmean
from pycircstat.descriptive import _complex_mean
from cmath import polar
import itertools as it
import warnings as wa
import cPickle as pickle
from copy import deepcopy
from random import sample
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr, linregress

import lab
from ..classes.place_cell_classes import pcExperimentGroup
import behavior_analysis as ba
import imaging_analysis as ia
from calc_activity import calc_activity
import filters
from .. import plotting
from ..misc import stats, memoize
from .. import misc
from ..classes import exceptions as exc
from ..misc.analysis_helpers import rewards_by_condition


@memoize
def sensitivity(
        exptGrp, roi_filter=None, includeFrames='running_only'):
    """
    Fraction of complete forward passes through the place field that trigger
    a significant calcium transient

    returns a place_cell_df
    """
    pfs_n = exptGrp.pfs_n(roi_filter=roi_filter)

    data_list = []
    n_wide_pfs = 0
    for expt in exptGrp:
        pfs = pfs_n[expt]
        if includeFrames == 'running_only':
            imaging_label = exptGrp.args['imaging_label']
            if imaging_label is None:
                imaging_label = expt.most_recent_key(
                    channel=exptGrp.args['channel'])
        for trial_idx, trial in enumerate(expt.findall('trial')):
            position = trial.behaviorData(imageSync=True)['treadmillPosition']
            transients = trial.transientsData(
                roi_filter=roi_filter, channel=exptGrp.args['channel'],
                label=exptGrp.args['imaging_label'],
                demixed=exptGrp.args['demixed'])
            if includeFrames == 'running_only':
                with open(expt.placeFieldsFilePath(), 'rb') as f:
                    p = pickle.load(f)
                running_kwargs = p[imaging_label][
                    'demixed' if exptGrp.args['demixed'] else 'undemixed'][
                    'running_kwargs']
                running_frames = ba.runningIntervals(
                    trial, returnBoolList=True, **running_kwargs)
            else:
                raise ValueError
            rois = expt.rois(
                roi_filter=roi_filter, channel=exptGrp.args['channel'],
                label=exptGrp.args['imaging_label'])

            assert len(rois) == len(pfs)
            assert len(rois) == len(transients)

            for roi_transients, roi_pfs, roi in it.izip(transients, pfs, rois):
                onsets = roi_transients['start_indices'].tolist()
                onsets = [onset for onset in onsets if running_frames[onset]]

                # At the moment, pfs wider than 0.5 will not be accurately
                # counted. This could potentially be fixed if needed.
                for pf in reversed(roi_pfs):
                    pf_len = pf[1] - pf[0]
                    if pf_len < 0:
                        pf_len += 1.
                    if pf_len > 0.5:
                        n_wide_pfs += 1
                        roi_pfs.remove(pf)

                if not len(roi_pfs):
                    # Sensitivity will be nan if roi has no place fields,
                    # or if all place fields were wider than 0.5
                    data_dict = {
                        'trial': trial, 'roi': roi, 'value': np.nan}
                    data_list.append(data_dict)
                    continue

                passes = 0
                hits = 0.
                for pf in roi_pfs:
                    current_frame = 0
                    while current_frame < expt.num_frames():
                        if pf[0] < pf[1]:
                            entries = np.argwhere(
                                (position[current_frame:] >= pf[0]) *
                                (position[current_frame:] < pf[1]))
                        else:
                            entries = np.argwhere(
                                (position[current_frame:] >= pf[0]) +
                                (position[current_frame:] < pf[1]))
                        if not len(entries):
                            break
                        next_entry = current_frame + entries[0, 0]

                        if pf[0] < pf[1]:
                            exits = np.argwhere(
                                (position[next_entry + 1:] >= pf[1]) +
                                (position[next_entry + 1:] < pf[0]))
                        else:
                            exits = np.argwhere(
                                (position[next_entry + 1:] >= pf[1]) *
                                (position[next_entry + 1:] < pf[0]))
                        if not len(exits):
                            # NOTE: a trial ending within the placefield does
                            # not count as a pass
                            break
                        next_exit = next_entry + 1 + exits[0, 0]

                        # if not good entry, continue
                        # if not (position[next_entry - 1] - pf[0] < 0 or
                        #         position[next_entry - 1] - pf[0] > 0.5):
                        #     current_frame = next_exit + 1
                        #     continue
                        previous_position = position[next_entry - 1]
                        if (0 <= previous_position - pf[0] < 0.5) or \
                                previous_position - pf[0] < -0.5:
                            current_frame = next_exit
                            continue
                        # if not good exit, continue
                        # if not (position[next_exit + 1] - pf[1] > 0 or
                        #         position[next_exit + 1] - pf[1] < -0.5):
                        #     current_frame = next_exit + 1
                        #     continue

                        next_position = position[next_exit]
                        if (-0.5 < next_position - pf[1] < 0.) or \
                                (next_position - pf[1] > 0.5):
                            current_frame = next_exit
                            continue

                        passes += 1
                        for onset in onsets:
                            if next_entry <= onset < next_exit:
                                hits += 1
                                break
                        current_frame = next_exit

                assert passes
                data_dict = {'trial': trial,
                             'roi': roi,
                             'value': hits / passes}
                data_list.append(data_dict)

    if n_wide_pfs:
        with wa.catch_warnings():
            wa.simplefilter('always')
            wa.warn(
                'Sensitivity not calculated for pf width >0.5: ' +
                '{} pfs skipped'.format(n_wide_pfs))

    return pd.DataFrame(data_list, columns=['trial', 'roi', 'value'])


@memoize
def specificity(
        exptGrp, roi_filter=None, includeFrames='running_only'):
    """
    Fraction of transient onsets that occur in a place field
    """

    pfs_n = exptGrp.pfs_n(roi_filter=roi_filter)

    data_list = []
    for expt in exptGrp:
        pfs = pfs_n[expt]
        for trial_idx, trial in enumerate(expt.findall('trial')):
            position = trial.behaviorData(imageSync=True)['treadmillPosition']
            transients = trial.transientsData(
                roi_filter=roi_filter, channel=exptGrp.args['channel'],
                label=exptGrp.args['imaging_label'],
                demixed=exptGrp.args['demixed'])
            if includeFrames == 'running_only':
                with open(expt.placeFieldsFilePath(), 'rb') as f:
                    p = pickle.load(f)
                running_kwargs = p[exptGrp.args['imaging_label']][
                    'demixed' if exptGrp.args['demixed'] else 'undemixed'][
                    'running_kwargs']
                running_frames = ba.runningIntervals(
                    trial, returnBoolList=True, **running_kwargs)
            rois = expt.rois(
                roi_filter=roi_filter, channel=exptGrp.args['channel'],
                label=exptGrp.args['imaging_label'])

            assert len(rois) == len(pfs)
            assert len(rois) == len(transients)

            for roi_transients, roi_pfs, roi in it.izip(transients, pfs, rois):
                if not len(roi_pfs):
                    continue
                onsets = roi_transients['start_indices'].tolist()
                onsets = [o for o in onsets if running_frames[o]]

                nTransients = 0
                hits = 0.
                for onset in onsets:
                    nTransients += 1
                    onset_position = position[onset]

                    for pf in roi_pfs:
                        if pf[0] < pf[1]:
                            if pf[0] < onset_position < pf[1]:
                                hits += 1
                                break
                        else:
                            if onset_position > pf[0] or \
                                    onset_position < pf[1]:
                                hits += 1
                                break
                if not nTransients:
                    value = np.nan
                else:
                    value = hits / nTransients
                data_dict = {'trial': trial,
                             'roi': roi,
                             'value': value}
                data_list.append(data_dict)
    return pd.DataFrame(data_list, columns=['trial', 'roi', 'value'])


def plotPositionHeatmap(
        exptGrp, roi_filter=None, ax=None, title='', plotting_order='all',
        cbar_visible=True, cax=None, norm=None, rasterized=False, cmap=None,
        show_belt=True, reward_in_middle=False):
    """
    Plot a heatmap of ROI activity at each place bin

    Keyword arguments:
    exptGrp -- pcExperimentGroup containing data to plot
    ax -- axis to plot on
    title -- label for the axis
    cbar_visible -- if False does not show a colorbar
    norm -- one of None, 'individual', an np.array of imagingData to
        determine normalization method
        'individual' scales each ROI individually to the same range
        np.array of imagingData normalizes to the raw imaging data by
        subtracting off the mean and dividing by the std on a per ROI basis,
        similar to a z-score
    reward_in_middle : bool
        If True, move reward to middle of plot

    """

    if ax is None:
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111, rasterized=rasterized)
    else:
        fig = ax.figure

    divider = make_axes_locatable(ax)
    if cbar_visible and cax is None:
        cax = divider.append_axes("right", size="5%", pad=0.05)

    if show_belt:
        if reward_in_middle:
            raise NotImplementedError
        belt_ax = divider.append_axes("bottom", size="5%", pad=0.05)
        exptGrp[0].belt().show(belt_ax, zeroOnLeft=True)

    def roller(data, expt):
        if not reward_in_middle:
            return data

        n_bins = exptGrp.args['nPositionBins']
        mid_bin = int(np.around(n_bins / 2.))
        reward_bin = int(np.around(
            expt.rewardPositions(units='normalized')[0] * n_bins))
        roll_factor = mid_bin - reward_bin

        return np.roll(data, roll_factor, axis=1)

    if plotting_order == 'all':
        roi_data_to_plot = []
        for expt in exptGrp:
            roi_data_to_plot.extend(
                roller(exptGrp.data(roi_filter=roi_filter)[expt], expt))
        data_to_plot = np.array(roi_data_to_plot)
    elif plotting_order == 'place_cells_first':
        place_cells = []
        for expt in exptGrp:
            place_cells.extend(roller(exptGrp.data(
                roi_filter=exptGrp.pcs_filter(roi_filter=roi_filter))[expt],
                expt))
        if len(place_cells) != 0:
            place_cells_array = np.array(place_cells)
            place_cells_array = place_cells_array[
                np.argsort(np.argmax(place_cells_array, axis=1))]

            empty_array = np.empty((1, place_cells_array.shape[1]))
            empty_array.fill(np.nan)
            place_cells_array = np.vstack((place_cells_array, empty_array))

        non_place_cells = []
        for expt in exptGrp:
            expt_filter = misc.filter_intersection(
                [misc.invert_filter(exptGrp.pcs_filter()), roi_filter])
            non_place_cells.extend(roller(
                exptGrp.data(roi_filter=expt_filter)[expt], expt))
        non_place_cells_array = np.array(non_place_cells)
        # non_place_cells_array = non_place_cells_array[
        #     np.argsort(np.argmax(non_place_cells_array, axis=1))]
        if len(place_cells) != 0:
            data_to_plot = np.vstack((place_cells_array, non_place_cells_array))
            n_pcs = place_cells_array.shape[0] - 1
        else:
            data_to_plot = non_place_cells_array
            n_pcs = 0
    elif plotting_order == 'place_cells_only':
        roi_data_to_plot = []
        for expt in exptGrp:
            roi_data_to_plot.extend(roller(exptGrp.data(
                roi_filter=exptGrp.pcs_filter(roi_filter=roi_filter))[expt],
                expt))
        roi_data_to_plot = np.array(roi_data_to_plot)
        data_to_plot = roi_data_to_plot[
            np.argsort(np.argmax(roi_data_to_plot, axis=1))]
        n_pcs = data_to_plot.shape[0]
    elif plotting_order is not None:
        allROIs = exptGrp.allROIs(channel=exptGrp.args['channel'],
                                  label=exptGrp.args['imaging_label'],
                                  roi_filter=None)
        roi_data_to_plot = []
        for roi_tuple in plotting_order:
            (expt, roi_idx) = allROIs[roi_tuple][0]
            roi_data_to_plot.append(roller(
                exptGrp.data()[expt], expt)[roi_idx])
        data_to_plot = np.array(roi_data_to_plot)

    if norm is 'individual':
        # Find the all zero rows, and just put back zeros there
        all_zero_rows = np.where(np.all(data_to_plot == 0, axis=1))[0]
        data_to_plot -= np.amin(data_to_plot, axis=1)[:, np.newaxis]
        data_to_plot /= np.amax(data_to_plot, axis=1)[:, np.newaxis]
        data_to_plot[all_zero_rows] = 0
    elif norm is 'pc_individual_non_pc_grouped':
        data_to_plot[:n_pcs] -= np.amin(
            data_to_plot[:n_pcs], axis=1)[:, np.newaxis]
        data_to_plot[:n_pcs] /= np.amax(
            data_to_plot[:n_pcs], axis=1)[:, np.newaxis]
        data_to_plot[n_pcs + 1:] -= np.amin(data_to_plot[n_pcs + 1:])
        data_to_plot[n_pcs + 1:] /= np.amax(data_to_plot[n_pcs + 1:])
    elif norm is 'all':
        data_to_plot -= np.amin(data_to_plot[np.isfinite(data_to_plot)])
        data_to_plot /= np.amax(data_to_plot[np.isfinite(data_to_plot)])
    elif norm is not None:
        try:
            try_data = data_to_plot - np.nanmean(norm, axis=1)
            try_data /= np.nanstd(norm, axis=1)
        except:
            print "Unable to normalize data"
        else:
            data_to_plot = try_data

    # Set the color scale based on a percentile of the data
    vmin = np.percentile(data_to_plot[np.isfinite(data_to_plot)], 40)
    vmax = np.percentile(data_to_plot[np.isfinite(data_to_plot)], 99)

    im = ax.imshow(
        data_to_plot, vmin=vmin, vmax=vmax, interpolation='none',
        aspect='auto', cmap=cmap)

    if cbar_visible:
        plt.colorbar(im, cax=cax, ticks=[vmin, vmax], format='%.2f')

    ax.set_xlabel('Normalized position')
    ax.set_xticks(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 5))
    ax.set_xticklabels(np.linspace(0, 1, 5))
    ax.set_ylabel('Cell')
    ax.set_title(title)

    # If the cells are sorted, the y labels will be wrong
    plt.setp(ax.get_yticklabels(), visible=False)

    return fig


def plotTuningCurve(
        data, roi, ax=None, polar=False, placeField=None,
        placeFieldColor='red', xlabel_visible=True, ylabel_visible=True,
        error_bars=None, axis_title=None, two_cycles=False, rasterized=False):
    # TODO: WHATEVER CALLS THIS SHOULD JUST PASS IN 1D ARRAYS FOR DATA AND
    # ERROR BARS!   NO NEED TO PASS IN WHOLE 2D ARRAYS?

    """Plot an ROI tuning curve on a normal or polar axis

    Keyword arguments:
    roi -- number of ROI to plot
    ax -- axis to plot on
    polar -- if True, will plot on polar plot
    placeField -- a single element from the normalized identifyPlaceField list,
        corresponding to the roi
    placeFieldColor -- color to shade the placefield
    error_bars -- if not None, plots error bounds on tuning curve, should be
        same shape as data
    axis_title -- title used to label the axis
    two_cycles -- if True, plots two identical cycles, ignored for polar plots

    """

    if ax is None and not polar:
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111)
    elif ax is None and polar:
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111, polar=True)

    if error_bars is not None:
        assert data.shape == error_bars.shape, \
            'Data and error bars shape mismatch'

    if not polar:
        if not two_cycles:
            x_range = np.linspace(0, 1, data.shape[1])
            ax.plot(x_range, data[roi], rasterized=rasterized)
        else:
            x_range = np.linspace(0, 2, data.shape[1] * 2)
            double_data = np.ma.hstack([data[roi]] * 2)
            ax.plot(x_range, double_data, rasterized=rasterized)
        if error_bars is not None:
            if not two_cycles:
                ax.fill_between(
                    x_range, data[roi], data[roi] + error_bars[roi],
                    facecolor='gray', edgecolor='none', alpha=0.2,
                    rasterized=rasterized)
            else:
                double_err = np.ma.hstack([error_bars[roi]] * 2)
                ax.fill_between(
                    x_range, double_data, double_data + double_err,
                    facecolor='gray', edgecolor='none', alpha=0.2,
                    rasterized=rasterized)
        if placeField is not None:
            for field in placeField:
                if field[0] <= field[1]:
                    ax.fill_between(
                        x_range, data[roi] if not two_cycles else double_data,
                        where=np.logical_and(
                            x_range % 1 >= field[0], x_range % 1 <= field[1]),
                        facecolor=placeFieldColor, alpha=0.5,
                        rasterized=rasterized)
                else:
                    ax.fill_between(
                        x_range, data[roi] if not two_cycles else double_data,
                        where=np.logical_or(
                            x_range % 1 >= field[0], x_range % 1 <= field[1]),
                        facecolor=placeFieldColor, alpha=0.5,
                        rasterized=rasterized)
        ax.set_xlim((x_range[0], x_range[-1]))
        ax.set_yticks([0, np.round(ax.get_ylim()[1], 2)])

    else:
        theta_range = np.linspace(0, 2 * np.pi, data.shape[1])
        ax.plot(theta_range, data[roi], rasterized=rasterized)
        if error_bars is not None:
            ax.fill_between(
                theta_range, data[roi], data[roi] + error_bars[roi],
                facecolor='gray', edgecolor='gray', alpha=0.2,
                rasterized=rasterized)

        ax.set_xticks([0, np.pi / 2., np.pi, 3 * np.pi / 2.])
        x_ticks = ax.get_xticks()
        x_tick_labels = [str(x / 2 / np.pi) for x in x_ticks]
        ax.set_xticklabels(x_tick_labels)
        if placeField:
            for field in placeField:
                # Convert to polar coordinates
                field = [x * 2 * np.pi for x in field]
                if field[0] <= field[1]:
                    ax.fill_between(
                        theta_range, data[roi], where=np.logical_and(
                            theta_range >= field[0], theta_range <= field[1]),
                        facecolor=placeFieldColor, alpha=0.5,
                        rasterized=rasterized)
                else:
                    ax.fill_between(
                        theta_range, data[roi], where=np.logical_or(
                            theta_range >= field[0], theta_range <= field[1]),
                        facecolor=placeFieldColor, alpha=0.5,
                        rasterized=rasterized)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.text(.86, .86, round(ax.get_ylim()[1], 3), transform=ax.transAxes,
                va='bottom', size=10)

    if not xlabel_visible:
        plt.setp(ax.get_xticklabels(), visible=False)
    else:
        ax.set_xlabel('Normalized position')

    if ylabel_visible:
        ax.set_ylabel(r'Average $\Delta$F/F')

    ax.set_title(axis_title)


def plotImagingData(
        roi_tSeries, ax=None, roi_transients=None, position=None,
        placeField=None, imaging_interval=1, xlabel_visible=True,
        ylabel_visible=True, right_label=False, placeFieldColor='red',
        transients_color='r', title='', rasterized=False, **plot_kwargs):

    if ax is None:
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111, rasterized=rasterized)

    x_range = np.arange(len(roi_tSeries)) * imaging_interval

    ax.plot(x_range, roi_tSeries, rasterized=rasterized, **plot_kwargs)
    if roi_transients is not None:
        for start, stop in zip(roi_transients['start_indices'],
                               roi_transients['end_indices']):
            if not np.isnan(start) and not np.isnan(stop):
                ax.plot(x_range[start:stop + 1], roi_tSeries[start:stop + 1],
                        color=transients_color, rasterized=rasterized)

    if placeField:
        # if behav data recording is shorter than imaging recording:
        x_range = x_range[:position.shape[0]]
        position = position[:x_range.shape[0]]
        yl = ax.get_ylim()
        for interval in placeField:
            if interval[0] <= interval[1]:
                ax.fill_between(
                    x_range, yl[0], yl[1],
                    where=np.logical_and(position >= interval[0],
                                         position <= interval[1]),
                    facecolor=placeFieldColor,
                    alpha=0.5,
                    rasterized=rasterized)
            else:
                ax.fill_between(
                    x_range, yl[0], yl[1],
                    where=np.logical_or(position >= interval[0],
                                        position <= interval[1]),
                    facecolor=placeFieldColor,
                    alpha=0.5,
                    rasterized=rasterized)
        ax.set_ylim(yl)

    if not xlabel_visible:
        plt.setp(ax.get_xticklabels(), visible=False)
    else:
        ax.set_xlabel('Time(s)')

    yl = ax.get_ylim()
    ax.set_yticks((0, yl[1]))
    ax.set_yticklabels(('', str(yl[1])))
    ax.tick_params(axis='y', direction='out')
    ax.set_ylim(yl)
    if not ylabel_visible:
        pass
        # plt.setp(ax.get_yticklabels(), visible=False)
    else:
        ax.set_ylabel(r'Mean $\Delta$F/F')

    if right_label:
        plotting.right_label(ax, title)
    else:
        ax.set_title(title)


def plotTransientVectors(exptGrp, roi_idx, ax, color='k', mean_color='r',
                         mean_zorder=1):
    """Plot running-related transients on a polar axis weighted by occupancy
    Accepts a single experiment pcExperimentGroup"""

    with open(exptGrp[0].placeFieldsFilePath(), 'rb') as f:
        pfs = pickle.load(f)
    demixed_key = 'demixed' if exptGrp.args['demixed'] else 'undemixed'
    imaging_label = exptGrp.args['imaging_label'] \
        if exptGrp.args['imaging_label'] is not None \
        else exptGrp[0].most_recent_key(channel=exptGrp.args['channel'])
    true_values = pfs[imaging_label][demixed_key]['true_values'][roi_idx]
    # true_counts = pfs[imaging_label][demixed_key]['true_counts'][roi_idx]
    true_counts = exptGrp[0].positionOccupancy()
    bins = 2 * np.pi * np.arange(0, 1, .01)

    magnitudes = []
    angles = []
    for pos_bin in xrange(len(true_values)):
        for v in xrange(int(true_values[pos_bin])):
            angles.append(bins[pos_bin])
            magnitudes.append(1. / true_counts[pos_bin])

    magnitudes /= np.amax(magnitudes)

    for a, m in zip(angles, magnitudes):
        ax.arrow(a, 0, 0, m, length_includes_head=True, color=color,
                 head_width=np.amin([0.1, m]), head_length=np.amin([0.1, m]),
                 zorder=2)
    p = polar(_complex_mean(bins, true_values / true_counts))
    mean_r = p[0]
    mean_angle = p[1]

    ax.arrow(mean_angle, 0, 0, mean_r, length_includes_head=True,
             color=mean_color, head_width=0, head_length=0, lw=1,
             zorder=mean_zorder)

    ax.set_xticklabels([])
    ax.set_yticklabels([])


def plotPosition(
        trial, ax=None, placeFields=None, placeFieldColors=None, polar=False,
        trans_roi_filter=None, trans_marker_size=10, running_trans_only=False,
        rasterized=False, channel='Ch2', label=None, demixed=False,
        behaviorData=None, trans_kwargs=None, position_kwargs=None,
        nan_lap_transitions=True):
    """Plot position over time

    Keyword arguments:
    trial -- an Experiment/Trial object to extract data from, if an Experiment
        is passed in, analyze the first Trial
    ax -- axes to plot on
    placeFields -- output from identifyPlaceFields, shades placefields, not
        sorted, so sort beforehand if desired
    placeFieldColors -- colors to use for each shaded placefield, should have
        length equal to number of place cells
    polar -- plot on polar axis
    trans_roi_filter -- either None or an roi_filter to mark transient peak
        times on the plot.  Filter must return 1 ROI!
    channel, label, demixed -- used to determine which transients to plot
    behaviorData -- optionally pass in imageSync'd behaviorData so it doesn't
        need to be reloaded
    nan_lap_transitions -- If True, NaN the last value before each reset.
        Should allow for plotting as a line instead of a scatter. Only applies
        to non-polar plots.

    """

    if ax is None and not polar:
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111, rasterized=rasterized)
    elif ax is None and polar:
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111, polar=True, rasterized=rasterized)

    if placeFields is not None:
        roi_list = [x for x in range(len(placeFields)) if
                    placeFields[x] != []]
        # roi_list = identifyPlaceCells(placeFields, sort=False)
        if placeFieldColors is None:
            placeFieldColors = [plt.cm.Set1(i) for i in
                                np.linspace(0, .9, len(roi_list))]

    if behaviorData is None:
        bd = trial.behaviorData(imageSync=True)
    else:
        bd = deepcopy(behaviorData)

    if 'treadmillPosition' not in bd:
        raise exc.MissingBehaviorData(' ''treadmillPosition'' not defined')

    position = bd['treadmillPosition']

    imagingInterval = trial.parent.frame_period()

    if trans_roi_filter is not None:
        try:
            trans_indices = trial.parent.transientsData(
                behaviorSync=True, roi_filter=trans_roi_filter,
                label=label)[
                trial.trialNum()][0]['start_indices']
        except TypeError:
            trans_roi_filter = None
        else:
            if running_trans_only:
                try:
                    with open(trial.parent.placeFieldsFilePath(
                            channel=channel), 'r') as f:
                        place_fields = pickle.load(f)
                except (IOError, exc.NoSimaPath, pickle.UnpicklingError):
                    wa.warn('No place fields found')
                    running_intervals = ba.runningIntervals(
                        trial, returnBoolList=False)
                else:
                    if label is None:
                        label = trial.parent.most_recent_key(channel=channel)
                    demix_label = 'demixed' if demixed else 'undemixed'
                    running_kwargs = place_fields[label][demix_label][
                        'running_kwargs']
                    running_intervals = ba.runningIntervals(
                        trial, returnBoolList=False, **running_kwargs)
                running_frames = np.hstack([np.arange(start, end + 1) for
                                            start, end in running_intervals])
                trans_indices = list(set(trans_indices).intersection(
                    running_frames))
        if len(trans_indices) == 0:
            trans_roi_filter = None

    if not polar:
        if position_kwargs is None:
            if nan_lap_transitions:
                position_kwargs = {}
                position_kwargs['linestyle'] = '-'
            else:
                position_kwargs = {}
                position_kwargs['linestyle'] = 'None'
                position_kwargs['marker'] = '.'

        x_range = np.arange(len(position)) * imagingInterval
        pos_copy = position.copy()
        if nan_lap_transitions:
            jumps = np.hstack([np.abs(np.diff(pos_copy)) > 0.4, False])
            pos_copy[jumps] = np.nan
        ax.plot(x_range, pos_copy, rasterized=rasterized, **position_kwargs)

        if placeFields is not None:
            for idx, roi in enumerate(roi_list):
                for interval in placeFields[roi]:
                    if interval[0] <= interval[1]:
                        ax.fill_between(
                            x_range, (idx / float(len(roi_list))),
                            ((idx + 1) / float(len(roi_list))),
                            where=np.logical_and(position >= interval[0],
                                                 position <= interval[1]),
                            facecolor=placeFieldColors[idx], alpha=0.5,
                            rasterized=rasterized)
                    else:
                        ax.fill_between(
                            x_range, (idx / float(len(roi_list))),
                            ((idx + 1) / float(len(roi_list))),
                            where=np.logical_or(position >= interval[0],
                                                position <= interval[1]),
                            facecolor=placeFieldColors[idx], alpha=0.5,
                            rasterized=rasterized)

        if trans_roi_filter is not None:
            ax.plot(x_range[trans_indices], position[trans_indices], 'r*',
                    markersize=trans_marker_size)

        y_ticks = ax.get_yticks()
        ax.set_yticks((y_ticks[0], y_ticks[-1]))

        ax.set_xlabel('Time(s)')
        ax.set_ylabel('Normalized position')

        ax.set_xlim((0, x_range[-1]))

    else:
        if position_kwargs is None:
            position_kwargs = {}
        position *= 2 * np.pi
        r_range = np.linspace(0.2, 1, len(position))
        ax.plot(position, r_range, rasterized=rasterized, **position_kwargs)

        pos = np.linspace(0, 2 * np.pi, 100)

        if placeFields is not None:
                for idx, roi in enumerate(roi_list):
                    for interval in placeFields[roi]:
                        interval = [x * 2 * np.pi for x in interval]
                        if interval[0] <= interval[1]:
                            ax.fill_between(
                                pos, r_range[-1],
                                where=np.logical_and(pos >= interval[0],
                                                     pos <= interval[1]),
                                facecolor=placeFieldColors[idx], alpha=0.5,
                                rasterized=rasterized)
                        else:
                            ax.fill_between(
                                pos, r_range[-1],
                                where=np.logical_or(pos >= interval[0],
                                                    pos <= interval[1]),
                                facecolor=placeFieldColors[idx], alpha=0.5,
                                rasterized=rasterized)

        if trans_roi_filter is not None:
            if trans_kwargs is None:
                trans_kwargs = {}
                trans_kwargs['color'] = 'r'
                trans_kwargs['marker'] = '*'
                trans_kwargs['linestyle'] = 'None'
                trans_kwargs['markersize'] = 10
            ax.plot(position[trans_indices], r_range[trans_indices],
                    **trans_kwargs)

        x_ticks = ax.get_xticks()
        x_tick_labels = [str(x / 2 / np.pi) for x in x_ticks]
        ax.set_xticklabels(x_tick_labels)

        y_ticks = ax.get_yticks()
        ax.set_yticks((y_ticks[0], y_ticks[-1]))

        ax.set_ylabel('Time(s)')
        ax.set_xlabel('Normalized position')
        ax.set_rmax(r_range[-1])


@memoize
def place_field_width(exptGrp, roi_filter=None, belt_length=200):
    """Calculate all place field widths.

    Keyword arguments:
    exptGrp -- pcExperimentGroup to analyze
    belt_length -- length of the belt in cm

    Output: Pandas DataFrame consisting of one value per observation of a
        place field

    """

    pfs_n = exptGrp.pfs_n(roi_filter=roi_filter)
    data_list = []
    for expt in exptGrp:
        try:
            belt_length = expt.belt().length()
        except exc.NoBeltInfo:
            print 'No belt information found for experiment {}.'.format(
                str(expt))
            print 'Using default belt length = {}'.format(str(belt_length))

        rois = expt.rois(
            roi_filter=roi_filter, channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'])

        assert len(rois) == len(pfs_n[expt])

        for roi, pfs in zip(rois, pfs_n[expt]):
            if not len(pfs):
                continue
            for pf_idx, pf in enumerate(pfs):
                if pf[0] <= pf[1]:
                    value = (pf[1] - pf[0]) * belt_length
                else:
                    value = (1 + pf[1] - pf[0]) * belt_length

                data_dict = {'expt': expt,
                             'roi': roi,
                             'value': value}
                data_list.append(data_dict)

    return pd.DataFrame(data_list, columns=['expt', 'roi', 'value'])


@memoize
def population_activity(
        exptGrp, stat, roi_filter=None, interval='all', dF='from_file',
        running_only=False, non_running_only=False, running_kwargs=None):
    """Calculate and plot the histogram of various activity properties

    Keyword arguments:
    exptGrp -- pcExperimentGroup to analyze
    stat -- activity statistic to calculate, see calc_activity for details
    interval -- 3 options: 'all' = all time, 'pf' = inside place field,
        'non pf' = outside place field
    running_only -- If True, only include running intervals
    dF -- dF method to use on imaging data
    average_trials -- if True, averages across trials

    Output:
    activity -- dict of lists of lists
        access as activity[expt][roiNum][trialNum]

    """

    activity_dfs = []
    if 'pf' in interval:
        pfs_n = exptGrp.pfs_n(roi_filter=roi_filter)

    for expt in exptGrp:
        (nROIs, nFrames, nTrials) = expt.imaging_shape(
            roi_filter=roi_filter, channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'])

        if interval == 'all':
            calc_intervals = None
        elif interval == 'pf' or interval == 'non pf':
            placeFields = pfs_n[expt]
            calc_intervals = np.zeros((nROIs, nFrames, nTrials), 'bool')
            for trial_idx, trial in enumerate(expt.findall('trial')):
                position = trial.behaviorData(
                    imageSync=True)['treadmillPosition']
                for roi_idx, roi in enumerate(placeFields):
                    for pf in roi:
                        if pf[0] <= pf[1]:
                            calc_intervals[roi_idx, np.logical_and(
                                position >= pf[0], position <= pf[1]),
                                trial_idx] = True
                        else:
                            calc_intervals[roi_idx, np.logical_or(
                                position >= pf[0], position <= pf[1]),
                                trial_idx] = True

            if interval == 'non pf':
                calc_intervals = ~calc_intervals
        else:
            calc_intervals = interval

        activity_dfs.append(ia.population_activity(
            exptGrp.subGroup([expt]), stat=stat,
            channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'], roi_filter=roi_filter,
            interval=calc_intervals, dF=dF, running_only=running_only,
            non_running_only=non_running_only))

    return pd.concat(activity_dfs)


def place_field_bins(expt_grp, roi_filter=None, n_bins=None):

    if n_bins is None:
        n_bins = expt_grp.args['nPositionBins']

    pfs = expt_grp.pfs_n(roi_filter=roi_filter)
    rois = expt_grp.rois(roi_filter=roi_filter, channel=expt_grp.args['channel'], label=expt_grp.args['imaging_label'])

    result = []
    for expt in expt_grp:
        assert len(rois[expt]) == len(pfs[expt])
        for roi, roi_pfs in zip(rois[expt], pfs[expt]):
            for start, stop in roi_pfs:
                if start < stop:
                    for bin in range(int(start * n_bins), int(stop * n_bins) + 1):
                        result.append({'expt': expt, 'roi': roi, 'bin': bin, 'value': 1})
                else:
                    for bin in range(0, int(stop * n_bins) + 1):
                        result.append({'expt': expt, 'roi': roi, 'bin': bin, 'value': 1})
                    for bin in range(int(start * n_bins), n_bins):
                        result.append({'expt': expt, 'roi': roi, 'bin': bin, 'value': 1})

    return pd.DataFrame(result, columns=['expt', 'roi', 'bin', 'value'])


def place_field_distribution(
        exptGrp, roi_filter=None, ax=None, normed=False, showBelt=False,
        nBins=None, label=None, color=None):
    """Plot density of place fields on the belt."""

    if showBelt:
        belt = exptGrp[0].belt()
        belt.addToAxis(ax)

    if nBins is None:
        nBins = exptGrp.args['nPositionBins']
    nBins = int(nBins)

    result = np.zeros(nBins, 'int')

    for expt in exptGrp:
        for roi in exptGrp.pfs_n(roi_filter=roi_filter)[expt]:
            for field in roi:
                if field[0] <= field[1]:
                    result[int(field[0] * nBins):int(field[1] * nBins + 1)] \
                        += 1
                else:
                    result[int(field[0] * nBins):] += 1
                    result[:int(field[1] * nBins + 1)] += 1
    if normed:
        result = result.astype('float') / float(np.sum(result))

    if ax:
        if color is None:
            color = lab.plotting.color_cycle().next()

        ax.plot(np.linspace(0, 1, nBins), result, label=label, color=color)
        ax.set_xlabel('Position')
        ax.set_title('Place field distribution')
        if normed:
            ax.set_ylabel('Normalized place field density')
        else:
            ax.set_ylabel('Number of place fields')
        ax.legend(frameon=False, loc='best')

    return result


@memoize
def recurrence_probability(exptGrp, roi_filter=None, circ_var_pcs=False):
    """Generate a plot of the probability of place field recurrence on
    subsequent days

    Keyword arguments:
    exptGrp -- ExperimentGroup of experiments to analyze
    shuffle -- calculate recurrence probability of shuffled cells
    circ_var_pcs -- If True, uses circular variance method for identifying
        place cells.

    Note: roi_filter filters the rois on the first day, but not the second

    """

    data_list = []
    shuffle_list = []
    for e1, e2 in exptGrp.genImagedExptPairs():

        td = e2 - e1

        shared_rois = exptGrp.subGroup([e1, e2]).sharedROIs(
            channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'])
        shared_rois = set(shared_rois)
        shared_filter = lambda x: x.id in shared_rois

        if len(shared_rois) == 0:
            continue

        shared_pcs = exptGrp.pcs_filter(
            roi_filter=shared_filter, circ_var=circ_var_pcs)
        shared_pcs1 = set(e1.roi_ids(
            channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'],
            roi_filter=misc.filter_intersection([shared_pcs, roi_filter])))
        shared_pcs2 = set(e2.roi_ids(
            channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'], roi_filter=shared_pcs))

        # The shuffle probability that a pc on day 1 is also a pc on day 2 is
        # just equal to the probability that a cell is a pc on day 2
        if len(shared_pcs1) < 1:
            continue
        num_recur = len(shared_pcs1.intersection(shared_pcs2))
        num_pcs_first_expt = len(shared_pcs1)
        num_pcs_second_expt = len(shared_pcs2)
        # num_recur_shuffle = len(shared_pcs2)
        # num_start_shuffle = len(shared_rois)
        num_shared_rois = len(shared_rois)

        data_dict = {'num_recur': num_recur,
                     'num_pcs_first_expt': num_pcs_first_expt,
                     'num_pcs_second_expt': num_pcs_second_expt,
                     'value': float(num_recur) / num_pcs_first_expt,
                     'time_diff': td,
                     'first_expt': e1,
                     'second_expt': e2}
        data_list.append(data_dict)

        shuffle_dict = {'num_recur': num_pcs_second_expt,
                        'num_shared_rois': num_shared_rois,
                        'value': float(num_pcs_second_expt) / num_shared_rois,
                        'time_diff': td,
                        'first_expt': e1,
                        'second_expt': e2}
        shuffle_list.append(shuffle_dict)

    return (
        pd.DataFrame(data_list, columns=[
            'num_recur', 'num_pcs_first_expt', 'num_pcs_second_expt', 'value',
            'time_diff', 'first_expt', 'second_expt']),
        pd.DataFrame(shuffle_list, columns=[
            'num_recur', 'num_shared_rois', 'value', 'time_diff', 'first_expt',
            'second_expt']))


@memoize
def recurrence_above_chance(expt_grp, roi_filter=None, circ_var_pcs=False):

    raise Exception('Code incomplete')

    recurrence_df, _ = recurrence_probability(
        expt_grp, roi_filter=roi_filter, circ_var_pcs=circ_var_pcs)


@memoize
def circular_variance(exptGrp, roi_filter=None, min_transients=0):
    data_list = []
    circular_variance = exptGrp.circular_variance(roi_filter=roi_filter)
    if min_transients:
        n_trans = {}
        for expt in exptGrp:
            n_trans[expt] = calc_activity(
                expt, 'n transients', roi_filter=roi_filter,
                interval='running',
                running_kwargs=exptGrp.running_kwargs())[:, 0]

    for expt in exptGrp:
        rois = expt.rois(
            roi_filter=roi_filter, channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'])
        assert len(rois) == len(circular_variance[expt])
        if min_transients:
            for roi, n, var in it.izip(
                    rois, n_trans[expt], circular_variance[expt]):
                if n > min_transients:
                    data_list.append({'expt': expt, 'roi': roi, 'value': var})
        else:
            for roi, var in it.izip(rois, circular_variance[expt]):
                data_list.append({'expt': expt, 'roi': roi, 'value': var})
    return pd.DataFrame(data_list, columns=['expt', 'roi', 'value'])


@memoize
def circular_variance_p(exptGrp, roi_filter=None):
    data_list = []
    circular_variance_p = exptGrp.circular_variance_p(roi_filter=roi_filter)
    for expt in exptGrp:
        rois = expt.rois(
            roi_filter=roi_filter, channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'])
        assert len(rois) == len(circular_variance_p[expt])
        for roi, p in it.izip(rois, circular_variance_p[expt]):
            data_list.append({'expt': expt, 'roi': roi, 'value': p})
    return pd.DataFrame(data_list, columns=['expt', 'roi', 'value'])


@memoize
def spatial_information_p(exptGrp, roi_filter=None):
    data_list = []
    spatial_information_p = exptGrp.spatial_information_p(
        roi_filter=roi_filter)
    for expt in exptGrp:
        rois = expt.rois(
            roi_filter=roi_filter, channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'])
        assert len(rois) == len(spatial_information_p[expt])
        for roi, p in it.izip(rois, spatial_information_p[expt]):
            data_list.append({'expt': expt, 'roi': roi, 'value': p})
    return pd.DataFrame(data_list, columns=['expt', 'roi', 'value'])


@memoize
def spatial_information(exptGrp, roi_filter=None):
    data_list = []
    spatial_information = exptGrp.spatial_information(
        roi_filter=roi_filter)
    for expt in exptGrp:
        rois = expt.rois(
            roi_filter=roi_filter, channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'])
        assert len(rois) == len(spatial_information[expt])
        for roi, p in it.izip(rois, spatial_information[expt]):
            data_list.append({'expt': expt, 'roi': roi, 'value': p})
    return pd.DataFrame(data_list, columns=['expt', 'roi', 'value'])


@memoize
def calcCentroids(data, pfs, returnAll=False, return_pfs=False):
    """
    Input:
        data (df/f by position)
        pfs  (not-normalized place field output from identifyPlaceFields)
        returnAll
            -False --> for cells with >1 place field, take the one with bigger
                peak
            -True  --> return centroids for each place field ordered by peak
                df/f in the pf
        return_pfs : If True, returns pfs matching centroids

    Output:
    centroids: nROIs length list, each element is either an empty list or a
    list containing the bin (not rounded) of the centroid

    """

    # if np.all(np.array(list(flatten(pfs)))<1):
    #    raise TypeError('Invalid argument, must pass in non-normalized pfs')

    centroids = [[] for x in range(len(pfs))]
    pfs_out = [[] for x in range(len(pfs))]
    # peaks_out  = [[] for x in range(len(pfs))]
    for pfIdx, roi, pfList in it.izip(it.count(), data, pfs):
        if len(pfList) > 0:
            peaks = []
            roi_centroids = []
            for pf in pfList:
                if pf[0] < pf[1]:
                    pf_data = roi[pf[0]:pf[1] + 1]
                    peaks.append(np.amax(pf_data))
                    roi_centroids.append(pf[0] + np.sum(
                        pf_data * np.arange(len(pf_data))) / np.sum(pf_data))
                else:
                    pf_data = np.hstack([roi[pf[0]:], roi[:pf[1] + 1]])
                    peaks.append(np.amax(pf_data))
                    roi_centroids.append((pf[0] + np.sum(
                        pf_data * np.arange(len(pf_data))) / np.sum(pf_data))
                        % data.shape[1])
            # sort the pf peaks in descending order
            order = np.argsort(peaks)[::-1]
            if returnAll:
                centroids[pfIdx] = [roi_centroids[x] for x in order]
                pfs_out[pfIdx] = [pfList[x] for x in order]
                # peaks_out[pfIdx] = [peaks[x] for x in order]
            else:
                centroids[pfIdx] = [roi_centroids[order[0]]]
                pfs_out[pfIdx] = [pfList[order[0]]]
                # peaks_out[pfIdx] = [peaks[order[0]]]
            assert not np.any(np.isnan(centroids[pfIdx]))
    if return_pfs:
        return centroids, pfs_out  # , peaks_out
    return centroids


@memoize
def calc_activity_centroids(exptGrp, roi_filter=None):
    """
    Output:
        list = (nROIs,)
    """

    bins = 2 * np.pi * np.arange(0, 1, 1. / exptGrp.args['nPositionBins'])

    result = {}
    for expt in exptGrp:
        expt_result = []
        for tuning_curve in exptGrp.data_raw(roi_filter=roi_filter)[expt]:
            finite_idxs = np.where(np.isfinite(tuning_curve))[0]
            p = _complex_mean(bins[finite_idxs], tuning_curve[finite_idxs])
            expt_result.append(p)
        result[expt] = expt_result
    return result


@memoize
def activity_centroid(exptGrp, roi_filter=None):

    centroids = calc_activity_centroids(exptGrp, roi_filter=roi_filter)

    data_list = []
    for expt in exptGrp:
        rois = expt.rois(
            roi_filter=roi_filter, channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'])
        assert len(rois) == len(centroids[expt])
        data_list.extend(
            [{'roi': roi, 'expt': expt, 'value': centroid}
             for roi, centroid in zip(rois, centroids[expt])])

    df = pd.DataFrame(data_list, columns=['expt', 'roi', 'value'])

    df['angle'] = df['value'].apply(np.angle)
    df['length'] = df['value'].apply(np.absolute)

    return df


@memoize
def centroid_shift(exptGrp, roi_filter=None, return_abs=False, shuffle=True):
    """Calculate the shift of place field centers over days

    Determines the center by calculating the center of 'mass' of the calcium
    signal

    Keyword arguments:
    exptGrp -- pcExperimentGroup of experiments to analyze
    return_abs -- If True, return absolute value of centroid shift

    """

    N_SHUFFLES = 10000

    nBins = exptGrp.args['nPositionBins']
    pcs_filter = exptGrp.pcs_filter(roi_filter=roi_filter)
    pfs = exptGrp.pfs(roi_filter=roi_filter)
    data = exptGrp.data(roi_filter=roi_filter)
    rois_by_id = {
        expt: {roi.id: roi for roi in expt.rois(
            channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'])} for expt in exptGrp}
    all_roi_ids = exptGrp.roi_ids(
        channel=exptGrp.args['channel'], label=exptGrp.args['imaging_label'],
        roi_filter=roi_filter)

    data_list = []
    shuffle_dicts = []
    centroids = {}
    for expt in exptGrp:
        # Calculate centroids and then discard all placeFields for an ROI
        # that has more than 1
        centroids[expt] = calcCentroids(data[expt], pfs[expt])
        centroids[expt] = [centroids[expt][idx] if len(pfs[expt][idx])
                           <= 1 else [] for idx in range(len(centroids[expt]))]

    for (e1, e2) in exptGrp.genImagedExptPairs():

        shared_pcs = exptGrp.subGroup([e1, e2]).sharedROIs(
            channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'], roi_filter=pcs_filter)

        if len(shared_pcs) == 0:
            continue

        e1_pairs, e2_pairs = [], []

        for pc in shared_pcs:
            centroid1 = centroids[e1][all_roi_ids[e1].index(pc)]
            centroid2 = centroids[e2][all_roi_ids[e2].index(pc)]

            if not len(centroid1) or not len(centroid2):
                continue

            first_roi = rois_by_id[e1][pc]
            second_roi = rois_by_id[e2][pc]

            shift = (centroid2[0] - centroid1[0]) / nBins
            shift = shift - 1 if shift > 0.5 else \
                shift + 1 if shift <= -0.5 else shift

            data_dict = {'value': shift,
                         'first_expt': e1,
                         'second_expt': e2,
                         'first_roi': first_roi,
                         'second_roi': second_roi}
            data_list.append(data_dict)

            if shuffle:
                e1_pairs.append((first_roi, centroid1[0]))
                e2_pairs.append((second_roi, centroid2[0]))

        if shuffle:
            shuffle_dicts.extend([
                {'expts': (e1, e2), 'rois': (r1, r2), 'data': (d1, d2)}
                for (r1, d1), (r2, d2) in it.product(e1_pairs, e2_pairs)])

    data_df = pd.DataFrame(data_list, columns=[
        'first_expt', 'second_expt', 'first_roi', 'second_roi', 'value'])

    if shuffle:
        if len(shuffle_dicts) < N_SHUFFLES:
            shuffler = shuffle_dicts
        else:
            shuffler = sample(shuffle_dicts, N_SHUFFLES)

        shuffle_list = []
        for pair in shuffler:
            shift = (pair['data'][1] - pair['data'][0]) / nBins
            shift = shift - 1 if shift > 0.5 else \
                shift + 1 if shift <= -0.5 else shift

            shuffle_dict = {'value': shift,
                            'first_expt': pair['expts'][0],
                            'second_expt': pair['expts'][1],
                            'first_roi': pair['rois'][0],
                            'second_roi': pair['rois'][1]}
            shuffle_list.append(shuffle_dict)
        shuffle_df = pd.DataFrame(shuffle_list, columns=[
            'first_expt', 'second_expt', 'first_roi', 'second_roi', 'value'])
    else:
        shuffle_df = None

    if return_abs:
        data_df['value'] = data_df['value'].abs()
        if shuffle:
            shuffle_df['value'] = shuffle_df['value'].abs()

    return data_df, shuffle_df


@memoize
def activity_centroid_shift(
        exptGrp, roi_filter=None, activity_filter='pc_either',
        circ_var_pcs=True, shuffle=True, units='rad'):
    """Calculate the angle between activity centroids.

    Parameters
    ----------
    activity_filter : {'pc_either', 'pc_both', 'active_either', 'active_both'}
        Determines which cells to include on a per-expt-pair basis.
    circ_var_pcs : bool
        If True, use circular variance place cell threshold instead of spatial
        information.
    shuffle : bool
        If True, calculate shuffle distributions.
    units : {'rad', 'norm', 'cm'}
        Determine the units of the returned result.

    Returns
    -------
    data_df : pandas.DataFrame
    shuffle_df {pandas.DataFrame, None}

    """
    if units not in ['rad', 'norm', 'cm']:
        raise ValueError("Unrecognized 'units' parameter value.")

    def dotproduct(v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))

    def length(v):
        return math.sqrt(dotproduct(v, v))

    def angle(v1, v2):
        return math.acos(
            np.round(dotproduct(v1, v2) / (length(v1) * length(v2)), 3))

    N_SHUFFLES = 10000

    data_list = []
    shuffle_dicts = []
    centroids = {}

    # Pre-calc and store things for speed
    centroids = calc_activity_centroids(exptGrp, roi_filter=None)
    if activity_filter is not None:
        if 'pc' in activity_filter:
            pcs_filter = exptGrp.pcs_filter(circ_var=circ_var_pcs)
        elif 'active' in activity_filter:
            active_filter = filters.active_roi_filter(
                exptGrp, min_transients=1, channel=exptGrp.args['channel'],
                label=exptGrp.args['imaging_label'], roi_filter=roi_filter)
    rois_by_id = {
        expt: {roi.id: roi for roi in expt.rois(
            channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'])} for expt in exptGrp}
    all_roi_ids = exptGrp.roi_ids(
        channel=exptGrp.args['channel'], label=exptGrp.args['imaging_label'],
        roi_filter=None)

    for (e1, e2) in exptGrp.genImagedExptPairs():
        e1_pairs, e2_pairs = [], []
        shared_rois = exptGrp.subGroup([e1, e2]).sharedROIs(
            channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'], roi_filter=roi_filter)
        if activity_filter == 'pc_either':
            e1_pcs = e1.roi_ids(roi_filter=pcs_filter)
            e2_pcs = e2.roi_ids(roi_filter=pcs_filter)
            roi_ids = list(set(shared_rois).intersection(
                set(e1_pcs).union(e2_pcs)))
        elif activity_filter == 'pc_both':
            e1_pcs = e1.roi_ids(roi_filter=pcs_filter)
            e2_pcs = e2.roi_ids(roi_filter=pcs_filter)
            roi_ids = list(set(shared_rois).intersection(
                           e1_pcs).intersection(e2_pcs))
        elif activity_filter == 'active_either':
            e1_active = e1.roi_ids(roi_filter=active_filter)
            e2_active = e2.roi_ids(roi_filter=active_filter)
            roi_ids = list(set(shared_rois).intersection(
                set(e1_active).union(e2_active)))
        elif activity_filter == 'active_both':
            e1_active = e1.roi_ids(roi_filter=active_filter)
            e2_active = e2.roi_ids(roi_filter=active_filter)
            roi_ids = list(set(shared_rois).intersection(
                set(e1_active).intersection(e2_active)))
        elif activity_filter:
            e1_rois = e1.roi_ids(roi_filter=activity_filter)
            e2_rois = e2.roi_ids(roi_filter=activity_filter)
            roi_ids = list(set(shared_rois).intersection(
                           set(e1_rois).union(set(e2_rois))))
        else:
            roi_ids = shared_rois

        for roi_id in roi_ids:
            c1 = centroids[e1][all_roi_ids[e1].index(roi_id)]
            c2 = centroids[e2][all_roi_ids[e2].index(roi_id)]

            roi1 = rois_by_id[e1][roi_id]
            roi2 = rois_by_id[e2][roi_id]

            value = angle([c1.real, c1.imag], [c2.real, c2.imag])

            if units == 'norm' or units == 'cm':
                value /= 2 * np.pi
                if units == 'cm':
                    belt_length = np.mean([
                        e1.belt().length(units='cm'),
                        e2.belt().length(units='cm')])
                    value *= belt_length

            data_dict = {'value': value,
                         'first_expt': e1,
                         'second_expt': e2,
                         'first_roi': roi1,
                         'second_roi': roi2,
                         'first_centroid': c1,
                         'second_centroid': c2}
            data_list.append(data_dict)

            if shuffle:
                e1_pairs.append((roi1, c1))
                e2_pairs.append((roi2, c2))

        if shuffle:
            shuffle_dicts.extend([
                {'expts': (e1, e2), 'rois': (r1, r2), 'data': (d1, d2)}
                for (r1, d1), (r2, d2) in it.product(e1_pairs, e2_pairs)])

    if shuffle:
        if len(shuffle_dicts) < N_SHUFFLES:
            shuffler = shuffle_dicts
        else:
            shuffler = sample(shuffle_dicts, N_SHUFFLES)

        shuffle_list = []
        for pair in shuffler:
            value = angle([pair['data'][0].real, pair['data'][0].imag],
                          [pair['data'][1].real, pair['data'][1].imag])
            if units == 'norm' or units == 'cm':
                value /= 2 * np.pi
                if units == 'cm':
                    belt_length = np.mean([
                        e1.belt().length(units='cm'),
                        e2.belt().length(units='cm')])
                    value *= belt_length

            shuffle_dict = {'value': value,
                            'first_expt': pair['expts'][0],
                            'second_expt': pair['expts'][1],
                            'first_roi': pair['rois'][0],
                            'second_roi': pair['rois'][1],
                            'first_centroid': pair['data'][0],
                            'second_centroid': pair['data'][1]}
            shuffle_list.append(shuffle_dict)

    data_df = pd.DataFrame(data_list, columns=[
        'first_expt', 'second_expt', 'first_roi', 'second_roi',
        'first_centroid', 'second_centroid', 'value'])
    shuffle_df = pd.DataFrame(shuffle_list, columns=[
        'first_expt', 'second_expt', 'first_roi', 'second_roi',
        'first_centroid', 'second_centroid', 'value']) \
        if shuffle else None

    return data_df, shuffle_df


@memoize
def sparsity(exptGrp, roi_filter=None):
    """Calculate single-cell sparsity index (equivalently
       'lifetime sparseness') as defined in Ahmed and Mehta (Trends in
       Neuroscience, 2009)

    """

    data = exptGrp.data(roi_filter=roi_filter)

    nBins = exptGrp.args['nPositionBins']
    data_list = []
    for expt in exptGrp:
        rois = expt.rois(
            roi_filter=roi_filter, channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'])

        assert len(rois) == len(data[expt])

        for roi, tuning_curve in zip(rois, data[expt]):

            num = (np.sum(tuning_curve) / float(nBins)) ** 2
            den = np.sum(np.square(tuning_curve)) / float(nBins)
            value = num / den

            data_dict = {'expt': expt,
                         'roi': roi,
                         'value': value}
            data_list.append(data_dict)

    return pd.DataFrame(data_list, columns=['expt', 'roi', 'value'])


@memoize
def place_field_correlation(
        exptGrp, roi_filter=None, activity_filter='pc_either', shuffle=True):
    """Calculate the mean correlation in spatial tuning over time.

    For each pair of experiments in the experiment group, find common place
    cells and then calculate correlation in spatial tuning curves between the
    two experiments.  Each pair of experiments is assigned a single mean
    correlation in the spatial tuning of the common place cells.

    Keyword arguments:
    activity_filter -- determines how to filter the cells for each expt pair
        Can be None, 'pc_either' or 'pc_both'
    """

    N_SHUFFLES = 10000

    data = exptGrp.data(roi_filter=roi_filter)
    pcs_filter = exptGrp.pcs_filter(roi_filter=roi_filter)
    rois_by_id = {
        expt: {roi.id: roi for roi in expt.rois(
            channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'])} for expt in exptGrp}
    all_roi_ids = exptGrp.roi_ids(
        channel=exptGrp.args['channel'], label=exptGrp.args['imaging_label'],
        roi_filter=roi_filter)

    data_list = []
    shuffle_dicts = []
    for e1, e2 in exptGrp.genImagedExptPairs():
        shared_rois = exptGrp.subGroup([e1, e2]).sharedROIs(
            channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'],
            demixed=exptGrp.args['demixed'], roi_filter=roi_filter)

        if activity_filter is None:
            roi_ids = shared_rois
        elif activity_filter == 'pc_either':
            e1_pcs = e1.roi_ids(roi_filter=pcs_filter)
            e2_pcs = e2.roi_ids(roi_filter=pcs_filter)
            roi_ids = list(set(shared_rois).intersection(
                set(e1_pcs).union(e2_pcs)))
        elif activity_filter == 'pc_both':
            e1_pcs = e1.roi_ids(roi_filter=pcs_filter)
            e2_pcs = e2.roi_ids(roi_filter=pcs_filter)
            roi_ids = list(set(shared_rois).intersection(
                set(e1_pcs).intersection(e2_pcs)))
        elif activity_filter == 'pc_first':
            e1_pcs = e1.roi_ids(roi_filter=pcs_filter)
            roi_ids = list(set(shared_rois).intersection(set(e1_pcs)))
        elif activity_filter == 'pc_second':
            e2_pcs = e2.roi_ids(roi_filter=pcs_filter)
            roi_ids = list(set(shared_rois).intersection(set(e2_pcs)))
        else:
            try:
                e1_rois = e1.roi_ids(label=exptGrp.args['imaging_label'], roi_filter=activity_filter)
                e2_rois = e2.roi_ids(label=exptGrp.args['imaging_label'], roi_filter=activity_filter)
                roi_ids = list(set(shared_rois).intersection(
                    set(e1_rois).union(set(e2_rois))))
            except:
                raise ValueError('Unrecognized activity filter')

        if len(roi_ids) == 0:
            continue

        e1_pairs, e2_pairs = [], []
        for roi in roi_ids:
            tuning1 = data[e1][all_roi_ids[e1].index(roi)]
            tuning2 = data[e2][all_roi_ids[e2].index(roi)]
            tuning_corr = np.corrcoef(tuning1, tuning2)[0, 1]

            first_roi = rois_by_id[e1][roi]
            second_roi = rois_by_id[e2][roi]

            data_dict = {'value': tuning_corr,
                         'first_expt': e1,
                         'second_expt': e2,
                         'first_roi': first_roi,
                         'second_roi': second_roi}
            data_list.append(data_dict)

            if shuffle:
                e1_pairs.append((first_roi, tuning1))
                e2_pairs.append((second_roi, tuning2))

        if shuffle:
            shuffle_dicts.extend(
                [{'expts': (e1, e2), 'rois': (r1, r2), 'data': (d1, d2)}
                 for (r1, d1), (r2, d2) in it.product(e1_pairs, e2_pairs)])

    if shuffle:
        if len(shuffle_dicts) < N_SHUFFLES:
            shuffler = shuffle_dicts
        else:
            shuffler = sample(shuffle_dicts, N_SHUFFLES)

        shuffle_list = []
        for pair in shuffler:
            tuning_corr = np.corrcoef(*pair['data'])[0, 1]
            shuffle_dict = {'value': tuning_corr,
                            'first_expt': pair['expts'][0],
                            'second_expt': pair['expts'][1],
                            'first_roi': pair['rois'][0],
                            'second_roi': pair['rois'][1]}
            shuffle_list.append(shuffle_dict)

    data_df = pd.DataFrame(data_list, columns=[
        'first_expt', 'second_expt', 'first_roi', 'second_roi', 'value'])
    shuffle_df = pd.DataFrame(shuffle_list, columns=[
        'first_expt', 'second_expt', 'first_roi', 'second_roi', 'value']) \
        if shuffle else None

    return data_df, shuffle_df


@memoize
def rank_order_correlation(
        exptGrp, roi_filter=None, method='centroids', min_shared_rois=1,
        shuffle=True, return_abs=False):
    """Calculate the rank order correlation between pairs of experiments

    For each pair of experiments in the experiment group, find common place
    cells and then calculate the Spearman rank order correlation in the order
    of their place fields  Note that a significant p-value will be returned if
    the order is either preserved or reversed
    """

    N_SHUFFLES = 10000

    if method == 'centroids':
        pcs_filter = exptGrp.pcs_filter(roi_filter=roi_filter)
        data = exptGrp.data(roi_filter=pcs_filter)
        pfs = exptGrp.pfs(roi_filter=pcs_filter)

        centroids = {}
        for expt in exptGrp:
            # Calculate centroids and then discard all placeFields for an ROI
            # that has more than 1
            centroids[expt] = calcCentroids(
                data[expt], pfs[expt], returnAll=True)
            centroids[expt] = [
                centroids[expt][idx] if len(pfs[expt][idx]) <= 1 else []
                for idx in range(len(centroids[expt]))]
    elif method == 'tuning_vectors':
        pcs_filter = exptGrp.pcs_filter(roi_filter=roi_filter, circ_var=True)
        centroids = calc_activity_centroids(exptGrp, roi_filter=pcs_filter)

        for key in centroids.iterkeys():
            complex_angles = centroids[key]
            centroids[key] = (np.angle(complex_angles) % (2 * np.pi)) / \
                (2 * np.pi)
            centroids[key] = [[x] for x in centroids[key]]
    else:
        raise('Not a valid method')

    all_roi_ids = exptGrp.roi_ids(
        channel=exptGrp.args['channel'], label=exptGrp.args['imaging_label'],
        roi_filter=pcs_filter)

    data_list = []
    for e1, e2 in exptGrp.genImagedExptPairs():
        shared_pcs = exptGrp.subGroup([e1, e2]).sharedROIs(
            channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'], roi_filter=pcs_filter)

        for pc in shared_pcs[::-1]:
            if not len(centroids[e1][all_roi_ids[e1].index(pc)]) or \
                    not len(centroids[e2][all_roi_ids[e2].index(pc)]):
                shared_pcs.remove(pc)

        if len(shared_pcs) < min_shared_rois:
            continue

        centroids1 = []
        centroids2 = []
        for pc in shared_pcs:

            c1 = centroids[e1][all_roi_ids[e1].index(pc)]
            c2 = centroids[e2][all_roi_ids[e2].index(pc)]

            centroids1.append(c1[0])
            centroids2.append(c2[0])

        template_order = np.argsort(np.array(centroids1))
        target_order = np.argsort(np.array(centroids2))
        template_order_ids = [shared_pcs[x] for x in template_order]
        target_order_ids = [shared_pcs[x] for x in target_order]

        template = np.arange(len(centroids1))
        target = np.array(
            [template_order_ids.index(x) for x in target_order_ids])

        p_val = 1.
        rho = 0.
        for shift in xrange(len(template)):
            r, p = spearmanr(template, np.roll(target, shift))
            if p < p_val:
                p_val = p
                rho = r

        data_list.append({'first_expt': e1,
                          'second_expt': e2,
                          'value': rho if not return_abs else np.abs(rho),
                          'p': p_val,
                          'n_shared_rois': len(shared_pcs)})

    return pd.DataFrame(data_list, columns=[
        'first_expt', 'second_expt', 'value', 'p', 'n_shared_rois'])


@memoize
def place_cell_percentage(exptGrp, roi_filter=None, circ_var=False):
    """Calculate the percentage of cells that are a place cell on each day."""
    pcs_filter = exptGrp.pcs_filter(roi_filter=roi_filter, circ_var=circ_var)
    data_list = []
    for expt in exptGrp:
        n_rois = len(expt.rois(
            roi_filter=roi_filter, channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label']))
        n_pcs = len(expt.rois(
            roi_filter=pcs_filter, channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label']))

        if n_rois == 0:
            result = np.nan
        else:
            result = float(n_pcs) / n_rois

        data_dict = {'expt': expt,
                     'value': result}
        data_list.append(data_dict)
    return pd.DataFrame(data_list, columns=['expt', 'value'])


@memoize
def n_place_fields(
        exptGrp, roi_filter=None, per_mouse_fractions=False,
        max_n_place_fields=None):
    """Calculate the number of place fields for each place cell"""
    data_list = []
    pfs = exptGrp.pfs(roi_filter=roi_filter)

    for expt in exptGrp:
        rois = expt.rois(
            roi_filter=roi_filter, channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'])
        assert len(rois) == len(pfs[expt])

        for roi, pc in zip(rois, pfs[expt]):
            n_place_fields = len(pc)
            if n_place_fields:
                data_dict = {'expt': expt,
                             'roi': roi,
                             'value': n_place_fields}
                data_list.append(data_dict)

    result = pd.DataFrame(data_list, columns=['expt', 'roi', 'value'])

    if per_mouse_fractions:
        new_data_list = []
        plotting.prepare_dataframe(result, include_columns=['mouse'])
        for mouse, mouse_df in result.groupby('mouse'):
            n_total_rois = len(mouse_df)
            mouse_counts = mouse_df.groupby('value', as_index=False).count()
            for n_pfs in np.arange(mouse_counts['value'].max()) + 1:
                if max_n_place_fields is None or n_pfs < max_n_place_fields:
                    n_rois = mouse_counts.ix[
                        mouse_counts['value'] == n_pfs, 'mouse'].sum()
                    number = n_pfs
                elif n_pfs == max_n_place_fields:
                    n_rois = mouse_counts.ix[
                        mouse_counts['value'] >= max_n_place_fields,
                        'mouse'].sum()
                    number = str(n_pfs) + '+'
                else:
                    break
                data_dict = {'mouse': mouse,
                             'number': n_pfs,
                             'n_rois': n_rois,
                             'n_total_rois': n_total_rois,
                             'value': n_rois / float(n_total_rois)}
                new_data_list.append(data_dict)
        result = pd.DataFrame(
            new_data_list, columns=[
                'mouse', 'number', 'n_rois', 'n_total_rois', 'value'])

    return result


def n_sessions_place_cell(
        exptGrp, roi_filter=None, ax=None, title_visible=True,
        minimum_observations=0, plotShuffle=True, color=None):

    pcs_filter = exptGrp.pcs_filter(roi_filter=roi_filter)
    pfs = exptGrp.pfs()

    placeCellPercentages = {}
    for e in exptGrp:
        nPCs = len(e.rois(roi_filter=pcs_filter))
        nCells = len(e.rois(roi_filter=roi_filter))

        placeCellPercentages[e] = float(nPCs) / nCells

    allROIs = exptGrp.allROIs(channel=exptGrp.args['channel'],
                              label=exptGrp.args['imaging_label'],
                              roi_filter=roi_filter)

    nSessionsPlaceCell = []
    shuffles = []
    for roi in allROIs.itervalues():
        shuffle_probabilities = []
        nSessions = 0
        for (expt, roi_idx) in roi:
            if len(pfs[expt][roi_idx]):
                nSessions += 1
            shuffle_probabilities.append(placeCellPercentages[expt])
        nSessionsPlaceCell.append(nSessions)
        shuffles.append(
            stats.poisson_binomial_distribution(shuffle_probabilities))

    shuffle_dist = np.empty(
        (len(allROIs), np.amax([len(x) for x in shuffles])))
    shuffle_dist.fill(np.nan)

    for x, dist in zip(shuffles, shuffle_dist):
        dist[:len(x)] = x

    shuffle_dist = nanmean(shuffle_dist, axis=0)

    if ax:
        if color is None:
            color = lab.plotting.color_cycle().next()
        if len(nSessionsPlaceCell):
            plotting.histogram(
                ax, nSessionsPlaceCell, bins=len(shuffle_dist),
                range=(0, len(shuffle_dist)), normed=True, plot_mean=True,
                label=exptGrp.label(), color=color)
            ax.set_ylabel('Normalized density')
            ax.set_xlabel('Number of sessions')
            if title_visible:
                ax.set_title('Number of sessions as place cell')
            ax.legend(frameon=False, loc='best')

        if plotShuffle:
            ax.step(np.arange(len(shuffle_dist)), shuffle_dist, where='post',
                    color='k', linestyle='dashed')

    return nSessionsPlaceCell


def n_sessions_imaged(exptGrp, roi_filter=None, ax=None, title_visible=True):
    """Of all cells in the group, how many days was each imaged?"""

    # dict by experiment
    roi_ids = exptGrp.roi_ids(channel=exptGrp.args['channel'],
                              label=exptGrp.args['imaging_label'],
                              roi_filter=roi_filter)

    data_list = []
    for all_rois in exptGrp.allROIs(channel=exptGrp.args['channel'],
                                    label=exptGrp.args['imaging_label'],
                                    roi_filter=roi_filter).itervalues():
        mouse_id = all_rois[0][0].parent.get('mouseID')
        roi_id = roi_ids[all_rois[0][0]][all_rois[0][1]]
        location = all_rois[0][0].get('uniqueLocationKey')
        nSessionsImaged = len(all_rois)

        data_dict = {'mouseID': mouse_id,
                     'roi_id': roi_id,
                     'location': location,
                     'value': nSessionsImaged}
        data_list.append(data_dict)

    return pd.DataFrame(data_list, columns=[
        'mouseID', 'roi_id', 'location', 'value'])


def is_ever_place_cell(
        expt_grps, roi_filters=None, ax=None, colors=None, groupby=None,
        **plot_kwargs):
    if roi_filters is None:
        roi_filters = [None] * len(expt_grps)

    dfs = []
    for expt_grp, roi_filter in zip(expt_grps, roi_filters):
        df = lab.ExperimentGroup.filtered_rois(
            expt_grp, roi_filter=expt_grp.pcs_filter(),
            include_roi_filter=roi_filter, channel=expt_grp.args['channel'],
            label=expt_grp.args['imaging_label'])

        plotting.prepare_dataframe(
            df, include_columns=['mouseID', 'uniqueLocationKey', 'roi_id',
                                 'session_number_in_df'])

        data = []
        for key, group in df.groupby(
                ['mouseID', 'uniqueLocationKey', 'roi_id']):
            values = np.array(group['value'])
            order = np.argsort(group['session_number_in_df']).tolist()
            for i, val in enumerate(np.maximum.accumulate(values[order])):
                data.append(
                    {'mouseID': key[0], 'uniqueLocationKey': key[1],
                     'roi_id': key[2], 'session_number': i, 'value': val})

        dfs.append(pd.DataFrame(data))

    if ax is not None:
        plotting.plot_dataframe(
            ax, dfs, labels=[expt_grp.label() for expt_grp in expt_grps],
            plotby=['session_number'], groupby=groupby,
            plot_method='grouped_bar', colors=colors,
            activity_label='Has ever been PC', **plot_kwargs)

        if groupby is None:
            ax.collections = []

    return dfs


@memoize
def population_vector_correlation(
        exptGrp, roi_filter=None, method='angle', activity_filter='pc_either',
        min_pf_density=0, shuffle=True, circ_var_pcs=False, reward_at_zero=False):
    """Calculates and plots the population similarity score over time

    Keyword arguments:
    method -- similarity method to use, either 'corr' for correlation
        or 'angle' for cosine of angle between the pop vectors
    activity_filter -- determines how to filter the cells for each expt pair
        Can be None, 'pf_either' or 'pf_both'
    min_pf_density -- only include position bins that have a minimum fraction
        of place cells with nonzero tuning curves at that position

    """

    N_SHUFFLES = 10000

    if activity_filter and not callable(activity_filter):
        if 'pc' in activity_filter:
            pcs_filter = exptGrp.pcs_filter(
                roi_filter=roi_filter, circ_var=circ_var_pcs)
        elif 'active' in activity_filter:
            active_filter = filters.active_roi_filter(
                exptGrp, min_transients=1, channel=exptGrp.args['channel'],
                label=exptGrp.args['imaging_label'], roi_filter=roi_filter)
    data = exptGrp.data(roi_filter=roi_filter)
    all_roi_ids = exptGrp.roi_ids(
        channel=exptGrp.args['channel'], label=exptGrp.args['imaging_label'],
        roi_filter=roi_filter)

    data_list = []
    shuffle_dicts = []

    for e1, e2 in exptGrp.genImagedExptPairs():
        grp = exptGrp.subGroup([e1, e2])

        shared_rois = grp.sharedROIs(channel=grp.args['channel'],
                                     label=grp.args['imaging_label'],
                                     demixed=grp.args['demixed'],
                                     roi_filter=roi_filter)
        if len(shared_rois) == 0:
            continue

        if activity_filter is None:
            roi_ids = shared_rois
        elif activity_filter == 'pc_either':
            e1_pcs = e1.roi_ids(roi_filter=pcs_filter)
            e2_pcs = e2.roi_ids(roi_filter=pcs_filter)
            roi_ids = list(set(shared_rois).intersection(
                set(e1_pcs).union(e2_pcs)))
        elif activity_filter == 'pc_both':
            e1_pcs = e1.roi_ids(roi_filter=pcs_filter)
            e2_pcs = e2.roi_ids(roi_filter=pcs_filter)
            roi_ids = list(set(shared_rois).intersection(
                set(e1_pcs).intersection(e2_pcs)))
        elif activity_filter == 'pc_first':
            e1_pcs = e1.roi_ids(roi_filter=pcs_filter)
            roi_ids = list(set(shared_rois).intersection(set(e1_pcs)))
        elif activity_filter == 'pc_second':
            e2_pcs = e2.roi_ids(roi_filter=pcs_filter)
            roi_ids = list(set(shared_rois).intersection(set(e2_pcs)))
        elif activity_filter == 'active_either':
            e1_active = e1.roi_ids(roi_filter=active_filter)
            e2_active = e2.roi_ids(roi_filter=active_filter)
            roi_ids = list(set(shared_rois).intersection(
                set(e1_active).union(e2_active)))
        elif activity_filter == 'active_both':
            e1_active = e1.roi_ids(roi_filter=active_filter)
            e2_active = e2.roi_ids(roi_filter=active_filter)
            roi_ids = list(set(shared_rois).intersection(
                set(e1_active).intersection(e2_active)))
        else:
            try:
                e1_rois = e1.roi_ids(label=exptGrp.args['imaging_label'], roi_filter=activity_filter)
                e2_rois = e2.roi_ids(label=exptGrp.args['imaging_label'], roi_filter=activity_filter)
                roi_ids = list(set(shared_rois).intersection(
                    set(e1_rois).intersection(set(e2_rois))))
            except:
                raise ValueError("Unrecognized value for 'activity_filter'")

        if len(roi_ids) == 0:
            continue

        e1_rois = np.array([all_roi_ids[e1].index(x) for x in roi_ids])
        e2_rois = np.array([all_roi_ids[e2].index(x) for x in roi_ids])

        # shape = (position, rois)
        e1_data = data[e1][e1_rois].swapaxes(0, 1)
        e2_data = data[e2][e2_rois].swapaxes(0, 1)

        if reward_at_zero:
            # Define mapping from [0, 99) to [-50, 50) with reward at center
            n_bins = exptGrp.args['nPositionBins']
            reward = e1.rewardPositions(units='normalized')[0]
            reward *= n_bins
            reward = int(reward)
            new_pos_bins = np.arange(0, n_bins) - reward
            for ix, x in enumerate(new_pos_bins):
                if x >= n_bins / 2:
                    new_pos_bins[ix] -= n_bins
                if x < -1 * (n_bins / 2):
                    new_pos_bins[ix] += n_bins

        # Iterate over positions
        for pos_bin, vect1, vect2 in it.izip(it.count(), e1_data, e2_data):
            # impose min_pf_density threshold
            if min_pf_density:
                if not (float(len(np.nonzero(vect1)[0])) / len(vect1) >
                        min_pf_density and
                        float(len(np.nonzero(vect2)[0])) / len(vect2) >
                        min_pf_density):
                    continue

            if method == 'corr':
                value = np.corrcoef(vect1, vect2)[0, 1]
            elif method == 'angle':
                value = np.dot(vect1, vect2) / np.linalg.norm(vect1) / \
                    np.linalg.norm(vect2)
            else:
                raise Exception('Unrecognized similarity method')
            position_bin = new_pos_bins[pos_bin] if reward_at_zero else pos_bin
            data_dict = {'value': value,
                         'first_expt': e1,
                         'second_expt': e2,
                         'position_bin_index': position_bin}
            data_list.append(data_dict)

        if shuffle:
            shuffle_dicts.extend([
                {'expts': (e1, e2), 'position_bin_indices': (b1, b2),
                 'data': (d1, d2)} for (b1, d1), (b2, d2) in it.product(
                    it.izip(it.count(), e1_data),
                    it.izip(it.count(), e2_data))])

    if shuffle:
        if len(shuffle_dicts) < N_SHUFFLES:
            shuffler = shuffle_dicts
        else:
            shuffler = sample(shuffle_dicts, N_SHUFFLES)

        shuffle_list = []
        for pair in shuffler:
            if method == 'corr':
                value = np.corrcoef(*pair['data'])[0, 1]
            elif method == 'angle':
                value = np.dot(*pair['data']) / np.linalg.norm(
                    pair['data'][0]) / np.linalg.norm(pair['data'][1])
            else:
                raise ValueError('Unrecognized similarity method')
            shuffle_dict = {
                'value': value, 'first_expt': pair['expts'][0],
                'second_expt': pair['expts'][1],
                'position_bin_index': pair['position_bin_indices'][0]}
            shuffle_list.append(shuffle_dict)

    data_df = pd.DataFrame(data_list, columns=[
        'first_expt', 'second_expt', 'position_bin_index', 'value'])
    shuffle_df = pd.DataFrame(shuffle_list, columns=[
        'first_expt', 'second_expt', 'position_bin_index', 'value']) \
        if shuffle else None

    return data_df, shuffle_df


@memoize
def overlap(
        exptGrp, roi_filter=None, activity_method='frequency',
        running_only=True, activity_filter='pc_either', shuffle=True,
        circ_var_pcs=False, **activity_filter_kwargs):

    N_SHUFFLES = 10000

    if activity_filter:
        if 'pc' in activity_filter:
            pcs_filter = exptGrp.pcs_filter(
                roi_filter=roi_filter, circ_var=circ_var_pcs)
        elif 'active' in activity_filter:
            active_filter = filters.active_roi_filter(
                exptGrp, channel=exptGrp.args['channel'],
                label=exptGrp.args['imaging_label'], roi_filter=roi_filter,
                **activity_filter_kwargs)
    rois_by_id = {
        expt: {roi.id: roi for roi in expt.rois(
            channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'])} for expt in exptGrp}
    all_roi_ids = exptGrp.roi_ids(
        channel=exptGrp.args['channel'], label=exptGrp.args['imaging_label'],
        roi_filter=roi_filter)
    activity = {}
    for expt in exptGrp:
        act = calc_activity(
            expt, interval='running' if running_only else None,
            method=activity_method, channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'], roi_filter=roi_filter,
            demixed=exptGrp.args['demixed'],
            running_kwargs=exptGrp.running_kwargs() if running_only else None)
        act = act.mean(1)
        activity[expt] = act

    data_list = []
    shuffle_dicts = []

    for e1, e2 in exptGrp.genImagedExptPairs():
        grp = exptGrp.subGroup([e1, e2])
        shared_rois = grp.sharedROIs(
            channel=grp.args['channel'], label=grp.args['imaging_label'],
            demixed=grp.args['demixed'], roi_filter=roi_filter)

        if activity_filter is None:
            roi_ids = shared_rois
        elif activity_filter == 'pc_either':
            e1_pcs = e1.roi_ids(roi_filter=pcs_filter)
            e2_pcs = e2.roi_ids(roi_filter=pcs_filter)
            roi_ids = list(set(shared_rois).intersection(
                set(e1_pcs).union(e2_pcs)))
        elif activity_filter == 'pc_both':
            e1_pcs = e1.roi_ids(roi_filter=pcs_filter)
            e2_pcs = e2.roi_ids(roi_filter=pcs_filter)
            roi_ids = list(set(shared_rois).intersection(
                set(e1_pcs).intersection(e2_pcs)))
        elif activity_filter == 'pc_first':
            e1_pcs = e1.roi_ids(roi_filter=pcs_filter)
            roi_ids = list(set(shared_rois).intersection(set(e1_pcs)))
        elif activity_filter == 'pc_second':
            e2_pcs = e2.roi_ids(roi_filter=pcs_filter)
            roi_ids = list(set(shared_rois).intersection(set(e2_pcs)))
        elif activity_filter == 'active_either':
            e1_active = e1.roi_ids(roi_filter=active_filter)
            e2_active = e2.roi_ids(roi_filter=active_filter)
            roi_ids = list(set(shared_rois).intersection(
                set(e1_active).union(e2_active)))
        elif activity_filter == 'active_both':
            e1_active = e1.roi_ids(roi_filter=active_filter)
            e2_active = e2.roi_ids(roi_filter=active_filter)
            roi_ids = list(set(shared_rois).intersection(
                set(e1_active).intersection(e2_active)))
        else:
            try:
                e1_rois = e1.roi_ids(roi_filter=activity_filter)
                e2_rois = e2.roi_ids(roi_filter=activity_filter)
                roi_ids = list(set(shared_rois).intersection(
                    set(e1_rois).union(set(e2_rois))))
            except:
                raise ValueError('Unrecognized activity filter')

        e1_pairs, e2_pairs = [], []
        for roi in roi_ids:
            a1 = activity[e1][all_roi_ids[e1].index(roi)]
            a2 = activity[e2][all_roi_ids[e2].index(roi)]
            roi1 = rois_by_id[e1][roi]
            roi2 = rois_by_id[e2][roi]

            value = min(a1, a2) / max(a1, a2)

            data_dict = {'value': value,
                         'value1': a1,
                         'value2': a2,
                         'first_expt': e1,
                         'second_expt': e2,
                         'first_roi': roi1,
                         'second_roi': roi2}
            data_list.append(data_dict)

            e1_pairs.append((a1, roi1))
            e2_pairs.append((a2, roi2))

        if shuffle:
            shuffle_dicts.extend([
                {'expts': (e1, e2), 'rois': (r1, r2), 'data': (d1, d2)}
                for (d1, r1), (d2, r2) in it.product(e1_pairs, e2_pairs)])

    if shuffle:
        if len(shuffle_dicts) < N_SHUFFLES:
            shuffler = shuffle_dicts
        else:
            shuffler = sample(shuffle_dicts, N_SHUFFLES)

        shuffle_list = []
        for pair in shuffler:
            value = min(*pair['data']) / max(*pair['data'])

            shuffle_dict = {'value': value,
                            'first_expt': pair['expts'][0],
                            'second_expt': pair['expts'][1],
                            'first_roi': pair['rois'][0],
                            'second_roi': pair['rois'][1]}
            shuffle_list.append(shuffle_dict)

    data_df = pd.DataFrame(data_list, columns=[
        'first_expt', 'second_expt', 'first_roi', 'second_roi', 'value'])
    shuffle_df = pd.DataFrame(shuffle_list, columns=[
        'first_expt', 'second_expt', 'first_roi', 'second_roi', 'value']) \
        if shuffle else None

    return data_df, shuffle_df


def plot_activity_stability_correlation(
        ax, exptGrp, activity_metric, stability_metric, stability_kwargs=None,
        activity_combine_method='mean', activity_filter='pc_either',
        roi_filter=None, z_score=True):
    """Scatter plots the activity of an ROI against it's stability

    Arguments:
    activity_metric: any metric argument to calc_activity
    stability_metric: any metric argument to place_field_stability_df,
        except similarity

    Note:
    To look at subsets of ROI pairs, pass in a paired_pcExperimentGroup

    """

    if stability_metric == population_vector_correlation:
        raise ValueError('Stability metric must be a per-cell metric')
    if stability_kwargs is None:
        stability_kwargs = {}

    activity = {}
    rois = {}
    for expt in exptGrp:
        # Calculate the desired activity metric and average across trials
        activity[expt] = calc_activity(
            expt, activity_metric, dF='from_file',
            channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'], roi_filter=roi_filter).mean(1)
        rois[expt] = expt.rois(
            channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'], roi_filter=roi_filter)

    data, _ = stability_metric(exptGrp, roi_filter, **stability_kwargs)

    activity_to_plot = []
    stability_to_plot = []

    for index, row in data.iterrows():
        e1 = row['first_expt']
        e2 = row['second_expt']
        r1 = row['first_roi']
        r2 = row['second_roi']

        a1 = activity[e1][rois[e1].index(r1)]
        a2 = activity[e2][rois[e2].index(r2)]

        if activity_combine_method == 'mean':
            row_activity = np.mean([a1, a2])
        elif activity_combine_method == 'first':
            row_activity = a1
        else:
            raise ValueError('Unrecognized method argument')

        if np.isfinite(row_activity) and np.isfinite(row['value']):
            activity_to_plot.append(row_activity)
            stability_to_plot.append(row['value'])

    activity_to_plot = np.array(activity_to_plot)
    stability_to_plot = np.array(stability_to_plot)

    if z_score:
        activity_to_plot -= np.mean(activity_to_plot)
        activity_to_plot /= np.std(activity_to_plot)

        stability_to_plot -= np.mean(stability_to_plot)
        stability_to_plot /= np.std(stability_to_plot)

    # Determine stability label
    if stability_metric == overlap:
        stability_label = \
            stability_kwargs.get('activity_method', '?') + ' overlap: '
    else:
        stability_label = stability_metric.__name__ + ': '
    if stability_metric == centroid_shift:
        stability_label += 'pc_both'
    elif activity_filter is None:
        stability_label += 'all_cells'
    else:
        stability_label += activity_filter
    if z_score:
        stability_label += ' (z-score)'

    plotting.scatterPlot(
        ax, [activity_to_plot, stability_to_plot],
        [activity_metric, stability_label], plotEqualLine=True,
        print_stats=True)

    return activity_to_plot, stability_to_plot


def plot_recur_vs_non_recur_activity(
        ax, exptGrp, metric_fn, fn_kwargs=None, roi_filter=None,
        groupby=None, plotby=('pair_condition',),
        orderby='pair_condition_order', z_score=False, circ_var_pcs=False,
        **plot_kwargs):
    """Compare the stability of cells that are/become place cells versus those
    that are not.

    """
    # TODO: pairing is not necessarily consecutive days...

    if fn_kwargs is None:
        fn_kwargs = {}

    # dict by roi id
    roi_activity_values = {}
    roi_activity_values

    day_minus1 = {}
    day_minus1['non_pcs'] = []
    day_minus1['pcs'] = []
    day_zero = {}
    day_zero['non_pcs'] = []
    day_zero['pcs'] = []
    day_plus1 = {}
    day_plus1['non_pcs'] = []
    day_plus1['pcs'] = []

    pcs_filter = exptGrp.pcs_filter(
        roi_filter=roi_filter, circ_var=circ_var_pcs)

    data = metric_fn(exptGrp, roi_filter=pcs_filter, **fn_kwargs)
    plotting.prepare_dataframe(data, include_columns=['expt', 'roi_id'])

    for e1, e2 in exptGrp.genImagedExptPairs():
        # Removed shared filer so: m1_npc is NOT:
        #     roi.id not in e2_pc_ids_set for roi in e1_roi_ids
        e1_pc_ids_set = set(e1.roi_ids(roi_filter=pcs_filter))
        e2_pc_ids_set = set(e2.roi_ids(roi_filter=pcs_filter))
        e1_pc_id_filter = lambda roi_id: roi_id in e1_pc_ids_set
        e2_pc_id_filter = lambda roi_id: roi_id in e2_pc_ids_set

        e1_npc_ids_set = set(
            e1.roi_ids(roi_filter=roi_filter)).difference(e1_pc_ids_set)
        e2_npc_ids_set = set(
            e2.roi_ids(roi_filter=roi_filter)).difference(e2_pc_ids_set)
        e1_npc_id_filter = lambda roi_id: roi_id in e1_npc_ids_set
        e2_npc_id_filter = lambda roi_id: roi_id in e2_npc_ids_set

        m1_pc = data[(data['expt'] == e2) &
                     (data['roi_id'].apply(e1_pc_id_filter))]

        m1_npc = data[(data['expt'] == e2) &
                      (data['roi_id'].apply(e1_npc_id_filter))]

        p1_pc = data[(data['expt'] == e1) &
                     (data['roi_id'].apply(e2_pc_id_filter))]

        p1_npc = data[(data['expt'] == e1) &
                      (data['roi_id'].apply(e2_npc_id_filter))]

        if z_score:
            m1_mean = np.nanmean(np.hstack([m1_pc['value'], m1_npc['value']]))
            p1_mean = np.nanmean(np.hstack([p1_pc['value'], p1_npc['value']]))

            m1_std = np.nanstd(np.hstack([m1_pc['value'], m1_npc['value']]))
            p1_std = np.nanstd(np.hstack([p1_pc['value'], p1_npc['value']]))

            m1_pc['value'] = m1_pc['value'] - m1_mean
            p1_pc['value'] = p1_pc['value'] - p1_mean

            m1_pc['value'] = m1_pc['value'] / m1_std
            p1_pc['value'] = p1_pc['value'] / p1_std

            m1_npc['value'] = m1_npc['value'] - m1_mean
            p1_npc['value'] = p1_npc['value'] - p1_mean

            m1_npc['value'] = m1_npc['value'] / m1_std
            p1_npc['value'] = p1_npc['value'] / p1_std

        day_minus1['pcs'].append(m1_pc)
        day_minus1['non_pcs'].append(m1_npc)
        day_plus1['pcs'].append(p1_pc)
        day_plus1['non_pcs'].append(p1_npc)

    if z_score:
        d0_mean = np.nanmean(data['value'])
        d0_std = np.nanstd(data['value'])
        data['value'] = data['value'] - d0_mean
        data['value'] = data['value'] / d0_std

    day_zero['pcs'] = data

    day_minus1_pc_df = pd.concat(day_minus1['pcs'])
    day_minus1_pc_df['pair_condition'] = 'Previous'
    day_minus1_pc_df['pair_condition_order'] = -1
    day_minus1_pc_df['is_place_cell'] = 'place cell'

    day_minus1_npc_df = pd.concat(day_minus1['non_pcs'])
    day_minus1_npc_df['pair_condition'] = 'Previous'
    day_minus1_npc_df['pair_condition_order'] = -1
    day_minus1_npc_df['is_place_cell'] = 'not a place cell'

    day_plus1_pc_df = pd.concat(day_plus1['pcs'])
    day_plus1_pc_df['pair_condition'] = 'Next'
    day_plus1_pc_df['pair_condition_order'] = 1
    day_plus1_pc_df['is_place_cell'] = 'place cell'

    day_plus1_npc_df = pd.concat(day_plus1['non_pcs'])
    day_plus1_npc_df['pair_condition'] = 'Next'
    day_plus1_npc_df['pair_condition_order'] = 1
    day_plus1_npc_df['is_place_cell'] = 'not a place cell'

    day_0_pc_df = day_zero['pcs']
    day_0_pc_df['pair_condition'] = 'Current'
    day_0_pc_df['pair_condition_order'] = 0
    day_0_pc_df['is_place_cell'] = 'place cell'

    pc_df = pd.concat([day_minus1_pc_df, day_plus1_pc_df, day_0_pc_df])
    n_pc_df = pd.concat([day_minus1_npc_df, day_plus1_npc_df])

    if ax is not None:
        plotting.plot_dataframe(
            ax, [pc_df, n_pc_df], labels=['place cell', 'not a place cell'],
            groupby=groupby, plotby=plotby, orderby=orderby,
            plot_method='grouped_bar', **plot_kwargs)

    return pc_df, n_pc_df


def plot_activity_versus_place_coding(
        ax, exptGrp, metric_fn, fn_kwargs=None, roi_filter=None,
        z_score=True, circ_var_pcs=False, **plot_kwargs):
    """Compare the stability of cells that are/become place cells versus those
    that are not.

    Determines if various metrics can predict future place cells

    """
    # TODO: pairing is not necessarily consecutive days...

    if fn_kwargs is None:
        fn_kwargs = {}

    # dict by roi id
    roi_activity_values = {}
    roi_activity_values

    day_minus1 = {}
    day_minus1['non_pcs'] = []
    day_minus1['pcs'] = []
    day_zero = {}
    day_zero['non_pcs'] = []
    day_zero['pcs'] = []
    day_plus1 = {}
    day_plus1['non_pcs'] = []
    day_plus1['pcs'] = []

    pcs_filter = exptGrp.pcs_filter(
        roi_filter=roi_filter, circ_var=circ_var_pcs)

    data = metric_fn(exptGrp, roi_filter=roi_filter, **fn_kwargs)
    plotting.prepare_dataframe(data, include_columns=['expt', 'roi_id'])

    for e1, e2 in exptGrp.genImagedExptPairs():
        # Removed shared filer so: m1_npc is NOT:
        #     roi.id not in e2_pc_ids_set for roi in e1_roi_ids
        e1_pc_ids_set = set(e1.roi_ids(roi_filter=pcs_filter))
        e2_pc_ids_set = set(e2.roi_ids(roi_filter=pcs_filter))
        e1_pc_id_filter = lambda roi_id: roi_id in e1_pc_ids_set
        e2_pc_id_filter = lambda roi_id: roi_id in e2_pc_ids_set

        e1_npc_ids_set = set(
            e1.roi_ids(roi_filter=roi_filter)).difference(e1_pc_ids_set)
        e2_npc_ids_set = set(
            e2.roi_ids(roi_filter=roi_filter)).difference(e2_pc_ids_set)
        e1_npc_id_filter = lambda roi_id: roi_id in e1_npc_ids_set
        e2_npc_id_filter = lambda roi_id: roi_id in e2_npc_ids_set

        m1_pc = data[(data['expt'] == e1) &
                     (data['roi_id'].apply(e2_pc_id_filter))]['value']

        m1_npc = data[(data['expt'] == e1) &
                      (data['roi_id'].apply(e2_npc_id_filter))]['value']

        p1_pc = data[(data['expt'] == e2) &
                     (data['roi_id'].apply(e1_pc_id_filter))]['value']

        p1_npc = data[(data['expt'] == e2) &
                      (data['roi_id'].apply(e1_npc_id_filter))]['value']

        if z_score:
            m1_mean = np.nanmean(np.hstack([m1_pc, m1_npc]))
            p1_mean = np.nanmean(np.hstack([p1_pc, p1_npc]))

            m1_std = np.nanstd(np.hstack([m1_pc, m1_npc]))
            p1_std = np.nanstd(np.hstack([p1_pc, p1_npc]))

            m1_pc -= m1_mean
            p1_pc -= p1_mean

            m1_pc /= m1_std
            p1_pc /= p1_std

            m1_npc -= m1_mean
            p1_npc -= p1_mean

            m1_npc /= m1_std
            p1_npc /= p1_std

        day_minus1['pcs'].extend(m1_pc)
        day_minus1['non_pcs'].extend(m1_npc)
        day_plus1['pcs'].extend(p1_pc)
        day_plus1['non_pcs'].extend(p1_npc)

    for e in exptGrp:
        pcs = set(e.roi_ids(roi_filter=pcs_filter))
        pcs_id_filter = lambda roi_id: roi_id in pcs
        npcs = set(e1.roi_ids(roi_filter=roi_filter)).difference(pcs)
        npcs_id_filter = lambda roi_id: roi_id in npcs

        d0_pc = data[(data['expt'] == e) &
                     (data['roi_id'].apply(pcs_id_filter))]['value']

        d0_npc = data[(data['expt'] == e) &
                      (data['roi_id'].apply(npcs_id_filter))]['value']

        if z_score:
            d0_mean = np.nanmean(np.hstack([d0_pc, d0_npc]))
            d0_std = np.nanstd(np.hstack([d0_pc, d0_npc]))

            d0_pc -= d0_mean
            d0_pc /= d0_std

            d0_npc -= d0_mean
            d0_npc /= d0_std

        day_zero['pcs'].extend(d0_pc)
        day_zero['non_pcs'].extend(d0_npc)

    values = [[day_minus1['non_pcs'],
               day_zero['non_pcs'],
               day_plus1['non_pcs']],
              [day_minus1['pcs'],
               day_zero['pcs'],
               day_plus1['pcs']]]

    plotting.grouped_bar(ax, values, ['non place cells', 'place cells'],
                         ['Previous', 'Current', 'Next'], **plot_kwargs)

    return values


def get_activity_label(metric_fn, activity_kwargs=None):
    if activity_kwargs is None:
        activity_kwargs = {}
    if metric_fn == sensitivity:
        if activity_kwargs.get('includeFrames', '') == 'running_only':
            return 'sensitivity during running by lap'
        else:
            return 'sensitivity by lap'
    if metric_fn == specificity:
        if activity_kwargs.get('includeFrames', '') == 'running_only':
            return 'specificity during running by lap'
        else:
            return 'specificity by lap'
    if metric_fn == place_field_width:
        return 'place field width'
    if metric_fn == population_activity:
        # This assumes that the default value for 'running_only' is False in
        # calc_activity_statistic
        running_only = activity_kwargs.get('running_only', False)
        if 'interval' in activity_kwargs:
            if activity_kwargs['interval'] == 'pf':
                return '{} in place fields{}'.format(
                    activity_kwargs['stat'],
                    ': running only' if running_only else '')
            elif activity_kwargs['interval'] == 'non pf':
                return '{} not in place fields{}'.format(
                    activity_kwargs['stat'],
                    ': running only' if running_only else '')
            elif activity_kwargs['interval'] == 'all':
                return '{} across all frames{}'.format(
                    activity_kwargs['stat'],
                    ': running only' if running_only else '')
            else:
                return '{} in interval{}'.format(
                    activity_kwargs['stat'],
                    ': running only' if running_only else '')
        else:
            return '{} across all frames{}'.format(
                activity_kwargs['stat'],
                ': running only' if running_only else '')
    if metric_fn == spatial_information:
        return 'spatial information'
    if metric_fn == sparsity:
        return 'single cell sparsity'
    if metric_fn == n_place_fields:
        return 'number of place fields'
    if metric_fn == n_sessions_imaged:
        return 'number of sessions imaged'
    if metric_fn == place_cell_percentage:
        return 'fraction place cells'
    if metric_fn == place_field_correlation:
        return 'PF correlation'
    if metric_fn == population_vector_correlation:
        if activity_kwargs.get('method', False):
            if activity_kwargs['method'] == 'angle':
                return 'PV correlation (angle)'
            elif activity_kwargs['method'] == 'corr':
                return 'PV correlation (corr)'
        return 'PV correlation'
    if metric_fn == recurrence_probability:
        return 'recurrence probability'
    if metric_fn == overlap:
        return 'overlap score ({})'.format(
            activity_kwargs.get('activity_method', 'unknown'))
    if metric_fn == centroid_shift:
        return 'centroid shift'
    if metric_fn == activity_centroid_shift:
        return 'activity centroid shift'
    if metric_fn == circular_variance:
        return 'circular variance'


def plot_acute_remapping_metric(
        ax, exptGrps_list, metric_fn, plot_method, roi_filters_list=None,
        group_labels=None, groupby=None, plotby=None, orderby=None,
        colorby=None, plot_shuffle=False, shuffle_plotby=False,
        pool_shuffle=False, plot_abs=False, activity_kwargs=None,
        **plot_kwargs):
    """Plotting function for acute remapping experiments.

    ExptGrps should already be split in to separate sub groups. You can plotby
    the different conditions with the 'condition' plotby column argument.

    exptGrps_list -- a list of lists of exptGrps. First index is the group and
        the second are the separate acute conditions
    roi_filters_list -- a list of list of filters, matched in shape to
        exptGrps_list
    group_labels -- labels for the overall exptGrps (not the individual
        conditions)

    See plot_place_cell_metric for description of additional arguments.

    """

    condition_order = {'SameAll': 0, 'DiffCtxs': 1, 'DiffAll': 2}

    if roi_filters_list is None:
        roi_filters_list = [
            [None] * len(conditions) for conditions in exptGrps_list]

    if group_labels is None:
        group_labels = ['Group ' + str(x) for x in range(len(exptGrps_list))]

    if activity_kwargs is None:
        activity_kwargs = {}

    if groupby is not None:
        include_columns = set(
            column for groupby_list in groupby for column in groupby_list)
    else:
        include_columns = set()
    if plotby is not None:
        include_columns.update(plotby)
    if orderby is not None:
        include_columns.update(orderby)
    try:
        include_columns.remove('condition')
    except KeyError:
        pass
    try:
        include_columns.remove('condition_order')
    except KeyError:
        pass

    dataframes, shuffles = [], []
    for exptGrps, roi_filters in it.izip(exptGrps_list, roi_filters_list):
        grp_data, grp_shuffle = [], []
        for condition_idx, condition_grp, roi_filter in it.izip(
                it.count(), exptGrps, roi_filters):

            condition_label = condition_grp.label() if condition_grp.label() \
                else "Condition " + str(condition_idx)

            data, shuffle = metric_fn(
                condition_grp, roi_filter=roi_filter, **activity_kwargs)

            try:
                prepped_data = plotting.prepare_dataframe(
                    data, include_columns=include_columns)
            except plotting.InvalidDataFrame:
                prepped_data = pd.DataFrame()
            prepped_data['condition'] = condition_label
            prepped_data['condition_order'] = \
                condition_order.get(condition_label, 10)
            grp_data.append(prepped_data)

            if plot_shuffle:
                try:
                    prepped_shuffle = plotting.prepare_dataframe(
                        shuffle, include_columns=include_columns)
                except plotting.InvalidDataFrame:
                    prepped_shuffle = pd.DataFrame()
                prepped_shuffle['condition'] = condition_label
                prepped_shuffle['condition_order'] = \
                    condition_order.get(condition_label, 10)
                grp_shuffle.append(prepped_shuffle)

        dataframes.append(pd.concat(grp_data))
        if plot_shuffle:
            shuffles.append(pd.concat(grp_shuffle))

    try:
        plotting.plot_dataframe(
            ax, dataframes, shuffles if plot_shuffle else None,
            activity_label=get_activity_label(metric_fn, activity_kwargs),
            groupby=groupby, plotby=plotby, orderby=orderby, colorby=colorby,
            plot_method=plot_method, plot_shuffle=plot_shuffle,
            shuffle_plotby=shuffle_plotby, pool_shuffle=pool_shuffle,
            labels=group_labels, **plot_kwargs)
    except plotting.InvalidDataFrame:
        pass
    return {label: df for label, df in zip(group_labels, dataframes)}


def plot_spatial_tuning_overlay(
        ax, exptGrp, plane=0, roi_filter=None, labels_visible=True,
        cax=None, alpha=0.2, **kwargs):
    """Plot place cell spatial tuning for a single expt exptGrp"""

    pcs_filter = exptGrp.pcs_filter(roi_filter=roi_filter)
    centroids = calcCentroids(
        exptGrp.data(roi_filter=pcs_filter)[exptGrp[0]],
        exptGrp.pfs(roi_filter=pcs_filter)[exptGrp[0]])
    nPositionBins = exptGrp.args['nPositionBins']
    centroid_vals = np.array([x[0] for x in centroids]) / float(nPositionBins)

    background_figure = exptGrp[0].returnFinalPrototype(
        channel=exptGrp.args['channel'])[plane, ...]
    roiVerts = exptGrp[0].roiVertices(
        channel=exptGrp.args['channel'], label=exptGrp.args['imaging_label'],
        roi_filter=pcs_filter)

    if not len(roiVerts):
        return

    imaging_parameters = exptGrp[0].imagingParameters()
    aspect_ratio = imaging_parameters['pixelsPerLine'] \
        / imaging_parameters['linesPerFrame']

    roi_inds = [i for i, v in enumerate(roiVerts) if v[0][0][2] == plane]
    plane_verts = np.array(roiVerts)[roi_inds].tolist()
    twoD_verts = []
    for roi in plane_verts:
        roi_polys = []
        for poly in roi:
            roi_polys.append(np.array(poly)[:, :2])
        twoD_verts.append(roi_polys)

    if labels_visible:
        pcLabels = exptGrp.roi_ids(
            channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'],
            roi_filter=pcs_filter)[exptGrp[0]]
        pcLabels = np.array(pcLabels)[roi_inds].tolist()
    else:
        pcLabels = None

    plotting.roiDataImageOverlay(
        ax, background_figure, twoD_verts,
        values=centroid_vals, vmin=0, vmax=1, labels=pcLabels, cax=cax,
        alpha=alpha, aspect=aspect_ratio, **kwargs)

    ax.set_title('Spatial tuning of place cells\nPlane {}'.format(plane))


def place_imaging_animation(
        expt, ax, n_position_bins=100, running_kwargs=None, channel='Ch2',
        **plot_kwargs):
    """Creates an animation where time is position on the belt and each
    frame is the average activity across all running frames at that
    position

    """

    if running_kwargs is None:
        running_kwargs = {}

    running_frames = expt.runningIntervals(
        imageSync=True, direction='forward', returnBoolList=True,
        **running_kwargs)

    imaging_dataset = expt.imaging_dataset()
    ch_idx = imaging_dataset.channel_names.index(channel)

    position_sums = np.zeros(
        (n_position_bins, imaging_dataset.num_rows,
         imaging_dataset.num_columns))
    position_counts = np.zeros(
        (n_position_bins, imaging_dataset.num_rows,
         imaging_dataset.num_columns), dtype=int)

    for trial, cycle, cycle_running in it.izip(
            expt, imaging_dataset, running_frames):
        position = trial.behaviorData(imageSync=True)['treadmillPosition']
        for frame, pos in it.compress(
                it.izip(cycle, position), cycle_running):
            pos_bin = int(pos * n_position_bins)
            non_nan_pixels = np.isfinite(frame[ch_idx])
            frame[ch_idx][np.isnan(frame[ch_idx])] = 0
            position_sums[pos_bin] += frame[ch_idx]
            position_counts[pos_bin] += non_nan_pixels.astype(int)

    position_average_movie = position_sums / position_counts

    imaging_parameters = expt.imagingParameters()
    aspect_ratio = imaging_parameters['pixelsPerLine'] \
        / imaging_parameters['linesPerFrame']

    image = ax.imshow(
        position_average_movie[0], cmap='gray', interpolation='none',
        aspect=aspect_ratio, **plot_kwargs)

    ax.set_axis_off()

    for frame in position_average_movie:
        image.set_data(frame)
        yield


def place_cell_tuning_animation(
        expt, ax, channel='Ch2', label=None, roi_filter=None,
        n_position_bins=100, add_end_frame=False):
    """Animation over belt position showing the spatial tuning of all place cells

    add_end_frame -- If True, adds an extra frame at end of the movie showing
        all place_cells shaded

    """

    background = expt.returnFinalPrototype(channel=channel)[..., 0]

    imaging_parameters = expt.imagingParameters()
    aspect_ratio = imaging_parameters['pixelsPerLine'] \
        / imaging_parameters['linesPerFrame']

    ax.imshow(
        background, cmap='gray', interpolation='none', aspect=aspect_ratio,
        zorder=0)

    exptGrp = pcExperimentGroup(
        [expt], channel=channel, imaging_label=label,
        nPositionBins=n_position_bins)

    pcs_filter = exptGrp.pcs_filter(roi_filter=roi_filter)

    data = exptGrp.data(roi_filter=pcs_filter)[expt]

    centroids, pfs = calcCentroids(
        data, exptGrp.pfs(roi_filter=pcs_filter)[expt], return_pfs=True)

    data /= np.amax(data, axis=1)[:, None]

    pf_mask = np.empty_like(data)
    pf_mask.fill(False)

    for pf_idx, pf in enumerate(pfs):
        if pf[0][0] < pf[0][1]:
            pf_mask[pf_idx, pf[0][0]:pf[0][1] + 1] = True
        else:
            pf_mask[pf_idx, pf[0][0]:] = True
            pf_mask[pf_idx, :pf[0][1] + 1] = True

    data *= pf_mask

    # Gather all the roi masks and color them
    # Probably would have been easier to keep the mask and color separate
    # until plotting, but this works
    roi_masks = []
    rois = expt.rois(
        roi_filter=pcs_filter, channel=exptGrp.args['channel'],
        label=exptGrp.args['imaging_label'])

    assert len(rois) == len(centroids)

    for roi, centroid in it.izip(rois, centroids):
        roi_mask = np.array(roi.mask[0].todense())
        color = np.array(plt.cm.hsv(centroid[0] / float(n_position_bins)))
        roi_masks.append(roi_mask[..., None] * color[None, None, :])
    roi_masks = np.array(roi_masks)

    rois = np.array(expt.roiVertices(roi_filter=pcs_filter))

    pf_image = ax.imshow(
        np.ma.masked_all_like(background), interpolation='none',
        aspect=aspect_ratio, zorder=1)

    ax.set_axis_off()

    for bin in data.T:
        bin_masks = roi_masks[bin > 0]
        if bin_masks.shape[0]:
            bin_masks_copy = bin_masks.copy()
            for mask_idx, roi, mask_mag in it.izip(
                    it.count(), rois[bin > 0], bin[bin > 0]):
                bin_masks_copy[mask_idx, :, :, 3] *= mask_mag
                # If this is the peak, add an ROI polygon
                if mask_mag == 1:
                    for poly in roi:
                        xlim = ax.get_xlim()
                        ylim = ax.get_ylim()
                        ax.plot(
                            poly[:, 0] - 0.5, poly[:, 1] - 0.5,
                            color=bin_masks[mask_idx].max(axis=0).max(axis=0),
                            zorder=2)
                        ax.set_xlim(xlim)
                        ax.set_ylim(ylim)

            bin_mask = bin_masks_copy.sum(axis=0)
            masked_pixels = ~np.any(bin_mask, axis=2)
            masked_pixels = np.tile(masked_pixels[..., None], (1, 1, 4))
            pf_image.set_data(np.ma.masked_where(masked_pixels, bin_mask))

        yield

    if add_end_frame:
        all_masks = roi_masks.sum(axis=0)
        all_masks[..., 3] *= 0.7  # Add some alpha
        masked_pixels = ~np.any(all_masks, axis=2)
        masked_pixels = np.tile(masked_pixels[..., None], (1, 1, 4))
        pf_image.set_data(np.ma.masked_where(masked_pixels, all_masks))

        yield


@memoize
def centroid_to_position_distance(
        exptGrp, positions, roi_filter=None, multiple_fields='closest',
        multiple_positions='closest', return_abs=False):
    """Calculates the distance from the centroid of each place field to any
    positions. Values < 0 denote the pf preceded the reward, > 0 followed.

    Parameters
    ----------
    positions : array or 'reward' or 'A', 'B', 'C',...
        Positions to calculate distance from. Either a list of positions
        (normalized) or 'reward' to use the reward locations.  If a single
        character is passed, use the reward position corresponding to
        that condition (calc'd on a per mouse basis and assuming 1 reward
        position per condition per mouse)
    multiple_fields : ['closest', 'largest']
        Determines hot to handle a PC with multiple fields, either return the
        closest field or the largest field.
    multiple_positions : ['closest']
        Determines how to handle multiple positions. Currently only 'closest'
        is implemented, where the closest position to the place field is
        considered.
    return_abs : boolean
        If True, returns absolute value of the distance

    """

    n_position_bins = float(exptGrp.args['nPositionBins'])

    if isinstance(positions, basestring):
        if positions == 'reward':
            rewards_by_expt = {
                expt: expt.rewardPositions(units='normalized')
                for expt in exptGrp}
        else:
            rewards_by_expt = rewards_by_condition(
                exptGrp, positions, condition_column='condition')
    else:
        rewards_by_expt = defaultdict(
            lambda: np.array(positions).astype(float))

    exptGrp_pfs = exptGrp.pfs(roi_filter=roi_filter)
    exptGrp_data = exptGrp.data(roi_filter=roi_filter)

    data_list = []
    for expt in exptGrp:
        calc_positions = rewards_by_expt[expt].copy()
        if calc_positions is None:
            continue
        # If any positions are >=1, they should be tick counts
        if np.any(calc_positions >= 1.):
            track_length = np.mean(
                [trial.behaviorData()['trackLength']
                 for trial in expt.findall('trial')])
            calc_positions /= track_length
        rois = expt.rois(
            roi_filter=roi_filter, channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'])
        data = exptGrp_data[expt]
        pfs = exptGrp_pfs[expt]
        centroids = calcCentroids(data, pfs, returnAll=True)
        assert len(rois) == len(centroids)
        for roi, roi_centroids in it.izip(rois, centroids):
            if not len(roi_centroids):
                continue
            centroid_distances = []
            for centroid in roi_centroids:
                centroid /= n_position_bins
                diffs = centroid - calc_positions
                diffs[diffs >= 0.5] -= 1.
                diffs[diffs < -0.5] += 1.
                if multiple_positions == 'closest':
                    centroid_distances.append(
                        diffs[np.argmin(np.abs(diffs))])
                else:
                    raise ValueError
                if multiple_fields == 'largest':
                    # calcCentroids should return pfs sorted by peak, so only
                    # consider the first pf for each ROI
                    break
            if multiple_fields == 'closest':
                distance = centroid_distances[
                    np.argmin(np.abs(centroid_distances))]
            elif multiple_fields == 'largest':
                distance = centroid_distances[0]
            else:
                raise ValueError
            data_list.append({'expt': expt, 'roi': roi, 'value': distance})

    dataframe = pd.DataFrame(data_list, columns=['expt', 'roi', 'value'])
    if return_abs:
        dataframe['value'] = dataframe['value'].abs()
    return dataframe


@memoize
def centroid_to_position_threshold(
        exptGrp, positions, threshold, method='centroid', **kwargs):

    if method == 'centroid':
        dataframe = centroid_to_position_distance(
            exptGrp, positions=positions, return_abs=True, **kwargs)
    elif method == 'resultant_vector':
        dataframe = mean_resultant_vector_to_position_angle(
            exptGrp, positions=positions, **kwargs)
    else:
        raise ValueError('Unrecognized centroid method')

    assert dataframe['value'].min() >= 0
    dataframe['distance'] = dataframe['value']
    dataframe['value'] = dataframe['distance'].apply(
        lambda x: int(x < threshold))

    return dataframe


@memoize
def mean_resultant_vector_to_position_angle(
        exptGrp, positions, roi_filter=None, multiple_positions='closest',
        pcs_only=False, circ_var_pcs=True, method='vector_angle'):
    """Calculates the angle between the mean resultant vector and any
    positions.

    Parameters
    ----------
    positions : array or 'reward' or 'A', 'B', 'C',...
        Positions to calculate distance from. Either a list of positions
        (normalized) or 'reward' to use the reward locations.  If a single
        character is passed, use the reward position corresponding to
        that condition (calc'd on a per mouse basis and assuming 1 reward
        position per condition per mouse)
    multiple_positions : {'closest', 'mean'}
        Determines how to handle multiple positions 'closest' takes the closest
        position and 'mean' takes the average.
    pcs_only : bool
        If True, only return place cells (as determined by circular variance)
    circ_var_pcs : bool
        If True and pcs_only is True, use circular variance PC criteria
    method : {'vector_angle', 'angle_difference'}
        Method for determining the distance from activity to reward. Either the
        angle between firing and reward, on [0, pi), or difference between the
        angles, on [-pi, pi). Positive angle and vector is in front of
        positions, negative is behind.

    """

    def dotproduct(v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))

    def length(v):
        return math.sqrt(dotproduct(v, v))

    def angle(v1, v2):
        return math.acos(
            np.round(dotproduct(v1, v2) / (length(v1) * length(v2)), 3))

    if pcs_only:
        roi_filter = exptGrp.pcs_filter(
            roi_filter=roi_filter, circ_var=circ_var_pcs)

    vectors = calc_activity_centroids(exptGrp, roi_filter=roi_filter)

    if isinstance(positions, basestring):
        if positions == 'reward':
            rewards_by_expt = {
                expt: expt.rewardPositions(units='normalized')
                for expt in exptGrp}
        else:
            rewards_by_expt = rewards_by_condition(
                exptGrp, positions, condition_column='condition')
    else:
        rewards_by_expt = defaultdict(
            lambda: np.array(positions).astype(float))

    data_list = []
    for expt in exptGrp:
        expt_positions = rewards_by_expt[expt].copy()
        if expt_positions is None:
            continue
        # If any positions are >=1, they should be tick counts
        if np.any(expt_positions >= 1.):
            track_length = np.mean(
                [trial.behaviorData()['trackLength']
                 for trial in expt.findall('trial')])
            expt_positions /= track_length
        expt_positions *= 2 * np.pi
        expt_positions = [np.complex(x, y) for x, y in zip(
            np.cos(expt_positions), np.sin(expt_positions))]
        expt_angles = np.angle(expt_positions)

        rois = expt.rois(
            roi_filter=roi_filter, channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'])
        assert len(rois) == len(vectors[expt])
        for roi, roi_resultant in it.izip(rois, vectors[expt]):
            if method == 'vector_angle':
                angles = [angle([roi_resultant.real, roi_resultant.imag],
                                [pos.real, pos.imag])
                          for pos in expt_positions]
            elif method == 'angle_difference':
                angles = np.array(
                    [np.angle(roi_resultant) - ang for ang in expt_angles])
                angles[angles < -np.pi] += 2 * np.pi
                angles[angles >= np.pi] -= 2 * np.pi

            if multiple_positions == 'closest':
                closest_position = np.argmin(np.abs(angles))
                min_angle = angles[closest_position]
                data_list.append(
                    {'expt': expt, 'roi': roi,
                     'closest_position_idx': closest_position,
                     'value': min_angle})
            elif multiple_positions == 'mean':
                raise NotImplementedError
            else:
                raise ValueError

    return pd.DataFrame(data_list, columns=[
        'expt', 'roi', 'closest_position_idx', 'value'])


@memoize
def distance_to_position_shift(
        exptGrp, roi_filter=None, method='resultant', **kwargs):

    if method == 'resultant':
        distance_fn = mean_resultant_vector_to_position_angle
    elif method == 'centroid':
        distance_fn = centroid_to_position_distance
    else:
        raise ValueError('Unrecognized method argument: {}'.format(method))

    data = distance_fn(exptGrp, roi_filter=roi_filter, **kwargs)
    plotting.prepare_dataframe(data, include_columns=['expt', 'roi_id'])

    df_list = []
    for e1, e2 in exptGrp.genImagedExptPairs():
        paired_df = pd.merge(
            data[data['expt'] == e1], data[data['expt'] == e2], on=['roi_id'],
            suffixes=('_e1', '_e2'))

        paired_df['value'] = paired_df['value_e2'] - paired_df['value_e1']

        paired_df['first_roi'] = paired_df['roi_e1']
        paired_df['second_roi'] = paired_df['roi_e2']
        paired_df['first_expt'] = paired_df['expt_e1']
        paired_df['second_expt'] = paired_df['expt_e2']

        paired_df = paired_df.drop(
            ['roi_e1', 'roi_e2', 'expt_e1', 'expt_e2', 'value_e1', 'value_e2',
             'roi_id'], axis=1)

        df_list.append(paired_df)

    return pd.concat(df_list, ignore_index=True)


def metric_correlation(
        expt_grp, first_metric_fn, second_metric_fn, correlate_by,
        groupby=None, roi_filter=None, first_metric_kwargs=None,
        second_metric_kwargs=None, method='pearson_r'):

    if groupby is None:
        groupby = [[]]

    if first_metric_kwargs is None:
        first_metric_kwargs = {}
    if second_metric_kwargs is None:
        second_metric_kwargs = {}

    try:
        first_metric_data = first_metric_fn(
            expt_grp, roi_filter=roi_filter, **first_metric_kwargs)
    except TypeError:
        first_metric_data = first_metric_fn(expt_grp, **first_metric_kwargs)
    if not isinstance(first_metric_data, pd.DataFrame) and \
            (len(first_metric_data) == 2 and
             isinstance(first_metric_data[0], pd.DataFrame) and
             (isinstance(first_metric_data[1], pd.DataFrame) or
              first_metric_data[1] is None)):
        first_metric_data = first_metric_data[0]

    try:
        second_metric_data = second_metric_fn(
            expt_grp, roi_filter=roi_filter, **second_metric_kwargs)
    except TypeError:
        second_metric_data = second_metric_fn(expt_grp, **second_metric_kwargs)
    if not isinstance(second_metric_data, pd.DataFrame) and \
            (len(second_metric_data) == 2 and
             isinstance(second_metric_data[0], pd.DataFrame) and
             (isinstance(second_metric_data[1], pd.DataFrame) or
              second_metric_data[1] is None)):
        second_metric_data = second_metric_data[0]

    grouped_dfs = []
    for df in (first_metric_data, second_metric_data):
        for groupby_list in groupby:
            plotting.prepare_dataframe(df, include_columns=groupby_list)
            df = df.groupby(groupby_list, as_index=False).mean()
        plotting.prepare_dataframe(df, include_columns=correlate_by)
        grouped_dfs.append(df)

    merge_on = tuple(groupby[-1]) + tuple(correlate_by)
    merged_df = pd.merge(
        grouped_dfs[0], grouped_dfs[1], how='inner', on=merge_on,
        suffixes=('_1', '_2'))

    result_dicts = []
    for key, group in merged_df.groupby(correlate_by):
        if method.startswith('pearson_r'):
            corr, _ = pearsonr(group['value_1'], group['value_2'])
        elif method.startswith('spearman_r'):
            corr, _ = spearmanr(group['value_1'], group['value_2'])
        if method.endswith('_squared'):
            corr **= 2

        group_dict = {c: v for c, v in zip(correlate_by, key)}
        group_dict['value'] = corr
        result_dicts.append(group_dict)

    return pd.DataFrame(result_dicts)


@memoize
def cue_cell_remapping(
        expt_grp, roi_filter=None, near_threshold=0.05 * 2 * np.pi,
        activity_filter=None, circ_var_pcs=False, shuffle=True):

    N_SHUFFLES = 10000

    centroids = calc_activity_centroids(expt_grp, roi_filter=None)
    if activity_filter is not None:
        if 'pc' in activity_filter:
            pcs_filter = expt_grp.pcs_filter(circ_var=circ_var_pcs)
        elif 'active' in activity_filter:
            active_filter = filters.active_roi_filter(
                expt_grp, min_transients=1, channel=expt_grp.args['channel'],
                label=expt_grp.args['imaging_label'], roi_filter=roi_filter)

    def dotproduct(v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))

    def length(v):
        return math.sqrt(dotproduct(v, v))

    def angle(v1, v2):
        return math.acos(
            np.round(dotproduct(v1, v2) / (length(v1) * length(v2)), 3))

    def complex_from_angle(angle):
        return np.complex(np.cos(angle), np.sin(angle))

    def expt_cues(expt):
        cues = expt.belt().cues(normalized=True)
        assert np.all(cues.start < cues.stop)
        cues['pos'] = np.array([cues.start, cues.stop]).mean(axis=0)
        cues['pos_complex'] = (cues['pos'] * 2 * np.pi).apply(complex_from_angle)
        return cues

    data_list = []
    for e1, e2 in expt_grp.genImagedExptPairs():

        shared_rois = expt_grp.subGroup([e1, e2]).sharedROIs(
            channel=expt_grp.args['channel'],
            label=expt_grp.args['imaging_label'], roi_filter=roi_filter)
        if activity_filter == 'pc_either':
            e1_pcs = e1.roi_ids(roi_filter=pcs_filter)
            e2_pcs = e2.roi_ids(roi_filter=pcs_filter)
            roi_ids = list(set(shared_rois).intersection(
                set(e1_pcs).union(e2_pcs)))
        elif activity_filter == 'pc_both':
            e1_pcs = e1.roi_ids(roi_filter=pcs_filter)
            e2_pcs = e2.roi_ids(roi_filter=pcs_filter)
            roi_ids = list(set(shared_rois).intersection(
                           e1_pcs).intersection(e2_pcs))
        elif activity_filter == 'active_either':
            e1_active = e1.roi_ids(roi_filter=active_filter)
            e2_active = e2.roi_ids(roi_filter=active_filter)
            roi_ids = list(set(shared_rois).intersection(
                set(e1_active).union(e2_active)))
        elif activity_filter == 'active_both':
            e1_active = e1.roi_ids(roi_filter=active_filter)
            e2_active = e2.roi_ids(roi_filter=active_filter)
            roi_ids = list(set(shared_rois).intersection(
                set(e1_active).intersection(e2_active)))
        elif activity_filter:
            e1_rois = e1.roi_ids(roi_filter=activity_filter)
            e2_rois = e2.roi_ids(roi_filter=activity_filter)
            roi_ids = list(set(shared_rois).intersection(
                           set(e1_rois).union(set(e2_rois))))
        else:
            roi_ids = shared_rois

        for roi in roi_ids:

            c1 = centroids[e1][e1.roi_ids().index(roi)]
            c2 = centroids[e2][e2.roi_ids().index(roi)]

            # Activity centroid will be NaN if there was no transients
            if np.isnan(c1) or np.isnan(c2):
                continue

            cue_positions = expt_cues(e1)['pos_complex']
            if not len(cue_positions):
                continue
            cue_distances = [
                angle((pos.real, pos.imag),
                      (c1.real, c1.imag)) for pos in cue_positions]
            assert all(dist <= np.pi for dist in cue_distances)

            closest_cue_idx = np.argmin(cue_distances)
            closest_cue = expt_cues(e1)['cue'][closest_cue_idx]
            e1_distance = cue_distances[closest_cue_idx]

            if e1_distance > near_threshold:
                continue

            if closest_cue not in set(expt_cues(e2)['cue']):
                continue

            cue2_position = expt_cues(e2).ix[
                expt_cues(e2).cue == closest_cue, 'pos_complex'].values[0]

            e2_distance = angle(
                [c2.real, c2.imag], [cue2_position.real, cue2_position.imag])
            data_dict = {'first_expt': e1,
                         'second_expt': e2,
                         'first_centroid': c1,
                         'second_centroid': c2,
                         'roi_id': roi,
                         'cue': closest_cue,
                         'value': e2_distance}
            data_list.append(data_dict)

    result_df = pd.DataFrame(
        data_list, columns=['first_expt', 'second_expt', 'first_centroid',
                            'second_centroid', 'roi_id', 'cue', 'value'])

    if shuffle:
        shuffle_list = []
        for _, row in result_df.sample(
                N_SHUFFLES, replace=True, axis=0).iterrows():
            same_expts_df = result_df.loc[
                (result_df.first_expt == row.first_expt) &
                (result_df.second_expt == row.second_expt)]
            # Instead of matching roi_id, choose a random ROI
            random_row = same_expts_df.sample(1).iloc[0]
            # How far is the random ROI in the second expt from the cue
            # preceding the original ROI in the first expt?
            c2 = random_row.second_centroid
            e2_cues = expt_cues(row.second_expt)
            e2_cue_pos = e2_cues.ix[e2_cues.cue == row.cue, 'pos_complex'].values[0]
            e2_distance = angle(
                [c2.real, c2.imag], [e2_cue_pos.real, e2_cue_pos.imag])

            shuffle_list.append(
                {'value': e2_distance,
                 'first_expt': row.first_expt,
                 'second_expt': row.second_expt,
                 'first_roi_id': row.roi_id,
                 'second_roi_id': random_row.roi_id,
                 'first_centroid': row.first_centroid,
                 'second_centroid': c2,
                 'cue': row.cue})

        shuffle_df = pd.DataFrame(shuffle_list, columns=[
            'first_expt', 'second_expt', 'first_roi_id', 'second_roi_id',
            'first_centroid', 'second_centroid', 'cue', 'value'])
    else:
        shuffle_df = None

    return result_df, shuffle_df


@memoize
def place_field_centroid(
        expt_grp, roi_filter=None, normalized=False,
        drop_multi_peaked_pfs=False):
    """Return the position of the centroid of each place field.

    Parameters
    ----------
    expt_grp : lab.classes.pcExperimentGroup
    roi_filter : filter function
    normalized : bool
        If True, return the centroid as a normalized belt position on [0, 1),
        otherwise return the centroid in position bins.
    drop_multi_peaked_pfs : bool
        If True, drop any place cells that have multiple peaks, otherwise will
        return 1 row per place field.

    Returns
    -------
    pd.DataFrame
        expt : lab.Experiment
        roi : sima.ROI.ROI
        idx : int
            Index of place field centroid, sorted by size of place field peak
        value : float
            Position of place field centroid.


    """
    pfs = expt_grp.pfs(roi_filter=roi_filter)
    data = expt_grp.data(roi_filter=roi_filter)
    rois = expt_grp.rois(
        channel=expt_grp.args['channel'], label=expt_grp.args['imaging_label'],
        roi_filter=roi_filter)
    centroids = []
    for expt in expt_grp:
        expt_centroids = calcCentroids(data[expt], pfs[expt], returnAll=True)
        assert len(expt_centroids) == len(rois[expt])
        for roi_centroids, roi in zip(expt_centroids, rois[expt]):
            if drop_multi_peaked_pfs and len(roi_centroids) > 1:
                continue
            for centroid_idx, roi_centroid in enumerate(roi_centroids):
                centroids.append({'expt': expt,
                                  'roi': roi,
                                  'idx': centroid_idx,
                                  'value': roi_centroid})

    df = pd.DataFrame(centroids, columns=['expt', 'roi', 'idx', 'value'])

    if normalized:
        df['value'] /= expt_grp.args['nPositionBins']

    return df


@memoize
def place_field_gain(expt_grp, roi_filter=None):
    """Return the gain of each place cell.

    The 'gain' is defined as:
        max(tuning_curve) - min(tuning_curve)

    Returns
    -------
    pd.DataFrame

    """
    df_list = []
    rois = expt_grp.rois(
        channel=expt_grp.args['channel'], label=expt_grp.args['imaging_label'],
        roi_filter=roi_filter)
    tuning_curves = expt_grp.data(roi_filter=roi_filter)
    for expt in expt_grp:
        for roi, tuning_curve in zip(rois[expt], tuning_curves[expt]):
            df_list.append({'expt': expt, 'roi': roi,
                            'value': tuning_curve.max() - tuning_curve.min()})

    return pd.DataFrame(df_list, columns=['expt', 'roi', 'value'])
