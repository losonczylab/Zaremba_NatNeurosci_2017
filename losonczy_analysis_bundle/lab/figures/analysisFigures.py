"""Figure generating functions to accompany behavior_analysis,
used by automatic scripts

All functions should return either a figure or list of figures.

"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
from scipy.misc import comb
try:
    from bottleneck import nanmean, nanstd
except ImportError:
    from numpy import nanmean, nanstd
from warnings import warn
import pandas as pd
import seaborn.apionly as sns
from collections import defaultdict
from copy import copy

import lab
from ..analysis import imaging_analysis as ia
from ..analysis import signals_analysis as sa
from ..analysis import reward_analysis as ra
from ..analysis import intervals as inter
from ..analysis import filters as af
from ..classes.classes import ExperimentGroup as eg
from .. import plotting
from .. import misc
from ..plotting import plot_metric, plot_paired_metrics, color_cycle

import lab.plotting.analysis_plotting as ap


def activityByExposureFigure(exptGrp, rasterized=False, **kwargs):

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    ap.activityByExposure(
        exptGrp, ax=axs[0][0], stat='mean', rasterized=rasterized, **kwargs)
    ap.activityByExposure(
        exptGrp, ax=axs[0][1], stat='responseMagnitude',
        rasterized=rasterized, **kwargs)
    ap.activityByExposure(
        exptGrp, ax=axs[0][2], stat='norm transient auc2',
        rasterized=rasterized, **kwargs)
    ap.activityByExposure(
        exptGrp, ax=axs[1][0], stat='amplitude', rasterized=rasterized,
        **kwargs)
    ap.activityByExposure(
        exptGrp, ax=axs[1][1], stat='duration', rasterized=rasterized,
        **kwargs)
    ap.activityByExposure(
        exptGrp, ax=axs[1][2], stat='frequency', rasterized=rasterized,
        **kwargs)

    return fig


def activityComparisonFigure(exptGrp, method='mean', rasterized=False):

    nCols = 4

    exposure = exptGrp.priorDaysOfExposure(ignoreContext=False)

    pairs = it.combinations(exptGrp, 2)
    nPairs = 0
    valid_pairs = []
    # A valid experiment pair is from the same mouse and either the same
    # context or the same day of exposure
    for pair in pairs:
        if (pair[0].parent == pair[1].parent) \
                and (pair[0].sameContext(pair[1]) or
                     (exposure[pair[0]] == exposure[pair[1]])):
            valid_pairs.append(pair)
            nPairs += 1

    nFigs = int(np.ceil(nPairs / float(nCols)))

    figs = []
    axs = []
    for f in range(nFigs):
        fig, ax = plt.subplots(2, nCols, figsize=(15, 8), squeeze=False)

        ax_pairs = [(ax[0][x], ax[1][x]) for x in range(nCols)]
        axs.extend(ax_pairs)
        figs.append(fig)

    n_extras = (nFigs * nCols) - nPairs
    if n_extras > 0:
        for a in axs[-n_extras:]:
            a[0].set_visible(False)
            a[1].set_visible(False)

    for pair, ax in it.izip(valid_pairs, axs):
        grp = lab.ExperimentGroup(pair)
        label1 = 'Day {}, Ctx {}'.format(
            exposure[grp[0]] + 1, grp[0].get('environment'))
        label2 = 'Day {}, Ctx {}'.format(
            exposure[grp[1]] + 1, grp[1].get('environment'))
        ap.activityComparisonPlot(
            grp, method=method, ax=ax[0], mask1=None, mask2=None, label1=label1,
            label2=label2, roiNamesToLabel=None, normalize=False,
            rasterized=rasterized, dF='from_file')

        grp2 = lab.ExperimentGroup(pair[::-1])
        ap.activityComparisonPlot(
            grp2, method=method, ax=ax[1], mask1=None, mask2=None, label1=label2,
            label2=label1, roiNamesToLabel=None, normalize=False,
            rasterized=rasterized, dF='from_file')

    return figs


def salience_responses_figures(
        exptGrp, stimuli, pre_time=None, post_time=None, channel='Ch2',
        label=None, roi_filter=None, exclude_running=False, rasterized=False):
    """Plot each ROI's response to each stim in stimuli"""

    if exclude_running:
        stimuli = [stim for stim in stimuli if 'running' not in stim]

    # Stims labeled 'off' just flip the tail of the responsive distribution
    # but are actually the same PSTH as the 'on' version
    # No need to plot both
    stimuli = [stim for stim in stimuli if 'off' not in stim]

    if not len(stimuli):
        warn("No stimuli to analyze, aborting.")
        return []

    cmap = matplotlib.cm.get_cmap(name='Spectral')
    color_cycle = [cmap(i) for i in np.linspace(0, 0.9, len(stimuli))]

    psths = []
    for stim in stimuli:
        psth, rois, x_ranges = ia.PSTH(
            exptGrp, stimulus=stim, channel=channel, label=label, roi_filter=roi_filter,
            pre_time=pre_time, post_time=post_time,
            exclude='running' if exclude_running else None)
        psths.append(psth)
    figs, axs, axs_to_label = plotting.layout_subplots(
        n_plots=len(psths[0]) + 1, rows=3, cols=4, polar=False,
        sharex=False, figsize=(15, 8), rasterized=rasterized)

    for fig in figs:
        fig.suptitle('Salience Responses: {}'.format(
            'running excluded' if exclude_running else 'running included'))

    for psth, color, stim in it.izip(psths, color_cycle, stimuli):
        for ax, roi_psth, roi, x_range in it.izip(axs, psth, rois, x_ranges):
            ax.plot(x_range, roi_psth, color=color)
            ax.set_title(roi[0].get('mouseID') + ', ' + roi[1] + ', ' + roi[2])
            ax.axvline(0, linestyle='dashed', color='k')
            ax.set_xlim(x_range[0], x_range[-1])
            ylims = np.round(ax.get_ylim(), 2)
            if ylims[1] != 0:
                ax.set_yticks([0, ylims[1]])
            elif ylims[0] != 0:
                ax.set_yticks([ylims[0], 0])
            else:
                ax.set_yticks([0])
            if ax not in axs_to_label:
                ax.tick_params(labelbottom=False)
        # Last axis will just be for labels
        axs[-1].plot([0, 1],
                     [-color_cycle.index(color), -color_cycle.index(color)],
                     color=color, label=stim)
    axs[-1].set_xlim(0, 1)
    axs[-1].set_ylim(-len(stimuli), 1)
    axs[-1].tick_params(labelbottom=False, labelleft=False, bottom=False,
                        left=False, top=False, right=False)
    axs[-1].legend()

    for ax in axs_to_label:
        ax.set_ylabel(r'Average $\Delta$F/F')
        ax.set_xlabel('Time (s)')

    return figs


def salience_expt_summary_figure(
        expt, stimuli, method='responsiveness', pre_time=None, post_time=None,
        channel='Ch2', label=None, roi_filter=None, exclude_running=False,
        rasterized=False, n_processes=1):
    """Summary of salience responses.
    Includes trialAverageHeatmap, psth of responsive ROIs and image overlay of
    responsive ROIs.

    """

    fig, axs = plt.subplots(3, len(stimuli), figsize=(15, 8), squeeze=False,
                            subplot_kw={'rasterized': rasterized})

    fig.suptitle('Salience Experiment Summary: {}'.format(
        'running excluded' if exclude_running else 'running included'))

    frame_period = expt.frame_period()
    pre_frames = None if pre_time is None else int(pre_time / frame_period)
    post_frames = None if post_time is None else int(post_time / frame_period)

    for stim_idx, stim in enumerate(stimuli):
        expt.trialAverageHeatmap(
            stimulus=stim, ax=axs[0, stim_idx], sort=False, smoothing=None,
            window_length=5, channel=channel, label=label,
            roi_filter=roi_filter, exclude_running=exclude_running)
        axs[0, stim_idx].set_title(stim)

        responsive_filter = af.identify_stim_responsive_cells(
            expt, stimulus=stim, method=method, pre_frames=pre_frames,
            post_frames=post_frames, data=None, ax=axs[1, stim_idx],
            conf_level=95, sig_tail='upper', transients_conf_level=95,
            plot_mean=True, exclude='running' if exclude_running else None,
            channel=channel, label=label, roi_filter=roi_filter,
            n_bootstraps=10000, save_to_expt=True, n_processes=n_processes)

        rois = expt.roiVertices(
            channel=channel, label=label, roi_filter=responsive_filter)

        plotting.roiDataImageOverlay(
            ax=axs[2, stim_idx],
            background=expt.returnFinalPrototype(channel=channel),
            rois=rois, values=None, vmin=0, vmax=.8)
    return fig


def salience_exptGrp_summary_figure(
        exptGrp, stimuli, method='responsiveness', pre_time=None,
        post_time=None, channel='Ch2', label=None, roi_filter=None,
        exclude_running=False, rasterized=False, save_data=False,
        n_processes=1, n_bootstraps=10000):

    STIMS_PER_FIG = 6

    data_to_save = {}

    if exclude_running:
        stimuli = [stim for stim in stimuli if 'running' not in stim]

    if not len(stimuli):
        warn("No stimuli to analyze, aborting.")
        return []

    n_figs = int(np.ceil(len(stimuli) / float(STIMS_PER_FIG)))

    figs, psth_axs, response_axs, fraction_axs, first_col_axs = \
        [], [], [], [], []
    for n in range(n_figs):
        fig, axs = plt.subplots(
            3, STIMS_PER_FIG, figsize=(15, 8), squeeze=False,
            subplot_kw={'rasterized': rasterized})
        fig.suptitle('Responsive ROIs summary: {}'.format(
            'running excluded' if exclude_running else 'running included'))
        figs.append(fig)
        psth_axs.append(axs[0, :])
        response_axs.append(axs[1, :])
        fraction_axs.append(axs[2, :])
        first_col_axs.append(axs[:, 0])

    psth_axs = np.hstack(psth_axs)
    response_axs = np.hstack(response_axs)
    fraction_axs = np.hstack(fraction_axs)
    first_col_axs = np.hstack(first_col_axs)

    min_psth_y_lim = np.inf
    max_psth_y_lim = -np.inf
    responsive_cells = {}
    for ax, stimulus in it.izip(psth_axs, stimuli):
        responsive_cells[stimulus] = ia.identify_stim_responsive_cells(
            exptGrp, stimulus=stimulus, method=method, ax=ax, pre_time=pre_time,
            post_time=post_time, data=None, conf_level=95, sig_tail='upper',
            plot_mean=True, exclude='running' if exclude_running else None,
            channel=channel, label=label, roi_filter=roi_filter,
            n_bootstraps=n_bootstraps, save_to_expt=True,
            n_processes=n_processes)
        ax.set_title(stimulus)
        min_psth_y_lim = np.amin([min_psth_y_lim, ax.get_ylim()[0]])
        max_psth_y_lim = np.amax([max_psth_y_lim, ax.get_ylim()[1]])

    max_bar_y_lim = 0
    n_responsive_rois = {}
    data_to_save['responsive_responses'] = []
    data_to_save['non_responsive_responses'] = []
    for ax, stimulus in it.izip(response_axs, stimuli):
        responses = ia.response_magnitudes(
            exptGrp, stimulus, method=method, pre_time=pre_time, post_time=post_time,
            data=None, exclude='running' if exclude_running else None,
            channel=channel, label=label,
            roi_filter=responsive_cells[stimulus])
        data_to_save['responsive_responses'].append(
            [stimulus] + ['{:f}'.format(val) for val in responses])
        plotting.scatter_bar(
            ax, [np.abs(responses)], labels=[''], jitter_x=True)
        max_bar_y_lim = np.amax([max_bar_y_lim, ax.get_ylim()[1]])
        ax.tick_params(bottom=False, labelbottom=False)
        n_responsive_rois[stimulus] = len(responses)

        non_responses = ia.response_magnitudes(
            exptGrp, stimulus, method=method, pre_time=pre_time, post_time=post_time,
            data=None, exclude='running' if exclude_running else None,
            channel=channel, label=label,
            roi_filter=misc.invert_filter(responsive_cells[stimulus]))
        data_to_save['non_responsive_responses'].append(
            [stimulus] + ['{:f}'.format(val) for val in non_responses])

    fractions = []
    n_rois = {}
    for ax, stimulus in it.izip(fraction_axs, stimuli):
        all_psths, _, _ = ia.PSTH(
            exptGrp, stimulus=stimulus, pre_time=pre_time, post_time=post_time,
            data=None, exclude='running' if exclude_running else None,
            channel=channel, label=label, roi_filter=roi_filter)
        # Find how many of the ROIs were imaged with the current stimulus
        n_rois[stimulus] = np.sum(
            [not np.all(np.isnan(psth)) for psth in all_psths])
        # n_responsive_rois = len(responsive_psths[stimulus])
        if n_rois[stimulus] > 0:
            fractions.append(
                n_responsive_rois[stimulus] / float(n_rois[stimulus]))
            plotting.scatter_bar(
                ax, [[fractions[-1]]],
                labels=['{} / {}'.format(
                    n_responsive_rois[stimulus], n_rois[stimulus])],
                jitter_x=False)
        else:
            fractions.append(np.nan)
        ax.set_ylim(0, 1)
        ax.tick_params(bottom=False)

    for ax in set(psth_axs).difference(first_col_axs):
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelleft=False, labelbottom=False)
    for ax in psth_axs:
        ax.set_ylim(min_psth_y_lim, max_psth_y_lim)

    for ax in set(response_axs).intersection(first_col_axs):
        ax.set_ylabel('Stim response')
    for ax in set(response_axs).difference(first_col_axs):
        ax.tick_params(labelleft=False)

    for ax in response_axs:
        ax.set_ylim(0, max_bar_y_lim)

    for ax in set(fraction_axs).intersection(first_col_axs):
        ax.set_ylabel('Responsive cell fraction')
    for ax in set(fraction_axs).difference(first_col_axs):
        ax.tick_params(labelleft=False)

    if len(stimuli) % STIMS_PER_FIG:
        extra_axs = len(stimuli) - n_figs * STIMS_PER_FIG
        for ax in it.chain(psth_axs[extra_axs:], response_axs[extra_axs:],
                           fraction_axs[extra_axs:]):
            ax.set_visible(False)

    if save_data:
        # Need to update for multiple pages
        raise NotImplemented
        psths = {}
        non_responsive_psths = {}
        for stimulus in stimuli:
            # Responders
            psth, x_range = ia.PSTH(
                exptGrp, stimulus=stimulus, pre_time=pre_time, post_time=post_time,
                channel=channel, label=label,
                roi_filter=responsive_cells[stimulus], return_full='norm',
                exclude='running' if exclude_running else None)
            psth_list = [x_range]
            for roi in psth:
                psth_list.append(['{:f}'.format(val) for val in roi])
            label_strs = np.array(
                ['Time (s)'] + ['ROI ' + str(x) for x in range(psth.shape[0])])
            psths[stimulus] = np.hstack([label_strs[:, None], psth_list])
            # Non-responders
            psth, x_range = ia.PSTH(
                exptGrp, stimulus=stimulus, pre_time=pre_time, post_time=post_time,
                channel=channel, label=label,
                roi_filter=misc.invert_filter(responsive_cells[stimulus]),
                return_full='norm', exclude='running' if exclude_running else None)
            psth_list = [x_range]
            for roi in psth:
                psth_list.append(['{:f}'.format(val) for val in roi])
            label_strs = np.array(
                ['Time (s)'] + ['ROI ' + str(x) for x in range(psth.shape[0])])
            non_responsive_psths[stimulus] = np.hstack([label_strs[:, None], psth_list])
        data_to_save['psths'] = psths
        data_to_save['non_responsive_psths'] = non_responsive_psths
        data_to_save['fractions'] = [stimuli, fractions]
        data_to_save['n_responding'] = [
            stimuli, [n_responsive_rois[stim] for stim in stimuli]]
        data_to_save['n_rois'] = [stimuli, [n_rois[stim] for stim in stimuli]]

        misc.save_data(
            data_to_save, fig=fig, label='salience_summary', method=save_data)

    return figs


def salience_expt_grp_dataframe_figure(
        expt_grps, stimuli, plotby, method='responsiveness', pre_time=None,
        post_time=None, channel='Ch2', label=None, roi_filters=None,
        colors=None, exclude_running=False, rasterized=False, save_data=False,
        n_bootstraps=10000, n_processes=1):

    # data_to_save = {}
    STIMS_PER_FIG = 4

    if roi_filters is None:
        roi_filters = [None] * len(expt_grps)

    if exclude_running:
        stimuli = [stim for stim in stimuli if 'running' not in stim]

    if not len(stimuli):
        warn("No stimuli to analyze, aborting.")
        return []

    n_figs = int(np.ceil(len(stimuli) / float(STIMS_PER_FIG)))

    figs, response_axs, fraction_axs, first_col_axs = [], [], [], []
    for n in range(n_figs):
        fig, axs = plt.subplots(
            2, STIMS_PER_FIG, figsize=(15, 8), squeeze=False,
            subplot_kw={'rasterized': rasterized})
        fig.suptitle('Responsive ROIs by {}: {}'.format(
            plotby,
            'running excluded' if exclude_running else 'running included'))
        figs.append(fig)
        response_axs.append(axs[0, :])
        fraction_axs.append(axs[1, :])
        first_col_axs.append(axs[:, 0])

    response_axs = np.hstack(response_axs)
    fraction_axs = np.hstack(fraction_axs)
    first_col_axs = np.hstack(first_col_axs)

    if method == 'responsiveness':
        activity_label = 'Responsiveness (dF/F)'
    elif method == 'peak':
        activity_label = 'Peak responsiveness (dF/F)'
    else:
        raise ValueError("Unrecognized 'method' value")

    responsive_cells = {}
    responsive_dfs = {}
    for stimulus in stimuli:
        responsive_cells[stimulus] = []
        responsive_dfs[stimulus] = []
        stimulus_filters = {}
        stimulus_dfs = {}
        for expt_grp, roi_filter in it.izip(expt_grps, roi_filters):
            stimulus_filters = []
            stimulus_dfs = []
            for key, grp in expt_grp.groupby(plotby):
                stimulus_filters.append(
                    ia.identify_stim_responsive_cells(
                        grp, stimulus=stimulus, method=method, pre_time=pre_time,
                        post_time=post_time, data=None, conf_level=95,
                        sig_tail='upper',
                        exclude='running' if exclude_running else None,
                        channel=channel, label=label, roi_filter=roi_filter,
                        n_bootstraps=n_bootstraps, save_to_expt=True,
                        n_processes=n_processes))
                df = ia.response_magnitudes(
                    grp, stimulus, method=method, pre_time=pre_time,
                    post_time=post_time, data=None,
                    exclude='running' if exclude_running else None,
                    channel=channel, label=label,
                    roi_filter=stimulus_filters[-1], return_df=True)
                # Put the grouping info back in the dataframe
                # For example:
                # plotby = ['condition_day']
                # keys will be ['A_0', 'A_1', 'B_0', etc...]
                # So df['condition_day'] == 'A_0' for the first group, etc.
                for key_value, grouping in zip(key, plotby):
                    df[grouping] = key_value
                stimulus_dfs.append(df)
            responsive_dfs[stimulus].append(pd.concat(
                stimulus_dfs, ignore_index=True))
            responsive_cells[stimulus].append(misc.filter_union(
                stimulus_filters))

    #
    # Plot mean PSTH for each stim/group
    #

    pass

    #
    # Plot the mean response of responsive cells
    #
    max_response_y_lim = 0
    for ax, stimulus in it.izip(response_axs, stimuli):
        plotting.plot_dataframe(
            ax, responsive_dfs[stimulus],
            labels=[expt_grp.label() for expt_grp in expt_grps],
            activity_label=activity_label, groupby=None, plotby=plotby,
            orderby=None, plot_method='line', plot_shuffle=False,
            shuffle_plotby=False, pool_shuffle=False,
            agg_fn=np.mean, colors=colors)

        max_response_y_lim = np.amax([max_response_y_lim, ax.get_ylim()[1]])
        ax.set_title(stimulus)
        plt.setp(ax.get_xticklabels(), rotation='40',
                 horizontalalignment='right')

    #
    # Plot fraction of responsive ROIs
    #

    groupby = [['mouseID', 'uniqueLocationKey', 'roi_id'] + plotby,
               ['mouseID'] + plotby]

    activity_kwargs = [
        {'channel': channel, 'label': label, 'include_roi_filter': inc_filter}
        for inc_filter in roi_filters]
    for ax, stimulus in it.izip(fraction_axs, stimuli):
        plot_metric(
            ax, expt_grps, eg.filtered_rois, 'line',
            roi_filters=responsive_cells[stimulus], groupby=groupby,
            plotby=plotby, orderby=None, plot_shuffle=False,
            shuffle_plotby=False, pool_shuffle=False, plot_abs=False,
            activity_kwargs=activity_kwargs,
            activity_label='Fraction responding', label_every_n=1,
            rotate_labels=True, colors=colors)

        # ax.set_ylim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    for ax in set(response_axs).difference(first_col_axs):
        ax.set_ylabel('')

    for ax in response_axs:
        ax.set_ylim(0, max_response_y_lim)
        ax.set_xlabel('')

    for ax in set(fraction_axs).difference(first_col_axs):
        ax.set_ylabel('')
        ax.set_title('')

    if len(stimuli) % STIMS_PER_FIG:
        extra_axs = len(stimuli) - n_figs * STIMS_PER_FIG
        for ax in it.chain(response_axs[extra_axs:], fraction_axs[extra_axs:]):
            ax.set_visible(False)

    return figs


def compare_psth_summary_figure(
        expt_grps, stimuli, pre_time=None, post_time=None, channel='Ch2',
        label=None, roi_filters=None, colors=None, exclude_running=False,
        rasterized=False):

    STIMS_PER_FIG = 6

    if colors is None:
        colors = sns.color_palette()

    data_to_save = {}

    if exclude_running:
        stimuli = [stim for stim in stimuli if 'running' not in stim]

    if not len(stimuli):
        warn("No stimuli to analyze, aborting.")
        return []

    n_figs = int(np.ceil(len(stimuli) / float(STIMS_PER_FIG)))

    figs, response_axs, first_col_axs = [], [], []
    psth_axs = defaultdict(list)
    for n in range(n_figs):
        fig, axs = plt.subplots(
            len(expt_grps) + 1, STIMS_PER_FIG, figsize=(15, 8), squeeze=False,
            subplot_kw={'rasterized': rasterized})
        fig.suptitle('All ROIs summary: {}'.format(
            'running excluded' if exclude_running else 'running included'))
        figs.append(fig)
        for expt_grp, grp_axs in zip(expt_grps, axs):
            psth_axs[expt_grp].append(grp_axs)
            plotting.right_label(grp_axs[-1], expt_grp.label())
        response_axs.append(axs[-1, :])
        first_col_axs.append(axs[:, 0])

    for expt_grp in expt_grps:
        psth_axs[expt_grp] = np.hstack(psth_axs[expt_grp])
    response_axs = np.hstack(response_axs)
    first_col_axs = np.hstack(first_col_axs)

    min_psth_y_lim = np.inf
    max_psth_y_lim = -np.inf
    for expt_grp, roi_filter, color in zip(expt_grps, roi_filters, colors):
        for ax, stimulus in it.izip(psth_axs[expt_grp], stimuli):
            ia.PSTH(
                expt_grp, stimulus, ax=ax, pre_time=pre_time, post_time=post_time,
                exclude='running' if exclude_running else None, data=None,
                shade_ste=False, plot_mean=True, channel=channel, label=label,
                roi_filter=roi_filter, color=color)
            ax.set_title(stimulus)
            min_psth_y_lim = np.amin([min_psth_y_lim, ax.get_ylim()[0]])
            max_psth_y_lim = np.amax([max_psth_y_lim, ax.get_ylim()[1]])

    max_bar_y_lim = 0
    data_to_save['responses'] = {expt_grp.label(): [] for expt_grp in expt_grps}
    for ax, stimulus in it.izip(response_axs, stimuli):
        responses = []
        for expt_grp, roi_filter, color in zip(expt_grps, roi_filters, colors):
            responses.append(ia.response_magnitudes(
                expt_grp, stimulus, method='responsiveness', pre_time=pre_time,
                post_time=post_time, data=None,
                exclude='running' if exclude_running else None,
                channel=channel, label=label, roi_filter=roi_filter))
            data_to_save['responses'][expt_grp.label()].append(
                [stimulus] + ['{:f}'.format(val) for val in responses[-1]])
        plotting.grouped_bar(
            ax, values=[[np.abs(r)] for r in responses], cluster_labels=[''],
            condition_labels=[expt_grp.label() for expt_grp in expt_grps],
            bar_colors=colors, scatter_points=True, jitter_x=True, s=20)
        max_bar_y_lim = np.amax([max_bar_y_lim, ax.get_ylim()[1]])
        ax.tick_params(bottom=False, labelbottom=False)

    for ax in set(it.chain(*psth_axs.itervalues())).difference(first_col_axs):
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(labelleft=False, labelbottom=False)
    for ax in it.chain(*psth_axs.itervalues()):
        ax.set_ylim(min_psth_y_lim, max_psth_y_lim)

    for ax in set(response_axs).intersection(first_col_axs):
        ax.set_ylabel('Stim response')
    for ax in set(response_axs).difference(first_col_axs):
        ax.tick_params(labelleft=False)
        legend = ax.get_legend()
        if legend is not None:
            legend.set_visible(False)

    for ax in response_axs:
        ax.set_ylim(0, max_bar_y_lim)

    # for ax in set(fraction_axs).intersection(first_col_axs):
    #     ax.set_ylabel('Responsive cell fraction')
    # for ax in set(fraction_axs).difference(first_col_axs):
    #     ax.tick_params(labelleft=False)

    if len(stimuli) % STIMS_PER_FIG:
        extra_axs = len(stimuli) - n_figs * STIMS_PER_FIG
        for ax in it.chain(response_axs[extra_axs:], *[
                grp_axs[extra_axs:] for grp_axs in psth_axs.itervalues()]):
            ax.set_visible(False)

    return figs


def plotRoisOverlay(expt, channel='Ch2', label=None, roi_filter=None,
                    rasterized=False):
    """Generate a figure of the imaging location with all ROIs overlaid"""

    figs = []
    background_image = expt.returnFinalPrototype(channel=channel)
    roiVerts = expt.roiVertices(
        channel=channel, label=label, roi_filter=roi_filter)
    labels = expt.roi_ids(channel=channel, label=label, roi_filter=roi_filter)
    imaging_parameters = expt.imagingParameters()
    aspect_ratio = imaging_parameters['pixelsPerLine'] \
        / imaging_parameters['linesPerFrame']
    for plane in xrange(background_image.shape[0]):
        fig = plt.figure()
        ax = fig.add_subplot(111, rasterized=rasterized)
        roi_inds = [i for i, v in enumerate(roiVerts) if v[0][0][2] == plane]

        # plane_verts = np.array(roiVerts)[roi_inds].tolist()
        plane_verts = [roiVerts[x] for x in roi_inds]
        twoD_verts = []
        for roi in plane_verts:
            roi_polys = []
            for poly in roi:
                roi_polys.append(poly[:, :2])
            twoD_verts.append(roi_polys)

        plotting.roiDataImageOverlay(
            ax, background_image[plane, :, :], twoD_verts, values=None,
            vmin=0, vmax=1, labels=np.array(labels)[roi_inds].tolist(),
            cax=None, alpha=0.2, aspect=aspect_ratio)
        ax.set_title('{}_{}: plane {}'.format(
            expt.parent.get('mouseID'), expt.get('startTime'), plane))
        figs.append(fig)
    return figs


def trial_responses(
        exptGrp, stimuli, channel='Ch2', label=None, roi_filter=None,
        exclude_running=False, rasterized=False, plot_mean=False,
        gray_traces=False, **psth_kwargs):
    """Plots the response to each stim in 'stimuli' for all rois and trials in
    'exptGrp'

    """

    if exclude_running:
        stimuli = [stim for stim in stimuli if 'running' not in stim]

    if not len(stimuli):
        warn("No stimuli to analyze, aborting.")
        return

    # Stims labeled 'off' just flip the tail of the responsive distribution
    # but are actually the same PSTH as the 'on' version
    # No need to plot both
    stimuli = [stim for stim in stimuli if 'off' not in stim]

    psths = {}
    for stimulus in stimuli:
        psths[stimulus], rois, x_range = ia.PSTH(
            exptGrp, stimulus, channel=channel, label=label, roi_filter=roi_filter,
            return_full=True, exclude='running' if exclude_running else None,
            **psth_kwargs)

    figs, axs, axs_to_label = plotting.layout_subplots(
        n_plots=len(rois) * len(stimuli), rows=4, cols=len(stimuli),
        polar=False, sharex=False, figsize=(15, 8), rasterized=rasterized)

    for fig in figs:
        fig.suptitle('Trial Responses: {}'.format(
            'running excluded' if exclude_running else 'running included'))

    for ax in axs_to_label:
        ax.set_ylabel(r'Average $\Delta$F/F')
        ax.set_xlabel('Time (s)')

    ax_idx = 0
    for roi_idx in xrange(len(rois)):
        for stimulus in stimuli:
            ax = axs[ax_idx]
            # If there are no trial psths for this roi, just move along
            if psths[stimulus][roi_idx].shape[1] > 0:
                if gray_traces:
                    ax.plot(x_range[roi_idx], psths[stimulus][roi_idx],
                            color='0.8')
                else:
                    ax.plot(x_range[roi_idx], psths[stimulus][roi_idx])
                if plot_mean:
                    ax.plot(
                        x_range[roi_idx],
                        np.nanmean(psths[stimulus][roi_idx], axis=1),
                        lw=2, color='k')
            ax.axvline(0, linestyle='dashed', color='k')
            ax.set_xlim(x_range[roi_idx][0], x_range[roi_idx][-1])
            ylims = np.round(ax.get_ylim(), 2)
            if ylims[1] != 0:
                ax.set_yticks([0, ylims[1]])
            elif ylims[0] != 0:
                ax.set_yticks([ylims[0], 0])
            else:
                ax.set_yticks([0])
            ax_geometry = ax.get_geometry()
            # If ax is in top row add a stim title
            if ax_geometry[2] <= ax_geometry[1]:
                ax.set_title(stimulus)
            # If ax is in last column add an roi label
            if ax_geometry[2] % ax_geometry[1] == 0:
                roi_label = rois[roi_idx][0].get('mouseID') + '\n' + \
                    rois[roi_idx][1] + '\n' + rois[roi_idx][2]
                # Bbox = ax.figbox
                # ax.figure.text(Bbox.p1[0] + 0.02,
                #                (Bbox.p1[1] + Bbox.p0[1]) / 2,
                #                roi_label, rotation='vertical',
                #                verticalalignment='center')
                plotting.right_label(
                    ax, roi_label, rotation='vertical',
                    verticalalignment='center', horizontalalignment='center')
            # Remove extra labels
            if ax not in axs_to_label:
                ax.tick_params(labelbottom=False)
            ax_idx += 1

        if np.mod(roi_idx, 4) == 3:
            yield figs[roi_idx / 4]


def compare_stim_responses(
        exptGrp, stimuli, channel='Ch2', label=None, roi_filter=None,
        exclude_running=False, rasterized=False, plot_method='scatter',
        z_score=True, **kwargs):
    """Plot of each pair of stims in stimuli against each other."""

    if exclude_running:
        stimuli = [stim for stim in stimuli if 'running' not in stim]

    if not len(stimuli):
        warn("No stimuli to analyze, aborting.")
        return []

    figs, axs, _ = plotting.layout_subplots(
        comb(len(stimuli), 2), rows=2, cols=4, figsize=(15, 8),
        sharex=False, rasterized=rasterized)

    rois = {}
    means = {}
    stds = {}
    for stimulus in stimuli:
        means[stimulus], stds[stimulus], _, rois[stimulus], _ = \
            ia.response_magnitudes(
                exptGrp, stimulus, channel=channel, label=label, roi_filter=roi_filter,
                return_full=True, z_score=z_score,
                exclude='running' if exclude_running else None, **kwargs)

    for ax, (stim1, stim2) in zip(axs, it.combinations(stimuli, 2)):
        if plot_method == 'ellipse':
            raise NotImplemented
            means_1 = []
            means_2 = []
            stds_1 = []
            stds_2 = []
            all_rois = rois[stim1] + rois[stim2]
            for roi in set(all_rois):
                if roi in rois[stim1] and roi in rois[stim2]:
                    idx_1 = rois[stim1].index(roi)
                    idx_2 = rois[stim2].index(roi)
                    means_1.append(means[stim1][idx_1])
                    means_2.append(means[stim2][idx_2])
                    stds_1.append(stds[stim1][idx_1])
                    stds_2.append(stds[stim2][idx_2])

            max_x = np.nanmax(np.array(means_1) + np.array(stds_1))
            x_std = 4 * nanstd(means_1)
            max_x = min([max_x, nanmean(means_1) + x_std])
            max_y = np.nanmax(np.array(means_2) + np.array(stds_2))
            y_std = 4 * nanstd(means_2)
            max_y = min([max_y, nanmean(means_2) + y_std])
            min_x = np.nanmin(np.array(means_1) - np.array(stds_1))
            min_x = max([min_x, nanmean(means_1) - x_std])
            min_y = np.nanmin(np.array(means_2) - np.array(stds_2))
            min_y = max([min_y, nanmean(means_2) - y_std])

            finite_means = np.isfinite(means_1) & np.isfinite(means_2)

            if not np.any(finite_means):
                continue

            plotting.ellipsePlot(ax, means_1, means_2, stds_1, stds_2,
                                 axesCenter=False, print_stats=True)

            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.axvline(0, linestyle=':', color='k')
            ax.axhline(0, linestyle=':', color='k')
            ax.set_xlabel(stim1)
            ax.set_ylabel(stim2)

        elif plot_method == 'scatter':
            means_1 = []
            means_2 = []
            all_rois = rois[stim1] + rois[stim2]
            for roi in set(all_rois):
                if roi in rois[stim1] and roi in rois[stim2]:
                    idx_1 = rois[stim1].index(roi)
                    idx_2 = rois[stim2].index(roi)
                    means_1.append(means[stim1][idx_1])
                    means_2.append(means[stim2][idx_2])

            finite_means = np.isfinite(means_1) & np.isfinite(means_2)

            if not np.any(finite_means):
                continue

            plotting.scatterPlot(
                ax, [means_1, means_2], [stim1, stim2], s=1.5,
                print_stats=True)

            ax.axvline(0, linestyle=':', color='k')
            ax.axhline(0, linestyle=':', color='k')

        else:
            raise ValueError

    for fig in figs:
        fig.suptitle('Stim response {}comparison: {}'.format(
            'z-score ' if z_score else '',
            'running excluded' if exclude_running else 'running included'))

    return figs


def quantify_multi_responses(
        exptGrp, stimuli, method='responsiveness', channel='Ch2', label=None,
        roi_filter=None, pre_time=None, post_time=None, rasterized=False,
        n_processes=1, n_bootstraps=10000):
    """Quantifies the number of stimuli that each ROI responds to,
    plots as a histogram"""

    fig, axs = plt.subplots(1, 2, subplot_kw={'rasterized': rasterized},
                            figsize=(15, 8))

    ia.plot_number_of_stims_responsive(
        exptGrp, axs[0], stimuli, method=method, pre_time=pre_time,
        post_time=post_time, exclude=None, channel=channel, label=label,
        roi_filter=roi_filter, n_processes=n_processes,
        n_bootstraps=n_bootstraps)

    ia.plot_number_of_stims_responsive(
        exptGrp, axs[1], stimuli, method=method, pre_time=pre_time,
        post_time=post_time, exclude='running', channel=channel, label=label,
        roi_filter=roi_filter, n_processes=n_processes,
        n_bootstraps=n_bootstraps)

    axs[0].set_title('Running included')
    axs[1].set_title('Running excluded')

    return fig


def response_linearity(
        exptGrp, paired_stimuli, channel='Ch2', label=None, roi_filter=None,
        exclude_running=False, responsive_method=None, rasterized=False,
        plot_method='ellipse', **kwargs):
    """Histogram of response linearities

    Calculated as combined_response / (single_response_1 + single_response_2)

    Parameters
    ----------
    paired_stimuli : list of paired stimuli to analyze
    responsive_method : None, to include all rois, or a method for identifying
        stim responsive rois

    """

    paired_stimuli = [stim for stim in paired_stimuli if 'Paired' in stim]

    if not paired_stimuli:
        return []

    figs, axs, _ = plotting.layout_subplots(
        len(paired_stimuli), rows=2, cols=4, figsize=(15, 8),
        sharex=False, rasterized=rasterized)

    for stimulus, ax in zip(paired_stimuli, axs):
        stims = stimulus.split()[1:]
        if responsive_method:
            stimulus_filter = ia.identify_stim_responsive_cells(
                exptGrp, stimulus=stimulus, method=responsive_method,
                channel=channel, label=label, roi_filter=roi_filter,
                exclude='running' if exclude_running else None,
                **kwargs)
        else:
            stimulus_filter = roi_filter

        psth1, rois_1, x_ranges1 = ia.PSTH(
            exptGrp, stims[0], channel=channel, label=label, roi_filter=stimulus_filter,
            return_full=True, exclude='running' if exclude_running else None,
            **kwargs)
        responses_1 = []
        for roi_psth, roi_x_range in zip(psth1, x_ranges1):
            responses_1.append(nanmean(roi_psth[roi_x_range > 0], axis=0) -
                               nanmean(roi_psth[roi_x_range < 0], axis=0))

        psth2, rois_2, x_ranges2 = ia.PSTH(
            exptGrp, stims[1], channel=channel, label=label, roi_filter=stimulus_filter,
            return_full=True, exclude='running' if exclude_running else None,
            **kwargs)
        responses_2 = []
        for roi_psth, roi_x_range in zip(psth2, x_ranges2):
            responses_2.append(nanmean(roi_psth[roi_x_range > 0], axis=0) -
                               nanmean(roi_psth[roi_x_range < 0], axis=0))

        psth_combo, rois_combo, x_ranges_combo = ia.PSTH(
            exptGrp, stimulus, channel=channel, label=label, roi_filter=stimulus_filter,
            return_full=True, exclude='running' if exclude_running else None,
            **kwargs)
        responses_combo = []
        for roi_psth, roi_x_range in zip(psth_combo, x_ranges_combo):
            responses_combo.append(
                nanmean(roi_psth[roi_x_range > 0], axis=0) -
                nanmean(roi_psth[roi_x_range < 0], axis=0))

        shared_rois = set(rois_1).intersection(rois_2).intersection(rois_combo)

        combined_mean = []
        combined_std = []
        summed_mean = []
        summed_std = []
        linearity_ratios = []
        for roi in shared_rois:
            combo = responses_combo[rois_combo.index(roi)]
            stim1 = responses_1[rois_1.index(roi)]
            stim2 = responses_2[rois_2.index(roi)]
            combined_mean.append(nanmean(combo))
            combined_std.append(nanstd(combo))
            summed_mean.append(nanmean(stim1) + nanmean(stim2))
            # Propagate summed std
            summed_std.append(
                np.sqrt(nanstd(stim1) ** 2 + nanstd(stim2) ** 2))
            linearity_ratios.append(combined_mean[-1] / summed_mean[-1])

        if np.all(np.isnan(linearity_ratios)):
            ax.set_visible(False)
            continue

        if plot_method == 'hist':
            linearity_ratios = [ratio for ratio in linearity_ratios
                                if not np.isnan(ratio)]

            if len(linearity_ratios) == 0:
                return []
            plotting.histogram(ax, linearity_ratios, bins=10, plot_mean=True)

            ax.set_title(stimulus)
            ax.set_xlabel('combined / (stim1 + stim2)')
            ax.set_ylabel('Number')
        elif plot_method == 'ellipse':
            plotting.ellipsePlot(
                ax, summed_mean, combined_mean, summed_std, combined_std,
                axesCenter=False, print_stats=True)
            ax.set_title(stimulus)
            ax.set_xlabel('stim1 + stim2')
            ax.set_ylabel('combined')
            combined_mean = np.array(combined_mean)
            combined_std = np.array(combined_std)
            summed_mean = np.array(summed_mean)
            summed_std = np.array(summed_std)
            max_x = np.nanmax(summed_mean + summed_std)
            x_std = 4 * nanstd(summed_mean)
            max_x = min([max_x, nanmean(summed_mean) + x_std])
            max_y = np.nanmax(combined_mean + combined_std)
            y_std = 4 * nanstd(combined_mean)
            max_y = min([max_y, nanmean(combined_mean) + y_std])
            min_x = np.nanmin(summed_mean - summed_std)
            min_x = max([min_x, nanmean(summed_mean) - x_std])
            min_y = np.nanmin(combined_mean - combined_std)
            min_y = max([min_y, nanmean(combined_mean) - y_std])
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.axvline(0, linestyle=':', color='k')
            ax.axhline(0, linestyle=':', color='k')
        elif plot_method == 'scatter':
            plotting.scatterPlot(
                ax, [summed_mean, combined_mean],
                ['stim1 + stim2', 'combined'], s=1, print_stats=True)
            ax.axvline(0, linestyle=':', color='k')
            ax.axhline(0, linestyle=':', color='k')
            ax.set_title(stimulus)
        else:
            raise ValueError(
                'Unrecognized plot method: {}'.format(plot_method))

    for fig in figs:
        fig.suptitle('Stim response linearity, {}: {}'.format(
            'all ROIs' if responsive_method is None else 'responsive ROIs only',
            'running excluded' if exclude_running else 'running included'))

    return figs


def run_duration_responsiveness(
        exptGrp, channel='Ch2', label=None, roi_filter=None, rasterized=False,
        method='responsiveness', **psth_kwargs):
    """Create figure comparing the magnitude of running responses versus
    duration of running bout.

    """

    figs = []

    fig, axs = plt.subplots(2, 2, subplot_kw={'rasterized': rasterized},
                            figsize=(15, 8))

    ia.compare_run_response_by_running_duration(
        exptGrp, axs[0, 0], run_intervals='running_start',
        response_method='responsiveness', plot_method='scatter',
        channel=channel, label=label, roi_filter=roi_filter,
        responsive_method=method, **psth_kwargs)

    ia.compare_run_response_by_running_duration(
        exptGrp, axs[0, 1], run_intervals='running_stop',
        response_method='responsiveness', plot_method='scatter',
        channel=channel, label=label, roi_filter=roi_filter,
        responsive_method=method, **psth_kwargs)

    # ia.compare_run_response_by_running_duration(
    #     exptGrp, axs[1, 0], run_intervals='running_stim',
    #     response_method='responsiveness', plot_method='scatter',
    #     channel=channel, label=label, roi_filter=roi_filter,
    #     responsive_method=method, **psth_kwargs)

    # ia.compare_run_response_by_running_duration(
    #     exptGrp, axs[1, 1], run_intervals='running_no_stim',
    #     response_method='responsiveness', plot_method='scatter',
    #     channel=channel, label=label, roi_filter=roi_filter,
    #     responsive_method=method, **psth_kwargs)

    # figs.append(fig)

    # fig, axs = plt.subplots(2, 2, subplot_kw={'rasterized': rasterized},
    #                         figsize=(15, 8))

    ia.compare_run_response_by_running_duration(
        exptGrp, axs[1, 0], run_intervals='running_start',
        response_method='mean', plot_method='scatter',
        channel=channel, label=label, roi_filter=roi_filter,
        responsive_method=method, **psth_kwargs)

    ia.compare_run_response_by_running_duration(
        exptGrp, axs[1, 1], run_intervals='running_stop',
        response_method='mean', plot_method='scatter',
        channel=channel, label=label, roi_filter=roi_filter,
        responsive_method=method, **psth_kwargs)

    # ia.compare_run_response_by_running_duration(
    #     exptGrp, axs[1, 0], run_intervals='running_stim',
    #     response_method='mean', plot_method='scatter',
    #     channel=channel, label=label, roi_filter=roi_filter,
    #     responsive_method=method, **psth_kwargs)

    # ia.compare_run_response_by_running_duration(
    #     exptGrp, axs[1, 1], run_intervals='running_no_stim',
    #     response_method='mean', plot_method='scatter',
    #     channel=channel, label=label, roi_filter=roi_filter,
    #     responsive_method=method, **psth_kwargs)

    figs.append(fig)

    return figs


def imaging_and_behavior_summary(
        exptGrp, channel='Ch2', label=None, roi_filter=None):
    """Creates a summary figure of imaging data and behavior data"""

    nTrials = sum([len(expt.findall('trial')) for expt in exptGrp])

    figs, axs, _ = plotting.layout_subplots(
        nTrials, rows=1, cols=2, figsize=(15, 8), sharex=False)

    for ax, trial in it.izip(
            axs, it.chain(*[expt.findall('trial') for expt in exptGrp])):
        if isinstance(trial.parent, lab.classes.SalienceExperiment):
            stim = trial.get('stimulus')
            if stim == 'air':
                stim = 'airpuff'
            stim_time = trial.parent.stimulusTime()
            if 'Paired' in stim:
                keys = stim.split(' ')[1:] + ['running', 'licking']
            else:
                keys = [stim, 'running', 'licking']
            ap.plot_imaging_and_behavior(
                trial, ax, keys=keys, channel=channel, label=label,
                roi_filter=roi_filter, include_empty=True)

            ax.axvline(stim_time, linestyle='dashed', color='k')
            ax.set_xticklabels(ax.get_xticks() - stim_time)
            ax.set_title('{}_{}: {}'.format(
                trial.parent.parent.get('mouseID'),
                trial.parent.get('uniqueLocationKey'), trial.get('time')))
        else:
            ap.plot_imaging_and_behavior(
                trial, ax, channel=channel, label=label, roi_filter=roi_filter,
                include_empty=False)

    return figs


def response_cdfs(
        exptGrp, stimuli, method='responsiveness', pre_time=None,
        post_time=None, channel='Ch2', label=None, roi_filter=None,
        rasterized=False):
    """Plot cdfs across all rois for each stim in stimuli.
    Plots all stims except running/licking, all running/licking stims, and
    all stims with running excluded"""

    fig, axs = plt.subplots(
        1, 3, figsize=(15, 8), subplot_kw={'rasterized': rasterized})

    cmap = matplotlib.cm.get_cmap(name='Spectral')

    #
    # Plot all stims except running/licking
    #
    axs[0].set_title('All stims (except running/licking)')
    stims = [stim for stim in stimuli
             if 'running' not in stim and 'licking' not in stim]

    colors = [cmap(i) for i in np.linspace(0, 0.9, len(stims))]

    for stim, color in zip(stims, colors):
        responses = ia.response_magnitudes(
            exptGrp, stim, method=method, pre_time=pre_time, post_time=post_time,
            channel=channel, label=label, roi_filter=roi_filter,
            return_full=False, exclude=None)
        non_nan_responses = responses[np.isfinite(responses)]
        if len(non_nan_responses):
            plotting.cdf(axs[0], non_nan_responses, bins='exact', color=color)

    axs[0].legend(stims, loc='lower right')

    #
    # Plot running/licking stims
    #
    axs[1].set_title('Running/licking responses')
    stims = [stim for stim in stimuli
             if 'running' in stim or 'licking' in stim]

    colors = [cmap(i) for i in np.linspace(0, 0.9, len(stims))]

    for stim, color in zip(stims, colors):
        responses = ia.response_magnitudes(
            exptGrp, stim, method=method, pre_time=pre_time, post_time=post_time,
            channel=channel, label=label, roi_filter=roi_filter,
            return_full=False, exclude=None)
        non_nan_responses = responses[np.isfinite(responses)]
        if len(non_nan_responses):
            plotting.cdf(axs[1], non_nan_responses, bins='exact', color=color)

    axs[1].legend(stims, loc='lower right')

    #
    # Plot all stims with running excluded
    #
    axs[2].set_title('All stims, running excluded')
    stims = [stim for stim in stimuli if 'running' not in stim]

    colors = [cmap(i) for i in np.linspace(0, 0.9, len(stims))]

    for stim, color in zip(stims, colors):
        responses = ia.response_magnitudes(
            exptGrp, stim, method=method, pre_time=pre_time, post_time=post_time,
            channel=channel, label=label, roi_filter=roi_filter,
            return_full=False, exclude='running')
        non_nan_responses = responses[np.isfinite(responses)]
        if len(non_nan_responses):
            plotting.cdf(axs[2], non_nan_responses, bins='exact', color=color)

    axs[2].legend(stims, loc='lower right')

    for ax in axs:
        ax.set_xlabel('Responsiveness')

    return fig


def paired_stims_response_heatmaps(
        exptGrp, stimuli, exclude_running=False, rasterized=False,
        **response_kwargs):
    """Plot heatmaps of response magnitude of paired stims versus
    single stims

    """

    paired_stims = [stim for stim in stimuli if 'Paired' in stim]

    fig, axs = plt.subplots(
        1, len(paired_stims), subplot_kw={'rasterized': rasterized})
    fig.subplots_adjust(wspace=0.5)

    for ax, paired_stim in it.izip(axs, paired_stims):
        stims_in_pair = paired_stim.split()[1:]

        stims_to_plot = [paired_stim] + stims_in_pair + \
            [stim for stim in exptGrp.stimuli()
             if 'Paired' not in stim and stim not in stims_in_pair]

        ap.stim_response_heatmap(
            exptGrp, ax, stims_to_plot, sort_by=paired_stim,
            exclude='running' if exclude_running else None,
            aspect_ratio=0.2, **response_kwargs)

        ax.axvline(0.5, linewidth=3, color='k')
        ax.axvline(2.5, linewidth=3, color='k')
        for label in ax.get_yticklabels():
            label.set_fontsize(7)
        x_labels = []
        for label in ax.get_xticklabels():
            label.set_fontsize(5)
            x_labels.append(''.join([s[0] for s in label.get_text().split()]))
        ax.set_xticklabels(x_labels)

    title = fig.suptitle(
        'Paired stim heatmap, sort by paired stim, running {}'.format(
            'excluded' if exclude_running else 'included'))
    title.set_fontsize(7)
    yield fig

    fig, axs = plt.subplots(
        1, len(paired_stims), subplot_kw={'rasterized': rasterized})
    fig.subplots_adjust(wspace=0.5)

    for ax, paired_stim in it.izip(axs, paired_stims):
        stims_in_pair = paired_stim.split()[1:]

        stims_to_plot = [paired_stim] + stims_in_pair + \
            [stim for stim in exptGrp.stimuli()
             if 'Paired' not in stim and stim not in stims_in_pair]

        ap.stim_response_heatmap(
            exptGrp, ax, stims_to_plot, sort_by=stims_in_pair,
            exclude='running' if exclude_running else None,
            aspect_ratio=0.2, **response_kwargs)

        ax.axvline(0.5, linewidth=3, color='k')
        ax.axvline(2.5, linewidth=3, color='k')
        for label in ax.get_yticklabels():
            label.set_fontsize(7)
        x_labels = []
        for label in ax.get_xticklabels():
            label.set_fontsize(5)
            x_labels.append(''.join([s[0] for s in label.get_text().split()]))
        ax.set_xticklabels(x_labels)

    title = fig.suptitle(
        'Paired stim heatmap, sort by single stims in pair, running {}'.format(
            'excluded' if exclude_running else 'included'))
    title.set_fontsize(7)
    yield fig


def compare_bouton_response_figure(
        exptGrp, stimuli, plot_method='cdf', save_data=False, rasterized=False,
        **response_kwargs):
    """Figure to compare different types of boutons"""

    fig, axs = plt.subplots(2, 3, subplot_kw={'rasterized': rasterized})

    data_to_save = {}

    data_to_save['angle'] = ap.compare_bouton_responses(
        exptGrp, axs[0, 0], stimuli, comp_method='angle', plot_method=plot_method,
        **response_kwargs)

    data_to_save['abs angle'] = ap.compare_bouton_responses(
        exptGrp, axs[1, 0], stimuli, comp_method='abs angle', plot_method=plot_method,
        **response_kwargs)

    data_to_save['corr'] = ap.compare_bouton_responses(
        exptGrp, axs[0, 1], stimuli, comp_method='corr', plot_method=plot_method,
        **response_kwargs)

    data_to_save['abs corr'] = ap.compare_bouton_responses(
        exptGrp, axs[1, 1], stimuli, comp_method='abs corr', plot_method=plot_method,
        **response_kwargs)

    data_to_save['mean diff'] = ap.compare_bouton_responses(
        exptGrp, axs[0, 2], stimuli, comp_method='mean diff', plot_method=plot_method,
        **response_kwargs)

    for line_idx, line in enumerate(axs[0, 2].lines):
        axs[1, 2].axhline(
            line_idx, color=line.get_color(), label=line.get_label())

    axs[1, 2].set_ylim(-1, len(axs[0, 2].lines))
    axs[1, 2].tick_params(labelbottom=False, labelleft=False, bottom=False,
                          left=False, top=False, right=False)
    axs[1, 2].legend()

    if save_data:
        misc.save_data(data_to_save, fig=fig, label='compare_bouton_responses',
                       method=save_data)

    return fig


def hidden_rewards_learning_summary(
        exptGrps, save_data=False, rasterized=False, groupby=None, plotby=None,
        orderby=None, colors=None, label_every_n=1):
    """Generates a summary figure of hidden reward analysis plots"""

    if groupby is None:
        groupby = [['expt', 'condition_day_session']]

    if plotby is None:
        plotby = ['condition_day_session']

    data_to_save = {}

    figs = []

    fig, axs = plt.subplots(
        2, 4, figsize=(15, 8), subplot_kw={'rasterized': rasterized})
    fig.subplots_adjust(hspace=0.3)

    data_to_save['time_per_lap'] = plot_metric(
        axs[0, 0], exptGrps, metric_fn=eg.time_per_lap,
        groupby=groupby, plotby=plotby, orderby=orderby, colors=colors,
        plot_method='line', activity_label='Time per lap (sec)',
        label_every_n=label_every_n, label_groupby=False)
    data_to_save['fraction_rewarded_laps'] = plot_metric(
        axs[0, 1], exptGrps, metric_fn=ra.fraction_of_laps_rewarded,
        groupby=groupby, plotby=plotby, orderby=orderby, colors=colors,
        plot_method='line', activity_label='Fraction of laps rewarded',
        label_every_n=label_every_n, label_groupby=False)
    data_to_save['rewards_per_lap'] = plot_metric(
        axs[0, 2], exptGrps, metric_fn=eg.stims_per_lap,
        groupby=groupby, plotby=plotby, orderby=orderby, colors=colors,
        plot_method='line', activity_label='Number of rewards per lap',
        activity_kwargs={'stimulus': 'water'},
        label_every_n=label_every_n, label_groupby=False)
    data_to_save['n_laps'] = plot_metric(
        axs[0, 3], exptGrps, metric_fn=eg.number_of_laps,
        groupby=groupby, plotby=plotby, orderby=orderby, colors=colors,
        plot_method='line', activity_label='Number of laps',
        label_every_n=label_every_n, label_groupby=False)

    data_to_save['water_rate'] = plot_metric(
        axs[1, 0], exptGrps, metric_fn=ra.rate_of_water_obtained,
        groupby=groupby, plotby=plotby, orderby=orderby, colors=colors,
        plot_method='line', activity_label='Rate of water obtained (ms/min)',
        label_every_n=label_every_n, label_groupby=False)
    data_to_save['rewarded_lick_duration'] = plot_metric(
        axs[1, 1], exptGrps, metric_fn=eg.lick_bout_duration,
        groupby=groupby, plotby=plotby, orderby=orderby, colors=colors,
        activity_kwargs={'bouts_to_include': 'rewarded', 'threshold': 0.5},
        activity_label='Duration of rewarded lick bouts (s)',
        plot_method='line', label_every_n=label_every_n, label_groupby=False)
    data_to_save['n_licks'] = plot_metric(
        axs[1, 2], exptGrps, metric_fn=eg.behavior_dataframe,
        groupby=groupby, plotby=plotby, orderby=orderby, colors=colors,
        activity_kwargs={'key': 'licking'}, activity_label='Number of licks',
        plot_method='line', agg_fn=np.sum, label_every_n=label_every_n,
        label_groupby=False)

    fig.suptitle('groupby = {}'.format(groupby))

    figs.append(fig)

    if save_data:
        misc.save_data(data_to_save, fig=figs, method=save_data,
                       label='hidden_rewards_behavior_1')

    data_to_save = {}

    fig, axs = plt.subplots(
        2, 4, figsize=(15, 8), subplot_kw={'rasterized': rasterized})
    fig.subplots_adjust(hspace=0.3)

    data_to_save['rewarded_lick_intervals'] = plot_metric(
        axs[0, 0], exptGrps, metric_fn=ra.fraction_rewarded_lick_intervals,
        groupby=groupby, plotby=plotby, orderby=orderby, plot_method='line',
        activity_label='Fraction of lick intervals rewarded', colors=colors,
        activity_kwargs={'threshold': 0.5}, label_every_n=label_every_n,
        label_groupby=False)
    data_to_save['licks_in_rewarded_intervals'] = plot_metric(
        axs[1, 0], exptGrps,
        metric_fn=ra.fraction_licks_in_rewarded_intervals,
        groupby=groupby, plotby=plotby, orderby=orderby, plot_method='line',
        activity_label='Fraction of licks in rewarded intervals',
        colors=colors, activity_kwargs={'threshold': 0.5},
        label_every_n=label_every_n, label_groupby=False)

    data_to_save['licks_in_reward_zone'] = plot_metric(
        axs[0, 1], exptGrps, metric_fn=ra.fraction_licks_in_reward_zone,
        groupby=groupby, plotby=plotby, orderby=orderby, plot_method='line',
        activity_label='Fraction of licks in reward zone', colors=colors,
        label_every_n=label_every_n, label_groupby=False)
    data_to_save['licks_near_rewards'] = plot_metric(
        axs[1, 1], exptGrps, metric_fn=ra.fraction_licks_near_rewards,
        groupby=groupby, plotby=plotby, orderby=orderby, plot_method='line',
        activity_label='Fraction of licks near rewards', colors=colors,
        label_every_n=label_every_n, label_groupby=False)

    data_to_save['licking_spatial_information'] = plot_metric(
        axs[0, 2], exptGrps, metric_fn=ra.licking_spatial_information,
        groupby=groupby, plotby=plotby, orderby=orderby, plot_method='line',
        activity_label='Licking spatial information (bits/sec)', colors=colors,
        label_every_n=label_every_n, label_groupby=False)
    # Licking circular variance

    data_to_save['lick_to_reward_distance'] = plot_metric(
        axs[0, 3], exptGrps, metric_fn=ra.lick_to_reward_distance,
        groupby=groupby, plotby=plotby, orderby=orderby, plot_method='line',
        activity_label='Lick distance to reward (norm units)', colors=colors,
        label_every_n=label_every_n, label_groupby=False)

    data_to_save['licks_outside_reward_vicinity'] = plot_metric(
        axs[1, 2], exptGrps, metric_fn=ra.licks_outside_reward_vicinity,
        groupby=groupby, plotby=plotby, orderby=orderby, plot_method='line',
        activity_label='Fraction of licks outside reward vicinity', colors=colors,
        label_every_n=label_every_n, label_groupby=False)

    # data_to_save['anticipatory_licks'] = plot_metric(
    #     axs[1, 3], exptGrps, metric_fn=ra.anticipatory_licking,
    #     groupby=groupby, plotby=plotby, orderby=orderby, plot_method='line',
    #     activity_label='Anticipatory licking', colors=colors,
    #     label_every_n=label_every_n, label_groupby=False)

    data_to_save['anticipatory_lick_fraction'] = plot_metric(
        axs[1, 3], exptGrps, metric_fn=ra.fraction_licks_near_rewards,
        groupby=groupby, plotby=plotby, orderby=orderby, plot_method='line',
        activity_label='Anticipatory lick fraction', colors=colors,
        label_every_n=label_every_n, label_groupby=False,
        activity_kwargs={'pre_window_cm': 5, 'exclude_reward': True})

    fig.suptitle('groupby = {}'.format(groupby))

    figs.append(fig)

    if save_data:
        misc.save_data(data_to_save, fig=figs, method=save_data,
                       label='hidden_rewards_behavior_2')

    return figs


def hidden_reward_behavior_control_summary(
        exptGrps, save_data=False, rasterized=False, groupby=None, plotby=None,
        orderby=None, colors=None, label_every_n=1):
    """Generate a control figure for hidden rewards behavior experiments."""
    if groupby is None:
        groupby = [['expt', 'condition_day_session']]

    if plotby is None:
        plotby = ['condition_day_session']

    data_to_save = {}

    fig, axs = plt.subplots(
        2, 2, figsize=(15, 8), subplot_kw={'rasterized': rasterized})
    fig.subplots_adjust(hspace=0.3)

    # Grouping by expt, trial, or mouse defeats the purpose of n_sessions plot
    n_ses_groupby = []
    for group in groupby:
        new_groupby = filter(
            lambda x: x not in ['expt', 'trial', 'mouseID'], group)
        if len(new_groupby):
            n_ses_groupby.append(new_groupby)
    if not len(n_ses_groupby):
        n_ses_groupby = None

    data_to_save['n_sessions'] = plot_metric(
        axs[0, 0], exptGrps, metric_fn=eg.dataframe,
        groupby=n_ses_groupby, plotby=plotby, orderby=orderby, colors=colors,
        plot_method='line', activity_label='Total number of sessions',
        label_every_n=label_every_n, agg_fn=np.sum)
    data_to_save['n_laps'] = plot_metric(
        axs[0, 1], exptGrps, metric_fn=eg.number_of_laps,
        groupby=groupby, plotby=plotby, orderby=orderby, colors=colors,
        plot_method='line', activity_label='Number of laps',
        label_every_n=label_every_n)
    data_to_save['reward_windows_per_lap'] = plot_metric(
        axs[1, 0], exptGrps, metric_fn=eg.stims_per_lap,
        groupby=groupby, plotby=plotby, orderby=orderby, colors=colors,
        plot_method='line', activity_label='Number of reward windows per lap',
        activity_kwargs={'stimulus': 'reward'},
        label_every_n=label_every_n)
    data_to_save['reward_position'] = plot_metric(
        axs[1, 1], exptGrps, metric_fn=eg.stim_position,
        groupby=groupby, plotby=plotby, orderby=orderby, colors=colors,
        plot_method='line', activity_label='Mean reward location',
        activity_kwargs={'stimulus': 'reward', 'normalized': False},
        label_every_n=label_every_n)

    try:
        expected_positions = [expt.rewardPositions(units=None)
                              for exptGrp in exptGrps for expt in exptGrp]
        expected_positions = set(it.chain(*expected_positions))
    except AttributeError:
        pass
    else:
        for position in expected_positions:
            axs[1, 1].axhline(position, color='red')

    if save_data:
        misc.save_data(data_to_save, fig=fig, method=save_data,
                       label='hidden_rewards_control')

    return fig


def hidden_rewards_move_rewards_learning(
        exptGrps, groupby=None, plotby=None, orderby=None, colors=None,
        label_every_n=1, rasterized=False, save_data=False,
        rewards='combined', by_condition=False):

    if groupby is None:
        groupby = [['expt', 'condition_day_session']]

    if plotby is None:
        plotby = ['condition_day_session']

    data_to_save = {}

    if rewards == 'combined':
        reward_positions = set()
        for exptGrp in exptGrps:
            if by_condition:
                conditions, _ = exptGrp.condition_label(by_mouse=True)
                reward_positions = reward_positions.union(conditions.values())
            else:
                for expt in exptGrp:
                    for pos in expt.rewardPositions(units=None):
                        reward_positions.add(pos)
        reward_positions = sorted(reward_positions)
    elif rewards == 'separate':
        reward_positions = {}
        for exptGrp in exptGrps:
            reward_positions[exptGrp] = set()
            if by_condition:
                conditions, _ = exptGrp.condition_label(by_mouse=True)
                reward_positions = reward_positions[exptGrp].union(
                    conditions.values())
            else:
                for expt in exptGrp:
                    for pos in expt.rewardPositions(units=None):
                        reward_positions[exptGrp].add(pos)
            reward_positions[exptGrp] = sorted(reward_positions[exptGrp])

    if colors is None:
        if rewards == 'combined':
            colors = sns.color_palette(
                "Paired", len(exptGrps) * len(reward_positions))
        if rewards == 'separate':
            colors = sns.color_palette(
                "Paired", len(exptGrps) * sum(map(len, reward_positions)))
    else:
        # Lightest is too light, so add an extra color that we'll ignore
        colors = [sns.light_palette(
            color, len(reward_positions) + 1,
            reverse=True)[:len(reward_positions)] for color in colors]
        colors = list(it.chain(*colors))

    new_exptGrps = []
    activity_kwargs = []
    for exptGrp in exptGrps:
        if rewards == 'combined':
            pos_iter = reward_positions
        elif rewards == 'separate':
            pos_iter = reward_positions[exptGrp]
        for pos in pos_iter:
            new_exptGrp = lab.classes.HiddenRewardExperimentGroup(exptGrp)
            if by_condition:
                new_exptGrp.label(exptGrp.label() + '_{}'.format(pos))
                activity_kwargs.append({'rewardPositions': pos})
            else:
                new_exptGrp.label(exptGrp.label() + '_{:0.1f}'.format(pos))
                activity_kwargs.append({'rewardPositions': [pos]})
            new_exptGrps.append(new_exptGrp)

    fig, axs = plt.subplots(
        1, 3, figsize=(15, 8), subplot_kw={'rasterized': rasterized},
        squeeze=False)
    fig.subplots_adjust(hspace=0.3)

    data_to_save['lick_to_reward_distance'] = plot_metric(
        axs[0, 0], new_exptGrps, metric_fn=ra.lick_to_reward_distance,
        groupby=groupby, plotby=plotby, orderby=orderby, colors=colors,
        plot_method='line', activity_kwargs=activity_kwargs,
        activity_label='lick distance to reward (norm units)',
        label_every_n=label_every_n, label_groupby=False)
    data_to_save['licks_near_rewards'] = plot_metric(
        axs[0, 1], new_exptGrps, metric_fn=ra.fraction_licks_near_rewards,
        groupby=groupby, plotby=plotby, orderby=orderby, colors=colors,
        plot_method='line', activity_kwargs=activity_kwargs,
        activity_label='fraction of licks near rewards',
        label_every_n=label_every_n, label_groupby=False)
    data_to_save['fraction_laps_licking'] = plot_metric(
        axs[0, 2], new_exptGrps,
        metric_fn=ra.fraction_of_laps_with_licking_near_reward,
        groupby=groupby, plotby=plotby, orderby=orderby, colors=colors,
        plot_method='line', activity_kwargs=activity_kwargs,
        activity_label='fraction of laps w/ licks near rewards',
        label_every_n=label_every_n, label_groupby=False)

    fig.suptitle('groupby = {}'.format(groupby))

    if save_data:
        misc.save_data(data_to_save, fig=fig, method=save_data,
                       label='hidden_rewards_behavior')

    return fig


def stim_response_summary(
        expt_grp, stimuli, pre_time=None, post_time=None, channel='Ch2',
        label=None, roi_filter=None):

    fig, axs = plt.subplots(2, len(stimuli), figsize=(15, 8))

    for stim, ax_pair in zip(stimuli, axs.T):
        ia.PSTH(
            expt_grp, stim, ax=ax_pair[0], pre_time=pre_time, post_time=post_time,
            shade_ste=False, plot_mean=True, channel=channel, label=label,
            roi_filter=roi_filter, gray_traces=True)
        ia.PSTH(
            expt_grp, stim, ax=ax_pair[1], pre_time=pre_time, post_time=post_time,
            shade_ste='sem', plot_mean=True, channel=channel, label=label,
            roi_filter=roi_filter)
        ax_pair[0].set_title(stim)
        ax_pair[0].set_xlabel('')
        ax_pair[0].tick_params(axis='x', labelbottom=False)

    min_y, max_y = np.inf, -np.inf
    for ax in axs[0, :]:
        min_y = np.amin([min_y, ax.get_ylim()[0]])
        max_y = np.amax([max_y, ax.get_ylim()[1]])
    for ax in axs[0, :]:
        ax.set_ylim(min_y, max_y)

    min_y, max_y = np.inf, -np.inf
    for ax in axs[1, :]:
        min_y = np.amin([min_y, ax.get_ylim()[0]])
        max_y = np.amax([max_y, ax.get_ylim()[1]])
    for ax in axs[1, :]:
        ax.set_ylim(min_y, max_y)

    for ax_row in axs[:, 1:]:
        for ax in ax_row:
            ax.set_ylabel('')

    return fig


def licktogram_summary(expt_grps, rasterized=False, polar=False):
    """Plots licktograms for every condition/day by mouse"""

    dataframes = [expt_grp.dataframe(
        expt_grp, include_columns=['mouseID', 'expt', 'condition', 'session'])
        for expt_grp in expt_grps]
    dataframe = pd.concat(dataframes)

    mouse_grp_dict = {
        mouse: expt_grp.label() for expt_grp in expt_grps for mouse in
        set(expt.parent.get('mouseID') for expt in expt_grp)}

    fig_dict = {}
    for mouse_id, df in dataframe.groupby('mouseID'):
        n_rows = len(set(df['condition']))
        n_cols = df['session'].max() + 1

        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(15, 8), sharey=not polar,
            subplot_kw={'rasterized': rasterized, 'polar': polar},
            squeeze=False)

        for c_idx, condition in enumerate(sorted(set(df['condition']))):
            for session in range(n_cols):
                df_slice = df[(df['condition'] == condition) &
                              (df['session'] == session)]
                if len(df_slice) == 1:
                    expt = df_slice['expt'].iloc[0]
                    if polar:
                        expt.polar_lick_plot(ax=axs[c_idx, session])
                    else:
                        expt.licktogram(
                            ax=axs[c_idx, session], plot_belt=False)
                else:
                    axs[c_idx, session].set_visible(False)

        for ax, condition in zip(axs[:, -1], sorted(set(df['condition']))):
            plotting.right_label(ax, condition)
        for ax, session in zip(axs[0, :], range(1, n_cols + 1)):
            ax.set_title('Session {}'.format(session))
        for ax in axs[:, 1:].flat:
            ax.set_ylabel('')
        for ax in axs[1:, :].flat:
            ax.set_title('')
        for ax in axs[:-1, :].flat:
            ax.set_xlabel('')

        fig.suptitle('{}: {}'.format(mouse_grp_dict[mouse_id], mouse_id))

        fig_dict[mouse_id] = fig

    return [fig_dict[mouse] for mouse in sorted(fig_dict.keys())]


def behavior_cross_correlation(
        expt_grps, roi_filters, behavior_key, channel='Ch2', label=None,
        rasterized=False, max_lag=10, thresh=0.5, colors=None):

    if colors is None:
        colors = sns.color_palette()

    fig, axs = plt.subplots(
        3, len(expt_grps) + 1, squeeze=False,
        subplot_kw={'rasterized': rasterized}, figsize=(15, 8))

    fig.suptitle('Imaging-behavior cross-correlation: {}'.format(behavior_key))

    corrs = {}

    zero_lag, peak_offset = [], []
    for grp_axs, color, expt_grp, roi_filter in zip(
            axs.T, colors, expt_grps, roi_filters):
        corr = sa.xcorr_imaging_behavior(
            expt_grp, behavior_key, max_lag=max_lag, thresh=thresh,
            return_full=False, channel=channel, label=label,
            roi_filter=roi_filter)

        assert 0. in corr.index

        corrs[expt_grp] = corr
        zero_lag.append([np.array(corr[corr.index == 0])[0]])
        peak_offset.append([np.array(
            [corr.index[i] for i in np.argmax(
                np.abs(np.array(corr)), axis=0)])])

        light_color = sns.light_palette(color)[1]

        grp_axs[0].plot(corr.index, corr, color=light_color)
        grp_axs[0].plot(corr.index, corr.mean(1), color=color)
        grp_axs[0].set_xlim(corr.index[0], corr.index[-1])
        grp_axs[0].set_title(expt_grp.label())
        grp_axs[0].set_xlabel('Lag (s)')
        grp_axs[0].set_ylabel('Cross-correlation')

        plotting.histogram(
            grp_axs[1], zero_lag[-1][0], bins=10,
            range=(-1, 1), color=color, normed=False,
            plot_mean=True, label=None, orientation='vertical', filled=True,
            mean_kwargs=None)

        grp_axs[1].set_xlabel('zero-lag cross-correlation')
        grp_axs[1].set_ylabel('ROIs')

        plotting.histogram(
            grp_axs[2], peak_offset[-1][0], bins=10,
            range=(-max_lag, max_lag), color=color, normed=False,
            plot_mean=True, label=None, orientation='vertical', filled=True,
            mean_kwargs=None)

        grp_axs[2].set_xlabel('Time to peak (s)')
        grp_axs[2].set_ylabel('ROIs')

    #
    # Directly compare
    #

    for expt_grp, color in zip(expt_grps, colors):
        corr = corrs[expt_grp]

        axs[0, -1].plot(
            corr.index, corr.mean(1), color=color, label=expt_grp.label())
        axs[0, -1].fill_between(
            corr.index, corr.mean(1) - corr.sem(1), corr.mean(1) + corr.sem(1),
            color=color, alpha=0.5)

    axs[0, -1].set_xlim(corr.index[0], corr.index[-1])
    axs[0, -1].set_xlabel('Lag (s)')
    axs[0, -1].set_ylabel('Cross-correlation')

    min_y, max_y = np.inf, - np.inf
    for ax in axs[0, :]:
        min_y = min(min_y, ax.get_ylim()[0])
        max_y = max(max_y, ax.get_ylim()[1])
    for ax in axs[0, :]:
        ax.set_ylim(min_y, max_y)

    axs[0, -1].legend(frameon=False, loc='best')

    plotting.grouped_bar(
        axs[1, -1], values=zero_lag, cluster_labels=[''],
        condition_labels=[expt_grp.label() for expt_grp in expt_grps],
        bar_colors=colors, scatter_points=True, jitter_x=True, s=20)

    axs[1, -1].set_ylabel('zero-lag cross-correlation')

    plotting.grouped_bar(
        axs[2, -1], values=peak_offset, cluster_labels=[''],
        condition_labels=[expt_grp.label() for expt_grp in expt_grps],
        bar_colors=colors, scatter_points=True, jitter_x=True, s=20)

    axs[2, -1].set_ylabel('Time to peak (s)')

    return fig


def plotControlSummary(
        exptGrps, roi_filters=None, channel='Ch2', label=None,
        rasterized=False, groupby=None, plotby=None, **plot_kwargs):
    """Plot a series of potentially control analysis, looking at similarity of
    data over time.

    """

    fig, axs = plt.subplots(
        2, 3, figsize=(15, 8), subplot_kw={'rasterized': rasterized})
    fig.subplots_adjust(hspace=0.3)

    base_kwargs = {'channel': channel, 'label': label}

    activity_kwargs = base_kwargs.copy()
    activity_kwargs.update({'stat': 'mean'})
    plot_metric(
        ax=axs[0, 0], exptGrps=exptGrps, roi_filters=roi_filters,
        metric_fn=ia.population_activity, plot_method='line',
        groupby=groupby, plotby=plotby, activity_kwargs=activity_kwargs,
        activity_label="Mean dF/F", **plot_kwargs)

    activity_kwargs = base_kwargs.copy()
    activity_kwargs.update({'stat': 'amplitude'})
    plot_metric(
        ax=axs[0, 1], exptGrps=exptGrps, roi_filters=roi_filters,
        metric_fn=ia.population_activity, plot_method='line',
        groupby=groupby, plotby=plotby, activity_kwargs=activity_kwargs,
        activity_label="Mean transient amplitude", **plot_kwargs)

    activity_kwargs = base_kwargs.copy()
    activity_kwargs.update({'stat': 'duration'})
    plot_metric(
        ax=axs[1, 0], exptGrps=exptGrps, roi_filters=roi_filters,
        metric_fn=ia.population_activity, plot_method='line',
        groupby=groupby, plotby=plotby, activity_kwargs=activity_kwargs,
        activity_label="Mean transient duration", **plot_kwargs)

    activity_kwargs = base_kwargs.copy()
    activity_kwargs.update({'stat': 'frequency'})
    plot_metric(
        ax=axs[1, 1], exptGrps=exptGrps, roi_filters=roi_filters,
        metric_fn=ia.population_activity, plot_method='line',
        groupby=groupby, plotby=plotby, activity_kwargs=activity_kwargs,
        activity_label="Mean transient frequency", **plot_kwargs)

    activity_kwargs = base_kwargs.copy()
    plot_metric(
        ax=axs[0, 2], exptGrps=exptGrps, roi_filters=roi_filters,
        metric_fn=ia.trace_sigma, plot_method='line',
        groupby=groupby, plotby=plotby, activity_kwargs=activity_kwargs,
        activity_label='Trace sigma', **plot_kwargs)

    activity_kwargs = base_kwargs.copy()
    plot_metric(
        ax=axs[1, 2], exptGrps=exptGrps, roi_filters=roi_filters,
        metric_fn=ia.mean_fluorescence, plot_method='line',
        groupby=groupby, plotby=plotby, activity_kwargs=activity_kwargs,
        activity_label='Mean raw fluorescence', **plot_kwargs)

    return fig


def plot_calcium_dynamics_summary(
        expt_grps, roi_filters=None, channel='Ch2', label=None,
        rasterized=False, groupby=None, plotby=None, plot_method='cdf',
        **plot_kwargs):
    """A set of control plots designed to compare baseline calcium properties
    between genotypes.

    """

    fig, axs = plt.subplots(
        2, 3, figsize=(15, 8), subplot_kw={'rasterized': rasterized})
    fig.subplots_adjust(hspace=0.3)

    # Trans psth
    # base_kwargs = {'channel': channel, 'label': label}
    base_kwargs = []
    for expt_grp in expt_grps:
        grp_kwargs = {}
        try:
            grp_kwargs['channel'] = expt_grp.args['channel']
        except KeyError:
            grp_kwargs['channel'] = channel

        try:
            grp_kwargs['label'] = expt_grp.args['imaging_label']
        except KeyError:
            grp_kwargs['label'] = label

    activity_kwargs = [dict(bkw.items() + [('stat', 'amplitude')])
                       for bkw in base_kwargs]
    plot_metric(
        ax=axs[0, 1], exptGrps=expt_grps, roi_filters=roi_filters,
        metric_fn=ia.population_activity, plot_method=plot_method,
        groupby=groupby, plotby=plotby, activity_kwargs=activity_kwargs,
        activity_label="Mean transient amplitude", **plot_kwargs)

    activity_kwargs = [dict(bkw.items() + [('stat', 'duration')])
                       for bkw in base_kwargs]
    plot_metric(
        ax=axs[1, 0], exptGrps=expt_grps, roi_filters=roi_filters,
        metric_fn=ia.population_activity, plot_method=plot_method,
        groupby=groupby, plotby=plotby, activity_kwargs=activity_kwargs,
        activity_label="Mean transient duration", **plot_kwargs)

    activity_kwargs = [dict(bkw.items() + [('stat', 'frequency')])
                       for bkw in base_kwargs]
    plot_metric(
        ax=axs[1, 1], exptGrps=expt_grps, roi_filters=roi_filters,
        metric_fn=ia.population_activity, plot_method=plot_method,
        groupby=groupby, plotby=plotby, activity_kwargs=activity_kwargs,
        activity_label="Mean transient frequency", **plot_kwargs)

    activity_kwargs = base_kwargs
    plot_metric(
        ax=axs[0, 2], exptGrps=expt_grps, roi_filters=roi_filters,
        metric_fn=ia.trace_sigma, plot_method=plot_method,
        groupby=groupby, plotby=plotby, activity_kwargs=activity_kwargs,
        activity_label='Trace sigma', **plot_kwargs)

    activity_kwargs = base_kwargs
    plot_metric(
        ax=axs[1, 2], exptGrps=expt_grps, roi_filters=roi_filters,
        metric_fn=ia.mean_fluorescence, plot_method=plot_method,
        groupby=groupby, plotby=plotby, activity_kwargs=activity_kwargs,
        activity_label='Mean raw fluorescence', **plot_kwargs)

    return fig


def transient_summary(
        expt_grps, plot_method, intervals='running', roi_filters=None,
        groupby=None, plotby=None, label_every_n=1, save_data=False,
        rasterized=False, interval_kwargs=None, channel='Ch2', label=None,
        **plot_kwargs):
    """Generate a summary plot of place field transient statistics."""

    if interval_kwargs is None:
        interval_kwargs = {}

    if roi_filters is None:
        roi_filters = [None] * len(expt_grps)

    if intervals == 'running':
        kwargs = {}
        kwargs.update(interval_kwargs)
        in_intervals = [inter.running_intervals(
            expt_grp, **kwargs) for expt_grp in expt_grps]
        out_intervals = [~ints for ints in in_intervals]
    elif intervals == 'place field':
        kwargs = {}
        kwargs.update(interval_kwargs)
        in_intervals = [inter.place_fields(
            expt_grp, roi_filter=roi_filter, **kwargs) for
            expt_grp, roi_filter in zip(expt_grps, roi_filters)]
        out_intervals = [~ints for ints in in_intervals]
    elif intervals == 'reward':
        kwargs = {'nearness': 0.1}
        kwargs.update(interval_kwargs)
        in_intervals = [inter.near_rewards(
            expt_grp, **kwargs) for expt_grp in expt_grps]
        out_intervals = [~ints for ints in in_intervals]
    else:
        raise ValueError("Unrecognized value for 'intervals' argument")

    data_to_save = {}

    fig, axs = plt.subplots(
        3, 5, figsize=(15, 8), subplot_kw={'rasterized': rasterized},
        sharey='col')
    fig.subplots_adjust(hspace=0.3)

    activity_kwargs = {'stat': 'amplitude', 'interval': None, 'channel': channel, 'label': label}
    data_to_save['amplitude_all'] = plot_metric(
        axs[0, 0], expt_grps, metric_fn=ia.population_activity_new,
        plot_method=plot_method, roi_filters=roi_filters, groupby=groupby,
        plotby=plotby, activity_kwargs=activity_kwargs, activity_label='',
        label_every_n=label_every_n, **plot_kwargs)
    axs[0, 0].set_title('amplitude')
    activity_kwargs = [
        {'stat': 'amplitude', 'interval': grp_interval, 'channel': channel, 'label': label}
        for grp_interval in in_intervals]
    data_to_save['amplitude_in'] = plot_metric(
        axs[1, 0], expt_grps, metric_fn=ia.population_activity_new,
        plot_method=plot_method, roi_filters=roi_filters, groupby=groupby,
        plotby=plotby, activity_kwargs=activity_kwargs, activity_label='',
        label_every_n=label_every_n, **plot_kwargs)
    activity_kwargs = [
        {'stat': 'amplitude', 'interval': grp_interval, 'channel': channel, 'label': label}
        for grp_interval in out_intervals]
    data_to_save['amplitude_out'] = plot_metric(
        axs[2, 0], expt_grps, metric_fn=ia.population_activity_new,
        plot_method=plot_method, roi_filters=roi_filters, groupby=groupby,
        plotby=plotby, activity_kwargs=activity_kwargs, activity_label='',
        label_every_n=label_every_n, **plot_kwargs)

    activity_kwargs = {'stat': 'duration', 'interval': None, 'channel': channel, 'label': label}
    data_to_save['duration_all'] = plot_metric(
        axs[0, 1], expt_grps, metric_fn=ia.population_activity_new,
        plot_method=plot_method, roi_filters=roi_filters, groupby=groupby,
        plotby=plotby, activity_kwargs=activity_kwargs, activity_label='',
        label_every_n=label_every_n, **plot_kwargs)
    axs[0, 1].set_title('duration')
    activity_kwargs = [
        {'stat': 'duration', 'interval': grp_interval, 'channel': channel, 'label': label}
        for grp_interval in in_intervals]
    data_to_save['duration_in'] = plot_metric(
        axs[1, 1], expt_grps, metric_fn=ia.population_activity_new,
        plot_method=plot_method, roi_filters=roi_filters, groupby=groupby,
        plotby=plotby, activity_kwargs=activity_kwargs, activity_label='',
        label_every_n=label_every_n, **plot_kwargs)
    activity_kwargs = [
        {'stat': 'duration', 'interval': grp_interval, 'channel': channel, 'label': label}
        for grp_interval in out_intervals]
    data_to_save['duration_out'] = plot_metric(
        axs[2, 1], expt_grps, metric_fn=ia.population_activity_new,
        plot_method=plot_method, roi_filters=roi_filters, groupby=groupby,
        plotby=plotby, activity_kwargs=activity_kwargs, activity_label='',
        label_every_n=label_every_n, **plot_kwargs)

    activity_kwargs = {'stat': 'responseMagnitude', 'interval': None, 'channel': channel, 'label': label}
    data_to_save['magnitude_all'] = plot_metric(
        axs[0, 2], expt_grps, metric_fn=ia.population_activity_new,
        plot_method=plot_method, roi_filters=roi_filters, groupby=groupby,
        plotby=plotby, activity_kwargs=activity_kwargs, activity_label='',
        label_every_n=label_every_n, **plot_kwargs)
    axs[0, 2].set_title('responseMagnitude')
    activity_kwargs = [
        {'stat': 'responseMagnitude', 'interval': grp_interval, 'channel': channel, 'label': label}
        for grp_interval in in_intervals]
    data_to_save['magnitude_in'] = plot_metric(
        axs[1, 2], expt_grps, metric_fn=ia.population_activity_new,
        plot_method=plot_method, roi_filters=roi_filters, groupby=groupby,
        plotby=plotby, activity_kwargs=activity_kwargs, activity_label='',
        label_every_n=label_every_n, **plot_kwargs)
    activity_kwargs = [
        {'stat': 'responseMagnitude', 'interval': grp_interval, 'channel': channel, 'label': label}
        for grp_interval in out_intervals]
    data_to_save['magnitude_out'] = plot_metric(
        axs[2, 2], expt_grps, metric_fn=ia.population_activity_new,
        plot_method=plot_method, roi_filters=roi_filters, groupby=groupby,
        plotby=plotby, activity_kwargs=activity_kwargs, activity_label='',
        label_every_n=label_every_n, **plot_kwargs)

    activity_kwargs = {'stat': 'norm transient auc2', 'interval': None, 'channel': channel, 'label': label}
    data_to_save['auc_all'] = plot_metric(
        axs[0, 3], expt_grps, metric_fn=ia.population_activity_new,
        plot_method=plot_method, roi_filters=roi_filters, groupby=groupby,
        plotby=plotby, activity_kwargs=activity_kwargs, activity_label='',
        label_every_n=label_every_n, **plot_kwargs)
    axs[0, 3].set_title('norm transient auc2')
    activity_kwargs = [
        {'stat': 'norm transient auc2', 'interval': grp_interval, 'channel': channel, 'label': label}
        for grp_interval in in_intervals]
    data_to_save['auc_in'] = plot_metric(
        axs[1, 3], expt_grps, metric_fn=ia.population_activity_new,
        plot_method=plot_method, roi_filters=roi_filters, groupby=groupby,
        plotby=plotby, activity_kwargs=activity_kwargs, activity_label='',
        label_every_n=label_every_n, **plot_kwargs)
    activity_kwargs = [
        {'stat': 'norm transient auc2', 'interval': grp_interval, 'channel': channel, 'label': label}
        for grp_interval in out_intervals]
    data_to_save['auc_out'] = plot_metric(
        axs[2, 3], expt_grps, metric_fn=ia.population_activity_new,
        plot_method=plot_method, roi_filters=roi_filters, groupby=groupby,
        plotby=plotby, activity_kwargs=activity_kwargs, activity_label='',
        label_every_n=label_every_n, **plot_kwargs)

    activity_kwargs = {'stat': 'frequency', 'interval': None, 'channel': channel, 'label': label}
    data_to_save['frequency_all'] = plot_metric(
        axs[0, 4], expt_grps, metric_fn=ia.population_activity_new,
        plot_method=plot_method, roi_filters=roi_filters, groupby=groupby,
        plotby=plotby, activity_kwargs=activity_kwargs, activity_label='',
        label_every_n=label_every_n, **plot_kwargs)
    axs[0, 4].set_title('frequency')
    activity_kwargs = [
        {'stat': 'frequency', 'interval': grp_interval, 'channel': channel, 'label': label}
        for grp_interval in in_intervals]
    data_to_save['frequency_in'] = plot_metric(
        axs[1, 4], expt_grps, metric_fn=ia.population_activity_new,
        plot_method=plot_method, roi_filters=roi_filters, groupby=groupby,
        plotby=plotby, activity_kwargs=activity_kwargs, activity_label='',
        label_every_n=label_every_n, **plot_kwargs)
    activity_kwargs = [
        {'stat': 'frequency', 'interval': grp_interval, 'channel': channel, 'label': label}
        for grp_interval in out_intervals]
    data_to_save['frequency_out'] = plot_metric(
        axs[2, 4], expt_grps, metric_fn=ia.population_activity_new,
        plot_method=plot_method, roi_filters=roi_filters, groupby=groupby,
        plotby=plotby, activity_kwargs=activity_kwargs, activity_label='',
        label_every_n=label_every_n, **plot_kwargs)

    # Remove extra labels
    for ax in axs[:2, :].flat:
        ax.set_xlabel('')
    for ax in axs[:, 1:].flat:
        ax.set_ylabel('')
    for ax in axs[1:, :].flat:
        ax.set_title('')

    plotting.right_label(axs[0, -1], 'all trans')
    plotting.right_label(axs[1, -1], 'trans in')
    plotting.right_label(axs[2, -1], 'trans out')

    fig.suptitle('Activity by {}\ngroupby={}'.format(intervals, groupby))

    if save_data:
        misc.save_data(data_to_save, fig=fig, label='transient_summary',
                       method=save_data)

    return fig


def thresholded_metric_vs_metric_figure(
        exptGrps, x_metric, y_metric, filter_metric, thresholds, roi_filters=None,
        x_metric_kwargs=None, y_metric_kwargs=None, filter_metric_kwargs=None,
        xlabel=None, ylabel=None, plot_method='scatter', groupby=None,
        colorby=None, filter_on=('roi',), title='', save_data=None, filter_fn=None,
        **plot_kwargs):

    fig, axs = plt.subplots(3, len(thresholds), figsize=(15, 8))

    data_to_save = {}

    if xlabel is None:
        xlabel = 'Metric 1'
    if ylabel is None:
        ylabel = 'Metric 2'

    filter_fns = [misc.df_filter_intersection([None, filter_fn]),
                  misc.df_filter_intersection([lambda df: df['filter_metric_value'] < threshold, filter_fn]),
                  misc.df_filter_intersection([lambda df: df['filter_metric_value'] > threshold, filter_fn])]

    filter_labels = ['all', 'less_than', 'greater_than']

    for col, threshold in enumerate(thresholds):
        for row, filter_fn, filter_label in zip(
                it.count(), filter_fns, filter_labels):
            label = '{}_{}'.format(filter_label, threshold)
            data_to_save[label] = plot_paired_metrics(
                exptGrps, roi_filters=roi_filters, ax=axs[row, col],
                first_metric_fn=x_metric, second_metric_fn=y_metric,
                first_metric_kwargs=x_metric_kwargs,
                second_metric_kwargs=y_metric_kwargs,
                first_metric_label=xlabel,
                second_metric_label=ylabel,
                plot_method=plot_method,
                groupby=groupby,
                colorby=colorby,
                filter_metric_fn=filter_metric,
                filter_metric_merge_on=filter_on,
                filter_metric_fn_kwargs=filter_metric_kwargs,
                filter_fn=filter_fn, **plot_kwargs)
        axs[0, col].set_title('Threshold = {}'.format(threshold))

    for ax, label in zip(axs[:, -1], filter_labels):
        plotting.right_label(ax, label)

    for ax in axs[:, 1:].flat:
        ax.set_ylabel('')

    for ax in axs[:-1, :].flat:
        ax.set_xlabel('')

    fig.suptitle(title)

    if save_data:
        misc.save_data(data_to_save, fig=fig,
                       label='thresholded_metric_vs_metric', method=save_data)

    return fig


def hidden_rewards_number_of_licks(
        expt_grps, rasterized=False, groupby=None, plotby=None,
        label_every_n=1, **plot_kwargs):
    """Plots the total number of licks in vs out of reward per mouse"""

    if groupby is None:
        groupby = [['expt', 'condition_day_session']]

    if plotby is None:
        plotby = ['condition_day_session']

    mice = {}
    max_mice = -1
    for expt_grp in expt_grps:
        mice[expt_grp] = {expt.parent for expt in expt_grp}
        max_mice = max(max_mice, len(mice[expt_grp]))

    fig, axs = plt.subplots(
        len(expt_grps), max_mice, figsize=(15, 8), squeeze=False,
        subplot_kw={'rasterized': rasterized})
    fig.subplots_adjust(hspace=0.3)

    for expt_grp, grp_axs in zip(expt_grps, axs):
        for mouse, ax in zip(sorted(mice[expt_grp]), grp_axs):
            mouse_expt_grp = expt_grp.subGroup(
                [expt for expt in expt_grp if expt.parent == mouse],
                label='near')

            colors = color_cycle()

            plot_metric(
                ax, [mouse_expt_grp],
                metric_fn=ra.number_licks_near_rewards,
                plot_method='line', groupby=groupby, plotby=plotby,
                label_every_n=label_every_n,  colors=[colors.next()],
                activity_label='Number of licks', **plot_kwargs)

            mouse_expt_grp.label('away')
            plot_metric(
                ax, [mouse_expt_grp],
                metric_fn=ra.number_licks_away_rewards,
                plot_method='line', groupby=groupby, plotby=plotby,
                label_every_n=label_every_n, colors=[colors.next()],
                activity_label='Number of licks', **plot_kwargs)

            ax.set_title(mouse.get('mouseID'))

    for ax in axs[:, 1:].flat:
        ax.set_ylabel('')
        ax.tick_params(labelleft=False)

    for ax in axs.flat:
        ax.set_xlabel('')
        ax.tick_params(top=False)

    max_licks = -np.inf
    for ax in axs.flat:
        max_licks = max(max_licks, ax.get_ylim()[1])
    for ax in axs.flat:
        ax.set_ylim(top=max_licks)

    for ax in list(axs.flat)[1:]:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_visible(False)

    for expt_grp, ax in zip(expt_grps, axs[:, -1]):
        plotting.right_label(ax, expt_grp.label())

    fig.suptitle(
        'Number of licks near/away from reward\ngroupby = {}'.format(groupby))

    return fig


def salience_responsiveness_figure_by_cell(
        expt_grp, stimuli, plotby, method='responsiveness', pre_time=None,
        post_time=None, channel='Ch2', label=None, roi_filter=None,
        exclude_running=False, rasterized=False, save_data=False,
        n_bootstraps=10000, n_processes=1):
    """Plots the stimulus responsiveness versus the 'plotby'. For example, the
    response to water rewards over days of exposure.

    Yields 1 figure per ROI with a grid of plots, 1 per stimulus in 'stimuli'.

    Parameters
    ----------
    expt_grp, channel, label, roi_filter
        Standard analysis arguments.
    stimuli : list
        List of stimuli.
    plotby : list
        List of keys that will determine the x-axis of the plot.
        See lab.plotting.plotting_helpers.prepare_dataframe
    method : 'responsiveness' or 'peak'
        Method to determine the response to the stimuli.
    pre_time, post_time : float
        Duration of baseline (pre_time) and response time (post_time).

    Yields
    ------
    mpl.pyplot.Figure

    """

    # data_to_save = {}
    N_COLS = 4
    n_rows = int(np.ceil(len(stimuli) / float(N_COLS)))
    n_extra_axs = (N_COLS * n_rows) % len(stimuli)

    if exclude_running:
        stimuli = [stim for stim in stimuli if 'running' not in stim]

    if not len(stimuli):
        warn("No stimuli to analyze, aborting.")
        return

    if method == 'responsiveness':
        activity_label = 'Responsiveness (dF/F)'
    elif method == 'peak':
        activity_label = 'Peak responsiveness (dF/F)'
    else:
        raise ValueError("Unrecognized 'method' value")

    responsiveness = {}
    all_roi_tuples = set()
    for stimulus in stimuli:
        stimulus_dfs = []
        for key, grp in expt_grp.groupby(plotby):
            df = ia.response_magnitudes(
                grp, stimulus, method=method, pre_time=pre_time,
                post_time=post_time, data=None,
                exclude='running' if exclude_running else None,
                channel=channel, label=label,
                roi_filter=roi_filter, return_df=True)
            # Put the grouping info back in the dataframe
            # For example:
            # plotby = ['condition_day']
            # keys will be ['A_0', 'A_1', 'B_0', etc...]
            # So df['condition_day'] == 'A_0' for the first group, etc.
            for key_value, grouping in zip(key, plotby):
                df[grouping] = key_value
            stimulus_dfs.append(df)
        joined_df = pd.concat(
            stimulus_dfs, ignore_index=True)
        joined_df['roi_tuple'] = zip(
            joined_df['mouse'].apply(lambda mouse: mouse.get('mouseID')),
            joined_df['uniqueLocationKey'],
            joined_df['roi_id'])
        responsiveness[stimulus] = joined_df
        all_roi_tuples = all_roi_tuples.union(joined_df['roi_tuple'])

    for roi_tuple in sorted(all_roi_tuples):
        fig, axs = plt.subplots(
            n_rows, N_COLS, figsize=(15, 8), squeeze=False,
            subplot_kw={'rasterized': rasterized})
        fig.subplots_adjust(hspace=0.3)
        first_col_axs = axs[:, 0]
        fig.suptitle(roi_tuple)
        min_response_y_lim, max_response_y_lim = np.inf, -np.inf
        for ax, stimulus in it.izip(axs.flat, stimuli):
            data = responsiveness[stimulus]
            data = data[data['roi_tuple'].apply(lambda val: val == roi_tuple)]
            plotting.plot_dataframe(
                ax, [data],
                activity_label=activity_label, groupby=None, plotby=plotby,
                orderby=None, plot_method='line', plot_shuffle=False,
                shuffle_plotby=False, pool_shuffle=False,
                agg_fn=np.mean)

            min_response_y_lim = np.amin([min_response_y_lim, ax.get_ylim()[0]])
            max_response_y_lim = np.amax([max_response_y_lim, ax.get_ylim()[1]])
            ax.set_title(stimulus)
            plt.setp(ax.get_xticklabels(), rotation='40',
                     horizontalalignment='right')

        if n_extra_axs:
            for ax in np.array(axs.flat)[-n_extra_axs:]:
                ax.set_visible(False)

        for ax in set(axs.flat).difference(first_col_axs):
            ax.set_ylabel('')

        for ax in axs.flat:
            ax.set_ylim(min_response_y_lim, max_response_y_lim)
            ax.set_xlabel('')
            legend = ax.get_legend()
            if legend is not None:
                legend.set_visible(False)

        yield fig


def behavior_psth_figure(
        expt_grps, stimulus_key, data_key, groupby, rasterized=False,
        **behaviorPSTH_kwargs):
    """Returns a figure of behavior data PSTHS of experiment subgroups.

    Figure will be an array of plots, n_expt_grps x n_groupby_groups.

    """

    all_expts = lab.ExperimentGroup([expt for expt in it.chain(*expt_grps)])

    n_groupbys = len(list(all_expts.groupby(groupby)))

    fig, axs = plt.subplots(
        len(expt_grps), n_groupbys, figsize=(15, 8), squeeze=False,
        subplot_kw={'rasterized': rasterized})

    for grp_axs, (grp_label, subgrp) in zip(axs.T, all_expts.groupby(groupby)):
        for ax, expt_grp in zip(grp_axs, expt_grps):
            expt_grp_subgrp = copy(expt_grp)
            expt_grp_subgrp.filter(lambda expt: expt in subgrp)

            if not len(expt_grp_subgrp):
                ax.set_visible(False)
                continue

            lab.analysis.behavior_analysis.plotBehaviorPSTH(
                expt_grp_subgrp, stimulus_key, data_key, ax=ax,
                **behaviorPSTH_kwargs)

        grp_axs[0].set_title(str(grp_label))

    for ax, expt_grp in zip(axs[:, -1], expt_grps):
        plotting.right_label(ax, expt_grp.label())

    fig.suptitle('{} triggered {} PSTH\ngroupby={}'.format(
        stimulus_key, data_key, groupby))

    return fig
