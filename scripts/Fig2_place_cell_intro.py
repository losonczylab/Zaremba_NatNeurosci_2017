"""Figure 2 - Place cell intro"""

FIG_FORMAT = 'svg'

import matplotlib as mpl
if FIG_FORMAT == 'svg':
    mpl.use('agg')
elif FIG_FORMAT == 'pdf':
    mpl.use('pdf')
elif FIG_FORMAT == 'interactive':
    mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn.apionly as sns

import lab
import lab.analysis.place_cell_analysis as place
import lab.plotting as plotting
import lab.misc as misc

import Df16a_analysis as df

# Colors
WT_color = df.WT_color
Df_color = df.Df_color
colors = (WT_color, Df_color)
linestyles = df.linestyles

# ROI filters
WT_filter = df.WT_filter
Df_filter = df.Df_filter
roi_filters = (WT_filter, Df_filter)

save_dir = df.fig_save_dir
filename = 'Fig2_place_cell_intro.{}'.format(FIG_FORMAT)


def prep_polar_ax(ax):
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    x_ticks = ax.get_xticks()
    x_tick_labels = [str(x / 2 / np.pi) for x in x_ticks]
    ax.set_xticklabels(x_tick_labels)
    labels = ax.get_xticklabels()
    labels[0].set_horizontalalignment('left')
    labels[1].set_horizontalalignment('center')
    labels[2].set_horizontalalignment('right')
    labels[3].set_horizontalalignment('center')

    labels[0].set_verticalalignment('center')
    labels[1].set_verticalalignment('bottom')
    labels[2].set_verticalalignment('center')
    labels[3].set_verticalalignment('top')


def main():
    all_grps = df.loadExptGrps('GOL')
    expts = lab.ExperimentSet(
        os.path.join(df.metadata_path, 'expt_metadata.xml'),
        behaviorDataPath=os.path.join(df.data_path, 'behavior'),
        dataPath=os.path.join(df.data_path, 'imaging'))

    WT_expt_grp = all_grps['WT_place_set']
    Df_expt_grp = all_grps['Df_place_set']
    expt_grps = [WT_expt_grp, Df_expt_grp]

    WT_label = WT_expt_grp.label()
    Df_label = Df_expt_grp.label()

    data_to_save = {}

    fig = plt.figure(figsize=(8.5, 11))

    gs2 = plt.GridSpec(48, 1, right=.6)
    wt_trace_ax = fig.add_subplot(gs2[12:16, 0])
    wt_position_ax = fig.add_subplot(gs2[16:18, 0])
    df_trace_ax = fig.add_subplot(gs2[18:22, 0])
    df_position_ax = fig.add_subplot(gs2[22:24, 0])

    gs3 = plt.GridSpec(2, 2, hspace=0.3, left=.62, bottom=0.5, top=0.7)
    wt_transients_ax = fig.add_subplot(gs3[0, 0], polar=True)
    df_transients_ax = fig.add_subplot(gs3[1, 0], polar=True)
    wt_vector_ax = fig.add_subplot(gs3[0, 1], polar=True)
    df_vector_ax = fig.add_subplot(gs3[1, 1], polar=True)

    gs4 = plt.GridSpec(1, 4, top=0.48, bottom=0.35, wspace=0.3)
    pf_fraction_ax = fig.add_subplot(gs4[0, 0])
    pf_per_cell_ax = fig.add_subplot(gs4[0, 1])
    pf_width_ax = fig.add_subplot(gs4[0, 2])
    circ_var_ax = fig.add_subplot(gs4[0, 3])

    pf_fraction_inset_ax = fig.add_axes([0.22, 0.36, 0.04, 0.06])
    pf_width_inset_ax = fig.add_axes([0.63, 0.41, 0.04, 0.06])
    circ_var_inset_ax = fig.add_axes([0.84, 0.36, 0.04, 0.06])

    #
    # PC Examples
    #

    wt_expt = expts.grabExpt('jz121', '2015-02-21-16h06m30s')
    df_expt = expts.grabExpt('jz098', '2014-11-08-16h14m02s')
    pc_expt_grp = place.pcExperimentGroup(
        [wt_expt, df_expt], imaging_label='soma')
    wt_id = '0422-0349'
    df_id = '0074-0339'
    wt_idx = wt_expt.roi_ids().index(wt_id)
    df_idx = df_expt.roi_ids().index(df_id)

    wt_imaging_data = wt_expt.imagingData(
        channel='Ch2', label='soma', dFOverF='from_file')
    df_imaging_data = df_expt.imagingData(
        channel='Ch2', label='soma', dFOverF='from_file')
    wt_transients = wt_expt.transientsData(
        threshold=95, channel='Ch2', label='soma')
    df_transients = df_expt.transientsData(
        threshold=95, channel='Ch2', label='soma')

    place.plotImagingData(
        roi_tSeries=wt_imaging_data[wt_idx, :, 0], ax=wt_trace_ax,
        roi_transients=wt_transients[wt_idx][0], position=None,
        imaging_interval=wt_expt.frame_period(), placeField=None,
        xlabel_visible=False, ylabel_visible=True, right_label=True,
        placeFieldColor=None, title='', rasterized=False, color='.4',
        transients_color=WT_color)
    sns.despine(ax=wt_trace_ax, top=True, left=False, bottom=True, right=True)
    wt_trace_ax.set_ylabel(WT_label, rotation='horizontal', ha='right')
    wt_trace_ax.tick_params(bottom=False, labelbottom=False)
    wt_trace_ax.tick_params(axis='y', direction='in', length=3, pad=3)
    wt_trace_ax.spines['left'].set_linewidth(1)
    wt_trace_ax.spines['left'].set_position(('outward', 5))

    place.plotImagingData(
        roi_tSeries=df_imaging_data[df_idx, :, 0], ax=df_trace_ax,
        roi_transients=df_transients[df_idx][0], position=None,
        imaging_interval=df_expt.frame_period(), placeField=None,
        xlabel_visible=False, ylabel_visible=True, right_label=True,
        placeFieldColor=None, title='', rasterized=False, color='.4',
        transients_color=Df_color)
    sns.despine(ax=df_trace_ax, top=True, left=False, bottom=True, right=True)
    df_trace_ax.set_ylabel(Df_label, rotation='horizontal', ha='right')
    df_trace_ax.tick_params(bottom=False, labelbottom=False)
    df_trace_ax.tick_params(axis='y', direction='in', length=3, pad=3)
    df_trace_ax.spines['left'].set_linewidth(1)
    df_trace_ax.spines['left'].set_position(('outward', 5))

    y_min = min(wt_trace_ax.get_ylim()[0], df_trace_ax.get_ylim()[0])
    y_max = max(wt_trace_ax.get_ylim()[1], df_trace_ax.get_ylim()[1])
    wt_trace_ax.set_ylim(y_min, y_max)
    df_trace_ax.set_ylim(y_min, y_max)
    wt_trace_ax.set_yticks([0, y_max])
    df_trace_ax.set_yticks([0, y_max])
    wt_trace_ax.set_yticklabels(['0', '{:0.1f}'.format(y_max)])
    df_trace_ax.set_yticklabels(['0', '{:0.1f}'.format(y_max)])
    wt_trace_ax.set_xlim(0, 600)
    df_trace_ax.set_xlim(0, 600)

    place.plotPosition(
        wt_expt.find('trial'), ax=wt_position_ax, rasterized=False,
        position_kwargs={'color': 'k'})
    sns.despine(
        ax=wt_position_ax, top=True, left=True, bottom=True, right=True)
    wt_position_ax.set_ylabel('')
    wt_position_ax.set_xlabel('')
    wt_position_ax.tick_params(
        left=False, bottom=False, top=False, right=False, labelleft=False,
        labelbottom=False, labelright=False, labeltop=False)
    plotting.add_scalebar(
        wt_position_ax, matchx=False, matchy=False, hidex=False, hidey=False,
        sizex=60, labelx='1 min', bar_thickness=.02, pad=0, loc=4)
    wt_position_ax.set_yticks([0, 1])
    wt_position_ax.set_ylim(0, 1)

    place.plotPosition(
        df_expt.find('trial'), ax=df_position_ax, rasterized=False,
        position_kwargs={'color': 'k'})
    sns.despine(ax=df_position_ax, top=True, bottom=True, right=True)
    df_position_ax.tick_params(
        bottom=False, top=False, right=False, labelbottom=False,
        labelright=False, labeltop=False, direction='in', length=3, pad=3)
    df_position_ax.spines['left'].set_linewidth(1)
    df_position_ax.spines['left'].set_position(('outward', 5))
    df_position_ax.set_ylabel('Position')
    df_position_ax.set_xlabel('')
    df_position_ax.set_yticks([0, 1])
    df_position_ax.set_ylim(0, 1)

    trans_kwargs = {
        'color': WT_color, 'marker': 'o', 'linestyle': 'None', 'markersize': 3}
    wt_pf = [pc_expt_grp.pfs_n()[wt_expt][wt_idx]]
    place.plotPosition(
        wt_expt.find('trial'), ax=wt_transients_ax, polar=True,
        placeFields=wt_pf, placeFieldColors=[WT_color],
        trans_roi_filter=lambda roi: roi.id == wt_id,
        rasterized=False, running_trans_only=True, demixed=False,
        position_kwargs={'color': '0.5'}, trans_kwargs=trans_kwargs)
    wt_transients_ax.set_xlabel('')
    wt_transients_ax.set_ylabel('')
    wt_transients_ax.set_rticks([])
    prep_polar_ax(wt_transients_ax)

    trans_kwargs['color'] = Df_color
    df_pf = [pc_expt_grp.pfs_n()[df_expt][df_idx]]
    place.plotPosition(
        df_expt.find('trial'), ax=df_transients_ax, polar=True,
        placeFields=df_pf, placeFieldColors=[Df_color],
        trans_roi_filter=lambda roi: roi.id == df_id,
        rasterized=False, running_trans_only=True, demixed=False,
        position_kwargs={'color': '0.5'}, trans_kwargs=trans_kwargs)
    df_transients_ax.set_xlabel('')
    df_transients_ax.set_ylabel('')
    df_transients_ax.set_rticks([])
    prep_polar_ax(df_transients_ax)

    place.plotTransientVectors(
        place.pcExperimentGroup([wt_expt], imaging_label='soma'), wt_idx,
        wt_vector_ax, mean_zorder=99, color=WT_color, mean_color='g')
    place.plotTransientVectors(
        place.pcExperimentGroup([df_expt], imaging_label='soma'), df_idx,
        df_vector_ax, mean_zorder=99, color=Df_color, mean_color='g')

    #
    # Stats
    #

    groupby = [['expt'], ['mouseID']]

    data_to_save['pc_percentage'] = plotting.plot_metric(
        pf_fraction_ax, expt_grps, metric_fn=place.place_cell_percentage,
        groupby=None, plotby=None, colorby=None, plot_method='cdf',
        roi_filters=roi_filters, activity_kwargs=None, colors=colors,
        activity_label='Place cell fraction', rotate_labels=False,
        return_full_dataframes=True, linestyles=linestyles)
    # pf_fraction_ax.get_legend().set_visible(False)
    pf_fraction_ax.legend(loc='upper left', fontsize=6)
    pf_fraction_ax.set_title('')
    pf_fraction_ax.set_ylabel('Cumulative fraction')
    pf_fraction_ax.set_xticks([0, .2, .4, .6, .8])
    pf_fraction_ax.set_xlim(0, .8)
    pf_fraction_ax.spines['left'].set_linewidth(1)
    pf_fraction_ax.spines['bottom'].set_linewidth(1)
    # plotting.stackedText(
    #     pf_fraction_ax, [WT_label, Df_label], colors=colors, loc=2, size=8)

    data_to_save['pc_percentage_inset'] = plotting.plot_metric(
        pf_fraction_inset_ax, expt_grps, metric_fn=place.place_cell_percentage,
        groupby=groupby,   # [['mouseID']],
        plotby=None, colorby=None, plot_method='swarm',
        roi_filters=roi_filters, activity_kwargs=None, colors=colors,
        activity_label='Place cell fraction', rotate_labels=False,
        linewidth=0.2, edgecolor='gray')
    pf_fraction_inset_ax.set_title('')
    pf_fraction_inset_ax.set_ylabel('')
    pf_fraction_inset_ax.set_xlabel('')
    pf_fraction_inset_ax.set_yticks([0, 0.5])
    pf_fraction_inset_ax.set_ylim([0, 0.5])
    pf_fraction_inset_ax.get_legend().set_visible(False)
    sns.despine(ax=pf_fraction_inset_ax)
    pf_fraction_inset_ax.tick_params(bottom=False, labelbottom=False)
    pf_fraction_inset_ax.spines['left'].set_linewidth(1)
    pf_fraction_inset_ax.spines['bottom'].set_linewidth(1)
    pf_fraction_inset_ax.set_xlim(-0.6, 0.6)

    n_pf_kwargs = {'per_mouse_fractions': True, 'max_n_place_fields': 3}
    data_to_save['n_place_fields'] = plotting.plot_metric(
        pf_per_cell_ax, expt_grps, metric_fn=place.n_place_fields,
        groupby=None, plotby=['number'], plot_method='swarm',
        roi_filters=roi_filters, activity_kwargs=n_pf_kwargs, colors=colors,
        activity_label='Fraction of place cells', rotate_labels=False,
        plot_bar=True, edgecolor='k', linewidth=0.5, size=3)
    sns.despine(ax=pf_per_cell_ax)
    pf_per_cell_ax.set_title('')
    pf_per_cell_ax.set_xlabel('Place fields per cell')
    pf_per_cell_ax.set_ylabel('Fraction of place cells')
    pf_per_cell_ax.set_xticklabels(['1', '2', '3+'])
    pf_per_cell_ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    data_to_save['width'] = plotting.plot_metric(
        pf_width_ax, expt_grps, metric_fn=place.place_field_width,
        groupby=[['roi_id', 'expt']], plotby=None, plot_method='hist',
        roi_filters=roi_filters, activity_kwargs=None,
        activity_label='Place field width (cm)', normed=True, plot_mean=True,
        bins=20, range=(0, 120), colors=colors, rotate_labels=False,
        filled=False, mean_kwargs={'ls': ':'}, return_full_dataframes=True,
        linestyles=linestyles)
    pf_width_ax.set_title('')
    # pf_width_ax.get_legend().set_visible(False)
    pf_width_ax.legend(loc='lower right', fontsize=6)
    pf_width_ax.set_xticks([0, 40, 80, 120])
    pf_width_ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
    pf_width_ax.set_ylim(0, 0.08)
    pf_width_ax.set_ylabel('Normalized density')
    pf_width_ax.spines['left'].set_linewidth(1)
    pf_width_ax.spines['bottom'].set_linewidth(1)

    data_to_save['width_inset'] = plotting.plot_metric(
        pf_width_inset_ax, expt_grps, metric_fn=place.place_field_width,
        groupby=[['roi_id', 'expt'], ['expt'], ['mouseID']],
        plotby=None, plot_method='swarm',
        roi_filters=roi_filters, activity_kwargs=None,
        activity_label='Place field width (cm)', colors=colors,
        rotate_labels=False, linewidth=0.2, edgecolor='gray')
    pf_width_inset_ax.set_title('')
    pf_width_inset_ax.set_ylabel('')
    pf_width_inset_ax.set_xlabel('')
    pf_width_inset_ax.get_legend().set_visible(False)
    sns.despine(ax=pf_width_inset_ax)
    pf_width_inset_ax.tick_params(bottom=False, labelbottom=False)
    pf_width_inset_ax.set_ylim(25, 40)
    pf_width_inset_ax.set_yticks([25, 40])
    pf_width_inset_ax.spines['left'].set_linewidth(1)
    pf_width_inset_ax.spines['bottom'].set_linewidth(1)
    pf_width_inset_ax.set_xlim(-0.6, 0.6)

    data_to_save['circular_variance'] = plotting.plot_metric(
        circ_var_ax, expt_grps, metric_fn=place.circular_variance,
        groupby=[['roi_id', 'expt']], plotby=None, plot_method='cdf',
        roi_filters=roi_filters, activity_kwargs=None,
        activity_label='Circular variance', colors=colors, rotate_labels=False,
        return_full_dataframes=True, linestyles=linestyles)
    circ_var_ax.set_title('')
    # circ_var_ax.get_legend().set_visible(False)
    circ_var_ax.legend(loc='upper left', fontsize=6)
    circ_var_ax.set_ylabel('Cumulative fraction')
    circ_var_ax.set_xlim(-0.1, 1)
    circ_var_ax.set_xticks([0, 0.5, 1])
    circ_var_ax.spines['left'].set_linewidth(1)
    circ_var_ax.spines['bottom'].set_linewidth(1)

    data_to_save['circular_variance_inset'] = plotting.plot_metric(
        circ_var_inset_ax, expt_grps, metric_fn=place.circular_variance,
        groupby=groupby,
        plotby=None, plot_method='swarm',
        roi_filters=roi_filters, activity_kwargs=None,
        activity_label='Circular variance', colors=colors, rotate_labels=False,
        linewidth=0.2, edgecolor='gray')
    circ_var_inset_ax.set_title('')
    circ_var_inset_ax.set_ylabel('')
    circ_var_inset_ax.set_xlabel('')
    circ_var_inset_ax.get_legend().set_visible(False)
    sns.despine(ax=circ_var_inset_ax)
    circ_var_inset_ax.tick_params(bottom=False, labelbottom=False)
    circ_var_inset_ax.set_ylim(0, 0.6)
    circ_var_inset_ax.set_yticks([0, 0.6])
    circ_var_inset_ax.spines['left'].set_linewidth(1)
    circ_var_inset_ax.spines['bottom'].set_linewidth(1)
    circ_var_inset_ax.set_xlim(-0.6, 0.6)

    misc.save_figure(fig, filename, save_dir=save_dir)

    plt.close('all')

if __name__ == '__main__':
    main()
