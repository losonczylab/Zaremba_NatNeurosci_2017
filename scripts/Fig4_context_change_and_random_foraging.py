"""Figure 4 - Context change and random foraging"""

FIG_FORMAT = 'svg'

import matplotlib as mpl
if FIG_FORMAT == 'svg':
    mpl.use('agg')
elif FIG_FORMAT == 'pdf':
    mpl.use('pdf')
elif FIG_FORMAT == 'interactive':
    mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import numpy as np
import pandas as pd
import math
from copy import copy

import lab.analysis.place_cell_analysis as place
from lab import plotting, misc

import Df16a_analysis as df

# Colors
WT_color = df.WT_color
Df_color = df.Df_color
colors = (WT_color, Df_color)

markers = df.markers
linestyles = df.linestyles

# ROI filters
WT_filter = df.WT_filter
Df_filter = df.Df_filter
roi_filters = (WT_filter, Df_filter)

save_dir = df.fig_save_dir
filename = 'Fig4_context_change_and_random_foraging.{}'.format(FIG_FORMAT)


def main():
    hidden_grps = df.loadExptGrps('GOL')

    WT_expt_grp_hidden = hidden_grps['WT_place_set']
    Df_expt_grp_hidden = hidden_grps['Df_place_set']
    expt_grps_hidden = [WT_expt_grp_hidden, Df_expt_grp_hidden]

    acute_grps = df.loadExptGrps('RF')

    WT_expt_grp_acute = acute_grps['WT_place_set'].unpair()
    Df_expt_grp_acute = acute_grps['Df_place_set'].unpair()
    expt_grps_acute = [WT_expt_grp_acute, Df_expt_grp_acute]

    WT_label = WT_expt_grp_hidden.label()
    Df_label = Df_expt_grp_hidden.label()
    labels = [WT_label, Df_label]

    fig = plt.figure(figsize=(8.5, 11))

    gs1 = plt.GridSpec(1, 1, top=0.9, bottom=0.7, left=0.1, right=0.20)
    across_ctx_ax = fig.add_subplot(gs1[0, 0])

    gs2 = plt.GridSpec(3, 1, top=0.9, bottom=0.7, left=0.25, right=0.35)
    wt_pie_ax = fig.add_subplot(gs2[0, 0])
    df_pie_ax = fig.add_subplot(gs2[1, 0])
    shuffle_pie_ax = fig.add_subplot(gs2[2, 0])
    pie_axs = (wt_pie_ax, df_pie_ax, shuffle_pie_ax)

    gs3 = plt.GridSpec(1, 1, top=0.9, bottom=0.7, left=0.4, right=0.5)
    cue_cell_bar_ax = fig.add_subplot(gs3[0, 0])

    gs5 = plt.GridSpec(1, 1, top=0.5, bottom=0.3, left=0.1, right=0.3)
    acute_stability_ax = fig.add_subplot(gs5[0, 0])

    acute_stability_inset_ax = fig.add_axes([0.23, 0.32, 0.05, 0.08])

    gs6 = plt.GridSpec(1, 1, top=0.5, bottom=0.3, left=0.4, right=0.5)
    task_compare_ax = fig.add_subplot(gs6[0, 0])

    #
    # RF Compare
    #
    params = {}
    params['filename'] = filename

    params_cent_shift_pc = {}
    params_cent_shift_pc['stability_fn'] = place.activity_centroid_shift
    params_cent_shift_pc['stability_kwargs'] = {
        'activity_filter': 'pc_both', 'circ_var_pcs': False, 'units': 'norm',
        'shuffle': True}
    params_cent_shift_pc['stability_label'] = \
        'Centroid shift (fraction of belt)'

    params_cent_shift_all = {}
    params_cent_shift_all['stability_fn'] = place.activity_centroid_shift
    params_cent_shift_all['stability_kwargs'] = {
        'activity_filter': 'active_both', 'circ_var_pcs': False,
        'units': 'norm', 'shuffle': True}
    params_cent_shift_all['stability_label'] = \
        'Centroid shift (fraction of belt)'
    params_cent_shift_all['stability_inset_ylim'] = (0.15, 0.30)
    params_cent_shift_all['stability_cdf_range'] = (0.15, 0.35)
    params_cent_shift_all['stability_cdf_ticks'] = \
        (0.15, 0.20, 0.25, 0.30, 0.35)
    params_cent_shift_all['stability_compare_ylim'] = (0.15, 0.27)
    params_cent_shift_all['stability_compare_yticks'] = (0.15, 0.20, 0.25)
    params_cent_shift_all['ctx_compare_ylim'] = (0.10, 0.30)
    params_cent_shift_all['ctx_compare_yticks'] = \
        (0.10, 0.15, 0.20, 0.25, 0.30)

    params_cent_shift_cm = {}
    params_cent_shift_cm['stability_fn'] = place.activity_centroid_shift
    params_cent_shift_cm['stability_kwargs'] = {
        'activity_filter': 'active_both', 'circ_var_pcs': False, 'units': 'cm',
        'shuffle': True}
    params_cent_shift_cm['stability_label'] = 'Centroid shift (cm)'

    params_pop_vect_corr = {}
    params_pop_vect_corr['stability_fn'] = place.population_vector_correlation
    params_pop_vect_corr['stability_kwargs'] = {
        'method': 'corr', 'activity_filter': 'pc_both', 'min_pf_density': 0.05,
        'circ_var_pcs': False}
    params_pop_vect_corr['stability_label'] = 'Population vector correlation'

    params_pf_corr = {}
    params_pf_corr['stability_fn'] = place.place_field_correlation
    params_pf_corr['stability_kwargs'] = {'activity_filter': 'pc_either'}
    params_pf_corr['stability_label'] = 'Place field correlation'
    params_pf_corr['stability_inset_ylim'] = (0, 0.50)
    params_pf_corr['stability_cdf_range'] = (0.15, 0.55)
    params_pf_corr['stability_cdf_ticks'] = (0.15, 0.25, 0.35, 0.45, 0.55)
    params_pf_corr['stability_compare_ylim'] = (0.22, 0.40)
    params_pf_corr['stability_compare_yticks'] = (0.25, 0.30, 0.35, 0.40)
    params_pf_corr['hidden_ctx_compare_ylim'] = (0.22, 0.40)
    params_pf_corr['hidden_ctx_compare_yticks'] = (0.25, 0.30, 0.35, 0.40)
    params_pf_corr['ctx_compare_ylim'] = (0.22, 0.40)
    params_pf_corr['ctx_compare_yticks'] = (0.25, 0.30, 0.35, 0.40)

    params.update(params_cent_shift_all)

    day_paired_grps_acute = [
        grp.pair('consecutive groups', groupby=['day_in_df'])
        for grp in expt_grps_acute]
    paired_grps_acute = day_paired_grps_acute
    paired_grps_hidden = [grp.pair(
        'consecutive groups',
        groupby=['X_condition', 'X_day']) for grp in expt_grps_hidden]

    filter_fn = lambda df: (df['expt_pair_label'] == 'SameAll')
    filter_columns = ['expt_pair_label']

    acute_stability = plotting.plot_metric(
        acute_stability_ax, paired_grps_acute,
        metric_fn=params['stability_fn'],
        groupby=[['expt_pair_label', 'second_expt']], plotby=None,
        plot_method='cdf', plot_abs=True, roi_filters=roi_filters,
        activity_kwargs=params['stability_kwargs'], plot_shuffle=True,
        shuffle_plotby=False, pool_shuffle=True,
        activity_label=params['stability_label'], colors=colors,
        rotate_labels=False, filter_fn=filter_fn,
        filter_columns=filter_columns, return_full_dataframes=False,
        linestyles=linestyles)
    acute_stability_ax.set_xlabel(params['stability_label'])
    acute_stability_ax.set_title('')
    sns.despine(ax=acute_stability_ax)
    acute_stability_ax.set_xlim(params['stability_cdf_range'])
    acute_stability_ax.set_xticks(params['stability_cdf_ticks'])
    acute_stability_ax.legend(loc='upper left', fontsize=6)

    plotting.plot_metric(
        acute_stability_inset_ax, paired_grps_acute,
        metric_fn=params['stability_fn'],
        groupby=[['second_expt'], ['second_mouseID']], plotby=None,
        plot_method='swarm', plot_abs=True, roi_filters=roi_filters,
        activity_kwargs=params['stability_kwargs'], plot_shuffle=True,
        shuffle_plotby=False, pool_shuffle=True,
        activity_label=params['stability_label'], colors=colors,
        rotate_labels=False, filter_fn=filter_fn,
        filter_columns=filter_columns, linewidth=0.2, edgecolor='gray',
        plot_shuffle_as_hline=True)
    acute_stability_inset_ax.get_legend().set_visible(False)
    sns.despine(ax=acute_stability_inset_ax)
    acute_stability_inset_ax.set_title('')
    acute_stability_inset_ax.set_ylabel('')
    acute_stability_inset_ax.set_xlabel('')
    acute_stability_inset_ax.tick_params(bottom=False, labelbottom=False)
    acute_stability_inset_ax.set_ylim(params['stability_inset_ylim'])
    acute_stability_inset_ax.set_yticks(params['stability_inset_ylim'])

    tmp_fig = plt.figure()
    tmp_ax = tmp_fig.add_subplot(111)
    hidden_stability = plotting.plot_metric(
        tmp_ax, paired_grps_hidden, metric_fn=params['stability_fn'],
        groupby=[['expt_pair_label', 'second_expt']],
        plotby=('expt_pair_label', ), plot_method='line', plot_abs=True,
        roi_filters=roi_filters, activity_kwargs=params['stability_kwargs'],
        plot_shuffle=True, shuffle_plotby=False, pool_shuffle=True,
        activity_label=params['stability_label'], colors=colors,
        rotate_labels=False, filter_fn=filter_fn,
        filter_columns=filter_columns, return_full_dataframes=False)
    plt.close(tmp_fig)

    wt_acute = acute_stability[WT_label]['dataframe']
    wt_acute_shuffle = acute_stability[WT_label]['shuffle']
    df_acute = acute_stability[Df_label]['dataframe']
    df_acute_shuffle = acute_stability[Df_label]['shuffle']

    wt_hidden = hidden_stability[WT_label]['dataframe']
    wt_hidden_shuffle = hidden_stability[WT_label]['shuffle']
    df_hidden = hidden_stability[Df_label]['dataframe']
    df_hidden_shuffle = hidden_stability[Df_label]['shuffle']

    for dataframe in (wt_acute, wt_acute_shuffle, df_acute, df_acute_shuffle):
        dataframe['task'] = 'RF'

    for dataframe in (wt_hidden, wt_hidden_shuffle,
                      df_hidden, df_hidden_shuffle):
        dataframe['task'] = 'GOL'

    WT_data = wt_acute.append(wt_hidden, ignore_index=True)
    Df_data = df_acute.append(df_hidden, ignore_index=True)

    WT_shuffle = wt_acute_shuffle.append(wt_hidden_shuffle, ignore_index=True)
    Df_shuffle = df_acute_shuffle.append(df_hidden_shuffle, ignore_index=True)

    filter_columns = ('expt_pair_label', )
    filter_fn = lambda df: (df['expt_pair_label'] == 'SameAll')

    order_dict = {'RF': 0, 'GOL': 1}
    WT_data['order'] = WT_data['task'].map(order_dict)
    Df_data['order'] = Df_data['task'].map(order_dict)
    line_kwargs = {'markersize': 4}
    plotting.plot_dataframe(
        task_compare_ax, [WT_data, Df_data], [WT_shuffle, Df_shuffle],
        labels=labels, activity_label='', groupby=[['task', 'second_mouseID']],
        plotby=('task',), plot_method='box_and_line', colors=colors,
        filter_fn=filter_fn, filter_columns=filter_columns, plot_shuffle=True,
        shuffle_plotby=False, pool_shuffle=True, orderby='order', notch=False,
        plot_shuffle_as_hline=True, markers=markers, linestyles=linestyles,
        line_kwargs=line_kwargs,
        flierprops={'markersize': 3, 'marker': 'o'}, whis='range')
    task_compare_ax.set_title('')
    sns.despine(ax=task_compare_ax)
    task_compare_ax.set_ylim(params['stability_compare_ylim'])
    task_compare_ax.set_yticks(params['stability_compare_yticks'])
    task_compare_ax.set_xlabel('')
    task_compare_ax.set_ylabel(params['stability_label'])
    task_compare_ax.legend(loc='upper right', fontsize=6)

    #
    # Stability across transition
    #
    groupby = [['second_expt'], ['second_mouse']]
    filter_fn = lambda df: (df['X_first_condition'] == 'A') \
        & (df['X_second_condition'] == 'B')
    filter_columns = ('X_first_condition', 'X_second_condition')
    plotting.plot_metric(
        across_ctx_ax, paired_grps_hidden, metric_fn=params['stability_fn'],
        groupby=groupby, plotby=None, plot_method='swarm',
        activity_kwargs=params['stability_kwargs'], plot_shuffle=True,
        shuffle_plotby=False, pool_shuffle=True, colors=colors,
        activity_label=params['stability_label'], rotate_labels=False,
        filter_fn=filter_fn, filter_columns=filter_columns,
        plot_shuffle_as_hline=True, return_full_dataframes=False,
        plot_bar=True, roi_filters=roi_filters)
    sns.despine(ax=across_ctx_ax)
    across_ctx_ax.set_ylim(0.0, 0.3)
    across_ctx_ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    across_ctx_ax.set_xticklabels([])
    across_ctx_ax.set_xlabel('')
    across_ctx_ax.set_title('')
    across_ctx_ax.get_legend().set_visible(False)
    plotting.stackedText(
        across_ctx_ax, labels, colors=colors, loc=2, size=10)

    #
    # Cue remapping
    #

    THRESHOLD = 0.05 * 2 * np.pi
    CUENESS_THRESHOLD = 0.33

    def first_cue_position(row):
        expt = row['first_expt']
        cue = row['cue']
        cues = expt.belt().cues(normalized=True)
        first_cue = cues.ix[cues['cue'] == cue]
        pos = (first_cue['start'] + first_cue['stop']) / 2
        angle = pos * 2 * np.pi
        return np.complex(np.cos(angle), np.sin(angle))

    def dotproduct(v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))

    def length(v):
        return math.sqrt(dotproduct(v, v))

    def angle(v1, v2):
        return math.acos(
            np.round(dotproduct(v1, v2) / (length(v1) * length(v2)), 3))

    def distance_to_first_cue(row):
        centroid = row['second_centroid']
        pos = row['first_cue_position']
        return angle((pos.real, pos.imag), (centroid.real, centroid.imag))

    WT_copy = copy(WT_expt_grp_hidden)
    WT_copy.filterby(
        lambda df: ~df['X_condition'].str.contains('C'), ['X_condition'])
    WT_paired = WT_copy.pair(
        'consecutive groups', groupby=['X_condition', 'X_day']).pair(
        'consecutive groups', groupby=['X_condition'])

    Df_copy = copy(Df_expt_grp_hidden)
    Df_copy.filterby(
        lambda df: ~df['X_condition'].str.contains('C'), ['X_condition'])
    Df_paired = Df_copy.pair(
        'consecutive groups', groupby=['X_condition', 'X_day']).pair(
        'consecutive groups', groupby=['X_condition'])

    WT_df, WT_shuffle_df = place.cue_cell_remapping(
        WT_paired, roi_filter=WT_filter, near_threshold=THRESHOLD,
        activity_filter='active_both', circ_var_pcs=False, shuffle=True)
    Df_df, Df_shuffle_df = place.cue_cell_remapping(
        Df_paired, roi_filter=Df_filter, near_threshold=THRESHOLD,
        activity_filter='active_both', circ_var_pcs=False, shuffle=True)

    shuffle_df = pd.concat([WT_shuffle_df, Df_shuffle_df], ignore_index=True)

    cueness, cueness_fraction = [], []
    cue_n, place_n, neither_n = [], [], []

    for grp_df in (WT_df, Df_df, shuffle_df):

        grp_df['first_cue_position'] = grp_df.apply(first_cue_position, axis=1)

        grp_df['second_distance_to_first_cue_position'] = grp_df.apply(
            distance_to_first_cue, axis=1)

        grp_df['cueness'] = grp_df['second_distance_to_first_cue_position'] / \
            (grp_df['value'] + grp_df['second_distance_to_first_cue_position'])

        plotting.prepare_dataframe(grp_df, ['first_mouse'])
        cueness_fraction.append([[]])
        cue_n.append([])
        place_n.append([])
        neither_n.append([])

        for mouse, mouse_df in grp_df.groupby('first_mouse'):
            cue_n[-1].append(
                (mouse_df['cueness'] > (1 - CUENESS_THRESHOLD)).sum())
            place_n[-1].append(
                (mouse_df['cueness'] < CUENESS_THRESHOLD).sum())
            neither_n[-1].append(
                mouse_df.shape[0] - cue_n[-1][-1] - place_n[-1][-1])
            cueness_fraction[-1][0].append(
                cue_n[-1][-1] / float(place_n[-1][-1]))
        cueness.append([grp_df['cueness']])

    cue_labels = labels + ['shuffle']

    plotting.swarm_plot(
        cue_cell_bar_ax, cueness_fraction[:2], condition_labels=labels,
        colors=colors, plot_bar=True)
    cue_cell_bar_ax.axhline(
        np.mean(cueness_fraction[-1][0]), ls='--', color='k')
    sns.despine(ax=cue_cell_bar_ax)
    cue_cell_bar_ax.set_ylim(0, 1.5)
    cue_cell_bar_ax.set_yticks([0, 0.5, 1.0, 1.5])
    cue_cell_bar_ax.set_xticklabels([])
    cue_cell_bar_ax.set_xlabel('')
    cue_cell_bar_ax.set_ylabel('Cue-to-position ratio')
    cue_cell_bar_ax.get_legend().set_visible(False)
    plotting.stackedText(
        cue_cell_bar_ax, labels, colors=colors, loc=2, size=10)

    WT_colors = sns.light_palette(WT_color, 7)[:-6:-2]
    Df_colors = sns.light_palette(Df_color, 7)[:-6:-2]
    shuffle_colors = sns.light_palette('k', 7)[:-6:-2]
    pie_colors = (WT_colors, Df_colors, shuffle_colors)
    pie_labels = ['cue', 'position', 'neither']
    orig_size = mpl.rcParams.get('xtick.labelsize')
    mpl.rcParams['xtick.labelsize'] = 5
    for grp_ax, grp_label, grp_cue_n, grp_place_n, grp_neither_n, p_cs in zip(
            pie_axs, cue_labels, cue_n, place_n, neither_n, pie_colors):
        grp_ax.pie(
            [sum(grp_cue_n), sum(grp_place_n), sum(grp_neither_n)],
            autopct='%1.0f%%', shadow=False, frame=False,
            labels=pie_labels, colors=p_cs, textprops={'fontsize': 5})
        grp_ax.set_title(grp_label)
        plotting.square_axis(grp_ax)
    mpl.rcParams['xtick.labelsize'] = orig_size

    misc.save_figure(fig, params['filename'], save_dir=save_dir)

    plt.close('all')

if __name__ == '__main__':
    main()
