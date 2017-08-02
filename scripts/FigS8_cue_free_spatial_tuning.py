"""Figure S8 - Cue-free spatial tuning"""

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
import pandas as pd
import os
import seaborn.apionly as sns

import lab
import lab.analysis.place_cell_analysis as place
import lab.plotting as plotting
from lab.misc import complex_to_norm, save_figure

import Df16a_analysis as df

# Colors
WT_color = df.WT_color
Df_color = df.Df_color
colors = (WT_color, Df_color)

# ROI filters
WT_filter = df.WT_filter
Df_filter = df.Df_filter
roi_filters = (WT_filter, Df_filter)

markers = df.markers
WT_marker, Df_marker = markers

save_dir = df.fig_save_dir
filename = 'FigS8_cue_free_spatial_tuning.{}'.format(FIG_FORMAT)

cue_free_json = os.path.join(df.expt_sets_path, 'cue_free_expts.json')


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
    all_expt_grps = df.loadExptGrps('GOL')

    WT_expt_grp_hidden = all_expt_grps['WT_place_set']
    Df_expt_grp_hidden = all_expt_grps['Df_place_set']
    expt_grps = [WT_expt_grp_hidden, Df_expt_grp_hidden]

    paired_grps = [grp.pair(
        'consecutive groups', groupby=['condition_day']) for grp in expt_grps]

    fig = plt.figure(figsize=(8.5, 11))

    gs1 = plt.GridSpec(
        3, 8, top=0.9, bottom=0.7, left=0.1, right=0.9, wspace=0.4)
    ax2 = fig.add_subplot(gs1[:, :2])
    pf_fraction_ax = fig.add_subplot(gs1[:, 2:4])
    circ_var_ax = fig.add_subplot(gs1[:, 4:6])

    trans_ax1 = fig.add_subplot(gs1[0, 6], polar=True)
    trans_ax2 = fig.add_subplot(gs1[0, 7], polar=True)
    trans_ax3 = fig.add_subplot(gs1[1, 6], polar=True)
    trans_ax4 = fig.add_subplot(gs1[1, 7], polar=True)
    trans_ax5 = fig.add_subplot(gs1[2, 6], polar=True)
    trans_ax6 = fig.add_subplot(gs1[2, 7], polar=True)
    trans_axs = [
        trans_ax1, trans_ax2, trans_ax3, trans_ax4, trans_ax5, trans_ax6]

    #
    # Stability by distance to fabric transitions
    #

    filter_fn = lambda df: df['second_condition_day_session'] == 'B_0_0'
    filter_columns = ['second_condition_day_session']

    label_order = ['before', 'middle', 'after']

    data_to_plot = [[], [], []]
    all_data, shuffles = [], []
    for expt_grp, roi_filter in zip(paired_grps, roi_filters):

        fabric_map = {expt: expt.belt().fabric_transitions(
            units='normalized') for expt in expt_grp}

        def norm_diff(n1, n2):
            d = n1 - n2
            d = d + 1.0 if d < -0.5 else d
            d = d - 1.0 if d >= 0.5 else d
            return d

        def closest_transition(row):
            expt = row['first_expt']
            centroid = complex_to_norm(row['first_centroid'])
            positions = fabric_map[expt]['position']
            distances = [norm_diff(centroid, t) for t in positions]
            row['closest'] = distances[np.argmin(np.abs(distances))]
            return row

        data, shuffle = place.activity_centroid_shift(
            expt_grp, roi_filter=roi_filter, activity_filter='active_both',
            circ_var_pcs=False, units='norm', shuffle=True)

        plotting.prepare_dataframe(data, include_columns=filter_columns)
        data = data[filter_fn(data)]

        plotting.prepare_dataframe(shuffle, include_columns=filter_columns)

        data = data.apply(closest_transition, axis=1)
        shuffle = shuffle.apply(closest_transition, axis=1)

        def categorize(row):
            if row['closest'] < 0 and -1 / 9. < row['closest']:
                row['category'] = 'before'
            elif row['closest'] > 0 and 1 / 9. > row['closest']:
                row['category'] = 'after'
            else:
                row['category'] = 'middle'
            return row

        data = data.apply(categorize, axis=1)
        shuffle = shuffle.apply(categorize, axis=1)

        groupby = [
            ['second_condition_day_session', 'second_mouse', 'category']]

        for gb in groupby:
            plotting.prepare_dataframe(data, include_columns=gb)
            plotting.prepare_dataframe(shuffle, include_columns=gb)
            data = data.groupby(gb, as_index=False).mean()
            shuffle = shuffle.groupby(gb, as_index=False).mean()

        for category, group in data.groupby(['category']):
            idx = label_order.index(category)
            data_to_plot[idx].append(group['value'])

        shuffles.append(shuffle)
        all_data.append(data)

    shuffle_df = pd.concat(shuffles, ignore_index=True)
    for category, group in shuffle_df.groupby(['category']):
        idx = label_order.index(category)
        data_to_plot[idx].append(group['value'])

    plotting.grouped_bar(
        ax2, data_to_plot, condition_labels=label_order,
        cluster_labels=df.labels + ('shuffle',),
        bar_colors=sns.color_palette('deep')[3:], scatter_points=False,
        scatterbar_colors=None, jitter_x=False, loc='best', error_bars='sem')
    sns.despine(ax=ax2)
    ax2.set_yticks([0, 0.1, 0.2, 0.3])
    ax2.set_ylabel('Centroid shift (fraction of belt)')

    #
    # Burlap belt
    #

    expts = lab.ExperimentSet(
        os.path.join(df.metadata_path, 'expt_metadata.xml'),
        behaviorDataPath=os.path.join(df.data_path, 'behavior'),
        dataPath=os.path.join(df.data_path, 'imaging'))

    burlap_expt_grp = lab.classes.pcExperimentGroup.from_json(
        cue_free_json, expts, imaging_label=df.IMAGING_LABEL, label='cue-free')

    acute_grps = df.loadExptGrps('RF')

    WT_expt_grp_acute = acute_grps['WT_place_set'].unpair()
    WT_expt_grp_acute.label('cue-rich')

    burlap_colors = ('k', '0.9')
    example_expt = expts.grabExptByPath('/jz128/TSeries-07262015-burlap-000')
    cv = place.circular_variance_p(burlap_expt_grp)
    cv = cv[cv['expt'] == example_expt]
    cv = cv.sort_values(by=['value'])

    trans_kwargs = {
        'color': '0.9', 'marker': 'o', 'linestyle': 'None',
        'markersize': 3}
    for ax, (idx, row) in zip(trans_axs, cv.iloc[:6].iterrows()):
        expt = row['expt']
        roi_idx = expt.rois().index(row['roi'])
        pf = None
        place.plotPosition(
            expt.find('trial'), ax=ax, polar=True,
            placeFields=pf, placeFieldColors=['0.9'],
            trans_roi_filter=lambda roi: roi.id == expt.roi_ids()[roi_idx],
            rasterized=False, running_trans_only=True, demixed=False,
            position_kwargs={'color': '0.5'}, trans_kwargs=trans_kwargs)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_rticks([])
        prep_polar_ax(ax)
    for ax in trans_axs[:-1]:
        ax.set_xticklabels(['', '', '', ''])

    activity_kwargs = {'circ_var': True}

    plotting.plot_metric(
        pf_fraction_ax, [WT_expt_grp_acute, burlap_expt_grp],
        metric_fn=place.place_cell_percentage,
        groupby=None, plotby=None, colorby=None, plot_method='swarm',
        roi_filters=[WT_filter, WT_filter], activity_kwargs=activity_kwargs,
        colors=burlap_colors, activity_label='Place cell fraction',
        rotate_labels=False, plot_bar=True)
    pf_fraction_ax.set_title('')
    pf_fraction_ax.set_xlabel('')
    sns.despine(ax=pf_fraction_ax)
    pf_fraction_ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])

    plotting.plot_metric(
        circ_var_ax, [WT_expt_grp_acute, burlap_expt_grp],
        metric_fn=place.circular_variance,
        groupby=[['roi_id', 'expt']], plotby=None, plot_method='cdf',
        roi_filters=[WT_filter, WT_filter], activity_kwargs=None,
        activity_label='Circular variance', colors=burlap_colors,
        rotate_labels=False)
    circ_var_ax.set_title('')
    circ_var_ax.get_legend().set_visible(False)
    circ_var_ax.set_xticks([0, 0.5, 1])

    save_figure(fig, filename, save_dir=save_dir)

    plt.close('all')

if __name__ == '__main__':
    main()
