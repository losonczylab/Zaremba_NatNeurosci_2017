"""Figure 5 - Goal zone enrichment by place cells"""

FIG_FORMAT = 'svg'
circ_var_pcs = False
MALES_ONLY = False

import matplotlib as mpl
if FIG_FORMAT == 'svg':
    mpl.use('agg')
elif FIG_FORMAT == 'pdf':
    mpl.use('pdf')
elif FIG_FORMAT == 'interactive':
    mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn.apionly as sns

import lab
import lab.analysis.place_cell_analysis as place
import lab.analysis.reward_analysis as ra
import lab.misc as misc
import lab.plotting as plotting

import Df16a_analysis as df
from Fig1_GOL_task_performance import day_number_only_label, label_conditions

# Colors
WT_color = df.WT_color
Df_color = df.Df_color
colors = (WT_color, Df_color)

# ROI filters
WT_filter = df.WT_filter
Df_filter = df.Df_filter
roi_filters = (WT_filter, Df_filter)

markers = df.markers
linestyles = df.linestyles

save_dir = df.fig_save_dir
filename = 'Fig5_goal_enrichment{}.{}'.format(
    '_males' if MALES_ONLY else '', FIG_FORMAT)


def fix_heatmap_ax(ax, expt_grp):
    plotting.right_label(
        ax, 'cell index', rotation=270, ha='center', va='center')
    ax.set_xlabel('Distance from reward (fraction of belt)')
    ax.tick_params(
        which='both', bottom=True, left=False, labelleft=False,
        labelbottom=True, top=False, right=True, direction='out')
    ax.yaxis.tick_right()
    n_cells = int(ax.get_ylim()[0] + 0.05)
    ax.set_yticks(ax.get_ylim())
    ax.set_yticklabels([str(n_cells), '1'])
    xlim = ax.get_xlim()
    ax.set_xticks([xlim[0], np.mean(xlim), xlim[1]])
    ax.set_xticklabels(['-0.5', '0.0', '0.5'])
    for spine in ax.spines.itervalues():
        spine.set_linewidth(1)

    # Figure out the reward window width
    windows = []
    for expt in expt_grp:
        track_length = expt[0].behaviorData()['trackLength']
        window = float(expt.get('operantSpatialWindow'))
        windows.append(window / track_length)
    ax.plot([0.5, 0.5], [0, 1], transform=ax.transAxes, color='k', ls='--')
    ax.plot([np.mean(windows) + 0.5, np.mean(windows) + 0.5],
            [0, 1], transform=ax.transAxes, color='k', ls='--')
    ax.set_xlim(xlim)


def main():
    all_grps = df.loadExptGrps('GOL')

    WT_expt_grp = all_grps['WT_place_set']
    Df_expt_grp = all_grps['Df_place_set']
    expt_grps = [WT_expt_grp, Df_expt_grp]
    if MALES_ONLY:
        for expt_grp in expt_grps:
            expt_grp.filter(lambda expt: expt.parent.get('sex') == 'M')

    WT_label = WT_expt_grp.label()
    Df_label = Df_expt_grp.label()

    fig = plt.figure(figsize=(8.5, 11))

    gs1 = plt.GridSpec(
        2, 5, left=0.1, right=0.3, top=0.90, bottom=0.67, hspace=0.2)
    gs1_2 = plt.GridSpec(
        2, 5, left=0.3, right=0.5, top=0.90, bottom=0.67, hspace=0.2)
    WT_1_heatmap_ax = fig.add_subplot(gs1[0, :-1])
    WT_3_heatmap_ax = fig.add_subplot(gs1_2[0, :-1])
    Df_1_heatmap_ax = fig.add_subplot(gs1[1, :-1])
    Df_3_heatmap_ax = fig.add_subplot(gs1_2[1, :-1])

    gs_cbar = plt.GridSpec(
        2, 10, left=0.3, right=0.5, top=0.90, bottom=0.67, hspace=0.2)
    WT_colorbar_ax = fig.add_subplot(gs_cbar[0, -1])
    Df_colorbar_ax = fig.add_subplot(gs_cbar[1, -1])

    gs2 = plt.GridSpec(1, 10, left=0.1, right=0.5, top=0.6, bottom=0.45)
    pf_close_fraction_ax = fig.add_subplot(gs2[0, :4])
    pf_close_behav_corr_ax = fig.add_subplot(gs2[0, 5:])

    frac_near_range_2 = (-0.051, 0.551)
    behav_range_2 = (-0.051, 0.551)

    #
    # Heatmaps
    #

    WT_cmap = sns.light_palette(WT_color, as_cmap=True)
    WT_dataframe = lab.ExperimentGroup.dataframe(
        WT_expt_grp, include_columns=['X_condition', 'X_day', 'X_session'])

    WT_1_expt_grp = WT_expt_grp.subGroup(list(
        WT_dataframe[
            (WT_dataframe['X_condition'] == 'C') &
            (WT_dataframe['X_day'] == '0') &
            (WT_dataframe['X_session'] == '0')]['expt']))
    place.plotPositionHeatmap(
        WT_1_expt_grp, roi_filter=WT_filter, ax=WT_1_heatmap_ax,
        norm='individual', cbar_visible=False, cmap=WT_cmap,
        plotting_order='place_cells_only', show_belt=False,
        reward_in_middle=True)
    fix_heatmap_ax(WT_1_heatmap_ax, WT_1_expt_grp)
    WT_1_heatmap_ax.set_title(r'Condition $\mathrm{III}$: Day 1')
    WT_1_heatmap_ax.set_ylabel(WT_label)
    WT_1_heatmap_ax.set_xlabel('')

    WT_3_expt_grp = WT_expt_grp.subGroup(list(
        WT_dataframe[
            (WT_dataframe['X_condition'] == 'C') &
            (WT_dataframe['X_day'] == '2') &
            (WT_dataframe['X_session'] == '0')]['expt']))
    place.plotPositionHeatmap(
        WT_3_expt_grp, roi_filter=WT_filter, ax=WT_3_heatmap_ax,
        norm='individual', cbar_visible=True, cax=WT_colorbar_ax, cmap=WT_cmap,
        plotting_order='place_cells_only', show_belt=False,
        reward_in_middle=True)
    fix_heatmap_ax(WT_3_heatmap_ax, WT_3_expt_grp)
    WT_3_heatmap_ax.set_title(r'Condition $\mathrm{III}$: Day 3')
    WT_3_heatmap_ax.set_ylabel('')
    WT_3_heatmap_ax.set_xlabel('')
    WT_colorbar_ax.set_yticklabels(['Min', 'Max'])

    Df_cmap = sns.light_palette(Df_color, as_cmap=True)
    Df_dataframe = lab.ExperimentGroup.dataframe(
        Df_expt_grp, include_columns=['X_condition', 'X_day', 'X_session'])

    Df_1_expt_grp = Df_expt_grp.subGroup(list(
        Df_dataframe[
            (Df_dataframe['X_condition'] == 'C') &
            (Df_dataframe['X_day'] == '0') &
            (Df_dataframe['X_session'] == '2')]['expt']))
    place.plotPositionHeatmap(
        Df_1_expt_grp, roi_filter=Df_filter, ax=Df_1_heatmap_ax,
        norm='individual', cbar_visible=False, cmap=Df_cmap,
        plotting_order='place_cells_only', show_belt=False,
        reward_in_middle=True)
    fix_heatmap_ax(Df_1_heatmap_ax, Df_1_expt_grp)
    Df_1_heatmap_ax.set_ylabel(Df_label)

    Df_3_expt_grp = Df_expt_grp.subGroup(list(
        Df_dataframe[
            (Df_dataframe['X_condition'] == 'C') &
            (Df_dataframe['X_day'] == '2') &
            (Df_dataframe['X_session'] == '0')]['expt']))
    place.plotPositionHeatmap(
        Df_3_expt_grp, roi_filter=Df_filter, ax=Df_3_heatmap_ax,
        norm='individual', cbar_visible=True, cax=Df_colorbar_ax, cmap=Df_cmap,
        plotting_order='place_cells_only', show_belt=False,
        reward_in_middle=True)
    fix_heatmap_ax(Df_3_heatmap_ax, Df_3_expt_grp)
    Df_3_heatmap_ax.set_ylabel('')
    Df_colorbar_ax.set_yticklabels(['Min', 'Max'])

    #
    # Fraction of PCs near reward
    #

    activity_metric = place.centroid_to_position_threshold
    activity_kwargs = {'method': 'resultant_vector', 'positions': 'reward',
                       'pcs_only': True, 'threshold': np.pi / 8}
    behavior_fn = ra.fraction_licks_in_reward_zone
    behavior_kwargs = {}
    behavior_label = 'Fraction of licks in reward zone'

    plotting.plot_metric(
        pf_close_fraction_ax, expt_grps, metric_fn=activity_metric,
        roi_filters=roi_filters, groupby=[['expt', 'X_condition', 'X_day']],
        plotby=['X_condition', 'X_day'], plot_abs=False, plot_method='line',
        activity_kwargs=activity_kwargs, rotate_labels=False,
        activity_label='Fraction of place cells near reward',
        label_every_n=1, colors=colors, markers=markers, markersize=5,
        return_full_dataframes=False, linestyles=linestyles)
    pf_close_fraction_ax.axhline(1 / 8., linestyle='--', color='k')
    pf_close_fraction_ax.set_title('')
    sns.despine(ax=pf_close_fraction_ax)
    pf_close_fraction_ax.set_xlabel('Day in Condition')
    day_number_only_label(pf_close_fraction_ax)
    label_conditions(pf_close_fraction_ax)
    pf_close_fraction_ax.legend(loc='upper left', fontsize=6)
    pf_close_fraction_ax.set_ylim(0, 0.40)
    pf_close_fraction_ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])

    scatter_kws = {'s': 5}
    colorby_list = [(expt_grp.label(), 'C') for expt_grp in expt_grps]
    pf_close_behav_corr_ax.set_xlim(frac_near_range_2)
    pf_close_behav_corr_ax.set_ylim(behav_range_2)
    plotting.plot_paired_metrics(
        expt_grps, first_metric_fn=place.centroid_to_position_threshold,
        second_metric_fn=behavior_fn,
        roi_filters=roi_filters, groupby=(('expt',),),
        colorby=('expt_grp', 'X_condition'),
        filter_fn=lambda df: df['X_condition'] == 'C',
        filter_columns=['X_condition'],
        first_metric_kwargs=activity_kwargs,
        second_metric_kwargs=behavior_kwargs,
        first_metric_label='Fraction of place cells near reward',
        second_metric_label=behavior_label, shuffle_colors=False, fit_reg=True,
        plot_method='regplot', colorby_list=colorby_list, colors=colors,
        markers=markers, ax=pf_close_behav_corr_ax, scatter_kws=scatter_kws,
        truncate=False, linestyles=linestyles)
    pf_close_behav_corr_ax.set_xlim(frac_near_range_2)
    pf_close_behav_corr_ax.set_ylim(behav_range_2)
    pf_close_behav_corr_ax.tick_params(direction='in')
    pf_close_behav_corr_ax.get_legend().set_visible(False)
    pf_close_behav_corr_ax.legend(loc='upper left', fontsize=6)

    misc.save_figure(fig, filename, save_dir=save_dir)

    plt.close('all')

if __name__ == '__main__':
    main()
