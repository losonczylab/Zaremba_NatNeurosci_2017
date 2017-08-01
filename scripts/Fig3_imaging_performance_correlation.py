"""Figure 3 - Imaging and task performance correlation

Also used to generate plots for:
    Supplemental Figure 5 - Place field correlation

"""

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
import seaborn.apionly as sns

import lab.analysis.place_cell_analysis as place
import lab.analysis.reward_analysis as ra
import lab.misc as misc
import lab.plotting as plotting

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
linestyles = df.linestyles

save_dir = df.fig_save_dir
filename = 'Fig3_imaging_performance_correlation{}.{}'.format(
    '_males' if MALES_ONLY else '', FIG_FORMAT)


def main():
    all_grps = df.loadExptGrps('GOL')

    WT_expt_grp = all_grps['WT_place_set']
    Df_expt_grp = all_grps['Df_place_set']
    expt_grps = [WT_expt_grp, Df_expt_grp]
    if MALES_ONLY:
        for expt_grp in expt_grps:
            expt_grp.filter(lambda expt: expt.parent.get('sex') == 'M')

    data_to_save = {}

    fig = plt.figure(figsize=(8.5, 11))

    recur_cdf_ax = fig.add_axes([0.1, 0.70, 0.22, 0.20])
    recur_inset_ax = fig.add_axes([0.26, 0.72, 0.05, 0.08])
    recur_day_sess_comp_ax = fig.add_axes([0.41, 0.70, 0.15, 0.20])
    recur_behav_corr_ax = fig.add_axes([0.65, 0.70, 0.27, 0.20])

    stability_over_time_ax = fig.add_axes([0.1, 0.40, 0.22, 0.20])
    stability_inset_ax = fig.add_axes([0.26, 0.42, 0.05, 0.08])
    stab_day_sess_comp_ax = fig.add_axes([0.41, 0.40, 0.15, 0.20])
    stability_behav_corr_ax = fig.add_axes([0.65, 0.40, 0.27, 0.20])

    params = {}
    params['recur_range'] = (-0.05, 0.85)
    params['recur_xticks'] = [0.0, 0.2, 0.4, 0.6, 0.8]

    fraction_in_params = {}
    fraction_in_params['behavior_fn'] = ra.fraction_licks_in_reward_zone
    fraction_in_params['behavior_kwargs'] = {}
    fraction_in_params['behavior_label'] = 'Fraction of licks in reward zone'
    fraction_in_params['behav_range'] = (-0.05, 0.65)

    anticipatory_licking_params = {}
    anticipatory_licking_params['behavior_fn'] = ra.fraction_licks_near_rewards
    anticipatory_licking_params['behavior_kwargs'] = {
        'pre_window_cm': 5, 'exclude_reward': True}
    anticipatory_licking_params['behavior_label'] = \
        'Anticipatory licking fraction'
    anticipatory_licking_params['behav_range'] = (-0.05, 0.95)

    activity_centroid_params = {}
    activity_centroid_params['stability_fn'] = place.activity_centroid_shift
    activity_centroid_params['stability_kwargs'] = {
        'activity_filter': 'active_both', 'circ_var_pcs': circ_var_pcs,
        'units': 'rad'}
    activity_centroid_params['stability_label'] = \
        'Centroid shift (fraction of belt)'
    activity_centroid_params['stab_range'] = (0.90, 2)

    act_cent_norm_params = {}
    act_cent_norm_params['stability_kwargs'] = {
        'activity_filter': 'active_both', 'circ_var_pcs': circ_var_pcs,
        'units': 'norm'}
    act_cent_norm_params['stab_range'] = (0.15, 0.31)
    act_cent_norm_params['stab_xticks'] = (0.15, 0.20, 0.25, 0.30)
    act_cent_norm_params['stab_day_ses_ylim'] = (0.10, 0.3)
    act_cent_norm_params['stab_day_ses_yticks'] = (0.1, 0.15, 0.20, 0.25, 0.30)
    act_cent_norm_params['stab_inset_ylim'] = (0.15, 0.3)
    act_cent_norm_params['stab_inset_yticks'] = (0.15, 0.3)

    act_cent_pc_parms = {}
    act_cent_pc_parms['stability_kwargs'] = {
        'activity_filter': 'pc_both', 'circ_var_pcs': circ_var_pcs,
        'units': 'rad'}

    act_cent_cm_params = {}
    act_cent_cm_params['stability_kwargs'] = {
        'activity_filter': 'active_both', 'circ_var_pcs': circ_var_pcs,
        'units': 'cm'}
    act_cent_cm_params['stability_label'] = 'Centroid shift (cm)'
    act_cent_cm_params['stab_range'] = (25, 65)
    act_cent_cm_params['stab_inset_ylim'] = (0, 50)
    act_cent_cm_params['stab_inset_yticks'] = (0, 50)
    act_cent_cm_params['stab_day_ses_ylim'] = (30, 50)

    pf_corr_params = {}
    pf_corr_params['stability_fn'] = place.place_field_correlation
    pf_corr_params['stability_kwargs'] = {'activity_filter': 'pc_either'}
    pf_corr_params['stability_label'] = 'Place field correlation'
    pf_corr_params['stab_range'] = (-0.1, 0.5)
    pf_corr_params['stab_xticks'] = (-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5)
    pf_corr_params['stab_inset_ylim'] = (0, 0.30)
    pf_corr_params['stab_inset_yticks'] = (0, 0.30)
    pf_corr_params['stab_day_ses_ylim'] = (0, 0.7)
    pf_corr_params['stab_day_ses_yticks'] = (0, 0.2, 0.4, 0.6)

    pop_vec_corr_params = {}
    pop_vec_corr_params['stability_fn'] = place.population_vector_correlation
    pop_vec_corr_params['stability_kwargs'] = {
        'method': 'corr', 'activity_filter': 'pc_both', 'min_pf_density': 0.05}
    pop_vec_corr_params['stability_label'] = 'Population vector correlation'

    #
    # Select parameters
    #
    params.update(fraction_in_params)
    params.update(activity_centroid_params)
    params.update(act_cent_norm_params)
    # params.update(pf_corr_params)  # For Supplemental Figure 5

    #
    # Recurrence
    #
    day_paired_grps = [grp.pair('same group', groupby=['X_session']).pair(
        'consecutive groups', groupby=['X_condition', 'X_day']) for grp in
        expt_grps]
    session_paired_grps = [
        grp.pair('consecutive groups',
                 groupby=['X_condition', 'X_day', 'X_session'])
        for grp in expt_grps]

    data_to_save['recurrence'] = plotting.plot_metric(
        recur_cdf_ax, day_paired_grps, metric_fn=place.recurrence_probability,
        groupby=(('second_expt',),), plot_method='cdf',
        roi_filters=roi_filters,
        activity_kwargs={'circ_var_pcs': circ_var_pcs}, plot_shuffle=True,
        shuffle_plotby=False, pool_shuffle=True,
        activity_label='Recurrence probability', colors=colors,
        rotate_labels=False, return_full_dataframes=True)
    recur_cdf_ax.legend(loc='upper left', fontsize=6)
    recur_cdf_ax.set_title('')
    recur_cdf_ax.set_xlim(params['recur_range'])
    recur_cdf_ax.set_xticks(params['recur_xticks'])

    data_to_save['recurrence_inset'] = plotting.plot_metric(
        recur_inset_ax, day_paired_grps,
        metric_fn=place.recurrence_probability,
        groupby=(('second_expt', 'second_mouseID'), ('second_mouseID',)),
        plot_method='swarm', roi_filters=roi_filters,
        activity_kwargs={'circ_var_pcs': circ_var_pcs}, plot_shuffle=True,
        shuffle_plotby=False, pool_shuffle=True,
        activity_label='Recurrence probability', colors=colors,
        rotate_labels=False, plot_shuffle_as_hline=True, linewidth=0.2,
        edgecolor='gray')
    recur_inset_ax.get_legend().set_visible(False)
    sns.despine(ax=recur_inset_ax)
    recur_inset_ax.set_title('')
    recur_inset_ax.set_ylabel('')
    recur_inset_ax.set_xlabel('')
    recur_inset_ax.tick_params(bottom=False, labelbottom=False)
    recur_inset_ax.set_ylim(0, 1.)
    recur_inset_ax.set_yticks([0, 1.])

    #
    # Recurrence - behavior correlation
    #
    scatter_kws = {'s': 5}
    colorby_list = [(expt_grp.label(),) for expt_grp in expt_grps]
    recur_behav_corr_ax.set_xlim(params['recur_range'])
    recur_behav_corr_ax.set_ylim(params['behav_range'])
    data_to_save['recur_behavior_scatter'] = plotting.plot_paired_metrics(
        day_paired_grps, first_metric_fn=place.recurrence_probability,
        second_metric_fn=params['behavior_fn'],
        roi_filters=roi_filters, groupby=(('second_expt',),),
        colorby=['expt_grp'], filter_fn=None, filter_columns=None,
        first_metric_kwargs=None,
        second_metric_kwargs=params['behavior_kwargs'],
        first_metric_label='Recurrence probability',
        second_metric_label=params['behavior_label'], shuffle_colors=False,
        fit_reg=True, plot_method='regplot', colorby_list=colorby_list,
        colors=colors, markers=markers, ax=recur_behav_corr_ax,
        scatter_kws=scatter_kws, truncate=False, linestyles=linestyles)
    recur_behav_corr_ax.set_xlim(params['recur_range'])
    recur_behav_corr_ax.set_ylim(params['behav_range'])
    recur_behav_corr_ax.tick_params(direction='in')
    recur_behav_corr_ax.legend(loc='upper left', fontsize=6)

    #
    # Stability
    #
    filter_fn = None
    filter_columns = None
    data_to_save['stability'] = plotting.plot_metric(
        stability_over_time_ax, day_paired_grps,
        metric_fn=params['stability_fn'], groupby=(('second_expt',),),
        plotby=None, plot_method='cdf', roi_filters=roi_filters,
        activity_kwargs=params['stability_kwargs'], plot_shuffle=True,
        shuffle_plotby=False, pool_shuffle=True,
        activity_label=params['stability_label'], colors=colors,
        rotate_labels=False, filter_fn=filter_fn,
        filter_columns=filter_columns, return_full_dataframes=True)
    stability_over_time_ax.legend(loc='upper left', fontsize=6)
    stability_over_time_ax.set_title('')
    stability_over_time_ax.set_xlabel(params['stability_label'])
    stability_over_time_ax.set_xlim(params['stab_range'])
    stability_over_time_ax.set_xticks(params['stab_xticks'])

    data_to_save['stability_inset'] = plotting.plot_metric(
        stability_inset_ax, day_paired_grps, metric_fn=params['stability_fn'],
        groupby=(('second_expt', 'second_mouseID'), ('second_mouseID',),),
        plotby=None, plot_method='swarm', roi_filters=roi_filters,
        activity_kwargs=params['stability_kwargs'], plot_shuffle=True,
        shuffle_plotby=False, pool_shuffle=True,
        activity_label=params['stability_label'], colors=colors,
        rotate_labels=False, filter_fn=filter_fn,
        filter_columns=filter_columns, plot_shuffle_as_hline=True,
        linewidth=0.2, edgecolor='gray')
    stability_inset_ax.get_legend().set_visible(False)
    sns.despine(ax=stability_inset_ax)
    stability_inset_ax.set_title('')
    stability_inset_ax.set_ylabel('')
    stability_inset_ax.set_xlabel('')
    stability_inset_ax.tick_params(bottom=False, labelbottom=False)
    stability_inset_ax.set_ylim(params['stab_inset_ylim'])
    stability_inset_ax.set_yticks(params['stab_inset_yticks'])

    #
    # Stability - behavior correlation
    #
    scatter_kws = {'s': 5}
    colorby_list = [(expt_grp.label(),) for expt_grp in expt_grps]
    stability_behav_corr_ax.set_xlim(params['stab_range'])
    stability_behav_corr_ax.set_ylim(params['behav_range'])
    data_to_save['stability_behavior_scatter'] = plotting.plot_paired_metrics(
        day_paired_grps, first_metric_fn=params['stability_fn'],
        second_metric_fn=params['behavior_fn'],
        roi_filters=roi_filters, groupby=(('second_expt',),),
        colorby=['expt_grp'], filter_fn=filter_fn,
        filter_columns=filter_columns,
        first_metric_kwargs=params['stability_kwargs'],
        second_metric_kwargs=params['behavior_kwargs'],
        first_metric_label=params['stability_label'],
        second_metric_label=params['behavior_label'],
        shuffle_colors=False, fit_reg=True, plot_method='regplot',
        colorby_list=colorby_list, colors=colors, markers=markers,
        ax=stability_behav_corr_ax, scatter_kws=scatter_kws, truncate=False,
        linestyles=linestyles)
    stability_behav_corr_ax.tick_params(direction='in')
    stability_behav_corr_ax.set_xlim(params['stab_range'])
    stability_behav_corr_ax.set_ylim(params['behav_range'])
    stability_behav_corr_ax.set_xticks(params['stab_xticks'])
    stability_behav_corr_ax.set_xlabel(params['stability_label'])
    stability_behav_corr_ax.legend(loc='upper right', fontsize=6)

    #
    # Day vs. session elapsed comparison
    #
    filter_fn = None
    filter_columns = None
    line_kwargs = {'markersize': 4}
    data_to_save['recur_day_sess_comp'] = plotting.plot_metric(
        recur_day_sess_comp_ax, session_paired_grps,
        metric_fn=place.recurrence_probability,
        groupby=(('elapsed_days_int', 'second_expt', 'second_mouseID'),),
        plotby=('elapsed_days_int',), plot_method='box_and_line',
        roi_filters=roi_filters, markers=markers,
        activity_kwargs={'circ_var_pcs': circ_var_pcs}, plot_shuffle=True,
        shuffle_plotby=False, pool_shuffle=True,
        activity_label='Recurrence probability', colors=colors,
        rotate_labels=False, plot_shuffle_as_hline=True,
        flierprops={'markersize': 2, 'marker': 'o'}, box_width=0.4,
        box_spacing=0.2, return_full_dataframes=True, whis='range',
        linestyles=linestyles, notch=False, line_kwargs=line_kwargs)
    sns.despine(ax=recur_day_sess_comp_ax)
    recur_day_sess_comp_ax.legend(loc='upper right', fontsize=6)
    recur_day_sess_comp_ax.set_title('')
    recur_day_sess_comp_ax.set_xlabel('')
    recur_day_sess_comp_ax.set_xticklabels(['S-S', 'D-D'])
    recur_day_sess_comp_ax.set_ylim(0.0, 1.0)
    recur_day_sess_comp_ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.])

    data_to_save['stab_day_sess_comp'] = plotting.plot_metric(
        stab_day_sess_comp_ax, session_paired_grps,
        metric_fn=params['stability_fn'],
        groupby=(('elapsed_days_int', 'second_expt', 'second_mouseID'),),
        plotby=('elapsed_days_int',), plot_method='box_and_line',
        roi_filters=roi_filters, activity_kwargs=params['stability_kwargs'],
        plot_shuffle=True, shuffle_plotby=False, pool_shuffle=True,
        activity_label=params['stability_label'], colors=colors,
        rotate_labels=False, filter_fn=filter_fn, markers=markers,
        filter_columns=filter_columns, plot_shuffle_as_hline=True,
        flierprops={'markersize': 2, 'marker': 'o'}, box_width=0.4,
        box_spacing=0.2, return_full_dataframes=True, whis='range',
        linestyles=linestyles, notch=False, line_kwargs=line_kwargs)
    sns.despine(ax=stab_day_sess_comp_ax)
    stab_day_sess_comp_ax.legend(loc='upper right', fontsize=6)
    stab_day_sess_comp_ax.set_title('')
    stab_day_sess_comp_ax.set_xlabel('')
    stab_day_sess_comp_ax.set_ylabel(params['stability_label'])
    stab_day_sess_comp_ax.set_xticklabels(['S-S', 'D-D'])
    stab_day_sess_comp_ax.set_ylim(params['stab_day_ses_ylim'])
    stab_day_sess_comp_ax.set_yticks(params['stab_day_ses_yticks'])

    misc.save_figure(fig, filename, save_dir=save_dir)

    plt.close('all')


if __name__ == '__main__':
    main()
