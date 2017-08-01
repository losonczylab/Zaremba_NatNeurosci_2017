"""Figure S6 - Performance and stability by Condition"""

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
import pandas as pd

import lab.plotting as plotting
import lab.analysis.place_cell_analysis as place
import lab.analysis.reward_analysis as ra
import lab.misc as misc

import Df16a_analysis as df

WT_color = df.WT_color
Df_color = df.Df_color
colors = (WT_color, Df_color)

WT_filter = df.WT_filter
Df_filter = df.Df_filter
roi_filters = (WT_filter, Df_filter)

markers = df.markers
linestyles= df.linestyles

WT_label = df.WT_label
Df_label = df.Df_label
labels = [WT_label, Df_label]

save_dir = df.fig_save_dir
filename = 'FigS6_stability_by_condition.{}'.format(FIG_FORMAT)


def z_score_value(dataframes, invert=False):
    all_dfs = pd.concat(dataframes, ignore_index=True)
    all_mean = all_dfs.value.mean()
    all_std = all_dfs.value.std(ddof=0)
    new_dataframes = []
    for dataframe in dataframes:
        new_dataframe = dataframe.copy()
        new_dataframe['value'] = (new_dataframe.value - all_mean) / all_std
        if invert:
            new_dataframe['value'] *= -1
        new_dataframes.append(new_dataframe)
    return new_dataframes


def main():

    all_grps = df.loadExptGrps('GOL')

    WT_expt_grp = all_grps['WT_hidden_behavior_set']
    Df_expt_grp = all_grps['Df_hidden_behavior_set']
    expt_grps = [WT_expt_grp, Df_expt_grp]

    WT_pc_expt_grp = all_grps['WT_place_set']
    Df_pc_expt_grp = all_grps['Df_place_set']
    pc_expt_grps = [WT_pc_expt_grp, Df_pc_expt_grp]

    data_to_save = {}

    paired_grps = [grp.pair(
        'consecutive groups', groupby=['condition_day_session']).pair(
            'same group', groupby=['condition']) for grp in pc_expt_grps]

    behavior_fn = ra.fraction_licks_in_reward_zone
    behavior_kwargs = {}
    behavior_label = 'Fraction of licks in reward zone'

    recurrence_fn = place.recurrence_probability
    recurrence_kwargs = {'circ_var_pcs': False}
    recurrence_label = 'Recurrence probability'

    stability_fn = place.activity_centroid_shift
    stability_kwargs = {
        'activity_filter': 'active_both', 'circ_var_pcs': False,
        'units': 'norm'}
    stability_label = 'Centroid shift\n(fraction of belt)'
    stability_save_label = 'cent_shift'

    fig = plt.figure(figsize=(8.5, 11))
    gs = plt.GridSpec(
        2, 3, left=0.1, bottom=0.5, right=0.89, top=0.9, wspace=0.4)
    behav_ax = fig.add_subplot(gs[0, 0])
    recurrence_ax = fig.add_subplot(gs[0, 1])
    stability_ax = fig.add_subplot(gs[0, 2])
    behav_z_ax = fig.add_subplot(gs[1, 0])
    recurrence_z_ax = fig.add_subplot(gs[1, 1])
    stability_z_ax = fig.add_subplot(gs[1, 2])

    lick_data = [
        behavior_fn(expt_grp, **behavior_kwargs) for expt_grp in expt_grps]
    data_to_save['licks_in_reward_zone'] = plotting.plot_dataframe(
        behav_ax, lick_data, labels=labels,
        groupby=[['expt'], ['mouseID', 'condition', 'condition_day_session']],
        plotby=['condition'], plot_method='line',
        activity_label=behavior_label, colors=colors, label_groupby=False,
        markers=markers, markersize=5, linestyles=linestyles)
    behav_ax.set_ylim(0, 0.4)
    behav_ax.set_yticks([0, .1, .2, .3, .4])
    behav_ax.set_xlabel('')
    behav_ax.set_title('')
    behav_ax.set_xticklabels(
        [r'$\mathrm{I}$', r'$\mathrm{II}$', r'$\mathrm{III}$'])

    recur_data, recur_shuffles = [], []
    for expt_grp in paired_grps:
        rd, rs = recurrence_fn(expt_grp, **recurrence_kwargs)
        recur_data.append(rd)
        recur_shuffles.append(rs)
    data_to_save['recurrence'] = plotting.plot_dataframe(
        recurrence_ax, recur_data, recur_shuffles, labels=labels,
        groupby=[['second_mouse', 'second_condition',
                  'second_condition_day_session']],
        plotby=['second_condition'], plot_method='line',
        activity_label=recurrence_label, colors=colors, label_groupby=False,
        plot_shuffle=True, shuffle_plotby=False, pool_shuffle=True,
        markers=markers, markersize=5, linestyles=linestyles)
    recurrence_ax.set_ylim(0.0, 0.6)
    recurrence_ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    recurrence_ax.set_xlabel('')
    recurrence_ax.set_title('')
    recurrence_ax.set_xticklabels(
        [r'$\mathrm{I}$', r'$\mathrm{II}$', r'$\mathrm{III}$'])

    stability_data, stability_shuffles = [], []
    for expt_grp in paired_grps:
        sd, ss = stability_fn(expt_grp, **stability_kwargs)
        stability_data.append(sd)
        stability_shuffles.append(ss)
    data_to_save[stability_save_label] = plotting.plot_dataframe(
        stability_ax, stability_data, stability_shuffles, labels=labels,
        groupby=[['second_mouse', 'second_condition',
                  'second_condition_day_session']],
        plotby=['second_condition'], plot_method='line',
        activity_label=stability_label, colors=colors,
        label_groupby=False, plot_shuffle=True,
        shuffle_plotby=False, pool_shuffle=True, markers=markers,
        markersize=5, linestyles=linestyles)
    stability_ax.set_ylim(0.15, 0.25)
    stability_ax.set_yticks([0.15, 0.17, 0.19, 0.21, 0.23, 0.25])
    stability_ax.set_xlabel('')
    stability_ax.set_title('')
    stability_ax.set_xticklabels(
        [r'$\mathrm{I}$', r'$\mathrm{II}$', r'$\mathrm{III}$'])

    lick_z_data = z_score_value(lick_data)
    plotting.plot_dataframe(
        behav_z_ax, lick_z_data, labels=labels,
        plotby=['condition'], plot_method='line',
        activity_label=behavior_label + '\n(z-score)', colors=colors,
        label_groupby=False, markers=markers, markersize=5,
        linestyles=linestyles)
    behav_z_ax.set_ylim(-1, 1)
    behav_z_ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    behav_z_ax.set_xlabel('')
    behav_z_ax.set_title('')
    behav_z_ax.set_xticklabels(
        [r'$\mathrm{I}$', r'$\mathrm{II}$', r'$\mathrm{III}$'])

    recur_z_data = z_score_value(recur_data)
    plotting.plot_dataframe(
        recurrence_z_ax, recur_z_data, labels=labels,
        plotby=['second_condition'], plot_method='line',
        activity_label=recurrence_label + '\n(z-score)', colors=colors,
        label_groupby=False, markers=markers, markersize=5,
        linestyles=linestyles)
    recurrence_z_ax.set_ylim(-1, 1)
    recurrence_z_ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    recurrence_z_ax.set_xlabel('')
    recurrence_z_ax.set_title('')
    recurrence_z_ax.set_xticklabels(
        [r'$\mathrm{I}$', r'$\mathrm{II}$', r'$\mathrm{III}$'])

    stability_z_data = z_score_value(stability_data, invert=True)
    plotting.plot_dataframe(
        stability_z_ax, stability_z_data, labels=labels,
        plotby=['second_condition'], plot_method='line',
        activity_label=stability_label + '\n(-1 * z-score)', colors=colors,
        label_groupby=False, markers=markers, markersize=5,
        linestyles=linestyles)
    stability_z_ax.set_ylim(-1, 1)
    stability_z_ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    stability_z_ax.set_xlabel('')
    stability_z_ax.set_title('')
    stability_z_ax.set_xticklabels(
        [r'$\mathrm{I}$', r'$\mathrm{II}$', r'$\mathrm{III}$'])

    sns.despine(fig)

    misc.save_figure(fig, filename, save_dir=save_dir)

    plt.close('all')

if __name__ == '__main__':
    main()
