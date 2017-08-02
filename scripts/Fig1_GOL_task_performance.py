"""Figure 1 - GOL task performance"""

FIG_FORMAT = 'svg'
MALES_ONLY = False

import matplotlib as mpl
if FIG_FORMAT == 'svg':
    mpl.use('agg')
elif FIG_FORMAT == 'pdf':
    mpl.use('pdf')
elif FIG_FORMAT == 'interactive':
    mpl.use('TkAgg')
import matplotlib.pyplot as plt
import itertools as it
import os
import seaborn.apionly as sns

import lab
import lab.analysis.reward_analysis as ra
import lab.misc as misc
from lab.plotting import plot_metric, stackedText, right_label

import Df16a_analysis as df

WT_color = df.WT_color
Df_color = df.Df_color
colors = (WT_color, Df_color)

linestyles = df.linestyles
markers = df.markers

save_dir = df.fig_save_dir
filename = 'Fig1_GOL_task_performance{}.{}'.format(
    '_males' if MALES_ONLY else '', FIG_FORMAT)

# Fig5
def day_number_only_label(ax):
    xticklabels = ax.get_xticklabels()
    new_labels = []
    for label in xticklabels:
        # _, day = label.get_text().split('_')
        _, day = label.get_text().split(", '")
        day = str(int(day[0]) + 1)
        new_labels.append(day)
    ax.set_xticklabels(new_labels, ha='center', va='top')


condition_map = {
    'A': r'$\mathrm{I}$', 'B': r'$\mathrm{II}$', 'C': r'$\mathrm{III}$'}


def two_level_condition_day_label(ax):
    xticklabels = ax.get_xticklabels()
    new_labels = []
    for label in xticklabels:
        condition, day = label.get_text().split('_')
        day = str(int(day) + 1)
        new_labels.append('{}\n{}'.format(day, condition_map[condition]))
    ax.set_xticklabels(new_labels, ha='center', va='top')


first_reward_map = {'A': '+', 'B': '+', 'C': '-'}
second_reward_map = {'A': '-', 'B': '-', 'C': '+'}


def four_level_condition_day_reward_label(ax):
    xticklabels = ax.get_xticklabels()
    new_labels = []
    for label in xticklabels:
        condition, day = label.get_text().split('_')
        day = str(int(day) + 1)
        new_labels.append('{}\n{}\n{}\n{}'.format(
            day, first_reward_map[condition], second_reward_map[condition],
            condition if condition != 'C' else 'B'))
    ax.set_xticklabels(new_labels, ha='center', va='top')

context_map = {'A': 1, 'B': 2, 'C': 2}
reward_map = {'A': 1, 'B': 1, 'C': 2}


def four_level_day_context_reward_condition(ax):
    xticklabels = ax.get_xticklabels()
    new_labels = []
    for label in xticklabels:
        condition, day = label.get_text().split('_')
        new_labels.append('{}\n{}\n{}\n{}'.format(
            day, context_map[condition], reward_map[condition], condition))
    ax.set_xticklabels(new_labels, ha='center', va='top')


# Fig 5
def label_conditions(ax):
    ax.text(1 / 6., .95, r'$\mathrm{I}$', ha='center', va='center',
            transform=ax.transAxes, fontsize=12)
    ax.text(1 / 2., .95, r'$\mathrm{II}$', ha='center', va='center',
            transform=ax.transAxes, fontsize=12)
    ax.text(5 / 6., .95, r'$\mathrm{III}$', ha='center', va='center',
            transform=ax.transAxes, fontsize=12)
    ax.axvline(2.5, color='k', linestyle=':')
    ax.axvline(5.5, color='k', linestyle=':')


def label_conditions_2(ax):
    ax.text(1 / 6., .95, r'$\mathrm{I}$', ha='center', va='center',
            transform=ax.transAxes, fontsize=12)
    ax.text(1 / 2., .95, r'$\mathrm{II}$', ha='center', va='center',
            transform=ax.transAxes, fontsize=12)
    ax.text(5 / 6., .95, r'$\mathrm{III}$', ha='center', va='center',
            transform=ax.transAxes, fontsize=12)
    ax.axvline(1.5, color='k', linestyle=':')
    ax.axvline(3.5, color='k', linestyle=':')


def main():
    all_grps = df.loadExptGrps('GOL')
    expts = lab.ExperimentSet(
        os.path.join(df.metadata_path, 'expt_metadata.xml'),
        behaviorDataPath=os.path.join(df.data_path, 'behavior'),
        dataPath=os.path.join(df.data_path, 'imaging'))

    WT_expt_grp = all_grps['WT_hidden_behavior_set']
    Df_expt_grp = all_grps['Df_hidden_behavior_set']
    expt_grps = [WT_expt_grp, Df_expt_grp]

    if MALES_ONLY:
        for expt_grp in expt_grps:
            expt_grp.filter(lambda expt: expt.parent.get('sex') == 'M')
    labels = [expt_grp.label() for expt_grp in expt_grps]

    fig = plt.figure(figsize=(8.5, 11))

    HORIZONTAL = False

    if HORIZONTAL:
        gs1 = plt.GridSpec(8, 6)
        wt_lick_axs = [fig.add_subplot(gs1[0, 0]),
                       fig.add_subplot(gs1[0, 1]),
                       fig.add_subplot(gs1[0, 2]),
                       fig.add_subplot(gs1[0, 3]),
                       fig.add_subplot(gs1[0, 4]),
                       fig.add_subplot(gs1[0, 5])]
        df_lick_axs = [fig.add_subplot(gs1[1, 0]),
                       fig.add_subplot(gs1[1, 1]),
                       fig.add_subplot(gs1[1, 2]),
                       fig.add_subplot(gs1[1, 3]),
                       fig.add_subplot(gs1[1, 4]),
                       fig.add_subplot(gs1[1, 5])]

        gs2 = plt.GridSpec(4, 2, hspace=0.5, wspace=0.2)
        reward_zone_ax = fig.add_subplot(gs2[1, 0])
    else:
        gs1 = plt.GridSpec(10, 6)
        wt_lick_axs = [fig.add_subplot(gs1[0, 0]),
                       fig.add_subplot(gs1[1, 0]),
                       fig.add_subplot(gs1[2, 0]),
                       fig.add_subplot(gs1[3, 0]),
                       fig.add_subplot(gs1[4, 0]),
                       fig.add_subplot(gs1[5, 0])]
        df_lick_axs = [fig.add_subplot(gs1[0, 1]),
                       fig.add_subplot(gs1[1, 1]),
                       fig.add_subplot(gs1[2, 1]),
                       fig.add_subplot(gs1[3, 1]),
                       fig.add_subplot(gs1[4, 1]),
                       fig.add_subplot(gs1[5, 1])]

        gs2 = plt.GridSpec(10, 1, hspace=0.5, wspace=0.8, left=0.47, right=0.9)
        reward_zone_ax = fig.add_subplot(gs2[0:4, :])
        gs3 = plt.GridSpec(10, 3, hspace=0.5, wspace=0.1, left=0.47, right=0.9)
        fraction_licks_by_session_A_ax = fig.add_subplot(gs3[5:7, 0])
        fraction_licks_by_session_B_ax = fig.add_subplot(gs3[5:7, 1])
        fraction_licks_by_session_C_ax = fig.add_subplot(gs3[5:7, 2])

    #
    # Lick plots
    #

    wt_lick_expts = [expts.grabExpt('jz101', '2014-11-06-23h37m54s'),
                     expts.grabExpt('jz101', '2014-11-08-22h53m27s'),
                     expts.grabExpt('jz101', '2014-11-09-23h06m56s'),
                     expts.grabExpt('jz101', '2014-11-11-23h13m16s'),
                     expts.grabExpt('jz101', '2014-11-12-19h29m41s'),
                     expts.grabExpt('jz101', '2014-11-14-19h59m09s')]
    df_lick_expts = [expts.grabExpt('jz106', '2014-12-11-17h06m49s'),
                     expts.grabExpt('jz106', '2014-12-13-19h00m01s'),
                     expts.grabExpt('jz106', '2014-12-14-17h17m17s'),
                     expts.grabExpt('jz106', '2014-12-16-17h43m05s'),
                     expts.grabExpt('jz106', '2014-12-17-17h57m51s'),
                     expts.grabExpt('jz106', '2014-12-19-17h13m52s')]

    shade_color = sns.xkcd_rgb['light green']
    for ax, expt in zip(wt_lick_axs, wt_lick_expts):
        expt.licktogram(
            ax=ax, plot_belt=False, nPositionBins=20, color=WT_color,
            linewidth=0, shade_reward=True, shade_color=shade_color)
    for ax, expt in zip(df_lick_axs, df_lick_expts):
        expt.licktogram(
            ax=ax, plot_belt=False, nPositionBins=20, color=Df_color,
            linewidth=0, shade_reward=True, shade_color=shade_color)

    for ax in wt_lick_axs + df_lick_axs:
        ax.set_ylim(0, 0.6)
        ax.set_yticks([0, 0.3, 0.6])
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(['0.0', '0.5', '1.0'])
        sns.despine(ax=ax)
        ax.set_title('')

    if HORIZONTAL:
        for ax in wt_lick_axs:
            ax.tick_params(labelbottom=False)
            ax.set_xlabel('')

        for ax in df_lick_axs[1:]:
            ax.set_xlabel('')

        for ax, label in zip(wt_lick_axs, [
                r'Condition $\mathrm{I}$' + '\nDay 1',
                r'Condition $\mathrm{I}$' + '\nDay 3',
                r'Condition $\mathrm{II}$' + '\nDay 1',
                r'Condition $\mathrm{II}$' + '\nDay 3',
                r'Condition $\mathrm{III}$' + '\nDay 1',
                r'Condition $\mathrm{III}$' + '\nDay 3']):
            ax.set_title(label)

        for ax in it.chain(wt_lick_axs[1:], df_lick_axs[1:]):
            ax.set_ylabel('')
            sns.despine(ax=ax, left=True, top=True, right=True)

        for ax in wt_lick_axs + df_lick_axs:
            ax.spines['bottom'].set_linewidth(0.5)

        wt_lick_axs[0].tick_params(labelbottom=False)
        for ax in wt_lick_axs[1:]:
            ax.tick_params(labelleft=False, left=False, labelbottom=False)
        for ax in df_lick_axs[1:]:
            ax.tick_params(labelleft=False, left=False)

        right_label(wt_lick_axs[-1], labels[0])
        right_label(df_lick_axs[-1], labels[1])

        df_lick_axs[0].set_yticks([0, 0.6])
        wt_lick_axs[0].set_yticks([0, 0.6])
        df_lick_axs[0].set_ylabel('Fraction of licks')
        wt_lick_axs[0].set_ylabel('')
        df_lick_axs[0].spines['left'].set_linewidth(0.5)
        wt_lick_axs[0].spines['left'].set_linewidth(0.5)
    else:
        for ax in wt_lick_axs + df_lick_axs:
            ax.spines['bottom'].set_linewidth(0.5)
        for ax in wt_lick_axs + df_lick_axs[1:]:
            sns.despine(ax=ax, left=True, top=True, right=True)
        sns.despine(ax=df_lick_axs[0], left=True, top=True, right=False)

        for ax in wt_lick_axs[:-1] + df_lick_axs[:-1]:
            ax.set_xlabel('')

        for ax in df_lick_axs:
            ax.set_ylabel('')

        for ax, label in zip(wt_lick_axs, [
                r'Condition $\mathrm{I}$' + '\nDay 1',
                r'Condition $\mathrm{I}$' + '\nDay 3',
                r'Condition $\mathrm{II}$' + '\nDay 1',
                r'Condition $\mathrm{II}$' + '\nDay 3',
                r'Condition $\mathrm{III}$' + '\nDay 1',
                r'Condition $\mathrm{III}$' + '\nDay 3']):
            ax.set_ylabel(
                label, rotation='horizontal', ha='right',
                multialignment='center', labelpad=3, va='center')

        for ax in wt_lick_axs[:-1] + df_lick_axs[:-1]:
            ax.tick_params(labelleft=False, left=False, labelbottom=False)

        for ax in (wt_lick_axs[-1], df_lick_axs[-1]):
            ax.tick_params(labelleft=False, left=False)

        wt_lick_axs[0].set_title(labels[0])
        df_lick_axs[0].set_title(labels[1])

        df_lick_axs[0].yaxis.tick_right()
        df_lick_axs[0].yaxis.set_label_position("right")
        df_lick_axs[0].set_yticks([0, 0.6])
        df_lick_axs[0].tick_params(axis='y', length=2, pad=2, direction='in')
        df_lick_axs[0].set_ylabel('Fraction of licks')
        df_lick_axs[0].spines['right'].set_linewidth(0.5)

    filter_fn = None
    filter_columns = None

    behavior_fn = ra.fraction_licks_in_reward_zone
    behavior_kwargs = {}
    activity_label = 'Fraction of licks in reward zone'

    plot_metric(
        reward_zone_ax, expt_grps,
        metric_fn=behavior_fn, activity_kwargs=behavior_kwargs,
        groupby=[['expt'], ['mouseID', 'X_condition', 'X_day']],
        plotby=['X_condition', 'X_day'], plot_method='line',
        activity_label=activity_label, colors=colors, linestyles=linestyles,
        label_every_n=1, label_groupby=False, markers=markers,
        markersize=5, rotate_labels=False, filter_fn=filter_fn,
        filter_columns=filter_columns, return_full_dataframes=False)
    reward_zone_ax.set_yticks([0, .1, .2, .3, .4])
    sns.despine(ax=reward_zone_ax)
    reward_zone_ax.set_xlabel('Day in Condition')
    reward_zone_ax.set_title('')
    day_number_only_label(reward_zone_ax)
    label_conditions(reward_zone_ax)
    reward_zone_ax.legend(loc='lower left', fontsize=8)
    # reward_zone_ax.get_legend().set_visible(False)
    # stackedText(reward_zone_ax, labels, colors=colors, loc=3, size=10)

    groupby = [['expt']]
    plotby = ['X_condition', 'X_session']

    filter_fn = lambda df: (df['X_session'] != '1') & (df['X_condition'] == 'A')
    filter_columns = ['X_session', 'X_condition']
    line_kwargs = {'markersize': 4}
    plot_metric(
        fraction_licks_by_session_A_ax, expt_grps,
        metric_fn=behavior_fn, activity_kwargs=behavior_kwargs,
        groupby=groupby, plotby=plotby, plot_method='box_and_line',
        activity_label=activity_label, colors=colors, notch=False,
        label_every_n=1, label_groupby=False, markers=markers,
        rotate_labels=False, line_kwargs=line_kwargs, linestyles=linestyles,
        filter_fn=filter_fn, filter_columns=filter_columns,
        flierprops={'markersize': 2, 'marker': 'o'}, box_width=0.4,
        box_spacing=0.2, return_full_dataframes=False, whis='range')
    sns.despine(ax=fraction_licks_by_session_A_ax, top=True, right=True)
    fraction_licks_by_session_A_ax.set_xticklabels(['first', 'last'])
    fraction_licks_by_session_A_ax.set_xlabel('')
    fraction_licks_by_session_A_ax.set_ylim(-0.02, 0.6)
    fraction_licks_by_session_A_ax.set_yticks([0, 0.2, 0.4, 0.6])
    fraction_licks_by_session_A_ax.set_title('')
    fraction_licks_by_session_A_ax.legend(loc='upper left', fontsize=6)
    # fraction_licks_by_session_A_ax.get_legend().set_visible(False)
    fraction_licks_by_session_A_ax.text(
        0.5, .95, r'$\mathrm{I}$', ha='center', va='center',
        transform=fraction_licks_by_session_A_ax.transAxes, fontsize=12)

    filter_fn = lambda df: (df['X_session'] != '1') & (df['X_condition'] == 'B')
    filter_columns = ['X_session', 'X_condition']
    plot_metric(
        fraction_licks_by_session_B_ax, expt_grps,
        metric_fn=behavior_fn, activity_kwargs=behavior_kwargs,
        groupby=groupby, plotby=plotby, plot_method='box_and_line',
        activity_label=activity_label, colors=colors,
        label_every_n=1, label_groupby=False, markers=markers,
        rotate_labels=False, line_kwargs=line_kwargs, linestyles=linestyles,
        filter_fn=filter_fn, filter_columns=filter_columns, notch=False,
        flierprops={'markersize': 2, 'marker': 'o'}, box_width=0.4,
        box_spacing=0.2, return_full_dataframes=False, whis='range')
    sns.despine(
        ax=fraction_licks_by_session_B_ax, top=True, right=True, left=True)
    fraction_licks_by_session_B_ax.tick_params(left=False, labelleft=False)
    fraction_licks_by_session_B_ax.set_xticklabels(['first', 'last'])
    fraction_licks_by_session_B_ax.set_xlabel('Session in day')
    fraction_licks_by_session_B_ax.set_ylabel('')
    fraction_licks_by_session_B_ax.set_ylim(-0.02, 0.6)
    fraction_licks_by_session_B_ax.set_title('')
    fraction_licks_by_session_B_ax.get_legend().set_visible(False)
    fraction_licks_by_session_B_ax.text(
        0.5, .95, r'$\mathrm{II}$', ha='center', va='center',
        transform=fraction_licks_by_session_B_ax.transAxes, fontsize=12)

    filter_fn = lambda df: (df['X_session'] != '1') & (df['X_condition'] == 'C')
    filter_columns = ['X_session', 'X_condition']
    plot_metric(
        fraction_licks_by_session_C_ax, expt_grps,
        metric_fn=behavior_fn, activity_kwargs=behavior_kwargs,
        groupby=groupby, plotby=plotby, plot_method='box_and_line',
        activity_label=activity_label, colors=colors, notch=False,
        label_every_n=1, label_groupby=False, markers=markers,
        rotate_labels=False, line_kwargs=line_kwargs, linestyles=linestyles,
        filter_fn=filter_fn, filter_columns=filter_columns,
        return_full_dataframes=False,
        flierprops={'markersize': 2, 'marker': 'o'}, box_width=0.4,
        box_spacing=0.2, whis='range')
    sns.despine(
        ax=fraction_licks_by_session_C_ax, top=True, right=True, left=True)
    fraction_licks_by_session_C_ax.tick_params(left=False, labelleft=False)
    fraction_licks_by_session_C_ax.set_xticklabels(['first', 'last'])
    fraction_licks_by_session_C_ax.set_xlabel('')
    fraction_licks_by_session_C_ax.set_ylabel('')
    fraction_licks_by_session_C_ax.set_ylim(-0.02, 0.6)
    fraction_licks_by_session_C_ax.set_title('')
    fraction_licks_by_session_C_ax.get_legend().set_visible(False)
    fraction_licks_by_session_C_ax.text(
        0.5, .95, r'$\mathrm{III}$', ha='center', va='center',
        transform=fraction_licks_by_session_C_ax.transAxes, fontsize=12)

    misc.save_figure(fig, filename, save_dir=save_dir)

    plt.close('all')

if __name__ == '__main__':
    main()
