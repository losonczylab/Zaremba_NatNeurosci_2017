"""Figure S1 - Task performance by mouse"""

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

import lab.analysis.reward_analysis as ra
import lab.plotting as plotting
import lab.misc as misc

import Df16a_analysis as df

from Fig1_GOL_task_performance import label_conditions

WT_color = df.WT_color
Df_color = df.Df_color
colors = (WT_color, Df_color)

markers = df.markers
WT_marker, Df_marker = markers

save_dir = df.fig_save_dir
filename = 'FigS1_performance_by_mouse.{}'.format(FIG_FORMAT)


def main():

    all_grps = df.loadExptGrps('GOL')

    WT_expt_grp = all_grps['WT_hidden_behavior_set']
    Df_expt_grp = all_grps['Df_hidden_behavior_set']

    data_to_save = {}

    behavior_fn = ra.fraction_licks_in_reward_zone
    behavior_kwargs = {}
    activity_label = 'Fraction of licks in reward zone'

    WT_colors = sns.light_palette(WT_color, 8)[::-1]
    Df_colors = sns.light_palette(Df_color, 7)[::-1]
    markers = ('o', 'v', '^', 'D', '*', 's')

    fig, axs = plt.subplots(4, 2, figsize=(8.5, 11))

    sns.despine(fig)

    wt_ax = axs[0, 0]
    df_ax = axs[0, 1]

    for ax in list(axs.flat)[2:]:
        ax.set_visible(False)

    wt_expt_grps = [WT_expt_grp.subGroup(list(expts['expt']), label=mouse)
                    for mouse, expts in WT_expt_grp.dataframe(
                        WT_expt_grp, include_columns=['mouseID']).groupby(
                            'mouseID')]
    df_expt_grps = [Df_expt_grp.subGroup(list(expts['expt']), label=mouse)
                    for mouse, expts in Df_expt_grp.dataframe(
                        Df_expt_grp, include_columns=['mouseID']).groupby(
                            'mouseID')]

    data_to_save['WT'] = plotting.plot_metric(
        wt_ax, wt_expt_grps,
        metric_fn=behavior_fn, activity_kwargs=behavior_kwargs,
        groupby=[['expt'], ['condition_day']],
        plotby=['condition_day'], plot_method='line', ms=5,
        activity_label=activity_label, colors=WT_colors, markers=markers,
        label_every_n=1, label_groupby=False, rotate_labels=False)
    wt_ax.set_xlabel('Day in Condition')
    wt_ax.set_title(WT_expt_grp.label())
    wt_ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    wt_ax.set_xticklabels(['1', '2', '3', '1', '2', '3', '1', '2', '3'])
    label_conditions(wt_ax)
    wt_ax.get_legend().set_visible(False)
    wt_ax.tick_params(length=3, pad=2)

    data_to_save['Df'] = plotting.plot_metric(
        df_ax, df_expt_grps,
        metric_fn=behavior_fn, activity_kwargs=behavior_kwargs,
        groupby=[['expt'], ['condition_day']],
        plotby=['condition_day'], plot_method='line', ms=5,
        activity_label=activity_label, colors=Df_colors, markers=markers,
        label_every_n=1, label_groupby=False, rotate_labels=False)
    df_ax.set_xlabel('Day in Condition')
    df_ax.set_title(Df_expt_grp.label())
    df_ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    df_ax.set_xticklabels(['1', '2', '3', '1', '2', '3', '1', '2', '3'])
    label_conditions(df_ax)
    df_ax.get_legend().set_visible(False)
    df_ax.tick_params(length=3, pad=2)

    misc.save_figure(fig, filename, save_dir=save_dir)

if __name__ == '__main__':
    main()
