"""Figure S2 - Initial behavior comparison"""

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
import seaborn.apionly as sns

import lab
import lab.misc as misc
import lab.plotting as plotting

import Df16a_analysis as df

# Colors
WT_color = df.WT_color
Df_color = df.Df_color
colors = (WT_color, Df_color)

WT_label = df.WT_label
Df_label = df.Df_label

save_dir = df.fig_save_dir
filename = 'FigS2_behavior_compare.{}'.format(FIG_FORMAT)


def main():
    all_grps = df.loadExptGrps('GOL')

    WT_expt_grp = all_grps['WT_place_set']
    Df_expt_grp = all_grps['Df_place_set']
    expt_grps = [WT_expt_grp, Df_expt_grp]

    data_to_save = {}

    fig, axs = plt.subplots(
        4, 3, figsize=(8.5, 11), gridspec_kw={'wspace': 0.4, 'hspace': 0.3})

    #
    # Velocity
    #
    data_to_save['velocity_A'] = plotting.plot_metric(
        axs[0, 0], expt_grps, metric_fn=lab.ExperimentGroup.velocity_dataframe,
        groupby=[['expt'], ['mouseID']], plotby=None, plot_method='swarm',
        activity_kwargs=None, filter_fn=lambda df: df['condition'] == 'A',
        filter_columns=['condition'], activity_label='Velocity (cm/s)',
        colors=colors, plot_bar=True, edgecolor='k', linewidth=0.5,
        return_full_dataframes=True)
    plotting.plot_metric(
        axs[0, 1], expt_grps, metric_fn=lab.ExperimentGroup.velocity_dataframe,
        groupby=[['expt'], ['session_in_day', 'mouseID']],
        plotby=['session_in_day'], plot_method='swarm', activity_kwargs=None,
        rotate_labels=False, filter_fn=lambda df: df['condition'] == 'A',
        filter_columns=['condition'], activity_label='Velocity (cm/s)',
        colors=colors, plot_bar=True, edgecolor='k', linewidth=0.5)

    #
    # Lap Rate
    #
    data_to_save['lap_rate_A'] = plotting.plot_metric(
        axs[1, 0], expt_grps, metric_fn=lab.ExperimentGroup.number_of_laps,
        groupby=[['expt'], ['mouseID']], plotby=None, plot_method='swarm',
        activity_kwargs={'rate': True},
        filter_fn=lambda df: df['condition'] == 'A',
        filter_columns=['condition'], activity_label='Lap rate (1/min)',
        colors=colors, agg_fn=[np.sum, np.mean], plot_bar=True, edgecolor='k',
        linewidth=0.5, return_full_dataframes=True)
    plotting.plot_metric(
        axs[1, 1], expt_grps, metric_fn=lab.ExperimentGroup.number_of_laps,
        groupby=[['expt'], ['session_in_day', 'mouseID']],
        plotby=['session_in_day'], plot_method='swarm',
        activity_kwargs={'rate': True}, rotate_labels=False,
        filter_fn=lambda df: df['condition'] == 'A',
        filter_columns=['condition'], activity_label='Lap rate (1/min)',
        colors=colors, agg_fn=[np.sum, np.mean], plot_bar=True, edgecolor='k',
        linewidth=0.5)

    #
    # Lick Rate
    #
    data_to_save['lick_rate_A'] = plotting.plot_metric(
        axs[2, 0], expt_grps, metric_fn=lab.ExperimentGroup.behavior_dataframe,
        groupby=[['trial'], ['mouseID']], plotby=None,
        plot_method='swarm', activity_kwargs={'key': 'licking', 'rate': True},
        filter_fn=lambda df: df['condition'] == 'A',
        filter_columns=['condition'], activity_label='Lick rate (Hz)',
        colors=colors, agg_fn=[np.sum, np.mean], plot_bar=True, edgecolor='k',
        linewidth=0.5, return_full_dataframes=True)
    plotting.plot_metric(
        axs[2, 1], expt_grps, metric_fn=lab.ExperimentGroup.behavior_dataframe,
        groupby=[['trial'], ['session_in_day', 'mouseID']],
        plotby=['session_in_day'], plot_method='swarm', roi_filters=None,
        activity_kwargs={'key': 'licking', 'rate': True},
        filter_fn=lambda df: df['condition'] == 'A',
        filter_columns=['condition'], activity_label='Lick rate (Hz)',
        colors=colors, agg_fn=[np.sum, np.mean], rotate_labels=False,
        plot_bar=True, edgecolor='k', linewidth=0.5)

    for ax in axs[3, :]:
        ax.set_visible(False)
    for ax in axs[:, 2]:
        ax.set_visible(False)

    sns.despine(fig=fig)

    for ax in axs.flat:
        ax.set_title('')
    for ax in axs[:, 0]:
        ax.set_xlabel('')
        ax.tick_params(bottom=False, labelbottom=False, length=3, pad=2)

    for ax in axs[:, 1]:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels(['1', '2', '3'])
        ax.tick_params(length=3, pad=2)
    for ax in axs[:3, :2].flat:
        ax.get_legend().set_visible(False)

    plotting.stackedText(
        axs[0, 0], [WT_label, Df_label], colors=colors, loc=2, size=10)

    axs[2, 1].set_xlabel('Session in day')

    for ax in axs[0, :]:
        ax.set_ylim(0, 13)
    for ax in axs[1, :]:
        ax.set_ylim(0, 2.0)
        ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    for ax in axs[2, :]:
        ax.set_ylim(0, 3)

    misc.save_figure(fig, filename, save_dir=save_dir)

    plt.close('all')

if __name__ == '__main__':
    main()
