"""Figure 7 - WT enrichment model"""

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
import cPickle as pickle
from scipy.special import i0
import seaborn.apionly as sns

import lab.misc as misc
import lab.plotting as plotting

import os
import sys
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..', 'enrichment_model'))
import enrichment_model_plotting as emp
import Df16a_analysis as df

# Colors
WT_color = df.WT_color
flat_color = sns.xkcd_rgb['burnt orange']
colors = (WT_color, flat_color)

WT_label = r'$WT\ model$'
flat_label = r'$flat\ model$'
labels = (WT_label, flat_label)

markers = ['o', 's']

save_dir = df.fig_save_dir
filename = 'Fig7_WT_enrichment_model.{}'.format(FIG_FORMAT)

simulations_path = os.path.join(
    df.data_path, 'enrichment_model',
    'WT_flat_enrichment_model_simulation_C.pkl')


def main():

    fig = plt.figure(figsize=(8.5, 11))

    gs = plt.GridSpec(
        3, 2, left=0.1, bottom=0.4, right=0.5, top=0.9, hspace=0.4, wspace=0.4)
    shift_schematic_ax = fig.add_subplot(gs[0, 1])
    model_enrich_by_time_ax = fig.add_subplot(gs[1, 0])
    model_final_enrich_ax = fig.add_subplot(gs[1, 1])
    model_swap_compare_enrich_ax = fig.add_subplot(gs[2, 0])
    model_WT_swap_final_enrich_ax = fig.add_subplot(gs[2, 1])

    #
    # Shift schematic
    #

    mu = 1
    k = 2.
    pos1 = 1.5

    def vms(x, mu=1, k=2.):
        return np.exp(k * np.cos(x - mu)) / (2 * np.pi * i0(k))

    xr = np.linspace(-np.pi, np.pi, 1000)

    shift_schematic_ax.plot(xr, [vms(x) for x in xr], color='k')
    shift_schematic_ax.axvline(pos1, color='r', ls='--')
    shift_schematic_ax.plot(
        [mu, mu], [0, vms(mu, mu=mu, k=k)], ls=':', color='0.3')
    sns.despine(ax=shift_schematic_ax, top=True, right=True)
    shift_schematic_ax.tick_params(labelleft=True, left=True, direction='out')
    shift_schematic_ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    shift_schematic_ax.set_xticklabels(['-0.50', '-0.25', '0', '0.25', '0.50'])
    shift_schematic_ax.set_xlim(-np.pi, np.pi)
    shift_schematic_ax.set_yticks([0, 0.2, 0.4, 0.6])
    shift_schematic_ax.set_ylim(0, 0.6)
    shift_schematic_ax.set_xlabel('Distance from reward (fraction of belt)')
    shift_schematic_ax.set_ylabel(
        'New place field centroid\nprobability density')

    #
    # Distance to reward
    #

    m = pickle.load(open(simulations_path))

    WT_enrich = emp.calc_enrichment(m['WT_no_swap_pos'], m['WT_no_swap_masks'])
    flat_enrich = emp.calc_enrichment(m['Df_no_swap_pos'], m['Df_no_swap_masks'])

    emp.plot_enrichment(
        model_enrich_by_time_ax, WT_enrich, WT_color, title='', rad=False)
    emp.plot_enrichment(
        model_enrich_by_time_ax, flat_enrich, flat_color, title='', rad=False)

    model_enrich_by_time_ax.set_xlabel("Iteration ('session' #)")
    plotting.stackedText(
        model_enrich_by_time_ax, [WT_label, flat_label], colors=colors, loc=2,
        size=10)

    #
    # Calc final distributions
    #

    WT_no_swap_final_dist = emp.calc_final_distributions(
        m['WT_no_swap_pos'], m['WT_no_swap_masks'])
    flat_no_swap_final_dist = emp.calc_final_distributions(
        m['Df_no_swap_pos'], m['Df_no_swap_masks'])

    WT_swap_recur_final_dist = emp.calc_final_distributions(
        m['WT_swap_recur_pos'], m['WT_swap_recur_masks'])
    flat_swap_recur_final_dist = emp.calc_final_distributions(
        m['Df_swap_recur_pos'], m['Df_swap_recur_masks'])

    WT_swap_shift_b_final_dist = emp.calc_final_distributions(
        m['WT_swap_shift_b_pos'], m['WT_swap_shift_b_masks'])
    flat_swap_shift_b_final_dist = emp.calc_final_distributions(
        m['Df_swap_shift_b_pos'], m['Df_swap_shift_b_masks'])

    WT_swap_shift_k_final_dist = emp.calc_final_distributions(
        m['WT_swap_shift_k_pos'], m['WT_swap_shift_k_masks'])
    flat_swap_shift_k_final_dist = emp.calc_final_distributions(
        m['Df_swap_shift_k_pos'], m['Df_swap_shift_k_masks'])

    #
    # Final distribution
    #

    emp.plot_final_distributions(
        model_final_enrich_ax,
        [WT_no_swap_final_dist, flat_no_swap_final_dist],
        colors, labels=labels, title='', rad=False)

    plotting.stackedText(
        model_final_enrich_ax, [WT_label, flat_label], colors=colors, loc=2,
        size=10)
    #
    # Compare enrichment
    #

    WT_bars = [
        WT_no_swap_final_dist, WT_swap_recur_final_dist,
        WT_swap_shift_k_final_dist, WT_swap_shift_b_final_dist]
    flat_bars = [
        flat_no_swap_final_dist, flat_swap_recur_final_dist,
        flat_swap_shift_k_final_dist, flat_swap_shift_b_final_dist]

    WT_bars = [np.pi / 2 - np.abs(bar) for bar in WT_bars]
    flat_bars = [np.pi / 2 - np.abs(bar) for bar in flat_bars]

    plotting.grouped_bar(
        model_swap_compare_enrich_ax, [WT_bars, flat_bars],
        [WT_label, flat_label],
        ['No\nswap', 'Swap\n' + r'$P_{recur}$',
         'Swap\nvariance', 'Swap\noffset'], [WT_color, flat_color],
        error_bars=None)

    sns.despine(ax=model_swap_compare_enrich_ax)
    model_swap_compare_enrich_ax.tick_params(length=3, pad=2, direction='out')
    model_swap_compare_enrich_ax.set_ylabel('Final enrichment (rad)')
    model_swap_compare_enrich_ax.get_legend().set_visible(False)
    model_swap_compare_enrich_ax.set_ylim(0, 0.08 * 2 * np.pi)
    y_ticks = np.array(['0', '0.02', '0.04', '0.06', '0.08'])
    model_swap_compare_enrich_ax.set_yticks(
        y_ticks.astype('float') * 2 * np.pi)
    model_swap_compare_enrich_ax.set_yticklabels(y_ticks)
    plotting.stackedText(
        model_swap_compare_enrich_ax, [WT_label, flat_label], colors=colors,
        loc=2, size=10)

    #
    # WT swap final distributions
    #

    hist_colors = iter(sns.color_palette())

    emp.plot_final_distributions(
        model_WT_swap_final_enrich_ax,
        [WT_no_swap_final_dist,
         WT_swap_recur_final_dist, WT_swap_shift_k_final_dist,
         WT_swap_shift_b_final_dist], colors=hist_colors,
        labels=['No swap', r'Swap $P_{recur}$',
                'Swap shift variance', 'Swap shift offset'], rad=False)
    model_WT_swap_final_enrich_ax.legend(
        loc='lower right', fontsize=5, frameon=False)

    misc.save_figure(fig, filename, save_dir=save_dir)

    plt.close('all')

if __name__ == '__main__':
    main()
