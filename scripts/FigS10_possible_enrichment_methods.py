"""Figure S10 - Possible enrichment methods"""

FIG_FORMAT = 'svg'

import matplotlib as mpl
if FIG_FORMAT == 'svg':
    mpl.use('agg')
elif FIG_FORMAT == 'pdf':
    mpl.use('pdf')
elif FIG_FORMAT == 'interactive':
    mpl.use('TkAgg')
import matplotlib.pyplot as plt
import cPickle as pkl
import numpy as np
import seaborn.apionly as sns

from lab.misc import save_figure

import os
import sys
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..', 'enrichment_model'))
import enrichment_model_plotting as emp
import enrichment_model as em
import enrichment_model_theoretical as emt
import Df16a_analysis as df

markers = df.markers

save_dir = df.fig_save_dir
filename = 'FigS10_possible_enrichment_methods.{}'.format(FIG_FORMAT)

params_path = os.path.join(
    df.data_path, 'enrichment_model', 'WT_model_params_C.pkl')


def show_parameters(axs, model, enrich, color='b'):

    positions = np.linspace(-np.pi, np.pi, 1000)

    bs, ks = model.shift_mean_var(positions)
    recur = model.recur_by_position(positions)

    axs[0].plot(positions, recur, color=color)
    axs[0].axvline(ls='--', color='0.4', lw=0.5)
    axs[0].set_xlim(-np.pi, np.pi)
    axs[0].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axs[0].set_xticklabels(['-0.50', '-0.25', '0', '0.25', '0.50'])
    axs[0].set_ylim(-0.3, 1.3)
    axs[0].set_yticks([0, 0.5, 1])
    axs[0].tick_params(length=3, pad=1, top=False)
    axs[0].set_xlabel('Distance from reward (fraction of belt)')
    axs[0].set_ylabel('Recurrence probability')

    axs[1].plot(positions, bs, color=color)
    axs[1].axvline(ls='--', color='0.4', lw=0.5)
    axs[1].axhline(ls='--', color='0.4', lw=0.5)
    axs[1].tick_params(length=3, pad=1, top=False)
    axs[1].set_xlim(-np.pi, np.pi)
    axs[1].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axs[1].set_xticklabels(['-0.50', '-0.25', '0', '0.25', '0.50'])
    axs[1].set_ylim(-0.10 * 2 * np.pi, 0.10 * 2 * np.pi)
    y_ticks = np.array(['-0.10', '-0.05', '0', '0.05', '0.10'])
    axs[1].set_yticks(y_ticks.astype('float') * 2 * np.pi)
    axs[1].set_yticklabels(y_ticks)
    axs[1].set_xlabel('Initial distance from reward (fraction of belt)')
    axs[1].set_ylabel(r'$\Delta$ position (fraction of belt)')

    axs[2].plot(positions, 1 / ks, color=color)
    axs[2].axvline(ls='--', color='0.4', lw=0.5)
    axs[2].tick_params(length=3, pad=1, top=False)
    axs[2].set_xlim(-np.pi, np.pi)
    axs[2].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axs[2].set_xticklabels(['-0.50', '-0.25', '0', '0.25', '0.50'])
    axs[2].set_ylim(0, 1)
    y_ticks = np.array(['0', '0.005', '0.010', '0.015', '0.020', '0.025'])
    axs[2].set_yticks(y_ticks.astype('float') * (2 * np.pi) ** 2)
    axs[2].set_yticklabels(y_ticks)
    axs[2].set_xlabel('Initial distance from reward (fraction of belt)')
    axs[2].set_ylabel(r'$\Delta$ position variance')

    axs[3].plot(range(9), np.mean(enrich, axis=0), color=color)
    axs[3].plot(range(9), np.percentile(enrich, 5, axis=0), ls='--',
                color=color)
    axs[3].plot(range(9), np.percentile(enrich, 95, axis=0), ls='--',
                color=color)
    axs[3].fill_between(
        range(9), np.percentile(enrich, 5, axis=0),
        np.percentile(enrich, 95, axis=0), facecolor=color, alpha=0.5)
    axs[3].axhline(0, ls='--', color='0.4', lw=0.5)

    sns.despine(ax=axs[3])
    axs[3].tick_params(length=3, pad=2, direction='out')
    axs[3].set_xlabel("Iteration ('session' #)")
    axs[3].set_ylabel('Enrichment (fraction of belt)')
    axs[3].set_xlim(-0.5, 8.5)
    axs[3].set_xticks([0, 2, 4, 6, 8])
    axs[3].set_ylim(-0.15, 0.10 * 2 * np.pi)
    y_ticks = np.array(['0', '0.05', '0.10'])
    axs[3].set_yticks(y_ticks.astype('float') * 2 * np.pi)
    axs[3].set_yticklabels(y_ticks)


def main():
    WT_params = pkl.load(open(params_path))

    WT_model = em.EnrichmentModel2(**WT_params)
    recur_model = emt.EnrichmentModel2_recur(
        kappa=1, span=0.8, mean_recur=0.4, **WT_params)
    offset_model = emt.EnrichmentModel2_offset(alpha=0.25, **WT_params)
    var_model = emt.EnrichmentModel2_var(
        kappa=1, alpha=10, mean_k=3, **WT_params)

    WT_model.initialize(n_cells=1000, flat_tol=1e-6)
    recur_model.initialize_like(WT_model)
    offset_model.initialize_like(WT_model)
    var_model.initialize_like(WT_model)
    initial_mask = WT_model.mask
    initial_positions = WT_model.positions

    recur_masks, recur_positions = [], []
    offset_masks, offset_positions = [], []
    var_masks, var_positions = [], []

    n_runs = 100

    color = sns.xkcd_rgb['forest green']

    for _ in range(n_runs):
        recur_model.initialize(
            initial_mask=initial_mask, initial_positions=initial_positions)
        offset_model.initialize(
            initial_mask=initial_mask, initial_positions=initial_positions)
        var_model.initialize(
            initial_mask=initial_mask, initial_positions=initial_positions)

        recur_model.run(8)
        offset_model.run(8)
        var_model.run(8)

        recur_masks.append(recur_model._masks)
        recur_positions.append(recur_model._positions)

        offset_masks.append(offset_model._masks)
        offset_positions.append(offset_model._positions)

        var_masks.append(var_model._masks)
        var_positions.append(var_model._positions)

    recur_enrich = emp.calc_enrichment(recur_positions, recur_masks)
    offset_enrich = emp.calc_enrichment(offset_positions, offset_masks)
    var_enrich = emp.calc_enrichment(var_positions, var_masks)

    fig, axs = plt.subplots(
        4, 3, figsize=(8.5, 11), gridspec_kw={'bottom': 0.3, 'wspace': 0.4})

    show_parameters(axs[:, 0], recur_model, recur_enrich, color=color)
    show_parameters(axs[:, 1], offset_model, offset_enrich, color=color)
    show_parameters(axs[:, 2], var_model, var_enrich, color=color)

    for ax in axs[:, 1:].flat:
        ax.set_ylabel('')
    for ax in axs[:2, :].flat:
        ax.set_xlabel('')
    for ax in axs[:, 0]:
        ax.set_xlabel('')
    for ax in axs[:, 2]:
        ax.set_xlabel('')

    axs[0, 0].set_title('Stable recurrence')
    axs[0, 1].set_title('Shift towards reward')
    axs[0, 2].set_title('Stable position')

    save_figure(fig, filename, save_dir=save_dir)

if __name__ == '__main__':
    main()
