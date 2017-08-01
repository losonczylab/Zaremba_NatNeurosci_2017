"""Figure S13 - Enrichment model parameter swap"""

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
import seaborn.apionly as sns

import lab.misc as misc
import lab.plotting as plotting

import os
import sys
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..', 'enrichment_model'))
import enrichment_model as em
import enrichment_model_plotting as emp
import Df16a_analysis as df

WT_label = r'$WT\ model$'
flat_label = r'$flat\ model$'

WT_color = df.WT_color
flat_color = sns.xkcd_rgb['burnt orange']
colors = [WT_color, flat_color]

save_dir = df.fig_save_dir
filename = 'FigS13_model_swap_parameters.{}'.format(FIG_FORMAT)

WT_params_path = os.path.join(
    df.data_path, 'enrichment_model', 'WT_model_params_C.pkl')


def main():

    fig, axs = plt.subplots(
        5, 2, figsize=(8.5, 11), gridspec_kw={
            'wspace': 0.4, 'hspace': 0.35, 'right': 0.75, 'top': 0.9,
            'bottom': 0.1})

    #
    # Run the model
    #

    n_cells = 1000
    n_runs = 100
    tol = 1e-4

    WT_params = pkl.load(open(WT_params_path, 'r'))

    model_cls = em.EnrichmentModel2

    WT_model = model_cls(**WT_params)
    flat_model = WT_model.copy()
    flat_model.flatten()

    WT_model.initialize(n_cells=n_cells, flat_tol=tol)
    flat_model.initialize_like(WT_model)
    initial_mask = WT_model.mask
    initial_positions = WT_model.positions

    def run_model(model1, model2=None, initial_positions=None,
                  initial_mask=None, n_runs=100, **interp_kwargs):
        masks, positions = [], []

        model = model1.copy()

        if model2 is not None:
            model.interpolate(model2, **interp_kwargs)

        for _ in range(n_runs):
            model.initialize(
                initial_mask=initial_mask, initial_positions=initial_positions)

            model.run(8)

            masks.append(model._masks)
            positions.append(model._positions)

        return positions, masks

    WT_no_swap_pos, WT_no_swap_masks = run_model(
        WT_model, initial_positions=initial_positions,
        initial_mask=initial_mask, n_runs=n_runs)
    flat_no_swap_pos, flat_no_swap_masks = run_model(
        flat_model, initial_positions=initial_positions,
        initial_mask=initial_mask, n_runs=n_runs)

    WT_swap_on_pos, WT_swap_on_masks = run_model(
        WT_model, flat_model, initial_positions=initial_positions,
        initial_mask=initial_mask, n_runs=n_runs, on=1)
    flat_swap_on_pos, flat_swap_on_masks = run_model(
        flat_model, WT_model, initial_positions=initial_positions,
        initial_mask=initial_mask, n_runs=n_runs, on=1)

    WT_swap_recur_pos, WT_swap_recur_masks = run_model(
        WT_model, flat_model, initial_positions=initial_positions,
        initial_mask=initial_mask, n_runs=n_runs, recur=1)
    flat_swap_recur_pos, flat_swap_recur_masks = run_model(
        flat_model, WT_model, initial_positions=initial_positions,
        initial_mask=initial_mask, n_runs=n_runs, recur=1)

    WT_swap_shift_b_pos, WT_swap_shift_b_masks = run_model(
        WT_model, flat_model, initial_positions=initial_positions,
        initial_mask=initial_mask, n_runs=n_runs, shift_b=1)
    flat_swap_shift_b_pos, flat_swap_shift_b_masks = run_model(
        flat_model, WT_model, initial_positions=initial_positions,
        initial_mask=initial_mask, n_runs=n_runs, shift_b=1)

    WT_swap_shift_k_pos, WT_swap_shift_k_masks = run_model(
        WT_model, flat_model, initial_positions=initial_positions,
        initial_mask=initial_mask, n_runs=n_runs, shift_k=1)
    flat_swap_shift_k_pos, flat_swap_shift_k_masks = run_model(
        flat_model, WT_model, initial_positions=initial_positions,
        initial_mask=initial_mask, n_runs=n_runs, shift_k=1)

    #
    # Distance to reward
    #

    WT_no_swap_enrich = emp.calc_enrichment(
        WT_no_swap_pos, WT_no_swap_masks)
    flat_no_swap_enrich = emp.calc_enrichment(
        flat_no_swap_pos, flat_no_swap_masks)

    WT_swap_recur_enrich = emp.calc_enrichment(
        WT_swap_recur_pos, WT_swap_recur_masks)
    flat_swap_recur_enrich = emp.calc_enrichment(
        flat_swap_recur_pos, flat_swap_recur_masks)

    WT_swap_shift_b_enrich = emp.calc_enrichment(
        WT_swap_shift_b_pos, WT_swap_shift_b_masks)
    flat_swap_shift_b_enrich = emp.calc_enrichment(
        flat_swap_shift_b_pos, flat_swap_shift_b_masks)

    WT_swap_shift_k_enrich = emp.calc_enrichment(
        WT_swap_shift_k_pos, WT_swap_shift_k_masks)
    flat_swap_shift_k_enrich = emp.calc_enrichment(
        flat_swap_shift_k_pos, flat_swap_shift_k_masks)

    emp.plot_enrichment(
        axs[0, 0], WT_no_swap_enrich, WT_color, '', rad=False)
    emp.plot_enrichment(
        axs[0, 0], flat_no_swap_enrich, flat_color, '', rad=False)
    plotting.right_label(axs[0, 1], 'No swap')
    axs[0, 0].set_ylabel('')

    emp.plot_enrichment(
        axs[1, 0], WT_swap_recur_enrich, WT_color, '', rad=False)
    emp.plot_enrichment(
        axs[1, 0], flat_swap_recur_enrich, flat_color, '', rad=False)
    plotting.right_label(axs[1, 1], 'Swap recurrence')

    emp.plot_enrichment(
        axs[2, 0], WT_swap_shift_k_enrich, WT_color, '', rad=False)
    emp.plot_enrichment(
        axs[2, 0], flat_swap_shift_k_enrich, flat_color, '', rad=False)
    plotting.right_label(axs[2, 1], 'Swap shift variance')
    axs[2, 0].set_ylabel('')

    emp.plot_enrichment(
        axs[3, 0], WT_swap_shift_b_enrich, WT_color, '', rad=False)
    emp.plot_enrichment(
        axs[3, 0], flat_swap_shift_b_enrich, flat_color, '', rad=False)
    plotting.right_label(axs[3, 1], 'Swap shift offset')
    axs[3, 0].set_ylabel('')

    plotting.stackedText(
        axs[0, 0], [WT_label, flat_label], colors=colors, loc=2, size=10)

    #
    # Final distribution
    #

    WT_no_swap_final_dist = emp.calc_final_distributions(
        WT_no_swap_pos, WT_no_swap_masks)
    flat_no_swap_final_dist = emp.calc_final_distributions(
        flat_no_swap_pos, flat_no_swap_masks)

    WT_swap_recur_final_dist = emp.calc_final_distributions(
        WT_swap_recur_pos, WT_swap_recur_masks)
    flat_swap_recur_final_dist = emp.calc_final_distributions(
        flat_swap_recur_pos, flat_swap_recur_masks)

    WT_swap_shift_b_final_dist = emp.calc_final_distributions(
        WT_swap_shift_b_pos, WT_swap_shift_b_masks)
    flat_swap_shift_b_final_dist = emp.calc_final_distributions(
        flat_swap_shift_b_pos, flat_swap_shift_b_masks)

    WT_swap_shift_k_final_dist = emp.calc_final_distributions(
        WT_swap_shift_k_pos, WT_swap_shift_k_masks)
    flat_swap_shift_k_final_dist = emp.calc_final_distributions(
        flat_swap_shift_k_pos, flat_swap_shift_k_masks)

    emp.plot_final_distributions(
        axs[0, 1], [WT_no_swap_final_dist, flat_no_swap_final_dist], colors,
        labels=None, title='', rad=False)
    emp.plot_final_distributions(
        axs[1, 1], [WT_swap_recur_final_dist, flat_swap_recur_final_dist],
        colors, labels=None, title='', rad=False)
    emp.plot_final_distributions(
        axs[2, 1], [WT_swap_shift_k_final_dist, flat_swap_shift_k_final_dist],
        colors, labels=None, title='', rad=False)
    emp.plot_final_distributions(
        axs[3, 1], [WT_swap_shift_b_final_dist, flat_swap_shift_b_final_dist],
        colors, labels=None, title='', rad=False)

    for ax in axs[:3, :].flat:
        ax.set_xlabel('')
    for ax in axs[4]:
        ax.set_visible(False)

    misc.save_figure(fig, filename, save_dir=save_dir)

    plt.close('all')

if __name__ == '__main__':
    main()
