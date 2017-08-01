"""Figure 8 - Df(16)A parameter fits and enrichment model"""

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
import seaborn.apionly as sns

import lab.misc as misc
import lab.misc.splines as splines

import os
import sys
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..', 'enrichment_model'))
import enrichment_model_data as emd
import enrichment_model_plotting as emp
import Df16a_analysis as df

Df_color = df.Df_color

markers = df.markers
_, Df_marker = df.markers

save_dir = df.fig_save_dir
filename = 'Fig8_Df_model.{}'.format(FIG_FORMAT)

simulations_path = os.path.join(
    df.data_path, 'enrichment_model',
    'WT_Df_enrichment_model_simulation_C.pkl')

params_path = os.path.join(
    df.data_path, 'enrichment_model', 'Df_model_params_C.pkl')


def main():
    Df_raw_data, Df_data = emd.load_data(
        'df', root=os.path.join(df.data_path, 'enrichment_model'))

    Df_params = pickle.load(open(params_path))

    fig = plt.figure(figsize=(8.5, 11))
    gs = plt.GridSpec(
        3, 2, left=0.1, bottom=0.5, right=0.5, top=0.9, wspace=0.3,
        hspace=0.3)
    Df_recur_ax = fig.add_subplot(gs[0, 0])
    Df_shift_ax = fig.add_subplot(gs[0, 1])
    shift_compare_ax = fig.add_subplot(gs[1, 0])
    var_compare_ax = fig.add_subplot(gs[1, 1])
    enrichment_ax = fig.add_subplot(gs[2, 0])
    final_enrichment_ax = fig.add_subplot(gs[2, 1])

    #
    # Recurrence by position
    #

    recur_x_vals = np.linspace(-np.pi, np.pi, 1000)

    Df_recur_data = emd.recurrence_by_position(Df_data, method='cv')
    Df_recur_knots = np.linspace(
        -np.pi, np.pi, Df_params['position_recurrence']['n_knots'])
    Df_recur_spline = splines.CyclicSpline(Df_recur_knots)
    Df_recur_N = Df_recur_spline.design_matrix(recur_x_vals)

    Df_recur_fit = splines.prob(
        Df_params['position_recurrence']['theta'], Df_recur_N)

    Df_recur_boots_fits = [splines.prob(boot, Df_recur_N) for boot in
                           Df_params['position_recurrence']['boots_theta']]
    Df_recur_ci_up_fit = np.percentile(Df_recur_boots_fits, 95, axis=0)
    Df_recur_ci_low_fit = np.percentile(Df_recur_boots_fits, 5, axis=0)

    Df_recur_ax.plot(recur_x_vals, Df_recur_fit, color=Df_color)
    Df_recur_ax.fill_between(
        recur_x_vals, Df_recur_ci_low_fit, Df_recur_ci_up_fit,
        facecolor=Df_color, alpha=0.5)
    sns.regplot(
        Df_recur_data[:, 0], Df_recur_data[:, 1], ax=Df_recur_ax,
        color=Df_color, y_jitter=0.2, fit_reg=False, scatter_kws={'s': 1},
        marker=Df_marker)
    Df_recur_ax.set_xlim(-np.pi, np.pi)
    Df_recur_ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    Df_recur_ax.set_xticklabels(['-0.50', '-0.25', '0', '0.25', '0.50'])
    Df_recur_ax.set_ylim(-0.3, 1.3)
    Df_recur_ax.set_yticks([0, 0.5, 1])
    Df_recur_ax.tick_params(length=3, pad=1, top=False)
    Df_recur_ax.set_xlabel('Initial distance from reward (fraction of belt)')
    Df_recur_ax.set_ylabel('Place cell recurrence probability')
    Df_recur_ax.set_title('')
    Df_recur_ax_2 = Df_recur_ax.twinx()
    Df_recur_ax_2.set_ylim(-0.3, 1.3)
    Df_recur_ax_2.set_yticks([0, 1])
    Df_recur_ax_2.set_yticklabels(['non-recur', 'recur'])
    Df_recur_ax_2.tick_params(length=3, pad=1, top=False)

    #
    # Place field stability
    #

    shift_x_vals = np.linspace(-np.pi, np.pi, 1000)

    Df_shift_knots = Df_params['position_stability']['all_pairs']['knots']
    Df_shift_spline = splines.CyclicSpline(Df_shift_knots)
    Df_shift_N = Df_shift_spline.design_matrix(shift_x_vals)
    Df_shift_theta_b = Df_params['position_stability']['all_pairs']['theta_b']
    Df_shift_b_fit = np.dot(Df_shift_N, Df_shift_theta_b)
    Df_shift_theta_k = Df_params['position_stability']['all_pairs']['theta_k']
    Df_shift_k_fit = splines.get_k(Df_shift_theta_k, Df_shift_N)
    Df_shift_fit_var = 1. / Df_shift_k_fit

    Df_shift_data = emd.paired_activity_centroid_distance_to_reward(Df_data)
    Df_shift_data = Df_shift_data.dropna()
    Df_shifts = Df_shift_data['second'] - Df_shift_data['first']
    Df_shifts[Df_shifts < -np.pi] += 2 * np.pi
    Df_shifts[Df_shifts >= np.pi] -= 2 * np.pi

    Df_shift_ax.plot(shift_x_vals, Df_shift_b_fit, color=Df_color)
    Df_shift_ax.fill_between(
        shift_x_vals, Df_shift_b_fit - Df_shift_fit_var,
        Df_shift_b_fit + Df_shift_fit_var, facecolor=Df_color, alpha=0.5)
    sns.regplot(
        Df_shift_data['first'], Df_shifts, ax=Df_shift_ax, color=Df_color,
        fit_reg=False, scatter_kws={'s': 1}, marker=Df_marker)

    Df_shift_ax.axvline(ls='--', color='0.4', lw=0.5)
    Df_shift_ax.axhline(ls='--', color='0.4', lw=0.5)
    Df_shift_ax.plot([-np.pi, np.pi], [np.pi, -np.pi], color='g', ls=':', lw=2)
    Df_shift_ax.tick_params(length=3, pad=1, top=False)
    Df_shift_ax.set_xlabel('Initial distance from reward (fraction of belt)')
    Df_shift_ax.set_ylabel(r'$\Delta$ position (fraction of belt)')
    Df_shift_ax.set_xlim(-np.pi, np.pi)
    Df_shift_ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    Df_shift_ax.set_xticklabels(['-0.50', '-0.25', '0', '0.25', '0.50'])
    Df_shift_ax.set_ylim(-np.pi, np.pi)
    Df_shift_ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    Df_shift_ax.set_yticklabels(['-0.50', '-0.25', '0', '0.25', '0.50'])
    Df_shift_ax.set_title('')

    #
    # Stability by distance to reward
    #

    shift_x_vals = np.linspace(-np.pi, np.pi, 1000)

    Df_shift_knots = Df_params['position_stability']['all_pairs']['knots']
    Df_shift_spline = splines.CyclicSpline(Df_shift_knots)
    Df_shift_N = Df_shift_spline.design_matrix(shift_x_vals)

    Df_shift_theta_b = Df_params['position_stability']['all_pairs']['theta_b']
    Df_shift_b_fit = np.dot(Df_shift_N, Df_shift_theta_b)
    Df_shift_boots_b_fit = [
        np.dot(Df_shift_N, boot) for boot in
        Df_params['position_stability']['all_pairs']['boots_theta_b']]
    Df_shift_b_ci_up_fit = np.percentile(Df_shift_boots_b_fit, 95, axis=0)
    Df_shift_b_ci_low_fit = np.percentile(Df_shift_boots_b_fit, 5, axis=0)

    shift_compare_ax.plot(shift_x_vals, Df_shift_b_fit, color=Df_color)
    shift_compare_ax.fill_between(
        shift_x_vals, Df_shift_b_ci_low_fit, Df_shift_b_ci_up_fit,
        facecolor=Df_color, alpha=0.5)

    shift_compare_ax.axvline(ls='--', color='0.4', lw=0.5)
    shift_compare_ax.axhline(ls='--', color='0.4', lw=0.5)
    shift_compare_ax.tick_params(length=3, pad=1, top=False)
    shift_compare_ax.set_xlim(-np.pi, np.pi)
    shift_compare_ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    shift_compare_ax.set_xticklabels(['-0.50', '-0.25', '0', '0.25', '0.50'])
    shift_compare_ax.set_ylim(-0.10 * 2 * np.pi, 0.10 * 2 * np.pi)
    y_ticks = np.array(['-0.10', '-0.05', '0', '0.05', '0.10'])
    shift_compare_ax.set_yticks(y_ticks.astype('float') * 2 * np.pi)
    shift_compare_ax.set_yticklabels(y_ticks)
    shift_compare_ax.set_xlabel(
        'Initial distance from reward (fraction of belt)')
    shift_compare_ax.set_ylabel(r'$\Delta$ position (fraction of belt)')

    Df_shift_theta_k = Df_params['position_stability']['all_pairs']['theta_k']
    Df_shift_k_fit = splines.get_k(Df_shift_theta_k, Df_shift_N)
    Df_shift_boots_k_fit = [splines.get_k(
        boot, Df_shift_N) for boot in
        Df_params['position_stability']['all_pairs']['boots_theta_k']]
    Df_shift_k_ci_up_fit = np.percentile(Df_shift_boots_k_fit, 95, axis=0)
    Df_shift_k_ci_low_fit = np.percentile(Df_shift_boots_k_fit, 5, axis=0)

    var_compare_ax.plot(shift_x_vals, 1. / Df_shift_k_fit, color=Df_color)
    var_compare_ax.fill_between(
        shift_x_vals, 1. / Df_shift_k_ci_low_fit, 1. / Df_shift_k_ci_up_fit,
        facecolor=Df_color, alpha=0.5)

    var_compare_ax.axvline(ls='--', color='0.4', lw=0.5)
    var_compare_ax.tick_params(length=3, pad=1, top=False)
    var_compare_ax.set_xlim(-np.pi, np.pi)
    var_compare_ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    var_compare_ax.set_xticklabels(['-0.50', '-0.25', '0', '0.25', '0.50'])
    y_ticks = np.array(['0.005', '0.010', '0.015', '0.020'])
    var_compare_ax.set_yticks(y_ticks.astype('float') * (2 * np.pi) ** 2)
    var_compare_ax.set_yticklabels(y_ticks)
    var_compare_ax.set_xlabel(
        'Initial distance from reward (fraction of belt)')
    var_compare_ax.set_ylabel(r'$\Delta$ position variance')

    #
    # Enrichment
    #

    m = pickle.load(open(simulations_path))
    Df_enrich = emp.calc_enrichment(m['Df_no_swap_pos'], m['Df_no_swap_masks'])

    emp.plot_enrichment(
        enrichment_ax, Df_enrich, Df_color, title='', rad=False)
    enrichment_ax.set_xlabel("Iteration ('session' #)")

    #
    # Final Enrichment
    #

    Df_no_swap_final_dist = emp.calc_final_distributions(
        m['Df_no_swap_pos'], m['Df_no_swap_masks'])

    emp.plot_final_distributions(
        final_enrichment_ax, [Df_no_swap_final_dist], [Df_color], title='',
        rad=False)

    misc.save_figure(fig, filename, save_dir=save_dir)

    plt.close('all')

if __name__ == '__main__':
    main()
