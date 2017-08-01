"""Figure S11 - Latent tuning"""

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
import cPickle as pkl
import pandas as pd
import seaborn.apionly as sns

import lab.plotting as plotting
import lab.misc.splines as splines
from lab.misc import save_figure

import os
import sys
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..', 'enrichment_model'))
import enrichment_model_data as emd
import Df16a_analysis as df

WT_color = df.WT_color
Df_color = df.Df_color
colors = (WT_color, Df_color)

WT_label = df.WT_label
Df_label = df.Df_label
labels = (WT_label, Df_label)

WT_marker = df.WT_marker
Df_marker = df.Df_marker
markers = [WT_marker, Df_marker]

save_dir = df.fig_save_dir
filename = 'FigS11_latent_tuning.{}'.format(FIG_FORMAT)

WT_params_path = os.path.join(
    df.data_path, 'enrichment_model', 'WT_model_params_C.pkl')
Df_params_path = os.path.join(
    df.data_path, 'enrichment_model', 'Df_model_params_C.pkl')


def main():

    WT_params = pkl.load(open(WT_params_path))
    Df_params = pkl.load(open(Df_params_path))

    WT_raw_data, WT_data = emd.load_data('wt')
    Df_raw_data, Df_data = emd.load_data('df')

    fig, axs = plt.subplots(
        3, 3, figsize=(11, 8.5), gridspec_kw={'wspace': 0.4})

    for ax in axs[1:, :].flat:
        ax.set_visible(False)

    #
    # 2 session stability
    #

    WT_2_shift_data = emd.tripled_activity_centroid_distance_to_reward(
        WT_data, prev_imaged=False)

    WT_2_shift_data = WT_2_shift_data.dropna(subset=['first', 'third'])
    WT_2_shifts = WT_2_shift_data['third'] - WT_2_shift_data['first']
    WT_2_shifts[WT_2_shifts < -np.pi] += 2 * np.pi
    WT_2_shifts[WT_2_shifts >= np.pi] -= 2 * np.pi

    WT_2_shift_data_prev = emd.tripled_activity_centroid_distance_to_reward(
        WT_data, prev_imaged=True)
    WT_2_shift_data_prev = WT_2_shift_data_prev.dropna(
        subset=['first', 'third'])
    WT_2_shifts_prev = WT_2_shift_data_prev['third'] - \
        WT_2_shift_data_prev['first']
    WT_2_shifts_prev[WT_2_shifts_prev < -np.pi] += 2 * np.pi
    WT_2_shifts_prev[WT_2_shifts_prev >= np.pi] -= 2 * np.pi
    WT_2_npc = np.isnan(WT_2_shift_data_prev['second'])

    sns.regplot(WT_2_shift_data_prev['first'][WT_2_npc],
                WT_2_shifts_prev[WT_2_npc], ax=axs[0, 0], color='m',
                fit_reg=False, scatter_kws={'s': 7}, marker='x')
    sns.regplot(WT_2_shift_data['first'], WT_2_shifts, ax=axs[0, 0],
                color=WT_color, fit_reg=False, scatter_kws={'s': 3},
                marker=WT_marker)
    axs[0, 0].axvline(ls='--', color='0.4', lw=0.5)
    axs[0, 0].axhline(ls='--', color='0.4', lw=0.5)
    axs[0, 0].plot([-np.pi, np.pi], [np.pi, -np.pi], color='g', ls=':', lw=2)
    axs[0, 0].tick_params(length=3, pad=1, top=False)
    axs[0, 0].set_xlabel('Initial distance from reward\n(fraction of belt)')
    axs[0, 0].set_ylabel(
        r'Two-session $\Delta$ position' + '\n(fraction of belt)')
    axs[0, 0].set_xlim(-np.pi, np.pi)
    axs[0, 0].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axs[0, 0].set_xticklabels(['-0.50', '-0.25', '0', '0.25', '0.50'])
    axs[0, 0].set_ylim(-np.pi, np.pi)
    axs[0, 0].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axs[0, 0].set_yticklabels(['-0.50', '-0.25', '0', '0.25', '0.50'])
    axs[0, 0].set_title(WT_label)

    Df_2_shift_data = emd.tripled_activity_centroid_distance_to_reward(
        Df_data, prev_imaged=False)
    Df_2_shift_data = Df_2_shift_data.dropna(subset=['first', 'third'])
    Df_2_shifts = Df_2_shift_data['third'] - Df_2_shift_data['first']
    Df_2_shifts[Df_2_shifts < -np.pi] += 2 * np.pi
    Df_2_shifts[Df_2_shifts >= np.pi] -= 2 * np.pi

    Df_2_shift_data_prev = emd.tripled_activity_centroid_distance_to_reward(
        Df_data, prev_imaged=True)
    Df_2_shift_data_prev = Df_2_shift_data_prev.dropna(
        subset=['first', 'third'])
    Df_2_shifts_prev = Df_2_shift_data_prev['third'] - \
        Df_2_shift_data_prev['first']
    Df_2_shifts_prev[Df_2_shifts_prev < -np.pi] += 2 * np.pi
    Df_2_shifts_prev[Df_2_shifts_prev >= np.pi] -= 2 * np.pi
    Df_2_npc = np.isnan(Df_2_shift_data_prev['second'])

    sns.regplot(Df_2_shift_data_prev['first'][Df_2_npc],
                Df_2_shifts_prev[Df_2_npc], ax=axs[0, 1], color='c',
                fit_reg=False, scatter_kws={'s': 7}, marker='x')
    sns.regplot(Df_2_shift_data['first'], Df_2_shifts, ax=axs[0, 1],
                color=Df_color, fit_reg=False, scatter_kws={'s': 3},
                marker=Df_marker)
    axs[0, 1].axvline(ls='--', color='0.4', lw=0.5)
    axs[0, 1].axhline(ls='--', color='0.4', lw=0.5)
    axs[0, 1].plot([-np.pi, np.pi], [np.pi, -np.pi], color='g', ls=':', lw=2)
    axs[0, 1].tick_params(length=3, pad=1, top=False)
    axs[0, 1].set_xlabel('Initial distance from reward\n(fraction of belt)')
    axs[0, 1].set_ylabel(
        r'Two-session $\Delta$ position' + '\n(fraction of belt)')
    axs[0, 1].set_xlim(-np.pi, np.pi)
    axs[0, 1].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axs[0, 1].set_xticklabels(['-0.50', '-0.25', '0', '0.25', '0.50'])
    axs[0, 1].set_ylim(-np.pi, np.pi)
    axs[0, 1].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    axs[0, 1].set_yticklabels(['-0.50', '-0.25', '0', '0.25', '0.50'])
    axs[0, 1].set_title(Df_label)

    #
    # Place field stability k
    #

    x_vals = np.linspace(-np.pi, np.pi, 1000)

    WT_knots = WT_params['position_stability']['all_pairs']['knots']
    WT_spline = splines.CyclicSpline(WT_knots)
    WT_N = WT_spline.design_matrix(x_vals)

    WT_all_theta_k = WT_params['position_stability']['all_pairs'][
        'boots_theta_k']
    WT_skip_1_theta_k = WT_params['position_stability']['skip_one'][
        'boots_theta_k']
    WT_skip_npc_theta_k = WT_params['position_stability']['skip_one_npc'][
        'boots_theta_k']
    WT_two_iter_theta_k = WT_params['position_stability']['two_iter'][
        'boots_theta_k']

    WT_all_k_fit_mean = [1. / splines.get_k(theta_k, WT_N).mean() for
                         theta_k in WT_all_theta_k]
    WT_skip_1_k_fit_mean = [1. / splines.get_k(theta_k, WT_N).mean() for
                            theta_k in WT_skip_1_theta_k]
    WT_skip_npc_k_fit_mean = [1. / splines.get_k(theta_k, WT_N).mean() for
                              theta_k in WT_skip_npc_theta_k]
    WT_two_iter_k_fit_mean = [1. / splines.get_k(theta_k, WT_N).mean() for
                              theta_k in WT_two_iter_theta_k]

    WT_all_k_fit_df = pd.DataFrame(
        {'value': WT_all_k_fit_mean, 'genotype': WT_label})
    WT_skip_1_k_fit_df = pd.DataFrame(
        {'value': WT_skip_1_k_fit_mean, 'genotype': WT_label})
    WT_skip_npc_k_fit_df = pd.DataFrame(
        {'value': WT_skip_npc_k_fit_mean, 'genotype': WT_label})
    WT_two_iter_k_fit_df = pd.DataFrame(
        {'value': WT_two_iter_k_fit_mean, 'genotype': WT_label})

    Df_knots = Df_params['position_stability']['all_pairs']['knots']
    Df_spline = splines.CyclicSpline(Df_knots)
    Df_N = Df_spline.design_matrix(x_vals)

    Df_all_theta_k = Df_params['position_stability']['all_pairs'][
        'boots_theta_k']
    Df_skip_1_theta_k = Df_params['position_stability']['skip_one'][
        'boots_theta_k']
    Df_skip_npc_theta_k = Df_params['position_stability']['skip_one_npc'][
        'boots_theta_k']
    Df_two_iter_theta_k = Df_params['position_stability']['two_iter'][
        'boots_theta_k']

    Df_all_k_fit_mean = [1. / splines.get_k(theta_k, Df_N).mean() for
                         theta_k in Df_all_theta_k]
    Df_skip_1_k_fit_mean = [1. / splines.get_k(theta_k, Df_N).mean() for
                            theta_k in Df_skip_1_theta_k]
    Df_skip_npc_k_fit_mean = [1. / splines.get_k(theta_k, Df_N).mean() for
                              theta_k in Df_skip_npc_theta_k]
    Df_two_iter_k_fit_mean = [1. / splines.get_k(theta_b, Df_N).mean() for
                              theta_b in Df_two_iter_theta_k]

    Df_all_k_fit_df = pd.DataFrame(
        {'value': Df_all_k_fit_mean, 'genotype': Df_label})
    Df_skip_1_k_fit_df = pd.DataFrame(
        {'value': Df_skip_1_k_fit_mean, 'genotype': Df_label})
    Df_skip_npc_k_fit_df = pd.DataFrame(
        {'value': Df_skip_npc_k_fit_mean, 'genotype': Df_label})
    Df_two_iter_k_fit_df = pd.DataFrame(
        {'value': Df_two_iter_k_fit_mean, 'genotype': Df_label})

    all_k_df = pd.concat([WT_all_k_fit_df, Df_all_k_fit_df])
    skip_1_k_df = pd.concat([WT_skip_1_k_fit_df, Df_skip_1_k_fit_df])
    skip_npc_k_df = pd.concat([WT_skip_npc_k_fit_df, Df_skip_npc_k_fit_df])
    two_iter_k_df = pd.concat([WT_two_iter_k_fit_df, Df_two_iter_k_fit_df])

    order_dict = {WT_label: 0, Df_label: 1}
    all_k_df['order'] = all_k_df['genotype'].apply(order_dict.get)
    skip_1_k_df['order'] = skip_1_k_df['genotype'].apply(order_dict.get)
    skip_npc_k_df['order'] = skip_npc_k_df['genotype'].apply(order_dict.get)
    two_iter_k_df['order'] = two_iter_k_df['genotype'].apply(order_dict.get)

    plotting.plot_dataframe(
        axs[0, 2], [all_k_df, skip_1_k_df, skip_npc_k_df, two_iter_k_df],
        labels=['1 ses elapsed', '2 ses elapsed (all)', '2 ses elapsed (nPC)',
                '2 iterations (model)'], activity_label='Mean shift variance',
        colors=sns.color_palette('deep'), plotby=['genotype'], orderby='order',
        plot_method='grouped_bar', plot_shuffle=False, shuffle_plotby=False,
        pool_shuffle=False, agg_fn=np.mean, markers=None, label_groupby=True,
        z_score=False, normalize=False, error_bars='std')
    axs[0, 2].set_xlabel('')
    axs[0, 2].set_ylim(0, 1.5)
    y_ticks = np.array(['0', '0.01', '0.02', '0.03'])
    axs[0, 2].set_yticks(y_ticks.astype('float') * (2 * np.pi) ** 2)
    axs[0, 2].set_yticklabels(y_ticks)
    axs[0, 2].tick_params(top=False, bottom=False)
    axs[0, 2].set_title('')

    save_figure(fig, filename, save_dir=save_dir)

if __name__ == '__main__':
    main()
