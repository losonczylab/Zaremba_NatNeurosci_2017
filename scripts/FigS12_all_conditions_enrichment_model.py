"""Figure S12 - All conditions enrichment model"""

FIG_FORMAT = 'svg'

import matplotlib as mpl
if FIG_FORMAT == 'svg':
    mpl.use('agg')
elif FIG_FORMAT == 'pdf':
    mpl.use('pdf')
elif FIG_FORMAT == 'interactive':
    mpl.use('TkAgg')
import matplotlib.pyplot as plt
import cPickle as pickle
import seaborn.apionly as sns

from lab.misc import save_figure
import lab.plotting as plotting

import os
import sys
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..', 'enrichment_model'))
import enrichment_model_plotting as emp
import Df16a_analysis as df

WT_color = df.WT_color
Df_color = df.Df_color
colors = (WT_color, Df_color)

save_dir = df.fig_save_dir
filename = 'FigS12_model_all_conditions.{}'.format(FIG_FORMAT)


def main():

    fig = plt.figure(figsize=(8.5, 11))

    gs1 = plt.GridSpec(
        2, 2, left=0.1, right=0.7, top=0.9, bottom=0.5, hspace=0.4, wspace=0.4)
    WT_enrich_ax = fig.add_subplot(gs1[0, 0])
    Df_enrich_ax = fig.add_subplot(gs1[0, 1])
    WT_final_dist_ax = fig.add_subplot(gs1[1, 0])
    Df_final_dist_ax = fig.add_subplot(gs1[1, 1])

    simulations_path_A = os.path.join(
        df.data_path, 'enrichment_model',
        'WT_Df_enrichment_model_simulation_A.pkl')
    simulations_path_B = os.path.join(
        df.data_path, 'enrichment_model',
        'WT_Df_enrichment_model_simulation_B.pkl')
    simulations_path_C = os.path.join(
        df.data_path, 'enrichment_model',
        'WT_Df_enrichment_model_simulation_C.pkl')

    m_A = pickle.load(open(simulations_path_A))
    m_B = pickle.load(open(simulations_path_B))
    m_C = pickle.load(open(simulations_path_C))

    WT_colors = sns.light_palette(WT_color, 7)[2::2]
    Df_colors = sns.light_palette(Df_color, 7)[2::2]

    condition_labels = [
        r'Condition $\mathrm{I}$',
        r'Condition $\mathrm{II}$',
        r'Condition $\mathrm{III}$']

    WT_final_dists, Df_final_dists = [], []

    for m, WT_c, Df_c in zip((m_A, m_B, m_C), WT_colors, Df_colors):

        WT_enrich = emp.calc_enrichment(
            m['WT_no_swap_pos'], m['WT_no_swap_masks'])
        Df_enrich = emp.calc_enrichment(
            m['Df_no_swap_pos'], m['Df_no_swap_masks'])

        WT_final_dists.append(emp.calc_final_distributions(
            m['WT_no_swap_pos'], m['WT_no_swap_masks']))
        Df_final_dists.append(emp.calc_final_distributions(
            m['Df_no_swap_pos'], m['Df_no_swap_masks']))

        emp.plot_enrichment(
            WT_enrich_ax, WT_enrich, WT_c, title='', rad=False)
        emp.plot_enrichment(
            Df_enrich_ax, Df_enrich, Df_c, title='', rad=False)

    WT_enrich_ax.set_xlabel("Iteration ('session' #)")
    Df_enrich_ax.set_xlabel("Iteration ('session' #)")
    plotting.stackedText(
        WT_enrich_ax, condition_labels, colors=WT_colors, loc=2, size=8)
    plotting.stackedText(
        Df_enrich_ax, condition_labels, colors=Df_colors, loc=2, size=8)

    emp.plot_final_distributions(
        WT_final_dist_ax, WT_final_dists,
        WT_colors, labels=condition_labels, title='', rad=False)
    emp.plot_final_distributions(
        Df_final_dist_ax, Df_final_dists,
        Df_colors, labels=condition_labels, title='', rad=False)

    WT_final_dist_ax.set_xlabel('Distance from reward\n(fraction of belt)')
    Df_final_dist_ax.set_xlabel('Distance from reward\n(fraction of belt)')
    plotting.stackedText(
        WT_final_dist_ax, condition_labels, colors=WT_colors, loc=2, size=8)
    plotting.stackedText(
        Df_final_dist_ax, condition_labels, colors=Df_colors, loc=2, size=8)
    WT_final_dist_ax.set_yticks([0, 0.1, 0.2, 0.3])
    Df_final_dist_ax.set_yticks([0, 0.1, 0.2, 0.3])

    save_figure(fig, filename, save_dir=save_dir)

if __name__ == '__main__':
    main()
