"""Figure 6 - WT parameter fits"""

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

import lab
import lab.misc as misc
import lab.misc.splines as splines
import lab.analysis.place_cell_analysis as place
import lab.plotting as plotting

import os
import sys
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..', 'enrichment_model'))
import enrichment_model_data as emd
import Df16a_analysis as df

WT_color = df.WT_color

WT_marker, _ = df.markers

save_dir = df.fig_save_dir
filename = 'Fig6_WT_parameters_fits.{}'.format(FIG_FORMAT)

params_path = os.path.join(
    df.data_path, 'enrichment_model', 'WT_model_params_C.pkl')


def plot_ROI_outlines(
        ax, expt, plane=0, channel='Ch2', label=None, roi_filter=None,
        **plot_kwargs):
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    roiVerts = expt.roiVertices(
        channel=channel, label=label, roi_filter=roi_filter)

    roi_inds = [i for i, v in enumerate(roiVerts) if v[0][0][2] == plane]

    plane_verts = [roiVerts[x] for x in roi_inds]
    roi_polys = []
    for roi in plane_verts:
        for poly in roi:
            roi_polys.append(poly[:, :2])

    for poly in roi_polys:
        ax.plot(poly[:, 0], poly[:, 1], **plot_kwargs)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)


def main():

    raw_data, data = emd.load_data(
        'wt', session_filter='C',
        root=os.path.join(df.data_path, 'enrichment_model'))
    expts = lab.ExperimentSet(
        os.path.join(df.metadata_path, 'expt_metadata.xml'),
        behaviorDataPath=os.path.join(df.data_path, 'behavior'),
        dataPath=os.path.join(df.data_path, 'imaging'))

    params = pickle.load(open(params_path))

    fig = plt.figure(figsize=(8.5, 11))
    gs1 = plt.GridSpec(
        1, 2, left=0.1, bottom=0.65, right=0.9, top=0.9, wspace=0.05)
    fov1_ax = fig.add_subplot(gs1[0, 0])
    fov2_ax = fig.add_subplot(gs1[0, 1])
    cmap_ax = fig.add_axes([0.49, 0.65, 0.02, 0.25])
    gs2 = plt.GridSpec(
        2, 2, left=0.1, bottom=0.3, right=0.5, top=0.6, wspace=0.5, hspace=0.5)
    recur_ax = fig.add_subplot(gs2[0, 0])
    shift_ax = fig.add_subplot(gs2[0, 1])
    shift_compare_ax = fig.add_subplot(gs2[1, 0])
    var_compare_ax = fig.add_subplot(gs2[1, 1])

    #
    # Tuning maps
    #

    e1 = expts.grabExpt('jz135', '2015-10-12-14h33m47s')
    e2 = expts.grabExpt('jz135', '2015-10-12-15h34m38s')

    cmap = mpl.colors.ListedColormap(sns.color_palette("husl", 256))

    for ax, expt in ((fov1_ax, e1), (fov2_ax, e2)):
        place.plot_spatial_tuning_overlay(
            ax, lab.classes.pcExperimentGroup([expt], imaging_label='soma'),
            labels_visible=False, alpha=0.9, lw=0.1, cmap=cmap)
        plot_ROI_outlines(
            ax, expt, channel='Ch2', label='soma', roi_filter=None, ls='-',
            color='k', lw=0.1)
        # Add a 50-um scale bar
        plotting.add_scalebar(
            ax=ax, matchx=False, matchy=False, sizey=0,
            sizex=50 / expt.imagingParameters()['micronsPerPixel']['XAxis'],
            bar_color='w', bar_thickness=3)

    fov1_ax.set_title('Session 1')
    fov2_ax.set_title('Session 2')

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient)).T
    cmap_ax.imshow(gradient, aspect='auto', cmap=cmap)
    sns.despine(ax=cmap_ax, top=True, left=True, right=True, bottom=True)
    cmap_ax.tick_params(
        left=False, labelleft=False, bottom=False, labelbottom=False)
    cmap_ax.set_ylabel('belt position')

    # Figure out the reward window width
    reward_poss, windows = [], []
    for expt in [e1, e2]:
        reward_poss.append(expt.rewardPositions(units='normalized')[0])
        track_length = expt[0].behaviorData()['trackLength']
        window = float(expt.get('operantSpatialWindow'))
        windows.append(window / track_length)
    reward_pos = np.mean(reward_poss)
    window = np.mean(windows)

    # Add reward zone
    cmap_ax.plot([0, 1], [reward_pos, reward_pos],
                 transform=cmap_ax.transAxes, color='k', ls=':')
    cmap_ax.plot([0, 1], [reward_pos + window, reward_pos + window],
                 transform=cmap_ax.transAxes, color='k', ls=':')
    cmap_ax.set_ylim(0, 256)

    #
    # Recurrence by position
    #

    recur_x_vals = np.linspace(-np.pi, np.pi, 1000)

    recur_data = emd.recurrence_by_position(data, method='cv')
    recur_knots = np.linspace(
        -np.pi, np.pi, params['position_recurrence']['n_knots'])
    recur_splines = splines.CyclicSpline(recur_knots)
    recur_n = recur_splines.design_matrix(recur_x_vals)

    recur_fit = splines.prob(
        params['position_recurrence']['theta'], recur_n)

    recur_boots_fits = [
        splines.prob(boot, recur_n) for boot in
        params['position_recurrence']['boots_theta']]
    recur_ci_up_fit = np.percentile(recur_boots_fits, 95, axis=0)
    recur_ci_low_fit = np.percentile(recur_boots_fits, 5, axis=0)

    recur_ax.plot(recur_x_vals, recur_fit, color=WT_color)
    recur_ax.fill_between(
        recur_x_vals, recur_ci_low_fit, recur_ci_up_fit,
        facecolor=WT_color, alpha=0.5)
    sns.regplot(
        recur_data[:, 0], recur_data[:, 1], ax=recur_ax,
        color=WT_color, y_jitter=0.2, fit_reg=False, scatter_kws={'s': 1},
        marker=WT_marker)
    recur_ax.axvline(ls='--', color='0.4', lw=0.5)
    recur_ax.set_xlim(-np.pi, np.pi)
    recur_ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    recur_ax.set_xticklabels(['-0.50', '-0.25', '0', '0.25', '0.50'])
    recur_ax.set_ylim(-0.3, 1.3)
    recur_ax.set_yticks([0, 0.5, 1])
    recur_ax.tick_params(length=3, pad=1, top=False)
    recur_ax.set_xlabel('Initial distance from reward (fraction of belt)')
    recur_ax.set_ylabel('Place cell recurrence probability')
    recur_ax.set_title('')
    recur_ax_2 = recur_ax.twinx()
    recur_ax_2.tick_params(length=3, pad=1, top=False)
    recur_ax_2.set_ylim(-0.3, 1.3)
    recur_ax_2.set_yticks([0, 1])
    recur_ax_2.set_yticklabels(['non-recur', 'recur'])

    #
    # Place field stability
    #

    shift_x_vals = np.linspace(-np.pi, np.pi, 1000)

    shift_knots = params['position_stability']['all_pairs']['knots']
    shift_spline = splines.CyclicSpline(shift_knots)
    shift_n = shift_spline.design_matrix(shift_x_vals)
    shift_theta_b = params['position_stability']['all_pairs']['theta_b']
    shift_b_fit = np.dot(shift_n, shift_theta_b)
    shift_theta_k = params['position_stability']['all_pairs']['theta_k']
    shift_k_fit = splines.get_k(shift_theta_k, shift_n)
    shift_fit_var = 1. / shift_k_fit

    shift_data = emd.paired_activity_centroid_distance_to_reward(data)
    shift_data = shift_data.dropna()
    shifts = shift_data['second'] - shift_data['first']
    shifts[shifts < -np.pi] += 2 * np.pi
    shifts[shifts >= np.pi] -= 2 * np.pi

    shift_ax.plot(shift_x_vals, shift_b_fit, color=WT_color)
    shift_ax.fill_between(
        shift_x_vals, shift_b_fit - shift_fit_var,
        shift_b_fit + shift_fit_var, facecolor=WT_color, alpha=0.5)
    sns.regplot(
        shift_data['first'], shifts, ax=shift_ax, color=WT_color,
        fit_reg=False, scatter_kws={'s': 1}, marker=WT_marker)

    shift_ax.axvline(ls='--', color='0.4', lw=0.5)
    shift_ax.axhline(ls='--', color='0.4', lw=0.5)
    shift_ax.plot([-np.pi, np.pi], [np.pi, -np.pi], color='g', ls=':', lw=2)
    shift_ax.tick_params(length=3, pad=1, top=False)
    shift_ax.set_xlabel('Initial distance from reward (fraction of belt)')
    shift_ax.set_ylabel(r'$\Delta$ position (fraction of belt)')
    shift_ax.set_xlim(-np.pi, np.pi)
    shift_ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    shift_ax.set_xticklabels(['-0.50', '-0.25', '0', '0.25', '0.50'])
    shift_ax.set_ylim(-np.pi, np.pi)
    shift_ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    shift_ax.set_yticklabels(['-0.50', '-0.25', '0', '0.25', '0.50'])
    shift_ax.set_title('')

    #
    # Stability by distance to reward
    #

    shift_x_vals = np.linspace(-np.pi, np.pi, 1000)

    shift_knots = params['position_stability']['all_pairs']['knots']
    shift_spline = splines.CyclicSpline(shift_knots)
    shift_n = shift_spline.design_matrix(shift_x_vals)

    shift_theta_b = params['position_stability']['all_pairs']['theta_b']
    shift_b_fit = np.dot(shift_n, shift_theta_b)
    shift_boots_b_fit = [
        np.dot(shift_n, boot) for boot in
        params['position_stability']['all_pairs']['boots_theta_b']]
    shift_b_ci_up_fit = np.percentile(shift_boots_b_fit, 95, axis=0)
    shift_b_ci_low_fit = np.percentile(shift_boots_b_fit, 5, axis=0)

    shift_compare_ax.plot(shift_x_vals, shift_b_fit, color=WT_color)
    shift_compare_ax.fill_between(
        shift_x_vals, shift_b_ci_low_fit, shift_b_ci_up_fit,
        facecolor=WT_color, alpha=0.5)

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
    shift_compare_ax.set_xlabel('Initial distance from reward (fraction of belt)')
    shift_compare_ax.set_ylabel(r'$\Delta$ position (fraction of belt)')

    shift_theta_k = params['position_stability']['all_pairs']['theta_k']
    shift_k_fit = splines.get_k(shift_theta_k, shift_n)
    shift_boots_k_fit = [
        splines.get_k(boot, shift_n) for boot in
        params['position_stability']['all_pairs']['boots_theta_k']]
    shift_k_ci_up_fit = np.percentile(shift_boots_k_fit, 95, axis=0)
    shift_k_ci_low_fit = np.percentile(shift_boots_k_fit, 5, axis=0)

    var_compare_ax.plot(shift_x_vals, 1. / shift_k_fit, color=WT_color)
    var_compare_ax.fill_between(
        shift_x_vals, 1. / shift_k_ci_low_fit, 1. / shift_k_ci_up_fit,
        facecolor=WT_color, alpha=0.5)

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

    misc.save_figure(fig, filename, save_dir=save_dir)

    plt.close('all')

if __name__ == '__main__':
    main()
