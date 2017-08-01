"""Figure S4 - Additional place cell metrics"""

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

import lab
import lab.analysis.place_cell_analysis as place
import lab.misc as misc
import lab.plotting as plotting

import Df16a_analysis as df

# Colors
WT_color = df.WT_color
Df_color = df.Df_color
colors = (WT_color, Df_color)

WT_label = df.WT_label
Df_label = df.Df_label

linestyles = df.linestyles

# ROI filters
WT_filter = df.WT_filter
Df_filter = df.Df_filter
roi_filters = (WT_filter, Df_filter)

save_dir = df.fig_save_dir
filename = 'FigS4_additional_place_cell_metrics.{}'.format(FIG_FORMAT)


def main():
    all_grps = df.loadExptGrps('GOL')

    expt_grps = [all_grps['WT_place_set'], all_grps['Df_place_set']]

    pc_filters = [expt_grp.pcs_filter(roi_filter=roi_filter) for
                  expt_grp, roi_filter in zip(expt_grps, roi_filters)]

    data_to_save = {}

    fig = plt.figure(figsize=(8.5, 11))

    gs1 = plt.GridSpec(
        3, 3, left=0.1, right=0.9, top=0.9, bottom=0.3, wspace=0.5, hspace=0.4)
    sensitivity_cdf_ax = fig.add_subplot(gs1[1, 0])
    sensitivity_bar_ax = fig.add_axes([0.24, 0.535, 0.05, 0.07])

    specificity_cdf_ax = fig.add_subplot(gs1[1, 1])
    specificity_bar_ax = fig.add_axes([0.43, 0.60, 0.05, 0.07])

    width_cdf_ax = fig.add_subplot(gs1[2, 0])
    width_bar_ax = fig.add_axes([0.24, 0.315, 0.05, 0.07])

    sparsity_cdf_ax = fig.add_subplot(gs1[2, 1])
    sparsity_bar_ax = fig.add_axes([0.54, 0.315, 0.05, 0.07])

    is_ever_pc_fraction_ax = fig.add_subplot(gs1[0, 0])

    pc_fraction_ax = fig.add_subplot(gs1[0, 1])
    pc_fraction_bar_ax = fig.add_axes([0.54, 0.755, 0.05, 0.07])

    cdf_axs = [
        sensitivity_cdf_ax, specificity_cdf_ax, width_cdf_ax, sparsity_cdf_ax,
        pc_fraction_ax]
    bar_axs = [
        sensitivity_bar_ax, specificity_bar_ax, width_bar_ax, sparsity_bar_ax,
        pc_fraction_bar_ax]

    sensitivity_range = (0, 1)
    data_to_save['sensitivity'] = plotting.plot_metric(
        sensitivity_cdf_ax, expt_grps, metric_fn=place.sensitivity,
        groupby=[['roi_id', 'expt']], plotby=None, plot_method='cdf',
        roi_filters=pc_filters, activity_kwargs=None,
        activity_label='Transient sensitivity', colors=colors,
        rotate_labels=False, linestyles=linestyles)
    sensitivity_cdf_ax.set_xlim(sensitivity_range)
    data_to_save['sensitivity_by_mouse'] = plotting.plot_metric(
        sensitivity_bar_ax, expt_grps, metric_fn=place.sensitivity,
        groupby=[['roi_id', 'expt'],
                 ['roi_id', 'uniqueLocationKey', 'mouseID'], ['mouseID']],
        plotby=None, plot_method='grouped_bar',
        roi_filters=pc_filters, activity_kwargs=None,
        activity_label='Transient sensitivity', colors=colors)
    sensitivity_bar_ax.set_ylim(sensitivity_range)
    sensitivity_bar_ax.set_yticks(sensitivity_range)

    plotting.stackedText(
        sensitivity_cdf_ax, [WT_label, Df_label], colors=colors, loc=2,
        size=10)

    specificity_range = (0, 1)
    data_to_save['specificity'] = plotting.plot_metric(
        specificity_cdf_ax, expt_grps, metric_fn=place.specificity,
        groupby=[['roi_id', 'expt']], plotby=None, plot_method='cdf',
        roi_filters=pc_filters, activity_kwargs=None,
        activity_label='Transient specificity', colors=colors,
        rotate_labels=False, linestyles=linestyles)
    specificity_cdf_ax.set_xlim(specificity_range)
    data_to_save['specificity_by_mouse'] = plotting.plot_metric(
        specificity_bar_ax, expt_grps, metric_fn=place.specificity,
        groupby=[['roi_id', 'expt'],
                 ['roi_id', 'uniqueLocationKey', 'mouseID'], ['mouseID']],
        plotby=None, plot_method='grouped_bar',
        roi_filters=pc_filters, activity_kwargs=None,
        activity_label='Transient specificity', colors=colors)
    specificity_bar_ax.set_ylim(specificity_range)
    specificity_bar_ax.set_yticks(specificity_range)

    width_range = (0, 100)
    data_to_save['pf_width'] = plotting.plot_metric(
        width_cdf_ax, expt_grps, metric_fn=place.place_field_width,
        groupby=None, plotby=None, plot_method='cdf',
        roi_filters=pc_filters, activity_kwargs=None,
        activity_label='Place field width (cm)', colors=colors,
        rotate_labels=False, linestyles=linestyles)
    width_cdf_ax.set_xlim(width_range)
    data_to_save['pf_width_by_mouse'] = plotting.plot_metric(
        width_bar_ax, expt_grps, metric_fn=place.place_field_width,
        groupby=[['roi_id', 'uniqueLocationKey', 'mouseID'], ['mouseID']],
        plotby=None, plot_method='grouped_bar',
        roi_filters=pc_filters, activity_kwargs=None,
        activity_label='Place field width', colors=colors)
    width_bar_ax.set_ylim(width_range)
    width_bar_ax.set_yticks(width_range)

    sparsity_range = (0, 1)
    data_to_save['sparsity'] = plotting.plot_metric(
        sparsity_cdf_ax, expt_grps, metric_fn=place.sparsity,
        groupby=[['roi_id', 'expt']], plotby=None, plot_method='cdf',
        roi_filters=pc_filters, activity_kwargs=None,
        activity_label='Single-cell sparsity', colors=colors,
        rotate_labels=False, linestyles=linestyles)
    sparsity_cdf_ax.set_xlim(sparsity_range)
    data_to_save['sparsity_by_mouse'] = plotting.plot_metric(
        sparsity_bar_ax, expt_grps, metric_fn=place.sparsity,
        groupby=[['roi_id', 'expt'],
                 ['roi_id', 'uniqueLocationKey', 'mouseID'], ['mouseID']],
        plotby=None, plot_method='grouped_bar',
        roi_filters=pc_filters, activity_kwargs=None,
        activity_label='Single-cell sparsity', colors=colors)
    sparsity_bar_ax.set_ylim(sparsity_range)
    sparsity_bar_ax.set_yticks(sparsity_range)

    fraction_ses_pc_range = (0, 0.5)
    data_to_save['fraction_ses_pc'] = plotting.plot_metric(
        pc_fraction_ax, expt_grps, metric_fn=lab.ExperimentGroup.filtered_rois,
        groupby=(('mouseID', 'uniqueLocationKey', 'roi_id'),), plotby=None,
        colorby=None, plot_method='cdf', roi_filters=pc_filters,
        activity_kwargs=[
            {'include_roi_filter': roi_filter} for roi_filter in roi_filters],
        colors=colors, activity_label='Fraction of sessions a place cell',
        rotate_labels=False, linestyles=linestyles)
    pc_fraction_ax.tick_params(length=3, pad=2)
    pc_fraction_ax.get_legend().set_visible(False)
    pc_fraction_ax.set_title('')
    pc_fraction_ax.set_xticks([0, .2, .4, .6, .8])
    pc_fraction_ax.set_xlim(0, .8)
    pc_fraction_ax.spines['left'].set_linewidth(1)
    pc_fraction_ax.spines['bottom'].set_linewidth(1)

    data_to_save['fraction_ses_pc_by_mouse'] = plotting.plot_metric(
        pc_fraction_bar_ax, expt_grps,
        metric_fn=lab.ExperimentGroup.filtered_rois,
        groupby=[['roi_id', 'uniqueLocationKey', 'mouseID'], ['mouseID']],
        plotby=None, plot_method='grouped_bar',
        roi_filters=pc_filters, activity_kwargs=[
            {'include_roi_filter': roi_filter} for roi_filter in roi_filters],
        activity_label='Fraction of sessions a place cell', colors=colors)
    pc_fraction_bar_ax.set_ylim(fraction_ses_pc_range)
    pc_fraction_bar_ax.set_yticks(fraction_ses_pc_range)

    data_to_save['is_ever_pc'] = place.is_ever_place_cell(
        expt_grps, roi_filters=roi_filters, ax=is_ever_pc_fraction_ax,
        colors=colors, filter_fn=lambda df: df['session_number'] < 15,
        filter_columns=['session_number'],
        groupby=[['mouseID', 'session_number']])
    is_ever_pc_fraction_ax.get_legend().set_visible(False)
    is_ever_pc_fraction_ax.tick_params(length=3, pad=2)
    is_ever_pc_fraction_ax.set_xlabel('Session number')
    is_ever_pc_fraction_ax.set_title('Lifetime place coding')
    is_ever_pc_fraction_ax.set_xticklabels([
        '1', '', '3', '', '5', '', '7', '', '9', '', '11', '', '13', '', '15'])

    for ax in cdf_axs + bar_axs:
        ax.tick_params(length=3, pad=2)
    for ax in cdf_axs:
        ax.set_title('')
    for ax in bar_axs:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.set_xticks([])
        ax.tick_params(bottom=False, labelbottom=False, length=3, pad=2)
        ax.get_legend().set_visible(False)

    sns.despine(fig)

    misc.save_figure(fig, filename, save_dir=save_dir)

    plt.close('all')

if __name__ == '__main__':
    main()
