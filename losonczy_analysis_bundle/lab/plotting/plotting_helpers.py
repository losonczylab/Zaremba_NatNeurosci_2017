"""Plotting helper functions"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox
from matplotlib.offsetbox import VPacker, HPacker, TextArea
import itertools as it
import pandas as pd
import datetime
from operator import itemgetter, methodcaller
from collections import defaultdict
from random import shuffle
import seaborn.apionly as sns
import copy

import lab
from ..classes.exceptions import InvalidDataFrame
# from lab.analysis import behaviorAnalysis as ba


def infer_expt_pair_label(expt1, expt2):

    same_belt = expt1.get('belt') == expt2.get('belt')
    matched_belts = expt1.get('belt')[:-1] == expt2.get('belt')[:-1] \
        and expt1.get('belt')[-1] in ['A', 'B'] \
        and expt2.get('belt')[-1] in ['A', 'B']
    same_context = expt1.get('environment') == expt2.get('environment')
    same_rewards = str(expt1.rewardPositions(units=None)) == \
        str(expt2.rewardPositions(units=None))

    if same_belt and same_context and same_rewards:
        return "SameAll"
    if same_belt and same_context and not same_rewards:
        return "SameAll_DiffRewards"
    if same_belt and not same_context and same_rewards:
        return "DiffCtxs"
    if not same_belt and not matched_belts and same_context and same_rewards:
        return "DiffBelts"
    if matched_belts and not same_context:
        return "SimBelts_DiffCtxs"
    if not same_belt and not same_context:
        return 'DiffAll'
    return None


# Adapted from mpl_toolkits.axes_grid2
# LICENSE: Python Software Foundation (http://docs.python.org/license.html)
# AnchoredScaleBar and add_scalebar from:
# https://gist.github.com/dmeliza/3251476
class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None,
                 loc=4, pad=0.1, borderpad=0.1, sep=2, prop=None,
                 bar_thickness=0, bar_color='k', **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).

        - transform : the coordinate frame (typically axes.transData)
        - sizex, sizey : width of x,y bar, in data units. 0 to omit
        - labelx, labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size
            (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(Rectangle((0, 0), sizex, bar_thickness,
                            color=bar_color))
        if sizey:
            bars.add_artist(Rectangle((0, 0), bar_thickness, sizey,
                            color=bar_color))

        if sizex and labelx:
            bars = VPacker(children=[bars, TextArea(labelx, minimumdescent=False)],
                           align="center", pad=0, sep=sep)
        if sizey and labely:
            bars = HPacker(children=[TextArea(labely), bars],
                           align="center", pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False,
                                   **kwargs)


def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True,
                 **kwargs):
    """ Add scalebars to axes

    Adds a set of scale bars to *ax*, matching the size to the ticks of the
    plot and optionally hiding the x and y axes

    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars

    Returns created scalebar object
    """
    def f(axis):
        l = axis.get_majorticklocs()
        return len(l) > 1 and (l[1] - l[0])

    if matchx:
        kwargs['sizex'] = f(ax.xaxis)
        kwargs['labelx'] = str(kwargs['sizex'])
    if matchy:
        kwargs['sizey'] = f(ax.yaxis)
        kwargs['labely'] = str(kwargs['sizey'])

    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)

    if hidex:
        ax.xaxis.set_visible(False)
    if hidey:
        ax.yaxis.set_visible(False)

    return sb


def right_label(ax, label, **kwargs):
    """Add a label to the right of an axes"""
    Bbox = ax.figbox
    ax.figure.text(
        Bbox.p1[0] + 0.02, (Bbox.p1[1] + Bbox.p0[1]) / 2, label, **kwargs)


def stackedText(ax, textList, colors=None, loc=1, size=None):
    """Add vertically stacked rows of text to the plot, for example to indicate
    a color scheme.
    Inputs:
        ax: the axis of the plot
        textList: a list of strings, each of which forms a row of text
        colors: a list of (r, g, b) tuples for the color of each text row.
            Defaults to black
        loc: the location of the text (following matplotlib standards)
        size: text size, if None, uses default font size
    """
    if colors is None:
        colors = [(0, 0, 0) for t in textList]
    textAreas = [
        TextArea(t, textprops=dict(color=c, size=size))
        for t, c in zip(textList, colors)]
    txt = VPacker(children=textAreas, pad=0, sep=2, align='baseline')
    anchored_box = AnchoredOffsetbox(
        loc=loc, child=txt, pad=0.5, frameon=False,
        bbox_transform=ax.transAxes, borderpad=0.)
    ax.add_artist(anchored_box)


def layout_subplots(
        n_plots, rows, cols, polar=False, sharex=True, sharey=False,
        figsize=(15, 8), rasterized=False):
    """Used to layout subplots in to multiple figures.
    Returns a list of figures, a flattened list of axs, a list of the bottom-
    left axis on each figure to label.

    """

    nFigs = int(np.ceil(n_plots / float(rows * cols)))

    figs = []
    axs = []
    axs_to_label = []
    for f in xrange(nFigs):
        if polar:
            fig, ax = plt.subplots(
                rows, cols, figsize=figsize, squeeze=False,
                subplot_kw={'polar': True, 'rasterized': rasterized})
        else:
            fig, ax = plt.subplots(
                rows, cols, sharex=sharex, sharey=sharey, figsize=figsize,
                squeeze=False, subplot_kw={'rasterized': rasterized})
        figs.append(fig)

        axs_to_label.append(ax[-1, 0])
        axs = np.hstack([axs, ax.flatten()])

    n_extras = int((nFigs * rows * cols) - n_plots)
    if n_extras > 0:
        extra_axs = ax.flatten()[-n_extras:]
        for a in extra_axs:
            a.set_visible(False)
            try:
                axs_to_label.remove(a)
            except:
                pass
        # Find the last ax in the first column on the last fig and add it to
        # axs_to_label
        removed_rows = int(n_extras) / int(cols)
        ax_to_add = ax[rows - 1 - removed_rows, 0]
        if ax_to_add not in axs_to_label:
            axs_to_label.append(ax_to_add)
        # Remove extra axs
        axs = np.array([a for a in axs if a not in extra_axs])

    return figs, axs, axs_to_label


def binTimes(times, combine_threshold=np.timedelta64(12, 'h'),
             returnTimes=False):
    """Takes a list of times and returns a group index for each time.

    Keyword arguments:
    combine_threshold -- a timedelta object corresponding to the maximum difference between times that should be combined

    """

    timeBins = np.array([-1] * len(times))
    for time_idx, time in enumerate(times):
        found = False
        for bin_idx in range(np.max(timeBins) + 1):
            binned_time = np.mean(times[timeBins == bin_idx])
            if (np.abs(binned_time - time) <= combine_threshold):
                timeBins[time_idx] = bin_idx
                found_bin = bin_idx
                found = True
                break
        if not found:
            timeBins[time_idx] = np.max(timeBins) + 1
            found_bin = timeBins[time_idx]
        # Check to see if any bins should be combined into 1
        found_binned_time = np.mean(times[timeBins == found_bin])
        for bin_idx in range(np.max(timeBins) + 1):
            if bin_idx != found_bin:
                binned_time = np.mean(times[timeBins == bin_idx])
                if np.abs(binned_time - found_binned_time) \
                        <= combine_threshold:
                    timeBins[timeBins == found_bin] = bin_idx
                    # Shift all time bins down one to replace the now empty bin
                    timeBins[timeBins > found_bin] -= 1
                    break
    if returnTimes:
        bin_means = [np.mean(times[timeBins == bin_idx]) for
                     bin_idx in sorted(np.unique(timeBins))]
        new_bins = []
        for t in timeBins:
            new_bins.append(bin_means[t])
        return np.array(new_bins).astype(np.timedelta64)
    return timeBins.astype(np.timedelta64)


def prepare_dataframe(dataframe, include_columns=None):
    """Adds calculatable columns to a DataFrame. Used to prep the DataFrame
    for further analysis. Always adds 'expt' and 'mouse' columns if not already
    there.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The pandas DataFrame to prep.
    include_columns : iterable
        Additional columns that should be included in the dataframe.

    Returns
    -------
    pandas.DataFrame
        The same dataframe as was passed in, with additional columns added.

    """

    if include_columns is None:
        return dataframe

    def check_column(column, target):
        return column in set(pre + target for pre in ('', 'first_', 'second_'))

    def all_expts():
        expt_set = set()
        for col in ['expt', 'first_expt', 'second_expt']:
            if col in dataframe.columns:
                expt_set = expt_set.union(dataframe[col])
        return expt_set

    # Add 'expt' and 'mouse' columns if possible
    for prefix in ['first_', 'second_', '']:
        if prefix + 'trial' in dataframe.columns and \
                prefix + 'expt' not in dataframe.columns:
            dataframe[prefix + 'expt'] = dataframe[prefix + 'trial'].apply(
                lambda trial: trial.parent)
        if prefix + 'expt' in dataframe.columns and \
                prefix + 'mouse' not in dataframe.columns:
            dataframe[prefix + 'mouse'] = dataframe[prefix + 'expt'].apply(
                lambda expt: expt.parent)

    for prefix in ('first', 'second'):
        for col in ('roi', 'expt', 'mouse'):
            test_col = prefix + '_' + col
            if test_col in include_columns and \
                    test_col not in dataframe.columns:
                dataframe[test_col] = dataframe[col]

    for column in include_columns:

        if column in dataframe.columns:
            continue

        # Should be careful with these, the if/else was added to be able to
        # treat statistics not based on paired experiments as if they were
        for prefix in ('first_', 'second_', ''):
            if prefix in column:
                mouse_col = prefix + 'mouse' if prefix + 'mouse' in \
                    dataframe.columns else 'mouse'
                expt_col = prefix + 'expt' if prefix + 'expt' in \
                    dataframe.columns else 'expt'
                roi_col = prefix + 'roi' if prefix + 'roi' in \
                    dataframe.columns else 'roi'
                break

        if column.startswith('X_'):
            column_trimmed = column.replace(
                'first_', '').replace('second_', '').replace('X_', '')
            dataframe[column] = dataframe[expt_col].apply(
                methodcaller('get', column_trimmed))
        elif check_column(column, 'mouse'):
            dataframe[column] = dataframe[expt_col].apply(
                lambda expt: expt.parent)
        elif check_column(column, 'mouseID'):
            try:
                dataframe[column] = dataframe[expt_col].apply(
                    lambda expt: expt.parent.get('mouseID'))
            except KeyError:
                dataframe[column] = dataframe[mouse_col].apply(
                    lambda mouse: mouse.get('mouseID'))
        elif check_column(column, 'uniqueLocationKey') or \
                check_column(column, 'location'):
            dataframe[column] = dataframe[expt_col].apply(
                lambda expt: expt.get('uniqueLocationKey'))
        elif column == 'startTime':
            dataframe['startTime'] = dataframe['expt'].apply(
                lambda expt: expt.get('startTime'))
        elif check_column(column, 'roi_id'):
            dataframe[column] = dataframe[roi_col].apply(
                lambda roi: roi.id)
        elif check_column(column, 'roi_tuple'):
            if 'first' in column:
                prefix = 'first_'
            elif 'second' in column:
                prefix = 'second_'
            else:
                prefix = ''
            prepare_dataframe(
                dataframe, include_columns=[
                    prefix + col for col in
                    ['mouseID', 'uniqueLocationKey', 'roi_id']])
            dataframe[column] = zip(
                dataframe[prefix + 'mouseID'],
                dataframe[prefix + 'uniqueLocationKey'],
                dataframe[prefix + 'roi_id'])
        elif check_column(column, 'exposure'):
            exptGrp = lab.ExperimentGroup(set(dataframe[expt_col]))
            exposure = exptGrp.priorDaysOfExposure()
            dataframe[column] = dataframe[expt_col].apply(
                lambda expt: exposure[expt])
        elif column == 'time_diff':
            dataframe['time_diff'] = dataframe['second_expt'] \
                - dataframe['first_expt']
        elif column == 'time_diff_d':
            dataframe['time_diff'] = dataframe['second_expt'] \
                - dataframe['first_expt']
            dataframe['time_diff_d'] = dataframe['time_diff'] \
                / np.timedelta64(1, 'D')
        elif check_column(column, 'belt_id'):
            dataframe[column] = dataframe[expt_col].apply(
                lambda expt: expt.belt().get('beltID'))
        elif column == 'elapsed_days':
            dataframe['time_diff'] = dataframe['second_expt'] - \
                dataframe['first_expt']
            time_bins = binTimes(
                np.array(dataframe['time_diff']), returnTimes=True)
            time_bins_d = time_bins / np.timedelta64(1, 'D')
            dataframe['elapsed_days'] = time_bins_d
        elif column == 'elapsed_days_int':
            dataframe['time_diff'] = dataframe['second_expt'] - \
                dataframe['first_expt']
            time_bins = binTimes(
                np.array(dataframe['time_diff']), returnTimes=True)
            time_bins_d = time_bins / np.timedelta64(1, 'D')
            dataframe['elapsed_days_int'] = np.around(time_bins_d)
        # elif 'condition_label' in column:
        #     exptGrp = analysis.HiddenRewardExperimentGroup(set(dataframe[expt_col]))
        #     condition_label, _ = exptGrp.condition_label()
        #     dataframe[column] = dataframe[expt_col].apply(
        #         lambda expt: condition_label[expt])
        elif check_column(column, 'condition_session'):
            # Labels each experiment with a condition label and overall
            # session number in that condition
            # exptGrp = lab.classes.HiddenRewardExperimentGroup(set(dataframe[expt_col]))
            exptGrp = lab.classes.HiddenRewardExperimentGroup(all_expts())
            condition_label, _ = exptGrp.condition_label(by_mouse=True)
            session_number = exptGrp.session_number(
                min_duration=datetime.timedelta(minutes=5))
            dataframe[column] = dataframe[expt_col].apply(
                lambda expt: '{}{}'.format(
                    condition_label[expt], session_number[expt]))
        elif check_column(column, 'condition_day_session'):
            # Labels each experiment with a condition label, days of
            # exposure to that condition, and session number of the day
            # exptGrp = lab.classes.HiddenRewardExperimentGroup(set(dataframe[expt_col]))
            exptGrp = lab.classes.HiddenRewardExperimentGroup(all_expts())
            condition_label, _ = exptGrp.condition_label(by_mouse=True)
            days = exptGrp.priorDaysOfExposure()
            session_number = exptGrp.session_number(
                min_duration=datetime.timedelta(minutes=8), per_day=True)
            dataframe[column] = dataframe[expt_col].apply(
                lambda expt: '{}_{}_{}'.format(
                    condition_label[expt], days[expt], session_number[expt]))
        elif check_column(column, 'condition_day'):
            # Labels each experiment with a condition label and days of
            # exposure to that condition
            # exptGrp = lab.classes.HiddenRewardExperimentGroup(set(dataframe[expt_col]))
            exptGrp = lab.classes.HiddenRewardExperimentGroup(all_expts())
            condition_label, _ = exptGrp.condition_label(by_mouse=True)
            days = exptGrp.priorDaysOfExposure()
            dataframe[column] = dataframe[expt_col].apply(
                lambda expt: '{}_{}'.format(condition_label[expt], days[expt]))
        elif check_column(column, 'condition'):
            # exptGrp = lab.classes.HiddenRewardExperimentGroup(set(dataframe[expt_col]))
            exptGrp = lab.classes.HiddenRewardExperimentGroup(all_expts())
            condition_label, _ = exptGrp.condition_label(by_mouse=True)
            dataframe[column] = dataframe[expt_col].apply(
                lambda expt: condition_label[expt])
        elif check_column(column, 'rewardPositions'):
            dataframe[column] = dataframe[expt_col].apply(
                lambda expt: expt.rewardPositions(units=None))
        elif check_column(column, 'session_number_in_df'):
            # Labels each experiment with the overall session number within
            # the experiments in the dataframe
            # exptGrp = lab.classes.ExperimentGroup(set(dataframe[expt_col]))
            exptGrp = lab.classes.ExperimentGroup(all_expts())
            session_number = exptGrp.session_number(
                ignoreBelt=True, ignoreContext=True,
                ignoreRewardPositions=True, number_in_group=True)
            dataframe[column] = dataframe[expt_col].apply(
                lambda expt: session_number[expt])
        elif check_column(column, 'day_in_df'):
            # Labels each experiment with the day within the experiments in the
            # dataframe
            # exptGrp = lab.classes.ExperimentGroup(set(dataframe[expt_col]))
            exptGrp = lab.classes.ExperimentGroup(all_expts())
            day_number = exptGrp.priorDaysOfExposure(
                ignoreBelt=True, ignoreContext=True,
                ignoreRewardPositions=True, number_in_group=True)
            dataframe[column] = dataframe[expt_col].apply(
                lambda expt: day_number[expt])
        elif check_column(column, 'day_session'):
            # exptGrp = lab.classes.ExperimentGroup(set(dataframe[expt_col]))
            exptGrp = lab.classes.ExperimentGroup(all_expts())
            day_number = exptGrp.priorDaysOfExposure()
            session_number = exptGrp.session_number(
                ignoreBelt=False, ignoreContext=False,
                min_duration=datetime.timedelta(minutes=8), per_day=True)
            dataframe[column] = dataframe[expt_col].apply(
                lambda expt: '{}_{}'.format(
                    day_number[expt], session_number[expt]))
        elif check_column(column, 'session_in_day'):
            # Labels each experiment with the session number of the day
            exptGrp = lab.classes.HiddenRewardExperimentGroup(
                set(dataframe[expt_col]))
            session_number = exptGrp.session_number(
                min_duration=datetime.timedelta(minutes=8), per_day=True)
            dataframe[column] = dataframe[expt_col].apply(
                lambda expt: int(session_number[expt]))
        elif check_column(column, 'session'):
            # Labels each experiment with the session number within the
            # condition/context
            # exptGrp = lab.classes.HiddenRewardExperimentGroup(
            #     set(dataframe[expt_col]))
            exptGrp = lab.classes.HiddenRewardExperimentGroup(all_expts())
            session_number = exptGrp.session_number(
                min_duration=datetime.timedelta(minutes=5))
            dataframe[column] = dataframe[expt_col].apply(
                lambda expt: session_number[expt])
        elif check_column(column, 'expt_pair_label'):
            # Data should be from paired expts, calculates the pair label,
            # i.e. SameAll, DiffCtxs, etc.
            def infer_label(df_row):
                return infer_expt_pair_label(
                    df_row['first_expt'], df_row['second_expt'])
            dataframe[column] = dataframe.apply(infer_label, axis=1)
        elif check_column(column, 'expt_pair_conditions'):
            # Data should be from paired expts, calculates the conditions,
            # i.e. A-A, A-B, B-B, etc.
            prepare_dataframe(
                dataframe, ['first_condition', 'second_condition'])

            def infer_conditions(df_row):
                labels = sorted(
                    [df_row['first_condition'], df_row['second_condition']])
                return '{}-{}'.format(*labels)

            dataframe[column] = dataframe.apply(infer_conditions, axis=1)
        elif check_column(column, 'ctx_from_tSeriesDir'):
            dataframe[column] = dataframe[expt_col].apply(
                lambda x: 'A' if 'ctxA' in x.get('tSeriesDirectory') else 'B')
        else:
            raise InvalidDataFrame('Unrecognized column name: ' + str(column))

    return dataframe


def plot_dataframe(
        ax, dataframes, shuffles=None, labels=None, plot_method='hist',
        groupby=None, plotby=None, orderby=None, colorby=None,
        activity_label='activity', filter_fn=None, filter_columns=None,
        plot_shuffle=False, shuffle_plotby=False, pool_shuffle=False,
        agg_fn=np.mean, colors=None, markers=None, label_groupby=True,
        z_score=False, normalize=False, plot_column='value', break_at=None,
        linestyles=None, **plot_kwargs):
    """A generic plotting function for plotting pandas.DataFrame's.

    Parameters
    ----------
    ax : matplotlib.axes
        Axis to plot on.
    dataframes : list of pd.DataFrame
        Dataframes to plot.
    shuffles : list of pd.Dataframe, optional
        Matching list of shuffled data.
    labels : list of str, optional
        List of labels, 1 per dataframe.
    plot_method : str
        The plotting method to be used. Note that not all methods
        support all arguments ('hist' must have plotby=None, for example).
    groupby : list of lists of str, optional
        Columns to sequentially group and then aggregate by.
    plotby : list of str, optional
        Columns in the dataframe to group and plot the data by.
    orderby : str, optional
        Dataframe column used to sort the x-values. If None, just sorts by the
        plotby value. Only works when also passing in a plotby argument,
        though this is not currently checked.
    colorby : list of str, optional
        A groupby list (e.g. ['mouseID', 'location']) used with 'scatter_bar'
        plotting method  for coloring the scatter points.
    activity_label : str, optional
        Label for the metric being plotted.
    filter_fn : function, optional
        Function used to filter rows of the dataframes. Should take a dataframe
        as the only argument and return a boolean array or pandas.Series of
        length equal to the number of rows. For example,
        lambda df: df['value'] >= 0, would filter out all values less than 0.
    filter_columns : list of str, optional
        List of columns to ensure are in the dataframes before filtering.
        Necessary since we can't inspect the function to know which columns are
        needed.
    plot_shuffle : bool
        If True, plot shuffle data along with the actual data.
    shuffle_plotby : bool
        If True, apply plotby to shuffle, otherwise aggregates all data.
    pool_shuffle : bool
        If True, pool all shuffle data across dataframes in to a single
        distribution, otherwise plot each dataframe's shuffle separately.
    agg_fn : function
        Function used to aggregate the data after grouping.
    colors : iterable of colors, optional
    markers : list of str, optional
        Only used for 'line' plot method. List of marker
        types passed to ax.plot, 1 element per dataframe.
    label_groupby bool
        If False, doesn't print the groupby in the axis title.
    z_score : bool
        If True, z-score data before plotting.
    normalize : bool
        If True, normalize data to the mean.
    plot_column : str
        The label of the column in the dataframe containing the data to plot.
        Defaults to 'value'.
    break_at : list, optional
        Only used for 'line' plot method. Adds breaks in all lines at each
        x-value in the list.
    **plot_kwargs -- any additional keyword arguments will be pass to the
        specific plotting functions. Note that each plotting method will have
        different allowable plot arguments

    """
    if shuffles is None:
        shuffles = [None] * len(dataframes)
    assert len(dataframes) == len(shuffles)
    if labels is not None:
        assert len(dataframes) == len(labels)

    if colors is None:
        colors = color_cycle()
        # colormaps are used by cdf plotting when 'plotby' is not None
        colormaps = [plt.cm.Reds, plt.cm.Blues, plt.cm.Greens]
    else:
        # colormaps = (sns.dark_palette(color, as_cmap=True) for color in colors)
        colormaps = (sns.light_palette(color, as_cmap=True) for color in colors)
        colors = iter(colors)

    if groupby is None or not label_groupby:
        groupby_label = ''
    else:
        groupby_label = ' by ' + str(groupby)

    if plotby is None:
        plotby_label = ''
    else:
        plotby_label = ' plot by ' + str(plotby)

    if labels is None:
        labels = ['Group ' + str(x) for x in range(len(dataframes))]

    if colorby is not None:
        colorby_dict = {}

    # Allow using a different aggregation method for each groupby
    if groupby is not None:
        try:
            iter(agg_fn)
        except TypeError:
            # Not iterable, make it iterable.
            agg_fn = [agg_fn] * len(groupby)

    try:
        iter(filter_fn)
    except TypeError:
        filter_fn = [filter_fn] * len(dataframes)
        filter_columns = [filter_columns] * len(dataframes)

    for idx, dataframe in it.izip(it.count(), dataframes):
        # Drop all Nan's
        try:
            dataframe.dropna(subset=[plot_column], inplace=True)
        except KeyError:
            raise InvalidDataFrame(
                'DataFrame missing \'{}\' column'.format(plot_column))
        if filter_fn[idx] is not None:
            prepare_dataframe(dataframe, include_columns=filter_columns[idx])
            dataframe = dataframe[filter_fn[idx](dataframe)]
        if groupby is not None and len(dataframe):
            for groupby_list, agg in it.izip(groupby, agg_fn):
                # NOTE: What is this supposed to be catching?
                # HAXXX
                if groupby_list == []:
                    groupby_list = ['expt']
                prepare_dataframe(dataframe, include_columns=groupby_list)
                dataframe = dataframe.groupby(
                    groupby_list, as_index=False).agg(agg)
        if colorby is not None:
            prepare_dataframe(dataframe, include_columns=colorby)
            colorby_colors = iter(sns.color_palette(
                'bright', n_colors=len(dataframe.groupby(colorby))))
            for key, group in dataframe.groupby(colorby):
                colorby_dict[key] = next(colorby_colors)
        if normalize:
            dataframe[plot_column] /= dataframe[plot_column].mean()
            activity_label += ' (normalized)'
        if z_score:
            dataframe[plot_column] -= dataframe[plot_column].mean()
            dataframe[plot_column] /= dataframe[plot_column].std()
            activity_label += ' (z-scored)'
        prepare_dataframe(dataframe, include_columns=plotby)
        dataframes[idx] = dataframe

    if plot_shuffle:
        for idx, shuffle_df in it.izip(it.count(), shuffles):
            try:
                shuffle_df.dropna(subset=[plot_column], inplace=True)
            except KeyError:
                raise InvalidDataFrame(
                    'Shuffle DataFrame missing \'value\' column')
            if filter_fn[idx] is not None:
                prepare_dataframe(
                    shuffle_df, include_columns=filter_columns[idx])
                shuffle_df = shuffle_df[filter_fn[idx](shuffle_df)]
            if groupby is not None:
                for groupby_list, agg in it.izip(groupby, agg_fn):
                    prepare_dataframe(shuffle_df, include_columns=groupby_list)
                    shuffle_df = shuffle_df.groupby(
                        groupby_list, as_index=False).agg(agg)
            prepare_dataframe(shuffle_df, include_columns=plotby)
            shuffles[idx] = shuffle_df

    if plot_shuffle and pool_shuffle:
        pooled_shuffle = pd.concat(shuffles)

    if plot_method in ['hist', 'kde']:
        if plotby is not None:
            raise ValueError(
                'Unable to plot by a column when using histograms')
        kwargs = copy.copy(plot_kwargs)
        if plot_method == 'hist':
            if plot_kwargs.get('uniform_bins') or \
                    plot_kwargs.get('unit_width'):
                data_min, data_max = np.inf, -np.inf
                for dataframe, shuffle_df in it.izip(dataframes, shuffles):
                    data_min = min(data_min, dataframe[plot_column].min())
                    data_max = max(data_max, dataframe[plot_column].max())
                    if plot_shuffle:
                        data_min = min(data_min, shuffle_df[plot_column].min())
                        data_max = max(data_min, shuffle_df[plot_column].max())
                if plot_kwargs.get('uniform_bins'):
                    kwargs['range'] = (data_min, data_max)
                else:
                    kwargs['range'] = None
                if plot_kwargs.get('unit_width'):
                    kwargs['bins'] = np.arange(
                        int(data_min), int(data_max) + 2)
                else:
                    kwargs['bins'] = plot_kwargs.get('bins', 10)
                kwargs.pop('uniform_bins', None)
                kwargs.pop('unit_width', None)
            else:
                kwargs['range'] = plot_kwargs.get('range', None)
                kwargs['bins'] = plot_kwargs.get('bins', 10)

            # Set a few more defaults
            kwargs['normed'] = kwargs.get('normed', True)
            kwargs['plot_mean'] = kwargs.get('plot_mean', True)

        for dataframe, shuffle_df, label, ls in it.izip(
                dataframes, shuffles, labels, linestyles):
            color = colors.next()
            if plot_method == 'hist':
                plotting.histogram(
                    ax, np.array(dataframe[plot_column]), color=color,
                    label=label, linestyle=ls, **kwargs)

                if plot_shuffle and not pool_shuffle:
                    plotting.histogram(
                        ax, shuffle_df[plot_column], color=color,
                        label=label + '_shuffle', hatch='//', linestyle=ls,
                        **kwargs)
            elif plot_method == 'kde':
                sns.kdeplot(
                    ax=ax, data=dataframe[plot_column], color=color,
                    label=label, linestyle=ls, **kwargs)

                if plot_shuffle and not pool_shuffle:
                    sns.kdeplot(
                        ax=ax, data=shuffle_df[plot_column], color=color,
                        label=label + '_shuffle', linestyle=ls, **kwargs)

        if plot_shuffle and pool_shuffle:
            if plot_method == 'hist':
                plotting.histogram(
                    ax, np.array(pooled_shuffle[plot_column]), color='k',
                    label='shuffle', hatch='//', **kwargs)
            elif plot_method == 'kde':
                sns.kdeplot(
                    ax=ax, data=np.array(pooled_shuffle[plot_column]),
                    color='k', label='shuffle', **kwargs)

        if plot_kwargs.get('normed', True):
            ax.set_ylabel('Normalized density')
        else:
            ax.set_ylabel('Number')

        ax.set_xlabel(activity_label)
        ax.set_title(activity_label + groupby_label)
        ax.legend(frameon=False, loc='best')

    elif plot_method in ['cdf', 'line-o-gram']:
        if linestyles is None:
            linestyles = ['-'] * len(dataframes)
        for dataframe, shuffle_df, label, cmap, ls in it.izip(
                dataframes, shuffles, labels, colormaps, linestyles):
            if plotby is None:
                color = colors.next()
                if plot_method == 'cdf':
                    plotting.cdf(
                        ax=ax, values=dataframe[plot_column], bins='exact',
                        label=label, linestyle=ls, color=color, **plot_kwargs)
                else:
                    plotting.line_o_gram(
                        ax=ax, values=dataframe[plot_column], linestyle=ls,
                        label=label, color=color, **plot_kwargs)
            else:
                if orderby is not None:
                    raise NotImplementedError
                grouped = dataframe.groupby(plotby)
                # grp_colors = iter([cmap(i) for i in
                #                    np.linspace(0.2, 1, len(grouped))])
                grp_colors = iter([cmap(i) for i in
                                   np.linspace(1, 0.4, len(grouped))])
                color = cmap(1.0)  # This is just for shuffle plotting
                for group_idx, group in grouped:
                    if plot_method == 'cdf':
                        plotting.cdf(
                            ax=ax, values=group[plot_column], bins='exact',
                            label=label + ': ' + str(group_idx), linestyle='-',
                            color=grp_colors.next(), **plot_kwargs)
                    else:
                        plotting.line_o_gram(
                            ax=ax, values=group[plot_column], linestyle='-',
                            label=label + ': ' + str(group_idx),
                            color=grp_colors.next(), **plot_kwargs)
            if plot_shuffle and not pool_shuffle:
                if not shuffle_plotby:
                    if plot_method == 'cdf':
                        plotting.cdf(
                            ax=ax, values=shuffle_df[plot_column],
                            bins='exact', linestyle='--', color=color,
                            label=label + '_shuffle', **plot_kwargs)
                    else:
                        plotting.line_o_gram(
                            ax=ax, values=shuffle_df[plot_column],
                            linestyle='--', color=color,
                            label=label + '_shuffle', **plot_kwargs)
                else:
                    grouped = shuffle_df.groupby(plotby)
                    grp_colors = iter([cmap(i) for i in
                                       np.linspace(0.2, 1, len(grouped))])
                    for group_idx, group in grouped:
                        if plot_method == 'cdf':
                            plotting.cdf(
                                ax=ax, values=group[plot_column], bins='exact',
                                label=label + '_shuffle: ' + str(group_idx),
                                linestyle='--', color=grp_colors.next(),
                                **plot_kwargs)
                        else:
                            plotting.line_o_gram(
                                ax=ax, values=group[plot_column],
                                linestyle='--',
                                label=label + '_shuffle: ' + str(group_idx),
                                color=grp_colors.next(), **plot_kwargs)

        if plot_shuffle and pool_shuffle:
            if shuffle_plotby:
                raise NotImplementedError
            if plot_method == 'cdf':
                plotting.cdf(
                    ax=ax, values=pooled_shuffle[plot_column], bins='exact',
                    label='shuffled', linestyle='--', color='k',
                    **plot_kwargs)
            else:
                plotting.line_o_gram(
                    ax=ax, values=pooled_shuffle[plot_column], label='shuffled',
                    linestyle='--', color='k', **plot_kwargs)

        if plot_method == 'cdf':
            ax.set_ylabel('Cumulative fraction')
        else:
            if plot_kwargs.get('hist_kwargs', {}).get('normed', False):
                ax.set_ylabel('Normalized density')
            else:
                ax.set_ylabel('Number')
            if plot_kwargs.get('hist_kwargs', {}).get('range', False):
                ax.set_xlim(plot_kwargs.get('hist_kwargs', {})['range'])
            ax.relim()
            ax.autoscale_view(tight=True, scalex=False, scaley=True)
        ax.set_xlabel(activity_label)
        ax.set_title(activity_label + groupby_label + plotby_label)
        ax.legend(frameon=False, ncol=len(dataframes), loc='best')

    elif plot_method in ['grouped_bar', 'grouped_box', 'box_and_line', 'swarm']:
        plot_shuffle_as_hline = plot_kwargs.pop('plot_shuffle_as_hline', False)
        values = []
        condition_labels = []  # The main comparison (genotype, cell-type, etc)
        bar_colors = []
        for dataframe, label in it.izip(dataframes, labels):
            values.append([])
            condition_labels.append(label)
            bar_colors.append(colors.next())
            if plotby is None:
                # If there is no plotby, there's only one set of bars, so
                # nothing to order
                assert orderby is None
                values[-1].append(
                    {'group_key': activity_label, 'order_key': activity_label,
                     plot_column: dataframe[plot_column]})
            else:
                for group_key, group in dataframe.groupby(plotby):
                    try:
                        group_key = float(group_key)
                    except (ValueError, TypeError):
                        group_key = str(group_key)
                    if orderby is None:
                        order_key = group_key
                    else:
                        order_key = group[orderby].mean()
                    group_dict = {'group_key': group_key,
                                  'order_key': order_key,
                                  plot_column: group[plot_column]}
                    values[-1].append(group_dict)

        #
        # Add shuffle data if needed
        #
        if plot_shuffle and shuffle_plotby:
            if pool_shuffle:
                values.append([])
                condition_labels.append('shuffle')
                bar_colors.append('0.9')
                if plotby is None:
                    values[-1].append(
                        {'group_key': activity_label,
                         'order_key': activity_label,
                         plot_column: pooled_shuffle[plot_column]})
                else:
                    for group_key, group in pooled_shuffle.groupby(plotby):
                        try:
                            group_key = float(group_key)
                        except ValueError:
                            group_key = str(group_key)
                        if orderby is None:
                            order_key = group_key
                        else:
                            order_key = group[orderby].mean()
                        values[-1].append(
                            {'group_key': group_key, 'order_key': order_key,
                             plot_column: group[plot_column]})

            else:
                for shuffle_df, label in it.izip(shuffles, labels):
                    values.append([])
                    condition_labels.append(label + '_shuffle')
                    bar_colors.append(colors.next())
                    for group_key, group in shuffle_df.groupby(plotby):
                        try:
                            group_key = float(group_key)
                        except ValueError:
                            group_key = str(group_key)
                        if orderby is None:
                            order_key = group_key
                        else:
                            order_key = group[orderby].mean()
                        values[-1].append(
                            {'group_key': group_key, 'order_key': order_key,
                             plot_column: group[plot_column]})
        elif plot_shuffle:
            if pool_shuffle:
                if plot_shuffle_as_hline:
                    ax.axhline(
                        pooled_shuffle[plot_column].mean(), linestyle='--',
                        color='k', label='shuffle')
                else:
                    values.append([
                        {'group_key': 'shuffle', 'order_key': np.inf,
                         plot_column: pooled_shuffle[plot_column]}])
                    condition_labels.append('shuffle')
                    bar_colors.append('0.9')
            else:
                if plot_shuffle_as_hline:
                    raise NotImplementedError(
                        'Plotting shuffle as a horizontal line only ' +
                        'implemented for pooled shuffle data.')
                for idx, shuffle_df in enumerate(shuffles):
                    values[idx].append(
                        {'group_key': 'shuffle', 'order_key': np.inf,
                         plot_column: shuffle_df[plot_column]})

        # Create a new dictionary with the keys all the plotby group_keys and
        # the values are all the ordering keys
        # For each group_key the ordering keys should either all be numeric so
        # that they can be averaged, or all identical
        sorting_dict = defaultdict(list)
        for val_dict in it.chain(*values):
            sorting_dict[val_dict['group_key']].append(val_dict['order_key'])

        # Once we have all the values, average across ordering keys, or make
        # sure they are all the same
        for group_key, order_keys in sorting_dict.iteritems():
            try:
                sorting_dict[group_key] = np.mean(order_keys)
            except TypeError:
                assert all([key == order_keys[0] for key in order_keys])
                sorting_dict[group_key] = order_keys[0]

        cluster_labels = [label for label, _ in sorted(
            sorting_dict.items(), key=itemgetter(1))]

        # if hyphen present in the x-axis labels, see if it's numeric data
        # ranges and sort ascending
        if np.all(['-' in str(x) for x in cluster_labels]):
            try:
                prefixes = [float(k.split('-')[0]) for k in cluster_labels]
            except ValueError:
                pass
            else:
                cluster_labels = [
                    cluster_labels[ix] for ix in np.argsort(prefixes)]

        if markers is None:
            markers = [None] * len(values)
        if linestyles is None:
            linestyles = ['-'] * len(values)

        data_to_plot = []
        for vals in values:
            data_to_plot.append([])
            for label in cluster_labels:
                # Find group
                group_values = [group[plot_column] for group in vals
                                if group['group_key'] == label]
                assert len(group_values) < 2
                if len(group_values):
                    data_to_plot[-1].append(group_values[0])
                else:
                    data_to_plot[-1].append([])

        #
        # Plot the data and set labels
        #
        if plot_method == 'grouped_bar':
            plotting.grouped_bar(
                ax, data_to_plot, condition_labels=condition_labels,
                cluster_labels=cluster_labels, bar_colors=bar_colors,
                loc='best', **plot_kwargs)
        elif plot_method == 'grouped_box':
            plotting.grouped_box(
                ax, data_to_plot, condition_labels=condition_labels,
                cluster_labels=cluster_labels, box_colors=bar_colors,
                loc='best', **plot_kwargs)
        elif plot_method == 'box_and_line':
            plotting.box_and_line(
                ax, data_to_plot, condition_labels=condition_labels,
                cluster_labels=cluster_labels, colors=bar_colors,
                loc='best', markers=markers, linestyles=linestyles,
                **plot_kwargs)
        elif plot_method == 'swarm':
            plotting.swarm_plot(
                ax, data_to_plot, condition_labels=condition_labels,
                cluster_labels=cluster_labels, colors=bar_colors,
                loc='best', **plot_kwargs)

        if plotby is not None:
            ax.set_xlabel(str(plotby))
        ax.set_ylabel(activity_label)
        ax.set_title(activity_label + groupby_label)

    elif plot_method == 'stacked_bar':
        if plot_shuffle:
            # Just not written yet
            raise NotImplementedError
        values = []
        for dataframe in dataframes:
            values.append({})
            for key, group in dataframe.groupby(plot_column):
                values[-1][key] = len(group)
        all_keys = set()
        for vals in values:
            all_keys.update(vals.keys())
        all_bars = sorted(all_keys)
        heights = []
        for vals in values:
            grp_heights = np.array(
                [vals.get(key, 0) for key in all_bars], dtype=float)
            grp_heights /= grp_heights.sum()
            heights.append(grp_heights)

        plotting.stackedBar(
            ax, range(len(labels)), heights, labels=all_bars, colors=colors,
            **plot_kwargs)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Fraction')
        ax.set_title(activity_label)

    elif plot_method == 'scatter_bar':
        values = []
        bar_colors = []
        if colorby is not None:
            scatter_colors = []
        condition_labels = []
        # Keep track of clusters to make sure they match between exptGrps
        cluster_labels = set()
        for dataframe, label in it.izip(dataframes, labels):
            bar_colors.append(colors.next())
            values.append({})
            if colorby is not None:
                scatter_colors.append({})
            if plotby is None:
                values[-1][activity_label] = dataframe[plot_column]
                cluster_labels.add(activity_label)
                if colorby is not None:
                    scatter_colors[-1][activity_label] = [
                        colorby_dict[key] for key in
                        dataframe.set_index(colorby).index]
            else:
                if orderby is not None:
                    raise NotImplementedError
                for group_idx, group in dataframe.groupby(plotby):
                    try:
                        group_idx = float(group_idx)
                    except (ValueError, TypeError):
                        group_idx = str(group_idx)
                    values[-1][group_idx] = group[plot_column]
                    cluster_labels.add(group_idx)

                    if colorby is not None:
                        group_colors = []
                        for row in group.iterrows():
                            k = tuple([row[1][c] for c in colorby])
                            group_colors.append(colorby_dict[k[0]])
                        scatter_colors[-1][group_idx] = group_colors
            condition_labels.append(label)

        cluster_labels = sorted(cluster_labels)
        data_to_plot = [[vals.get(cluster, []) for cluster in cluster_labels]
                        for vals in values]
        if colorby is not None:
            colorby_list = [
                [color.get(cluster, []) for cluster in cluster_labels]
                for color in scatter_colors]

        #
        # Add shuffle data if needed
        #
        if plot_shuffle and shuffle_plotby:
            if pool_shuffle:
                condition_labels.append('shuffle')
                bar_colors.append('0.9')
                shuffle_values = {}
                for group_idx, group in pooled_shuffle.groupby(plotby):
                    try:
                        group_idx = float(group_idx)
                    except ValueError:
                        group_idx = str(group_idx)
                    shuffle_values[group_idx] = group[plot_column]
                # If there are clusters in shuffle not in the real data
                # they will just be dropped
                data_to_plot.append(
                    [shuffle_values.get(cluster, [])
                     for cluster in cluster_labels])
                # Make all shuffle scatterbar points black
                if colorby is not None:
                    colorby_list.append(['k'] * len(cluster_labels))
            else:
                for shuffle_df, label in it.izip(shuffles, labels):
                    shuffle_values = {}
                    bar_colors.append(colors.next())
                    for group_idx, group in shuffle_df.groupby(plotby):
                        try:
                            group_idx = float(group_idx)
                        except ValueError:
                            group_idx = str(group_idx)
                        shuffle_values[group_idx] = group[plot_column]
                    data_to_plot.append(
                        [shuffle_values.get(cluster, [])
                         for cluster in cluster_labels])
                    condition_labels.append(label + '_shuffle')
                    if colorby is not None:
                        colorby_list.append(['k'] * len(cluster_labels))
        elif plot_shuffle:
            raise NotImplementedError
            cluster_labels.append('shuffle')
            if pool_shuffle:
                data_to_plot[0].append(pooled_shuffle[plot_column])
                for condition_data in data_to_plot[1:]:
                    condition_data.append([])
                bar_colors.append('0.9')
            else:
                for condition_data, shuffle_df in it.izip(
                        data_to_plot, shuffles):
                    condition_data.append(shuffle_df[plot_column])
            if colorby:
                for color in colorby_list:
                    color.append('k')
        #
        # Plot the data and set labels
        #
        scatterbar_colors = colorby_list if colorby else None
        plotting.grouped_bar(
            ax, data_to_plot, condition_labels=condition_labels,
            cluster_labels=cluster_labels, bar_colors=bar_colors,
            scatter_points=True, scatterbar_colors=scatterbar_colors,
            jitter_x=True, loc='best', **plot_kwargs)
        if plotby is not None:
            ax.set_xlabel(str(plotby))
        ax.set_ylabel(activity_label)
        ax.set_title(activity_label + groupby_label)

    elif plot_method == 'line':
        if plotby is None:
            raise ValueError(
                'Must include a plotby argument when plotting as a line')

        if 'markeredgecolor' not in plot_kwargs:
            plot_kwargs['markeredgecolor'] = 'k'
        if 'markeredgewidth' not in plot_kwargs:
            plot_kwargs['markeredgewidth'] = 0.5
        if 'capsize' not in plot_kwargs:
            plot_kwargs['capsize'] = 2.

        values = []
        shuffle_values = []
        # Keep track of clusters to make sure they match between exptGrps
        # Labels will be added as a (label, sorting) tuple.
        # If orderby is None, label=sorting, otherwise they will be sorted
        # by the orderby column
        for dataframe in dataframes:
            values.append([])
            for group_key, group in dataframe.groupby(plotby):
                try:
                    group_key = float(group_key)
                except (ValueError, TypeError):
                    group_key = str(group_key)
                if orderby is None:
                    order_key = group_key
                else:
                    order_key = group[orderby].mean()
                group_dict = {'group_key': group_key, 'order_key': order_key,
                              plot_column: group[plot_column]}
                values[-1].append(group_dict)
        if plot_shuffle and shuffle_plotby:
            if pool_shuffle:
                all_shuffles = (pooled_shuffle, )
            else:
                all_shuffles = shuffles
            for shuffle_df in all_shuffles:
                shuffle_values.append([])
                for group_key, group in shuffle_df.groupby(plotby):
                    try:
                        group_key = float(group_key)
                    except (ValueError, TypeError):
                        group_key = str(group_key)
                    if orderby is None:
                        order_key = group_key
                    else:
                        order_key = group[orderby].mean()
                    group_dict = {'group_key': group_key,
                                  'order_key': order_key,
                                  plot_column: group[plot_column]}
                    shuffle_values[-1].append(group_dict)

        # Create a new dictionary with the keys all the plotby group_keys and
        # the values are all the ordering keys
        # For each group_key the ordering keys should either all be numeric so
        # that they can be averaged, or all identical
        sorting_dict = defaultdict(list)
        for val_dict in it.chain(*(values + shuffle_values)):
            sorting_dict[val_dict['group_key']].append(val_dict['order_key'])

        # Once we have all the values, average across ordering keys, or make
        # sure they are all the same
        for group_key, order_keys in sorting_dict.iteritems():
            try:
                sorting_dict[group_key] = np.mean(order_keys)
            except TypeError:
                assert all([key == order_keys[0] for key in order_keys])
                sorting_dict[group_key] = order_keys[0]

        # If any of the ordering keys are non-numeric, use their order as the
        # x value, otherwise use the actual value of the sorting_dict
        if any([isinstance(x, basestring) for x in sorting_dict.values()]):
            x_vals = {group: order for order, (group, val)
                      in enumerate(sorted(sorting_dict.items(),
                                          key=itemgetter(1)))}
        else:
            x_vals = sorting_dict

        if markers is None:
            markers = [None] * len(values)
        if linestyles is None:
            linestyles = ['-'] * len(values)

        line_colors = []
        for df_label, vals, marker, ls in it.izip(
                labels, values, markers, linestyles):
            x_y_sem_tuples = [(x_vals[val['group_key']],
                              val[plot_column].mean(),
                              val[plot_column].std() / np.sqrt(len(val[plot_column])))
                              for val in vals]
            if break_at is not None:
                x_y_sem_tuples.extend(
                    (val, np.nan, np.nan) for val in break_at)
            sorted_tuples = sorted(x_y_sem_tuples, key=itemgetter(0))
            # Save colors for shuffle plotting if needed
            line_colors.append(colors.next())
            ax.errorbar([t[0] for t in sorted_tuples],
                        [t[1] for t in sorted_tuples],
                        yerr=[t[2] for t in sorted_tuples],
                        color=line_colors[-1], ecolor=line_colors[-1],
                        marker=marker, label=df_label, ls=ls, **plot_kwargs)
        #
        # Plot shuffle data if needed
        #
        if plot_shuffle and shuffle_plotby:
            if pool_shuffle:
                shuffle_labels = ['shuffled']
                shuffle_colors = ['k']
            else:
                shuffle_labels = [label + '_shuffled' for label in labels]
                shuffle_colors = line_colors
            for shuffle_label, vals, color in it.izip(
                    shuffle_labels, shuffle_values, shuffle_colors):
                # Pull out the clusters that are in the current dataframe,
                # while maintaining the sorted order
                x_y_sem_tuples = [
                    (x_vals[val['group_key']],
                     val[plot_column].mean(),
                     val[plot_column].std() / np.sqrt(len(val[plot_column])))
                    for val in vals]
                if break_at is not None:
                    x_y_sem_tuples.extend(
                        (val, np.nan, np.nan) for val in break_at)
                sorted_tuples = sorted(x_y_sem_tuples, key=itemgetter(0))

                ax.errorbar([t[0] for t in sorted_tuples],
                            [t[1] for t in sorted_tuples],
                            yerr=[t[2] for t in sorted_tuples],
                            color=color, ecolor=color, label=shuffle_label,
                            linestyle='--', **plot_kwargs)

        elif plot_shuffle:
            if pool_shuffle:
                ax.axhline(pooled_shuffle[plot_column].mean(), linestyle='--',
                           color='k', label='shuffle')
            else:
                for shuffle_df, label, line_color in it.izip(
                        shuffles, labels, line_colors):
                    ax.axhline(
                        shuffle_df[plot_column].mean(), linestyle='--',
                        color=line_color, label=label + '_shuffled')
        #
        # Set labels
        #
        # Find the clusters that were in the real data (not shuffled)
        # and just use those to set the labels
        all_groups = set([val['group_key'] for vals in values for val in vals])
        x_val_label_tuples = sorted(
            [(x_vals[group], group) for group in all_groups],
            key=itemgetter(0))
        all_x_vals = [t[0] for t in x_val_label_tuples]
        ax.set_xlim(min(all_x_vals) - 1, max(all_x_vals) + 1)
        ax.set_xticks([t[0] for t in x_val_label_tuples])
        ax.set_xticklabels([t[1] for t in x_val_label_tuples])
        ax.set_xlabel(str(plotby))
        ax.set_ylabel(activity_label)
        ax.set_title(activity_label + groupby_label)
        ax.legend(frameon=False, loc='best')

    elif plot_method == 'scatter':
        if len(plotby) != 1:
            # Could potentially allow categorical values, but for now just
            # numeric
            raise ValueError(
                "'plotby' must be a single column of numeric values")
        if markers is None:
            markers = ['o'] * len(dataframes)
        df_colors = []
        for dataframe, label, color, marker in zip(
                dataframes, labels, colors, markers):
            sns.regplot(
                plotby[0], plot_column, dataframe, ax=ax, marker=marker,
                color=color, line_kws={'label': label}, **plot_kwargs)
            df_colors.append(color)

        if plot_shuffle:
            if not shuffle_plotby:
                # Possible, just not implemented yet
                raise NotImplementedError
            if pool_shuffle:
                sns.regplot(
                    plotby[0], plot_column, pooled_shuffle, ax=ax, color='0.5',
                    line_kws={'label': 'shuffle'}, scatter=False,
                    **plot_kwargs)
            else:
                for shuffle_df, label, color in zip(
                        shuffles, labels, df_colors):
                    sns.regplot(
                        plotby[0], plot_column, shuffle_df, ax=ax, color=color,
                        line_kws={'label': '{}_shuffle'.format(label),
                                  'linestyle': ':'}, scatter=False,
                        **plot_kwargs)
        ax.set_xlabel(str(plotby))
        ax.set_ylabel(activity_label)
        ax.set_title(activity_label + groupby_label)
        ax.legend(frameon=False, loc='best')

    else:
        raise ValueError('Unrecognized plot method: ' + str(plot_method))

    # Don't want to actually return the exptGrp objects as keys
    return {
        label: dataframe for label, dataframe in it.izip(labels, dataframes)}


def plot_paired_dataframes(
        ax, first_dataframe, second_dataframe, labels, plot_method='scatter',
        groupby=(('expt',),), colorby=None, filter_fn=None,
        filter_columns=None, post_pair_groupby=None, post_pair_plotby=None,
        post_pair_filter_fn=None, post_pair_filter_columns=None,
        post_pair_colorby=None, x_bins=None, colorby_list=None, colors=None,
        z_score=False, shuffle_colors=False, plot_column='value',
        linestyles=None, **plot_kwargs):

    """
    plotby the last groupby tuple

    groupby = list of lists
    colorby = list
    filterby = lambda df: df['column'] != 'A'
    x_bins : Optional, int or array-like. Bin the x-axis value and convert to a
        categorical label. Bins specified either as a number of bins
        which will then be evenly spaced throughout the range of values, or
        bin boundaries as an array.
    plot_column : string
        The label of the column in the dataframe containing the data to plot.

    """

    # TODO: refactor to merge dataframe and use pandas groupby

    dataframes = [first_dataframe, second_dataframe]

    assert len(labels) == 2 and labels[0] != labels[1]

    plotby = list(groupby[-1])
    groupby = list(groupby[:-1])

    if colorby is not None:
        for c in colorby:
            if c in plotby:
                plotby.remove(c)
        plotby.extend(colorby)

    # Only applying filter to first dataframe
    if filter_fn is not None:
        prepare_dataframe(dataframes[0], include_columns=filter_columns)
        dataframes[0] = dataframes[0][filter_fn(dataframes[0])]
    # Try to filter second dataframe as well (as code was written originally)
    try:
        prepare_dataframe(dataframes[1], include_columns=filter_columns)
        dataframes[1] = dataframes[1][filter_fn(dataframes[1])]
    except:
        pass

    dataframes = tuple(dataframes)

    grouped_dfs = []
    for df in dataframes:
        for groupby_list in groupby:
            prepare_dataframe(df, include_columns=groupby_list)
            df = df.groupby(groupby_list, as_index=False).mean()
        prepare_dataframe(df, include_columns=plotby)
        df = df.groupby(plotby, as_index=False).mean()
        grouped_dfs.append(df)

    # Get all the groupby values common to both datasets
    dataframe_plotbys = []
    for df in grouped_dfs:
        df_plotby_tuples = set()
        for row in df.iterrows():
            g = []
            for key in plotby:
                g.append(row[1][key])
            df_plotby_tuples.add(tuple(g))
        dataframe_plotbys.append(df_plotby_tuples)
    plotby_tuples = list(
        dataframe_plotbys[0].intersection(dataframe_plotbys[1]))

    if colorby:
        l = np.array([list(a) for a in plotby_tuples], dtype='object')
        if not colorby_list:
            # length of this == number of colors
            colorby_list = sorted(
                set([tuple(c) for c in l[:, -1 * len(colorby):]]))
        if colors:
            assert len(colors) == len(colorby_list)
        else:
            colors = sns.husl_palette(len(colorby_list), l=.5, s=1)
            colors = [tuple(color) for color in colors]
            if shuffle_colors:
                shuffle(colors)
        color_legend = {c: l for c, l in it.izip(colors, colorby_list)}

    vals = [[], []]
    color_list = []  # len == len(vals[0]) == len(vals[1])
    for plotby_tuple in plotby_tuples:
        for i, df in enumerate(grouped_dfs):
            for g, val in zip(plotby, plotby_tuple):
                df = df[df[g] == val]
            assert len(df) == 1
            vals[i].append(df[plot_column].values[0])
        if colorby:
            l = []
            for c in colorby:
                l.append(df[c].values[0])
            t = tuple(l)
            color_list.append(colors[colorby_list.index(t)])
        else:
            color_list.append('k')

    assert len(vals[0]) == len(vals[1])
    assert len(color_list) == len(vals[0])
    vals = [np.array(v) for v in vals]
    if z_score:
        vals = [(v - np.mean(v)) / np.std(v) for v in vals]

    # Put the data back in to a single dataframe
    df = pd.DataFrame({labels[0]: vals[0], labels[1]: vals[1]})
    for idx, factor in enumerate(plotby):
        df[factor] = [tup[idx] for tup in plotby_tuples]
    if colorby:
        colorby_tuples = [color_legend[color] for color in color_list]
        df['colorby_tuple'] = colorby_tuples

    if x_bins is not None:
        df.dropna(subset=[labels[0]], inplace=True)
        if isinstance(x_bins, int):
            x_bins = np.linspace(
                df[labels[0]].min(), df[labels[0]].max(), x_bins + 1)
        binned_value = np.digitize(
            df[labels[0]], bins=np.hstack([-np.inf, x_bins, np.inf])) - 2
        df[labels[0]] = binned_value
        df = df[(df[labels[0]] >= 0) & (df[labels[0]] < len(x_bins) - 1)]

    if plot_method == 'scatter':
        plotting.scatterPlot(
            ax, vals, labels, colors=color_list,
            color_legend=color_legend if colorby else None, **plot_kwargs)
    elif plot_method == 'lmplot':
        return sns.lmplot(labels[0], labels[1], df, **plot_kwargs)
    elif plot_method == 'regplot':
        if colorby is not None:
            kwargs = copy.copy(plot_kwargs)
            markers = kwargs.pop('markers', None)
            line_kws = kwargs.pop('line_kws', {})
            for group_key, group in df.groupby(colorby):
                key = group_key if len(colorby) > 1 else (group_key,)
                if markers is None:
                    marker = 'o'
                else:
                    marker = markers[colorby_list.index(key)]
                if linestyles is None:
                    pass
                else:
                    line_kws['linestyle'] = linestyles[colorby_list.index(key)]
                color = colors[colorby_list.index(key)]
                sns.regplot(
                    labels[0], labels[1], group, ax=ax, marker=marker,
                    color=color, label=str(group_key), line_kws=line_kws,
                    **kwargs)
            ax.legend(frameon=False, loc='best')
        else:
            sns.regplot(labels[0], labels[1], df, ax=ax, **plot_kwargs)
    elif plot_method == 'jointplot':
        return sns.jointplot(labels[0], labels[1], df, **plot_kwargs)
    else:
        # Fall back to plotting with plot_dataframes
        # Re-split by expt_grp and copy the second metric to plot_column
        # 'plotby' should probably be in the plot_kwargs
        if x_bins is not None:
            bin_labels = ['{:.2f}-{:.2f}'.format(start, stop)
                          for start, stop in zip(
                              x_bins[:-1], x_bins[1:])]
            df[labels[0]] = [bin_labels[b] for b in df[labels[0]]]
        dataframes = []
        df_labels = []
        new_colors = []
        for group_key, group in df.groupby(colorby):
            group_df = group.copy()
            group_df[plot_column] = group_df[labels[1]]
            dataframes.append(group_df)
            df_labels.append(str(group_key))
            new_colors.append(colors[colorby_list.index(
                group_key if len(colorby) > 1 else (group_key,))])

        plot_dataframe(
            ax, dataframes, labels=df_labels, activity_label=labels[1],
            groupby=post_pair_groupby, plotby=post_pair_plotby,
            colorby=post_pair_colorby, filter_fn=post_pair_filter_fn,
            filter_columns=post_pair_filter_columns, plot_method=plot_method,
            colors=new_colors, plot_column=plot_column, **plot_kwargs)

    return df


def mark_bad(ax):
    """Puts a red X through an ax."""

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    ax.plot(x_lim, y_lim, 'r--')
    ax.plot(x_lim, y_lim[::-1], 'r--')

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)


def replace_ax(old_ax, new_ax):
    """Takes an old ax and replaces it with a different one"""

    fig = old_ax.figure
    new_ax.set_figure(fig)
    new_ax.set_position(old_ax.get_position())
    fig.delaxes(old_ax)
    fig.add_axes(new_ax)


def stagger_xticklabels(ax):
    """Staggers xticklabels such that every other label is lower."""
    labels = ax.get_xticklabels()
    new_labels = ['\n' + label.get_text() if idx % 2 else label for idx, label in enumerate(labels)]
    ax.set_xticklabels(new_labels)


def color_cycle(method='rcParams', **kwargs):
    """Returns a repeating cycle of colors.

    Parameters
    ----------
    method : string
        Name of method to use for generating colors.
    **kwargs
        Additional keyword arguments are used by the specific color cycle
        methods.

    Returns
    -------
    A repeating iterator of colors (Using itertools.cycle)

    """
    if method == 'rcParams':
        return it.cycle(
            prop['color'] for prop in mpl.rcParams['axes.prop_cycle'])
    if method == 'seaborn':
        return it.cycle(
            sns.color_palette(kwargs.get('palette', 'husl'),
                              kwargs.get('n_colors', None)))


def formatAxes(ax, xlabel, ylabel, topVisible=False, rightVisible=False,
               tickLableSize=7, yTickLabelRotation="vertical",
               tickPad=1, tickDirection="out", tickLength=2,
               labelSize=7):

    ax.spines["top"].set_visible(topVisible)
    ax.spines["right"].set_visible(rightVisible)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.tick_params(labelsize=tickLableSize, pad=tickPad,
                   direction=tickDirection, length=tickLength)
    ax.set_ylabel(ylabel, fontsize=labelSize)
    ax.set_xlabel(xlabel, fontsize=labelSize)

    yTLabels = ax.get_yticklabels()
    for i in np.r_[0:len(yTLabels)]:
        yTLabels[i].set_rotation(yTickLabelRotation)


def square_axis(ax):
    r = (
        ax.get_xlim()[1] - ax.get_xlim()[0]) / (
        ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.set_aspect(r)


import plotting