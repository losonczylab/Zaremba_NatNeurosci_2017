"""Plotting lab metrics."""

import pandas as pd
import itertools as it
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import warnings

from plotting_helpers import plot_dataframe, plot_paired_dataframes, \
    prepare_dataframe
from lab.classes.exceptions import InvalidDataFrame


def plot_metric(
        ax, exptGrps, metric_fn, plot_method, roi_filters=None, plot_abs=False,
        activity_kwargs=None, label_every_n=1, return_full_dataframes=False,
        rotate_labels=True, recalc=False, **plot_kwargs):
    """Plotting method for plotting any activity metric.

    Metric should return a dataframe or a pair of dataframes
    (data and shuffled data).

    Note that not all combinations of plotbys and shuffle plotting are not
    possible for all plot methods.

    exptGrps -- list of exptGrps
    metric_fn -- function to call, should take at least an exptGrp and
        potentially an roi_filter and return a dataframe with the calculated
        value in 'value'
    plot_method -- method used to plot the data, should be a valid method
        in plotting.plot_dataframe
    plot_abs -- instead of plotting 'value' plot abs('value')
    activity_kwargs -- a dictionary of keyword arguments that are passed in to
        metric_fn or a list of dictionaries, one per exptGrp
    label_every_n -- hides extra xticklabels
    rotate_labels -- If True, rotate the x-axis labels by 40 degrees
    **plot_kwargs -- any additional keyword arguments are passed directly to
        the plotting function

    """
    if roi_filters is None:
        roi_filters = [None] * len(exptGrps)

    if activity_kwargs is None:
        activity_kwargs = {}
    if not isinstance(activity_kwargs, list):
        activity_kwargs = [activity_kwargs] * len(exptGrps)

    dataframes, shuffles = [], []
    if return_full_dataframes:
        orig_dataframes, orig_shuffles = [], []
    for exptGrp, roi_filter, kwargs in it.izip(
            exptGrps, roi_filters, activity_kwargs):
        try:
            metric_data = metric_fn(
                exptGrp, roi_filter=roi_filter, **kwargs)
        except TypeError as e:
            if "got an unexpected keyword argument 'roi_filter'" in e.message:
                metric_data = metric_fn(exptGrp, **kwargs)
            else:
                raise

        if not isinstance(metric_data, pd.DataFrame) and \
                (len(metric_data) == 2 and
                 isinstance(metric_data[0], pd.DataFrame) and
                 (isinstance(metric_data[1], pd.DataFrame) or
                  metric_data[1] is None)):
            grp_data = metric_data[0]
            grp_shuffle = metric_data[1]
        else:
            grp_data = metric_data
            grp_shuffle = None

        if plot_abs:
            try:
                grp_data['value'] = grp_data['value'].apply(np.abs)
            except KeyError:
                raise InvalidDataFrame(
                    'DataFrame missing \'value\' column')

        if grp_shuffle is not None and plot_abs:
            try:
                grp_shuffle['value'] = grp_shuffle['value'].apply(np.abs)
            except KeyError:
                raise InvalidDataFrame(
                    'DataFrame missing \'value\' column')

        dataframes.append(grp_data)
        shuffles.append(grp_shuffle)
        if return_full_dataframes:
            orig_dataframes.append(grp_data.copy(deep=True))
            orig_shuffles.append(
                grp_shuffle if grp_shuffle is None else
                grp_shuffle.copy(deep=True))

    grp_labels = []
    for grp_idx, exptGrp in it.izip(it.count(), exptGrps):
        if exptGrp.label() is not None:
            grp_labels.append(exptGrp.label())
        else:
            grp_labels.append('Group ' + str(grp_idx))

    # Make sure metric_fn returned data
    try:
        plot_dataframe(
            ax, dataframes, shuffles, labels=grp_labels,
            plot_method=plot_method,
            **plot_kwargs)
    except:
        # Should this do something else?
        print 'Plotting of {} failed for {}'.format(metric_fn.__name__, exptGrp.label())
        # pass
        raise
    else:
        # Remove extra labels on the x-axis
        if label_every_n > 1:
            # Hide labels
            for label_idx, label in it.izip(it.count(), ax.get_xticklabels()):
                if label_idx % label_every_n:
                    label.set_visible(False)

        if rotate_labels:
            plt.setp(ax.get_xticklabels(), rotation='40',
                     horizontalalignment='right')

    if return_full_dataframes:
        return {exptGrp.label(): {'dataframe': dataframe, 'shuffle': shuffle}
                for exptGrp, dataframe, shuffle in it.izip(
            exptGrps, orig_dataframes, orig_shuffles)}

    return {exptGrp.label(): {'dataframe': dataframe, 'shuffle': shuffle}
            for exptGrp, dataframe, shuffle in it.izip(
        exptGrps, dataframes, shuffles)}


def plot_paired_metrics(
        expt_grps, first_metric_fn, second_metric_fn, ax=None,
        roi_filters=None, groupby=(('expt',),), colorby=None,
        first_metric_kwargs=None, second_metric_kwargs=None,
        first_metric_label='First metric', second_metric_label='Second metric',
        include_columns=None, plot_abs=False, filter_metric_fn=None,
        filter_metric_fn_kwargs=None, filter_metric_merge_on=('roi',),
        **plot_kwargs):
    """Plotting method for plotting two activity metrics against each other.

    expt_grps -- list of expt_grps
    ax -- Axis to plot on
    first_metric_fn, second_metric_fn -- function to call, should take at least
        an exptGrp and possibly an roi_filter and return a dataframe with the
        pass to each metric_fn. Either a dictionary of kwargs or a list of
        dicts, one dict per expt_grp.
    first_metric_label, second_metric_label -- Axis labels.
    groupby -- List of lists of columns to aggregate data over before plotting.
        The last list will be used for pairing data across dataframes.
    colorby -- If not None, another key to group the data by and color uniquely
    plot_abs -- If True, plot the absolute value instead of the raw value
    filter_metric_fn -- function to call, on whose value first_metric_fn
        results will be filtered.  e.g. if you were looking at stability vs.
        performance, you may want to look at stability of cells near the reward
        vs. performance, in which case your filter_metric_fn would be
        distance_to_reward
    filter_metric_fn_kwargs -- dict of kwargs corresponding to the
        filter_metric_fn
    filter_metric_merge_on -- tuple of columns on which to merge
        first_metric_fn
        with filter_metric_fn
    **plot_kwargs -- any additional keyword arguments are passed directly to
        the plotting function

    """

    if roi_filters is None:
        roi_filters = [None] * len(expt_grps)

    if first_metric_kwargs is None:
        first_metric_kwargs = {}
    if second_metric_kwargs is None:
        second_metric_kwargs = {}
    if not isinstance(first_metric_kwargs, list):
        first_metric_kwargs = [first_metric_kwargs] * len(expt_grps)
    if not isinstance(second_metric_kwargs, list):
        second_metric_kwargs = [second_metric_kwargs] * len(expt_grps)

    metric_data = []
    for metric_fn, fn_kwargs_list in (
            (first_metric_fn, first_metric_kwargs),
            (second_metric_fn, second_metric_kwargs)):
        df_list = []
        for expt_grp, roi_filter, fn_kwargs in it.izip(
                expt_grps, roi_filters, fn_kwargs_list):
            try:
                data = metric_fn(expt_grp, roi_filter=roi_filter,
                                 **fn_kwargs)
            except TypeError as e:
                if "got an unexpected keyword argument 'roi_filter'" in e.message:
                    data = metric_fn(expt_grp, **fn_kwargs)
                else:
                    raise

            # Check for a pair of dataframes, assume the second is shuffled
            # and toss it out
            # The second element could also be None, if a stat that
            # normally shuffles data, but it was explicitly skipped
            if not isinstance(data, pd.DataFrame) and \
                    (len(data) == 2 and isinstance(
                        data[0], pd.DataFrame) and
                     (isinstance(data[1], pd.DataFrame) or
                      data[1] is None)):
                data = data[0]

            data['expt_grp'] = expt_grp.label()
            if plot_abs:
                try:
                    data['value'] = data['value'].apply(np.abs)
                except KeyError:
                    raise InvalidDataFrame(
                        'DataFrame missing \'value\' column')

            df_list.append(data)
        metric_data.append(pd.concat(df_list))

    if filter_metric_fn is not None:
        if filter_metric_fn_kwargs is None:
            filter_metric_fn_kwargs = {}

        df_list = []
        for expt_grp, roi_filter in it.izip(expt_grps, roi_filters):
            try:
                data = filter_metric_fn(
                    expt_grp, roi_filter=roi_filter, **filter_metric_fn_kwargs)
            except TypeError as e:
                if "got an unexpected keyword argument 'roi_filter'" in e.message:
                    data = filter_metric_fn(expt_grp, **filter_metric_fn_kwargs)
                else:
                    raise

            # Toss shuffle
            if not isinstance(data, pd.DataFrame) and \
                    (len(data) == 2 and isinstance(
                        data[0], pd.DataFrame) and
                     (isinstance(data[1], pd.DataFrame) or
                      data[1] is None)):
                data = data[0]
            df_list.append(data)
        filter_metric_df = pd.concat(df_list)
        filter_metric_df = prepare_dataframe(
            filter_metric_df, filter_metric_merge_on)

        filter_metric_df['filter_metric_value'] = filter_metric_df['value']
        del filter_metric_df['value']

        metric_data[0]['imaged'] = 1.

        metric_data[0] = pd.merge(
            metric_data[0], filter_metric_df, how='left',
            on=filter_metric_merge_on)

        metric_data[0] = metric_data[0][np.isfinite(metric_data[0]['imaged'])]
        del metric_data[0]['imaged']

    combined_df = plot_paired_dataframes(
        ax, metric_data[0], metric_data[1],
        [first_metric_label, second_metric_label],
        groupby=groupby, colorby=colorby, **plot_kwargs)

    return combined_df
