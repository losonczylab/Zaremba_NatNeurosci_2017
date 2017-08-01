"""Definition of Interval objects.

Used to filter/mask data in time.

"""

try:
    from bottleneck import nanmin, nanmax
except ImportError:
    from numpy import nanmin, nanmax
import numpy as np
import pandas as pd


class Interval(pd.DataFrame):
    """Class for defining intervals of interest across ROIs and Trials.

    Inherits from and stores data as a pandas DataFrame.

    Each row is an interval defined by the values in the 'start' and 'stop'
    columns. In addition, there should be at either an 'experiment' or 'trial'
    column and optionally an 'roi' column.

    Parameters
    ----------
    intervals : pd.DataFrame, list, dict
        Interval data will be passed to pd.DataFrame, so see pd.DataFrame for
        details of possible initialization structure. The result DataFrame must
        at least have 'start' and 'stop' columns.

    sampling_interval : float
        Conversion factor for values stored in 'start'/'stop'; 1. if intervals
        are in seconds, or the frame period if the intervals are in imaging
        frames.

    num_frames : int
        Duration of all time for the given experiments/trials. Used for
        converting to mask.

    Note
    ----
    For details of DataFrame subclassing, see:
    http://pandas.pydata.org/pandas-docs/stable/internals.html#subclassing-pandas-data-structures

    """

    _metadata = ['_sampling_interval']

    def __init__(self, intervals, sampling_interval=1., **kwargs):
        super(Interval, self).__init__(intervals, **kwargs)

        assert 'start' in self.columns
        assert 'stop' in self.columns

        self._sampling_interval = sampling_interval

    @property
    def _constructor(self):
        return Interval

    @property
    def _constructor_sliced(self):
        return IntervalSeries

    @property
    def sampling_interval(self):
        """Sampling interval of data in seconds."""
        return self._sampling_interval

    @sampling_interval.setter
    def sampling_interval(self, new_sampling_interval):
        self.resample(float(new_sampling_interval), inplace=True)

    @classmethod
    def from_mask(cls, mask, sampling_interval=1., data=None, **kwargs):
        """Create an interval object from a boolean mask.

        Parameters
        ----------
        mask : Nx1 np.ndarray of booleans
            True within interval, False outside of interval.

        sampling_interval : float
            Time between samples, used to convert intervals to time (in
            seconds).

        data : dict, optional
            Additional columns to add to Interval DataFrame. Each key will be a
            column and the value will be the same for each row.

        kwargs : optional
            Additional keyword arguments are passed to the DataFrame init.

        Returns
        -------
        Interval
            New Interval object representation of the mask.

        """
        if data is None:
            data = {}

        df_list = []

        for start, stop in _mask_to_intervals_1d(mask):
            df_list.append(dict(start=start, stop=stop, **data))

        return cls(df_list, sampling_interval=sampling_interval, **kwargs)

    def resample(self, new_sampling_interval=1., inplace=False):
        """Change the time units of the data.

        Parameters
        ----------
        new_sampling_interval : float
            The new sampling interval of the data.

        inplace : boolean, optional
            If True, edit the Interval object in place, if False return a new
            Interval object.

        Returns
        -------
        Interval
            Either the original Interval with new 'start'/'stop' values, or
            a new Interval

        """
        scale_factor = self.sampling_interval / float(new_sampling_interval)
        if inplace:
            dataframe = self
        else:
            dataframe = self.copy()
        dataframe['start'] *= scale_factor
        dataframe['stop'] *= scale_factor
        dataframe._sampling_interval = new_sampling_interval
        if not inplace:
            return dataframe

    def merge_intervals(self, inplace=False):
        """Merge overlapping intervals.

        As a side-effect of the merging, also sorts all intervals by 'start'.

        Parameters
        ----------
        inplace : bool

        """
        sampling_interval = self.sampling_interval

        def join_wrapper(group_df):
            """Wrapper to correctly wrap and unwrap data for reduce."""
            sorted_df = group_df.sort_values(by='start', na_position='first')
            df_rows = (row for _, row in sorted_df.iterrows())
            reduced = reduce(_joiner, df_rows, [])
            return type(self)(reduced).reset_index(drop=True)

        if not inplace:
            dataframe = self.copy()
        else:
            dataframe = self
        other_columns = [col for col in dataframe.columns.values
                         if col not in ['start', 'stop']]
        dataframe = dataframe.groupby(other_columns).apply(
            join_wrapper).reset_index(drop=True)
        dataframe._sampling_interval = sampling_interval
        if not inplace:
            return dataframe

    def durations(self, end_time):
        """Calculate total durations of intervals.

        Parameters
        ----------
        end_time : float
            Max time (in seconds) of the interval. Replaces NaN in 'stop'
            column.

        Returns
        -------
        pd.DataFrame

        """
        other_columns = [col for col in self.columns.values
                         if col not in ['start', 'stop']]

        resampled = self.resample(inplace=False).fillna(
            {'start': 0, 'stop': end_time})
        resampled['duration'] = resampled['stop'] - resampled['start']
        result = resampled.groupby(other_columns, as_index=False).agg(
            lambda x: x['duration'].sum())
        return result.drop(['start', 'stop'], axis=1)

    def __invert__(self):
        """Invert the time that is in/out of the interval."""
        # Need a way to invert ROIs/Trials not present in original df.
        raise NotImplementedError('Does not reliably invert intervals.')
        other_columns = [col for col in self.columns.values
                         if col not in ['start', 'stop']]

        def invert_wrapper(group_df):
            def row_gen(group_df):
                for _, row in group_df.iterrows():
                    # row['_sampling_interval'] = group_df.sampling_interval
                    yield row
            reduced = reduce(_invert_intersector, row_gen(group_df), [])
            return reduced

        result = self.groupby(other_columns).apply(invert_wrapper)
        result._sampling_interval = self.sampling_interval
        return result.reset_index(drop=True)

    def __and__(self, other):
        """Combine Interval objects to only include overlapping intervals."""
        # If we are combining with a non-Interval, use it as a filter
        # Might also just be able to add all NaN 'start' and 'stop' columns.
        if all(col not in other.columns for col in ['start', 'stop']):
            return pd.merge(self, other, how='inner').merge_intervals()

        other_resampled = other.resample(self.sampling_interval, inplace=False)

        other_columns = set(self.columns).intersection(
            other.columns).difference(['start', 'stop'])

        merged = pd.merge(
            pd.DataFrame(self), pd.DataFrame(other_resampled),
            on=list(other_columns), how='inner')

        merged_rows = (row for _, row in merged.iterrows())

        reduced = Interval(
            reduce(_intersector, merged_rows, []),
            columns=set(self.columns,).union(other.columns),
            sampling_interval=self.sampling_interval)

        return reduced.merge_intervals()

    def __or__(self, other):
        """Combine Interval objects as the union of all intervals."""
        if all(col not in other.columns for col in ['start', 'stop']):
            # if all(col in self.columns for col in other.columns):
            #     raise Exception
            # return pd.merge(self, other, how='outer')
            raise ValueError(
                'Unrecognized other Interval, expecting Interval object.')

        other_resampled = other.resample(self.sampling_interval, inplace=False)

        other_columns = set(self.columns).intersection(
            other.columns).difference(['start', 'stop'])

        merged = pd.merge(
            pd.DataFrame(self), pd.DataFrame(other_resampled),
            on=list(other_columns), how='outer', indicator=True)

        merged_rows = (row for _, row in merged.iterrows())

        reduced = Interval(
            reduce(_unioner, merged_rows, []),
            columns=set(self.columns).union(other.columns),
            sampling_interval=self.sampling_interval)

        return reduced.merge_intervals()

    def filter_events(self, events_df, key='value', invert=False):
        """Filter a DataFrame of event times.

        Parameters
        ----------
        events_df : pd.DataFrame
            A DataFrame containing times to filter. Should have columns that
            match the Interval dataframe: expt, trial, roi, etc.
        key : string
            The column containing the data to filter on. Should be at same
            sampling interval as current Interval.

        Returns
        -------
        pd.DataFrame
            A new DataFrame only including events that occurred within the
            current Interval.

        """
        # Create a copy and also convert to a basic DataFrame (might be an
        # Interval)
        events_df = pd.DataFrame(events_df)
        events_df['_index'] = np.arange(events_df.shape[0])

        # 'start' and 'stop' are special columns in an Interval dataframe.
        # Ensure that we never try to merge on them, and if you want to filter
        # on one of them, rename to something new.
        columns = set(list(events_df.columns)).intersection(
            self.columns).difference(['start', 'stop'])
        if key in ['start', 'stop']:
            orig_key = key
            key = '_value'
            events_df.rename(columns={orig_key: key}, inplace=True)
        events = events_df[list(columns) + [key, '_index']]

        merged = pd.merge(events, self, how='inner')
        merged['_test'] = (np.isnan(merged['start']) | (merged['start'] <= merged[key])) & \
                          (np.isnan(merged['stop']) | (merged[key] < merged['stop']))
        merged_test = merged[['_index', '_test']]
        grouped = merged_test.groupby('_index').any().reset_index()

        result = pd.merge(events_df, grouped, on='_index', how='left')
        result['_test'] = result['_test'].fillna(False)
        if invert:
            result = result[~result['_test']]
        else:
            result = result[result['_test']]
        del result['_test']
        del result['_index']

        if key is '_value':
            result.rename(columns={key: orig_key}, inplace=True)

        return result


class IntervalSeries(pd.Series):

    def __init__(self, *args, **kwargs):
        super(IntervalSeries, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return IntervalSeries

    @property
    def _constructor_expanddim(self):
        return Interval


def concat(intervals, ignore_index=True, **kwargs):
    """Same functionality as pd.concat.

    Sampling interval of resulting Interval will match the first Interval.

    Parameters
    ----------
    ignore_index : bool
        Changed default to True, but see pd.concat for details.

    kwargs : optional
        All other arguments are passed to pd.concat.

    """
    new_sampling_interval = intervals[0].sampling_interval
    resampled_intervals = [
        interval.resample(new_sampling_interval, inplace=False)
        for interval in intervals]
    concat_intervals = pd.concat(
        resampled_intervals, ignore_index=ignore_index, **kwargs)
    concat_intervals._sampling_interval = new_sampling_interval
    return concat_intervals


def _mask_to_intervals_1d(mask):
    """Convert a 1d boolean array to Nx2 array of starts/stops.

    Parameters
    ----------
    mask : Nx1 np.ndarray

    Returns
    -------
    intervals : Mx2 np.ndarray

    """
    mask_diff = np.diff(mask.astype('int'))
    starts = np.where(mask_diff == 1)[0]
    stops = np.where(mask_diff == -1)[0]
    if len(stops) and (not len(starts) or stops[0] < starts[0]):
        starts = np.hstack([np.nan, starts])
    if len(starts) and (not len(stops) or starts[-1] > stops[-1]):
        stops = np.hstack([stops, np.nan])

    assert len(starts) == len(stops)

    stacked_intervals = np.vstack([starts, stops]).T

    stacked_intervals += 1  # result of diff is trimmed down by 1

    return stacked_intervals


def _joiner(acc, int_df_row):
    # https://stackoverflow.com/questions/37496759/combining-discrete-and-or-overlapping-time-sequences-from-lists
    # if an empty list, return the new interval
    if not len(acc):
        return [int_df_row]
    # pop the last interval from the list
    last = acc.pop()
    # if the intervals are disjoint, return both
    if int_df_row['start'] > last['stop']:
        return acc + [last, int_df_row]
    # otherwise, join them together
    last['stop'] = np.max([int_df_row['stop'], last['stop']])
    return acc + [last]


def _unioner(acc, row):
    # Expect '_merge', start_x', 'stop_x', 'start_y', and 'stop_y'
    other_columns = set(row.index).difference(
        ['start_x', 'stop_x', 'start_y', 'stop_y', '_merge'])
    indicator = row['_merge']
    base_dict = {col: row[col] for col in other_columns}
    intervals = [{'start': row['start_x'], 'stop': row['stop_x']},
                 {'start': row['start_y'], 'stop': row['stop_y']}]
    intervals[0].update(base_dict)
    intervals[1].update(base_dict)
    if indicator is 'left_only':
        return acc + [intervals[0]]
    elif indicator is 'right_only':
        return acc + [intervals[1]]
    if np.isnan(intervals[1]['start']) or \
            intervals[1]['start'] < intervals[0]['start']:
        intervals = intervals[::-1]

    # if the intervals are disjoint, return both
    if intervals[0]['stop'] < intervals[1]['start']:
        return acc + intervals
    # otherwise, join them together
    joined_int = intervals[0]
    joined_int['stop'] = np.max([intervals[0]['stop'], intervals[1]['stop']])
    return acc + [joined_int]


def _intersector(acc, row):
    # Expect 'start_x', 'stop_x', 'start_y', and 'stop_y'
    other_columns = set(row.index).difference(
        ['start_x', 'stop_x', 'start_y', 'stop_y'])
    base_dict = {col: row[col] for col in other_columns}

    merged_int = dict(start=nanmax([row['start_x'], row['start_y']]),
                      stop=nanmin([row['stop_x'], row['stop_y']]), **base_dict)
    if merged_int['start'] < merged_int['stop'] or \
            np.any(np.isnan([merged_int['start'], merged_int['stop']])):
        return acc + [merged_int]
    return acc


def _invert_intersector(acc, row):
    row_dict = {key: [val] * 2 for key, val in row.iteritems()
                # if key not in ['start', 'stop', '_sampling_interval']}
                if key not in ['start', 'stop']}
    row_dict['start'] = [np.nan, row['stop']]
    row_dict['stop'] = [row['start'], np.nan]
    # row_int = Interval(row_dict, sampling_interval=row['_sampling_interval'])
    row_int = Interval(row_dict)
    row_int.dropna(how='all', subset=['start', 'stop'], inplace=True)
    if not len(acc):
        return row_int
    else:
        return acc & row_int


if __name__ == '__main__':
    run_int = Interval([{'expt': 1, 'start': 2, 'stop': 5},
                        {'expt': 1, 'start': 7, 'stop': 9},
                        {'expt': 2, 'start': np.nan, 'stop': 10},
                        {'expt': 2, 'start': 15, 'stop': np.nan}])
    pf_int = Interval([{'expt': 1, 'start': 10, 'stop': 12, 'roi': 1},
                       {'expt': 1, 'start': 3, 'stop': 4, 'roi': 1},
                       {'expt': 1, 'start': 8, 'stop': 10, 'roi': 1},
                       {'expt': 1, 'start': 8, 'stop': 11, 'roi': 2},
                       {'expt': 2, 'start': np.nan, 'stop': 2, 'roi': 1},
                       {'expt': 2, 'start': 2, 'stop': 12, 'roi': 1},
                       {'expt': 2, 'start': 11, 'stop': 13, 'roi': 2},
                       {'expt': 3, 'start': 2, 'stop': 4, 'roi': 1}])
    roi_df = pd.DataFrame([{'expt': 1, 'roi': 1},
                           {'expt': 1, 'roi': 2},
                           {'expt': 1, 'roi': 3},
                           {'expt': 2, 'roi': 1},
                           {'expt': 3, 'roi': 1}])
    trans_events = pd.DataFrame([{'expt': 1, 'roi': 1, 'max': 2, 'amp': 12},
                                 {'expt': 1, 'roi': 1, 'max': 11, 'amp': 1},
                                 {'expt': 1, 'roi': 1, 'max': 20, 'amp': 1},
                                 {'expt': 1, 'roi': 5, 'max': 3, 'amp': 7},
                                 {'expt': 2, 'roi': 1, 'max': 1, 'amp': 3}])
    stim_events = pd.DataFrame([{'expt': 1, 'value': 3},
                                {'expt': 1, 'value': 20}])

    z = pd.merge(run_int, roi_df, on=['expt'], how='outer').merge_intervals()

    x = z | pf_int

    union = run_int | pf_int
    intersection = run_int & pf_int

    # inverted_run = ~ run_int

    pf_int.filter_events(trans_events, key='max')

    run_int.duration(100)
    from pudb import set_trace; set_trace()
