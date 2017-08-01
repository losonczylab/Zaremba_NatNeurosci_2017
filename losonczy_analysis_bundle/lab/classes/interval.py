"""Definition of Interval objects that will be used to filter/mask data in
time.

"""

import numpy as np


class Interval(object):
    """Base interval object.

    Parameters
    ----------
    mask : ndarray or list of list of Nx2 arrays
        If ndarray, format should be (ROIs x Time x Trials).
        If list of intervals, should be [ROI][Trial][Nx2 array], where Nx2
        array is start, stop times of each interval.
    sampling_interval : float
        Time interval between sample bins.
    num_frames : int
        The size of the data in frames. Required if initialized with intervals.

    Attributes
    ----------
    mask : ndarray
        Boolean mask
    intervals : list of list of ndarray
        start, stop intervals
    shape : tuple
        shape of mask
    sampling_interval : float
        Sampling interval (in seconds) of underlying data

    Note
    ----

    Data is stored internally as a boolean 3D mask.

    Masks are (ROIs x Time x Trial)
    Intervals are [Trial][ROIs][Nx2 array]

    Bitwise arithmetic is defined for Intervals, namely union (|),
    intersection (&), and inverse (~)

    """

    # Core utilities

    def __init__(self, mask, sampling_interval=1., num_frames=None):

        if isinstance(mask, np.ndarray):
            self._mask = mask.astype('bool')
        else:
            assert num_frames
            self._mask = _intervals_to_mask(
                mask, num_frames=num_frames)

        self._sampling_interval = float(sampling_interval)

    def __getitem__(self, indices):
        indices = indices if isinstance(indices, tuple) else (indices,)
        # Reformat integer slices to avoid dimension collapse
        new_indices = []
        for i in indices:
            try:
                i = int(i)
            except TypeError:
                new_indices.append(i)
            else:
                new_indices.append(slice(i, i + 1))

        new_mask = self.mask[tuple(new_indices)]
        return type(self)(
            mask=new_mask, sampling_interval=self.sampling_interval)

    def __array__(self):
        return self.mask

    def __invert__(self):
        return type(self)(
            mask=~self.mask, sampling_interval=self._sampling_interval)

    def __repr__(self):
        return repr(self.mask)

    def __and__(self, other):
        if not type(self) == type(other):
            raise TypeError("Objects must be of same type")

        if not self.sampling_interval == other.sampling_interval:
            other = other.resample(self.sampling_interval)

        assert self.shape == other.shape

        new_mask = np.logical_and(self.mask, other.mask)

        return type(self)(new_mask, sampling_interval=self.sampling_interval)

    def __or__(self, other):
        if not type(self) == type(other):
            raise TypeError("Objects must be of same type")

        if not self.sampling_interval == other.sampling_interval:
            other = other.resample(self.sampling_interval)

        assert self.shape == other.shape

        new_mask = np.logical_or(self.mask, other.mask)

        return type(self)(new_mask, sampling_interval=self.sampling_interval)

    # Properties

    @property
    def shape(self):
        return self.mask.shape

    @property
    def sampling_interval(self):
        return self._sampling_interval

    @property
    def mask(self):
        return self._mask

    @property
    def intervals(self):
        return _mask_to_intervals(self._mask)

    # Methods

    def resample(self, sampling_interval):
        """Resample an interval to a new sampling rate.

        Parameters
        ----------
        sampling_interval : float
            New sampling interval of data

        Returns
        -------
        Interval : type(self)
            New Interval object of same type with new sampling interval.

        """

        intervals = _mask_to_intervals(self._mask)
        scale_factor = self.sampling_interval / float(sampling_interval)
        new_intervals = _rescale_intervals(intervals, scale_factor)
        num_frames = int(self.shape[1] * scale_factor)
        return type(self)(
            mask=new_intervals, sampling_interval=sampling_interval,
            num_frames=num_frames)


class IntervalDict(dict):
    """A dictionary of interval objects.
    Intended to be used for analysis where the keys will be experiments.
    Allows for manipulation of all intervals simultaneously.

    """

    def __invert__(self):
        """Inverts all the Intervals in an IntervalDict"""
        return type(self)({
            key: ~interval for key, interval in self.iteritems()})


class BehaviorInterval(Interval):
    """Interval object for behavior data, one mask per trial.

    Parameters
    ----------
    mask : ndarray or list of arrays
        Either a (Trials x Time) boolean array or a list
        (one element per trial) of Nx2 arrays of start/stop times.

    See also
    --------
    Interval
    ImagingInterval

    """

    def __init__(self, mask, *args, **kwargs):
        if isinstance(mask, np.ndarray):
            assert mask.ndim == 2  # (Trials x Time)
            mask = mask.T
            mask = mask[None, ...]
        else:  # list of intervals
            mask = [mask]
        super(BehaviorInterval, self).__init__(mask, *args, **kwargs)

    @property
    def mask(self):
        """Boolean mask of included frames.

        Returns
        -------
        mask : ndarray
            (Trials x Time) boolean array

        """

        return self._mask[0].T

    @property
    def intervals(self):
        """List of Nx2 arrays of start, stop times, 1 per trial.

        Returns
        -------
        intervals : list of ndarray
            1 element per trial, Nx2 arrays of start, stop times for each
            interval

        """

        return _mask_to_intervals(self._mask)[0]

    # TODO: code mostly copied from Interval, need to refactor
    def resample(self, sampling_interval):
        """Resample an interval to a new sampling rate.

        Parameters
        ----------
        sampling_interval : float
            New sampling interval of data

        Returns
        -------
        Interval : type(self)
            New Interval object of same type with new sampling interval.

        """

        intervals = _mask_to_intervals(self._mask)
        scale_factor = self.sampling_interval / float(sampling_interval)
        new_intervals = _rescale_intervals(intervals, scale_factor)
        num_frames = int(self.shape[1] * scale_factor)
        return type(self)(
            mask=new_intervals[0], sampling_interval=sampling_interval,
            num_frames=num_frames)


class ImagingInterval(Interval):
    """Interval object for masking imaging data, one mask per ROI x Trial

    Parameters
    ----------
    mask : BehaviorInterval, ndarray, or list of list of Nx2 arrays
        If ndarray, format should be (ROIs x Time x Trials).
        If list of intervals, should be [ROI][Trial][Nx2 array], where Nx2
        array is start, stop times of each interval.
    num_rois : int
        The number of ROIs to expand a BehaviorInterval to. Only necessary if
        initializing with a BehaviorInterval.

    See also
    --------
    Interval
    BehaviorInterval

    """

    def __init__(self, mask, num_rois=None, **kwargs):
        if isinstance(mask, BehaviorInterval):
            if 'sampling_interval' in kwargs:
                sampling_interval = kwargs.pop('sampling_interval')
                if sampling_interval != mask.sampling_interval:
                    mask = mask.resample(sampling_interval)
            else:
                sampling_interval = mask.sampling_interval
            new_mask = mask.mask
            tiled_mask = np.tile(new_mask.T, (num_rois, 1, 1))
            super(ImagingInterval, self).__init__(
                tiled_mask, sampling_interval=sampling_interval, **kwargs)
        else:
            if isinstance(mask, np.ndarray):
                assert mask.ndim == 3
            super(ImagingInterval, self).__init__(mask, **kwargs)


def _mask_to_intervals(mask):
    # (ROIs x Time X Trial)
    # intervals are [ROIs][Trial][Nx2 array]
    assert np.ndim(mask) == 3

    intervals = []
    for roi_mask in mask:
        intervals.append([])
        for trial_mask in roi_mask.T:
            mask_diff = np.diff(trial_mask.astype('int'))
            starts = np.where(mask_diff == 1)[0]
            stops = np.where(mask_diff == -1)[0]
            if len(stops) and (not len(starts) or stops[0] < starts[0]):
                starts = np.hstack([np.nan, starts])
            if len(starts) and (not len(stops) or starts[-1] > stops[-1]):
                stops = np.hstack([stops, np.nan])

            assert len(starts) == len(stops)

            stacked_intervals = np.vstack([starts, stops]).T

            stacked_intervals += 1  # result of diff is trimmed down by 1

            intervals[-1].append(stacked_intervals)

    return intervals


def _intervals_to_mask(intervals, num_frames):
    mask = np.zeros(
        (len(intervals), num_frames, len(intervals[0])), dtype='bool')

    for roi_idx, roi_intervals in enumerate(intervals):
        for trial_idx, trial_intervals in enumerate(roi_intervals):
            for start, stop in trial_intervals.astype('int'):
                if np.isnan(start):
                    start = 0
                if np.isnan(stop):
                    stop = num_frames
                mask[roi_idx, start:stop, trial_idx] = True

    return mask


def _rescale_intervals(intervals, scale_factor):
    new_intervals = []
    for roi_intervals in intervals:
        new_intervals.append([])
        for trial_intervals in roi_intervals:
            trial_intervals = trial_intervals.astype('float')
            trial_intervals *= scale_factor
            new_intervals[-1].append(trial_intervals)

    return new_intervals

if __name__ == '__main__':
    a = np.array([False, True, True, False, False, True, False])
    b = np.array([True, False, True, True, False, False, True])
    c = np.array([True, False, False, False, False, False, False])
    d = np.array([False, False, False, False, False, False, True])
    mask = np.vstack([a, b, c, d])
    masks = np.dstack([mask, mask[::-1, :]])
    print masks.shape

    interval = Interval(masks)

    assert np.all(masks == _intervals_to_mask(_mask_to_intervals(masks), masks.shape[1]))

    from pudb import set_trace; set_trace()
