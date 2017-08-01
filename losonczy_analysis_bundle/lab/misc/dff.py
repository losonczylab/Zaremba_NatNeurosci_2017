"""dF/F methods for 'on-the-fly' calculation"""

import numpy as np
from scipy import percentile


def mean(im_data):
    """Use the mean of each ROI across all time and trials as the baseline"""
    for i in xrange(im_data.shape[0]):
        m = np.mean(im_data[i, :, :])
        im_data[i, :, :] = (im_data[i, :, :] - m) / m

    return im_data


def median(im_data):
    """Use the median of each ROI across all time and trials as the baseline"""
    for i in xrange(im_data.shape[0]):
        m = np.median(im_data[i, :, :])
        im_data[i, :, :] = (im_data[i, :, :] - m) / m

    return im_data


def sliding_window(im_data, window_width, baseline_percentile):
    """Calculate df/f using a sliding window of given width centered
    around each timepoint and take the nth percentile as the baseline.

    """

    result = np.empty(im_data.shape)
    half_width = np.ceil(window_width / 2)

    for (roi, timepoint, cycle), value in np.ndenumerate(im_data):
        # define the window extent
        if timepoint - half_width < 0:
            window_start = 0
        else:
            window_start = timepoint - half_width

        if timepoint + half_width > im_data.shape[1]:
            window_end = im_data.shape[1]
        else:
            window_end = timepoint + half_width
        # calculate the baseline as a percentile within the window
        baseline = percentile(
            im_data[roi, window_start:window_end, cycle], baseline_percentile)
        baseline = percentile(
            im_data[roi, window_start:window_end, cycle], baseline_percentile)

        # calculate df/f
        result[roi, timepoint, cycle] = (value - baseline) / baseline

    return result


def sliding_window_jia(im_data, t0, t1, t2, baseline_percentile, frame_period):
    """Using the method from Jia et al, Nature Protocols (2011)"""

    smooth_im_data = np.empty(im_data.shape)
    baseline = np.empty(im_data.shape)

    half_width = np.ceil(t1 / 2)

    # smooth raw data and calculate baseline
    for (roi, timepoint, cycle), value in np.ndenumerate(im_data):
        if timepoint - half_width < 0:
            window_start = 0
        else:
            window_start = timepoint - half_width
        if timepoint + half_width > im_data.shape[1]:
            window_end = im_data.shape[1]
        else:
            window_end = timepoint + half_width

        smooth_im_data[roi, timepoint, cycle] = \
            np.mean(im_data[roi, window_start:window_end, cycle])

        if timepoint - t2 < 0:
            window_start = 0
        else:
            window_start = timepoint - t2

        baseline[roi, timepoint, cycle] = percentile(
            smooth_im_data[roi, window_start:timepoint + 1, cycle],
            baseline_percentile)

    # calculate raw df/f
    raw_dfOverF = ((im_data - baseline) / baseline)

    # smooth df/f by a decaying exponential with time constant t0
    smooth_dfOverF = np.empty(im_data.shape)
    tau = np.linspace(0, im_data.shape[1] - 1, im_data.shape[1])
    w0 = np.exp(-tau / t0)
    for (roi, cycle) in np.ndindex(
            raw_dfOverF.shape[0], raw_dfOverF.shape[2]):
        timeseries = raw_dfOverF[roi, :, cycle]
        if np.isnan(np.min(timeseries)):
            smooth_dfOverF[roi, :, cycle] = np.nan
            continue
        else:
            for timepoint in range(timeseries.size):
                if timepoint == 0:
                    smooth_dfOverF[roi, timepoint, cycle] = \
                        timeseries[timepoint]
                else:
                    d = timeseries[:timepoint + 1]
                    w = w0[:timepoint + 1]

                    num = np.trapz(d[::-1] * w, dx=frame_period)
                    den = np.trapz(w, dx=frame_period)
                    smooth_dfOverF[roi, timepoint, cycle] = num / den

    return smooth_dfOverF


def non_running_baseline(im_data, running_intervals):
    dfof = np.empty(im_data.shape)
    for roiIdx, roi in enumerate(im_data):
        for cycle in range(roi.shape[1]):
            runningFrames = []
            for start, stop in running_intervals[cycle]:
                runningFrames.extend(range(stop - start + 1) + start)
            nonRunningFrames = set(range(
                len(roi[:, cycle]))).difference(set(runningFrames))

            baseline = np.mean(roi[list(nonRunningFrames), cycle])
            dfof[roiIdx, :, cycle] = \
                (roi[:, cycle] - baseline) / baseline

    return dfof
