"""Identify place fields"""

import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import random

from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from scipy.stats import mode, percentileofscore
from pycircstat import var

from multiprocessing import Pool
from collections import Counter


def generate_tuning_curve(start_indices, end_indices, response_magnitudes,
                          position_unsynced, behavior_rate, position_synced,
                          imaging_rate, n_position_bins, return_squared,
                          initial_counts):
    """
    start_indices : frame indices where transient events start.
    end_indices   : frame indices where transient events end.
    response_magnitudes : areas under the included transients
    position_synced : animal's position bin as a function of imaging frame.
        nan'ed for frames to exclude. position[start/end] cannot be nan for
        start/end in start_indices, end_indices
    position_unsynced: the animal's position bin as a function of time.  This
        should be nan'ed for frames to exclude.
    behavior_rate: period (sec) of unsynced data
    imaging_rate: period (sec) of image-synced data

    arguments are for a single roi for a single cycle
    """

    initial_counts = initial_counts.copy()

    values = np.zeros(n_position_bins)

    if return_squared:
        values_squared = np.zeros(n_position_bins)

    roi_counter = Counter()
    for event_idx, start, end, mag in zip(
            it.count(), start_indices, end_indices, response_magnitudes):

        start_frame_end = int(
            np.round((start + 1) * imaging_rate / behavior_rate))
        end_frame = int(np.round((end + 1) * imaging_rate / behavior_rate))

        # what position bin is the animal in at time start?
        current_pos = position_synced[start]

        values[current_pos] += mag
        if return_squared:
            values_squared[current_pos] += mag ** 2
        roi_counter.update(position_unsynced[start_frame_end:end_frame])

        # position_unsynced[start_frame_end:end_frame] = np.nan

    initial_counts.subtract(roi_counter)
    counts = np.array([initial_counts[x] * behavior_rate
                       for x in xrange(n_position_bins)])
    # counts = np.array([np.sum(position_unsynced == x) *
    #                   behavior_rate for x in range(n_position_bins)])

    # zeros counts implies no valid observations for the entire belt
    # counts = np.ones(len(values)) * np.ceil(counts.mean())
    assert np.sum(counts) != 0

    if return_squared:
        return values, counts, values_squared

    # assert np.all(values[counts == 0] == 0)

    return values, counts, None


def smooth_tuning_curves(tuning_curves, smooth_length=3, nan_norm=True):

    # mean_zeroed = np.array(tuning_curves)
    # mean_zeroed[np.isnan(tuning_curves)] = 0
    mean_zeroed = np.nan_to_num(tuning_curves)
    gaus_mean = gaussian_filter1d(mean_zeroed, smooth_length, mode='wrap')

    if nan_norm:
        isfinite = np.isfinite(tuning_curves).astype(float)
        sm_isfinite = gaussian_filter1d(isfinite, smooth_length, mode='wrap')
        return gaus_mean / sm_isfinite
    else:
        return gaus_mean


def transients_to_include(transients, frames_to_include):
    """
    transients is the ROI transients data for a single cycle
    frames_to_include is a list of the frames to include for a given cycle

    returns list of start and stop frames for transients that begin in
    frames_to_include (nROIs long)
    """

    start_frames = []
    end_frames = []
    for trans in transients:
        start_frames.append([])
        end_frames.append([])
        for start_frame, end_frame in zip(trans['start_indices'],
                                          trans['end_indices']):

            if start_frame in frames_to_include:
                start_frames[-1].append(start_frame)
                end_frames[-1].append(end_frame)

    return start_frames, end_frames


def _nantrapz_1d(y, x=None, dx=1.0):
    if x is None:
        x_vals = np.arange(0, len(y) * dx, step=dx)
    else:
        x_vals = x

    nans = np.isnan(y)

    return np.trapz(y[~nans], x=x_vals[~nans])


def calcResponseMagnitudes(imData, starts, ends, im_period):
    """
    imData is for a single cycle -- nROIs x nFrames
    starts, ends are for a single cycle
    """

    responses = []
    for roi_imData, roi_starts, roi_ends in it.izip(imData, starts, ends):
        responses.append([])
        for start, end in zip(roi_starts, roi_ends):
            # response = _nantrapz_1d(
            #     roi_imData[start:end + 1], dx=im_period)
            response = 1.
            responses[-1].append(response)
    return responses


def shuffle_transients(true_starts, true_ends, frames_to_include):

    durs = np.array(true_ends) - np.array(true_starts)
    true_starts = [x for (y, x) in sorted(zip(durs, true_starts))][::-1]
    true_ends = [x for (y, x) in sorted(zip(durs, true_ends))][::-1]

    transients_frames = set([])
    shuffle_starts = []
    shuffle_ends = []

    for start, end in zip(true_starts, true_ends):
        valid = False
        while not valid:
            frame = random.sample(frames_to_include, 1)[0]
            trans_frames = set(range(frame, frame + end - start + 1))
            valid = not len(trans_frames.intersection(transients_frames))

        shuffle_starts.append(frame)
        shuffle_ends.append(frame + end - start)
        transients_frames = transients_frames.union(trans_frames)

    return shuffle_starts, shuffle_ends


def calcSpatialResponses(tuning_curve, putative_place_fields):

    responses = []
    for start, end in zip(*putative_place_fields):
        if start <= end:
            # response = np.trapz(tuning_curve[start:end + 1], axis=0)
            # response = np.amax(tuning_curve[start:end + 1])
            response = np.sum(tuning_curve[start:end + 1])
        else:
            # response = np.trapz(tuning_curve[start:], axis=0) + \
            #           np.trapz(tuning_curve[:end + 1], axis=0)
            # response = np.amax([np.amax(tuning_curve[start:]),
            #                    np.amax(tuning_curve[:end + 1])])
            response = np.sum(tuning_curve[start:]) + \
                np.sum(tuning_curve[:end + 1])
        responses.append(response)
    return responses


def calc_spatial_information(inputs):
    event_counts, obs_counts, smooth_length = inputs
    # This is devived from Skaggs with some algebraic simplifications

    info = []
    for roi_events, roi_obs in it.izip(event_counts, obs_counts):

        O_sum = roi_obs.sum()
        idx = np.nonzero(roi_events)[0]

        roi_events = roi_events[idx]
        roi_obs = roi_obs[idx]

        E_sum = roi_events.sum()

        R = roi_events / roi_obs

        i = np.dot(roi_events, np.log2(R)) / E_sum - np.log2(E_sum / O_sum)

        info.append(i)

    return info


def identifyPutativeFields(tuning_curve):
    """Returns the start and end indices of non-zero intervals in
    tuning_curve.  End indices are inclusive (ie tuning_curve[end] > 0)

    Corrects for the wrap-around case
    """

    tuning_curve = np.copy(tuning_curve)
    tuning_curve = np.around(tuning_curve, decimals=3)
    elevated_indices = np.nonzero(tuning_curve)[0]
    if len(elevated_indices) == 0:
        return [[], []]

    putative_starts = []
    putative_ends = []

    putative_start = elevated_indices[0]

    while putative_start is not None:
        putative_starts.append(putative_start)
        # when does it return to zero?
        try:
            putative_end = putative_start + np.amin(
                np.where(tuning_curve[putative_start:] == 0)) - 1
        except ValueError:
            # does not return to 0
            putative_end = len(tuning_curve) - 1

        putative_ends.append(putative_end)

        try:
            putative_start = np.amin(
                elevated_indices[elevated_indices > putative_end])
        except ValueError:
            # no more intervals > 0
            putative_start = None

    # correct the wrap-around case
    if 0 in putative_starts and len(tuning_curve) - 1 in putative_ends and \
            len(putative_starts) > 1:
        putative_starts.pop(0)
        putative_ends[-1] = putative_ends.pop(0)

    return putative_starts, putative_ends


def define_fields(tuning_curve):

    def gaussian_func(x, a, c):
        # gaussian with mean zero and zero vertical displacement
        return a * np.exp(-(x ** 2) / (2 * c ** 2))

    local_maxima = argrelextrema(
        np.around(tuning_curve, decimals=5),np.greater, mode='wrap')[0]
    if len(local_maxima) == 0:
        all_plateaus = argrelextrema(np.array(tuning_curve), np.greater_equal,
                                     mode='wrap')[0]
        non_zero_vals = np.where(np.array(tuning_curve) != 0)[0]
        plateaus = np.intersect1d(all_plateaus, non_zero_vals)

        if len(plateaus) == 0:
            local_maxima = np.array([])
        else:
            first_bins = np.hstack([True, np.diff(plateaus) > 1])

            local_maxima = plateaus[first_bins]

            # Check wraparound case
            if 0 in local_maxima and len(tuning_curve) - 1 in local_maxima:
                local_maxima = local_maxima[1:]

    local_minima = argrelextrema(
        np.around(tuning_curve, decimals=5), np.less_equal, mode='wrap')[0]

    frames = set([])
    mid_point = len(tuning_curve) / 2
    for local_max in local_maxima:
        offset = mid_point - local_max
        rolled_tuning = np.roll(tuning_curve, offset)
        rolled_mins = (local_minima + offset) % len(tuning_curve)

        try:
            left_local_min = np.amax(rolled_mins[rolled_mins < mid_point])
        except ValueError:
            # If there are no mins to the left of the midpoint try to find
            # a point that is 25% of the current peak
            try:
                left_boundary = np.amax(np.where(rolled_tuning[:mid_point] <
                                        0.25 * rolled_tuning[mid_point])[0])
            except ValueError:
                # If it never falls down to 25% of the current peak, fit the
                # entire left half
                left_boundary = 0
        else:
            # Check to see if it tuning falls down to 25% of peak before the
            # closest local min
            try:
                left_peak_edge = np.amax(np.where(rolled_tuning[:mid_point] <
                                         0.25 * rolled_tuning[mid_point])[0])
            except ValueError:
                left_boundary = left_local_min
            else:
                left_boundary = np.amax((left_peak_edge, left_local_min))

        # Same thing for the right side
        try:
            right_local_min = np.amin(rolled_mins[rolled_mins > mid_point])
        except ValueError:
            try:
                right_boundary = mid_point + 1 + np.amin(
                    np.where(rolled_tuning[mid_point + 1:] <
                             0.25 * rolled_tuning[mid_point])[0])
            except ValueError:
                right_boundary = len(tuning_curve) - 1
        else:
            try:
                right_peak_edge = mid_point + 1 + np.amin(
                    np.where(rolled_tuning[mid_point + 1:] <
                             0.25 * rolled_tuning[mid_point])[0])
            except ValueError:
                right_boundary = right_local_min
            else:
                right_boundary = np.amin((right_peak_edge, right_local_min))

        x_data = np.arange(left_boundary, right_boundary + 1) - mid_point

        data_to_fit = rolled_tuning[left_boundary:right_boundary + 1]
        data_to_fit -= np.amin(data_to_fit)
        popt, pcov = curve_fit(
            gaussian_func, x_data, data_to_fit, p0=[.1, 1])

        if np.abs(popt[1]) > len(tuning_curve) / 2.:
            frames = frames.union(range(0, len(tuning_curve)))
        else:
            pf_start = int(local_max - 2 * np.abs(popt[1]))
            pf_end = int(local_max + 2 * np.abs(popt[1]))
            if pf_start < 0:
                frames = frames.union(range(pf_start + len(tuning_curve),
                                            len(tuning_curve)))
                pf_start = 0
            if pf_end >= len(tuning_curve):
                frames = frames.union(range(0, pf_end - len(tuning_curve) + 1))
                pf_end = len(tuning_curve) - 1
            frames = frames.union(range(pf_start, pf_end + 1))

    sorted_frames = sorted(frames)
    gaps = np.where(np.diff(sorted_frames) != 1)[0]

    starts = [sorted_frames[0]]
    ends = []
    for gap in gaps:
        ends.append(sorted_frames[gap])
        starts.append(sorted_frames[gap + 1])
    ends.append(sorted_frames[-1])

    # Correct wraparound case
    # i.e starts = [0, 90], ends = [10, 99]
    if len(starts) > 1 and starts[0] == 0 \
            and ends[-1] == len(tuning_curve) - 1:
        starts[0] = starts.pop()
        ends.pop()

    # filter by area
    areas = []
    for start, end in zip(starts, ends):
        if start <= end:
            area = np.trapz(tuning_curve[start:end + 1])
        else:
            area = np.trapz(tuning_curve[start:]) + \
                np.trapz(tuning_curve[:end + 1])
        areas.append(area)
    maximum = np.amax(areas)
    include = [a >= 0.5 * maximum for a in areas]
    starts = [start for i, start in enumerate(starts) if include[i]]
    ends = [end for i, end in enumerate(ends) if include[i]]

    return starts, ends


def _shuffler(inputs):
    """Calculates shuffled place fields.
    Used to split across pools"""

    (true_starts, true_ends, transient_responses, position_unsynced,
        behav_period, position_synced, framePeriod, frames_to_include,
        nROIs, n_position_bins, initial_counts) = inputs

    shuffle_values = np.zeros((nROIs, n_position_bins))
    shuffle_counts = np.zeros((nROIs, n_position_bins))
    for cycle_true_starts, cycle_true_ends, cycle_responses, cycle_pos, \
            cycle_pos_synced, cycle_frames, cycle_counts in it.izip(
            true_starts, true_ends, transient_responses, position_unsynced,
            position_synced, frames_to_include, initial_counts):

        for roi_idx, roi_starts, roi_ends, roi_responses in zip(
                it.count(), cycle_true_starts, cycle_true_ends,
                cycle_responses):

            shuffle_starts, shuffle_ends = shuffle_transients(
                true_starts=roi_starts, true_ends=roi_ends,
                frames_to_include=cycle_frames)

            v, c, _ = generate_tuning_curve(
                start_indices=shuffle_starts, end_indices=shuffle_ends,
                response_magnitudes=roi_responses,
                position_unsynced=cycle_pos, behavior_rate=behav_period,
                position_synced=cycle_pos_synced, imaging_rate=framePeriod,
                n_position_bins=n_position_bins, return_squared=False,
                initial_counts=cycle_counts)

            shuffle_values[roi_idx] += v
            shuffle_counts[roi_idx] += c

    # shuffled counts may become zero if there's an issue with the behavior
    # sampling rate
    assert np.any(np.sum(shuffle_counts, axis=0)) != 0

    return shuffle_values, shuffle_counts


def binned_positions(expt, imData, frames_to_include, MAX_N_POSITION_BINS):
    """Calculate the binned positions for each cycle"""
    nROIs, nFrames, nCycles = imData.shape
    framePeriod = expt.frame_period()
    behav_period = expt.find('trial').behaviorData()['samplingInterval']
    position_unsynced = []  # position bins as a function of behavioral frame
    position_synced = []  # position bins as a function of imaging frame
    initial_counts = []
    for idx, cycle in enumerate(expt.findall('trial')):

        position_unsynced.append((cycle.behaviorData(
            sampling_interval='actual')['treadmillPosition'] *
            MAX_N_POSITION_BINS).astype(int))

        position_synced.append(np.zeros(nFrames, dtype='int'))

        # exclude frames, e.g. when animal is not running
        exclude_frames = list(set(np.arange(nFrames)).difference(
            set(frames_to_include[idx])))

        for frame in xrange(nFrames):

            start = int(np.round(frame * framePeriod / behav_period))
            end = int(np.round((frame + 1) * framePeriod / behav_period))

            position_array = position_unsynced[idx][start:end]

            assert np.all(position_array >= 0)
            assert np.all(np.isfinite(position_array))
            pos = int(np.mean(position_array))

            if pos not in position_array:
                pos_mode, _ = mode(position_array)
                pos = int(pos_mode)

            assert not np.isnan(pos)

            position_synced[idx][frame] = pos

            if frame in exclude_frames:
                position_unsynced[idx][start:end] = -1

        initial_counts.append(Counter(position_unsynced[idx]))
    return position_unsynced, position_synced, initial_counts


def _calc_information(
        MAX_N_POSITION_BINS, true_values, true_counts, bootstrap_values,
        bootstrap_counts, n_bins_list, smooth_lengths, n_processes):
    """
    Returns
    -------
    true_information : ndarray(ROIs, nbins)
    shuffle_information : ndarray(ROIs, bootstraps, nbins)
    """
    nROIs = len(true_counts)
    n_bootstraps = bootstrap_values.shape[2]
    true_information = np.empty((nROIs, len(n_bins_list)))
    shuffle_information = np.empty((nROIs, n_bootstraps, len(n_bins_list)))
    if n_processes > 1:
        pool = Pool(processes=n_processes)

    for bin_idx, (n_bins, factor_smoothing) in enumerate(zip(
            n_bins_list, smooth_lengths)):
        true_information_by_shift = np.empty(
            (nROIs, MAX_N_POSITION_BINS / n_bins))
        for bin_shift in np.arange(MAX_N_POSITION_BINS / n_bins):
            values = np.roll(true_values, shift=bin_shift, axis=1).reshape(
                [nROIs, n_bins, -1]).sum(2)
            counts = np.roll(true_counts, shift=bin_shift, axis=1).reshape(
                [nROIs, n_bins, -1]).sum(2)

            true_information_by_shift[:, bin_shift] = calc_spatial_information(
                (values, counts, factor_smoothing))

        true_information[:, bin_idx] = np.max(
            true_information_by_shift, axis=1)

        shuffle_information_by_shift = np.empty(
            (nROIs, n_bootstraps, MAX_N_POSITION_BINS / n_bins))
        for bin_shift in np.arange(MAX_N_POSITION_BINS / n_bins):

            # Need to round for non-integer values
            assert np.all(np.around(
                np.std(np.sum(bootstrap_values, axis=1), axis=1), 12) == 0)

            shuffle_values = np.rollaxis(np.roll(
                bootstrap_values, shift=bin_shift, axis=1).reshape(
                [nROIs, n_bins, -1, n_bootstraps]).sum(2), 2, 0)

            assert np.all(np.around(
                np.std(np.sum(shuffle_values, axis=2), axis=0), 12) == 0)

            shuffle_counts = np.rollaxis(np.roll(
                bootstrap_counts, shift=bin_shift, axis=1).reshape(
                [nROIs, n_bins, -1, n_bootstraps]).sum(2), 2, 0)

            if n_processes > 1:
                chunksize = 1 + n_bootstraps / n_processes
                map_generator = pool.imap_unordered(
                    calc_spatial_information, zip(
                        shuffle_values, shuffle_counts,
                        it.repeat(factor_smoothing)),
                    chunksize=chunksize)
            else:
                map_generator = map(
                    calc_spatial_information, zip(
                        shuffle_values, shuffle_counts,
                        it.repeat(factor_smoothing)))

            idx = 0
            for info in map_generator:
                shuffle_information_by_shift[:, idx, bin_shift] = info
                idx += 1
            shuffle_information[:, :, bin_idx] = np.max(
                shuffle_information_by_shift, axis=2)

    if n_processes > 1:
        pool.close()
        pool.join()

    return true_information, np.rollaxis(shuffle_information, 1, 0)


def _calc_variances(
        true_values, true_counts, bootstrap_values, bootstrap_counts):

    true_variances = []
    p_vals = []

    bins = 2 * np.pi * np.arange(0, 1, 1. / len(true_values[0]))
    for values, counts, shuffle_values, shuffle_counts in it.izip(
            true_values, true_counts, bootstrap_values, bootstrap_counts):
        true_value = var(bins, values / counts)
        true_variances.append(true_value)

        roi_shuffles = []
        for shuffle in range(shuffle_values.shape[1]):
            roi_shuffles.append(var(
                bins, shuffle_values[:, shuffle] / shuffle_counts[:, shuffle]))
        p_vals.append(percentileofscore(roi_shuffles, true_value) / 100.)

    return true_variances, p_vals


def _shuffle_bin_counts(
        MAX_N_POSITION_BINS, position_unsynced, position_synced,
        frames_to_include, im_period, behav_period, true_starts,
        true_ends, transient_responses, n_bootstraps, n_processes,
        initial_counts):
    """Create shuffled versions of transient and observation counts per bin"""
    nROIs = len(true_starts[0])
    if n_processes > 1:
        pool = Pool(processes=n_processes)
    inputs = (true_starts, true_ends, transient_responses, position_unsynced,
              behav_period, position_synced, im_period, frames_to_include,
              nROIs, MAX_N_POSITION_BINS, initial_counts)
    if n_processes > 1:
        # chunksize = min(1 + n_bootstraps / n_processes, 200)
        chunksize = 1 + n_bootstraps / n_processes
        map_generator = pool.imap_unordered(
            _shuffler, it.repeat(inputs, n_bootstraps), chunksize=chunksize)
    else:
        map_generator = map(_shuffler, it.repeat(inputs, n_bootstraps))
    bootstrap_values = np.empty((nROIs, MAX_N_POSITION_BINS, n_bootstraps))
    bootstrap_counts = np.empty((nROIs, MAX_N_POSITION_BINS, n_bootstraps))
    bootstrap_idx = 0
    for values, counts in map_generator:
        bootstrap_values[:, :, bootstrap_idx] = values
        bootstrap_counts[:, :, bootstrap_idx] = counts
        bootstrap_idx += 1
    if n_processes > 1:
        pool.close()
        pool.join()
    return bootstrap_values, bootstrap_counts


def find_truth(
        imData, transients, MAX_N_POSITION_BINS, position_unsynced,
        position_synced, frames_to_include, im_period, behav_period,
        initial_counts):
    """
    Returns
    -------
    true_values :
        tranients per position bin. Dims=(nROis, nBins)
    true_counts :
        Number of observation per position bin
    true_starts :
        Imaging frame indices of transients starts.
    true_ends :
        Imaging frame indices of transients stops (inclusive).
    transient_responses : list of list of list
        Used to weight. Currently all ones. Index order [cycle][roi][event].

    """
    nROIs = imData.shape[0]
    true_values = np.zeros((nROIs, MAX_N_POSITION_BINS))
    true_values_squared = np.zeros((nROIs, MAX_N_POSITION_BINS))
    true_counts = np.zeros((nROIs, MAX_N_POSITION_BINS))
    true_starts = []  # by cycle
    true_ends = []  # by cycle
    transient_responses = []  # by cycle

    for idx, cycle_imData, cycle_transients, cycle_pos, \
            cycle_frames_to_include, cycle_counts in it.izip(
                it.count(), np.rollaxis(imData, 2, 0),
                np.rollaxis(transients, 1, 0),
                position_unsynced, frames_to_include, initial_counts):

        starts, ends = transients_to_include(
            cycle_transients, cycle_frames_to_include)
        true_starts.append(starts)
        true_ends.append(ends)

        responses = calcResponseMagnitudes(imData=cycle_imData, starts=starts,
                                           ends=ends,
                                           im_period=im_period)
        transient_responses.append(responses)

        for roi_idx, roi_starts, roi_ends, roi_responses in zip(
                it.count(), starts, ends, responses):

            v, c, vs = generate_tuning_curve(
                start_indices=roi_starts, end_indices=roi_ends,
                response_magnitudes=roi_responses,
                position_unsynced=cycle_pos, behavior_rate=behav_period,
                position_synced=position_synced[idx],
                imaging_rate=im_period,
                n_position_bins=MAX_N_POSITION_BINS, return_squared=True,
                initial_counts=cycle_counts)

            true_values[roi_idx] += v
            true_values_squared[roi_idx] += vs
            true_counts[roi_idx] += c
    return (
        true_values,
        true_values_squared,
        true_counts,
        true_starts,
        true_ends,
        transient_responses,
    )


def id_place_fields(expt, intervals='running', n_position_bins=100,
                    dFOverF='from_file', channel='Ch2', label=None,
                    demixed=False, smooth_length=3, n_bootstraps=1000,
                    confidence=95, transient_confidence=95,
                    n_processes=1, debug=False, isolated=False):

    params = {'intervals': intervals, 'n_position_bins': n_position_bins,
              'dFOverF': dFOverF, 'smooth_length': smooth_length,
              'n_bootstraps': n_bootstraps, 'confidence': confidence,
              'transient_confidence': transient_confidence}
    running_kwargs = {'min_duration': 1.0, 'min_mean_speed': 0,
                      'end_padding': 0, 'stationary_tolerance': 0.5,
                      'min_peak_speed': 5, 'direction': 'forward'}
    # running_kwargs = {'min_duration': 0, 'min_mean_speed': 0,
    #                   'end_padding': 0, 'stationary_tolerance': 2.,
    #                   'min_peak_speed': 0, 'direction': 'forward'}

    # every element of n_bins_list must divide into the first element evenly
    # to alow for fast re-binning during bin-shift calculations
    n_bins_list = [100, 50, 25, 20, 10, 5, 4, 2]
    # same length as n_bins_list
    smooth_lengths = [3, 1, 1, 0, 0, 0, 0, 0]
    assert np.all([n_bins_list[0] % x == 0 for x in n_bins_list[2:]])
    MAX_N_POSITION_BINS = n_bins_list[0]
    assert MAX_N_POSITION_BINS % n_position_bins == 0

    imData = expt.imagingData(
        dFOverF=dFOverF, channel=channel, label=label, demixed=demixed)

    if isolated:
        transients = expt.transSubset('isolated',
            channel=channel, label=label, demixed=demixed, threshold=transient_confidence)
    else:
        transients = expt.transientsData(
            channel=channel, label=label, demixed=demixed, threshold=transient_confidence)

    # imaging framerate
    im_period = expt.frame_period()
    # behavioral data framerate
    behav_period = expt.find('trial').behaviorData()['samplingInterval']

    nROIs, nFrames, nCycles = imData.shape
    if intervals == 'all':
        frames_to_include = [np.arange(nFrames) for x in range(nCycles)]
    elif intervals == 'running':
        running_intervals = expt.runningIntervals(returnBoolList=False,
                                                  **running_kwargs)
        # list of frames, one per list element per cycle
        frames_to_include = [np.hstack([np.arange(start, end) for
                             start, end in cycle]) for
                             cycle in running_intervals]
    else:
        # Assume frames_to_include was passed in directly
        frames_to_include = intervals

    position_unsynced, position_synced, initial_counts = binned_positions(
        expt, imData, frames_to_include, MAX_N_POSITION_BINS)
    true_values, true_values_squared, true_counts, true_starts, true_ends, \
        transient_responses = find_truth(
            imData, transients, MAX_N_POSITION_BINS, position_unsynced,
            position_synced, frames_to_include, im_period, behav_period,
            initial_counts)

    bootstrap_values, bootstrap_counts = _shuffle_bin_counts(
        MAX_N_POSITION_BINS, position_unsynced, position_synced,
        frames_to_include, im_period, behav_period, true_starts,
        true_ends, transient_responses, n_bootstraps, n_processes, initial_counts)

    true_information, shuffle_information = _calc_information(
        MAX_N_POSITION_BINS, true_values, true_counts, bootstrap_values,
        bootstrap_counts, n_bins_list, smooth_lengths, n_processes)

    true_circ_variances, circ_variance_p_vals = _calc_variances(
        true_values, true_counts, bootstrap_values, bootstrap_counts)

    # which num of bins give max difference in info
    shuffle_means = shuffle_information.mean(axis=0)  # rois x bins
    true_diffed = true_information - shuffle_means  # rois x bins
    shuffle_diffed = shuffle_information - shuffle_means  # bootstraps x rois x bins
    # TODO: adjustment for self contribution to the mean

    # take best bin for true and for each shuffle
    optimal_true_info = true_diffed.max(axis=1)
    optimal_shuffle_info = shuffle_diffed.max(axis=2)

    thresholds = np.percentile(optimal_shuffle_info, confidence, axis=0)

    true_values = true_values.reshape([nROIs, n_position_bins, -1]).sum(2)
    true_values_squared = true_values_squared.reshape(
        [nROIs, n_position_bins, -1]).sum(2)
    true_counts = true_counts.reshape([nROIs, n_position_bins, -1]).sum(2)
    if smooth_length > 0:
        true_result = smooth_tuning_curves(true_values / true_counts,
                                           smooth_length=smooth_length,
                                           nan_norm=False)
    else:
        true_result = true_values / true_counts

    pfs = []
    information_p_vals = []
    for roi_idx, tuning_curve, response, threshold, shuffle_scores in it.izip(
            it.count(), true_result, optimal_true_info, thresholds,
            optimal_shuffle_info.T):
        information_p_vals.append(1. - percentileofscore(shuffle_scores, response) / 100.)
        if response > threshold:
            starts, ends = define_fields(tuning_curve)
            if len(starts):
                pfs.append(zip(starts, ends))
            else:
                print('Tuned cell w/ no field: roi ' +
                      '{}, response {}, threshold {}'.format(
                          roi_idx, response, threshold))
        else:
            pfs.append([])

    result = {'spatial_tuning': true_values / true_counts,
              'true_values': true_values,
              'true_counts': true_counts,
              'true_circ_variances': np.array(true_circ_variances),
              'circ_variance_p_vals': np.array(circ_variance_p_vals),
              'std': np.sqrt(true_values_squared / true_counts -
                             (true_values / true_counts) ** 2),
              'pfs': pfs,
              'parameters': params,
              'spatial_information': optimal_true_info,
              'thresholds': thresholds,
              'information_p_values': np.array(information_p_vals)}
    if intervals == 'running':
        result['running_kwargs'] = running_kwargs
    if smooth_length > 0:
        result['spatial_tuning_smooth'] = true_result
        result['std_smooth'] = smooth_tuning_curves(
            result['std'], smooth_length=smooth_length, nan_norm=False)

    if debug:
        fig = plt.figure()
        tuning = np.copy(true_result)
        pcs = np.where(pfs)[0]
        for pc in pcs:
            # bring place cells to the top
            tuning = np.vstack((tuning[pc], tuning))
            tuning = np.delete(tuning, pc + 1, 0)

        # add in a blank row
        arranged = np.vstack((tuning[:len(pcs)], [np.nan] * n_position_bins,
                             tuning[len(pcs):]))

        plt.imshow(arranged, vmin=0, vmax=np.percentile(true_result, 95),
                   interpolation='nearest')

        fig, axs = plt.subplots(10, 10, sharex=True, sharey=True)
        roi_names = expt.roiNames(channel=channel, label=label)
        for idx1 in range(10):
            for idx2 in range(10):
                index = 10 * idx1 + idx2
                if index >= len(pcs):
                    continue
                idx = pcs[index]

                axs[idx1, idx2].plot(true_result[idx])
                axs[idx1, idx2].set_title(roi_names[idx])
                for pf in pfs[idx]:
                    color = plt.cm.Set1(np.random.rand(1))
                    start = pf[0]
                    end = pf[1]
                    if start < end:
                        x_range = np.arange(start, end + 1)
                        y_min = np.zeros(len(x_range))
                        y_max = true_result[idx, start:end + 1]
                        axs[idx1, idx2].fill_between(x_range, y_min, y_max,
                                                     color=color)
                    else:
                        x_range = np.arange(start, true_result.shape[1])
                        y_min = np.zeros(len(x_range))
                        y_max = true_result[idx, start:]
                        axs[idx1, idx2].fill_between(x_range, y_min, y_max,
                                                     color=color)

                        x_range = np.arange(0, end + 1)
                        y_min = np.zeros(len(x_range))
                        y_max = true_result[idx, :end + 1]
                        axs[idx1, idx2].fill_between(x_range, y_min, y_max,
                                                     color=color)
        plt.xlim((0, n_position_bins))
        plt.ylim((-0.02, np.amax(true_result)))

        plt.show()

        result['true_information'] = optimal_true_info
        result['bootstrap_distributions'] = bootstrap_values
        result['thresholds'] = thresholds

    return result
