"""Functions that generate various time interval objects"""

import numpy as np
from collections import Mapping, defaultdict

from ..classes.interval import BehaviorInterval, ImagingInterval, IntervalDict
import behavior_analysis as ba


def near_positions(expt_grp, positions, nearness=0.1):
    """Calculates the time when a mouse was near a particular location.

    Parameters
    ----------
    expt_grp : lab.ExperimentGroup
    positions : {array-like, dict}
        Either an array-like of normalized positions or a dictionary with
        experiments as keys and values an array of positions
    nearness : float
        How close is 'near'?

    Returns
    -------
    IntervalDict of BehaviorInterval

    """

    if not isinstance(positions, Mapping):
        positions = defaultdict(lambda: positions)

    result = {}
    for expt in expt_grp:
        expt_positions = positions[expt]

        assert all(0. <= position < 1. for position in expt_positions)

        intervals = []
        for position in expt_positions:
            trial_intervals = []
            sampling_intervals = []
            for trial in expt.findall('trial'):
                bd = trial.behaviorData(sampling_interval='actual')
                pos = bd['treadmillPosition']
                sampling_intervals.append(bd['samplingInterval'])

                distance_to_reward = pos - position
                distance_to_reward[distance_to_reward < -0.5] += 1.
                distance_to_reward[distance_to_reward >= 0.5] -= 1.

                assert np.all(distance_to_reward < 0.5)
                assert np.all(distance_to_reward >= -0.5)

                in_window = np.abs(distance_to_reward) <= nearness

                trial_intervals.append(in_window)

            assert all(s == sampling_intervals[0] for s in sampling_intervals)

            intervals.append(BehaviorInterval(
                np.array(trial_intervals),
                sampling_interval=sampling_intervals[0]))

        if len(expt_positions) >= 2:
            expt_result = intervals[0]
            for interval in intervals[1:]:
                expt_result = expt_result | interval
        else:
            expt_result = intervals[0]

        result[expt] = expt_result

    return IntervalDict(result)


def near_rewards(expt_grp, nearness=0.1):
    """Calculates the time when the mouse was near a reward location."""

    positions = {
        expt: expt.rewardPositions(units='normalized') for expt in expt_grp}

    return near_positions(expt_grp, positions=positions, nearness=nearness)


def running_intervals(expt_grp, **kwargs):
    """Calculates running intervals.

    Parameters
    ----------
    kwargs
        All arguments are passed to ba.runningIntervals

    Returns
    -------
    IntervalDict of BehaviorInterval

    """

    result = {}
    for expt in expt_grp:
        expt_result = []
        sampling_intervals = []
        for trial in expt:
            sampling_intervals.append(trial.behavior_sampling_interval())
            expt_result.append(ba.runningIntervals(
                trial, imageSync=False, returnBoolList=True, **kwargs))

        assert all([s == sampling_intervals[0] for s in sampling_intervals])

        result[expt] = BehaviorInterval(
            np.array(expt_result), sampling_interval=sampling_intervals[0])

    return IntervalDict(result)


def place_fields(expt_grp, roi_filter=None):
    """Calculate the times a mouse was within each cell's place field.

    Parameters
    ----------
    expt_grp : lab.classes.pcExperimentGroup
        Must be a pcExperimentGroup to find place place fields

    Returns
    -------
    IntervalDict of ImagingInterval

    """

    result = {}
    pfs = expt_grp.pfs(roi_filter=roi_filter)
    n_bins = float(expt_grp.args['nPositionBins'])
    for expt in expt_grp:
        assert len(expt.findall('trial')) == 1
        bd = expt.find('trial').behaviorData(imageSync=True)
        sampling_interval = expt.frame_period()
        pos = bd['treadmillPosition']
        intervals = []
        for roi_pfs in pfs[expt]:
            in_pfs = np.zeros_like(pos, dtype='bool')
            for start, stop in roi_pfs:
                if start > stop:
                    in_pfs = np.logical_or(
                        in_pfs, np.logical_or(pos < stop / n_bins,
                                              pos >= start / n_bins))
                else:
                    in_pfs = np.logical_or(
                        in_pfs, np.logical_and(pos >= start / n_bins,
                                               pos < stop / n_bins))
            intervals.append(in_pfs)
        intervals_array = np.vstack(intervals)[..., None]
        assert intervals_array.shape == expt.imaging_shape(
            channel=expt_grp.args['channel'],
            label=expt_grp.args['imaging_label'], roi_filter=roi_filter)
        result[expt] = ImagingInterval(
            intervals_array, sampling_interval=sampling_interval)

    return IntervalDict(result)


def in_transient(expt_grp, **trans_kwargs):

    result = {}
    for expt in expt_grp:
        transients = expt.transientsData(**trans_kwargs)
        expt_trans = []
        for roi_trans in transients:
            expt_trans.append([])
            for trial_trans in roi_trans:
                expt_trans[-1].append(np.array(zip(
                    trial_trans['start_indices'], trial_trans['end_indices'])))
        frame_period = expt.frame_period()
        _, num_frames, _ = expt.imaging_shape()
        result[expt] = ImagingInterval(
            expt_trans, sampling_interval=frame_period, num_frames=num_frames)

    return IntervalDict(result)
