"""Functions that generate various time interval objects."""

from collections import Mapping, defaultdict

import numpy as np

from ..classes.new_interval import Interval, concat
from . import behavior_analysis as ba
from . import imaging_analysis as ia


def near_positions(expt_grp, positions, nearness=0.1, invert=False):
    """Calculate the time when a mouse was near a particular location.

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
    Interval

    """
    if not isinstance(positions, Mapping):
        positions_by_expt = defaultdict(lambda: positions)
    else:
        positions_by_expt = positions

    intervals = []
    for expt in expt_grp:
        expt_positions = positions_by_expt[expt]

        assert all(0. <= position < 1. for position in expt_positions)

        for trial in expt.findall('trial'):
            bd = trial.behaviorData(sampling_interval='actual')
            pos = bd['treadmillPosition']
            for position in expt_positions:
                distance_to_position = pos - position
                distance_to_position[distance_to_position < -0.5] += 1.
                distance_to_position[distance_to_position >= 0.5] -= 1.

                in_window = np.abs(distance_to_position) <= nearness

                if invert:
                    in_window = ~in_window

                intervals.append(Interval.from_mask(
                    in_window,
                    sampling_interval=trial.behavior_sampling_interval(),
                    data={'trial': trial}))
    return concat(intervals).merge_intervals()


def near_rewards(expt_grp, **kwargs):
    """Calculate the time when the mouse was near a reward location."""
    positions = {
        expt: expt.rewardPositions(units='normalized') for expt in expt_grp}

    return near_positions(expt_grp, positions=positions, **kwargs)


def near_cues(expt_grp, nearness=0.1):
    """Calculate the time when the mouse was near a cue."""
    cue_positions = {}
    for expt in expt_grp:
        cues = expt.belt().cues(normalized=True)
        cue_positions[expt] = list(np.array(cues[['start', 'stop']]).flat)

    return near_positions(expt_grp, positions=cue_positions, nearness=nearness)


def running_intervals(expt_grp, **running_kwargs):
    """Calculate running intervals.

    Parameters
    ----------
    running_kwargs : optional
        All additional arguments are passed to ba.runningIntervals

    Returns
    -------
    Interval

    """
    result = []
    for expt in expt_grp:
        for trial in expt.findall('trial'):
            run_ints = ba.runningIntervals(
                trial, imageSync=False, returnBoolList=False, **running_kwargs)
            # Can't do *= since there is an int-to-float type conversion
            run_ints = run_ints * \
                trial.behavior_sampling_interval()
            for start, stop in run_ints:
                result.append({'trial': trial, 'start': start, 'stop': stop})

    return Interval(result)


def stationary_intervals(expt_grp, **running_kwargs):
    """Calculate stationary/non-running intervals.

    Parameters
    ----------
    running_kwargs : optional
        All additional arguments are passed to ba.runningIntervals

    Returns
    -------
    Interval

    """
    intervals = []
    for expt in expt_grp:
        for trial in expt.findall('trial'):
            run_ints = ba.runningIntervals(
                trial, imageSync=False, returnBoolList=True, **running_kwargs)
            intervals.append(Interval.from_mask(
                ~run_ints,
                sampling_interval=trial.behavior_sampling_interval(),
                data={'trial': trial}))
    return concat(intervals).merge_intervals()


def place_fields(expt_grp, roi_filter=None):
    """Calculate the times a mouse was within each cell's place field.

    Parameters
    ----------
    expt_grp : lab.classes.pcExperimentGroup
        Must be a pcExperimentGroup to find place place fields

    Returns
    -------
    Interval

    """
    result = []
    pfs = expt_grp.pfs(roi_filter=roi_filter)
    rois = expt_grp.rois(roi_filter=roi_filter)
    n_bins = float(expt_grp.args['nPositionBins'])
    for expt in expt_grp:
        assert len(rois[expt]) == len(pfs[expt])
        for trial in expt.findall('trial'):
            bd = trial.behaviorData(
                imageSync=False, sampling_interval='actual')
            sampling_interval = trial.behavior_sampling_interval()
            pos = bd['treadmillPosition']
            for roi, roi_pfs in zip(rois[expt], pfs[expt]):
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
                    result.append(Interval.from_mask(
                        in_pfs, data={'trial': trial, 'roi': roi},
                        sampling_interval=sampling_interval))

    return concat(result).merge_intervals()


def in_transient(expt_grp, **trans_kwargs):
    """Return intervals where each cell was within a transient."""
    transients = ia.transients(expt_grp, **trans_kwargs)

    transients['start'] = transients.apply(
        lambda df: df['start_frame'] * df['trial'].parent.frame_period(),
        axis=1)
    transients['stop'] = transients.apply(
        lambda df: df['stop_frame'] * df['trial'].parent.frame_period(),
        axis=1)

    return Interval(transients, columns=['trial', 'roi', 'start', 'stop'])


def behavior(expt_grp, key):
    """Return interval corresponding to behavior key."""
    behavior_df = expt_grp.behavior_dataframe(expt_grp, key=key, rate=False)
    behavior_df.rename(
        columns={'on_time': 'start', 'off_time': 'stop'}, inplace=True)

    return Interval(behavior_df, columns=['trial', 'start', 'stop'])
