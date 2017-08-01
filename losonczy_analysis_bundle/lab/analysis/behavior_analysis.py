"""Analysis of mouse behavior during in vivo calcium imaging"""

import numpy as np
from matplotlib import pyplot as plt
import itertools as it
from scipy.ndimage.filters import gaussian_filter1d
import warnings

from ..classes import exceptions as exc

from .. import plotting


# def infer_expt_pair_condition(expt1, expt2):

#     same_belt = expt1.get('belt') == expt2.get('belt')
#     same_context = expt1.get('environment') == expt2.get('environment')
#     same_rewards = expt1.get('rewardPositions') == expt2.get('rewardPositions')

#     if same_belt and same_context and same_rewards:
#         return "SameAll"
#     if same_belt and same_context and not same_rewards:
#         return "SameAll_DiffRewards"
#     if same_belt and not same_context and same_rewards:
#         return "DiffCtxs"
#     if not same_belt and same_context and same_rewards:
#         return "DiffBelts"
#     if not same_belt and not same_context:
#         return 'DiffAll'
#     return None


"""
Experiment behavior functions
"""


def calculateRewardedLickIntervals(expt, threshold=1.0, imageSync=False):
    """Separates lick intervals in to rewarded and unrewarded intervals"""

    lick_intervals = expt.lickingIntervals(
        imageSync=imageSync, sampling_interval=None,
        threshold=threshold, returnBoolList=False)
    rewards = []
    for trial in expt.findall('trial'):
        if imageSync:
            sampling_interval = expt.frame_period()
        else:
            sampling_interval = trial.behavior_sampling_interval()
        water = trial.behaviorData(imageSync=False)['water'] \
            / float(sampling_interval)
        rewards.append(water[:, 0] if water.shape[0] > 0 else np.array([]))

    rewarded_intervals = []
    unrewarded_intervals = []

    for trial_idx, intervals_trial, rewards_trial in it.izip(
            it.count(), lick_intervals, rewards):
        rewarded_intervals.append([])
        unrewarded_intervals.append([])
        for interval in intervals_trial:
            if np.any((interval[0] <= rewards_trial)
                      & (interval[1] >= rewards_trial)):
                rewarded_intervals[trial_idx].append(interval)
            else:
                unrewarded_intervals[trial_idx].append(interval)
        rewarded_intervals[-1] = np.array(rewarded_intervals[-1]) \
            if len(rewarded_intervals[-1]) else np.empty((0, 2))
        unrewarded_intervals[-1] = np.array(unrewarded_intervals[-1]) \
            if len(unrewarded_intervals[-1]) else np.empty((0, 2))

    return np.array(rewarded_intervals), np.array(unrewarded_intervals)


"""
Trial behavior functions
"""


def runningIntervals(
        trial, imageSync=True, stationary_tolerance=2.0,
        returnBoolList=False, direction='both', min_duration=0,
        min_mean_speed=0, min_peak_speed=0, end_padding=0,
        preceding_still_time=0):
    """Return running interval start and stop times (Nx2 array).

    Parameters
    ----------
    imageSync : bool
        If True, returns data synced to image frames.
    stationary_tolerance : float
        Amount of time (s) where mouse is still that is allowed before starting
        a new interval.
    returnBoolList : bool
        If True, converts intervals to boolean array that is True when there is
        running.
    direction : {'both', forward', 'backwards'}
        Determines direction of running to include.
    min_duration : float, optional
        Minimum duration (in seconds) of a valid running interval.
    min_mean_speed : float, optional
        Minimum mean speed of a valid running interval.
    min_peak_speed : float, optional
        Minimum peak speed of a valid running interval.
    end_padding : float, optional
        Time (s) to add on to the end of every running interval.

    Examples
    --------
    If running=[0 1 1 0 0 0 0 1 1 1 0] and stationary_tolerance=2,
        returns [[1, 2], [7, 9]].
    Using a smaller stationary_tolerance results in more granular intervals,
        while a larger stationary_tolerance leads to longer running intervals

    Note
    ----
    The last frame of each interval does not actually contain running activity
    They are formatted for easy array slicing

    See Also
    --------
    lickingIntervals

    """
    behaviorData = trial.behaviorData(imageSync=imageSync)

    if imageSync:
        period = trial.parent.frame_period()
    else:
        period = trial.behavior_sampling_interval()

    if 'treadmillPosition' not in behaviorData:
        warnings.warn(
            'Quadrature data not available. Analyzing both directions of' +
            ' motion')
        direction = 'both'
        vel = None
    else:
        if imageSync:
            vel = velocity(trial, imageSync=True, sampling_interval=None)
        else:
            vel = velocity(trial, imageSync=False, sampling_interval='actual')

    if direction == 'both':
        if vel is not None:
            running_inds = np.where(vel != 0)[0]
        else:
            if imageSync:
                running_times = []
                for key in ['treadmillTimes', 'treadmillTimes2']:
                    try:
                        running_times.append(
                            np.where(behaviorData[key] != 0)[0])
                    except KeyError:
                        pass
                if len(running_times):
                    running_inds = np.hstack(running_times)
                    running_inds.sort()
                else:
                    running_inds = np.array([])
            else:
                running_times = []
                for key in ['treadmillTimes', 'treadmillTimes2']:
                    try:
                        treadmill_times = behaviorData[key] / period
                    except KeyError:
                        pass
                    else:
                        if len(treadmill_times):
                            running_times.append(treadmill_times)
                if len(running_times):
                    running_inds = np.hstack(running_times)
                    running_inds.sort()
                else:
                    running_inds = np.array([])

    elif direction == 'forward':
        running_inds = np.where(vel > 0)[0]
    elif direction == 'backwards':
        running_inds = np.where(vel < 0)[0]
    else:
        raise ValueError(
            "Invalid direction, must be one of 'forward', 'backwards', " +
            "or 'both'")

    if running_inds.size:
        running_inds = running_inds.astype('uint32')
        # find end indices where the gap between running frames is above
        # stationary_tolerance
        ends = np.where(
            np.diff(running_inds) > stationary_tolerance / period + 1)[0]
        ends = ends.astype('uint32')
        starts = ends + 1

        # The first one is always a start, and the last is always an end
        ends = np.hstack([ends, running_inds.size - 1])
        starts = np.hstack([0, starts])

        good_inds = np.ones(len(starts), 'bool')

        # Check for various interval validity criteria
        if min_mean_speed or min_peak_speed or min_duration or \
                preceding_still_time:
            for idx, (start, end) in enumerate(zip(starts, ends)):
                if idx == 0:
                    previous_run_frame = 0
                else:
                    previous_run_frame = running_inds[ends[idx - 1]]
                if (running_inds[start] - previous_run_frame) * period < \
                        preceding_still_time:
                    good_inds[idx] = False
                if (running_inds[end] - running_inds[start] + 1) * period < \
                        min_duration:
                    good_inds[idx] = False
                if vel is not None:
                    if np.mean(np.abs(
                        vel[running_inds[start]:running_inds[end] + 1])) \
                            < min_mean_speed or \
                       np.amax(np.abs(
                            vel[running_inds[start]:running_inds[end] + 1])) \
                            < min_peak_speed:
                        good_inds[idx] = False
                else:
                    warnings.warn(
                        'Unable to determine velocity, ignoring speed criteria')

        result = np.array(
            [running_inds[starts[good_inds]], running_inds[ends[good_inds]]]).T

        # pad frames to the ends of the running intervals
        if end_padding and len(result) > 0:
            if imageSync:
                result[:, 1] += int(end_padding / period)
                result = result[
                    result[:, 1] < behaviorData['treadmillTimes'].shape[0]]
            else:
                result[:, 1] += int(end_padding)
                result = result[result[:, 1] <= behaviorData[
                    'recordingDuration']]

            padded_result = []
            interval_idx = 0
            while interval_idx < result.shape[0]:
                if interval_idx != result.shape[0] - 1 and result[
                        interval_idx, 1] >= result[interval_idx + 1, 0]:
                    padded_result.append(
                        [result[interval_idx, 0], result[interval_idx + 1, 1]])
                    interval_idx += 2
                else:
                    padded_result.append(
                        [result[interval_idx, 0], result[interval_idx, 1]])
                    interval_idx += 1

            result = np.array(padded_result)

    else:
        result = np.zeros([0, 2], 'uint32')

    if returnBoolList:
        if imageSync:
            try:
                boolList = np.zeros(
                    behaviorData['treadmillPosition'].shape[0], dtype='bool')
            except KeyError:
                boolList = np.zeros(
                    behaviorData['treadmillTimes'].shape[0], dtype='bool')
        else:
            boolList = np.zeros(
                int(behaviorData['recordingDuration'] /
                    behaviorData['samplingInterval']), dtype='bool')
        for interval in result:
            boolList[interval[0]:interval[1]] = True
        return boolList
    return result


def lickingIntervals(trial, imageSync=False, sampling_interval=None,
                     threshold=2.0, returnBoolList=False):
    """Return licking interval start and stop frames (Nx2 array).

    Parameters
    ----------
    imageSync : bool
        If True, synchronizes the output structure to imaging frames
    sampling_interval : float, optional
        In place of `imageSync`, you can set a particular rate to re-sample the
        data at.
        If left None and `imageSync=False`, defaults to sampling rate of the
        behavior data.
    threshold : float, optional
        Combines intervals separated by less than `threshold` (in seconds)
    returnBoolList : bool
        If True, returns data as a boolean mask of in/out of licking intervals.
        If False, returns interval start/stop frames.

    Examples
    --------
    If sampling_interval=0.5, licking=[0 1 1 0 0 0 0 1 1 1 0] and
        threshold=1 frame, this returns [[1, 2], [7, 9]].
    Using a smaller threshold results in more granular intervals, while a
        larger threshold leads to longer licking intervals

    Note
    ----
    The last frame of each interval does not actually contain licking activity.
    They are formatted for easy array slicing.

    See Also
    --------
    runningIntervals

    """
    if imageSync:
        behaviorData = trial.behaviorData(imageSync=True)
        sampling_interval = trial.parent.frame_period()
    else:
        if sampling_interval is None:
            # Default to behavior data sampling rate
            sampling_interval = trial.behavior_sampling_interval()
        behaviorData = trial.behaviorData(
            imageSync=False, sampling_interval=sampling_interval)

    lickingFrames = behaviorData['licking']
    licking_inds = np.where(lickingFrames != 0)[0]

    if licking_inds.size:
        licking_inds = licking_inds.astype('uint32')
        # end indices where the gap between licking frames is above threshold
        ends = np.where(np.diff(
            licking_inds) > threshold / float(sampling_interval))[0]
        ends = ends.astype('uint32')
        starts = ends + 1
        # The first one is always a start, and the last is always an end
        ends = np.hstack((ends, licking_inds.size - 1))
        starts = np.hstack((0, starts))
        result = np.array([licking_inds[starts], licking_inds[ends] + 1]).T
    else:
        result = np.zeros([0, 2])

    if returnBoolList:
        boolList = np.zeros(len(lickingFrames), dtype='bool')
        for interval in result:
            boolList[interval[0]:interval[1] + 1] = True
        return boolList
    else:
        return result


def lickCount(trial, startTime=0, endTime=-1, duration_threshold=0.1):
        """Count licks in a given interval if entire lick falls
        within interval

        """

        behavior_data = trial.behaviorData()
        if 'licking' not in behavior_data:
            print "No licking data"
            return -1
        if endTime == -1:
            endTime = behavior_data['recordingDuration']
        licks = [lick for lick in behavior_data['licking']
                 if lick[0] >= startTime and lick[1] <= endTime]
        licks = [lick[0] for lick in licks if
                 (lick[1] - lick[0]) <= duration_threshold]
        return len(licks)


def absolutePosition(trial, imageSync=True, sampling_interval=None):
    """Returns the normalized absolute position of the mouse at each imaging time frame

    Keyword arguments:
    imageSync -- if True, syncs to imaging data

    absolutePosition % 1 = behaviorData()['treadmillPosition']

    """

    assert not (imageSync and sampling_interval is not None)

    if not imageSync and sampling_interval is None:
        raise(Exception,
            "Should be either image sync'd or at an explicit sampling " +
            "interval, defaulting to 'actual' sampling interval")
        sampling_interval = 'actual'

    bd = trial.behaviorData(
        imageSync=imageSync, sampling_interval=sampling_interval)
    try:
        position = bd['treadmillPosition']
    except KeyError:
        raise exc.MissingBehaviorData(
            'No treadmillPosition, unable to calculate absolute position')

    # if not imageSync:
    #     full_position = np.empty(
    #         int(bd['recordingDuration'] / bd['samplingInterval']))
    #     for tt, pos in position:
    #         full_position[int(tt / bd['samplingInterval']):] = pos
    #     position = full_position

    lap_starts = np.where(np.diff(position) < -0.5)[0]
    lap_back = np.where(np.diff(position) > 0.5)[0].tolist()
    lap_back.reverse()

    # Need to check for backwards steps around the lap start point
    if len(lap_back) > 0:
        next_back = lap_back.pop()
    else:
        next_back = np.inf

    for start in lap_starts:
        if next_back < start:
            position[next_back + 1:] -= 1
            position[start + 1:] += 1
            if len(lap_back) > 0:
                next_back = lap_back.pop()
            else:
                next_back = np.inf
        else:
            position[start + 1:] += 1

    return position


def velocity(trial, imageSync=True, sampling_interval=None, belt_length=200,
             smoothing=None, window_length=5, tick_count=None):
    """Return the velocity of the mouse.

    Parameters
    ----------
    imageSync : bool
        If True, syncs to imaging data.
    belt_length : float
        Length of belt, will return velocity in units/second.
    smoothing {None, str}
        Window function to use, should be 'flat' for a moving average or
        np.'smoothing' (hamming, hanning, bartltett, etc.).
    window_length int
        Length of window function, should probably be odd.
    tick_count : float
        if not None velocity is calculated based on the treadmill
        times by counting ticks and dividing by the tick_count. i.e.
        tick _count should be in ticks/m (or ticks/cm) to get m/s (cm/s)
        returned.

    """
    assert not (imageSync and sampling_interval is not None)

    if not imageSync and sampling_interval is None:
        warnings.warn(
            "Should be either image sync'd or at an explicit sampling " +
            "interval, defaulting to 'actual' sampling interval")
        sampling_interval = 'actual'

    try:
        b = trial.parent.belt().length()
        assert b > 0
        belt_length = b
    except (exc.NoBeltInfo, AssertionError):
        warnings.warn('No belt information found for experiment %s.  \nUsing default belt length = %f' % (str(trial.parent), belt_length))

    if tick_count is not None:
        bd = trial.behaviorData(imageSync=imageSync)
        times = bd['treadmillTimes']
        duration = bd['recordingDuration']
        if imageSync:
            times = np.where(times != 0)[0] * trial.parent.frame_period()
            bincounts = np.bincount(
                times.astype(int), minlength=duration)[:duration]
            bincounts = bincounts.astype(float) / tick_count
            interpFunc = scipy.interpolate.interp1d(
                range(len(bincounts)), bincounts)
            xnew = np.linspace(
                0, len(bincounts) - 1, len(bd['treadmillTimes']))
            return interpFunc(xnew)
        else:
            bincounts = np.bincount(
                times.astype(int), minlength=duration)[:duration]
            bincounts = bincounts.astype(float) / tick_count
            return bincounts

    try:
        position = absolutePosition(
            trial, imageSync=imageSync, sampling_interval=sampling_interval)
    except exc.MissingBehaviorData:
        raise exc.MissingBehaviorData(
            'Unable to calculate position based velocity')

    if imageSync:
        samp_int = trial.parent.frame_period()
    elif sampling_interval == 'actual':
        samp_int = trial.behavior_sampling_interval()
    else:
        samp_int = sampling_interval

    vel = np.hstack([0, np.diff(position)]) * belt_length / samp_int

    if smoothing is not None and np.any(vel != 0):
        if smoothing == 'flat':  # moving average
            w = np.ones(window_length, 'd')
        else:
            # If 'smoothing' is not a valid method this will throw an AttributeError
            w = eval('np.' + smoothing + '(window_length)')
        s = np.r_[vel[window_length - 1:0:-1], vel, vel[-1:-window_length:-1]]
        vel = np.convolve(w / w.sum(), s, mode='valid')
        # Trim away extra frames
        vel = vel[window_length / 2 - 1:-window_length / 2]

    return vel


"""
ExperimentGroup functions
"""


def averageBehavior(exptGrp, ax=None, key='velocity', sampling_interval=1,
                    smooth_length=None, trim_length=None):
    """Plots the average behavior data over the course of the experiment
        For example, average lick rate over time since experiment start

    Keyword arguments:
    key -- behavior data to plot
    sampling_interval -- sampling interval of final output (in seconds)
    smooth_length -- smoothing window length in seconds
    trim_length -- length to trim the final average to (in seconds)

    """

    data_sum = np.array([])
    data_count = np.array([])

    for expt in exptGrp:
        for trial in expt.findall('trial'):
            try:
                bd = trial.behaviorData(imageSync=False)
                bd['samplingInterval']
                bd['recordingDuration']
            except (exc.MissingBehaviorData, KeyError):
                continue
            if bd['samplingInterval'] > sampling_interval:
                warnings.warn(
                    "{}_{}: Sampling interval too low, skipping experiment.".format(
                        expt.parent.get('mouseID'), expt.get('startTime')))
                continue
            if key == 'velocity':
                try:
                    vel = velocity(
                        trial, imageSync=False, sampling_interval='actual',
                        smoothing=None)
                except exc.MissingBehaviorData:
                    warnings.warn(
                        "{}_{}: Unable to determine velocity, skipping experiment.".format(
                            expt.parent.get('mouseID'), expt.get('startTime')))
                    continue
                else:
                    data = np.zeros(
                        int(bd['recordingDuration'] / sampling_interval))
                    starts, step = np.linspace(0, len(vel), num=len(data),
                                               retstep=True)
                    for idx, start in enumerate(starts):
                        data[idx] = np.mean(
                            vel[int(start):int(np.ceil(start + step))])
                    intervals = None
            elif key == 'running':
                intervals = runningIntervals(trial, imageSync=False) * \
                    bd['samplingInterval']
            else:
                try:
                    intervals = bd[key]
                except KeyError:
                    continue
            if intervals is not None:
                data = np.zeros(
                    int(bd['recordingDuration'] / sampling_interval))
                for start, stop in intervals:
                    if np.isnan(start):
                        start = 0
                    if np.isnan(stop):
                        stop = bd['recordingDuration']
                    data[int(start / sampling_interval):
                         int(np.ceil(stop / sampling_interval))] = 1

            if len(data) > len(data_sum):
                data_sum = np.hstack(
                    [data_sum, np.zeros(len(data) - len(data_sum))])
                data_count = np.hstack(
                    [data_count, np.zeros(len(data) - len(data_count))])
            data_sum[:len(data)] += data
            data_count[:len(data)] += 1

    if np.sum(data_count) == 0:
        return None

    final_average = data_sum / data_count

    if trim_length is not None:
        final_average = final_average[:(trim_length / sampling_interval)]

    if smooth_length is not None:
        smooth = int(smooth_length / sampling_interval)
        final_average = gaussian_filter1d(
            final_average, smooth if smooth % 2 == 0 else smooth + 1)

    if ax is not None:
        ax.plot(
            np.linspace(0, len(final_average) * float(sampling_interval),
                        len(final_average)),
            final_average, label=exptGrp.label())

        ax.set_xlabel('Time (s)')
        if key == 'velocity':
            ax.set_ylabel('Average velocity')
        else:
            ax.set_ylabel('Average activity (% of trials)')
        ax.set_title('{} averaged across trials'.format(key))

    return final_average


def getBehaviorTraces(
        exptGrp, stimulus_key, data_key, pre_time=5, post_time=5,
        sampling_interval=0.01, imageSync=False, use_rebinning=False,
        deduplicate=False):
    """Grab behavior data traces triggered by a stimulus.

    Parameters
    ----------
    exptGrp : lab.ExperimentGroup
        contains the group of experiments to get the triggered traces.
    stimulus_key : {str, dict}
        stimulus for triggering the traces. If a string, it should be a key in
        behaviorData. If a dict, then the keys should be experiments. For each
        experiment key the values should be a list of start times.
    data_key : str
        the behavior data type, e.g., velocity.
    pre_time, post_time : float
        time (in seconds) before and after stimuli
    sampling_interval : float
        sampling interval to convert all behavior data to
    imageSync : bool
        if True, sync to the imaging interval, defaults to False.
    use_rebinning : bool
        if True, use average of points to resample, and not interpolation.
        Most useful for downsampling binary signals such as licking. Defaults
        to False.
    deduplicate : bool
         if True, then triggers within the pre_time+post_time window of the
         first trigger is ignored. Defaults to False.

    Returns
    -------
    pandas.DataFrame
        Each row is a behavior trace from a single triggered event.

    Notes
    -----
    Returned dataframe has columns:
        expt
            The experiment that this trace belongs to.
        stimulus
            The stimulus type that triggered the trace.
        dataKey
            The type of behavior trace that is stored, e.g., velocity.
        stimStart
            The frame # that triggered the trace.
        data
            The actual behavior trace.
        time
            The time values corresponding to the behavior trace, with the
            triggered event centered at 0.
        lapNum
            The lap # that the triggered event occured.

    """
    df = dict()

    df["expt"] = []
    df["stimulus"] = []
    df["dataKey"] = []
    df["stimStart"] = []
    df["data"] = []
    df["time"] = []
    df["lapNum"] = []

    for expt in exptGrp:
        data_raw = []
        data_lapNum = []
        if isinstance(stimulus_key, dict):
            try:
                starts = stimulus_key[expt]
            except KeyError:
                continue
        else:
            try:
                starts = stimStarts(expt, stimulus_key, imageSync=imageSync,
                                    deduplicate=deduplicate,
                                    duplicate_window=pre_time + post_time)
            except (exc.MissingBehaviorData, KeyError):
                continue

        if imageSync:
            sampling_interval = expt.frame_period()
            stim_starts = starts
        else:
            stim_starts = [np.around(trial / sampling_interval).astype(int)
                           for trial in starts]

        pre_frames = int(pre_time / sampling_interval)
        post_frames = int(post_time / sampling_interval)

        if data_key == 'velocity':
            if imageSync:
                behaviorData = [velocity(
                                trial, imageSync=imageSync)
                                for trial in expt.findall('trial')]
            else:
                behaviorData = [velocity(
                                trial, imageSync=imageSync, sampling_interval=sampling_interval)
                                for trial in expt.findall('trial')]
        else:
            try:
                if imageSync:
                    behaviorData = [trial.behaviorData(
                                    imageSync=imageSync,
                                    use_rebinning=use_rebinning)[data_key]
                                    for trial in expt.findall('trial')]
                else:
                    behaviorData = [trial.behaviorData(
                                    imageSync=imageSync, sampling_interval=sampling_interval,
                                    use_rebinning=use_rebinning)[data_key]
                                    for trial in expt.findall('trial')]
            except KeyError:
                continue

        try:
            if imageSync:
                lapNums = [np.array(absolutePosition(
                    trial, imageSync=imageSync)).astype("int32")
                    for trial in expt.findall("trial")]
            else:
                lapNums = [np.array(absolutePosition(
                    trial, imageSync=imageSync, sampling_interval=sampling_interval)).astype("int32")
                    for trial in expt.findall("trial")]
        except exc.MissingBehaviorData:
            lapNums = [None for trial in expt.findall('trial')]

        for stimFrames, data, lapNum in it.izip(stim_starts, behaviorData, lapNums):
            for stim in stimFrames:
                if np.isnan(stim):
                    stim = 0
                # Check for running off the ends
                if stim - pre_frames >= 0:
                    data_start = 0
                    start_frame = stim - pre_frames
                else:
                    data_start = pre_frames - stim
                    start_frame = 0
                if stim + post_frames < len(data):
                    data_end = pre_frames + post_frames + 1
                    stop_frame = stim + post_frames + 1
                else:
                    data_end = len(data) - stim - post_frames - 1
                    stop_frame = len(data)

                dataRow = np.empty(pre_frames + post_frames + 1)
                dataRow.fill(np.nan)
                dataRow[data_start:data_end] = data[start_frame:stop_frame]
                data_raw.append(dataRow)

                if lapNum is not None:
                    if(stim >= lapNum.size):
                        stim = lapNum.size - 1
                    data_lapNum.append(lapNum[stim])
                else:
                    data_lapNum.append(np.nan)

        numTraces = len(data_raw)
        df["expt"].extend([expt] * numTraces)
        df["stimulus"].extend([stimulus_key] * numTraces)
        df["dataKey"].extend([data_key] * numTraces)
        df["stimStart"].extend([val for sublist in stim_starts for val in sublist])
        df["data"].extend(data_raw)
        df["time"].extend([np.r_[-(pre_frames * sampling_interval):
                                 ((post_frames + 1) * sampling_interval):sampling_interval]] * numTraces)
        df["lapNum"].extend(data_lapNum)

    return pd.DataFrame(df)


def behaviorPSTH(exptGrp, stimuli_key, data_key, pre_time=5,
                 post_time=10, sampling_interval=0.01, smoothing=None,
                 window_length=1):
    """calculates a PSTH of behavior data versus a stimuli

    Parameters
    ----------
    exptGrp: lab.ExperimentGroup
        contains the group of experiments to calculate the PSTH on
    stimuli_key: str
        stimulus to trigger the PSTH, should be a key in behaviorData or
        'running' which will be the start of running intervals
    data_key: str
        behaviorData key used to generate the histogram
    pre_time, post_time: float
        time (in seconds) before and after the stimuli
    sampling_interval: float
        sampling interval to convert all behavior data to
    smoothing: func
        window function to use, should be 'flat' for a moving average
        or np.'smoothing' (hamming, hanning, bartltett, etc.)
    window_length: float
        length of smoothing window function in seconds

    Returns
    -------
    numpy.ndarray
        the PSTH of the behavior data triggered on the stimulus for the exptGrp

    """

    pre_frames = int(pre_time / sampling_interval)
    post_frames = int(post_time / sampling_interval)
    window_length = int(window_length / sampling_interval)
    if window_length % 2 == 0:
        window_length += 1

    data_sum = np.zeros(pre_frames + post_frames + 1)
    data_count = np.zeros(data_sum.shape)
    for expt in exptGrp:
        try:
            starts = stimStarts(expt, stimuli_key, imageSync=False)
        except (exc.MissingBehaviorData, KeyError):
            continue
        stim_starts = [np.around(trial / sampling_interval).astype(int) for
                       trial in starts]
        if data_key == 'velocity':
            behaviorData = [velocity(
                trial, imageSync=False, sampling_interval=sampling_interval)
                for trial in expt.findall('trial')]
        else:
            try:
                behaviorData = [trial.behaviorData(
                    sampling_interval=sampling_interval)[data_key]
                    for trial in expt.findall('trial')]
            except KeyError:
                continue
        for stimFrames, data in it.izip(stim_starts, behaviorData):
            for stim in stimFrames:
                if np.isnan(stim):
                    stim = 0
                # Check for running off the ends
                if stim - pre_frames >= 0:
                    data_start = 0
                    start_frame = stim - pre_frames
                else:
                    data_start = pre_frames - stim
                    start_frame = 0
                if stim + post_frames < len(data):
                    data_end = len(data_sum)
                    stop_frame = stim + post_frames + 1
                else:
                    data_end = len(data) - stim - post_frames - 1
                    stop_frame = len(data)

                data_sum[data_start:data_end] += data[start_frame:stop_frame]
                data_count[data_start:data_end] += 1

    result = data_sum / data_count

    if smoothing is not None:
        if smoothing == 'flat':  # moving average
            w = np.ones(window_length, 'd')
        else:
            # If 'smoothing' is not a valid method, will throw AttributeError
            w = eval('np.' + smoothing + '(window_length)')
        s = np.r_[result[window_length - 1:0:-1], result,
                  result[-1:-window_length:-1]]
        tmp = np.convolve(w / w.sum(), s, mode='valid')
        # Trim away extra frames
        result = tmp[window_length / 2 - 1:-window_length / 2]

    return result


def plotBehaviorPSTH(exptGrp, stimulus_key, data_key, ax,
                     pre_time=5, post_time=10, color="b", **kwargs):
    """caltulates and plots a PSTH of behavior data vs a stimuli.

    Parameters
    ----------
    exptGrp: lab.ExperimentGroup
        contains the group of experiments to plot the PSTH on
    stimulus_key: str
        stimulus to trigger the PSTH, should be a key in behaviorData or
        'running' which will be the start of running intervals
    data_key: str
        stimulus to trigger the PSTH, should be a key in behaviorData or
        'running' which will be the start of running intervals
    ax: matplotlib.axes
        the PSTH will be plotted on the axes instance.
    pre_time, post_time: float
        time (in seconds) before and after the stimulus
    color:
        the color of the PSTH lines, use any convention accepted by matplotlib
    **kwargs: dict
        see BehaviorPSTH for other keyword arguments

    Returns
    -------
    None

    """
    result = behaviorPSTH(exptGrp, stimulus_key, data_key,
                          pre_time=pre_time, post_time=post_time, **kwargs)

    xAxis = np.linspace(-pre_time, post_time, len(result))
    ax.plot(xAxis, result, label=exptGrp.label(), color=color)

    ax.axvline(0, 0, 1, linestyle='dashed', color='k')
    ax.set_xlim((-pre_time, post_time))

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean data')
    ax.set_title('{} triggered {} PSTH'.format(stimulus_key, data_key))


def positionOccupancy(exptGrp, ax=None, nBins=100, normed=True, showBelt=True,
                      running_only=False, running_kwargs=None,
                      **plot_kwargs):
    """Calculate and plot the time spent at each position on the belt"""

    if not exptGrp.sameBelt():
        warnings.warn('Not all experiments recorded on same belt')

    binSize = 1.0 / nBins
    binStarts = np.arange(0, 1.0, binSize)

    # framePeriod will be equal to the slowest behavior data sampling
    # interval of all the trials
    framePeriod = 0
    for expt in exptGrp:
        for trial in expt.findall('trial'):
            framePeriod = np.amax(
                [framePeriod, trial.behavior_sampling_interval()])

    occupancy = np.zeros(nBins)
    for expt in exptGrp:
        for trial in expt.findall('trial'):
            treadmillPosition = trial.behaviorData(
                imageSync=False,
                sampling_interval=framePeriod)['treadmillPosition']
            if running_only:
                if running_kwargs:
                    running_intervals = runningIntervals(
                        trial, returnBoolList=True,
                        imageSync=False, **running_kwargs)
                else:
                    running_intervals = runningIntervals(
                        trial, returnBoolList=True,
                        imageSync=False)
                treadmillPosition[~running_intervals] = -1

            for bin_ind, bin_start in enumerate(binStarts):
                bins = np.logical_and(
                    treadmillPosition >= bin_start,
                    treadmillPosition < bin_start + binSize)
                occupancy[bin_ind] += np.sum(bins) * framePeriod

    if normed:
        occupancy /= np.sum(occupancy)

    if ax:
        ax.plot(binStarts, occupancy, **plot_kwargs)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Position')
        if normed:
            ax.set_ylabel('Time (percent)')
        else:
            ax.set_ylabel('Time (s)')

        if showBelt and exptGrp.sameBelt():
            exptGrp[0].belt().addToAxis(ax)
        ax.legend(frameon=False, loc='best')

    return occupancy


"""
ExperimentGroup compare functions
"""


def compareLickRate(exptGrps, ax=None):

    if ax is None:
        ax = plt.axes()

    bar_labels = [exptGrp.label() if exptGrp.label() is not None else
                  'Group {}'.format(idx + 1)
                  for idx, exptGrp in enumerate(exptGrps)]

    lickRates = []
    for exptGrp in exptGrps:
        lickRates.append(
            {mouseID: [] for mouseID in set([expt.parent.get('mouseID')
             for expt in exptGrp])})
        for expt in exptGrp:
            for trial in expt.findall('trial'):
                bd = trial.behaviorData()
                duration = bd['recordingDuration']
                try:
                    lick_rate = lickCount(trial) / duration
                except KeyError:
                    pass
                else:
                    # If lick rate is exactly 0, assume recording did not work
                    if lick_rate > 0:
                        lickRates[-1][expt.parent.get('mouseID')].append(
                            lick_rate)

    values = [
        [exptGrp[mouseID] for mouseID in exptGrp] for exptGrp in lickRates]
    group_labels = [[mouseID for mouseID in exptGrp] for exptGrp in lickRates]
    plotting.scatter_1d(ax, values=values, group_labels=group_labels,
                        bar_labels=bar_labels)
    ax.set_title('Lick rate compare')
    ax.set_ylabel('Lick rate (Hz)')


def compareLapRate(exptGrps, ax=None):

    if ax is None:
        ax = plt.axes()

    bar_labels = [exptGrp.label() if exptGrp.label() is not None
                  else 'Group {}'.format(idx + 1)
                  for idx, exptGrp in enumerate(exptGrps)]

    lapRates = []
    for exptGrp in exptGrps:
        lapRates.append({mouseID: [] for mouseID in
                        set([expt.parent.get('mouseID') for expt in exptGrp])})
        for expt in exptGrp:
            for trial in expt.findall('trial'):
                bd = trial.behaviorData()
                duration = bd['recordingDuration']
                try:
                    lap_rate = np.sum(bd['lapCounter'][:, 1] == 1) \
                        / duration * 60
                except KeyError:
                    pass
                else:
                    lapRates[-1][expt.parent.get('mouseID')].append(lap_rate)

    values = [
        [exptGrp[mouseID] for mouseID in exptGrp] for exptGrp in lapRates]
    group_labels = [[mouseID for mouseID in exptGrp] for exptGrp in lapRates]
    plotting.scatter_1d(ax, values=values, group_labels=group_labels,
                        bar_labels=bar_labels)
    ax.set_title('Lap rate compare')
    ax.set_ylabel('Lap rate (laps/minute)')


def compareBehaviorPSTH(exptGrps, stimuli_key, data_key, ax, pre_time=4,
                        post_time=10, **kwargs):
    """Compare behavior PSTHs. See exptGrp.behaviorPSTH for details.

    Parameters
    ----------
    exptGrps: iterable
        contains an iterable of lab.ExperimentGroup instances. Plot the
        BehaviorPSTH for each ExperimentGroup instance on the same axes.
    stimuli_key: str
        stimulus to trigger the PSTH, should be a key in behaviorData or
        'running' which will be the start of running intervals
    data_key: str
        stimulus to trigger the PSTH, should be a key in behaviorData or
        'running' which will be the start of running intervals
    ax: matplotlib.axes
        the PSTHs will be plotted on the axes instance.
    pre_time, post_time: float
        time (in seconds) before and after the stimulus
    **kwargs: dict
        see BehaviorPSTH for other keyword arguments


    Returns
    -------
    None
    """

    for exptGrp in exptGrps:
        result = behaviorPSTH(
            exptGrp, stimuli_key, data_key, pre_time=pre_time,
            post_time=post_time, **kwargs)

        xAxis = np.linspace(-pre_time, post_time, len(result))
        ax.plot(xAxis, result, label=exptGrp.label())

    ylim = ax.get_ylim()
    ax.vlines(0, 0, 1, linestyles='dashed', color='k')
    ax.set_ylim(ylim)
    ax.set_xlim((-pre_time, post_time))

    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean data')
    ax.set_title('{} triggered {} PSTH'.format(stimuli_key, data_key))


def stimStarts(expt, stimulus, exclude_paired_from_single=True,
               imageSync=True, deduplicate=False, duplicate_window=None):
    """Return stimulus start times, formatted for psth().
    Returns list of np.arrrays, one item per trial and then an array of
    start frames if imageSync is True and start times if False

    If the stim is present at the start of the trial, does NOT return the first
    frame.

    """

    POST_STIM_DELAY = 10.0
    REWARDED_RUN_INTERVAL = 3.0

    if stimulus == 'air':
        stimulus = 'airpuff'

    if stimulus == 'all':
        stimuli = expt.stimuli()
        all_starts = stimStarts(expt, stimuli[0], imageSync=imageSync)
        for stim in stimuli[1:]:
            starts = stimStarts(expt, stim, imageSync=imageSync)
            for trial_idx, trial_starts in enumerate(starts):
                all_starts[trial_idx] = np.sort(np.unique(np.hstack(
                    (all_starts[trial_idx], trial_starts))))
        return all_starts

    if stimulus == 'running' or stimulus == 'running_start':
        starts = [interval[:, 0] * trial.behavior_sampling_interval()
                  for interval, trial in zip(
                      expt.runningIntervals(imageSync=False),
                      expt.findall('trial'))]
    elif stimulus in ['running_start_5', 'running_start_5_5']:
        starts = [interval[:, 0] * trial.behavior_sampling_interval()
                  for interval, trial in zip(
                      expt.runningIntervals(
                          imageSync=False, min_duration=5,
                          preceding_still_time=5 if '5_5' in stimulus else 0),
                      expt.findall('trial'))]
    elif stimulus in [
            'running_stop_5', 'running_stop_5_off', 'running_stop_5_5',
            'running_stop_5_5_off']:
        starts = [interval[:, 1] * trial.behavior_sampling_interval()
                  for interval, trial in zip(
                      expt.runningIntervals(
                          imageSync=False, min_duration=5,
                          preceding_still_time=5 if '5_5' in stimulus else 0),
                      expt.findall('trial'))]
    elif 'running_stop' in stimulus:
        starts = [interval[:, 1] * trial.behavior_sampling_interval()
                  for interval, trial in zip(
                  expt.runningIntervals(imageSync=False),
                  expt.findall('trial'))]
        if 'rewarded' in stimulus:
            water = [trial.behaviorData(imageSync=False)['water']
                     for trial in expt.findall('trial')]
            result = []
            for water_trial, run_trial in it.izip(water, starts):
                frame_diff = np.abs(run_trial.reshape((-1, 1)) -
                                    water_trial.reshape((1, -1)))
                rewarded = np.any(
                    frame_diff < REWARDED_RUN_INTERVAL, axis=1)
                if 'unrewarded' in stimulus:
                    result.append(run_trial[~rewarded])
                else:
                    result.append(run_trial[rewarded])
            starts = result
    elif stimulus == 'running_no_stim':
        starts = []
        stim_time = expt.stimulusTime()
        for trial_running, trial in zip(
                expt.runningIntervals(imageSync=False),
                expt.findall('trial')):
            trial_interval = trial.behavior_sampling_interval()
            starts.append(np.array(
                [interval[0] * trial_interval
                 for interval in trial_running
                 if interval[1] * trial_interval < stim_time or
                 interval[0] * trial_interval > stim_time +
                 POST_STIM_DELAY]))
    elif stimulus == 'running_stim':
        starts = []
        stim_time = expt.stimulusTime()
        for trial_running, trial in zip(
                expt.runningIntervals(imageSync=False),
                expt.findall('trial')):
            trial_interval = trial.behavior_sampling_interval()
            starts.append(np.array(
                [interval[0] * trial_interval
                 for interval in trial_running
                 if interval[0] * trial_interval >= stim_time and
                 interval[1] * trial_interval < stim_time +
                 POST_STIM_DELAY]))
    elif stimulus == 'running_stim_no_pair':
        starts = []
        stim_time = expt.stimulusTime()
        for trial_running, trial in zip(
                expt.runningIntervals(imageSync=False),
                expt.findall('trial')):
            if 'Paired' in trial.get('stimulus', ''):
                continue
            trial_interval = trial.behavior_sampling_interval()
            starts.append(np.array(
                [interval[0] * trial_interval
                 for interval in trial_running
                 if interval[0] * trial_interval >= stim_time and
                 interval[1] * trial_interval < stim_time +
                 POST_STIM_DELAY]))
    elif 'running_stim_' in stimulus:
        stim = stimulus[13:]
        starts = []
        stim_time = expt.stimulusTime()
        for trial_running, trial in zip(
                expt.runningIntervals(imageSync=False),
                expt.findall('trial')):
            if trial.get('stimulus', '') != stim:
                continue
            trial_interval = trial.behavior_sampling_interval()
            starts.append(np.array(
                [interval[0] * trial_interval
                 for interval in trial_running
                 if interval[0] * trial_interval >= stim_time and
                 interval[1] * trial_interval < stim_time +
                 POST_STIM_DELAY]))
    elif stimulus == 'licking':
        starts = [interval[:, 0] * trial.behavior_sampling_interval()
                  for interval, trial in zip(
                      expt.lickingIntervals(imageSync=False),
                      expt.findall('trial'))]
    elif stimulus == 'licking_no_stim':
        starts = []
        stim_time = expt.stimulusTime()
        for trial_running, trial in zip(
                expt.lickingIntervals(imageSync=False),
                expt.findall('trial')):
            trial_interval = trial.behavior_sampling_interval()
            starts.append(np.array(
                [interval[0] * trial_interval
                 for interval in trial_running
                 if interval[1] * trial_interval < stim_time or
                 interval[0] * trial_interval > stim_time +
                 POST_STIM_DELAY]))
    elif stimulus == 'licking_stim':
        starts = []
        stim_time = expt.stimulusTime()
        for trial_running, trial in zip(
                expt.lickingIntervals(imageSync=False),
                expt.findall('trial')):
            trial_interval = trial.behavior_sampling_interval()
            starts.append(np.array(
                [interval[0] * trial_interval
                 for interval in trial_running
                 if interval[0] * trial_interval >= stim_time and
                 interval[0] * trial_interval < stim_time +
                 POST_STIM_DELAY]))
    elif 'licking_stop' in stimulus:
        starts = [interval[:, 1] * trial.behavior_sampling_interval()
                  for interval, trial in zip(
                      expt.lickingIntervals(imageSync=False),
                      expt.findall('trial'))]
    elif 'licking_reward' in stimulus:
        rewarded_intervals, unrewarded_intervals = \
            calculateRewardedLickIntervals(expt, imageSync=False)
        starts = []
        # Loop over trials
        for trial, trial_intervals in it.izip(
                expt.findall('trial'), rewarded_intervals):
            sampling_interval = \
                trial.behavior_sampling_interval()
            if len(trial_intervals):
                intervals = trial_intervals[:, 0] * sampling_interval
            else:
                intervals = trial_intervals
            starts.append(intervals)
    elif 'licking_no_reward' in stimulus:
        rewarded_intervals, unrewarded_intervals = \
            calculateRewardedLickIntervals(expt, imageSync=False)
        starts = []
        # Loop over trials
        for trial, trial_intervals in it.izip(
                expt.findall('trial'), unrewarded_intervals):
            sampling_interval = \
                trial.behavior_sampling_interval()
            if len(trial_intervals):
                intervals = trial_intervals[:, 0] * sampling_interval
            else:
                intervals = trial_intervals
            starts.append(intervals)
    elif 'Paired' in stimulus:
        starts = []
        stims = stimulus.split()[1:]
        for trial in expt.findall('trial'):
            trial_stim_times = []
            if trial.get('stimulus', '') == stimulus:
                try:
                    bd = trial.behaviorData(imageSync=False)
                except exc.MissingBehaviorData:
                    starts.append(np.array([]))
                    continue
                for stim in stims:
                    if stim == 'air':
                        stim = 'airpuff'
                    if bd[stim].shape[1] > 0:
                        if len(trial_stim_times):
                            trial_stim_times = np.intersect1d(
                                trial_stim_times, bd[stim][:, 0])
                        else:
                            trial_stim_times = bd[stim][:, 0]
                starts.append(trial_stim_times)
            else:
                starts.append(np.array([]))
    elif 'position_' in stimulus or stimulus == 'reward':
        # Finds the first frame where the mouse passed the goal position
        # each lap.
        # Running backwards and then forwards again will not trigger
        # multiple positions, so the max number of starts is the number of
        # laps (actually 1 more than the number of completed laps)
        if stimulus == 'reward':
            rewards = expt.rewardPositions(units=None)
            assert len(rewards) == 1  # Only works for a single reward for now
            goal = rewards[0]
        else:
            goal = int(stimulus[9:])
        starts = []
        for trial in expt.findall('trial'):
            trial_starts = []
            position = absolutePosition(
                trial, imageSync=False, sampling_interval='actual')
            trial_goal = goal / trial.behaviorData()['trackLength']
            if position[0] > trial_goal:
                position -= 1
            while True:
                position[position < 0] = np.nan
                pos_bins = np.where(position >= trial_goal)[0]
                if not len(pos_bins):
                    break
                trial_starts.append(pos_bins[0])
                position -= 1
            starts.append(
                np.array(trial_starts) *
                trial.behavior_sampling_interval())
    else:
        starts = []
        for trial in expt.findall('trial'):
            if exclude_paired_from_single \
                    and 'Paired' in trial.get('stimulus', ''):
                starts.append(np.array([]))
                continue
            try:
                bd = trial.behaviorData(imageSync=False)[stimulus]
            except exc.MissingBehaviorData:
                starts.append(np.array([]))
                continue
            if bd.shape[1] > 0:
                starts.append(bd[:, 0])
            else:
                starts.append(np.array([]))

    if(deduplicate):
        assert(duplicate_window is not None)
        dedup = []
        for trialStarts in starts:
            if(trialStarts.size <= 0):
                continue
            dedupTrial = []
            curInd = 0
            while(True):
                dedupTrial.append(trialStarts[curInd])
                nextInds = np.nonzero((trialStarts - trialStarts[curInd]) > duplicate_window)[0]
                if(nextInds.size <= 0):
                    break
                curInd = nextInds[0]
            dedup.append(np.array(dedupTrial))
        starts = dedup

    # Drop NaN values, which correspond to a stim at the start of a trial.
    starts = [
        trial_starts[np.isfinite(trial_starts)] for trial_starts in starts]

    if imageSync:
        syncd_starts = []
        for trial_starts in starts:
            trial_starts /= expt.frame_period()
            # Make sure all the frames are unique
            trial_starts = np.sort(np.unique(trial_starts.astype('int')))
            # Drop frames acquired after imaging stopped
            trial_starts = trial_starts[trial_starts < expt.num_frames()]
            syncd_starts.append(trial_starts)
        starts = syncd_starts

    return starts


def total_absolute_position(
        expt_grp, imageSync=True, sampling_interval=None, by_condition=False):
    """Calculates the position (in laps) as the total laps run per mouse.

    Returns a dictionary with trials as keys and an array of positions with the
    same format as returned by absolutePosition as values.

    """

    if by_condition:
        raise NotImplementedError

    result = {}
    for mouse, mouse_df in expt_grp.dataframe(
            expt_grp, include_columns=['mouse', 'trial']).groupby('mouse'):
        trials = sorted(mouse_df['trial'])
        prev_last_lap = 0
        for trial in trials:
            trial_pos = absolutePosition(
                trial, imageSync=imageSync,
                sampling_interval=sampling_interval)
            last_lap = int(trial_pos.max())
            result[trial] = trial_pos + prev_last_lap
            prev_last_lap += last_lap + 1

    return result


def licks_near_position(
        expt_grp, position, pre=None, post=None, nbins=100):
    """Return normalized licks near a specific position.

    Parameters
    ----------
    expt_grp : lab.ExperimentGroup
    position : {str, float}
        Position to center lick counts on. Argument is passed to expt.locate
        for each experiment.
    pre, post : float, optional
        If not None, filter the resulting dataframe to only include values
        within the interval [-pre, post], centered about 'position'. Should be
        in normalized belt units: [0, 1).
    nbins : int, optional
        Number of bins for resulting lick histogram.

    """
    result = pd.DataFrame([], columns=['expt', 'pos', 'value'])
    for expt in expt_grp:
        licks, bins = expt.licktogram(
            normed=True, nPositionBins=nbins)
        pos = expt.locate(position)

        pos_bin = np.argmin(np.abs(bins - pos))
        rolled_licks = np.roll(licks, nbins / 2 - pos_bin)
        result = pd.concat([result, pd.DataFrame({
            'expt': [expt] * nbins,
            'pos': bins - 0.5,
            'value': rolled_licks})], ignore_index=True)

    if pre is not None:
        result = result[result['pos'] >= -pre]
    if post is not None:
        result = result[result['pos'] <= post]

    return result

# TODO: REMOVE -- Temporary for compatibility
from ..classes import *
from ..misc import *
from imaging_analysis import *
