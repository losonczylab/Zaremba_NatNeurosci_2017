"""Reward analysis functions."""
from __future__ import absolute_import

import numpy as np
import pandas as pd
import itertools as it
from collections import defaultdict
import warnings

from scipy.ndimage.morphology import binary_dilation

from . import behavior_analysis as ba
from . import new_intervals as ints
from ..misc.analysis_helpers import rewards_by_condition
from ..classes.classes import ExperimentGroup
from ..classes import exceptions as exc


def lick_to_reward_distance(expt_grp, rewardPositions=None):
    """Calculate the average lick to reward distance.

    Parameters
    ----------
    rewardPositions : {str, None, np.ndarray}
        If a string, assumed to be a condition label, and will use the
        reward positions used for each mouse during the condition.
        If 'None', uses the actual reward positions during the experiment.
        Otherwise pass in normalized reward positions.

    Returns
    -------
    pd.DataFrame

    """
    result = []

    if rewardPositions is None:
        rewards_by_expt = {
            expt: expt.rewardPositions(units='normalized')
            for expt in expt_grp}
    elif isinstance(rewardPositions, basestring):
        rewards_by_expt = rewards_by_condition(
            expt_grp, rewardPositions, condition_column='condition')
    else:
        rewards_by_expt = defaultdict(lambda: np.array(rewardPositions))

    for expt in expt_grp:

        rewards = rewards_by_expt[expt]

        for trial in expt.findall('trial'):
            bd = trial.behaviorData(imageSync=False)
            position = ba.absolutePosition(
                trial, imageSync=False, sampling_interval='actual')

            if np.any(rewards >= 1.0):
                trial_rewards = rewards / bd['trackLength']
            else:
                trial_rewards = rewards

            licking = bd['licking'][:, 0]
            licking = licking[np.isfinite(licking)]
            licking = licking / bd['samplingInterval']
            licking = licking.astype('int')

            licking_positions = position[licking] % 1

            # meshgrid sets up the subtraction below
            # basically tile expands the arrays
            rewards_mesh, licking_mesh = np.meshgrid(
                trial_rewards, licking_positions)

            reward_distance = licking_mesh - rewards_mesh
            # All distances should be on [-0.5, 0.5)
            reward_distance[reward_distance >= 0.5] -= 1.0
            reward_distance[reward_distance < -0.5] += 1.0

            reward_distance = np.amin(np.abs(reward_distance), axis=1)

            assert len(licking_positions) == len(reward_distance)
            for lick, position in it.izip(
                    reward_distance, licking_positions):
                result.append(
                    {'trial': trial, 'position': position, 'value': lick})
    return pd.DataFrame(result, columns=['trial', 'position', 'value'])


def fraction_rewarded_lick_intervals(expt_grp, **lick_interval_kwargs):
    """Fraction of lick intervals that were rewarded.

    Parameters
    ----------
    **lick_interval_kwargs : dict
        All additional keyword parameters are passed to
        ba.calculateRewardedLickIntervals.

    Returns
    -------
    pd.DataFrame

    """
    result = []
    for expt in expt_grp:
        rewarded_intervals, unrewarded_intervals \
            = ba.calculateRewardedLickIntervals(expt, **lick_interval_kwargs)
        rewarded_count = sum(
            [intervals.shape[0] for intervals in rewarded_intervals])
        unrewarded_count = sum(
            [intervals.shape[0] for intervals in unrewarded_intervals])

        try:
            fraction = rewarded_count \
                / float(rewarded_count + unrewarded_count)
        except ZeroDivisionError:
            fraction = np.nan
        result.append({
            'expt': expt, 'rewarded_count': rewarded_count,
            'unrewarded_counts': unrewarded_count, 'value': fraction})
    return pd.DataFrame(result, columns=[
        'expt', 'rewarded_count', 'unrewarded_count', 'value'])


def fraction_licks_in_rewarded_intervals(expt_grp, **lick_interval_kwargs):
    """Fraction of licks that are within a rewarded lick interval.

    Parameters
    ----------
    **lick_interval_kwargs : dict
        All additional keyword parameters are passed to
        ba.calculateRewardedLickIntervals.

    Returns
    -------
    pd.DataFrame

    """
    result = []
    for expt in expt_grp:
        rewarded_intervals, _ = ba.calculateRewardedLickIntervals(
            expt, **lick_interval_kwargs)

        licks = []
        for trial in expt.findall('trial'):
            sampling_interval = trial.behavior_sampling_interval()
            licks.append(
                trial.behaviorData(imageSync=False)['licking'][:, 0] /
                float(sampling_interval))

        rewarded_licks = np.zeros(len(rewarded_intervals))
        total_licks = 0

        for trial_idx, intervals_trial, licks_trial in it.izip(
                it.count(), rewarded_intervals, licks):
            total_licks += licks_trial.shape[0]
            for interval in intervals_trial:
                rewarded_licks[trial_idx] += np.sum(
                    (interval[0] <= licks_trial) &
                    (interval[1] >= licks_trial))

        total_rewarded_licks = np.sum(rewarded_licks)

        try:
            fraction = total_rewarded_licks / float(total_licks)
        except ZeroDivisionError:
            fraction = np.nan
        result.append({
            'expt': expt, 'rewarded_licks': total_rewarded_licks,
            'total_licks': total_licks, 'value': fraction})

    return pd.DataFrame(result, columns=[
        'expt', 'rewarded_licks', 'total_licks', 'value'])


def fraction_licks_rewarded(expt_grp):
    """Fraction of possible licks rewarded.

    Takes in to account the operant reward rate.

    Returns
    -------
    pd.DataFrame

    """
    result = []
    for expt in expt_grp:

        totalLicks = sum([
            trial.behaviorData()['licking'].shape[0]
            for trial in expt.findall('trial')])

        totalWater = sum([
            trial.behaviorData()['water'].shape[0]
            for trial in expt.findall('trial')])

        rewardRate = expt.reward_parameters().get('operant_rate', 1)

        try:
            fraction = float(totalWater) / (totalLicks / float(rewardRate))
        except ZeroDivisionError:
            fraction = np.nan

        result.append({
            'expt': expt, 'lick': totalLicks, 'water': totalWater,
            'value': fraction})
    return pd.DataFrame(result, columns=['expt', 'lick', 'water', 'value'])


def fractionLicksNearRewardsPerLap(
        expGrp, anticipStartCM=-5, anticipEndCM=-0.1, compareStartCM=-15,
        compareEndCM=-0.1, fractionColName="value", rewardPositions=None,
        exclude_reward=False, exclude_reward_duration=10.0):
    """Fraction of licks in the anticipatory zone vs a compare zone, per lap.

    Parameters
    ----------
    anticipStartCM, anticipEndCM : float
        licks in this spatial window is counted anticipatory.
        Units are in cm. The reward zone start is considered as 0.
        Prereward space is negative, and post reward space is positive.
    copareStartCM, compareEndCM : float
        licks in this window is counted toward total licks.
        Usually, this window should contain the anticipatory window.
    fractionColName : str
        the name to give to the lick fraction column of the returned
        dataframe,  default is "value"
    rewardPositions : {str, None, np.ndarray}
        If a string, assumed to be a condition label, and will use the
        reward positions used for each mouse during the condition.
        If 'None', uses the actual reward positions during the experiment.
        Otherwise pass in normalized reward positions.
    exclude_reward : bool
        exclude licks that occurs after water rewards. This is trying to
        ignore licks that are for drinking water. Default is false.
    exclude_duration : float
        number of seconds after the onset of water reward for which the
        licking should be ignored. Default is 10 seconds.

    Returns
    -------
    pd.DataFrame
        Each row is the licking calculation of a lap, with columns:
            trial - trial instance
            rewardPosition - reward location
            lapNum - the lap number
            anticipLicks - number of licks in the anticipatory zone
            compareLicks - number of licks in the compare zone
            value - anticipLicks/compareLicks.
                the name of this column is set by the kwarg fractionColName,
                when the compare zone contains the anticipatory zone, it is
                similar to fraction of licks near rewards
    """
    result = []

    if(rewardPositions is None):
        rewards_by_exp = {exp: exp.rewardPositions(units='normalized')
                          for exp in expGrp}
    elif(isinstance(rewardPositions, basestring)):
        rewards_by_exp = rewards_by_condition(
            expGrp, rewardPositions, condition_column="condition")
    else:
        rewards_by_exp = defaultdict(lambda: np.array(rewardPositions))

    for exp in expGrp:
        try:
            belt_length = exp.belt().length()
        except exc.NoBeltInfo:
            warnings.warn('Missing belt length. All results may be wrong.')
            belt_length = 220
        anticipStart = anticipStartCM / float(belt_length)
        anticipEnd = anticipEndCM / float(belt_length)
        compareStart = compareStartCM / float(belt_length)
        compareEnd = compareEndCM / float(belt_length)

        rewards = rewards_by_exp[exp]

        for trial in exp.findall("trial"):
            position = ba.absolutePosition(
                trial, imageSync=False, sampling_interval="actual")
            bd = trial.behaviorData(
                imageSync=False, sampling_interval="actual")

            lapNum = position.astype("int32")

            for reward in rewards:
                for i in np.r_[0:np.max(lapNum)]:
                    absRewardPos = i + reward
                    anticipS = absRewardPos + anticipStart
                    anticipE = absRewardPos + anticipEnd
                    compareS = absRewardPos + compareStart
                    compareE = absRewardPos + compareEnd

                    anticipBA = (position >= anticipS) & (position < anticipE)
                    compareBA = (position >= compareS) & (position < compareE)

                    if exclude_reward:
                        numExcPoints = np.int(np.float(
                            exclude_reward_duration)/bd["samplingInterval"])
                        try:
                            firstWater = np.where(compareBA & bd["water"])[0]
                            if(firstWater.size > 0):
                                firstWater = firstWater[0]
                                compareBA[firstWater:firstWater +
                                          numExcPoints] = False
                                anticipBA[firstWater:firstWater +
                                          numExcPoints] = False
                        except KeyError:
                            print("No water signal exist, " +
                                  "exclude_reward not in effect")

                    numAnticipLicks = np.sum(bd["licking"][anticipBA])
                    numCompareLicks = np.sum(bd["licking"][compareBA])

                    fraction = numAnticipLicks / float(numCompareLicks)

                    result.append({"trial": trial,
                                   "rewardPos": reward,
                                   "lapNum": i,
                                   "anticipLicks": numAnticipLicks,
                                   "compareLicks": numCompareLicks,
                                   fractionColName: fraction})

    return pd.DataFrame(result, columns=[
        'trial', 'rewardPos', 'lapNum', 'anticipLicks', 'compareLicks'])


def fractionLicksNearRewards(expt_grp, **kwargs):
    """Aggregate fractionLicksNearRewardsPerLap by Trial."""
    fraction_per_lap = fractionLicksNearRewardsPerLap(expt_grp, **kwargs)
    result = fraction_per_lap.groupby('trial', as_index=False).apply(
        lambda x: pd.Series({
            'trial': x.trial.iloc[0],
            'value': x.anticipLicks.sum() / float(x.compareLicks.sum())}))
    return result


def fraction_licks_near_rewards(
        expt_grp, pre_window_cm=5, post_window_cm=10, rewardPositions=None,
        exclude_reward=False):
    """Fraction of licks near the reward locations.

    Parameters
    ----------
    pre_window_cm, post_window_cm : float
        Window to consider "near" the rewards
    rewardPositions : {str, None, np.ndarray}
        If a string, assumed to be a condition label, and will use the
        reward positions used for each mouse during the condition.
        If 'None', uses the actual reward positions during the experiment.
        Otherwise pass in normalized reward positions.
    exlude_reward : boolean
        Do not consider licks in the reward zone after post_window_cm

    Returns
    -------
    pd.DataFrame

    """
    result = []

    if rewardPositions is None:
        rewards_by_expt = {
            expt: expt.rewardPositions(units='normalized')
            for expt in expt_grp}
    elif isinstance(rewardPositions, basestring):
        rewards_by_expt = rewards_by_condition(
            expt_grp, rewardPositions, condition_column='condition')
    else:
        rewards_by_expt = defaultdict(lambda: np.array(rewardPositions))

    for expt in expt_grp:

        belt_length = expt.belt().length()
        pre = float(pre_window_cm) / belt_length
        post = float(post_window_cm) / belt_length

        rewards = rewards_by_expt[expt]

        # #check to ensure the spatial windows around each reward are non-overlapping
        # rewardPositionCombos = it.combinations(rewards, 2)
        # for (r1, r2) in rewardPositionCombos:
        #     diff = np.abs(r1 - r2)
        #     if diff > .5:
        #         diff = 1 - diff

        #     assert diff > pre + post, "Rewards at %f and %f are too close for pre_window_cm = %f and post_window_cm = %f" % (r1, r2, pre_window_cm, post_window_cm)

        for trial in expt.findall('trial'):
            bd = trial.behaviorData(imageSync=False)
            position = ba.absolutePosition(
                trial, imageSync=False, sampling_interval='actual')

            if np.any(rewards >= 1.0):
                trial_rewards = rewards / bd['trackLength']
            else:
                trial_rewards = rewards

            licking = bd['licking'][:, 0]
            licking = licking[np.isfinite(licking)]
            licking = licking / bd['samplingInterval']
            licking = licking.astype('int')

            licking_positions = position[licking] % 1

            # meshgrid sets up the subtraction below
            # basically tile expands the arrays
            rewards_mesh, licking_mesh = np.meshgrid(
                trial_rewards, licking_positions)

            reward_distance = licking_mesh - rewards_mesh
            # All distances should be on [-0.5, 0.5)
            reward_distance[reward_distance >= 0.5] -= 1.0
            reward_distance[reward_distance < -0.5] += 1.0

            lick_near_reward = np.bitwise_and(
                -pre < reward_distance, reward_distance < post)
            lick_near_reward = np.any(lick_near_reward, axis=1)

            near_licks = np.sum(lick_near_reward)

            if exclude_reward:
                reward_zone_length = expt.reward_parameters(
                    distance_units='normalized')['window_length']

                licks_to_exclude = np.bitwise_and(
                    post < reward_distance,
                    reward_distance < reward_zone_length)
                licks_to_exclude = np.any(licks_to_exclude, axis=1)

                total_licks = len(licking) - np.sum(licks_to_exclude)
            else:
                total_licks = len(licking)

            try:
                fraction = near_licks / float(total_licks)
            except ZeroDivisionError:
                fraction = np.nan

            result.append({
                'trial': trial, 'near_licks': int(near_licks),
                'total_licks': total_licks, 'value': fraction})

    return pd.DataFrame(result, columns=[
        'trial', 'near_licks', 'total_licks', 'value'])


def number_licks_near_rewards(*args, **kwargs):
    df = fraction_licks_near_rewards(*args, **kwargs)

    df['fraction'] = df['value']
    df['value'] = df['near_licks']

    return df


def number_licks_away_rewards(*args, **kwargs):
    df = fraction_licks_near_rewards(*args, **kwargs)

    df['fraction'] = df['value']
    df['value'] = df['total_licks'] - df['near_licks']

    return df


def licking_spatial_information(expt_grp):
    """Calculate spatial information rate (bits/sec) of the licking signal
    calculated across trials

    """

    # TODO: make sure that this is calculating information correctly
    nBins = 100.

    result = []
    for expt in expt_grp:
        nLicks_by_position, _ = expt.licktogram(
            nPositionBins=nBins, rewardPositions=None, normed=False)
        time_by_position = expt.positionOccupancy(
            nPositionBins=nBins, normed=False)
        occupancy = expt.positionOccupancy(nPositionBins=nBins, normed=True)

        lick_rate_by_position = nLicks_by_position.astype(float) \
            / time_by_position
        overall_lick_rate = float(np.sum(nLicks_by_position)) \
            / np.sum(time_by_position)

        integrand = lick_rate_by_position * np.log(
            lick_rate_by_position / float(overall_lick_rate)) * occupancy

        # take nansum because of positions at which there was zero licking (log undefined)
        result.append({'expt': expt, 'value': np.nansum(integrand)})

    return pd.DataFrame(result, columns=['expt', 'value'])


def rate_of_water_obtained(expt_grp):
    """Calculates the rate of water obtained during the experiment.
    Returns result in milliseconds / minute

    """

    result = []

    for expt in expt_grp:
        for trial in expt.findall('trial'):
            bd = trial.behaviorData(imageSync=False)

            ms_water = np.sum([x[1] - x[0] for x in bd['water']]) * 1000.
            trial_duration_min = bd['recordingDuration'] / 60.
            fraction = float(ms_water) / trial_duration_min

            result.append({
                'trial': trial, 'ms_water': ms_water,
                'trial_duration': trial_duration_min, 'value': fraction})

    return pd.DataFrame(result, columns=['trial', 'ms_water', 'trial_duration', 'value'])


# def fraction_licks_in_reward_zone_old(expt_grp):
#     """Calculates the fraction of licks that were within the reward zone"""

#     result = []
#     for expt in expt_grp:
#         for trial in expt.findall('trial'):
#             bd = trial.behaviorData(
#                 imageSync=False, sampling_interval='actual')
#             n_licks = np.sum(
#                 np.diff(np.hstack([0, bd['licking']]).astype('int')) > 0)

#             licks_in_reward = bd['licking'] * bd['reward']
#             n_licks_in_reward = np.sum(
#                 np.diff(np.hstack([0, licks_in_reward]).astype('int')) > 0)
#             try:
#                 fraction = n_licks_in_reward / float(n_licks)
#             except ZeroDivisionError:
#                 fraction = np.nan

#             result.append({
#                 'trial': trial, 'total_licks': n_licks,
#                 'licks_in_reward': n_licks_in_reward, 'value': fraction})

#     return pd.DataFrame(result, columns=[
#         'trial', 'total_licks', 'licks_in_reward', 'value'])


def fraction_licks_in_reward_zone(expt_grp):
    """Calculate the fraction of licks that were within the reward zone."""
    rew_intervals = ints.behavior(expt_grp, 'reward')
    licking_intervals = ints.behavior(expt_grp, 'licking')

    n_licks = licking_intervals.groupby('trial', as_index=False).agg(len)
    n_licks.rename(columns={'start': 'total_licks'}, inplace=True)
    del n_licks['stop']

    licks_in_reward = rew_intervals.filter_events(
        licking_intervals, 'start').groupby('trial', as_index=False).agg(len)
    licks_in_reward.rename(columns={'start': 'licks_in_reward'}, inplace=True)
    del licks_in_reward['stop']

    result = pd.merge(licks_in_reward, n_licks, on='trial', how='outer')
    result['licks_in_reward'] = result['licks_in_reward'].fillna(0)
    result['value'] = result['licks_in_reward'] / \
        result['total_licks'].astype('float')

    return result


def fraction_of_laps_rewarded(expt_grp):
    """Fraction of laps with at least one reward."""
    result = []
    for expt in expt_grp:
        for trial in expt.findall('trial'):
            water = trial.behaviorData(
                imageSync=False, sampling_interval='actual')['water']
            pos = ba.absolutePosition(
                trial, imageSync=False, sampling_interval='actual')
            n_laps = int(pos.max())
            # Need at least 1 full lap
            n_laps -= 1
            if n_laps <= 0:
                continue
            reward_laps = 0
            for lap in range(1, n_laps):
                lap_pos = np.logical_and(pos >= lap, pos < lap + 1)
                if water[lap_pos].sum() > 0:
                    reward_laps += 1
            try:
                fraction = reward_laps / float(n_laps)
            except ZeroDivisionError:
                fraction = np.nan
            result.append({'trial': trial, 'n_laps': n_laps,
                           'rewarded_laps': reward_laps,
                           'value': fraction})

    return pd.DataFrame(result)


def licks_outside_reward_vicinity(expt_grp):
    result = []
    for expt in expt_grp:
        for trial in expt.findall('trial'):
            bd = trial.behaviorData(
                imageSync=False, sampling_interval='actual')
            n_licks = np.sum(
                np.diff(np.hstack([0, bd['licking']]).astype('int')) > 0)
            # "Reward vicinity" is 5s either side of reward zone (based on PSTH)
            rewards = bd['reward']
            five_secs = int(round((5. / expt.duration().seconds) * len(rewards)))
            reward_vicinity = binary_dilation(rewards, iterations=five_secs)
            licks_outside_reward_vicinity = bd['licking'] * ~reward_vicinity
            lick_array = np.diff(np.hstack(
                [0, licks_outside_reward_vicinity]).astype('int')) > 0
            n_licks_outside_reward = np.sum(lick_array)
            try:
                fraction = n_licks_outside_reward / float(n_licks)
            except ZeroDivisionError:
                fraction = np.nan

            result.append({
                'trial': trial, 'total_licks': n_licks,
                'licks_outside_reward': n_licks_outside_reward,
                'value': fraction})

    return pd.DataFrame(result, columns=[
        'trial', 'total_licks', 'licks_outside_reward', 'value'])


def anticipatory_licking(expt_grp):
    result = []
    for expt in expt_grp:
        for trial in expt.findall('trial'):
            bd = trial.behaviorData(
                imageSync=False, sampling_interval='actual')
            n_licks = np.sum(
                np.diff(np.hstack([0, bd['licking']]).astype('int')) > 0)
            reward_zones = bd['reward']
            five_secs = int(
                round((5. / expt.duration().seconds) * len(reward_zones)))
            expanded_reward = binary_dilation(
                reward_zones, iterations=five_secs, structure=[1, 1, 0])
            anticipation_zones = expanded_reward - reward_zones
            anticipatory_licks = bd['licking'] * anticipation_zones
            lick_array = np.diff(
                np.hstack([0, anticipatory_licks]).astype('int')) > 0
            n_anticipatory_licks = sum(lick_array)
            try:
                fraction = n_anticipatory_licks / float(n_licks)
            except ZeroDivisionError:
                fraction = np.nan
            result.append({
                'trial': trial, 'total_licks': n_licks,
                'anticipatory_licks': n_anticipatory_licks,
                'value': fraction})

    return pd.DataFrame(result, columns=[
        'trial', 'total_licks', 'anticipatory_licks', 'value'])


def fraction_of_laps_with_licking_near_reward(
        expt_grp, pre_window_cm=5, post_window_cm=10,
        rewardPositions=None):
    """Fraction of laps that have licking near the reward positions.

    Arguments:
    pre_window_cm, post_window_cm -- Window to consider "near" the rewards
    rewardPositions : {str, None, np.ndarray}
        If a string, assumed to be a condition label, and will use the
        reward positions used for each mouse during the condition.
        If 'None', uses the actual reward positions during the experiment.
        Otherwise pass in normalized reward positions.

    """

    result = []

    if rewardPositions is None:
        rewards_by_expt = {
            expt: expt.rewardPositions(units='normalized')
            for expt in expt_grp}
    elif isinstance(rewardPositions, basestring):
        rewards_by_expt = rewards_by_condition(
            expt_grp, rewardPositions, condition_column='condition')
    else:
        rewards_by_expt = defaultdict(lambda: np.array(rewardPositions))

    licking_pos = ExperimentGroup.stim_position(
        expt_grp, 'licking', abs_pos=True)

    for expt in expt_grp:

        belt_length = expt.belt().length()
        pre = float(pre_window_cm) / belt_length
        post = float(post_window_cm) / belt_length

        rewards = rewards_by_expt[expt]

        for trial in expt.findall('trial'):
            n_laps = int(ba.absolutePosition(
                trial, imageSync=False, sampling_interval='actual').max())

            # Don't include the first incomplete lap
            n_laps -= 1
            if n_laps <= 0:
                continue

            trial_pos = licking_pos[
                licking_pos['trial'] == trial]['value'].values

            if np.any(rewards >= 1.0):
                track_length = trial.behaviorData(
                    imageSync=False)['trackLength']
                trial_rewards = rewards / track_length
            else:
                trial_rewards = rewards

            reward_laps = 0
            for lap in range(1, n_laps):
                for reward in trial_rewards:
                    lap_rew_start = lap + reward - pre
                    lap_rew_stop = lap + reward + post
                    in_rew = np.logical_and(
                        trial_pos >= lap_rew_start,
                        trial_pos < lap_rew_stop)
                    if in_rew.sum():
                        reward_laps += 1
                        break

            result.append({
                'trial': trial, 'n_laps': n_laps,
                'rewarded_laps': reward_laps,
                'value': reward_laps / float(n_laps)})

    return pd.DataFrame(result, columns=[
        'trial', 'n_laps', 'rewarded_laps', 'value'])
