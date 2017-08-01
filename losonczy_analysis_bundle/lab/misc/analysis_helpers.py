"""Helper functions for analysis"""

import numpy as np
from copy import copy


def rewards_by_condition(expt_grp, condition, condition_column='condition'):
    """Determines the reward positions used during a particular condition.

    Parameters
    ----------
    condition : str
        Return the reward positions used during this condition.
    conditio_column : str
        Column in the experiment dataframe to match the 'condition' to.

    Returns
    -------
    dict
        Dictionary with expts as keys as an array of non-normalized positions
        as values.

    """

    df = expt_grp.dataframe(
        expt_grp, include_columns=[
            condition_column, 'rewardPositions', 'mouseID'])
    mouse_condition_positions = dict()

    sliced_df = df[df[condition_column] == condition]
    for mouseID in set(sliced_df['mouseID'].values):
        sliced_df1 = sliced_df[sliced_df['mouseID'] == mouseID]
        sliced_df1['rewardPositions'] = sliced_df1[
            'rewardPositions'].apply(lambda x: float(x[0]))
        mouse_cond_reward_pos = list(set(
            sliced_df1['rewardPositions'].values))
        mouse_condition_positions[mouseID] = np.array(mouse_cond_reward_pos)

    return {expt: copy(
        mouse_condition_positions.get(expt.parent.get('mouseID'), None))
        for expt in expt_grp}
