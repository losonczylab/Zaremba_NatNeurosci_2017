import pandas as pd
import numpy as np
import os

import lab.plotting as plotting

session_order = [
    '{}_{}_{}'.format(context, day, session) for context in ('A', 'B', 'C')
    for day in range(3) for session in range(3)]

# session_pairs = (
#     ('C_0_0', 'C_0_1'), ('C_0_1', 'C_0_2'), ('C_1_0', 'C_1_1'),
#     ('C_1_1', 'C_1_2'), ('C_2_0', 'C_2_1'), ('C_2_1', 'C_2_2'),
#     ('C_0_2', 'C_1_0'), ('C_1_2', 'C_2_0')
# )
session_pairs = zip(session_order, session_order[1:])

# session_triples = (
#     ('C_0_0', 'C_0_1', 'C_0_2'), ('C_1_0', 'C_1_1', 'C_1_2'), ('C_2_0', 'C_2_1', 'C_2_2'),
#     ('C_0_1', 'C_0_2', 'C_1_0'), ('C_1_1', 'C_1_2', 'C_2_0'),
#     ('C_0_2', 'C_1_0', 'C_1_1'), ('C_1_2', 'C_2_0', 'C_2_1'),
# )
session_triples = zip(session_order, session_order[1:], session_order[2:])

# session_quads = (
#     ('C_0_0', 'C_0_1', 'C_0_2', 'C_1_0'), ('C_1_0', 'C_1_1', 'C_1_2', 'C_2_0'),
#     ('C_0_1', 'C_0_2', 'C_1_0', 'C_1_1'), ('C_1_1', 'C_1_2', 'C_2_0', 'C_2_1'),
#     ('C_0_2', 'C_1_0', 'C_1_1', 'C_1_2'), ('C_1_2', 'C_2_0', 'C_2_1', 'C_2_2'),
# )
session_quads = zip(session_order, session_order[1:], session_order[2:],
                    session_order[3:])

# session_quints = (
#     ('C_0_0', 'C_0_1', 'C_0_2', 'C_1_0', 'C_1_1'), ('C_1_0', 'C_1_1', 'C_1_2', 'C_2_0', 'C_2_1'),
#     ('C_0_1', 'C_0_2', 'C_1_0', 'C_1_1', 'C_1_2'), ('C_1_1', 'C_1_2', 'C_2_0', 'C_2_1', 'C_2_2'),
#     ('C_0_2', 'C_1_0', 'C_1_1', 'C_1_2', 'C_2_0'),
# )
session_quints = zip(session_order, session_order[1:], session_order[2:],
                     session_order[3:], session_order[4:])

# session_hexs = (
#     ('C_0_0', 'C_0_1', 'C_0_2', 'C_1_0', 'C_1_1', 'C_1_2'), ('C_1_0', 'C_1_1', 'C_1_2', 'C_2_0', 'C_2_1', 'C_2_2'),
#     ('C_0_1', 'C_0_2', 'C_1_0', 'C_1_1', 'C_1_2', 'C_2_0'),
#     ('C_0_2', 'C_1_0', 'C_1_1', 'C_1_2', 'C_2_0', 'C_2_1'),
# )
session_hexs = zip(session_order, session_order[1:], session_order[2:],
                   session_order[3:], session_order[4:], session_order[5:])

not_pc_threshold = 0.2
pc_threshold = 0.05


def prep_data(df, session_filter='C', pivotby='condition_day_session'):

    plotting.prepare_dataframe(
        df,
        include_columns=['mouseID', 'uniqueLocationKey', 'roi_id'] + [pivotby])

    df['imaged'] = 1.

    df['roi_tuple'] = zip(df['mouseID'], df['uniqueLocationKey'], df['roi_id'])

    del df['mouseID']
    del df['uniqueLocationKey']
    del df['roi_id']
    del df['expt']
    del df['roi']

    if session_filter is not None:
        df = df[df['condition_day_session'].str.contains(session_filter)]

    # After the pivot, df[(df['expt_grp'] == grp_label).any(1)],
    # will filter out to a single expt_grp

    return df.pivot(index='roi_tuple', columns=pivotby)


def load_data(
        geno, session_filter='C',
        root='/data/enrichment_model/'):
    if 'wt' in geno.lower():
        raw_data = pd.read_pickle(os.path.join(root, 'WT_place_cell_data.pkl'))
    elif 'df' in geno.lower():
        raw_data = pd.read_pickle(os.path.join(root, 'Df_place_cell_data.pkl'))
    prepped_data = prep_data(
        raw_data, session_filter=session_filter,
        pivotby='condition_day_session')
    return raw_data, prepped_data


def new_recur_pc_distances(data, method='cv'):
    #
    # The positions firing positions of new and recurring PCs
    #
    not_pc_to_pc_distances = []
    pc_to_pc_distances = []
    for first, second in session_pairs:
        if any(session not in data.columns.levels[1]
               for session in (first, second)):
            continue
        if method == 'cv':
            new_pcs = data[(data['circ_var_p', first] > not_pc_threshold) &
                           (data['circ_var_p', second] < pc_threshold)]
            recur_pcs = data[(data['circ_var_p', first] < pc_threshold) &
                             (data['circ_var_p', second] < pc_threshold)]
            not_pc_to_pc_distances.extend(
                new_pcs['activity_centroid_reward_distance', second].values)
            pc_to_pc_distances.extend(
                recur_pcs['activity_centroid_reward_distance', second].values)
        elif method == 'si':
            new_pcs = data[(data['n_place_fields', first] < 1) &
                           (data['n_place_fields', second] >= 1)]
            recur_pcs = data[(data['n_place_fields', first] >= 1) &
                             (data['n_place_fields', second] >= 1)]
            not_pc_to_pc_distances.extend(
                new_pcs['centroid_reward_distance', second].values)
            pc_to_pc_distances.extend(
                recur_pcs['centroid_reward_distance', second].values)

    return np.array(not_pc_to_pc_distances), np.array(pc_to_pc_distances)


def pc_fraction(raw_data, method='cv'):

    n_rois = raw_data.shape[0]

    if method == 'cv':
        n_pcs = (raw_data['circ_var_p'] < pc_threshold).sum()

    pc_fraction = n_pcs / float(n_rois)

    print('pc_fraction: {}'.format(pc_fraction))

    return pc_fraction


def p_on_off(data, method='cv'):
    #
    # P_on
    #
    old_non_pcs_count = 0
    not_pc_to_pc_count = 0
    not_pc_to_not_pc_count = 0
    for first, second in session_pairs:
        if any(session not in data.columns.levels[1]
               for session in (first, second)):
            continue
        if method == 'cv':
            old_non_pcs = data[(data['circ_var_p', first] > pc_threshold) &
                               data['imaged', second]]
            new_pcs = old_non_pcs[
                old_non_pcs['circ_var_p', second] < pc_threshold]
            new_non_pcs = old_non_pcs[
                old_non_pcs['circ_var_p', second] > not_pc_threshold]
        elif method == 'si':
            raise NotImplementedError
        old_non_pcs_count += len(old_non_pcs)
        not_pc_to_pc_count += len(new_pcs)
        not_pc_to_not_pc_count += len(new_non_pcs)

    p_on = not_pc_to_pc_count / float(old_non_pcs_count)
    p_stay_off = not_pc_to_not_pc_count / float(old_non_pcs_count)
    print "P_on = {}".format(p_on)
    print "P_stay_off = {}".format(p_stay_off)

    #
    # P_off
    #
    old_pcs_count = 0
    pc_to_pc_count = 0
    pc_to_not_pc_count = 0
    for first, second in session_pairs:
        if any(session not in data.columns.levels[1]
               for session in (first, second)):
            continue
        old_pcs = data[(data['circ_var_p', first] < pc_threshold) &
                       data['imaged', second]]
        new_pcs = old_pcs[old_pcs['circ_var_p', second] < pc_threshold]
        new_non_pcs = old_pcs[old_pcs['circ_var_p', second] > not_pc_threshold]
        old_pcs_count += len(old_pcs)
        pc_to_pc_count += len(new_pcs)
        pc_to_not_pc_count += len(new_non_pcs)

    p_off = pc_to_not_pc_count / float(old_pcs_count)
    p_stay_on = pc_to_pc_count / float(old_pcs_count)
    print "P_off = {}".format(p_off)
    print "P_stay_on = {}".format(p_stay_on)

    return p_on, p_stay_off, p_off, p_stay_on


def recurrence_by_position(data, method='cv'):
    #
    # (first_position, second_is_pc) tuples
    #
    distance_recurrence = []
    for first, second in session_pairs:
        if any(session not in data.columns.levels[1]
               for session in (first, second)):
            continue
        if method == 'cv':
            old_pcs = data[(data['circ_var_p', first] < pc_threshold) &
                           data['imaged', second]]
            distance_recurrence.extend(zip(
                old_pcs['activity_centroid_reward_distance', first],
                old_pcs['circ_var_p', second] < pc_threshold))
        elif method == 'si':
            old_pcs = data[(data['n_place_fields', first] >= 1) &
                           data['imaged', second]]
            distance_recurrence.extend(zip(
                old_pcs['centroid_reward_distance', first],
                old_pcs['n_place_fields', second] >= 1))
    return np.array(distance_recurrence)


def firing_stability(data):
    #
    # Firing stability
    #
    position_pairs = []
    for first, second in session_pairs:
        if any(session not in data.columns.levels[1]
               for session in (first, second)):
            continue
        both_pcs = data[(data['circ_var_p', first] < pc_threshold) &
                        (data['circ_var_p', second] < pc_threshold)]
        position_pairs.extend(zip(
            both_pcs['activity_centroid_reward_distance', first],
            both_pcs['activity_centroid_reward_distance', second]))
    return np.array(position_pairs)


def paired_centroid_distance_to_reward(data):
    #
    # Paired centroid distance to reward, nan if not a PC
    #
    paired_centroid_list = []
    for first, second in session_pairs:
        if any(session not in data.columns.levels[1]
               for session in (first, second)):
            continue
        session_data = pd.DataFrame({
            'first': data['centroid_reward_distance', first],
            'second': data['centroid_reward_distance', second]})
        session_data = session_data[(data['imaged', first] == 1) &
                                    (data['imaged', second] == 1)]
        paired_centroid_list.append(session_data)
    return pd.concat(paired_centroid_list)


def paired_activity_centroid_distance_to_reward(
        data, skip_across_conditions=False):
    #
    # Paired activity centroid distance to reward, nan if not a PC
    #
    paired_activity_centroid_list = []
    for first, second in session_pairs:
        if any(session not in data.columns.levels[1]
               for session in (first, second)):
            continue
        if skip_across_conditions and first[0] != second[0]:
            continue
        session_data = pd.DataFrame({
            'first': data['activity_centroid_reward_distance', first],
            'second': data['activity_centroid_reward_distance', second]})
        session_data['first'][data['circ_var_p', first] >= pc_threshold] = \
            np.nan
        session_data['second'][data['circ_var_p', second] >= pc_threshold] = \
            np.nan
        session_data = session_data[
            (data['imaged', first] == 1) & (data['imaged', second] == 1)]
        paired_activity_centroid_list.append(session_data)
    return pd.concat(paired_activity_centroid_list)


def tripled_activity_centroid_distance_to_reward(data, prev_imaged=True):

    activity_centroid_list = []
    for first, second, third in session_triples:
        if any(session not in data.columns.levels[1]
               for session in (first, second, third)):
            continue
        session_data = pd.DataFrame({
            'first': data['activity_centroid_reward_distance', first],
            'second': data['activity_centroid_reward_distance', second],
            'third': data['activity_centroid_reward_distance', third]})
        session_data['first'][data['circ_var_p', first] >= pc_threshold] = \
            np.nan
        session_data['second'][data['circ_var_p', second] >= pc_threshold] = \
            np.nan
        session_data['third'][data['circ_var_p', third] >= pc_threshold] = \
            np.nan
        if prev_imaged:
            session_data = session_data[
                (data['imaged', first] == 1) & (data['imaged', second] == 1) &
                (data['imaged', third] == 1)]
        else:
            session_data = session_data[
                (data['imaged', first] == 1) & (data['imaged', third] == 1)]
        activity_centroid_list.append(session_data)
    return pd.concat(activity_centroid_list)


def quad_activity_centroid_distance_to_reward(data, prev_imaged=True):

    activity_centroid_list = []
    for first, second, third, fourth in session_quads:
        if any(session not in data.columns.levels[1]
               for session in (first, second, third, fourth)):
            continue
        session_data = pd.DataFrame({
            'first': data['activity_centroid_reward_distance', first],
            'second': data['activity_centroid_reward_distance', second],
            'third': data['activity_centroid_reward_distance', third],
            'fourth': data['activity_centroid_reward_distance', fourth]})
        session_data['first'][data['circ_var_p', first] >= pc_threshold] = \
            np.nan
        session_data['second'][data['circ_var_p', second] >= pc_threshold] = \
            np.nan
        session_data['third'][data['circ_var_p', third] >= pc_threshold] = \
            np.nan
        session_data['fourth'][data['circ_var_p', fourth] >= pc_threshold] = \
            np.nan
        if prev_imaged:
            session_data = session_data[
                (data['imaged', first] == 1) & (data['imaged', third] == 1) &
                (data['imaged', fourth] == 1)]
        else:
            session_data = session_data[
                (data['imaged', first] == 1) & (data['imaged', fourth] == 1)]
        activity_centroid_list.append(session_data)
    return pd.concat(activity_centroid_list)


def quint_activity_centroid_distance_to_reward(data, prev_imaged=True):

    activity_centroid_list = []
    for first, second, third, fourth, fifth in session_quints:
        if any(session not in data.columns.levels[1]
               for session in (first, second, third, fourth, fifth)):
            continue
        session_data = pd.DataFrame({
            'first': data['activity_centroid_reward_distance', first],
            'second': data['activity_centroid_reward_distance', second],
            'third': data['activity_centroid_reward_distance', third],
            'fourth': data['activity_centroid_reward_distance', fourth],
            'fifth': data['activity_centroid_reward_distance', fifth]})
        session_data['first'][data['circ_var_p', first] >= pc_threshold] = \
            np.nan
        session_data['second'][data['circ_var_p', second] >= pc_threshold] = \
            np.nan
        session_data['third'][data['circ_var_p', third] >= pc_threshold] = \
            np.nan
        session_data['fourth'][data['circ_var_p', fourth] >= pc_threshold] = \
            np.nan
        session_data['fifth'][data['circ_var_p', fifth] >= pc_threshold] = \
            np.nan
        if prev_imaged:
            session_data = session_data[
                (data['imaged', first] == 1) & (data['imaged', fourth] == 1) &
                (data['imaged', fifth] == 1)]
        else:
            session_data = session_data[
                (data['imaged', first] == 1) & (data['imaged', fifth] == 1)]
        activity_centroid_list.append(session_data)
    return pd.concat(activity_centroid_list)


def hex_activity_centroid_distance_to_reward(data, prev_imaged=True):

    activity_centroid_list = []
    for first, second, third, fourth, fifth, sixth in session_hexs:
        if any(session not in data.columns.levels[1]
               for session in (first, second, third, fourth, fifth, sixth)):
            continue
        session_data = pd.DataFrame({
            'first': data['activity_centroid_reward_distance', first],
            'second': data['activity_centroid_reward_distance', second],
            'third': data['activity_centroid_reward_distance', third],
            'fourth': data['activity_centroid_reward_distance', fourth],
            'fifth': data['activity_centroid_reward_distance', fifth],
            'sixth': data['activity_centroid_reward_distance', sixth]})
        session_data['first'][data['circ_var_p', first] >= pc_threshold] = \
            np.nan
        session_data['second'][data['circ_var_p', second] >= pc_threshold] = \
            np.nan
        session_data['third'][data['circ_var_p', third] >= pc_threshold] = \
            np.nan
        session_data['fourth'][data['circ_var_p', fourth] >= pc_threshold] = \
            np.nan
        session_data['fifth'][data['circ_var_p', fifth] >= pc_threshold] = \
            np.nan
        session_data['sixth'][data['circ_var_p', sixth] >= pc_threshold] = \
            np.nan
        if prev_imaged:
            session_data = session_data[
                (data['imaged', first] == 1) & (data['imaged', fifth] == 1) &
                (data['imaged', sixth] == 1)]
        else:
            session_data = session_data[
                (data['imaged', first] == 1) & (data['imaged', sixth] == 1)]
        activity_centroid_list.append(session_data)
    return pd.concat(activity_centroid_list)


def tripled_activity_centroid(data, prev_imaged=True):

    activity_centroid_list = []
    for first, second, third in session_triples:
        if any(session not in data.columns.levels[1]
               for session in (first, second, third)):
            continue
        session_data = pd.DataFrame({
            'first': data['activity_centroid', first],
            'second': data['activity_centroid', second],
            'third': data['activity_centroid', third]})
        session_data['first'][data['circ_var_p', first] >= pc_threshold] = \
            np.nan
        session_data['second'][data['circ_var_p', second] >= pc_threshold] = \
            np.nan
        session_data['third'][data['circ_var_p', third] >= pc_threshold] = \
            np.nan
        if prev_imaged:
            session_data = session_data[
                (data['imaged', first] == 1) & (data['imaged', second] == 1) &
                (data['imaged', third] == 1)]
        else:
            session_data = session_data[
                (data['imaged', first] == 1) & (data['imaged', third] == 1)]
        activity_centroid_list.append(session_data)
    return pd.concat(activity_centroid_list)
