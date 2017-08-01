"""Functions that generate specific ROI filters"""

import os
import warnings
from os.path import normpath
import numpy as np
import random
import itertools as it
import cPickle as pickle

import lab

from calc_activity import calc_activity
import imaging_analysis as ia


def interneurons_filter(expt, channel='Ch2', label=None, demixed=False):
    def _cluster_points(X, mu):
        clusters = {}
        for x in X:
            bestmukey = min([(i[0], np.linalg.norm(x - mu[i[0]]))
                             for i in enumerate(mu)], key=lambda t: t[1])[0]
            try:
                clusters[bestmukey].append(x)
            except KeyError:
                clusters[bestmukey] = [x]
        return clusters

    def _reevaluate_centers(mu, clusters):
        newmu = []
        keys = sorted(clusters.keys())
        for k in keys:
            newmu.append(np.mean(clusters[k], axis=0))
        return newmu

    def _has_converged(mu, oldmu):
        return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])

    def _find_centers(X, K):
        oldmu = random.sample(X, K)
        mu = random.sample(X, K)
        while not _has_converged(mu, oldmu):
            oldmu = mu
            # Assign all points in X to clusters
            clusters = _cluster_points(X, mu)
            # Reevaluate centers
            mu = _reevaluate_centers(oldmu, clusters)
        return(mu, clusters)

    if label is None:
        label = expt.most_recent_key(channel=channel)
    raw = expt.imagingData(
        channel=channel, label=label, demixed=demixed)[:, :, 0]
    # dff = expt.imagingData(dFOverF='from_file',
    #     channel=channel, label=label, demixed=demixed)[:, :, 0]
    # std = np.nanstd(raw, axis=1)
    # area = np.nansum(dff, axis=1) / np.array(
    #     [sum(np.isfinite(x)) for x in dff])
    t = expt.transientsData()[:, 0]
    frame_period = expt.frame_period()
    peak_fractions = []
    for starts, peaks, durs in it.izip(
            t['start_indices'], t['max_indices'], t['durations_sec']):
        roi_vals = []
        for s, p, d in it.izip(starts, peaks, durs):
            if d < 5:
                continue
            else:
                roi_vals.append((p - s) * frame_period / d)
        if len(roi_vals) >= 5:
            peak_fractions.append(np.mean(roi_vals))
        else:
            peak_fractions.append(np.nan)
    peak_fractions = np.array(peak_fractions)

    v = expt.velocity()[0]
    velocity_corr = np.array(
        [np.corrcoef(np.nan_to_num(x), v)[0, 1] for x in it.izip(raw)])
    durations = calc_activity(expt, method='duration').mean(1)
    active_fraction = np.sum(
        ia.isActive(expt)[:, :, 0], axis=1) / float(expt.imaging_shape()[1])

    # IN_idxs = set(np.where(velocity_corr > 0.3)[0]).intersection(set(
    #     np.where(active_fraction > 0.1)[0])).intersection(
    #     set(np.where(peak_fractions > 0.4)[0]))
    IN_idxs = set(np.where(peak_fractions > 0.3)[0]).intersection(set(
        np.where(active_fraction > 0.10)[0])).intersection(
        set(np.where(np.abs(velocity_corr) > 0.3)[0])).intersection(
        set(np.where(durations > 2.5)[0]))

    # IN_idxs = set(np.where(velocity_corr > 0.3)[0]).intersection(set(
    #     np.where(active_fraction > 0.2)[0]).union(
    #     set(np.where(durations > 2.5)[0])))
    rois = expt.rois(channel=channel, label=label)
    return [rois[x] for x in IN_idxs]

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(std, velocity_corr)


def validROIs(expt, fraction_isnans_threshold=0.,
              contiguous_isnans_threshold=50, **imaging_kwargs):
    """Return a filter that selects valid ROIs as determined by NaN
    thresholds

    Parameters
    ----------
    fraction_isnans_threshold : float
        If any trial contains greater than X% NaN's, throw out the ROI
    contiguous_isnans_threshold : int
        If any trial contains more than X consecutive NaN's, throw out the
        ROI

    """
    raw_imData = expt.imagingData(**imaging_kwargs)

    first_threshold_junk_cells = np.where(np.amax(np.sum(np.isnan(
        raw_imData), axis=1), axis=1) / float(raw_imData.shape[1]) >
        fraction_isnans_threshold)[0]

    nanFrames = np.isnan(raw_imData)
    longest_nan_stretch = np.zeros(nanFrames.shape[0])
    for rowInd, row in enumerate(nanFrames):
        cycle_result = []
        for cycleInd in range(row.shape[1]):
            nanFrames_starts_stops = np.where(np.diff(np.hstack(
                (0, row[:, cycleInd].astype(int), 0))))[0]
            nanInterval_lengths = np.diff(
                nanFrames_starts_stops)[np.arange(0, len(
                    nanFrames_starts_stops), 2)]
            try:
                cycle_result.append(np.amax(nanInterval_lengths))
            except:
                pass
        try:
            longest_nan_stretch[rowInd] = np.amax(cycle_result)
        except:
            pass

    second_threshold_junk_cells = np.where(longest_nan_stretch >
                                           contiguous_isnans_threshold)[0]

    # take the union of the two thresholding results
    exclude_cells = set(first_threshold_junk_cells) | \
        set(second_threshold_junk_cells)

    # return the indices of the good cells as a list
    # return sorted(
    #     set(np.arange(raw_imData.shape[0])).difference(exclude_cells))
    return lambda x: True if x in expt.rois() and expt.rois().index(x) \
        not in exclude_cells else False


def n_tran_max_filter(expt, channel=None, label=None, roi_filter=None,
                      n_tran=0, dof='from_file', interval=None):
    """Return a filter for ROIs that have a minimum number of transients
    whose peaks fall within the intervals of interest.

    Parameters
    ----------
    channel : string
        The name of the channel containing the transient data to be used for
        filtering.
    label : string or None
        The label associated with the ROI set to be used.
    roi_filter : function or None
        A boolean-valued function that is used to filter the ROI set.
    n_tran : int
        The minimum number of transients an ROI must have in order to be
        included in the filter
    interval : tuple or string
        The imaging intervals of interest.
    """

    expt_activity = calc_activity(expt, roi_filter=roi_filter,
                                  method='n transients', interval=interval,
                                  dF=dof, channel=channel, label=label)
    expt_roi_ids = expt.roi_ids(roi_filter=roi_filter, channel=channel,
                                label=label)
    rois = []
    for idx, roi_id in enumerate(expt_roi_ids):
        if expt_activity[idx] >= n_tran:
            rois.append(roi_id)

    def shared_rois(roi):
        return roi.id in rois

    return shared_rois


def n_tran_grp_filter(expt_grp, channel='Ch2', label=None, roi_filter=None,
                      n_tran=0, n_sessions=0, interval=None,
                      method='n transients'):
    """Return a shared filter for rois that are in all expts and have a minimum
    number of transients in a minimum number of sessions.
    n_tran -- the integer value of the minimum number of transients that must
              be present in a session
    n_sessions -- the integer value of the minimum number of sessions where the
                 number of transients of an roi is greater than the n_tran
    interval -- count transients only in the specific interval of an expt
    method -- 'n transients' will return the number of transients that peak
              in the interval of interest.
              NOTE: 'n transients start' currently not implemented in
              calc_activity
              'n transients start' will return the number of transients that
              start in the interval of interest.
    """

    shared_filter = shared_rois_filter(expt_grp, channel=channel, label=label,
                                       roi_filter=roi_filter)
    shared_rois_ids = expt_grp.sharedROIs(channel=channel, label=label,
                                          roi_filter=shared_filter)
    roi_ids = []
    temp_list = []
    for expt in expt_grp:
        n_tran_rois = n_tran_max_filter(expt, channel=channel, label=label,
                                        roi_filter=shared_filter,
                                        n_tran=n_tran, interval=interval)
        temp_list.extend(expt.roi_ids(roi_filter=n_tran_rois, channel=channel,
                                      label=label))
    for roi_id in shared_rois_ids:
        count = temp_list.count(roi_id)
        if count >= n_sessions:
            roi_ids.append(roi_id)

    def shared_rois(roi):
        return roi.id in roi_ids

    return shared_rois


def responsive_roi_filter(
        expt, stimulus, trials, method='responsiveness', pre_frames=None,
        post_frames=None, channel='Ch2', label=None, roi_filter=None,
        data=None, conf_level=95, sig_tail='upper', exclude=None,
        transients_conf_level=99, shuffle_exclude='exclude',
        n_bootstraps=10000, dFOverF='from_file',
        return_roi_tuple_filter=False):
    """Return the responsive roi_filter for the given experiment.
    Raises a ValueError if responsive rois have not been identified yet
    for this expt.

    Keyword arguments:
    stimulus -- str, return rois responsive to this stimulus
    trials -- list, all the trials used for analysis. This should match
        the same trials that were originally used to calculate
        responsiveness.
    method -- 'responsiveness' or 'peak'
    pre_frame, post_frames -- number of pre and post frames for psth
    channel, label, roi_filter, dFOverF -- used for imaging data
    data -- lets you pass in data other than the imaging data to calculate
        the psth
    conf_level -- confidence threshold for rois to survive shuffling
    sig_tail -- 'upper', 'lower', 'both', which end of the shuffle
        distribution to use for cutoff
    exclude -- frames to exclude, can also be 'running' to exclude running
    transients_conf_level -- used if data is transients data
    shuffle_exclude -- determines the frames to exclude when shuffling
        data. 'exclude' matches the exclude argument.
    n_bootstraps -- number of bootstraps for shuffling
    return_roi_tuple_filter : bool
        The default return will filter on ROI objects present in the current
        ExperimentGroup. Alternatively, return a filter on ROI
        (mouse, location, id) tuples that will also filter ROIs not in the
        current ExperimentGroup.

    """

    if stimulus == 'air':
        stimulus = 'airpuff'

    if shuffle_exclude == 'exclude':
        shuffle_exclude = exclude

    responsive_rois_path = normpath(os.path.join(
        expt.sima_path(), 'responsive_rois.pkl'))
    try:
        with open(responsive_rois_path, 'r') as f:
            responsive_rois = pickle.load(f)
    except (IOError, pickle.UnpicklingError):
        raise ValueError(
            'No responsive_rois.pkl file, run ' +
            'identify_stim_responsive_cells')

    if label is None:
        label = expt.most_recent_key(channel=channel)

    # trial times will get concatenated in to 1 long string, since
    # lists are not hashable
    trials_string = ''.join(
        sorted(trial.get('time') for trial in trials))
    trials_grp = lab.ExperimentGroup(set(trial.parent for trial in trials))
    update_time = trials_grp.updateTime(
        channel=channel, label=label, ignore_signals=False,
        ignore_transients=(data != 'trans'),
        ignore_dfof=(dFOverF != 'from_file'), ignore_place_fields=True)
    key_tuple = (trials_string, method, pre_frames, post_frames, channel,
                 label, data, conf_level, sig_tail, transients_conf_level,
                 exclude, shuffle_exclude, n_bootstraps, dFOverF)
    if stimulus in responsive_rois \
            and key_tuple in responsive_rois[stimulus]:
        roi_ids = responsive_rois[stimulus][key_tuple]['roi_ids']
        timestamp = responsive_rois[stimulus][key_tuple]['timestamp']
        if update_time > timestamp:
            raise ValueError(
                'Data has been updated since rois identified, re-run ' +
                'identify_stim_responsive_cells')
        rois = expt.rois(
            channel=channel, label=label, roi_filter=roi_filter)
        responsive_rois = [roi for roi in rois if roi.id in roi_ids]

        if return_roi_tuple_filter:
            responsive_roi_tuples = set(
                [(roi.expt.parent.get('mouseID'),
                 roi.expt.get('uniqueLocationKey'),
                 roi.id) for roi in responsive_rois])

            def responsive_roi_tuple_filter(roi):
                return (roi.expt.parent.get('mouseID'),
                        roi.expt.get('uniqueLocationKey'),
                        roi.id) in responsive_roi_tuples

            return responsive_roi_tuple_filter

        else:
            def responsive_filter(roi):
                return roi in responsive_rois

            return responsive_filter

    else:
        raise ValueError(
            'No responsive rois for given parameters, re-run ' +
            'identify_stim_responsive_cells')


def identifyStimReponsiveCells(expt, channel='Ch2', label=None,
                               roi_filter=None, **kwargs):
    warnings.warn('Use identify_stim_reponsive_cells to get an ' +
                  'roi_filter instead of a boolean mask',
                  DeprecationWarning)
    responsive_filter = identify_stim_responsive_cells(
        expt, channel=channel, label=label, roi_filter=roi_filter, **kwargs)
    return [responsive_filter(roi) for roi in expt.rois(
        channel=channel, label=label, roi_filter=roi_filter)]


def identify_stim_responsive_cells(
        expt, stimulus, pre_frames=None, post_frames=None, channel='Ch2',
        label=None, returnStrNames=False, **kwargs):
    """Return a filter of stim responsive ROIS.
    See ExperimentGroup.identify_stim_responsive_cells for details.

    """

    if pre_frames is None:
        pre_time = None
    else:
        pre_time = pre_frames * expt.frame_period()
    if post_frames is None:
        post_time = None
    else:
        post_time = post_frames * expt.frame_period()

    final_filter = ia.identify_stim_responsive_cells(
        lab.ExperimentGroup([expt]), stimulus=stimulus, pre_time=pre_time,
        post_time=post_time, **kwargs)

    if returnStrNames:
        return expt.roi_ids(
            channel=channel, label=label, roi_filter=final_filter)
    return final_filter


"""
ExperimentGroup filter functions
"""


def shared_rois_filter(
        exptGrp, roi_filter=None, channel='Ch2', label=None, demixed=False):
    """Return a shared filter for rois that are in all expts."""

    # Check to make sure all expts are from the same field
    fields = [
        (expt.parent, expt.get('uniqueLocationKey')) for expt in exptGrp]
    if not all([field == fields[0] for field in fields[1:]]):
        warnings.warn(
            'Multiple fields of views in exptGrp, no shared ROIs')
        return lambda x: False

    rois = set(
        [(roi.expt.parent.get("mouseID"),
          roi.expt.get("uniqueLocationKey"),
          roi.id) for roi in exptGrp[0].rois(channel=channel,
                                             label=label,
                                             roi_filter=roi_filter)])
    for expt in exptGrp[1:]:
        rois = rois.intersection(
            set([(roi.expt.parent.get("mouseID"),
                  roi.expt.get("uniqueLocationKey"),
                  roi.id) for roi in expt.rois(channel=channel,
                                               label=label,
                                               roi_filter=roi_filter)]))

    def shared_rois(roi):
        return (roi.expt.parent.get("mouseID"),
                roi.expt.get("uniqueLocationKey"),
                roi.id) in rois

    return shared_rois


def active_roi_filter(
        exptGrp, min_transients=1, channel='Ch2', label=None,
        roi_filter=None):
    """Identify active cells, returns an ROI filter"""
    active_rois_list = []
    for expt in exptGrp:
        transients = expt.transientsData(
            channel=channel, label=label, roi_filter=roi_filter)
        n_transients = np.sum(
            np.vectorize(len)(transients['start_indices']), axis=1)
        rois = expt.rois(
            channel=channel, label=label, roi_filter=roi_filter)
        active_rois_list.extend(
            it.compress(rois, n_transients >= min_transients))
    active_rois_set = set(active_rois_list)

    def active_rois(roi):
        return roi in active_rois_set

    return active_rois


def reward_cell_filter(expt_grp, threshold=0.1, roi_filter=None):
    """Return an ROI filter that includes all ROIs near rewards.

    Parameters
    ----------
    expt_grp : lab.classes.pcExperimentGroup
    threshold : float
        Any place cell with a centroid within 'threshold' of the reward will
        be included. Normalized units.
    roi_filter : filter function
        Initial ROI filter.

    Returns
    -------
    reward_cell_roi_filter : filter function

    """
    distances = lab.analysis.place_cell_analysis.centroid_to_position_distance(
        expt_grp, positions='reward', roi_filter=roi_filter, return_abs=True,
        multiple_fields='largest')

    rois = set(distances.ix[distances['value'] < threshold, 'roi'])

    def reward_cell_roi_filter(roi):
        return roi in rois

    return reward_cell_roi_filter
