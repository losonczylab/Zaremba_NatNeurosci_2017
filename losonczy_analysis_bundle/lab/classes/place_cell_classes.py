"""Place cell classes"""

import numpy as np
import cPickle as pickle
from copy import copy, deepcopy
import pandas as pd

from classes import Mouse, ExperimentGroup
import exceptions as exc


class pcExperimentGroup(ExperimentGroup):
    """Place cell experiment group"""

    def __init__(self, experiment_list, nPositionBins=100,
                 channel='Ch2', imaging_label=None, demixed=False, **kwargs):

        super(pcExperimentGroup, self).__init__(experiment_list, **kwargs)

        # Store all args as a dictionary
        self.args = {}
        self.args['nPositionBins'] = nPositionBins
        self.args['channel'] = channel
        self.args['imaging_label'] = imaging_label
        self.args['demixed'] = demixed

        self._data, self._data_raw, self._pfs, self._std, self._circ_var, \
            self._circ_var_p, self._info, self._info_p = \
            {}, {}, {}, {}, {}, {}, {}, {}

    def __repr__(self):
        return super(pcExperimentGroup, self).__repr__() + \
               '(args: {})'.format(repr(self.args))

    def __copy__(self):
        return type(self)(
            copy(self._list), label=self._label, **copy(self.args))

    def __deepcopy__(self):
        return type(self)(
            deepcopy(self._list), label=self._label, **deepcopy(self.args))

    def __delitem__(self, i):
        expt = self[i]
        super(pcExperimentGroup, self).__delitem__(i)
        self._data.pop(expt, 0)
        self._data_raw.pop(expt, 0)
        self._pfs.pop(expt, 0)
        self._std.pop(expt, 0)

    def data(self, roi_filter=None, dataframe=False):
        # tuning curves by experiment
        indices = {}
        for expt in self:
            indices[expt] = expt._filter_indices(
                roi_filter, channel=self.args['channel'],
                label=self.args['imaging_label'])

            if expt not in self._data:
                try:
                    # check for existence of place_fields.pkl
                    with open(expt.placeFieldsFilePath(
                            channel=self.args['channel']), 'rb') as f:
                        place_fields = pickle.load(f)
                except IOError:
                    self._data[expt] = None
                    self._data_raw[expt] = None
                else:
                    demixed_key = 'demixed' if self.args['demixed'] \
                        else 'undemixed'
                    imaging_label = self.args['imaging_label'] \
                        if self.args['imaging_label'] is not None \
                        else expt.most_recent_key(channel=self.args['channel'])
                    try:
                        self._data[expt] = place_fields[
                            imaging_label][demixed_key][
                            'spatial_tuning_smooth']
                        self._data_raw[expt] = place_fields[
                            imaging_label][demixed_key][
                            'spatial_tuning']
                    except KeyError:
                        self._data[expt] = None
                        self._data_raw[expt] = None

        return_data = [] if dataframe else {}
        if dataframe:
            rois = self.rois(
                roi_filter=roi_filter, channel=self.args['channel'],
                label=self.args['imaging_label'])
        for expt in self:
            try:
                expt_data = self._data[expt][indices[expt], :]
            except (TypeError, IndexError):
                expt_data = None
            if dataframe:
                assert len(rois[expt]) == len(expt_data)
                for roi, dat in zip(rois[expt], expt_data):
                    return_data.append(
                        {'expt': expt, 'roi': roi, 'value': dat})
            else:
                return_data[expt] = expt_data
        if dataframe:
            return pd.DataFrame(return_data)
        return return_data

    def data_raw(self, roi_filter=None, dataframe=False):
        self.data()
        return_data = [] if dataframe else {}
        if dataframe:
            rois = self.rois(
                roi_filter=roi_filter, channel=self.args['channel'],
                label=self.args['imaging_label'])
        for expt in self:
            indices = expt._filter_indices(
                roi_filter, channel=self.args['channel'],
                label=self.args['imaging_label'])
            try:
                expt_data = self._data_raw[expt][indices, :]
            except (TypeError, IndexError):
                expt_data = None
            if dataframe:
                assert len(rois[expt]) == len(expt_data)
                for roi, dat in zip(rois[expt], expt_data):
                    return_data.append(
                        {'expt': expt, 'roi': roi, 'value': dat})
            else:
                return_data[expt] = expt_data
        if dataframe:
            return pd.DataFrame(return_data)
        return return_data

    def circular_variance(self, roi_filter=None):
        indices = {}
        for expt in self:
            indices[expt] = expt._filter_indices(
                roi_filter, channel=self.args['channel'],
                label=self.args['imaging_label'])
            if expt not in self._circ_var:
                with open(expt.placeFieldsFilePath(), 'rb') as f:
                    result = pickle.load(f)
                demixed_key = 'demixed' if self.args['demixed'] \
                    else 'undemixed'
                imaging_label = self.args['imaging_label'] \
                    if self.args['imaging_label'] is not None \
                    else expt.most_recent_key(channel=self.args['channel'])
                self._circ_var[expt] = result[
                    imaging_label][demixed_key]['true_circ_variances']
        return {expt: self._circ_var[expt][indices[expt]] for expt in self}

    def circular_variance_p(self, roi_filter=None):
        indices = {}
        for expt in self:
            indices[expt] = expt._filter_indices(
                roi_filter, channel=self.args['channel'],
                label=self.args['imaging_label'])
            if expt not in self._circ_var_p:
                with open(expt.placeFieldsFilePath(), 'rb') as f:
                    result = pickle.load(f)
                demixed_key = 'demixed' if self.args['demixed'] \
                    else 'undemixed'
                imaging_label = self.args['imaging_label'] \
                    if self.args['imaging_label'] is not None \
                    else expt.most_recent_key(channel=self.args['channel'])
                self._circ_var_p[expt] = result[
                    imaging_label][demixed_key]['circ_variance_p_vals']
        return {expt: self._circ_var_p[expt][indices[expt]] for expt in self}

    def spatial_information_p(self, roi_filter=None):
        indices = {}
        for expt in self:
            indices[expt] = expt._filter_indices(
                roi_filter, channel=self.args['channel'],
                label=self.args['imaging_label'])
            if expt not in self._info_p:
                with open(expt.placeFieldsFilePath(), 'rb') as f:
                    result = pickle.load(f)
                demixed_key = 'demixed' if self.args['demixed'] \
                    else 'undemixed'
                imaging_label = self.args['imaging_label'] \
                    if self.args['imaging_label'] is not None \
                    else expt.most_recent_key(channel=self.args['channel'])
                self._info_p[expt] = result[
                    imaging_label][demixed_key]['information_p_values']
        return {expt: self._info_p[expt][indices[expt]] for expt in self}

    def spatial_information(self, roi_filter=None):
        indices = {}
        for expt in self:
            indices[expt] = expt._filter_indices(
                roi_filter, channel=self.args['channel'],
                label=self.args['imaging_label'])
            if expt not in self._info:
                with open(expt.placeFieldsFilePath(), 'rb') as f:
                    result = pickle.load(f)
                demixed_key = 'demixed' if self.args['demixed'] \
                    else 'undemixed'
                imaging_label = self.args['imaging_label'] \
                    if self.args['imaging_label'] is not None \
                    else expt.most_recent_key(channel=self.args['channel'])
                self._info[expt] = result[
                    imaging_label][demixed_key]['spatial_information']
        return {expt: self._info[expt][indices[expt]] for expt in self}

    def pfs(self, roi_filter=None):
        indices = {}
        return_data = {}
        for expt in self:
            indices[expt] = expt._filter_indices(
                roi_filter, channel=self.args['channel'],
                label=self.args['imaging_label'])
            if expt not in self._pfs:
                if self.data()[expt] is None:
                    self._pfs[expt] = None
                else:
                    with open(expt.placeFieldsFilePath(), 'rb') as f:
                        result = pickle.load(f)
                    demixed_key = 'demixed' if self.args['demixed'] \
                        else 'undemixed'
                    imaging_label = self.args['imaging_label'] \
                        if self.args['imaging_label'] is not None \
                        else expt.most_recent_key(channel=self.args['channel'])
                    self._pfs[expt] = result[
                        imaging_label][demixed_key]['pfs']
            try:
                return_data[expt] = [
                    self._pfs[expt][idx] for idx in indices[expt]]
            except (TypeError, IndexError):
                return_data[expt] = None
        return return_data

    def pfs_n(self, roi_filter=None):
        pfs = self.pfs(roi_filter=roi_filter)
        pfs_n = {}
        nBins = self.args['nPositionBins']
        for expt in self:
            if pfs[expt] is None:
                pfs_n[expt] = None
            else:
                pfs_n[expt] = []
                for roi in pfs[expt]:
                    roiPfs = []
                    for pf in roi:
                        roiPfs.append(
                            [pf[0] / float(nBins), pf[1] / float(nBins)])
                    pfs_n[expt].append(roiPfs)
        return pfs_n

    def std(self, roi_filter=None):
        indices = {}
        for expt in self:
            indices[expt] = expt._filter_indices(
                roi_filter, channel=self.args['channel'],
                label=self.args['imaging_label'])
            if expt not in self._std:
                if self.data()[expt] is None:
                    self._std[expt] = None
                else:
                    with open(expt.placeFieldsFilePath(), 'rb') as f:
                        result = pickle.load(f)
                    demixed_key = 'demixed' if self.args['demixed'] \
                        else 'undemixed'
                    imaging_label = self.args['imaging_label'] \
                        if self.args['imaging_label'] is not None \
                        else expt.most_recent_key(channel=self.args['channel'])
                    self._std[expt] = result[
                        imaging_label][demixed_key]['std_smooth']

        return {expt: self._std[expt][indices[expt], :] for expt in self}

    def pcs_filter(self, roi_filter=None, circ_var=False):
        pcs = []

        if not circ_var:
            pfs = self.pfs(roi_filter=roi_filter)
        else:
            circular_variance_p = self.circular_variance_p(
                roi_filter=roi_filter)
        for expt in self:
            if circ_var:
                pc_inds = np.where(circular_variance_p[expt] < 0.05)[0]
            else:
                pc_inds = np.where(pfs[expt])[0]
            rois = expt.rois(channel=self.args['channel'],
                             label=self.args['imaging_label'],
                             roi_filter=roi_filter)
            pcs.extend([rois[x] for x in pc_inds])
        pcs = set(pcs)

        def pc_filter(roi):
            return roi in pcs
        return pc_filter

    def running_kwargs(self):
        if not hasattr(self, '_running_kwargs'):
            running_kwargs = []
            for expt in self:
                with open(expt.placeFieldsFilePath(), 'rb') as f:
                    p = pickle.load(f)
                imaging_label = self.args['imaging_label'] \
                    if self.args['imaging_label'] is not None \
                    else expt.most_recent_key(channel=self.args['channel'])
                demixed_key = 'demixed' if self.args['demixed'] \
                    else 'undemixed'
                running_kwargs.append(
                    p[imaging_label][demixed_key]['running_kwargs'])
            if np.all(
                [running_kwargs[0].items() == x.items()
                 for x in running_kwargs[1:]]):
                self._running_kwargs = running_kwargs[0]
            else:
                raise(
                    'Place fields calculated with different running kwargs')
        return self._running_kwargs

    def removeDatalessExperiments(self, **kwargs):
        super(pcExperimentGroup, self).removeDatalessExperiments(**kwargs)
        for expt in reversed(self):
            if self.data()[expt] is None:
                self.remove(expt)

    def subGroup(self, expts, label=None):
        """Returns a new pcExperimentGroups containing the experiments in
        expts, with the same args and data/pcs as the original group.

        """

        if not label:
            new_grp = type(self)(expts, label=self.label(), **self.args)
        else:
            new_grp = type(self)(expts, label=label, **self.args)
        for expt in expts:
            if expt not in self:
                raise Exception('Invalid experiment, experiment not found')
            if expt in self._data:
                new_grp._data[expt] = self._data[expt]
            if expt in self._data_raw:
                new_grp._data_raw[expt] = self._data_raw[expt]
            if expt in self._pfs:
                new_grp._pfs[expt] = self._pfs[expt]
            if expt in self._std:
                new_grp._std[expt] = self._std[expt]
            if expt in self._circ_var:
                new_grp._circ_var[expt] = self._circ_var[expt]
            if expt in self._circ_var_p:
                new_grp._circ_var_p[expt] = self._circ_var_p[expt]
            if expt in self._info_p:
                new_grp._info_p[expt] = self._info_p[expt]
            if expt in self._std:
                new_grp._std[expt] = self._std[expt]

        return new_grp

    def extend(self, exptGrp):
        """Extend a pcExperimentGroup with another ExperimentGroup"""

        super(pcExperimentGroup, self).extend(exptGrp)

        for expt in exptGrp:
            try:
                if expt in exptGrp._data:
                    self._data[expt] = exptGrp._data[expt]
                if expt in exptGrp._data_raw:
                    self._data_raw[expt] = exptGrp._data_raw[expt]
                if expt in exptGrp._pfs:
                    self._pfs[expt] = exptGrp._pfs[expt]
                if expt in exptGrp._std:
                    self._std[expt] = exptGrp._std[expt]
            except AttributeError:
                return

    def set_args(self, **kwargs):
        # change arguments and clear data/pfs
        for key, value in kwargs.iteritems():
            self.args[key] = value
        self._data, self._data_raw, self._pfs, self._std = {}, {}, {}, {}

    @classmethod
    def fromMice(cls, mice, tag='', channel='Ch2', demixed=False,
                 imaging_label=None, **kwargs):
        """Takes a list of mice and pulls out all the experiment that have
        place fields for this channel/label/demixed

        """

        if isinstance(mice, Mouse):
            mice = [mice]

        group = []
        for mouse in mice:
            for expt in mouse.findall('experiment'):
                if tag.lower() in expt.get('tSeriesDirectory', '').lower():

                    try:
                        expt.find('trial').behaviorData()['treadmillPosition']
                    except (KeyError, exc.MissingBehaviorData):
                        continue

                    try:
                        with open(expt.placeFieldsFilePath(
                                channel=channel), 'rb') as f:
                            place_fields = pickle.load(f)
                    except (exc.NoSimaPath, exc.NoTSeriesDirectory, IOError,
                            pickle.UnpicklingError):
                        continue
                    else:
                        if imaging_label not in place_fields:
                            continue
                        else:
                            demixed_key = 'demixed' if demixed else 'undemixed'
                            if demixed_key in place_fields[imaging_label]:
                                group.append(expt)

        return cls(group, channel=channel, demixed=demixed,
                   imaging_label=imaging_label, **kwargs)

    def transform(self, *args, **kwargs):
        return transformed_pcExperimentGroup(self, *args, **kwargs)


class transformed_pcExperimentGroup(object):
    """A wrapped pcExperimentGroup where the belt positions are re-ordered/
    transformed.

    """

    def __init__(
            self, expt_grp, transforms):
        self.expt_grp = copy(expt_grp)
        self.transforms = transforms

    def __copy__(self):
        return type(self)(self.expt_grp, self.transforms)

    def __deepcopy__(self):
        return type(self)(
            deepcopy(self.expt_grp), deepcopy(self.transforms))

    def __getattr__(self, name):
        # return getattr(self.expt_grp, name)
        return self.expt_grp.__getattribute__(name)

    def __setitem__(self, i, v):
        self.expt_grp[i] = v

    def __delitem__(self, i):
        del self.expt_grp[i]

    def __len__(self):
        return len(self.expt_grp)

    def __getitem__(self, i):
        return self.expt_grp[i]

    def __iter__(self):
        return self.expt_grp.__iter__()

    def __reversed__(self):
        return self.expt_grp.__reversed__()

    def __str__(self):
        return "<transformed_pcExperimentGroup " + \
            self.expt_grp.__str__() + ">"

    def __repr__(self):
        return self.expt_grp.__repr__()

    def subGroup(self, expts):
        new_grp = self.expt_grp.subGroup(expts)
        new_transforms = {expt: tran for expt, tran in
                          self.transforms.iteritems() if expt in new_grp}
        return new_grp.transform(transforms=new_transforms)

    def untransform(self):
        return self.expt_grp

    def data(self, roi_filter=None):
        return_data = self.expt_grp.data(roi_filter=roi_filter)
        for expt in return_data:
            if return_data[expt] is not None:
                if expt in self.transforms:
                    return_data[expt] = \
                        return_data[expt][:, self.transforms[expt]]
        return return_data

    def data_raw(self, roi_filter=None):
        return_data = self.expt_grp.data_raw(roi_filter=roi_filter)
        for expt in return_data:
            if return_data[expt] is not None:
                if expt in self.transforms:
                    return_data[expt] = \
                        return_data[expt][:, self.transforms[expt]]
        return return_data

    def pfs(self, roi_filter=None):
        raise TypeError('PFs not calculated for transformed belts')

    def pfs_n(self, roi_filter=None):
        raise TypeError('PFs not calculated for transformed belts')

    def std(self, roi_filter=None):
        raise NotImplemented
