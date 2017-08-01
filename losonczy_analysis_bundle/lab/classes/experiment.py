"""Experiment subclasses"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from xml.etree import ElementTree
import cPickle as pkl
import os
import glob
import time
import itertools as it
from copy import copy, deepcopy
import pandas as pd

import sima
from sima.ROI import ROI, ROIList

from classes import ExperimentGroup, Belt
import exceptions as exc
from ..analysis import behavior_analysis as ba
from ..analysis import imaging_analysis as ia
from ..plotting import plotting
from ..plotting import analysis_plotting as ap
from .. import misc
from ..misc import dff


class Experiment(ElementTree.Element):
    """An object representing a single recording session.

    Experiment instances comprise an ExperimentSet and are instantiated upon
    initialization of an ExperimentSet.  They are not directly initialized.

    Example
    -------

    >>> from lab import ExperimentSet
    >>> expt_set = ExperimentSet(
        '/analysis/experimentSummaries/.clean_code/experiments/behavior.xml')
    >>> experiment = expt_set.grabExpt('mouseID', 'startTime')

    Note
    ----
    The Experiment class inherits from the ElementTree.Element class, and as
    such, it retains the hierarchical organization of the .xml file.
    Experiment.parent returns the mouse object associated with that experiment,
    and Experiment.findall('trial') returns the list of trials associated with
    the Experiment.

    """

    def __init__(self):
        """Initialization occurs upon initialization of an ExperimentSet."""
        self._rois = {}

    #
    # Defines simple relationship operations for convenience
    #
    def __lt__(self, other):
        """Sortable by startTime."""
        return self.get('startTime') < other.get('startTime')

    def __eq__(self, other):
        """Check the equivalence of two experiments.

        Experiments are equivalent if they have the same associated mouse
        and startTime
        """
        try:
            self_mouse = self.parent.get('mouseID', np.nan)
            other_mouse = other.parent.get('mouseID', np.nan)
            self_time = self.get('startTime', np.nan)
            other_time = other.get('startTime', np.nan)
        except AttributeError:
            return False
        return (type(self) == type(other)) and (self_mouse == other_mouse) \
            and (self_time == other_time)

    def __sub__(self, other):
        """Take the difference of two experiment startTimes."""
        try:
            return self.startTime() - other.startTime()
        except AttributeError:
            raise NotImplemented

    def totuple(self):
        """Return a unique tuple representation of the experiment."""
        return (self.parent.get('mouseID'), self.get('startTime'))

    def getparent(self):
        """Simple getter function to return mouse. Same as self.parent,
        used for compatibility with lxml etree

        """
        return self.parent

    @property
    def trial_id(self):
        try:
            return int(self.get('trial_id'))
        except TypeError:
            pass

    def field_tuple(self):
        return (self.parent.get('mouseID'), self.get('uniqueLocationKey'))

    def _filter_indices(self, roi_filter, channel='Ch2', label=None):
        if roi_filter is not None:
            indices = [i for i, r in enumerate(
                self.rois(channel=channel, label=label)) if roi_filter(r)]
        else:
            indices = np.arange(len(self.rois(channel=channel, label=label)))
        return indices

    def has_drug_condition(self, **kwargs):
        """Checks a list of drug fields for matching values. If the fields
        exist in any drugs present during the experiment return True,
        otherwise False. Breaks as soon as it can't find a matching drug field.

        """

        drugs = self.findall('drug')
        for key, value in kwargs.iteritems():
            field_found = False
            for drug in drugs:
                if drug.get(key, '') == value:
                    field_found = True
                    break
            if not field_found:
                return False
        return True

    def frame_shape(self):
        # (z, y, x, c)
        return self.imaging_dataset().frame_shape

    def num_frames(self, trim_to_behavior=True, channel='Ch2', label=None):
        return self.imaging_shape(
            trim_to_behavior=trim_to_behavior, channel=channel, label=label)[1]

    def num_rois(self, label=None, roi_filter=None):
        return len(self.rois(label=label, roi_filter=roi_filter))

    def imaging_shape(
            self, channel='Ch2', label=None, roi_filter=None,
            trim_to_behavior=True):
        """Lazy stores and returns the shape of imaging data.
        (n_rois, n_frames, n_cycles)

        """

        # channel is unnecessary here, remove eventually

        if not hasattr(self, '_imaging_shape'):
            self._imaging_shape = {}
        if (channel, label, roi_filter, trim_to_behavior) not in \
                self._imaging_shape:
            self._imaging_shape[
                (channel, label, roi_filter, trim_to_behavior)] = \
                self.imagingData(
                    channel=channel, label=label, roi_filter=roi_filter,
                    trim_to_behavior=trim_to_behavior).shape
        return self._imaging_shape[
            (channel, label, roi_filter, trim_to_behavior)]

    def frame_period(self, round_result=True):
        """Returns the time between zyxc volumes.

        This is complicated by Prairie's inconsistent handling of planes/cycles
        in the xml.

        """

        if not hasattr(self, '_frame_period'):
            if self.frame_shape()[0] == 1:
                # If the imaging data is only 1 plane, the frame period in the
                # xml should be correct.
                self._frame_period = self.imagingParameters()['framePeriod']
                # Alternatively, this should be correct for an experiment
                # consisting of a single element.
                # config = ElementTree.parse(
                #     self.prairie_xml_path().replace('.xml', '.env'))
                # self._frame_period = float(
                #     config.find('TSeries')[0].get('repititionPeriod'))
            else:
                # First make sure there is only 1 element in the tSeries, if
                # there's more than 1 this might still work, but check...
                n_elements = -1
                for _, elem in ElementTree.iterparse(
                        self.prairie_xml_path().replace('.xml', '.env')):
                    if elem.tag == 'TSeries':
                        n_elements = len(elem)
                        break
                if n_elements != 1:
                    raise ValueError(
                        'Invalid experiment type, unable to determine frame ' +
                        'period')
                times = []
                # If this time seems variable, adding more might even this out,
                # but it seems reliable enough to just take the first 2 frames
                n_seqs = 2
                for _, elem in ElementTree.iterparse(self.prairie_xml_path()):
                    if elem.tag == 'Sequence':
                        bidirectional = elem.get('bidirectionalZ') == 'True'
                        times.append(
                            float(elem.find('Frame').get('absoluteTime')))
                        if len(times) >= n_seqs:
                            break
                if bidirectional:
                    self._frame_period = np.diff(times).mean() * 2.0
                else:
                    self._frame_period = np.diff(times).mean()

        if round_result:
            return np.around(self._frame_period, 6)
        return self._frame_period

    # def saveCorrected(self, filenames=None, fmt='TIFF', fill_gaps=True):
    #     '''Saves a motion-corrected tiff stack of the given experiment'''
    #     self.imaging_dataset().export_frames(filenames, fmt, fill_gaps)

    def sameField(self, expt):
        """Determine if two experiments are from the same field
        Returns true if non-imaged experiments are from the same mouse

        """
        return (self.parent == expt.parent) and \
               (self.get('uniqueLocationKey') == expt.get('uniqueLocationKey'))

    def sameConditions(self, expt, ignoreBelt=False, ignoreContext=False,
                       ignoreRewardPositions=False):
        """Determine if two experiments were under the same conditions,
        context and belt

        """
        if not ignoreRewardPositions:
            reward_check = all(a == b for a, b in it.izip_longest(
                self.rewardPositions(units=None),
                expt.rewardPositions(units=None)))

        return (ignoreBelt or self.get('belt') == expt.get('belt')) and \
               (ignoreContext or self.get('environment') ==
                expt.get('environment')) and \
               (ignoreRewardPositions or reward_check)

    def sameContext(self, expt):
        """Determines whether two experiments were in the same context
        (same belt and same environment)

        """

        return self.sameConditions(
            expt, ignoreContext=False, ignoreBelt=False,
            ignoreRewardPositions=True)

    def updateTime(self, **kwargs):
        return ExperimentGroup([self]).updateTime(**kwargs)

    def startTime(self):
        return misc.parseTime(self.get('startTime'))

    def imaging_dataset(self, dataset_path=None, reload_dataset=False):
        if dataset_path is None:
            dataset_path = self.sima_path()
        if reload_dataset or not hasattr(self, '_dataset'):
            self._dataset = sima.ImagingDataset.load(dataset_path)
        return self._dataset

    def most_recent_key(self, channel='Ch2'):
        if not hasattr(self, '_most_recent_key'):
            self._most_recent_key = {}
        if channel not in self._most_recent_key:
            try:
                self._most_recent_key[channel] = sima.misc.most_recent_key(
                    self.imaging_dataset().signals(channel=channel))
            except ValueError:
                raise exc.NoSignalsData('No signals for channel {}'.format(
                    channel))
        return self._most_recent_key[channel]

    def rois(self, channel='Ch2', label=None, roi_filter=None):
        if label is None:
            label = self.most_recent_key(channel=channel)
        if label not in self._rois:
            # If the desired label is not extracted, return the original ROIs
            # TODO: Should we catch errors here?
            #
            # TODO: Do we really want to return the original rois?
            # All the analysis assumes that the ROIs are from the extracted
            # data, I don't think this is a good idea.
            #
            try:
                signals = self.imaging_dataset().signals(
                    channel=channel)[label]
            except KeyError:
                warnings.warn(
                    "LABEL MISSING FROM SIGNALS FILE, LOADING rois.pkl ROIS")
                with open(self.roisFilePath(), 'rb') as f:
                    roi_list = pkl.load(f)[label]['rois']
            else:
                roi_list = signals['rois']
            self._rois[label] = [ROI(**roi) for roi in roi_list]
            # Fill in empty IDs with what should be a unique id
            id_str = '_{}_{}_'.format(self.parent.get('mouseID'),
                                      self.get('startTime'))
            id_idx = 0
            for roi in self._rois[label]:
                if roi.id is None:
                    roi.id = id_str + str(id_idx)
                    id_idx += 1
                roi.expt = self
        if roi_filter is None:
            return ROIList(self._rois[label])

        return ROIList([roi for roi in self._rois[label] if roi_filter(roi)])

    def roi_ids(self, label=None, channel='Ch2', roi_filter=None):
        return [roi.id for roi in self.rois(
                channel=channel, label=label, roi_filter=roi_filter)]

    def roi_tuples(self, channel='Ch2', label=None, roi_filter=None):
        mouse_id = self.parent.get('mouseID')
        loc = self.get('uniqueLocationKey')
        return [(mouse_id, loc, roi_id) for roi_id in self.roi_ids(
            channel=channel, label=label, roi_filter=roi_filter)]

    def duration(self):
        try:
            return misc.parseTime(self.get('stopTime')) - \
                misc.parseTime(self.get('startTime'))
        except:
            return None

    def prairie_xml_path(self):
        root_dir = os.path.join(self.parent.parent.dataPath,
                                self.get('tSeriesDirectory').lstrip('/'))
        xml_files = glob.glob(os.path.join(root_dir, '*.xml'))
        if len(xml_files) == 1:
            return xml_files[0]
        elif len(xml_files) > 1:
            raise ValueError(
                "Unable to determine Prairie xml path: too many xml files")
        else:
            raise ValueError(
                "Unable to determine Prairie xml path: no xml files found")

    def imagingParameters(self, param=None):

        if not hasattr(self, '_imagingParameters'):
            self._imagingParameters = \
                sima.imaging_parameters.prairie_imaging_parameters(
                    self.prairie_xml_path())

        if param is None:
            return self._imagingParameters
        else:
            warnings.warn('param argument is deprecated, index the dictionary',
                          DeprecationWarning)
            return self._imagingParameters[param]

    def set_sima_path(self, path):
        ''' Setter function for the absolute path to the .sima folder
        To be used in cases where the dataset of interest is stored locally,
        separately from its tSeries
        '''
        self._sima_path = path
        self._dataset = sima.ImagingDataset.load(path)

    def sima_path(self):
        """Returns the path to the .sima folder
        If > 1 .sima directory exists, raise an ambiguity error
        """
        if hasattr(self, '_sima_path'):
            return self._sima_path

        if not self.get('tSeriesDirectory'):
            raise exc.NoTSeriesDirectory()
        tSeriesDirectory = os.path.normpath(
            os.path.join(self.parent.parent.dataPath,
                         self.get('tSeriesDirectory').lstrip('/')))
        sima_dirs = glob.glob(os.path.join(tSeriesDirectory, '*.sima'))

        if len(sima_dirs) == 1:
            return sima_dirs[0]
        elif len(sima_dirs) > 1:
            raise exc.NoSimaPath(
                'Multiple .sima directories contained in t-series directory')
        else:
            raise exc.NoSimaPath('Unable to locate .sima directory')

    def signalsFilePath(self, channel='Ch2'):
        signals_path = os.path.normpath(os.path.join(
            self.sima_path(), 'signals_{}.pkl'.format(
                self.imaging_dataset()._resolve_channel(channel))))
        return signals_path

    def transientsFilePath(self, channel='Ch2'):
        transients_path = os.path.normpath(os.path.join(
            self.sima_path(), 'transients_{}.pkl'.format(
                self.imaging_dataset()._resolve_channel(channel))))
        return transients_path

    def dfofFilePath(self, channel='Ch2'):
        dfof_path = os.path.normpath(os.path.join(
            self.sima_path(), 'dFoF_{}.pkl'.format(
                self.imaging_dataset()._resolve_channel(channel))))
        return dfof_path

    def placeFieldsFilePath(self, channel='Ch2'):
        place_path = os.path.normpath(os.path.join(
            self.sima_path(), 'place_fields_{}.pkl'.format(
                self.imaging_dataset()._resolve_channel(channel))))
        return place_path

    def spikesFilePath(self, channel='Ch2'):
        spikes_path = os.path.normpath(os.path.join(
            self.sima_path(), 'spikes_{}.pkl'.format(
                self.imaging_dataset()._resolve_channel(channel))))
        return spikes_path

    def roisFilePath(self):
        return os.path.normpath(os.path.join(self.sima_path(), 'rois.pkl'))

    def ripplesFilePath(self):
        return os.path.normpath(os.path.join(self.sima_path(), 'ripples.pkl'))

    def hasSignalsFile(self, channel='Ch2'):
        try:
            with open(self.signalsFilePath(channel=channel)) as _:
                return True
        except:
            return False

    def hasTransientsFile(self, channel='Ch2'):
        """Return whether there are transients saved for the experiment"""
        try:
            with open(self.transientsFilePath(channel=channel)) as _:
                return True
        except:
            return False

    def hasDfofTracesFile(self, channel='Ch2'):
        """Return whether dFoF traces have been saved for the experiment"""
        try:
            with open(self.dfofFilePath(channel=channel)) as _:
                return True
        except:
            return False

    def hasPlaceFieldsFile(self, channel='Ch2'):
        """Return whether place fields have been saved for the experiment"""
        try:
            with open(self.placeFieldsFilePath(channel=channel)) as _:
                return True
        except:
            return False

    def hasAutoRoisFile(self):
        try:
            with open(self.roisFilePath(), "rb") as f:
                rois = pkl.load(f)
        except (IOError, pkl.UnpicklingError):
            return False
        return 'auto' in rois

    def signalsModTime(self):
        """Returns the modification time of the corresponding signals file
        as a text string or 'None'"""
        if not self.hasSignalsFile():
            return 'None'
        return time.strftime(
            '%Y-%m-%d-%Hh%Mm%Ss', time.localtime(
                os.path.getmtime(self.signalsFilePath())))

    def transientsModTime(self):
        """Returns the modification time of the corresponding transients file
        as a text string or 'None'"""
        if not self.hasTransientsFile():
            return 'None'
        return time.strftime(
            '%Y-%m-%d-%Hh%Mm%Ss', time.localtime(
                os.path.getmtime(self.transientsFilePath())))

    def dfof_tracesModTime(self):
        """Returns the modification time of the corresponding dfof traces file
        as a text string or 'None'"""
        if not self.hasDfofTracesFile():
            return 'None'
        return time.strftime(
            '%Y-%m-%d-%Hh%Mm%Ss', time.localtime(
                os.path.getmtime(self.dfofFilePath())))

    def belt(self):
        """Returns a Belt object corresponding to the beltID in the experiment
        attributes

        """
        if not hasattr(self, '_belt'):
            beltXmlPath = self.parent.parent.beltXmlPath
            beltID = self.get('belt', '')
            if beltID == '':
                raise exc.NoBeltInfo()

            with open(beltXmlPath) as f:
                doc = ElementTree.parse(f)

            root = doc.getroot()

            belt = root.find("./*[@beltID='" + beltID + "']")

            if belt is None:
                raise exc.NoBeltInfo('beltID not found in belts.xml: {}'.format(
                    beltID))

            belt.__class__ = Belt
            belt.backwards = (self.get('beltBackwards') == 'yes')

            self._belt = belt
        return self._belt

    def locate(self, feature):
        """Locate the position of a given feature.

        Parameters
        ----------
        feature : string or float
            Feature to locate. Valid values include any cue on the belt,
            'reward', a distance in cm as a string ('42cm'), or an already
            normalized position that will just be returned as is.

        Returns
        -------
        position : float
            Normalized position of desired feature.

        """
        if isinstance(feature, float):
            return feature
        elif feature == 'reward':
            reward_positions = self.rewardPositions(units='normalized')
            assert len(reward_positions) == 1
            return reward_positions[0]
        elif isinstance(feature, str) and \
                (feature.endswith('cm') or feature.endswith('mm')):
            belt_length = self.belt().length(
                units='cm' if feature.endswith('cm') else 'mm')
            feature_position = float(feature.rstrip('cm').rstrip('mm'))
            assert feature_position < belt_length
            return feature_position / belt_length
        else:
            try:
                return float(self.belt().cues(normalized=True).loc[
                    lambda df: df['cue'] == feature]['start'])
            except TypeError:
                raise ValueError("Unable to locate '{}'".format(feature))

    def imagingData(
            self, dFOverF=None, demixed=False, roi_filter=None,
            linearTransform=None, window_width=100, dFOverF_percentile=8,
            removeNanBoutons=False, channel='Ch2', label=None,
            trim_to_behavior=True, dataframe=False, dFOFTraceType="dFOF"):
        """Return a 3D array (with axes for ROI, time, and cycle number) of
        imaging data for the experiment.

        Parameters
        ----------
        dFOverF : {'from_file', 'mean', 'median', 'sliding_window', None}
            Method for converting to dFOverF
        demixed : bool
            whether to use ICA-demixed imaging data
        roi_filter : func
            Filter for selecting ROIs
        linearTransform : np.array
            Array specifying a linear transform to be performed on the input
            arguments, e.g. a matrix from PCA
        window_width : int
            Number of surrounding frames from which to calculate dFOverF using
            the sliding_baseline method
        dFOverF_percentile : int
            Percentile of the window to take as the baseline
        trim_to_behavior : bool
            If True, trim imaging data down to the length of
            the recorded behavior data
        dFOFTraceType : {"dFOF", "baseline", None}
            type of traces to get if dFoverF is "from_file", does not apply otherwise
            defaults to "dFOF"

        Returns
        -------
        imaging_data : np.array
            3D array (ROI, time, trial) of imaging data

        """

        if self.get('ignoreImagingData'):
            raise Exception('Imaging data ignored.')

        if label is None:
            label = self.most_recent_key(channel=channel)

        if trim_to_behavior:
            min_time = np.inf
            for trial in self.findall('trial'):
                trial_duration = trial.behaviorData()['recordingDuration']
                min_time = min(min_time, trial_duration)
            assert np.isfinite(min_time)
            trimmed_frames = int(min_time / self.frame_period())
        if dFOverF == 'from_file':
            if removeNanBoutons:
                warnings.warn(
                    "NaN boutons not removed when dF method is 'from_file'")
            path = self.dfofFilePath(channel=channel)
            try:
                with open(path, 'rb') as f:
                    dfof_traces = pkl.load(f)
            except (IOError, pkl.UnpicklingError):
                raise exc.NoDfofTraces('No dfof traces')


            if(dFOFTraceType == "baseline"):
                traceKey = "baseline"
            else:
                traceKey = "traces" if not demixed else "demixed_traces"

            try:
                traces = dfof_traces[label][traceKey]
#                 ['traces' if not demixed else 'demixed_traces']
            except KeyError:
                raise exc.NoDfofTraces('This label does not exist in dfof file')

            if trim_to_behavior:
                traces = traces[:, :trimmed_frames, :]

            if roi_filter is None:
                imData = traces
            else:
                try:
                    imData = traces[self._filter_indices(
                        roi_filter, channel=channel, label=label)]
                except KeyError:
                    raise exc.NoDfofTraces(
                        'No signals found for ch: {}, label: '.format(
                            channel, label))

        else:
            signals = self.imaging_dataset().signals(channel=channel)
            if len(signals) == 0:
                raise exc.NoSignalsData
            try:
                imData = signals[label]
            except KeyError:
                raise exc.NoSignalsData
            else:
                imData = imData['demixed_raw' if demixed else 'raw']
                # Check to make sure that all cycles have the same number of
                # frames, trim off any that don't match the first cycle
                frames_per_cycle = [cycle.shape[1] for cycle in imData]
                cycle_iter = zip(it.count(), frames_per_cycle)
                cycle_iter.reverse()
                for cycle_idx, cycle_frames in cycle_iter:
                    if cycle_frames != frames_per_cycle[0]:
                        warnings.warn(
                            'Dropping cycle with non-matching number of ' +
                            'frames: cycle_0: {}, cycle_{}: {}'.format(
                                frames_per_cycle[0], cycle_idx,
                                frames_per_cycle[cycle_idx]))
                        imData.pop(cycle_idx)
                imData = np.array(imData)
                imData = np.rollaxis(imData, 0, 3)

            if imData.ndim == 2:
                # Reshape data to always contain 3 dimensions
                imData = imData.reshape(imData.shape + (1,))

            if trim_to_behavior:
                imData = imData[:, :trimmed_frames, :]

            if roi_filter is None:
                indices = np.arange(imData.shape[0])
            else:
                try:
                    indices = self._filter_indices(
                        roi_filter, channel=channel, label=label)
                except KeyError:
                    raise exc.NoDfofTraces(
                        'No signals found for ch: {}, label: '.format(
                            channel, label))

            if removeNanBoutons:
                nan_indices = np.nonzero(np.any(np.isnan(
                    imData.reshape([imData.shape[0], -1], order='F')),
                    axis=1))[0]
                indices = list(set(indices).difference(nan_indices))

            # filter if filter passed in
            imData = imData[indices, :, :]

            # perform a linear transformation on the space of ROIs
            if linearTransform is not None:
                imData = np.tensordot(linearTransform, imData, axes=1)

            if dFOverF == 'mean':
                imData = dff.mean(imData)

            elif dFOverF == 'median':
                imData = dff.median(imData)

            elif dFOverF == 'sliding_window':
                imData = dff.sliding_window(
                    imData, window_width, dFOverF_percentile)

            elif dFOverF == 'sliding_window2':
                t0 = 2.  # exponential decay constant for df/f smoothing
                t1 = 8.  # size of sliding window for smoothing
                t2 = 400.  # size of baseline
                baselinePercentile = 5

                imData = dff.sliding_window_jia(
                    imData, t0, t1, t2, baselinePercentile,
                    self.frame_period())

            elif dFOverF == 'non-running-baseline':
                imData = dff.non_running_baseline(imData,
                                                  self.runningIntervals())

        if dataframe:
            data_list = []
            rois = self.rois(channel=channel, label=label,
                             roi_filter=roi_filter)
            for trial_idx, trial in enumerate(self.findall('trial')):
                for roi_idx, roi in enumerate(rois):
                    data = {"roi": roi,
                            "trial": trial,
                            "im_data": imData[roi_idx, :, trial_idx]
                            }
                    data_list.append(data)
            imData = pd.DataFrame(data_list)

        return imData

    def transientsData(
            self, threshold=95, behaviorSync=True, demixed=False,
            label=None, channel='Ch2', roi_filter=None, dataframe=False):
        """Load the transients data and return the corresponding transients
        structure for 95 or 99% confidence

        Keyword arguments:
        threshold -- transient confidence threshold, currently either 95 or 99
        behaviorSync -- if True, discards transients that occurred after
            behaviorData stopped recording

        Optional:
        dataframe --- bool, if True, returns dataframe of transients

        Returned structure is [ROI,cycle][key] record array

        """

        if not self.get('tSeriesDirectory'):
            raise exc.NoTSeriesDirectory
        path = self.transientsFilePath(channel=channel)

        if label is None:
            try:
                label = self.most_recent_key(channel=channel)
            except exc.NoSignalsData:
                raise exc.NoTransientsData(
                    'No signals for channel \'{}\''.format(channel))
        try:
            indices = self._filter_indices(
                roi_filter, channel=channel, label=label)
        except KeyError:
            raise exc.NoTransientsData(
                'No signals found for ch: {}, label: '.format(channel, label))

        if not hasattr(self, '_transients'):
            try:
                with open(path, 'rb') as file:
                    self._transients = pkl.load(file)
            except (IOError, pkl.UnpicklingError):
                raise exc.NoTransientsData
        transients = deepcopy(self._transients)

        p = np.round((100 - threshold) / 100., 3)
        try:
            trans = transients[label][
                'demixed_transients' if demixed else 'transients'][p][indices]
        except KeyError:
            raise exc.NoTransientsData(
                "No {} transients data for label '{}'".format(
                    'demixed' if demixed else 'non-demixed', label))

        if behaviorSync:
            # The point of this option is to remove all transients that occur
            # after behavior data was finished recording
            # num_frames is the length of the imageSync'd behaviorData, which
            # could be shorter than imagingData
            num_frames = self.num_frames(
                trim_to_behavior=True, channel=channel, label=label)
            for trialIdx in range(trans.shape[1]):
                for roiIdx in range(trans.shape[0]):
                    for transIdx, (start, stop, t_max) in enumerate(zip(
                            trans[roiIdx, trialIdx]['start_indices'],
                            trans[roiIdx, trialIdx]['end_indices'],
                            trans[roiIdx, trialIdx]['max_indices'])):
                        if start >= num_frames or stop >= num_frames or \
                                t_max >= num_frames:
                            # As soon as we find any index outside the duration
                            # of the behaviorData, chop off the rest of the
                            # transients and go on to the next ROI
                            for key in ['start_indices', 'end_indices',
                                        'max_indices', 'max_amplitudes',
                                        'durations_sec']:
                                trans[roiIdx, trialIdx][key] = \
                                    trans[roiIdx, trialIdx][key][:transIdx]
                            break

        if dataframe:
            rois = self.rois(
                channel=channel, label=label, roi_filter=roi_filter)
            assert len(rois) == trans.shape[0]

            data_list = []
            for roi, roi_trans in zip(rois, trans):
                for trial, roi_trial_trans in zip(
                        self.findall('trial'), roi_trans):
                    for t_idx, (start, end, max_amp, dur, max_idx) in \
                            enumerate(zip(
                                roi_trial_trans['start_indices'],
                                roi_trial_trans['end_indices'],
                                roi_trial_trans['max_amplitudes'],
                                roi_trial_trans['durations_sec'],
                                roi_trial_trans['max_indices'])):
                        data = {
                            'trial': trial, 'roi': roi,
                            'roi_id': roi.id,
                            'mouseID': roi.expt.parent.get('mouseID'),
                            'uniqueLocationKey': roi.expt.get(
                                'uniqueLocationKey'),
                            'sigma': roi_trial_trans['sigma']}
                        data['trans_idx'] = t_idx
                        data['start_frame'] = start
                        data['stop_frame'] = end
                        data['max_amplitude'] = max_amp
                        data['duration'] = dur
                        data['max_frame'] = max_idx
                        data_list.append(data)

            return pd.DataFrame(data_list)

        return trans

    def spikes(self, channel='Ch2', label=None, roi_filter=None):
        """Return spike times in seconds."""
        if not self.get('tSeriesDirectory'):
            raise exc.NoTSeriesDirectory
        path = self.spikesFilePath(channel=channel)

        if label is None:
            try:
                label = self.most_recent_key(channel=channel)
            except exc.NoSignalsData:
                raise exc.NoTransientsData(
                    'No signals for channel \'{}\''.format(channel))
        try:
            indices = self._filter_indices(
                roi_filter, channel=channel, label=label)
        except KeyError:
            raise exc.NoTransientsData(
                'No signals found for ch: {}, label: '.format(channel, label))

        if not hasattr(self, '_spikes'):
            try:
                with open(path, 'rb') as file:
                    self._spikes = pkl.load(file)
            except (IOError, pkl.UnpicklingError):
                raise exc.NoSpikesData
        spikes = deepcopy(self._spikes)

        return [cycle[indices] for cycle in spikes[label]['spike_times']]

    def transSubset(self, transtype, threshold=99, label=None,
                    channel='Ch2', demixed=False, roi_filter=None,
                    pre=0.5, post=0.75, percentile=None):

        """Load the subset of the transients data that represent
        dendritic spiking in the absence of coincident somatic activity

        """
        # Warning: cycle 0 is hard-coded

        transients = self.transientsData(threshold=threshold, label=label,
                                         channel=channel, demixed=demixed,
                                         roi_filter=roi_filter)

        rois = self.rois(label=label, roi_filter=roi_filter)

        pre_tol = int(pre / self.frame_period())
        post_tol = int(post / self.frame_period())

        col_names = ['start_indices',
                     'end_indices',
                     'max_amplitudes',
                     'durations_sec',
                     'max_indices']

        cell_names = [r.label for r in rois if '_' not in r.label]

        def in_somatic_window(soma_start, dend_start,
                              pre_tolerance, post_tolerance):

            if dend_start >= soma_start:
                if dend_start - soma_start < post_tolerance:
                    return True
            else:
                if soma_start - dend_start < pre_tolerance:
                    return True

            return False

        for cell_name in cell_names:

            cell_idx = [i for i, r in enumerate(rois) if
                        r.label == cell_name][0]

            d_idxs = [i for i, r in enumerate(rois) if
                      r.label.startswith(cell_name + '_')]

            if percentile and len(transients[cell_idx][0]['max_amplitudes']):
                cut_off = np.percentile(transients[cell_idx][0]['max_amplitudes'], percentile)
                soma_starts = [s for s, a in zip(transients[cell_idx][0]['start_indices'],
                                                 transients[cell_idx][0]['max_amplitudes'])
                               if a <= cut_off]
            else:
                soma_starts = transients[cell_idx][0]['start_indices']

            for di in d_idxs:

                dend_starts = transients[di][0]['start_indices']

                if transtype == 'isolated':
                    # Delete any bAPs, i.e. any event in a somatic window
                    del_indices = [i for i, dstart in enumerate(dend_starts) if
                                   any([in_somatic_window(soma_start, dstart,
                                                          pre_tol, post_tol)
                                        for soma_start in soma_starts])]

                elif transtype == 'somatic':

                    del_indices = [i for i, dstart in enumerate(dend_starts) if
                                   not any([in_somatic_window(soma_start, dstart,
                                                              pre_tol, post_tol)
                                           for soma_start in soma_starts])]

                for col_name in col_names:

                    transients[di][col_name][0] = np.delete(
                        transients[di][col_name][0], del_indices)

        # Remove all transients for those dendrites
        # where the associated soma was not imaged
        no_soma_idxs = [i for i, r in enumerate(rois) if
                        r.label.split('_')[0] not in cell_names]

        for idx in no_soma_idxs:

            n_events = transients[idx]['start_indices'][0].shape[0]

            del_events = range(n_events)

            for col_name in col_names:

                transients[idx][col_name][0] = np.delete(
                    transients[idx][col_name][0], del_events)

        return transients

    def ripple_frames(self, trigger='start_times'):
        path = self.ripplesFilePath()

        try:
            with open(path, 'rb') as fp:
                ripples = pkl.load(fp)[trigger]
        except (IOError, KeyError):
            return None
        else:
            return [int(np.floor(r / (self.frame_period())))
                    for r in ripples]

    def transientsParameters(self, label=None, channel='Ch2'):
        transients_path = self.transientsFilePath(channel)

        if label is None:
            try:
                label = self.most_recent_key(channel=channel)
            except exc.NoSignalsData:
                raise exc.NoTransientsData(
                    "No signals for channel '{}'".format(channel))

        with open(transients_path, 'r') as file:
            transients = pkl.load(file)[label]

        return transients['parameters']

    def pca(self, **kwargs):
        """Perform principle component analysis on the experiment imaging ROI signals.

        See ExperimentGroup.PCA

        """

        return ExperimentGroup([self]).pca(**kwargs)

    def imagingIndex(self, time):
        """ get the imaging index associated with the given time """
        return int(time / self.frame_period())

    def imagingTimes(self, channel='Ch2'):
        """Return a list containing the times in second at which images
        individual frames were acquired in the cycle"""

        framePeriod = self.frame_period()
        return np.array([float(x) * framePeriod for x in range(
            self.num_frames(channel=channel))])

    def is_imaged(self):
        """Return whether the experiment has imaging data"""
        return bool(self.get('tSeriesDirectory'))

    def is_motion_corrected(self):
        """Return whether the experiment has already been motion corrected."""
        warnings.warn(DeprecationWarning)
        # TODO: REMOVE THIS FUNCTION, NOT A GOOD CHECK, MIGHT AS WELL CALL
        # HAS_SIMA_DIR
        try:
            self.sima_path()
        except (exc.NoTSeriesDirectory, exc.NoSimaPath):
            return False
        return True

    def imaged(self):
        """ Return whether the experiment has imaging data"""

        # TODO: REMOVE THIS FUNCTION, REDUNDANT

        warnings.warn('imaged method is deprecated, replaced by is_imaged',
                      DeprecationWarning)
        return self.is_imaged()

    def running(self, **kwargs):
        """ Returns running activity from behavioral data, for each trial,
        within the given start and stop times

        """

        result = []
        for cycle in self.findall('trial'):
            result.append(cycle.running(**kwargs))
        return result


    def runningIntervals(self, **kwargs):
        """ Returns a list (one entry per cycle) of np arrays, each containing
        the starting and stopping imaging index for each running interval.

        See Trial.runningIntervals for details.

        """

        result = []
        for cycle in self.findall('trial'):
            result.append(ba.runningIntervals(cycle, **kwargs))
        return result

    def rippleIntervals(self, window=1, returnBoolList=False):
        result = []
        ripples = self.ripple_frames()
        for ripple in ripples:
            rstart = np.max([0, ripple - window])
            rstop = np.min([self.num_frames(), ripple + window])
            result.append([rstart, rstop])

        result = np.array(result)

        if returnBoolList:
            boolList = np.zeros(self.num_frames(), dtype=bool)
            for interval in result:
                boolList[interval[0]:interval[1]+1] = True

            return boolList

        return result

    def runningIntervals_old(self, **kwargs):
        """ Returns a list (one entry per cycle) of np arrays, each containing
        the starting and stopping imaging index for each running interval.

        See Trial.runningIntervals for details.
        """
        result = []
        for cycle in self.findall('trial'):
            result.append(cycle.runningIntervals_old(**kwargs))
        return result

    def velocity(self, **kwargs):
        """ Returns a list (one entry per cycle) of np arrays, each containing
        the velocity of the mouse at the current imaging frame

        See Trial.velocity for details.

        """

        result = []
        for cycle in self.findall('trial'):
            result.append(ba.velocity(cycle, **kwargs))
        return result

    def lickingIntervals(self, **kwargs):
        """ Returns a list (one entry per cycle) of np arrays, each containing
        the starting and stopping imaging index for each licking interval.

        See Trial.runningIntervals for details.

        """

        result = []
        for cycle in self.findall('trial'):
            result.append(ba.lickingIntervals(cycle, **kwargs))
        return result

    def runningModulation(self, **kwargs):
        return ia.runningModulation(ExperimentGroup([self]), **kwargs)

    def lickingModulation(self, **kwargs):
        return ia.lickingModulation(ExperimentGroup([self]), **kwargs)

    def drugString(self):
        """Give a string describing the drug status"""
        s = ""
        for drug in self.findall('drug'):
            s += drug.get('name')
            if drug.get('method') is not None:
                s += ', ' + drug.get('method')
        s += self.get('drug', '')
        if s in [None, "", "pre", "control", "ctrl", "Vehicle"]:
            return 'control'
        return s

    def drug_parameters(self):
        if 'drug' in self.attrib:
            return eval(self.get('drug'))
        # Need to add parsing of drug XML entries like in drugString
        return []

    def exptNum(self):
        """Return the unique sequential number of this experiment for it's
        mouse.

        """
        return self.parent.findall('experiment').index(self)

    def returnFinalPrototype(self, channel='Ch2', plane=None):
        """Returns the uint16 data from final_prototype.tif"""

        time_averages = self.imaging_dataset().time_averages

        prototype = time_averages[..., self.imaging_dataset()._resolve_channel(
            channel)]
        if plane is not None:
            prototype = prototype[plane, ...]

        return prototype

    def roiVertices(
            self, roi_filter=None, channel='Ch2', label=None, two_d=False):
        """Returns a list of lists of vertices for the ROIs"""

        rois = self.rois(channel=channel, label=label, roi_filter=roi_filter)
        verts = []
        for roi in rois:
            roi_verts = roi.coords
            if two_d:
                for idx, poly in enumerate(roi_verts):
                    roi_verts[idx] = poly[:, :2]
            verts.append(roi_verts)

        return verts

    def pcaCleanupMatrix(self, **kwargs):
        return ExperimentGroup([self]).pcaCleanupMatrix(**kwargs)

    def positionOccupancy(self, **kwargs):
        return ba.positionOccupancy(ExperimentGroup([self]), **kwargs)

    def identify_active_cells(self, **kwargs):
        """Identify active cells, returns an ROI filter.

        See ExperimentGroup.identify_active_cells for details

        """
        return ExperimentGroup([self]).identify_active_cells(**kwargs)

    def reward_parameters(self, distance_units='mm'):
        """Return water reward parameters.

        Parameters
        ----------
        distance_units : {'mm', 'normalized', None}
            Determines units of distance reward parameters. 'mm' converts to
            real units, 'normalized' normalizes to the length of the belt
            (all values will be on [0, 1)), and None does not convert at all.

        Returns
        -------
        params : dict
            'type' : None, 'fixed', 'fixed_number', 'moving', 'continuous'
                The type of reward schedule. None is no automatic rewards,
                'fixed' is at specific locations (which should also be
                specified), 'fixed number' is a fixed number of rewards evenly
                spaced throughout the belt (same locations each lap, legacy
                paradigm), 'moving' is a fixed number of random rewards per
                lap, and 'continuous' is a constant reward window (lick
                training).
            'locations' : list of float
                The locations of fixed reward start positions (in mm from start
                of belt).
            'number' : int
                The number of rewards per lap for 'fixed number' or 'moving'
                reward paradigms.
            'initial_open' : float
                Time of initial valve opening when the mouse enters a reward
                zone (in ms).
            'duration' : float
                Time of valve opening for each water reward (in seconds).
            'window_length' : float
                Length of operant reward window (in mm).
            'window_duration' : float
                Duration of operant reward window (in seconds).
            'operant_rate' : int
                Number of licks per operant reward.

        """
        if 'reward' in self.attrib:
            json_behavior = True

            # need to eval this if type is string instead of dictionary
            try:
                params = eval(self.get('reward'))
            except TypeError:
                params = self.get('reward')
        else:
            json_behavior = False
            key_list = [
                'operantTemporalWindow', 'operantSpatialWindow',
                'operantRewardRate', 'rewardType', 'rewardPositions',
                'rewardDuration', 'spatialRewardRate', 'automatedRewards']
            params = {k: self.get(k) for k in key_list if k in self.attrib}

        rename_keys = [('rewardType', 'type'),
                       ('operantTemporalWindow', 'window_duration'),
                       ('operantRewardRate', 'operant_rate'),
                       ('rewardDuration', 'duration'),
                       ('spatialRewardRate', 'number'),
                       ('drop_size', 'duration'),
                       ('max_duration', 'window_duration'),
                       ]
        for old, new in rename_keys:
            if params.get(old):
                params[new] = params.pop(old)

        rename_types = [('scattered', 'fixed'),
                        ('spatial', 'fixed_number'),
                        ('probabilistic', 'moving')]
        for old, new in rename_types:
            if params.get('type') == old:
                params['type'] = new
        if params.get('automatedRewards') == 'yes' and not params.get('type'):
            # Original reward experiments didn't have a 'type' key or equivalent
            params['type'] = 'fixed_number'

        if params.get('radius'):
            params['window_length'] = 2 * float(params.pop('radius'))
            if distance_units == 'normalized':
                belt_length = self.belt().length(units='mm')
                params['window_length'] /= belt_length

        if params.get('operantSpatialWindow'):
            if distance_units is None:
                params['window_length'] = params.pop('operantSpatialWindow')
            else:
                track_length = np.mean([trial.behaviorData()['trackLength']
                                        for trial in self.findall('trial')])
                params['window_length'] = float(
                    params.pop('operantSpatialWindow')) / track_length
                if distance_units == 'mm':
                    belt_length = self.belt().length(units='mm')
                    params['window_length'] *= belt_length

        if isinstance(params.get('locations', 0), list) and json_behavior:
            if distance_units == 'normalized':
                belt_length = self.belt().length(units='mm')
                params['locations'] = [
                    loc / belt_length for loc in params['locations']]
            params['locations'] = [loc - params['window_length'] / 2.
                                   for loc in params.get('locations')]
            # Random reward locations
        elif isinstance(params.get('locations', None), int) and json_behavior:
            params.pop('locations', None)
        elif not json_behavior and (params.get('rewardPositions') or
                                    params.get('type') == 'fixed_number'):
            if params.get('type') == 'fixed_number':
                number = int(params.get('number', 0))
                try:
                    track_length = np.mean(
                        [trial.behaviorData()['trackLength'] for trial in
                         self.findall('trial')])
                except KeyError:
                    track_length = int(self.get('trackLength'), 0)
                locations = [
                    n / float(number) * track_length for n in range(number)] \
                    if number else []
            else:
                locations = [float(x) for x in
                             params.pop('rewardPositions').split(',')]
            if distance_units is None:
                params['locations'] = locations
            else:
                track_length = np.mean([trial.behaviorData()['trackLength']
                                        for trial in self.findall('trial')])
                params['locations'] = [loc / track_length for loc in locations]
                if distance_units == 'mm':
                    belt_length = self.belt().length(units='mm')
                    params['locations'] = [
                        loc * belt_length for loc in params['locations']]

        if params.get('automatedRewards') == 'no' or not len(params):
            params['type'] = None

        params.pop('automatedRewards', '')

        return params

    def rewardPositions(self, units='normalized'):
        """Return positions of rewards.

        Parameters
        ----------
        units : {'mm', 'normalized', None}
            Determines units of reward positions 'mm' converts to
            real units, 'normalized' normalizes to the length of the belt
            (all values will be on [0, 1)), and None does not convert at all.

        """
        positions = self.reward_parameters(
            distance_units=units).get('locations', [])
        return np.array(positions)

    def __repr__(self):
        s = "< Experiment: " + self.parent.get('mouseID', '')
        for key, value in self.attrib.iteritems():
            s = s + ", " + str(key) + ' = ' + str(value)
        s = s + '>'
        return s

    def __str__(self):
        s = "< Experiment: {}_{}, {}, signals={}".format(
            self.parent.get('mouseID'), self.get('startTime'),
            self.get('experimentType'), self.signalsModTime())

        reward_params = self.reward_parameters(distance_units=None)
        for key in sorted(reward_params):
            s += ", {}={}".format(key, reward_params[key])

        return s + ">"


class SalienceExperiment(Experiment):

    def stimuli(self):
        """Return the set of different stimuli that were presented"""
        return list(set(
            [trial.get('stimulus') for trial in self.findall('trial')
             if trial.get('stimulus') is not None]))

    def trialIndices(self, stimulus, duration=None, power=None,
                     check_shape=True, excludeRunning=False):
        """Return the indices of the trials for which a given stimulus/duration
        is given.

        """

        indices = [i for i, trial in enumerate(self.findall('trial'))
                   if trial.get('stimulus') == stimulus
                   and (not check_shape or i < self.imaging_shape()[2])
                   and (power is None or int(trial.get('power')) == int(power))
                   and (duration is None or trial.duration() == int(duration))]
        if excludeRunning:
            tmpIndices = []
            for idx in indices:
                try:
                    if len([x for x in self.findall('trial')[idx].behaviorData()['treadmillTimes'] if x > self.stimulusTime() - 1.0
                        and x < self.stimulusTime()+ 2.0]) < 2:
                      tmpIndices.append(idx)
                except exc.MissingBehaviorData:
                    pass
            indices = tmpIndices
        return indices

    def stimulusTime(self):
        """Returns the time after the start of data acquisition at which
        the stimulus was presented.

        Returns NaN if no stimulus presentation recorded.

        """

        if self.get('stimulusTime', None) is not None:
            return float(self.get('stimulusTime'))
        else:
            onsetTimes = []
            for trial in self.findall('trial'):
                stimulus = trial.get('stimulus', '')
                if stimulus.lower().count('puff') \
                        or stimulus.lower().count('air'):
                    try:
                        onsetTimes.append(
                            trial.behaviorData()['airpuff'][0, 0])
                    except:
                        pass
                else:
                    try:
                        onsetTimes.append(trial.behaviorData()[stimulus][0, 0])
                    except:
                        pass
            return np.median(onsetTimes)

    def baselineActivityCorrelations(self, **kwargs):
        """Return the correlation matrix for the activity across ROIs before the stimuli"""
        return ia.baselineActivityCorrelations(ExperimentGroup([self]), **kwargs)

    def trialAverages(self, stimulus, duration=None, excludeRunning=False, power=None, removeNanBoutons=False, **kwargs):
        """Return the average signal timeseries for trials with a given stimulus type"""
        return ia.trialAverages(
            ExperimentGroup([self]), stimulus, duration=duration, excludeRunning=excludeRunning,
            power=power, **kwargs)

    def peakAverageResponses(self, stimulus, **kwargs):
        """Calculate the peak of the trial-averaged response of each roi to the stimulus"""
        trialAv = self.trialAverages(stimulus, **kwargs)
        stimIdx = self.imagingIndex(self.stimulusTime())
        baseline = np.mean(trialAv[:, :stimIdx], axis=1)
        return max(trialAv[:, stimIdx:(2 * stimIdx)] - baseline, axis=1) / baseline

    def averageResponseIntegrals(
            self, stimulus, duration=None, power=None, excludeRunning=False,
            demixed=False, dFOverF=None, postStimDuration=1.5,
            sharedBaseline=True, linearTransform=None, channel='Ch2',
            label=None, roi_filter=None):
        """Return the integral of the trial-averaged response of each roi to the stimulus"""
        return self.responseIntegrals(
            stimulus, postStimDuration=postStimDuration, duration=duration,
            linearTransform=linearTransform, excludeRunning=excludeRunning,
            demixed=demixed, dFOverF=dFOverF,
            sharedBaseline=sharedBaseline, channel=channel, label=label,
            roi_filter=roi_filter).mean(axis=1)

    def responseIntegrals(
            self, stimulus, postStimDuration=1.5, duration=None, power=None,
            excludeRunning=False, demixed=False, dFOverF=None,
            sharedBaseline=False, linearTransform=None, channel='Ch2',
            label=None, roi_filter=None):

        timeSeries = self.imagingData(
            demixed=demixed, dFOverF=dFOverF,
            linearTransform=linearTransform, channel=channel, label=label,
            roi_filter=roi_filter)
        trialIndices = self.trialIndices(stimulus, duration, power=power,
                                         excludeRunning=excludeRunning)
        stimIdx = self.imagingIndex(self.stimulusTime())
        maxIdx = stimIdx + self.imagingIndex(postStimDuration)
        timeSeries = timeSeries[:, :, trialIndices]
        if sharedBaseline:
            baseline = timeSeries[:, :stimIdx, :].mean(axis=2).mean(axis=1)
            for i in range(timeSeries.shape[0]):
                timeSeries[i, :, :] -= baseline[i]
                if dFOverF in ['none', None]:
                    timeSeries[i, :, :] /= baseline[i]
        else:
            baseline = timeSeries[:, :stimIdx, :].mean(axis=1)
            for i in range(timeSeries.shape[0]):
                for j in range(timeSeries.shape[2]):
                    timeSeries[i, :, j] -= baseline[i, j]
                    if dFOverF in ['none', None]:
                        timeSeries[i, :, j] /= baseline[i, j]

        return timeSeries[:, stimIdx:maxIdx, :].sum(axis=1) * \
            self.frame_period()

    def responseZScores(self, stimulus, postStimDuration=1.5, duration=None,
                        sharedBaseline=False, excludeRunning=False,
                        demixed=False, dFOverF=None,
                        linearTransform=None, roi_filter=None):
        responseIntegrals = self.responseIntegrals(
            stimulus, duration=duration, excludeRunning=excludeRunning,
            demixed=demixed, linearTransform=linearTransform,
            dFOverF=dFOverF, roi_filter=roi_filter,
            postStimDuration=postStimDuration, sharedBaseline=sharedBaseline)
        return responseIntegrals.mean(axis=1) / np.sqrt(
            np.var(responseIntegrals, axis=1)) * np.sqrt(
            responseIntegrals.shape[1])

    def responsePairPlot(self, ax, stim1, stim2, excludeRunning=False,
                         **kwargs):
        ap.responsePairPlot(
            ExperimentGroup([self]), ax, stim1, stim2,
            excludeRunning=excludeRunning, **kwargs)

    def trialAverageHeatmap(self, stimulus, ax=None, roi_list=None, sort=False,
                            smoothing=None, window_length=5, vmin=None,
                            vmax=None, exclude_running=False, **imaging_kwargs):
        """ Plots a trial averaged heatmap for a given stimulus type

        Keyword arguments:
        stimulus -- stimulus to analyze, passed to trialAverages
        roi_list -- list of roi numbers to plot separately at the top in order
        sort -- if True, sorts rois by their peak activity
        smoothing -- determines smoothing, can be 'None', 'flat', or any np.X smoothing function ('hamming', 'hanning', 'barltett', etc)
        window_length -- smoothing window function length

        """

        if ax is None:
            ax = plt.axes()

        data = self.trialAverages(
            stimulus, removeNanBoutons=False, excludeRunning=exclude_running,
            **imaging_kwargs)

        data_to_plot = data.copy()

        if smoothing is not None:
            if smoothing == 'flat':  # moving average
                w = np.ones(window_length, 'd')
            else:
                # If 'smoothing' is not a valid method this will throw an AttributeError
                w = eval('np.' + smoothing + '(window_length)')
            for idx, row in enumerate(data_to_plot):
                s = np.r_[row[window_length - 1:0:-1], row, row[-1:-window_length:-1]]
                tmp = np.convolve(w / w.sum(), s, mode='valid')
                # Trim away extra frames
                data_to_plot[idx] = tmp[window_length / 2 - 1:-window_length / 2]

        if roi_list is not None:
        # Add an empty row to the end that will divide cell groups
            empty_array = np.empty((1, data_to_plot.shape[1]))
            empty_array.fill(np.nan)
            data_to_plot = np.vstack((data_to_plot, empty_array))
            roi_list = list(roi_list)
            other_cells = np.setdiff1d(range(data.shape[0]), roi_list)
            if sort:
                sorted = np.argsort(np.amax(data, axis=1))
                order = roi_list + [-1] + sorted[np.in1d(sorted, other_cells)].tolist()
            else:
                order = roi_list + [-1] + other_cells.tolist()
        else:
            if sort:
                order = np.argsort(np.amax(data, axis=1)).tolist()
            else:
                order = range(data.shape[0])

        stim_time = self.stimulusTime()
        imaging_times = self.imagingTimes()

        left = imaging_times[0] - stim_time
        right = imaging_times[-1] - stim_time
        bottom = data_to_plot.shape[0]
        top = 0

        if vmin is None:
            vmin = np.percentile(data_to_plot, 10)
        if vmax is None:
            vmax = np.percentile(data_to_plot, 99)
        #vmin = None
        #vmax = None
        ax.imshow(data_to_plot[order], vmin=vmin, vmax=vmax,
                  interpolation='none', aspect='auto',
                  extent=(left, right, bottom, top))
        ylim = ax.get_ylim()
        ax.vlines(0, ylim[0], ylim[1], color='w', linestyles='dashed')
        ax.set_ylim(ylim)

        ax.set_xlabel('Time from stimulus (s)')
        ax.set_ylabel('ROIs')
        ax.tick_params(which='both', direction='out', left='off', right='off',
                       top='off', labelleft='off')

    def response_matrix(self, stimuli, **kwargs):
        return ia.response_matrix(ExperimentGroup([self]), stimuli, **kwargs)

    def stim_response_overlay(
            self, ax, stimuli=None, roi_filter=None, plot_method='angle',
            **response_kwargs):
        """Plot an ROI overlay where ROIs are colored by similarity"""

        if stimuli is None:
            stimuli = self.stimuli()

        data = self.response_matrix(stimuli, roi_filter=roi_filter, **response_kwargs)

        unity = np.ones(data.shape[1])

        if len(stimuli) == 1:
            values = data[:, 0] - np.amin(data)
            values /= np.amax(values)
        else:
            values = []
            for roi in data:
                if plot_method == 'angle':
                    values.append(
                        np.dot(roi, unity) / np.linalg.norm(roi)
                        / np.linalg.norm(unity))
                elif plot_method == 'corr':
                    values.append(np.corrcoef(roi, unity)[0, 1])
                else:
                    raise ValueError('Invalid plot method')

        imaging_parameters = self.imagingParameters()
        aspect_ratio = imaging_parameters['pixelsPerLine'] \
            / imaging_parameters['linesPerFrame']

        plotting.roiDataImageOverlay(
            ax, self.returnFinalPrototype(plane=0), self.roiVertices(roi_filter=roi_filter, two_d=True),
            values=np.array(values), aspect=aspect_ratio, vmin=0, vmax=1)

    # END SalienceExperiment class


class DoubleStimulusExperiment(SalienceExperiment):

    def singleStimTrialAverage(self, stimulus):
        raise Exception('Code incomplete')

    def doubleStimulusTrialAverage(self, stimulus, delay):
        raise Exception('Code incomplete')


class IntensityRangeExperiment(SalienceExperiment):

    def durations(self, stimulus):
        d = []
        for trial in self.findall('trial'):
            if trial.get('stimulus') == stimulus:
                try:
                    d.append(int(trial.get('duration')))
                except TypeError:
                    if stimulus == 'air':
                        d.append(int(trial.parent.get('airpuffDuration')))
                    else:
                        raise
        return sorted(list(set(d)))

    def tonePowers(self):
        d = []
        for trial in self.findall('trial'):
            if trial.get('stimulus') in ['tone', 'noise']:
                d.append(int(trial.get('power')))
        return sorted(list(set(d)))

    def intensityStats(self, stimulus, excludeRunning=False, demixed=False,
                       dFOverF=None, channel='Ch2',
                       label=None, roi_filter=None):
        data = []
        if stimulus == 'tone':
            for power in self.tonePowers():
                data.append(self.averageResponseIntegrals(
                    stimulus, power=power, excludeRunning=excludeRunning,
                    demixed=demixed, dFOverF=dFOverF,
                    channel=channel, label=label, roi_filter=roi_filter))
        else:
            for duration in self.durations(stimulus):
                data.append(self.averageResponseIntegrals(
                    stimulus, duration=duration, excludeRunning=excludeRunning,
                    demixed=demixed, dFOverF=dFOverF,
                    channel=channel, label=label, roi_filter=roi_filter))
        assert np.all(np.isfinite(data[0]))
        return np.array(data)


class FearConditioningExperiment(Experiment):

    def lickCountInContext(self, duration=None):
        """ Returns total licks during the context. FearConditioningExperiements should have only one trial """
        context = self.contextInterval(mask=False)
        if duration is None:
            endTime = context[1]
        else:
            endTime = context[0] + duration
        licks = ba.lickCount(self.find('trial'), context[0], endTime)
        return licks

    def lickRateInContext(self):
        """ Returns licks per second during the context. FearConditioningExperiements should have only one trial """
        context = self.contextInterval(mask=False)
        licks = ba.lickCount(self.find('trial'), context[0], context[1])
        return licks / float((context[1] - context[0]))

    def lickRatePreContext(self):
        """Returns licks per second before the context. FearConditioningExperiments should have only one trial """
        context = self.contextInterval(mask=False)
        licks = ba.lickCount(self.find('trial'), 0, context[0])
        return licks / float((context[0] - 0))

    def lickRateTracePeriod(self):
        """Returns licks per second during the trace period. FearConditioningExperiments should have only one trial """
        context = self.contextInterval(mask=False)
        licks = ba.lickCount(self.find('trial'), context[1], context[1] + 15)
        return licks / float((15))

    def meanVelocityInContext(self, belt_length=200):
        """ Returns average velocity during the context. """

        try:
            belt_length = self.belt().length(units='cm')
        except exc.NoBeltInfo:
            warnings.warn('No belt information found for experiment %s.  \nUsing default belt length = %f' % (str(self), belt_length))

        context = self.contextInterval(mask=False)
        velocity = ba.velocity(
            self.find('trial'), imageSync=False, belt_length=belt_length, smoothing='hanning',
            window_length=71)
        samplingInterval = self.find('trial').behavior_sampling_interval()
        return np.mean(velocity[context[0] / samplingInterval:
                                context[1] / samplingInterval + 1])

    def isPuffed(self):
        """
        Returns whether or not this experiment was puffed.
        First checks for a 'autoPuff' attribute, then attempts to load behavior data
        """

        if 'autoPuff' in self.keys():
            return (self.attrib['autoPuff'] == 'yes')
        elif 'airpuffAuto' in self.keys():
            return (self.attrib['airpuffAuto'] == 'yes')
        else:
            return (self[0].behaviorData()['airpuff'].shape[0] > 0)

    def contextInterval(self, mask=False):
        """
        Returns the bounds of the context, first by checking for attributes, and if missing
        then checks the 'odor' field in the behavior data
        If mask=False, returns the context start and stop times.
        If mask=True, returns a boolean array for indexing
        """

        bd = self.find('trial').behaviorData()

        # Determine which field to use
        field = None
        if 'odor' in bd.keys() and len(bd['odor']) is 1 and len(bd['odor'][0]) is 2 and \
            not np.isnan(bd['odor'][0][0]) and not np.isnan(bd['odor'][0][1]):
            field = 'odor'
        # if 'odorA' looks good use it, else check 'odorB'
        elif 'odorA' in bd.keys() and len(bd['odorA']) is 1 and len(bd['odorA'][0]) is 2 and \
            not np.isnan(bd['odorA'][0][0]) and not np.isnan(bd['odorA'][0][1]):
            field = 'odorA'
        elif 'odorB' in bd.keys() and len(bd['odorB']) is 1 and len(bd['odorB'][0]) is 2 and \
            not np.isnan(bd['odorB'][0][0]) and not np.isnan(bd['odorB'][0][1]):
            field = 'odorB'

        if field is not None:
            interval = [ int(bd[field][0][0]), int(bd[field][0][1]) ]
        elif 'environmentOnsetDelay' in self.keys() and 'environmentDuration' in self.keys():
            interval =  [ float(self.attrib['environmentOnsetDelay']), \
                      float(self.attrib['environmentOnsetDelay']) + float(self.attrib['environmentDuration']) ]
        else:
            raise Exception('Unable to determine context onset and offset: ' + self.get('startTime'))

        if not mask:
            return interval
        else:
            boolMask = np.zeros(self.imaging_shape()[1], 'bool')
            boolMask[interval[0]:interval[1]] = True
            return boolMask

    def fullPlot(self, ax, style='color', yOffsets=None, linearTransform=None,
                 channel='Ch2', label=None, roi_filter=None):
        trial = self.find('trial')
        ROIs = ExperimentGroup([self]).sharedROIs(channel=channel, label=label,
                                                  roi_filter=roi_filter)
        shared_filter = lambda x: x.id in ROIs
        ap.activityPlot(trial, ax, style=style, yOffsets=yOffsets,
                           linearTransform=linearTransform, channel=channel,
                           label=label, roi_filter=shared_filter)
        ymin = ax.get_ylim()[0]
        ax.set_ylim(bottom=ymin - 2)
        behaviorData = trial.behaviorData()
        odorA = filter(lambda k: k in behaviorData.keys(), ['odorA', 'odor A', 'odor'])[0]
        odorB = filter(lambda k: k in behaviorData.keys(), ['odorB', 'odor B', 'odor'])[0]
        if behaviorData[odorA].size:
            ctx = behaviorData[odorA]
        elif behaviorData[odorB].size:
            ctx = behaviorData[odorB]
        else:
            ctx = None
        if self.get('environment') == 'A':
            ctxColor = 'c'
        else:
            ctxColor = 'y'
        if ctx is not None:
            ax.add_patch(
                Rectangle((ctx[0][0], ymin - 1), ctx[0][1] - ctx[0][0], 1,
                          facecolor=ctxColor, lw=0, alpha=0.5))

    def lickPlot(self, axis):
        colors = {'A': 'g', 'B': 'r'}
        behavior_data = self.find('trial').behaviorData()
        if any('odor' in key for keys in behavior_data.keys()):
            ctx_keys = [key for keys in behavior_data.keys() if 'odor' in key]
            for ctx_key in ctx_keys:
                if behavior_data[ctx_key].size:
                    ctx = behavior_data[ctx_key]
                    if 'A' in ctx_key:
                        color = colors['A']
                    elif 'B' in ctx_key:
                        color = colors['B']
                else:
                    continue
        # if behavior_data['odor A'].size:
        #     ctx = behavior_data['odor A']
        #     color = 'g'
        # elif behavior_data['odorB'].size:
        #     ctx = behavior_data['odor B']
        #     color = 'r'
        else:
            ctx = None
        #else:
        #    raise Exception('Unable to determine context onset and offset')
        if ctx is not None:
            axis.axvspan(ctx[0][0], ctx[0][1], facecolor=color, alpha=0.3)
        for lick in behavior_data['licking']:
            axis.axvline(lick[0], color='b', linewidth=1)
        for step in behavior_data['treadmillTimes']:
            axis.axvline(step[0], color='k')

    def responseIntegrals(self, stimuli, postStimDuration=1.5, duration=None,
                          power=None, excludeRunning=False, demixed=False,
                          dFOverF=None, sharedBaseline=False,
                          linearTransform=None, channel='Ch2', label=None,
                          roi_filter=None):
        ROIs = ExperimentGroup([self]).sharedROIs(channel=channel, label=label,
                                                  roi_filter=roi_filter)
        shared_filter = lambda x: x.id in ROIs
        rIntegrals = []

        timeSeries = self.imagingData(
            demixed=demixed, dFOverF=dFOverF,
            linearTransform=linearTransform, channel=channel, label=label,
            roi_filter=shared_filter)
        baselineIndices = set(range(self.imagingIndex(self.contextInterval()[0] - 1)))
        for runInt in self.runningIntervals()[0]:
            baselineIndices -= set(range(runInt[0], runInt[1] + 1))
        baseline = timeSeries[:, list(baselineIndices), :].mean(axis=2).mean(axis=1)
        for i in range(timeSeries.shape[0]):
            timeSeries[i, :, :] -= baseline[i]
            timeSeries[i, :, :] /= baseline[i]
        if stimuli == 'air':
            try:
                puffs = self.find('trial').behaviorData()['airpuff']
            except KeyError:
                puffs = self.find('trial').behaviorData()['air']
            for puff in puffs:
                stimIdx = self.imagingIndex(puff[0])
                maxIdx = stimIdx + self.imagingIndex(postStimDuration)
                rIntegrals.append(timeSeries[:, stimIdx:maxIdx, :].sum(axis=1) * self.frame_period())
        elif stimuli == 'context':
            stimIdx = self.imagingIndex(self.contextInterval()[0])
            maxIdx = stimIdx + self.imagingIndex(postStimDuration)
            rIntegrals.append(timeSeries[:, stimIdx:maxIdx, :].sum(axis=1) * self.frame_period())
        if rIntegrals:
            return np.concatenate(rIntegrals, axis=1)
        else:
            return np.empty([len(ROIs), 0])


class toneTraceConditioningExperiment(Experiment):

    def periodStartTimes(self, period='all', mask=False, channel='Ch2',
                         label=None, roi_filter=None):
        periods = ['Baseline', 'CS', 'Trace', 'CS and Trace',
                   'US', 'Post US', 'all']
        intervals = [[0, 10], [10, 30], [30, 45],
                     [10, 45], [45, 50], [50, 60], [0, 60]]
        experimentPeriods = defaultdict()
        for p, interval in it.izip(periods, intervals):
            experimentPeriods[p] = interval
        if not mask:
            return experimentPeriods[period]
        else:
            (nROIs, nFrames, nTrials) = self.imaging_shape(
                roi_filter=roi_filter, channel=channel,
                label=label)
            calc_intervals = np.zeros((nROIs, nFrames, nTrials), 'bool')
            if period == 'all':
                calc_intervals = None
            elif period == 'Baseline':
                calc_intervals[:, 0:self.imagingIndex(10)-1,:] = True
            elif period == 'CS':
                calc_intervals[:, self.imagingIndex(10):self.imagingIndex(30)-1,:] = True
            elif period == 'Trace':
                calc_intervals[:, self.imagingIndex(30):self.imagingIndex(45)-1,:] = True
            elif period == 'US':
                calc_intervals[:, self.imagingIndex(45):self.imagingIndex(50)-1,:] = True
            elif period == 'Post US':
                calc_intervals[:, self.imagingIndex(50):self.imagingIndex(60),:] = True
            elif period == 'CS and Trace':
                calc_intervals[:, self.imagingIndex(10):self.imagingIndex(45)-1,:] = True
            return calc_intervals

    def lickCountInContext(self, duration=None):
        """Return total licks during the context.

        Tone Trace Conditioning Experiments should have only one trial.
        """
        context = self.contextInterval(mask=False)
        if duration is None:
            endTime = context[1]
        else:
            endTime = context[0] + duration
        licks = ba.lickCount(self.find('trial'), context[0], endTime)
        return licks

    def lickRateInContext(self):
        """
        Return licks per second during the context.

        FearConditioningExperiements should have only one trial
        """
        context = self.contextInterval(mask=False)
        licks = ba.lickCount(self.find('trial'), context[0], context[1])
        return licks / float((context[1] - context[0]))

    def lickRatePreContext(self):
        """Returns licks per second before the context.
            FearConditioningExperiments should have only one trial
        """
        context = self.contextInterval(mask=False)
        licks = ba.lickCount(self.find('trial'), 0, context[0])
        return licks / float((context[0] - 0))

    def lickRateTracePeriod(self):
        """Returns licks per second during the trace period.
            FearConditioningExperiments should have only one trial
        """
        context = self.contextInterval(mask=False)
        licks = ba.lickCount(self.find('trial'), context[1], context[1] + 15)
        return licks / float((15))

    def meanVelocityInContext(self, belt_length=200):
        """ Returns average velocity during the context. """

        try:
            belt_length = self.belt().length(units='cm')
        except exc.NoBeltInfo:
            warnings.warn('No belt information found for experiment %s.  \nUsing default belt length = %f' % (str(self), belt_length))

        context = self.contextInterval(mask=False)
        velocity = ba.velocity(
            self.find('trial'), imageSync=False, belt_length=belt_length, smoothing='hanning',
            window_length=71)
        samplingInterval = self.find('trial').behavior_sampling_interval()
        return np.mean(velocity[context[0] / samplingInterval:
                                context[1] / samplingInterval + 1])

    def isPuffed(self):
        """
        Returns whether or not this experiment was puffed.
        First checks for a 'autoPuff' attribute, then attempts to load behavior data
        """

        if 'autoPuff' in self.keys():
            return (self.attrib['autoPuff'] == 'yes')
        elif 'airpuffAuto' in self.keys():
            return (self.attrib['airpuffAuto'] == 'yes')
        else:
            return (self[0].behaviorData()['airpuff'].shape[0] > 0)

    def contextInterval(self, mask=False):
        """
        Returns the bounds of the context, first by checking for attributes, and if missing
        then checks the 'odor' field in the behavior data
        If mask=False, returns the context start and stop times.
        If mask=True, returns a boolean array for indexing
        """

        bd = self.find('trial').behaviorData()

        # Determine which field to use
        field = None
        if 'odor' in bd.keys() and len(bd['odor']) is 1 and len(bd['odor'][0]) is 2 and \
            not np.isnan(bd['odor'][0][0]) and not np.isnan(bd['odor'][0][1]):
            field = 'odor'
        # if 'odorA' looks good use it, else check 'odorB'
        elif 'odorA' in bd.keys() and len(bd['odorA']) is 1 and len(bd['odorA'][0]) is 2 and \
            not np.isnan(bd['odorA'][0][0]) and not np.isnan(bd['odorA'][0][1]):
            field = 'odorA'
        elif 'odorB' in bd.keys() and len(bd['odorB']) is 1 and len(bd['odorB'][0]) is 2 and \
            not np.isnan(bd['odorB'][0][0]) and not np.isnan(bd['odorB'][0][1]):
            field = 'odorB'

        if field is not None:
            interval = [ int(bd[field][0][0]), int(bd[field][0][1]) ]
        elif 'environmentOnsetDelay' in self.keys() and 'environmentDuration' in self.keys():
            interval =  [ float(self.attrib['environmentOnsetDelay']), \
                      float(self.attrib['environmentOnsetDelay']) + float(self.attrib['environmentDuration']) ]
        else:
            raise Exception('Unable to determine context onset and offset: ' + self.get('startTime'))

        if not mask:
            return interval
        else:
            boolMask = np.zeros(self.imaging_shape()[1], 'bool')
            boolMask[interval[0]:interval[1]] = True
            return boolMask

    def fullPlot(self, ax, style='color', yOffsets=None, linearTransform=None,
                 channel='Ch2', label=None, roi_filter=None):
        trial = self.find('trial')
        ROIs = ExperimentGroup([self]).sharedROIs(channel=channel, label=label,
                                                  roi_filter=roi_filter)
        shared_filter = lambda x: x.id in ROIs
        ap.activityPlot(trial, ax, style=style, yOffsets=yOffsets,
                           linearTransform=linearTransform, channel=channel,
                           label=label, roi_filter=shared_filter)
        ymin = ax.get_ylim()[0]
        ax.set_ylim(bottom=ymin - 2)
        behaviorData = trial.behaviorData()
        odorA = filter(lambda k: k in behaviorData.keys(), ['odorA', 'odor A', 'odor'])[0]
        odorB = filter(lambda k: k in behaviorData.keys(), ['odorB', 'odor B', 'odor'])[0]
        if behaviorData[odorA].size:
            ctx = behaviorData[odorA]
        elif behaviorData[odorB].size:
            ctx = behaviorData[odorB]
        else:
            ctx = None
        if self.get('environment') == 'A':
            ctxColor = 'c'
        else:
            ctxColor = 'y'
        if ctx is not None:
            ax.add_patch(
                Rectangle((ctx[0][0], ymin - 1), ctx[0][1] - ctx[0][0], 1,
                          facecolor=ctxColor, lw=0, alpha=0.5))

    def lickPlot(self, axis):
        colors = {'A': 'g', 'B': 'r'}
        behavior_data = self.find('trial').behaviorData()
        if any('odor' in key for key in behavior_data.keys()):
            ctx_keys = [key for key in behavior_data.keys() if 'odor' in key]
            for ctx_key in ctx_keys:
                if behavior_data[ctx_key].size:
                    ctx = behavior_data[ctx_key]
                    if 'A' in ctx_key:
                        color = colors['A']
                    elif 'B' in ctx_key:
                        color = colors['B']
                else:
                    ctx = None
        # if behavior_data['odor A'].size:
        #     ctx = behavior_data['odor A']
        #     color = 'g'
        # elif behavior_data['odorB'].size:
        #     ctx = behavior_data['odor B']
        #     color = 'r'
        else:
            ctx = None
        #else:
        #    raise Exception('Unable to determine context onset and offset')
        if ctx is not None:
            axis.axvspan(ctx[0][0], ctx[0][1], facecolor=color, alpha=0.3)

        for lick in behavior_data['licking']:
            axis.axvline(lick[0], color='b', linewidth=1)
        # for step in behavior_data['treadmillTimes']:
        #     axis.axvline(step[0], color='k')

    def responseIntegrals(self, stimuli, postStimDuration=1.5, duration=None,
                          power=None, excludeRunning=False, demixed=False,
                          dFOverF=None, sharedBaseline=False,
                          linearTransform=None, channel='Ch2', label=None,
                          roi_filter=None):
        ROIs = ExperimentGroup([self]).sharedROIs(channel=channel, label=label,
                                                  roi_filter=roi_filter)
        shared_filter = lambda x: x.id in ROIs
        rIntegrals = []

        timeSeries = self.imagingData(
            demixed=demixed, dFOverF=dFOverF,
            linearTransform=linearTransform, channel=channel, label=label,
            roi_filter=shared_filter)
        baselineIndices = set(range(self.imagingIndex(self.contextInterval()[0] - 1)))
        for runInt in self.runningIntervals()[0]:
            baselineIndices -= set(range(runInt[0], runInt[1] + 1))
        baseline = timeSeries[:, list(baselineIndices), :].mean(axis=2).mean(axis=1)
        for i in range(timeSeries.shape[0]):
            timeSeries[i, :, :] -= baseline[i]
            timeSeries[i, :, :] /= baseline[i]
        if stimuli == 'air':
            try:
                puffs = self.find('trial').behaviorData()['airpuff']
            except KeyError:
                puffs = self.find('trial').behaviorData()['air']
            for puff in puffs:
                stimIdx = self.imagingIndex(puff[0])
                maxIdx = stimIdx + self.imagingIndex(postStimDuration)
                rIntegrals.append(timeSeries[:, stimIdx:maxIdx, :].sum(axis=1) * self.frame_period())
        elif stimuli == 'context':
            stimIdx = self.imagingIndex(self.contextInterval()[0])
            maxIdx = stimIdx + self.imagingIndex(postStimDuration)
            rIntegrals.append(timeSeries[:, stimIdx:maxIdx, :].sum(axis=1) * self.frame_period())
        if rIntegrals:
            return np.concatenate(rIntegrals, axis=1)
        else:
            return np.empty([len(ROIs), 0])


class RunTrainingExperiment(FearConditioningExperiment):
    pass


class HiddenRewardExperiment(RunTrainingExperiment):

    def licktogram(
            self, ax=None, nPositionBins=100, rewardPositions=None,
            normed=True, plot_belt=True, shade_reward=False, shade_color='0.7',
            **plot_kwargs):
        """Returns the percentage of all licks occurring in each position bin.
        Calculated across trials
        Accepts normalized reward positions only
        If axis is passed in, a histogram is plotted
        """

        if rewardPositions is None:
            rewards = self.rewardPositions(units='normalized')
        else:
            rewards = rewardPositions

        nLicks_by_position = np.zeros(int(nPositionBins))
        for trial in self.findall('trial'):
            bd = trial.behaviorData(imageSync=False)
            position = ba.absolutePosition(
                trial, imageSync=False, sampling_interval='actual')

            licking = bd['licking'][:, 0]
            licking = licking[np.isfinite(licking)]
            licking = licking / bd['samplingInterval']
            licking = licking.astype('int')

            licking_positions = position[licking] % 1

            nLicks_by_position += np.histogram(
                licking_positions, bins=nPositionBins, range=(0, 1))[0]

        binEdges = np.arange(0, 1, 1. / nPositionBins)

        if normed:
            licks = nLicks_by_position.astype(float) \
                / np.sum(nLicks_by_position)
        else:
            licks = nLicks_by_position

        if ax:
            for p in rewards:
                if shade_reward:
                    window = self.reward_parameters(
                        distance_units='normalized').get('window_length')
                    reward_regions = [
                        (reward, reward + window) for reward in rewards]
                    for reward_window in reward_regions:
                        ax.fill_between(
                            reward_window, 0, 1, color=shade_color)
                else:
                    ax.axvline(x=p, linewidth=1, color='k', linestyle='--')

            ax.bar(binEdges, licks, width=1. / nPositionBins, **plot_kwargs)

            ax.set_xlabel('Normalized position')
            ax.set_ylabel('Percent of total licks')
            ax.set_title('Licking by position')

            if plot_belt:
                self.belt().addToAxis(ax)

        return (licks, binEdges)

    def polar_lick_plot(self, ax):

        trial = self.find('trial')

        bd = trial.behaviorData(imageSync=False)
        position = bd['treadmillPosition']
        position[:, 1] *= 2 * np.pi
        # pos = position[:, 1] * 2 * np.pi

        r_range = np.interp(
            position[:, 0], [0, bd['recordingDuration']], [0.2, 1])

        ax.plot(position[:, 1], r_range, 'k-')

        licking = bd['licking'][:, 0]

        theta_lick = np.interp(licking, position[:, 0], position[:, 1])
        r_lick = np.interp(licking, [0, bd['recordingDuration']], [0.2, 1])

        ax.plot(theta_lick, r_lick, 'bo', markersize=3)

        reward_positions = self.rewardPositions(units='normalized')
        spatial_window = \
            self.reward_parameters(
                distance_units='normalized')['window_length']

        all_pos = np.linspace(0, 2 * np.pi, 100)
        for reward in reward_positions:
            if reward + spatial_window >= 1:
                raise NotImplementedError
            ax.fill_between(
                all_pos, 1, where=np.logical_and(
                    all_pos >= reward * 2 * np.pi,
                    all_pos <= (reward + spatial_window) * 2 * np.pi),
                color='r', alpha=0.5)

        ax.set_xlabel('Position')
        ax.set_rmax(1.)

        x_ticks = ax.get_xticks()
        x_tick_labels = [str(x / 2 / np.pi) for x in x_ticks]
        ax.set_xticklabels(x_tick_labels)

        ax.tick_params(
            axis='y', bottom=False, top=False, left=False, right=False,
            labelbottom=False, labeltop=False, labelleft=False,
            labelright=False)
