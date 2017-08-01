"""Losonczy Lab class structures for analysis."""

import warnings
from xml.etree import ElementTree
import scipy.signal
import numpy as np
from os.path import join, normpath
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
import json
import itertools as it
import cPickle as pickle
from copy import copy, deepcopy
import pandas as pd
from string import ascii_uppercase
from collections import defaultdict

from .. import misc
import exceptions as exc
from ..plotting import plotting_helpers as ph
from ..analysis import imaging_analysis as ia
from ..analysis import behavior_analysis as ba
from ..misc import norm_to_complex

import sima
import sima.misc
import sima.imaging_parameters
import sima.segment

global hits
hits = 0


class Mouse(ElementTree.Element):

    # Make them sortable by mouseID
    def __lt__(self, other):
        return self.get('mouseID') < other.get('mouseID')

    #
    # Collect information about the experiments within the mouse
    #

    def hasInjection(self, virus='', location=''):
        for inj in self.findall('injection'):
            for vir in inj.findall('virus'):
                if location.lower() in inj.get('location', '').lower() and \
                        virus.lower() in vir.get('name', '').lower():
                    return True
        return False

    def locationsImaged(self):
        """Returns a list of unique imaging locations with signals files"""
        locs = []
        for exp in self.findall('experiment'):
            if exp.hasSignalsFile():
                locs.append(exp.get('uniqueLocationKey'))
        return np.unique(locs).tolist()

    #
    # Return subsets of experiments
    #

    def exptsAtLocation(self, location):
        """Returns a list of experiments with signals files from 'loc'"""
        expts = []
        for exp in self.findall('experiment'):
            if exp.hasSignalsFile() and \
                    exp.get('uniqueLocationKey') == location:
                expts.append(exp)

        return expts

    def exptsAtLayer(self, layer):
        """Returns a list of experiments with signals files from layer"""
        expts = []
        for exp in self.findall('experiment'):
            if exp.hasSignalsFile() and \
                    exp.get('imagingLayer') == layer:
                expts.append(exp)

        return expts

    def imagingExperiments(self, channels=['Ch1', 'Ch2']):
        """Returns a list of the experiments with non-empty signals files for
        any of the above channels."""
        expts = []
        for expt in self.findall('experiment'):
            for channel in channels:
                if expt.hasSignalsFile(channel=channel) \
                        and not expt.get('ignoreImagingData'):
                    expts.append(expt)
                    break
        return expts

    #
    # Print more useful string representations of Mouse objects
    #

    def __repr__(self):
        s = "<Mouse: "
        for key, value in self.attrib.iteritems():
            s = s + key + ' = ' + str(value) + ", "
        s = s[:-1] + '>'
        return s

    def __str__(self):
        return "<Mouse: {}, genotype={}, nExpts={}>".format(
            self.get('mouseID', ''), self.get('genotype', ''),
            len(self.findall('experiment')))


class Belt(ElementTree.Element):
    """Items of the belts.xml are cast as Belt objects when called by the
    Experiment class

    """

    backwards = False

    def __str__(self):
        return '<Belt: {}, length: {} cm, direction: {}>'.format(
            self.get('beltID'), self.length(),
            'backwards' if self.backwards else 'forwards')

    def __lt__(self, other):
        return self.get('beltID') < other.get('beltID')

    def __eq__(self, other):
        return self.get('beltID') == other.get('beltID')

    def length(self, units='cm'):
        """Returns length of the belt in cm."""
        if not hasattr(self, '_length'):
            self._length = np.sum(
                [float(seg.get('length_cm')) for seg in self.findall('segment')])
        if units == 'cm':
            return self._length
        if units == 'mm':
            return self._length * 10.
        raise ValueError('Unrecognized units.')

    def zeroPosition(self):
        """Returns the relative position of the 'lap start' RFIDtag along
        the belt (e.g. 0.5 indicates middle of belt)

        """

        nSegments = len(self.findall('segment'))
        pos = 0.
        for seg in range(nSegments):
            for segment in self.findall('segment'):
                if segment.get('ordering') == str(seg + 1):
                    if len(segment.findall('RFIDtag')):
                        lapStart = None
                        for tag in segment.findall('RFIDtag'):
                            if tag.get('lapStart') == 'True':
                                lapStart = tag
                        if lapStart is not None:
                            pos += float(
                                lapStart.get('distanceFromSegmentStart'))
                            if self.backwards:
                                return 1 - (pos / self.length())
                            return pos / self.length()
                        else:
                            pos += float(segment.get('length_cm'))

                    else:
                        pos += float(segment.get('length_cm'))

    def materials(self):
        """List of materials used in the belt."""
        return np.unique(
            [seg.get('material') for seg in self.findall('segment')]).tolist()

    def show(self, ax=None, aspect='auto', zeroOnLeft=False):
        """Display the belt on an axis, preserving aspect ratio.
        If zeroOnLeft, the RFID start tag is displayed on the left

        """

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        imPath = self.get('imageLocation')
        im = Image.open(imPath)
        imData = np.asarray(im)

        if self.backwards:
            imData = np.fliplr(imData)

        if zeroOnLeft:
            # swapping two blocks of columns of imData
            new_im = np.empty(imData.shape)
            zeroIdx = int(self.zeroPosition() * imData.shape[1])

            new_im[:, :imData.shape[1] - zeroIdx, :] = imData[:, zeroIdx:, :]
            new_im[:, imData.shape[1] - zeroIdx:, :] = imData[:, :zeroIdx, :]

            imData = new_im

        ax.imshow(imData.astype('uint8'), aspect=aspect)
        ax.set_axis_off()

    def addToAxis(self, ax, plot_ratio=(8, 1)):
        """Given an axis, use make_axes_locatable to place the belt underneath
        the axis.

        """

        divider = make_axes_locatable(ax)
        belt_ax = divider.append_axes("bottom", size="5%", pad=0.05)

        self.show(belt_ax, zeroOnLeft=True)

    def animation(self, ax, position):
        """Creates an animation of position scrolling along the belt, denoted
        by a moving red vertical line. The animation will be len(position)
        frames long.

        position -- Normalized location of the mouse at each time point.
            Should be a 1D array with all values on [0, 1)

        """

        self.show(ax, zeroOnLeft=True)

        x_min, x_max = ax.get_xlim()
        x_position = np.array(position) * (x_max - x_min) + x_min

        for pos in x_position:
            ax.lines = []
            ax.axvline(pos, color='r')
            yield

    def cues(self, normalized=False):
        """Return cues along with absolute position ranges"""
        segs = self.findall('segment')

        zero_position = self.zeroPosition() * self.length()

        seq_start = 0.
        cue_list = []
        for seq in sorted(segs, key=lambda x: x.get('ordering')):
            cues = seq.findall('cue')
            for cue in sorted(
                    cues, key=lambda x: x.get('distanceFromSegmentStart')):
                cue_start = seq_start + \
                    float(cue.get('distanceFromSegmentStart')) - zero_position
                cue_stop = cue_start + float(cue.get('length'))
                cue_start %= self.length()
                cue_stop %= self.length()
                cue_list.append(
                    {'cue': cue.get('type'), 'start': cue_start,
                     'stop': cue_stop})
            seq_start += float(seq.get('length_cm'))

        result = pd.DataFrame(
            cue_list, columns=['cue', 'start', 'stop']).sort_values(
            by='start').reset_index(drop=True)

        if normalized:
            result['start'] /= self.length()
            result['stop'] /= self.length()

        return result

    def fabric_transitions(self, units='cm'):
        """Return position of fabric transitions."""
        segs = self.findall('segment')

        zero_position = self.zeroPosition() * self.length()

        first_transition_position = self.length() - zero_position if zero_position > 0 else np.abs(zero_position)

        if len(segs) == 0:
            raise ValueError(
                'Belt information incomplete: {}'.format(self.get('beltID')))
        elif len(segs) == 1:
            transitions = [{'first_material': segs[0].get('material'),
                            'second_material': segs[0].get('material'),
                            'position': first_transition_position}]
        else:
            transitions = []
            sorted_segs = sorted(segs, key=lambda x: x.get('ordering'))
            total_pos = -zero_position
            for seg1, seg2 in zip(sorted_segs[:-1], sorted_segs[1:]):
                total_pos += float(seg1.get('length_cm'))
                transitions.append(
                    {'first_material': seg1.get('material'),
                     'second_material': seg2.get('material'),
                     'position': total_pos})
            transitions.append(
                {'first_material': sorted_segs[-1].get('material'),
                 'second_material': sorted_segs[0].get('material'),
                 'position': first_transition_position})

        result = pd.DataFrame(
            transitions,
            columns=['first_material', 'second_material', 'position'])

        if units == 'normalized':
            result['position'] /= self.length()
        elif units == 'complex':
            result['position'] = norm_to_complex(
                result['position'] / self.length())
        elif units != 'cm':
            raise ValueError('Unrecognized units: {}'.format(units))

        return result

    def transform(self, other, method='cues', n_bins=100):
        if method != 'cues':
            raise NotImplementedError

        src = self.cues()
        trg = other.cues()

        assert(all(src['cue'].sort(inplace=False) ==
                   trg['cue'].sort(inplace=False)))

        src[['start', 'stop']] /= self.length()
        trg[['start', 'stop']] /= other.length()

        final_order = np.empty(n_bins)
        final_order.fill(np.nan)

        for idx, row in trg.iterrows():
            src_idx = src[src['cue'] == row['cue']].index.tolist()
            assert len(src_idx) == 1
            src_idx = src_idx[0]
            start = int(src.loc[src_idx]['start'] * n_bins)
            stop = int(src.loc[src_idx]['stop'] * n_bins)
            if stop > start:
                src_range = np.arange(start, stop)
            else:
                src_range = np.hstack(
                    [np.arange(start, n_bins), np.arange(0, stop)])
            # TODO: should this be n_bins?
            trg_start = int(row['start'] * 100)
            if trg_start + len(src_range) >= n_bins:
                to_end = n_bins - trg_start
                final_order[trg_start:] = src_range[:to_end]
                final_order[:len(src_range) - to_end] = src_range[to_end:]
            else:
                final_order[trg_start:trg_start + len(src_range)] = src_range

        available_bins = set(np.arange(n_bins)).difference(final_order)

        def fill_bins(arr, vals):
            return_val = False
            diffs = np.diff(np.isnan(np.hstack([arr[-1], arr])).astype(int))
            for next_bin in np.where(diffs != 0)[0]:
                if diffs[next_bin] < 0:
                    b = next_bin - 1
                    # Check if already filled
                    if not np.isnan(arr[b]):
                        continue
                    next_val = arr[b + 1] - 1
                    if next_val == -1:
                        next_val = len(arr)
                    if next_val in vals:
                        arr[b] = next_val
                        vals.remove(next_val)
                        return_val = True
                    else:
                        # Can't expand in this direction anymore
                        pass
                elif diffs[next_bin] > 0:
                    b = next_bin
                    # Check if already filled
                    if not np.isnan(arr[b]):
                        continue
                    next_val = arr[b - 1] + 1
                    if next_val == len(arr):
                        next_val = 0
                    if next_val in vals:
                        arr[b] = next_val
                        vals.remove(next_val)
                        return_val = True
                    else:
                        # Can't expand in this direction anymore
                        pass
            return return_val

        while fill_bins(final_order, available_bins):
            pass

        return final_order[np.isfinite(final_order)]
    # END Belt class


class Trial(ElementTree.Element):

    def __lt__(self, other):
        return self.startTime() < other.startTime()

    def __eq__(self, other):
        try:
            self_expt = self.parent
            other_expt = other.parent
            self_time = self.get('time', np.nan)
            other_time = other.get('time', np.nan)
        except AttributeError:
            return False
        return isinstance(other, Trial) and (self_expt == other_expt) \
            and (self_time == other_time)

    def startTime(self):
        return misc.parseTime(self.get('time'))

    def image_sync_behavior_length(self):
        """Returns the number of frames of image sync'd behaviorData
        Can be less than number of imaging frames
        """
        if not hasattr(self, '_image_sync_behavior_length'):
            self._image_sync_behavior_length = len(
                self.behaviorData(imageSync=True)['treadmillTimes'])
        return self._image_sync_behavior_length

    def behaviorDataPath(self):
        """Returns the path to the behaviorData pkl file"""
        if 'filename' not in self.keys():
            raise exc.MissingBehaviorData(
                'Missing filename field, no behavior data recorded')
        return normpath(join(self.parent.parent.parent.behaviorDataPath,
                             self.get('filename').replace('.csv', '.pkl')))

    # TODO: Remove this
    def transientsData(self, **kwargs):
        """Load the transients data and return the corresponding transients
        structure for 95 or 99% confidence

        See Expt.transientsData() for details

        """
        trans = self.parent.transientsData(**kwargs)
        return trans[:, self.trialNum()]

    def behavior_sampling_interval(self):
        """Shortcut method to just return the behavior data sampling_interval.
        Saves it as well, so it's only loaded once."""
        if not hasattr(self, '_behavior_sampling_interval'):
            bd = pickle.load(open(self.behaviorDataPath(), 'r'))
            try:
                self._behavior_sampling_interval = float(bd['samplingInterval'])
            except KeyError:
                self._behavior_sampling_interval = 0.01

        return self._behavior_sampling_interval

    def _resample_position(self, positions, sampling_interval=None):
        rate = np.min(np.diff(positions[:, 0]))
        gaps = np.where(np.diff(positions[:,0]) > rate*2)[0]
        for gap in gaps:
            positions = np.insert(positions, gap+1, [positions[gap][0]+rate,
                positions[gap][1]], axis=0)
            gaps += 1

        lap_times = np.where(np.diff(positions[:,1]) < -0.5)[0]
        for ti in lap_times:
            positions[ti+1:, 1] += 1.0

        lap_times = np.where(np.diff(positions[:,1]) > 0.5)[0]
        for ti in lap_times:
            positions[ti+1:, 1] -= 1.0

        if sampling_interval is not None:
            rate = sampling_interval

        position_func = scipy.interpolate.interp1d(
            positions[:, 0], positions[:, 1])
        new_times = np.arange(0, max(positions[:, 0]), rate)
        new_positions = position_func(new_times) % 1.0

        return np.vstack(([new_times], [new_positions])).T, rate

    def behaviorData(self, imageSync=False, sampling_interval=None, discard_initial=False,
                     use_rebinning=False):
        """Return a dictionary containing the the behavioral data.

        Parameters
        ----------
        imageSync : bool
             If False, the structure will represent the sparse times at
            which the stimuli/behavioral variables changed.
            If True, the structure will contain a boolean array corresponding
            to the stimulus intervals, with timepoints separated by the
            'framePeriod' up to the length of the imaging data.
            For 'treadmillTimes' the structure will contain the number of beam
            breaks within each sampling interval.
            For 'lapCounter' the times will of each marker will be converted to
            frame numbers for imageSync.
            The default 'lapCounter' format is an Nx2 array of times for each
            marker, the first column being the time and the second column being
            the marker number with '1' being the lap start marker.
        sampling_interval : {None, 'actual', float}
             The sampling interval (in seconds) of the output
            data structure or 'actual' to use the sampling interval that the
            data was recorded at or None to return sparse intervals.
        discard_initial : bool
            Discards data from first partial lap and starts at first 0 position
            if True
        Note
        ----
        imageSync=True is mostly the same as
        sampling_interval=self.parent.frame_period(),
        though additionally all arrays are trimmed down to the length of the
        imaging data

        If sampling_interval is not None or imageSync = True, data is converted
        from time (in seconds) to frame (in units of sampling_interval).
        For this conversion, frame n = [n, n+1) so the times of the final
        output arrays is [0, recording_duration)

        """

        # make sure the behavior data is there and load it
        if not hasattr(self, '_behavior_data'):
            try:
                self._behavior_data = {'original': pickle.load(
                    open(self.behaviorDataPath(), 'rb'))}
            except:
                raise exc.MissingBehaviorData('Unable to find behavior data')

        if ((imageSync, sampling_interval) in self._behavior_data) and (not use_rebinning):
            return deepcopy(self._behavior_data[imageSync, sampling_interval])

        dataDict = deepcopy(self._behavior_data['original'])

        # All of these conversions are unnecessary with pickled behaviorData
        # TODO: remove them all?
        try:
            assert discard_initial is True
            d = self._behavior_data['original']
            first_lap_start = d['lapCounter'][d['lapCounter'][:,1] == 1][0,0] / 100.
            dataDict['recordingDuration'] = float(dataDict['recordingDuration']) - first_lap_start
            dataDict['samplingInterval'] = float(dataDict['samplingInterval'])
            dataDict['trackLength'] = float(dataDict['trackLength'])
            dataDict['lapCounter'] = dataDict['lapCounter'][dataDict['lapCounter'][:,0] >= first_lap_start * 100]
            dataDict['licking'] = dataDict['licking'][dataDict['licking'][:, 0] > first_lap_start]
            dataDict['water'] = dataDict['water'][dataDict['water'][:,0] > first_lap_start]
            dataDict['laser'] = dataDict['laser'][dataDict['laser'][:,0] > first_lap_start]
            dataDict['reward'] = dataDict['reward'][dataDict['reward'][:,0] > first_lap_start]
            dataDict['light'] = dataDict['light'][dataDict['light'][:,0] > first_lap_start]
            dataDict['tone'] = dataDict['tone'][dataDict['tone'][:,0] > first_lap_start]
            dataDict['shock'] = dataDict['shock'][dataDict['shock'][:,0] > first_lap_start]
            dataDict['airpuff'] = dataDict['airpuff'][dataDict['airpuff'][:,0] > first_lap_start]
            dataDict['odorA'] = dataDict['odorA'][dataDict['odorA'][:,0] > first_lap_start]
            dataDict['odorB'] = dataDict['odorB'][dataDict['odorB'][:,0] > first_lap_start]
            try:
                dataDict['treadmillPosition'] = dataDict['treadmillPosition'][dataDict['treadmillPosition'][:,0] > first_lap_start]
            except KeyError:
                print """{} doesn't have treadmillPosition??""".format(self.behaviorDataPath())
        except:
            if discard_initial is True:
                print """Bad stuff happened with {}""".format(self.behaviorDataPath())
            try:
                dataDict['recordingDuration'] = \
                    float(dataDict['recordingDuration'])
            except:
                pass
            try:
                dataDict['samplingInterval'] = float(dataDict['samplingInterval'])
            except:
                pass
            try:
                dataDict['trackLength'] = float(dataDict['trackLength'])
            except:
                pass
            # This shouldn't be necessary, what happened that changed this?
            try:
                dataDict['treadmillTimes'] = dataDict['treadmillTimes'].reshape(-1)
            except:
                pass
            try:
                dataDict['treadmillTimes2'] = \
                    dataDict['treadmillTimes2'].reshape(-1)
            except:
                pass

        orig_samp_int = sampling_interval
        if imageSync:
            sampling_interval = self.parent.frame_period()
            nFrames = self.parent.num_frames()

        if 'samplingInterval' not in dataDict:
            if type(sampling_interval) == type(1.0) or \
                    type(sampling_interval) == type(np.float64(1.0)):
                dataDict['samplingInterval'] = sampling_interval
            else:
                dataDict["samplingInterval"] = 0.01

        if sampling_interval == 'actual':
            sampling_interval = dataDict['samplingInterval']

        # If we don't want the data resampled, just return as intervals
        if sampling_interval is None:
            self._behavior_data[(imageSync, sampling_interval)] = dataDict
            return deepcopy(dataDict)

        output_interval = sampling_interval
        uprateFactor = None
        if(use_rebinning):
            uprateFactor = int((1 / dataDict["samplingInterval"]) * 5 / (1 / sampling_interval));
            upRate = uprateFactor * (1 / sampling_interval)
            sampling_interval = 1.0 / upRate

        recordingDuration = dataDict['recordingDuration']

        # Changed 4/6 by Jeff, will now include frames with full behavior data
        # numberBehaviorFrames = \
        #     int(np.ceil(recordingDuration / sampling_interval))
        numberBehaviorFrames = int(recordingDuration / sampling_interval)
        # Return data as boolean array, matched to imaging data
        for stim in dataDict:
            if not stim.startswith('__') and stim not in [
                    'treadmillTimes', 'treadmillTimes2', 'lapCounter',
                    'treadmillPosition', 'samplingInterval',
                    'recordingDuration', 'trackLength', 'lap']:
                out = np.zeros(numberBehaviorFrames, 'bool')
                for start, stop in dataDict[stim]:
                    if np.isnan(start):
                        start = 0
                    if np.isnan(stop):
                        stop = recordingDuration
                    start_frame = int(start / sampling_interval)
                    # Changed 4/6 by Jeff, this is more correct
                    stop_frame = int(stop / sampling_interval) + 1
                    # stop_frame = int(np.ceil(stop / sampling_interval))
                    out[start_frame:stop_frame] = True

                if(use_rebinning):
                    numFrames = int(numberBehaviorFrames / uprateFactor)
                    out = out[:(numFrames * uprateFactor)]
                    out = np.sum(np.reshape(out, (numFrames, uprateFactor)), axis=1) / float(uprateFactor)

                if imageSync and len(out) > nFrames:
                    out = out[:nFrames]
                dataDict[stim] = out
        # treadmillTimes will be the number of beam breaks in each interval
        for treadmill_times in ['treadmillTimes', 'treadmillTimes2']:
            if treadmill_times in dataDict:
                out = np.zeros(numberBehaviorFrames)
                for tick_time in dataDict[treadmill_times]:
                    behavior_bin = int(tick_time / sampling_interval)
                    if behavior_bin < numberBehaviorFrames:
                        out[behavior_bin] += 1
                    # # Correct for the edge case
                    # if tick_time == recordingDuration:
                    #     tick_time -= np.spacing(1)
                    # out[int(tick_time / sampling_interval)] += 1
                if(use_rebinning):
                    numFrames = int(numberBehaviorFrames / uprateFactor)
                    out = out[:(numFrames * uprateFactor)]
                    out = np.sum(np.reshape(out, (numFrames, uprateFactor)), axis=1)

                if imageSync and len(out) > nFrames:
                    out = out[:nFrames]
                dataDict[treadmill_times] = out

        sampling_interval = output_interval
        # lapCounter will just convert the real times into frame numbers
        if 'lapCounter' in dataDict and len(dataDict['lapCounter']) > 0:
            dataDict['lapCounter'][:, 0] /= sampling_interval
            dataDict['lapCounter'][:, 0] = np.floor(
                dataDict['lapCounter'][:, 0])
        # treadmillPosition will be the mean position during each frame
        if 'treadmillPosition' in dataDict:
            # This is a little complicated since, just rounding down the
            # treadmill times to the nearest bin biases the result more for
            # low sampling rates than for high ones.
            # To get around this, always calculate the full position at the
            # original sampling rate, and then downsample from there.

            out = np.empty(numberBehaviorFrames)

            original_sampling_interval = dataDict['samplingInterval']
            assert sampling_interval >= original_sampling_interval

            treadmill_position = dataDict['treadmillPosition']

            assert treadmill_position[0, 0] == 0.
            # Make sure time 0 is in the positions, so the fill will be for
            # times after the last change in position (which will be constant)
            position_interp = scipy.interpolate.interp1d(
                treadmill_position[:, 0], treadmill_position[:, 1],
                kind='zero', bounds_error=False,
                fill_value=treadmill_position[-1, 1])

            times = np.arange(0., recordingDuration,
                              original_sampling_interval)

            position = position_interp(times)

            if sampling_interval == original_sampling_interval:
                out = position[:numberBehaviorFrames]
            else:
                frames = np.arange(numberBehaviorFrames)
                start_frames = np.around(
                    frames * sampling_interval / original_sampling_interval,
                    0).astype(int)
                stop_frames = np.around(
                    (frames + 1) * sampling_interval /
                    original_sampling_interval, 0).astype(int)

                for frame, start_frame, stop_frame in it.izip(
                        it.count(), start_frames, stop_frames):

                    data = position[start_frame:stop_frame]

                    data_sorted = sorted(data)
                    if len(data_sorted) >= 2 and \
                            data_sorted[-1] - data_sorted[0] > 0.9:
                        high_vals = data[data >= 0.5]
                        low_vals = data[data < 0.5]

                        out[frame] = np.mean(
                            np.hstack((high_vals, low_vals + 1))) % 1
                    else:
                        out[frame] = np.mean(data)

            assert np.all(out < 1.)
            assert np.all(out >= 0.)

            # Trim down if behavior data runs longer
            if imageSync and len(out) > nFrames:
                out = out[:nFrames]

            dataDict['treadmillPosition'] = np.round(out, 8)

        if(not use_rebinning):
            self._behavior_data[(imageSync, orig_samp_int)] = dataDict
        return deepcopy(dataDict)

    # CAN WE REMOVE THIS?
    def trialNum(self):
        """Return the unique sequential number of this trial within
        it's experiment

        """
        return self.parent.findall('trial').index(self)

    # LEAVE THIS OR EDIT .XML'S
    def duration(self):
        try:
            return int(self.get('duration'))
        except TypeError:
            if self.get('stimulus') == 'air':
                return int(self.parent.get('airpuffDuration'))
            else:
                raise

    def __repr__(self):
        s = "<  Trial: " + self.parent.parent.get('mouseID', '')
        for key, value in self.attrib.iteritems():
            s = s + ", " + key + ' = ' + str(value)
        s = s + '>'
        return s

    def __str__(self):
        return "<  Trial: " + self.parent.parent.get('mouseID', '') + \
               ", stimulus = " + self.get('stimulus', '') + \
               ", time = " + self.get('time', '') + ">"

    # END trial class


class ExperimentGroup(object):
    """Grouping of experiments, e.g. by same location and experiment type.

    Example
    -------
    >>> from lab import ExperimentSet
    >>> expt_set = ExperimentSet(
        '/analysis/experimentSummaries/.clean_code/experiments/behavior_jeff.xml')

    >>> e1 = expt_set.grabExpt('sample_mouse', 'startTime1')
    >>> e2 = expt_set.grabExpt('sample_mouse', 'startTime2')

    >>> expt_grp = ExperimentGroup([e1, e2], label='example_group')

    Parameters
    ----------
    experiment_list : list
        A list of lab.classes.Experiment objects comprising the group.

    label : string
        A string describing the contents of the group.  For example, if you are
        comparing two ExperimentGroups (e.g. WT vs. mutant), you could label
        them as such.
    """

    def __init__(self, experiment_list, label=None, **kwargs):
        """Initialize the group."""
        super(ExperimentGroup, self).__init__(**kwargs)
        self._list = list(experiment_list)
        self._label = label

    """
    CONTAINER TYPE FUNCTIONS
    """
    def __len__(self):
        return len(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def __setitem__(self, i, v):
        self._list[i] = v

    def __delitem__(self, i):
        self._list.__delitem__(i)

    def __iter__(self):
        return self._list.__iter__()

    def __reversed__(self):
        return self._list.__reversed__()

    def __str__(self):
        return "<Experiment group: label={label}, nExpts={nExpts}>".format(
            label=self.label(), nExpts=len(self))

    def __repr__(self):
        return '{}({})'.format(repr(type(self)), repr(self._list))

    def __copy__(self):
        return type(self)(copy(self._list), self._label)

    def __deepcopy__(self):
        return type(self)(deepcopy(self._list), self._label)

    def index(self, expt):
        return self._list.index(expt)

    def remove(self, expt):
        self._list.remove(expt)

    def append(self, expt):
        self._list.append(expt)

    def extend(self, expt_grp):
        self._list.extend(expt_grp)

    def label(self, newLabel=None):
        """Return or set label for the ExperimentGroup.

        Parameters
        ----------
        newLabel : None or string, optional
            If not None, change the label to 'newLabel'. Otherwise, just return
            the label.

        Returns
        -------
        label : string
            The label of the ExperimentGroup.

        """
        if newLabel is not None:
            self._label = str(newLabel)
        return self._label

    def subGroup(self, expts, inplace=False, label=None):
        """Create a mini group based on a full ExperimentGroup.

        Parameters
        ----------
        expts : list
            A list of lab.classes.Experiment instances (each of which is
            present in the full group) from which to make a new ExperimentGroup
            instance.

        label : string, optional
            A label for the new group.  By default, the label for the original
            group will be used.

        Returns
        -------
        ExperimentGroup (preserves class of the original experimentGroup)

        """

        # if label is None:
        #     return type(self)(expts, label=self.label())
        # else:
        #     return type(self)(expts, label=label)
        return self.filter(
            lambda expt: expt in expts, inplace=inplace, label=label)

    def filter(self, function, inplace=True, label=None):
        if inplace:
            self._list = filter(function, self)
            self.label(label)
        else:
            new_grp = copy(self)
            new_grp.filter(function, inplace=True, label=label)
            return new_grp

    def filterby(
            self, filter_fn, include_columns=None, inplace=True, label=None):

        if include_columns is None:
            include_columns = []

        dataframe = self.dataframe(self, include_columns=include_columns)
        if len(dataframe) <= 0:
            warnings.warn(
                'Empty ExperimentGroup, inplace and label arguments ignored.')
            return self
        dataframe = dataframe[filter_fn(dataframe)]
        dataframe = ph.prepare_dataframe(dataframe, ['expt'])

        expts = set(dataframe['expt'])

        return self.filter(
            lambda expt: expt in expts, inplace=inplace, label=label)

    def pair(self, method='time', **kwargs):
        if method == 'time':
            return PairedExperimentGroup(self, **kwargs)
        elif method == 'consecutive groups':
            return ConsecutiveGroupsPairedExperimentGroup(self, **kwargs)
        elif method == 'same group':
            return SameGroupPairedExperimentGroup(self, **kwargs)
        else:
            raise ValueError(
                "Unrecognized 'method' argument: {}".format(method))

    def groupby(self, groupby):
        """A generator that acts similar to pd.DataFrame.groupby.
        Returns key, group pairs.

        """

        assert isinstance(groupby, list) or isinstance(groupby, tuple)
        unsqueeze_key = (len(groupby) == 1)
        df = self.dataframe(self, include_columns=groupby)

        for key, group in df.groupby(groupby):
            expts = group['expt'].tolist()
            new_label = '{}: {}'.format(self.label(), key)
            # Pandas seems to squeeze keys by default...
            if unsqueeze_key:
                key = (key, )
            new_grp = self.subGroup(expts, label=new_label)
            yield key, new_grp

    def to_json(self, path=None):
        """Dump an ExperimentGroup to a JSON representation that can be used
        to re-initialize the same experiments.

        Parameters
        ----------
        path : string, optional
            The full file path of the output .json file. If 'None', prints
            JSON.

        Notes
        -----
        The keys of the output JSON are mouseIDs and the elements are lists of
        experiment startTimes (as indicated in the .xml / SQL database).

        Returns
        -------
        None

        """

        grp_dict = defaultdict(list)
        for expt in self:
            grp_dict[expt.parent.get('mouseID')].append(expt.get('startTime'))

        for mouse, expt_list in grp_dict.iteritems():
            grp_dict[mouse] = sorted(expt_list)

        save_dict = {'experiments': grp_dict}
        if path is None:
            print json.dumps(save_dict, sort_keys=True, indent=4)
        else:
            with open(path, 'wb') as f:
                json.dump(save_dict, f, sort_keys=True, indent=4)

    @classmethod
    def from_json(cls, path, expt_set, **kwargs):
        """Initializes a new ExperimentGroup with the experiments from the JSON.

        Parameters
        ----------
        path : string
            Path to the JSON file from which to load the experiments

        expt_set : lab.classes.ExperimentSet
            An instance of the ExperimentSet class containing the experiments
            of interest.

        Returns
        -------
        ExperimentGroup

        """

        expts = json.load(open(path, 'r'))

        expt_list = []
        for mouseID, mouse_expts in expts['experiments'].iteritems():
            for startTime in mouse_expts:
                expt_list.append(expt_set.grabExpt(mouseID, startTime))

        return cls(expt_list, **kwargs)

    """
    Static methods
    """

    @staticmethod
    def dictByMouse(exptList):
        """Takes a list of experiments and returns a dictionary sorted by
        mouse

        """
        mice = set([expt.parent for expt in exptList])
        expts = {}
        for mouse in mice:
            expts[mouse] = []
        for expt in exptList:
            expts[expt.parent].append(expt)
        return expts

    @staticmethod
    def dictByExposure(exptList, **kwargs):
        """Takes a list of experiments and returns a dictionary grouped by days
        of exposure

        """
        exposure = ExperimentGroup(exptList).priorDaysOfExposure(**kwargs)
        expts = {}
        for expt in exposure:
            if exposure[expt] in expts:
                expts[exposure[expt]].append(expt)
            else:
                expts[exposure[expt]] = [expt]
        return expts

    """
    Class specific functions
    """

    def splitExptGrpByContext(self):
        new_expt_lists = []
        for expt in self:
            match_found = False
            for idx, expt_list in enumerate(new_expt_lists):
                if expt.sameContext(expt_list[0]):
                    new_expt_lists[idx].append(expt)
                    match_found = True
                    break
            if not match_found:
                new_expt_lists.append([expt])
        grps = [type(self)(expt_list) for expt_list in new_expt_lists]
        for grp in grps:
            grp.label(grp.inferLabel())
        return grps

    def inferLabel(self):
        if all([self[0].sameContext(expt) for expt in self]):
            return "Ctx{}_{}".format(
                self[0].get('environment'), self[0].get('belt'))
        return None

    def stimuli(self):
        """Return all stimuli presented."""
        stimuli = []
        for expt in self:
            try:
                stimuli.extend(expt.stimuli())
            except AttributeError:
                pass
        return list(set(stimuli))

    def genImagedExptPairs(self, ignore_conditions=False):
        """Returns pairs of experiments matched for mouse and uniqueLocationKey

        Returns experiment pairs with the oldest experiment first.
        Will not return experiment pairs with the same start time.

        """

        for (e1, e2) in it.combinations(self, 2):
            if e1.sameField(e2) and e1.is_imaged() and e2.is_imaged() and \
                    (ignore_conditions or e1.sameConditions(e2)):
                if e2 > e1:
                    yield e1, e2
                elif e1 > e2:
                    yield e2, e1

    def rois(self, channel='Ch2', label=None, roi_filter=None):
        rois = {}
        for expt in self:
            # If the expt wasn't imaged, wasn't MC'd, wasn't extracted,
            # or is missing the desired label, rois[expt] = None
            try:
                rois[expt] = expt.rois(
                    channel=channel, label=label, roi_filter=roi_filter)
            except (exc.NoTSeriesDirectory, exc.NoSignalsData, exc.NoSimaPath, KeyError):
                rois[expt] = None
        return rois

    def roi_ids(self, **kwargs):

        ids = {}
        for expt in self:
            try:
                ids[expt] = expt.roi_ids(**kwargs)
            except (exc.NoTSeriesDirectory, exc.NoSignalsData):
                ids[expt] = None
        return ids

    # TODO: LOOKS BROKEN
    def roi_tuples(self, **kwargs):

        tuples = {}
        for expt in self:
            try:
                tuples[expt] = expt.roi_tuples(**kwargs)
            except (exc.NoTSeriesDirectory, exc.NoSignalsData):
                tuples[expt] = None

        return tuples

    # TODO: Move to lab/analysis/cleanup.py
    def removeMissingBehaviorData(
            self, key=None, minimum=0, min_per_lap=0, verbose=False):
        """Remove trials that are missing behavior data.
        Additionally if 'key' is not None, remove experiments with any trials
        missing that behaviorData key.
        If 'key' is not None and minimum > 0, remove experiments with any
        trials that have fewer than 'minimum' events/intervals

        We could eventually just remove trials missing the key, but at the
        moment we don't have a system for filtering out trials of imaging data.

        """

        for expt in reversed(self):
            bad_behavior = False
            for trial in expt.findall('trial'):
                try:
                    bd = trial.behaviorData(imageSync=False)
                except exc.MissingBehaviorData:
                    if verbose:
                        print "Missing behavior data:", trial
                    expt.remove(trial)
                else:
                    if key is not None:
                        if key not in bd:
                            bad_behavior = True
                            if verbose:
                                print "Missing behavior key,", key, ":", trial
                            break
                        elif bd[key].shape[0] < minimum:
                            bad_behavior = True
                            if verbose:
                                print "Too few values ({}), {}:".format(
                                    bd[key].shape[0], key), trial
                            break
                        else:
                            n_laps = ba.absolutePosition(
                                trial, imageSync=False,
                                sampling_interval='actual').max()
                            if bd[key].shape[0] / n_laps < min_per_lap:
                                bad_behavior = True
                                if verbose:
                                    print "Rate too low ({}), {}:".format(
                                        bd[key].shape[0] / n_laps, key), trial
                                break

            if len(expt.findall('trial')) == 0:
                if verbose:
                    print "Empty expt:", expt
                self.remove(expt)
            elif bad_behavior:
                self.remove(expt)

    # TODO: Move to lab/analysis/cleanup.py
    def removeBadTickCounts(
            self, percent_off=None, minimum=None, maximum=None,
            check_reward_positions=True, verbose=False):
        for expt in reversed(self):
            if percent_off is not None:
                expected_length = expt.get('trackLength', None)
                if expected_length is None:
                    if verbose:
                        print 'Missing xml track length, skipping: ', expt
                    continue
                expected_length = int(expected_length)
            if check_reward_positions:
                try:
                    reward = max(expt.rewardPositions(units=None))
                except AttributeError:
                    reward = 0
            else:
                reward = 0
            for trial in expt.findall('trial'):
                track_length = trial.behaviorData()['trackLength']
                if track_length < reward:
                    if verbose:
                        print 'Track length off, reward at {}, actual {}: {}'.format(
                            reward, track_length, expt)
                    self.remove(expt)
                    break
                if percent_off is not None \
                        and (track_length < (1. - percent_off) * expected_length
                             or track_length > (1. + percent_off) * expected_length):
                    if verbose:
                        print 'Track length off, expected {}, actual {}: {}'.format(
                            expected_length, track_length, expt)
                    self.remove(expt)
                    break
                if minimum is not None and track_length < minimum:
                    if verbose:
                        print "Track length too short ({}): {}".format(
                            track_length, expt)
                    self.remove(expt)
                    break
                if maximum is not None and track_length > maximum:
                    if verbose:
                        print "Track length too long ({}): {}".format(
                            track_length, expt)
                    self.remove(expt)
                    break

    # TODO: Move to lab/analysis/cleanup.py
    def removeDatalessExperiments(
            self, channel='Ch2', label=None, check_trans=False, verbose=False):
        self.removeMissingBehaviorData(verbose=verbose)
        for expt in reversed(self):
            try:
                im_data = expt.imaging_dataset().signals(channel=channel)
                # expt.imagingData(channel=channel, label=label)
            except (exc.NoTSeriesDirectory, exc.NoSignalsData, exc.NoSimaPath):
                if verbose:
                    "Missing imaging data:", expt
                self.remove(expt)
            else:
                if not len(im_data):
                    if verbose:
                        "Missing imaging data:", expt
                    self.remove(expt)
                    continue
                else:
                    if label is None:
                        expt_label = sima.misc.most_recent_key(im_data)
                    else:
                        expt_label = label
                    if not len(im_data.get(expt_label, {}).get('raw', [])):
                        if verbose:
                            "Missing imaging data:", expt
                        self.remove(expt)
                        continue
                if expt.get("ignoreImagingData"):
                    if verbose:
                        print "Marked to ignore:", expt
                    self.remove(expt)
                    continue
                if check_trans:
                    try:
                        expt.transientsData(channel=channel, label=label)
                    except (exc.NoTSeriesDirectory, exc.NoTransientsData):
                        if verbose:
                            print "Missing transients data:", expt
                        self.remove(expt)
                        continue

    # TODO: Move to lab/analysis/cleanup.py
    def removeDatalessTrials(
            self, channel='Ch2', label=None, check_trans=False, verbose=False):
        self.removeDatalessExperiments(
            channel=channel, label=label, check_trans=check_trans,
            verbose=verbose)
        for expt in self:
            imaging_shape = expt.imaging_shape(channel=channel, label=label)
            trials = expt.findall('trial')
            if len(trials) > imaging_shape[2]:
                for trial in trials[imaging_shape[2]:]:
                    if verbose:
                        print "More trials than data:", trial
                    expt.remove(trial)
            if len(expt.findall('trial')) == 0:
                if verbose:
                    print "Empty expt:", expt
                self.remove(expt)

    # TODO: Move to lab/analysis/cleanup.py
    def removeShortExperiments(
            self, duration=datetime.timedelta(seconds=60), laps=0,
            remove_incomplete=False, verbose=False):
        """Remove short experiments from the ExperimentGroup.

        Experiments can be removed either by time, by lap, or by an
        incomplete duration.

        NOTE: remove_incomplete will only work for fearconditioning
        (or related, ie runTraining) experiments

        """
        self.removeMissingBehaviorData(verbose=verbose)
        for expt in reversed(self):
            if laps:
                try:
                    nLaps = np.sum([int(np.amax(
                        ba.absolutePosition(
                            trial, imageSync=False,
                            sampling_interval='actual')))
                        for trial in expt.findall('trial')])
                except exc.MissingBehaviorData:
                    if verbose:
                        print "Missing behavior data:", expt
                    self.remove(expt)
                    continue
                else:
                    if nLaps < laps:
                        if verbose:
                            print "Too few laps ({}): {}".format(nLaps, expt)
                        self.remove(expt)
                        continue
            if expt.duration() is None or expt.duration() < duration:
                if verbose:
                    print "Expt too short:", expt
                self.remove(expt)
                continue
            if remove_incomplete and expt.duration() < \
                    datetime.timedelta(seconds=int(expt.get('environmentOnsetDelay', 0))) \
                    + datetime.timedelta(seconds=int(expt.get('environmentDuration', 0))) \
                    + datetime.timedelta(seconds=int(expt.get('postEnvironmentDelay', 0))):
                if verbose:
                    print "Expt incomplete:", expt
                self.remove(expt)
                continue

    def updateTime(
            self, channel='Ch2', label=None, demixed=False,
            ignore_signals=False, ignore_transients=False, ignore_dfof=False,
            ignore_place_fields=False):
        last_update = 0
        for expt in self:
            if label is None:
                expt_label = expt.most_recent_key(channel=channel)
            else:
                expt_label = label
            if not ignore_signals:
                try:
                    with open(expt.signalsFilePath(channel=channel), 'rb') as f:
                        signals = pickle.load(f)
                except (exc.NoSimaPath, exc.NoTSeriesDirectory, IOError, pickle.UnpicklingError):
                    pass
                else:
                    try:
                        update_time = signals[expt_label]['timestamp']
                    except KeyError:
                        pass
                    else:
                        if update_time > last_update:
                            last_update = update_time

            if not ignore_transients:
                try:
                    with open(expt.transientsFilePath(channel=channel), 'rb') as f:
                        transients = pickle.load(f)
                except (exc.NoSimaPath, exc.NoTSeriesDirectory, IOError, pickle.UnpicklingError):
                    pass
                else:
                    try:
                        update_time = transients[expt_label]['timestamp']
                    except KeyError:
                        pass
                    else:
                        if update_time > last_update:
                            last_update = update_time

            if not ignore_dfof:
                try:
                    with open(expt.dfofFilePath(channel=channel), 'rb') as f:
                        dfof = pickle.load(f)
                except (exc.NoSimaPath, exc.NoTSeriesDirectory, IOError, pickle.UnpicklingError):
                    pass
                else:
                    try:
                        update_time = dfof[expt_label]['timestamp']
                    except KeyError:
                        pass
                    else:
                        if update_time > last_update:
                            last_update = update_time

            if not ignore_place_fields:
                try:
                    with open(expt.placeFieldsFilePath(channel=channel), 'rb') as f:
                        place_fields = pickle.load(f)
                except (exc.NoSimaPath, exc.NoTSeriesDirectory, IOError, pickle.UnpicklingError):
                    pass
                else:
                    demixed_key = 'demixed' if demixed else 'undemixed'
                    try:
                        update_time = place_fields[expt_label][demixed_key]['timestamp']
                    except KeyError:
                        pass
                    else:
                        if update_time > last_update:
                            last_update = update_time

        return last_update

    # TODO: Cleanup
    def sharedROIs(self, roiType=None, fraction_isnans_threshold=0.,
                   contiguous_isnans_threshold=50, demixed=False,
                   checkValidity=False, ignoreXmarks=False,
                   orderByCluster=False, excludeXmarks=True, roi_filter=None,
                   label=None, channel='Ch2'):
        """Return list of ROIs that are present with valid data in all experiments"""
        # TODO: remove ignoreXmarks and excludeXmarks options, replaced by tags
        # and filters

        # Check to make sure all expts are from the same field
        fields = [(expt.parent, expt.get('uniqueLocationKey')) for expt in self]
        if not all([field == fields[0] for field in fields]):
            warnings.warn('Multiple fields of views in exptGrp, no shared ROIs')
            return []

        # initialize list of shared ROIs, don't filter
        initial_rois = self[0].roi_ids(label=label, channel=channel)
        if ignoreXmarks:
            for r_idx, r in enumerate(initial_rois):
                if r[:2] in ['x-', 'X-']:
                    initial_rois[r_idx] = r[2:]
        sharedROIs = set(initial_rois)
        # take intersection of the valid ROIs for each experiment
        for expt in self:
            if checkValidity:
                valid_filter = filters.validROIs(
                    expt, fraction_isnans_threshold=fraction_isnans_threshold,
                    contiguous_isnans_threshold=contiguous_isnans_threshold,
                    demixed=demixed, label=label,
                    channel=channel)
                f = misc.filter_intersection([valid_filter, roi_filter])
            else:
                f = roi_filter
            rois = expt.roi_ids(label=label, channel=channel, roi_filter=f)
            if ignoreXmarks:
                for r_idx, r in enumerate(rois):
                    if r[:2] in ['x-', 'X-']:
                        rois[r_idx] = r[2:]
            sharedROIs.intersection_update(rois)
        if excludeXmarks:
            sharedROIs = [r for r in sharedROIs if not r.lower().startswith('x-')]
        if roiType is not None:
            try:
                typeList = classifyROIs(expt, sharedROIs)
                if isinstance(roiType, RoiInfo):
                    roiType = [roiType]
                sharedROIs = [r for i, r in enumerate(sharedROIs) if typeList[i] in roiType]
            except exc.NoSignalsData:
                raise
            except:
                # TODO: this should become obsolete
                if isinstance(roiType, str) and roiType == 'GABAergic':
                    if any([any([v.get('name') == 'bz-183-gcamp5-flex' for v in
                                 inj.findall('virus')]) for inj in
                            self[0].parent.findall('injection')]):
                        sharedROIs = [r for r in sharedROIs if not
                                      (r.lower().startswith('x') or
                                       r.lower().startswith('axon'))]
                    else:
                        allROIs = sharedROIs
                        sharedROIs = [r for r in sharedROIs if r.lower().startswith('c')]
                        # include ROIs if any other ROI on the same axon contacts a cell
                        axonGroups = BoutonSet(allROIs).axonGroups()
                        for g in axonGroups:
                            if any([r in sharedROIs for r in g]):
                                sharedROIs |= g
                elif isinstance(roiType, RoiInfo):
                    sharedROIs = [r for r in sharedROIs if classifyROI(r, self[0].get('imagingLayer'),
                                  self[0].get('imagingLayer')) == roiType]
                elif isinstance(roiType, list) and isinstance(roiType[0], RoiInfo):
                    sharedROIs = [r for r in sharedROIs if classifyROI(r, self[0].get('imagingLayer'),
                                  self[0].get('imagingLayer')) in roiType]

        if orderByCluster:
            shared_filter = lambda x: x.id in sharedROIs
            _, sharedROIs = ia.baselineActivityCorrelations(
                self, includeStimResponse=True, offset=True, cluster='complete',
                channel=channel, label=label, roi_filter=shared_filter)
            return sharedROIs

        return sorted(sharedROIs)

    def allROIs(self, channel='Ch2', label=None, roi_filter=None):
        """Return a dictionary containing all the ROIs in the experiment group.
        The keys are a tuple of the format (mouse, uniqueLocationKey, roi_id)
        The values are a list of tuples of the format (experiment, roi_number)
        for all the experiments that contain the ROI

        """

        if not hasattr(self, '._all_rois'):
            self._all_rois = {}

        if (channel, label, roi_filter) in self._all_rois:
            # Check if exptGrp has changed
            expts = self._all_rois[(channel, label, roi_filter)]['expts']
            same_check = all(
                [a == b for a, b in it.izip_longest(self, expts)])
        else:
            same_check = False
        if not same_check:
            rois = defaultdict(list)
            for expt in self:
                for roi_idx, roi in enumerate(self.roi_ids(
                        channel=channel, label=label,
                        roi_filter=roi_filter)[expt]):
                    key = (expt.parent, expt.get('uniqueLocationKey'), roi)
                    value = (expt, roi_idx)
                    rois[key].append(value)
            self._all_rois[(channel, label, roi_filter)] = {}
            self._all_rois[(channel, label, roi_filter)]['expts'] = copy(self)
            self._all_rois[(channel, label, roi_filter)]['rois'] = dict(rois)
        return self._all_rois[(channel, label, roi_filter)]['rois']

    # TODO: REMOVE THIS
    def identify_active_cells(self, **kwargs):
        warnings.warn('DeprecationWarning')
        return filters.active_roi_filter(self, **kwargs)

    # TODO: Move to imaging_analysis.py
    def pca(self, trialAveraged=True, demixed=False,
            timeIndices=None, offset=False, dFOverF=None, channel='Ch2',
            label=None, roi_filter=None):
        """Perform principle component analysis on the experiment imaging ROI
        signals.

        Inputs:
            trialAveraged:
                average across trials if True
            demixed:
                use signal data in which the channels have been demixed with
                independent component analysis
            timeIndices:
                subset of time indices to be used when determining PCs
            offset:
                if true, perform a PCA variant the cross correlations offset by
                one time index

        Returns:
            coefficientTimeSeries:
                a 3D array with the first index determining the PC,
                the second index determining the time index, and the third
                index determining the trial number
            pcVariances:
                the variance of each PC, i.e. the eigenvalues of the covariance
                matrix
            pcWeights:
                the transformation matrix from the PCs to the ROIs
        """

        # get imaging data, with one column per array
        ROIs = self.sharedROIs(channel=channel, label=label,
                               roi_filter=roi_filter)
        shared_filter = lambda x: x.id in ROIs
        linearizedImDataList = []
        for experiment in self:
            imData = experiment.imagingData(
                dFOverF=dFOverF, demixed=demixed,
                channel=channel, label=label, roi_filter=shared_filter)
            #roiIndices = np.nonzero(np.all(np.isfinite(imData.reshape(
            #  [imData.shape[0], -1], order='F')), axis=1))[0]
            #imData = imData[roiIndices, :, :]
            #pcaRois = [ROIs[i] for i in roiIndices]
            if timeIndices is not None:
                imData = imData[:, timeIndices, :]
            if trialAveraged:
                try:
                    # if there are specific stimuli, don't average across stimuli
                    linearizedImData = []
                    for stimulus in experiment.stimuli():
                        linearizedImData.append(np.mean(
                            imData[:, :, experiment.trialIndices(stimulus)],
                            axis=2))
                    linearizedImData = np.concatenate(linearizedImData, axis=1)
                except AttributeError:
                    linearizedImData = np.mean(imData, axis=2)
            else:
                linearizedImData = imData.reshape([imData.shape[0], -1], order='F')
            assert len(linearizedImData.shape) == 2
            linearizedImDataList.append(linearizedImData)
        linearizedImData = np.concatenate(linearizedImDataList, axis=1)
        if offset:
            variances, pcs = offsetPCA(linearizedImData.T)
        else:
            import mdp
            pcaNode = mdp.nodes.PCANode(svd=True)  # create a PCA node object
            pcaResult = pcaNode.execute(linearizedImData.T)  # apply PCA to imaging data
            pcs = pcaNode.v
            variances = pcaNode.d
        for i in range(pcs.shape[1]):
            pcs[:, i] = pcs[:, i] * np.sign(np.mean(pcs[:, i]))
        coefficientTimeSeries = np.tensordot(pcs.T, imData, axes=1)
        return coefficientTimeSeries, variances, pcs, ROIs

    def pcaCleanupMatrix(self, channel='Ch2', label=None, roi_filter=None):
        imData = []
        for experiment in self:
            imData.append(experiment.imagingData(
                dFOverF=None, channel=channel, label=label,
                roi_filter=roi_filter))
        imData = [x.reshape([x.shape[0], -1], order='F') for x in imData]
        imData = np.concatenate(imData, axis=1)
        return ia.pcaCleanupMatrix(imData)

    def group_by_condition(self, ignoreBelt=False, ignoreContext=False,
                           ignoreRewardPositions=False, by_mouse=False):
        """Returns all experiments as a list of lists, grouped by experiments
        performed under the same conditions.

        """
        expts_by_condition = []
        for expt in self:
            match_found = False
            for idx, condition_expts in it.izip(it.count(), expts_by_condition):
                if expt.sameConditions(
                        condition_expts[0], ignoreBelt=ignoreBelt,
                        ignoreContext=ignoreContext,
                        ignoreRewardPositions=ignoreRewardPositions) and \
                        (not by_mouse or expt.parent ==
                         condition_expts[0].parent):
                    expts_by_condition[idx].append(expt)
                    match_found = True
                    break
            if not match_found:
                expts_by_condition.append([expt])

        return expts_by_condition

    def priorDaysOfExposure(
            self, combineTimes=datetime.timedelta(hours=12), ignoreBelt=False,
            ignoreContext=False, ignoreRewardPositions=False,
            number_in_group=False):
        """Returns a dictionary of the total number of days of pre-exposure to
        the belt for the given experiment.

        """

        days = {}
        for expt in self:
            exptTime = expt.startTime()
            if number_in_group:
                mouse = expt.parent.get('mouseID')
                expts = filter(
                    lambda ee: ee.parent.get('mouseID') == mouse, self)
            else:
                expts = expt.parent.findall('experiment')
            # Pull out all experiments prior to the current experiment and
            # compare to make sure only 1 per day
            times = np.array([])
            for ee in expts:
                try:
                    currentTime = ee.startTime()
                except:
                    continue
                timeDiff = np.abs(times - currentTime)
                # Don't count experiments that occurred on the same day
                if exptTime - currentTime > combineTimes and \
                        all(timeDiff > combineTimes) and \
                        expt.sameConditions(
                            ee, ignoreBelt=ignoreBelt,
                            ignoreContext=ignoreContext,
                            ignoreRewardPositions=ignoreRewardPositions):
                    times = np.hstack((times, currentTime))
            days[expt] = len(times)
        return days

    def session_number(
            self, per_day=False, min_duration=datetime.timedelta(seconds=30),
            ignoreBelt=False, ignoreContext=False, ignoreRewardPositions=False,
            number_in_group=False):
        """The number of the session for all experiments

        Arguments:
        per_day : if True, return the session number within the day, otherwise
            the session number over all experiments
        min_duration : Ignore sessions for counting purposes shorter than
            min_duration. Ignored if 'number_in_group' is True.
        ignoreBelt, ignoreContext, ignoreRewardPositions : passed to
            sameConditions to find all experiments that match conditions
        number_in_group : If True, determine the session number within the
            experimentGroup, otherwise across all experiments for that mouse

        """

        sessions = {}
        for expt in self:
            if number_in_group:
                mouse = expt.parent.get('mouseID')
                expts = filter(
                    lambda ee: ee.parent.get('mouseID') == mouse, self)
            else:
                expts = expt.parent.findall('experiment')
            exptGrp = ExperimentGroup(expts)
            if per_day:
                days = exptGrp.priorDaysOfExposure(
                    combineTimes=datetime.timedelta(hours=12),
                    ignoreBelt=ignoreBelt, ignoreContext=ignoreContext,
                    ignoreRewardPositions=ignoreRewardPositions)
                expt_exposure = days[expt]
            # Make a list of experiments that match the same conditions
            # If 'per_day', only include experiments on the same day
            # If not 'number_in_group', toss experiments that are too short
            # (except for the expt itself, to ensure it's always included)
            same_exposure_expts = []
            for ee in exptGrp:
                try:
                    if expt.sameConditions(
                            ee, ignoreBelt=ignoreBelt,
                            ignoreContext=ignoreContext,
                            ignoreRewardPositions=ignoreRewardPositions) and \
                            (not per_day or days[ee] == expt_exposure) and \
                            (number_in_group or ee == expt or ee.duration() >
                             min_duration):
                        same_exposure_expts.append(ee)
                except:
                    # TODO: What is the excepting? Is this correct?
                    same_exposure_expts.append(ee)
                    print(ee)
            sessions[expt] = sorted(same_exposure_expts).index(expt)

        return sessions

    def condition_label(
            self, ignoreBelt=False, ignoreContext=False,
            ignoreRewardPositions=False, by_mouse=False):
        """Infers a condition label for each experiment
        Also returns the condition for each label as a
        (belt, context, rewardPositions) tuple

        """

        expts_by_condition = self.group_by_condition(
            ignoreBelt=ignoreBelt, ignoreContext=ignoreContext,
            ignoreRewardPositions=ignoreRewardPositions, by_mouse=by_mouse)

        session_number = self.session_number(
            ignoreBelt=True, ignoreContext=True, ignoreRewardPositions=True,
            number_in_group=True)

        mean_session_number_by_condition = [
            np.mean([session_number[expt] for expt in condition])
            for condition in expts_by_condition]

        labels = {}
        conditions = defaultdict(dict)

        if by_mouse:
            expts_by_mouse_by_condition = defaultdict(list)
            mean_session_number_by_mouse_by_condition = defaultdict(list)
            for expts, session_mean in zip(
                    expts_by_condition, mean_session_number_by_condition):
                expts_by_mouse_by_condition[expts[0].parent].append(expts)
                mean_session_number_by_mouse_by_condition[
                    expts[0].parent].append(session_mean)

            for mouse in expts_by_mouse_by_condition:
                for label_idx, group_idx in enumerate(
                        np.argsort(mean_session_number_by_mouse_by_condition[mouse])):
                    for expt in expts_by_mouse_by_condition[mouse][group_idx]:
                        labels[expt] = ascii_uppercase[label_idx]

                    conditions[mouse][ascii_uppercase[label_idx]] = (
                        'IGNORE' if ignoreBelt else expt.get('belt'),
                        'IGNORE' if ignoreContext else expt.get('environment'),
                        'IGNORE' if ignoreRewardPositions else str(
                            expt.rewardPositions(units=None)))

        else:
            for label_idx, group_idx in enumerate(
                    np.argsort(mean_session_number_by_condition)):
                for expt in expts_by_condition[group_idx]:
                    labels[expt] = ascii_uppercase[label_idx]

                conditions[ascii_uppercase[label_idx]] = (
                    'IGNORE' if ignoreBelt else expt.get('belt'),
                    'IGNORE' if ignoreContext else expt.get('environment'),
                    'IGNORE' if ignoreRewardPositions else str(
                        expt.rewardPositions(units=None)))

        return labels, conditions

    def sameBelt(self):
        """Returns True if all experiments were on the same belt,
        otherwise returns False

        """

        try:
            belts = [expt.belt().get('beltID') for expt in self]
        except exc.NoBeltInfo:
            return False
        return len(set(belts)) == 1

    # TODO: Move to behavior_analysis.py
    @staticmethod
    def time_per_lap(exptGrp):
        """Calculates the time (in seconds) per lap"""

        result = []
        for expt in exptGrp:
            for trial in expt.findall('trial'):
                try:
                    sampling_interval = trial.behavior_sampling_interval()
                    pos = ba.absolutePosition(
                        trial, imageSync=False, sampling_interval='actual')
                    n_laps = int(pos.max())
                    #set_trace()
                    # Need at least 1 full lap to calculate time
                    if n_laps < 2:
                        continue
                    for lap in range(1, n_laps):
                        start_frame = np.where(pos >= lap)[0][0]
                        end_frame = np.where(pos >= lap + 1)[0][0]
                        time = (end_frame - start_frame) * sampling_interval
                        result.append({'trial': trial, 'lap': lap, 'value': time})
                except:
                    print "{} failed to calculate, not included in final analysis".format(trial)
        return pd.DataFrame(result)

    @staticmethod
    def dataframe(exptGrp, include_columns=None, add_as_attrib=False):
        """Returns a dataframe just containing all the trials in the group.
        Used to calculate stats on the number of experiments/trials."""

        result = []
        for expt in exptGrp:
            for trial in expt.findall('trial'):
                result.append({'trial': trial, 'value': 1})

        if len(result) == 0:
            all_cols = ['trial', 'value']
            if include_columns is not None:
                all_cols += include_columns
            return pd.DataFrame(result, columns=include_columns)

        df = pd.DataFrame(result)
        if include_columns:
            ph.prepare_dataframe(df, include_columns=include_columns)

        if add_as_attrib and include_columns is not None:
            for column in include_columns:
                map_dict = {trial.parent: col_val for trial, col_val in zip(
                    df['trial'], df[column])}
                for expt in exptGrp:
                    expt.attrib[column] = map_dict[expt]

        return df

    # TODO: Move to behavior_analysis.py
    @staticmethod
    def lick_bout_duration(exptGrp, bouts_to_include='all', threshold=1.0):
        """Returns the duration of lick bouts (in sec)."""

        result = []
        for expt in exptGrp:
            rewarded, unrewarded = ba.calculateRewardedLickIntervals(
                expt, imageSync=False, threshold=threshold)
            for trial, rew, unrew in it.izip(
                    expt.findall('trial'), rewarded, unrewarded):
                sampling_interval = trial.behavior_sampling_interval()
                if bouts_to_include == 'all':
                    bouts = np.vstack((rew, unrew))
                elif bouts_to_include == 'rewarded':
                    bouts = rew
                elif bouts_to_include == 'unrewarded':
                    bouts = unrew
                else:
                    raise ValueError('Unrecognized bouts_to_include value')

                if not len(bouts):
                    continue

                durations = (bouts[:, 1] - bouts[:, 0]) * sampling_interval
                for d in durations:
                    result.append({'trial': trial, 'value': d})

        return pd.DataFrame(result)

    @staticmethod
    def filtered_rois(expt_grp, roi_filter, include_roi_filter=None,
                      channel='Ch2', label=None):
        """Returns 1/0 for whether or not an roi is in a filter.

        roi_filter is the filter that is tested, and include_roi_filter is a
        filter for which ROIs to test at all.

        For example, to see which red cells had 2 transient, roi_filter would
        be the 2 transient filter and include_roi_filter would be the red cell
        filter

        """

        if include_roi_filter is None:
            def include_roi_filter(roi):
                return True

        result = []
        all_rois = expt_grp.rois(channel=channel, label=label, roi_filter=None)
        for expt in expt_grp:
            all_rs = all_rois[expt]
            if all_rs is None:
                continue
            for roi in all_rs:
                if include_roi_filter(roi):
                    result.append(
                        {'expt': expt, 'roi': roi,
                         'value': int(roi_filter(roi))})

        return pd.DataFrame(result)

    @staticmethod
    def stims_per_lap(expt_grp, stimulus, trim_incomplete=True):
        """Return the number of stims per lap.

        Parameters
        ----------
        exptGrp : lab.classes.ExperimentGroup
        stimulus: str
            String should be a key in return of lab.classes.Trial.behaviorData.
        trim_incomplete : optional, bool
            If True, only include full laps.

        """
        result = []
        for expt in expt_grp:
            for trial in expt.findall('trial'):
                stim = trial.behaviorData(
                    imageSync=False, sampling_interval='actual')[stimulus]
                pos = ba.absolutePosition(
                    trial, imageSync=False, sampling_interval='actual')
                n_laps = int(pos.max())
                if trim_incomplete:
                    if n_laps < 2:
                        continue
                    laps = range(1, n_laps)
                else:
                    laps = range(0, n_laps + 1)
                stim_starts = np.diff(np.hstack([0, stim]).astype('int')) > 0
                for lap in laps:
                    lap_pos = np.logical_and(pos >= lap, pos < lap + 1)
                    stims = stim_starts[lap_pos].sum()
                    result.append({'trial': trial, 'lap': lap, 'value': stims})

        return pd.DataFrame(result)

    @staticmethod
    def number_of_laps(exptGrp, rate=False):

        result = []
        for expt in exptGrp:
            for trial in expt.findall('trial'):
                pos = ba.absolutePosition(
                    trial, imageSync=False, sampling_interval='actual')
                laps = pos.max() - pos.min()
                if rate:
                    time = trial.behaviorData(imageSync=False)['recordingDuration'] / 60.
                    laps /= time
                result.append({'trial': trial, 'value': laps})

        return pd.DataFrame(result)

    @staticmethod
    def number_of_licks(exptGrp):

        result = []
        for expt in exptGrp:
            for trial in expt.findall('trial'):
                n_licks = len(trial.behaviorData()['licking'])
                result.append({'trial': trial, 'value': n_licks})
        return pd.DataFrame(result)

    @staticmethod
    def behavior_dataframe(exptGrp, key, start=0, stop=None, rate=False):

        result = []
        for expt in exptGrp:
            for trial in expt.findall('trial'):
                bd = trial.behaviorData()
                behavior = bd[key]
                if rate:
                    value = 1. / bd['recordingDuration']
                else:
                    value = 1
                for stim in behavior:
                    if stim[0] >= start and (stop is None or stim[1] < stop):
                        result.append(
                            {'trial': trial, 'on_time': stim[0],
                             'off_time': stim[1], 'value': value})

        return pd.DataFrame(result)

    @staticmethod
    def stim_position(exptGrp, stimulus, normalized=True, abs_pos=False):

        result = []
        for expt in exptGrp:
            for trial in expt.findall('trial'):
                bd = trial.behaviorData(
                    imageSync=False, sampling_interval='actual')
                stim = bd[stimulus]
                track_length = bd['trackLength']
                if abs_pos:
                    pos = ba.absolutePosition(
                        trial, imageSync=False, sampling_interval='actual')
                else:
                    pos = bd['treadmillPosition']
                stim_starts = np.diff(np.hstack([0, stim]).astype('int')) > 0
                for stim_pos in pos[stim_starts]:
                    if not normalized:
                        stim_pos = stim_pos * track_length
                    result.append({'trial': trial, 'value': stim_pos})

        return pd.DataFrame(result)

    @staticmethod
    def velocity_dataframe(expt_grp, running_only=True, running_kwargs=None):

        if running_kwargs is None:
            running_kwargs = {}

        result = []
        for expt in expt_grp:
            for trial in expt.findall('trial'):
                vel = ba.velocity(trial, imageSync=False)
                if running_only:
                    running = ba.runningIntervals(
                        trial, imageSync=False, returnBoolList=True,
                        **running_kwargs)
                    running = running.astype('float')
                    running[running == False] = np.nan
                    assert vel.shape == running.shape
                    vel *= running
                    value = np.nanmean(vel)
                else:
                    value = np.mean(vel)
                result.append({'trial': trial, 'value': value})

        return pd.DataFrame(result)

    # END ExperimentGroup class


class HiddenRewardExperimentGroup(ExperimentGroup):

    @classmethod
    def fromMice(cls, mice, **kwargs):
        expt_list = []
        for mouse in mice:
            # print mouse
            for expt in mouse.findall('experiment'):
                if expt.get('experimentType') == 'hiddenRewards':
                    expt_list.append(expt)
        return cls(expt_list, **kwargs)

    def compareLicktograms(self, ax=None, nPositionBins=100):

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        for expt in self:
            [percentLicks, binEdges] = expt.licktogram(
                nPositionBins=nPositionBins)
            ax.bar(binEdges, percentLicks, width=1. / nPositionBins,
                   color=np.random.rand(3), alpha=0.4,
                   label=expt.get('startTime'))

        rewardPositions = self[0].rewardPositions(units='normalized')
        for p in rewardPositions:
            ax.axvline(x=p, linewidth=2, color='k', linestyle='--')

        ax.legend()
        ax.set_xlabel('Normalized position')
        ax.set_ylabel('Percent of total licks')
        ax.set_title('Licking by position')

        self[0].belt().addToAxis(ax)

    # END HiddenRewardsExperimentGroup class


class PairedExperimentGroup(object):
    """Wrapper class to pair experiments by time."""

    def __init__(self, expt_grp, max_pair_delta=datetime.timedelta(hours=6),
                 min_pair_delta=datetime.timedelta(0)):
        self.expt_grp = copy(expt_grp)
        self.max_pair_delta = max_pair_delta
        self.min_pair_delta = min_pair_delta
        self._removeUnpairedExperiments()
        if self.label() == 'infer':
            self.label(self.inferLabel())

    def __copy__(self):
        return type(self)(
            copy(self.expt_grp), max_pair_delta=self.max_pair_delta,
            min_pair_delta=self.min_pair_delta)

    def __deepcopy__(self):
        return type(self)(
            deepcopy(self.expt_grp), max_pair_delta=self.max_pair_delta,
            min_pair_delta=self.min_pair_delta)

    def __getattr__(self, name):
        return getattr(self.expt_grp, name)
        # return self.expt_grp.__getattribute__(name)

    def __setitem__(self, i, v):
        self.expt_grp[i] = v
        self._removeUnpairedExperiments()

    def __delitem__(self, i):
        del self.expt_grp[i]
        self._removeUnpairedExperiments()

    def __len__(self):
        return len(self.expt_grp)

    def __getitem__(self, i):
        return self.expt_grp[i]

    def __iter__(self):
        return self.expt_grp.__iter__()

    def __reversed__(self):
        return self.expt_grp.__reversed__()

    def __str__(self):
        return "<PairedExperimentGroup " + self.expt_grp.__str__() + ">"

    def __repr__(self):
        return '{}(expt_grp={}, min_pair_delta={}, max_pair_delta={})'.format(
            repr(type(self)), repr(self.expt_grp),
            repr(self.min_pair_delta), repr(self.max_pair_delta))

    def remove(self, expt):
        self.expt_grp.remove(expt)
        self._removeUnpairedExperiments()

    def append(self, expt):
        self.expt_grp.append(expt)
        self._removeUnpairedExperiments()

    def extend(self, expt_grp):
        self.expt_grp.extend(expt_grp)
        self._removeUnpairedExperiments()

    def subGroup(self, expts, **kwargs):
        new_grp = self.expt_grp.subGroup(expts, **kwargs)
        return new_grp.pair(method='time', min_pair_delta=self.min_pair_delta,
                            max_pair_delta=self.max_pair_delta)

    def pair(self, method='time', **kwargs):
        if method == 'time':
            return PairedExperimentGroup(self, **kwargs)
        elif method == 'consecutive groups':
            return ConsecutiveGroupsPairedExperimentGroup(self, **kwargs)
        elif method == 'same group':
            return SameGroupPairedExperimentGroup(self, **kwargs)
        else:
            raise ValueError(
                "Unrecognized 'method' argument: {}".format(method))

    def unpair(self, label=None):
        if label is None:
            return copy(self.expt_grp)
        else:
            grp = copy(self.expt_grp)
            grp.label(label)
            return grp

    def _removeUnpairedExperiments(self):
        """Remove all experiments that can not be paired up"""
        all_paired = []
        for e1, e2 in self.genImagedExptPairs():
            all_paired.extend([e1, e2])

        all_paired_unique = set(all_paired)

        self.expt_grp.filter(lambda expt: expt in all_paired_unique)

    def genImagedExptPairs(self):
        """Returns pairs of experiments matched for imaging field"""
        for e1, e2 in self.expt_grp.genImagedExptPairs(
                ignore_conditions=True):
            if self.min_pair_delta < e2 - e1 < self.max_pair_delta:
                yield e1, e2

    def splitExptGrpByConditions(self, **conditions_kwargs):
        """Return a list of PairedExperimentGroups matched by conditions"""
        new_expt_lists = []
        for e1, e2 in self.genImagedExptPairs():
            match_found = False
            for idx, expt_list in enumerate(new_expt_lists):
                last_expt_pair = expt_list[-2:]
                if (e1.sameConditions(
                    last_expt_pair[0], **conditions_kwargs) and
                        e2.sameConditions(
                            last_expt_pair[1], **conditions_kwargs)) or \
                   (e2.sameConditions(
                    last_expt_pair[0], **conditions_kwargs) and
                        e1.sameConditions(
                            last_expt_pair[1], **conditions_kwargs)):
                    match_found = True
                    new_expt_lists[idx].extend([e1, e2])
                    break
            if not match_found:
                new_expt_lists.append([e1, e2])

        grps = [self.subGroup(expt_list) for expt_list in new_expt_lists]
        for grp in grps:
            grp.label(grp.inferLabel())
        return grps

    def inferLabel(self):
        """Used to infer a label for the group if possible"""
        total = 0
        same_belt_same_ctx = 0
        same_belt_diff_ctxs = 0
        diff_belts_same_ctx = 0
        diff_belts_diff_ctxs = 0
        same_belt_same_ctx_diff_rewards = 0

        for e1, e2 in self.genImagedExptPairs():
            if e1.sameConditions(e2, ignoreBelt=False, ignoreContext=False, ignoreRewardPositions=False):
                same_belt_same_ctx += 1
            elif e1.sameConditions(e2, ignoreBelt=False, ignoreContext=False, ignoreRewardPositions=True):
                same_belt_same_ctx_diff_rewards += 1
            elif e1.sameConditions(e2, ignoreBelt=False, ignoreContext=True):
                same_belt_diff_ctxs += 1
            elif e1.sameConditions(e2, ignoreBelt=True, ignoreContext=False):
                diff_belts_same_ctx += 1
            else:
                diff_belts_diff_ctxs += 1
            total += 1
        if same_belt_same_ctx == total:
            return "SameAll"
        if same_belt_diff_ctxs == total:
            return "DiffCtxs"
        if diff_belts_same_ctx == total:
            return "DiffBelts"
        if diff_belts_diff_ctxs == total:
            return "DiffAll"
        if same_belt_same_ctx_diff_rewards == total:
            return "SameAll_DiffRewards"

        return None


class ConsecutiveGroupsPairedExperimentGroup(object):
    """Wrapper class to pair experiments from consecutive group."""

    def __init__(self, expt_grp, groupby):
        self.expt_grp = copy(expt_grp)
        self._groupby = copy(groupby)
        self._removeUnpairedExperiments()

    def __copy__(self):
        return type(self)(copy(self.expt_grp), groupby=copy(self._groupby))

    def __deepcopy__(self):
        return type(self)(deepcopy(self.expt_grp),
                          groupby=deepcopy(self._groupby))

    def __getattr__(self, name):
        return getattr(self.expt_grp, name)

    def __setitem__(self, i, v):
        self.expt_grp.__setitem__(i, v)
        self._removeUnpairedExperiments()

    def __delitem__(self, i):
        self.expt_grp.__delitem__(i)
        self._removeUnpairedExperiments()

    def __len__(self):
        return len(self.expt_grp)

    def __getitem__(self, i):
        return self.expt_grp[i]

    def __iter__(self):
        return self.expt_grp.__iter__()

    def __reversed__(self):
        return self.expt_grp.__reversed__()

    def __str__(self):
        return "<ConsecutiveGroupsPairedExperimentGroup " + \
            self.expt_grp.__str__() + ">"

    def __repr__(self):
        return '{}(expt_grp={}, groupby={})'.format(
            repr(type(self)), repr(self.expt_grp), repr(self._groupby))

    def remove(self, expt):
        self.expt_grp.remove(expt)
        self._removeUnpairedExperiments()

    def append(self, expt):
        self.expt_grp.append(expt)
        self._removeUnpairedExperiments()

    def extend(self, expt_grp):
        self.expt_grp.extend(expt_grp)
        self._removeUnpairedExperiments()

    def subGroup(self, expts, **kwargs):
        new_grp = self.expt_grp.subGroup(expts, **kwargs)
        return new_grp.pair(method='consecutive groups', groupby=self._groupby)

    def pair(self, method='time', **kwargs):
        if method == 'time':
            return PairedExperimentGroup(self, **kwargs)
        elif method == 'consecutive groups':
            return ConsecutiveGroupsPairedExperimentGroup(self, **kwargs)
        elif method == 'same group':
            return SameGroupPairedExperimentGroup(self, **kwargs)
        else:
            raise ValueError(
                "Unrecognized 'method' argument: {}".format(method))

    def unpair(self, label=None):
        if label is None:
            return copy(self.expt_grp)
        else:
            grp = copy(self.expt_grp)
            grp.label(label)
            return grp

    def _removeUnpairedExperiments(self):
        """Remove all experiments that cannot be paired up"""
        all_paired = []
        for e1, e2 in self.genImagedExptPairs():
            all_paired.extend([e1, e2])

        all_paired_unique = set(all_paired)

        self.expt_grp.filter(lambda expt: expt in all_paired_unique)

    def genImagedExptPairs(self, **kwargs):
        """Returns pairs of experiments matched for imaging field"""
        keys = []
        group_by_expt = {}
        for key, group in self.groupby(self._groupby):
            keys.append(key)
            for expt in group:
                group_by_expt[expt] = key
        keys = sorted(keys)

        for e1, e2 in self.expt_grp.genImagedExptPairs(
                ignore_conditions=True):

            if e1 not in group_by_expt or e2 not in group_by_expt:
                continue

            e1_key_index = keys.index(group_by_expt[e1])
            if e1_key_index + 1 < len(keys) and \
                    group_by_expt[e2] == keys[e1_key_index + 1]:
                yield e1, e2

    def splitExptGrpByConditions(self, **conditions_kwargs):
        """Return a list of PairedExperimentGroups matched by conditions"""
        raise NotImplemented
        new_grps = []
        for key, group in self.groupby(self._groupby):
            new_grps.append()
        return new_grps


class SameGroupPairedExperimentGroup(object):
    """Wrapper class to pair experiments by matching group."""

    def __init__(self, expt_grp, groupby):
        self.expt_grp = copy(expt_grp)
        self._groupby = copy(groupby)
        self._removeUnpairedExperiments()

    def __copy__(self):
        return type(self)(copy(self.expt_grp), groupby=copy(self._groupby))

    def __deepcopy__(self):
        return type(self)(deepcopy(self.expt_grp),
                          groupby=deepcopy(self._groupby))

    def __getattr__(self, name):
        return getattr(self.expt_grp, name)

    def __setitem__(self, i, v):
        self.expt_grp.__setitem__(i, v)
        self._removeUnpairedExperiments()

    def __delitem__(self, i):
        self.expt_grp.__delitem__(i)
        self._removeUnpairedExperiments()

    def __len__(self):
        return len(self.expt_grp)

    def __getitem__(self, i):
        return self.expt_grp[i]

    def __iter__(self):
        return self.expt_grp.__iter__()

    def __reversed__(self):
        return self.expt_grp.__reversed__()

    def __str__(self):
        return "<SameGroupPairedExperimentGroup " + \
            self.expt_grp.__str__() + ">"

    def __repr__(self):
        return '{}(expt_grp={}, groupby={})'.format(
            repr(type(self)), repr(self.expt_grp), repr(self._groupby))

    def remove(self, expt):
        self.expt_grp.remove(expt)
        self._removeUnpairedExperiments()

    def append(self, expt):
        self.expt_grp.append(expt)
        self._removeUnpairedExperiments()

    def extend(self, expt_grp):
        self.expt_grp.extend(expt_grp)
        self._removeUnpairedExperiments()

    def subGroup(self, expts, **kwargs):
        new_grp = self.expt_grp.subGroup(expts, **kwargs)
        return new_grp.pair(method='same group', groupby=self._groupby)

    def pair(self, method='time', **kwargs):
        if method == 'time':
            return PairedExperimentGroup(self, **kwargs)
        elif method == 'consecutive groups':
            return ConsecutiveGroupsPairedExperimentGroup(self, **kwargs)
        elif method == 'same group':
            return SameGroupPairedExperimentGroup(self, **kwargs)
        else:
            raise ValueError(
                "Unrecognized 'method' argument: {}".format(method))

    def unpair(self, label=None):
        if label is None:
            return copy(self.expt_grp)
        else:
            grp = copy(self.expt_grp)
            grp.label(label)
            return grp

    def _removeUnpairedExperiments(self):
        """Remove all experiments that cannot be paired up"""
        all_paired = []
        for e1, e2 in self.genImagedExptPairs():
            all_paired.extend([e1, e2])

        all_paired_unique = set(all_paired)

        self.expt_grp.filter(lambda expt: expt in all_paired_unique)

    def genImagedExptPairs(self, **kwargs):
        """Returns pairs of experiments matched for imaging field"""
        group_by_expt = {}
        for key, group in self.groupby(self._groupby):
            for expt in group:
                group_by_expt[expt] = key

        for e1, e2 in self.expt_grp.genImagedExptPairs(
                ignore_conditions=True):

            if e1 not in group_by_expt or e2 not in group_by_expt:
                continue

            if group_by_expt[e1] == group_by_expt[e2]:
                yield e1, e2

    def splitExptGrpByConditions(self, **conditions_kwargs):
        """Return a list of PairedExperimentGroups matched by conditions"""
        raise NotImplemented
        new_grps = []
        for key, group in self.groupby(self._groupby):
            new_grps.append()
        return new_grps


class CFCExperimentGroup(ExperimentGroup):
    """Experiment group for CFC experiments

    NOTE: This assumes only one set of CFC experiments (habituation, conditioning, recall) per mouse

    """

    def __init__(self, experimentList, label=None):
        self._label = label
        self._habituation, self._conditioning, self._recall = self.inferTrialType(experimentList)

    """
    Re-implement container functions
    """
    def __len__(self):
        return len(self._habituation) + len(self._conditioning) + len(self._recall)

    def __getitem__(self, i):
        return (self._habituation + self._conditioning + self._recall)[i]

    def __setitem__(self, i, v):
        replace = self[i]
        if replace in self._habituation:
            self._habituation[self._habituation.index(replace)] = v
        elif replace in self._conditioning:
            self._conditioning[self._conditioning.index(replace)] = v
        elif replace in self._recall:
            self._recall[self._recall.index(replace)] = v
        else:
            raise IndexError('CFCExperiemntGroup.__setitem__: Invalid index')

    def __delitem__(self, i):
        self.remove(self[i])

    def __iter__(self):
        return (self._habituation + self._conditioning + self._recall).__iter__()

    def __reversed__(self):
        return (self._habituation + self._conditioning + self._recall).__reversed__()

    def __str__(self):
        return "<CFC Experiment group: label={label}, nHab={nHab}, nCon={nCon}, nRec={nRec}>".format(
            label=self.label(), nHab=len(self._habituation),
            nCon=len(self._conditioning), nRec=len(self._recall))

    def __repr__(self):
        return (self._habituation + self._conditioning + self._recall).__repr__()

    def remove(self, expt):
        if expt in self._habituation:
            self._habituation.remove(expt)
        elif expt in self._conditioning:
            self._conditioning.remove(expt)
        else:
            self._recall.remove(expt)

    @staticmethod
    def inferTrialType(exptList,
                       conditioningWindow=datetime.timedelta(hours=12)):
        """For each experiment in exptList attempt to infer the trial type (habituation, conditioning, recall)

        Looks at each mouse individually for any trial that had airpuffs, anything within same_day_threshold is considered a conditioning trial
        Any trials before are habituation and any after are recall
        Throws a warning if unable to determine trial type

        Returns a tuple (habituation_experiments, conditioning_experiments, recall_experiments)

        Keyword arguments:
        conditioningWindow -- any trials within conditioningWindow of

        """

        # Need to figure out why isinstance never seems to work...
        #exptList = [expt for expt in exptList if isinstance(expt, FearConditioningExperiment)]
        exptList = [expt for expt in exptList if
                    expt.get('experimentType') == 'contextualFearConditioning']

        expts_by_mouse = ExperimentGroup.dictByMouse(exptList)

        habituation_expts = []
        conditioning_expts = []
        recall_expts = []

        for mouse in expts_by_mouse:
            puffed_expts = []
            for expt in expts_by_mouse[mouse]:
                try:
                    if expt.isPuffed():
                        puffed_expts.append(expt)
                except (exc.MissingBehaviorData, KeyError):
                    pass
            if len(puffed_expts) == 0:
                warnings.warn('Unable to determine conditioning experiment ' +
                              'for {}'.format(mouse.get('mouseID')))
                continue

            puffed_expt_times = np.sort([expt.startTime() for expt in puffed_expts])

            for expt in expts_by_mouse[mouse]:
                expt_time = expt.startTime()
                if np.any(np.abs(expt_time - puffed_expt_times) <= conditioningWindow):
                    conditioning_expts.append(expt)
                elif expt_time < puffed_expt_times[0]:
                    habituation_expts.append(expt)
                elif expt_time > puffed_expt_times[-1]:
                    recall_expts.append(expt)
                else:
                    warnings.warn(
                        'Unable to determine CFC trial type: {}_{}'.format(
                            mouse.get('mouseID'), expt.get('startTime')))

        return (habituation_expts, conditioning_expts, recall_expts)

    def addExperiments(self, exptList):
        """Add additional experiments to an already existing group"""
        hab, con, rec = CFCExperimentGroup.inferTrialType(exptList)
        self._habituation.extend(hab)
        self._conditioning.extend(con)
        self._recall.extend(rec)

    def puffedContexts(self):
        """Returns a dictionary with mice as keys and the puffed context as values"""
        puffed = {}
        for expt in self._conditioning:
            if expt.isPuffed():
                if expt.parent in puffed:
                    if expt.get('environment') not in puffed[expt.parent]:
                        puffed[expt.parent].append(expt.get('environment'))
                else:
                    puffed[expt.parent] = [expt.get('environment')]
        return puffed

    def getExperiments(self, type='all', context='all'):
        """Returns subsets of the experiments

        Keyword arguments:
        type -- one of: 'all', 'habituation', 'conditioning', 'recall'
        context -- one of 'all', 'conditioned'/'puffed', 'safe'/'neutral', 'A'/'B'/'None'...

        """

        if type == 'all':
            expts = self._conditioning + self._habituation + self._recall
        elif type in ['habituation', 'hab']:
            expts = self._habituation
        elif type in ['conditioning', 'cond', 'con']:
            expts = self._conditioning
        elif type in ['recall', 'rec']:
            expts = self._recall
        else:
            raise Exception("Invalid argument: unrecognized type value, '{}'".format(type))

        if context == 'all':
            return expts
        elif context in ['conditioned', 'puffed']:
            return [expt for expt in expts if self.isConditionedContext(expt)]
        elif context in ['safe', 'neutral']:
            return [expt for expt in expts if not self.isConditionedContext(expt)]
        else:
            return [expt for expt in expts if expt.get('environment') == context]

    def isConditionedContext(self, expt):
        """Returns true/false if the experiments is/is not in the conditioned context"""
        if expt not in self:
            raise Exception('Experiment not in experiment group')
        for con_expt in self._conditioning:
            if con_expt.parent == expt.parent and con_expt.get('environment') == expt.get('environment') and con_expt.isPuffed():
                return True
        return False

    def lickRatePlot(self, ax=None, separateDays=False, plotCon=True):
        """Plot average lick rate by hab/con/rec

        Keyword arguments:
        separateDays -- if False averages all experiments within hab/con/rec, otherwise separate by days
        plotCon -- if True plots conditioning data

        """

        if ax is None:
            fig, ax = plt.subplots()

        colors = ['r', 'b']
        bar_width = 0.35

        habC = self.getExperiments(type='hab', context='conditioned')
        habN = self.getExperiments(type='hab', context='neutral')
        conC = self.getExperiments(type='con', context='conditioned')
        conN = self.getExperiments(type='con', context='neutral')
        recC = self.getExperiments(type='rec', context='conditioned')
        recN = self.getExperiments(type='rec', context='neutral')

        if not separateDays:
            neu_means = [np.mean([expt.lickRateInContext() for expt in habN]),
                         np.mean([expt.lickRateInContext() for expt in conN]),
                         np.mean([expt.lickRateInContext() for expt in recN])]
            con_means = [np.mean([expt.lickRateInContext() for expt in habC]),
                         np.mean([expt.lickRateInContext() for expt in conC]),
                         np.mean([expt.lickRateInContext() for expt in recC])]
            neu_sem = [np.std([expt.lickRateInContext() for expt in habN]) / np.sqrt(len(habN)),
                       np.std([expt.lickRateInContext() for expt in conN]) / np.sqrt(len(conN)),
                       np.std([expt.lickRateInContext() for expt in recN]) / np.sqrt(len(recN))]
            con_sem = [np.std([expt.lickRateInContext() for expt in habC]) / np.sqrt(len(habC)),
                       np.std([expt.lickRateInContext() for expt in conC]) / np.sqrt(len(conC)),
                       np.std([expt.lickRateInContext() for expt in recC]) / np.sqrt(len(recC))]

            if not plotCon:
                # Remove conditioning experiments if we aren't going to plot them
                neu_means = neu_means[::2]
                con_means = con_means[::2]
                neu_sem = neu_sem[::2]
                con_sem = con_sem[::2]

            index = np.arange(len(neu_means))

            ax.bar(index, neu_means, bar_width, color=colors[0], yerr=neu_sem, label='Neutral')
            ax.bar(index + bar_width, con_means, bar_width, color=colors[1], yerr=con_sem, label='Conditioned')
        else:
            raise Exception('Not implemented')

        ax.set_ylabel('Mean lick rate (Hz)')
        ax.set_xticks(index + bar_width)
        if plotCon:
            ax.set_xticklabels(('Hab', 'Con', 'Rec'))
        else:
            ax.set_xticklabels(('Hab', 'Rec'))
        plt.legend()

        return fig
