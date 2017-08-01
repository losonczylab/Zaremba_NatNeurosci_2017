"""README for structure of the output data & how to load.

The .csv output files are tab-delimited

The first 3 rows contain the following information:
    * ROW 1: ROI ID's (there is one unique ID for each field-of-view)
    * ROW 2: ROI tags (e.g. 'r' indicates that the ROI was tdTomato positive)
    * ROW 3: Column headers:
        ** Time (s)
        ** Running (boolean for whether the animal was classified as running)
        ** Velocity (cm/s)
        ** Position (normalized)
        ** Licking (boolean for whether a lick was detected)
        ** ...ROI ID... (df/f data)
        ** ...ROI ID... (is active data -- boolean on whether the ROI was
           firing a significant transient at that time point)

Here is some sample Python code for pulling out the data:

>>> import csv
>>> import numpy as np

>>> time_s = []; running = []; velocity = []; position = []; licking = [];
>>> dff_data = []; is_active = [];

>>> with open('path_to_csv', 'rb') as f:
>>>     reader = csv.reader(f, delimiter='\t')
>>>     roi_ids = reader.next()[5:]  # First row is ROI IDs
>>>     tags = reader.next()[5:]  # Second row is ROI tags
>>>     reader.next()  # Pop off third row (column headers)
>>>     for row in reader:
>>>         time_s.append(row[0])
>>>         running.append(row[1])
>>>         velocity.append(row[2])
>>>         position.append(row[3])
>>>         licking.append(row[4])
>>>         dff_data.append(row[5:5 + len(roi_ids)])
>>>         is_active.append(row[5 + len(roi_ids):])

>>> time_s = np.array(time_s).astype(float)
>>> running = np.array(running).astype(int)
>>> velocity = np.array(velocity).astype(float)
>>> position = np.array(position).astype(float)
>>> licking = np.array(licking).astype(int)
>>> dff_data = np.array(dff_data).astype(float).T  # rois x time
>>> is_active = np.array(is_active).astype(int).T  # rois x time

"""

import argparse
import csv
import os
import itertools as it

from lab.classes import ExperimentSet, ExperimentGroup
import lab.analysis.behavior_analysis as ba
import lab.analysis.imaging_analysis as ia

channel = 'Ch2'
label = None

argParser = argparse.ArgumentParser()
argParser.add_argument(
    "xml", action='store', type=str, default='behavior.xml',
    help="name of xml file to parse")
argParser.add_argument(
    "path", action="store", type=str,
    help="Path to store the dumped data")
argParser.add_argument(
    "-m", "--mouse", type=str,
    help="enter a single mouseID string to data dump")
argParser.add_argument(
    "-d", "--directory", action="store", type=str, default='',
    help="All data found below this directory will be dumped")
args = argParser.parse_args()

experimentSet = ExperimentSet(
    '/analysis/experimentSummaries/.clean-code/experiments/' + args.xml,
    '/data/BehaviorData')

if args.mouse is None and args.directory == '':
    raise Exception("Must pass in either a mouse or directory to dump")

if args.mouse:
    miceToAnalyze = [experimentSet.grabMouse(args.mouse)]
else:
    miceToAnalyze = experimentSet.root.findall('mouse')

for mouse in miceToAnalyze:
    exptGrp = ExperimentGroup(
        [expt for expt in mouse.findall('experiment')
         if expt.get('tSeriesDirectory', '') and args.directory
         in expt.get('tSeriesDirectory', '')])
    exptGrp.removeDatalessTrials()

    for expt in exptGrp:
        print('Dumping data for {}: {}'.format(
            expt.parent.get('mouseID'),
            expt.sima_path().split('/')[-1].split('.')[0]))
        imaging_data = expt.imagingData(dFOverF='from_file')
        trans_data = ia.isActive(expt)
        imaging_times = expt.imagingTimes()
        for trial_idx, trial in enumerate(expt.findall('trial')):

            filename = '{}_{}_{}.csv'.format(
                expt.parent.get('mouseID'),
                expt.sima_path().split('/')[-1].split('.')[0],
                trial_idx)
            labels = ['Time(s)']
            data = [imaging_times]
            try:
                running = ba.runningIntervals(
                    trial, imageSync=True, returnBoolList=True)
            except:
                pass
            else:
                labels.append('run')
                data.append([int(run) for run in running])
            try:
                velocity = ba.velocity(trial, imageSync=True)
            except:
                pass
            else:
                labels.append('velocity')
                data.append(velocity)
            try:
                position = trial.behaviorData(
                    imageSync=True)['treadmillPosition']
            except:
                pass
            else:
                labels.append('position')
                data.append(position)
            try:
                licking = ba.lickingIntervals(
                    trial, imageSync=True, returnBoolList=True)
            except:
                pass
            else:
                labels.append('licking')
                data.append([int(lick) for lick in licking])

            data.extend(
                [imag.tolist() for imag in imaging_data[..., trial_idx]])
            data.extend(
                [trans.astype('int').tolist()
                 for trans in trans_data[..., trial_idx]])

            with open(os.path.join(args.path, filename), 'wb') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')

                writer.writerow(
                    [''] * (len(labels) - 1) + ['id'] +
                    [roi.id for roi in expt.rois()])
                writer.writerow(
                    [''] * (len(labels) - 1) + ['tags'] +
                    [''.join(t + ',' for t in sorted(roi.tags))[:-1]
                        for roi in expt.rois()])
                writer.writerow(
                    labels + [roi.label for roi in expt.rois()] + [
                        roi.label for roi in expt.rois()])
                for row in it.izip(*data):
                    writer.writerow(row)

                # writer.writerow(
                #     ['Time(s)', 'run', 'velocity', 'position', 'licking']
                #     + [roi.label for roi in expt.rois()]
                #     + [roi.label for roi in expt.rois()])
                # writer.writerow(
                #     ['', '', '', '', 'id']
                #     + [roi.id for roi in expt.rois()])
                # writer.writerow(
                #     ['', '', '', '', 'tags']
                #     + [''.join(t + ',' for t in sorted(roi.tags))[:-1]
                #         for roi in expt.rois()])
                # for time, run, vel, pos, lick, imag, tran in it.izip(
                #         imaging_times, running, velocity,
                #         position, licking, imaging_data[..., trial_idx].T,
                #         trans_data[..., trial_idx].T):
                #     writer.writerow([time, int(run), vel, pos, int(lick)]
                #                     + imag.tolist()
                #                     + tran.astype('int').tolist())
