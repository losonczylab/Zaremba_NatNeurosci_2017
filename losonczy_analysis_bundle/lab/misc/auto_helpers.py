"""Helper functions mostly used by auto scripts"""

import fnmatch

import os
import time
from datetime import datetime
import cPickle as pickle
from collections import defaultdict

import lab.classes
import lab.classes.exceptions as exc

from xml.etree import ElementTree
from distutils.version import LooseVersion


def behavior_file_check(exptGrps, filename, overwrite_old=False):
    """Checks behavior data for all trials in all exptGrps to see if any
    are newer than a particular file.
    Used to see if summary pdfs need to be updated.

    Returns True if there is behavior data newer than 'filename'

    If overwrite_old is > 0, also returns True if 'filename' is older than
    that X hours.

    """

    if not os.path.isfile(filename):
        return True
    mod_time = os.path.getmtime(filename)
    if overwrite_old:
        now = time.time()
        if now - mod_time > overwrite_old * 60 * 60:
            return True
    for exptGrp in exptGrps:
        for expt in exptGrp:
            for trial in expt.findall('trial'):
                behavior_file = trial.behaviorDataPath()
                if os.path.isfile(behavior_file) and \
                        os.path.getmtime(behavior_file) > mod_time:
                    return True
    return False


def imaging_file_check(
        exptGrps, channel, label, demixed, path, overwrite_old=False):
    """Checks all imaging data in exptGrps to see if any are newer than a
    particular file.
    Used to see if summary pdfs need to be updated.

    Returns True if imaging data has been updated since 'path' was last
    modified.

    If overwrite_old is > 0, also returns True if 'path' is older than
    that X hours.

    """
    if not os.path.isfile(path):
        return True
    mod_time = os.path.getmtime(path)
    if overwrite_old:
        now = time.time()
        if now - mod_time > overwrite_old * 60 * 60:
            return True
    file_modtime = datetime.fromtimestamp(mod_time)
    file_modtime = datetime.strftime(file_modtime, '%Y-%m-%d-%Hh%Mm%Ss')
    for exptGrp in exptGrps:
        if exptGrp.updateTime(
                channel=channel, label=label, demixed=demixed) > file_modtime:
            return True
    return False


def channels_to_process(expt_list):
    # which channels have place field data?
    channels_to_process = set()
    for expt in expt_list:
        for channel in expt.imaging_dataset().channel_names:
            if expt.hasPlaceFieldsFile(channel=channel):
                channels_to_process.add(channel)
    return channels_to_process


def labels_to_process(expt_list, channel, demixed, check_place=False):
    # which labels are contained within the channel and have the correct
    # demixed condition?
    labels_to_process = set()
    for expt in expt_list:
        try:
            if check_place:
                with open(
                        expt.placeFieldsFilePath(channel=channel), 'rb') as f:
                    pkl_data = pickle.load(f)
            else:
                with open(expt.dfofFilePath(channel=channel), 'rb') as f:
                    pkl_data = pickle.load(f)
        except (exc.NoSimaPath, exc.NoTSeriesDirectory, IOError,
                pickle.UnpicklingError):
            continue
        else:
            if check_place:
                demixed_key = 'demixed' if demixed else 'undemixed'
            else:
                demixed_key = 'demixed_traces' if demixed else 'traces'
            for label in pkl_data:
                try:
                    pkl_data[label][demixed_key]
                except KeyError:
                    continue
                else:
                    labels_to_process.add(label)
    return labels_to_process


def condition_label_text(exptGrps, by_mouse=False):
    text = ''
    for exptGrp in exptGrps:
        text += exptGrp.label() + '\n'
        _, conditions = exptGrp.condition_label(by_mouse=by_mouse)
        if by_mouse:
            all_conditions = defaultdict(list)
            for mouse in conditions:
                for condition, l in conditions[mouse].iteritems():
                    all_conditions[condition].append(l)
            conditions = all_conditions
        for condition, l in sorted(conditions.iteritems()):
            text += ' {}: {}\n'.format(condition, l)
        text += '\n'
    return text


# def imaging_channel(x):
#     """
#     Determines the appropriate channel name to use as a key for calling up
#     signals
#     """

#     return 'Ch2'


def locate(pattern, root=os.curdir, ignore=None, max_depth=None):
    """Locate all files matching supplied filename pattern
       in and below supplied root directory."""
    if ignore is None:
        ignore = []
    root = os.path.abspath(root)
    for path, dirs, files in os.walk(os.path.abspath(root)):
        if (max_depth is None) or \
                (path.count(os.sep) - root.count(os.sep) <= max_depth):
            for filename in fnmatch.filter(files, pattern):
                dirs[:] = [dn for dn in dirs
                           if os.path.join(path, dn) not in ignore]
                yield os.path.join(path, filename)


def get_prairieview_version(xml_filepath):
    """Return Prairieview version number"""
    for _, elem in ElementTree.iterparse(xml_filepath, events=("start",)):
        if elem.tag == 'PVScan':
            return LooseVersion(elem.get('version'))


def get_element_size_um(xml_filepath, prairie_version):
    """Determine the size in um of x and y in order to store it with the
    data. The HDF5 plugin for ImageJ will read this metadata"""
    if prairie_version >= LooseVersion('5.2'):
        for _, elem in ElementTree.iterparse(xml_filepath):
            if elem.get('key') == 'micronsPerPixel':
                for value in elem.findall('IndexedValue'):
                    if value.get('index') == "XAxis":
                        x = float(value.get('value'))
                    elif value.get('index') == "YAxis":
                        y = float(value.get('value'))
                return (1, y, x)
    else:
        for _, elem in ElementTree.iterparse(xml_filepath):
            if elem.tag == 'PVStateShard':
                for key in elem.findall('Key'):
                    if key.get('key') == 'micronsPerPixel_XAxis':
                        x = float(key.get('value'))
                    elif key.get('key') == 'micronsPerPixel_YAxis':
                        y = float(key.get('value'))
                return (1, y, x)
    print('Unable to identify element size, returning default value')
    return (1, 1, 1)
