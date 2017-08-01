"""Infers line phase offset of rows of imaging data acquired with the
resonant scanner"""

import os
import h5py
import itertools as it
import numpy as np
from collections import Counter
from distutils.version import LooseVersion
from xml.etree import ElementTree

from sys import path
path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..', '..',
    'automaticScripts'))
from lab.misc.auto_helpers import get_prairieview_version

MAX_OFFSET = 10

FRAMES_PER_BIN = 1000


def is_resonant_data(prairie_xml):
    """Determine if the data was imaged with the resonant scanners"""
    prairie_version = get_prairieview_version(prairie_xml)
    if prairie_version >= LooseVersion('5.2'):
        for _, elem in ElementTree.iterparse(prairie_xml):
            if elem.get('key') == 'activeMode':
                return elem.get('value') == 'ResonantGalvo'
    else:
        for _, elem in ElementTree.iterparse(prairie_xml):
            if elem.tag == 'PVStateShard':
                for key in elem.findall('Key'):
                    if key.get('key') == 'activeMode':
                        return key.get('description') == 'ResonantGalvo'


def identify_resonant_phase_offset(h5_file_path):

    f = h5py.File(h5_file_path, 'r')
    data = f['imaging']

    channels = f['imaging'].attrs['channel_names'].tolist()
    try:
        ch_index = channels.index('Ch2')
    except ValueError:
        ch_index = len(channels) - 1

    cnt = Counter()

    frame_bounds = np.linspace(
        0, data.shape[0], data.shape[0] / FRAMES_PER_BIN + 1).astype('int')
    if len(frame_bounds) < 2:
        frame_bounds = [0, data.shape[0]]

    for vol_idx, frame_start, frame_end in it.izip(
            it.count(), frame_bounds[:-1], frame_bounds[1:]):
        vol1 = data[frame_start:frame_end, :, 0::2, :, ch_index].mean(0)
        vol2 = data[frame_start:frame_end, :, 1::2, :, ch_index].mean(0)
        best_corr = np.corrcoef([
            vol1[:, :, :-MAX_OFFSET].flatten(),
            vol2[:, :, :-MAX_OFFSET].flatten()])[0, 1]
        best_offset = 0
        for x in range(1, MAX_OFFSET + 1):
            pos_corr = np.corrcoef([
                vol1[:, :, :-MAX_OFFSET].flatten(),
                np.roll(vol2, -x, 2)[:, :, :-MAX_OFFSET].flatten()])[0, 1]
            neg_corr = np.corrcoef([
                np.roll(vol1, -x, 2)[:, :, :-MAX_OFFSET].flatten(),
                vol2[:, :, :-MAX_OFFSET].flatten()])[0, 1]
            if pos_corr > best_corr:
                best_corr = pos_corr
                best_offset = x
            if neg_corr > best_corr:
                best_corr = neg_corr
                best_offset = -x
        cnt[best_offset] += 1

    return cnt.most_common()[0][0]


def identify_phase_offset_array(array):
    """Infers resonant phase offset of a single 2D numpy array"""

    first = array[0::2, :]
    second = array[1::2, :]

    best_corr = np.corrcoef([first[:, :-MAX_OFFSET].flatten(),
                             second[:, :-MAX_OFFSET].flatten()])[0, 1]

    best_offset = 0
    for x in range(1, MAX_OFFSET + 1):
        pos_corr = np.corrcoef([
            first[:, :-MAX_OFFSET].flatten(),
            np.roll(second, -x, 1)[:, :-MAX_OFFSET].flatten()])[0, 1]
        neg_corr = np.corrcoef([
            np.roll(first, -x, 1)[:, :-MAX_OFFSET].flatten(),
            second[:, :-MAX_OFFSET].flatten()])[0, 1]
        if pos_corr > best_corr:
            best_corr = pos_corr
            best_offset = x
        if neg_corr > best_corr:
            best_corr = neg_corr
            best_offset = -x

    return best_offset
