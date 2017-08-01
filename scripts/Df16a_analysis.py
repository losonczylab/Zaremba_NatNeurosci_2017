"""Initialize experiment sets and parameters for Df(16)A analysis."""

import os.path
import pandas as pd
import ConfigParser

import lab
import lab.classes.exceptions as exc
import lab.plotting as plotting

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import seaborn.apionly as sns
sns.set_style("ticks")
sns.set_context(rc={'lines.linewidth': 1,
                    'axes.titlesize': 7,
                    'axes.labelsize': 'medium',
                    'xtick.labelsize': 'medium',
                    'ytick.labelsize': 'medium'})

import matplotlib as mpl
# mpl.rcParams['font.sans-serif'].append(u'DejaVu Sans')
mpl.rcParams.update({'lines.linewidth': 1,
                     'axes.titlesize': 9,
                     'axes.labelsize': 9,
                     'xtick.labelsize': 7,
                     'ytick.labelsize': 7,
                     'mathtext.fontset': 'dejavuserif',
                     'font.size': 9,
                     'pdf.fonttype': 42,
                     'xtick.major.size': 3,
                     'xtick.major.pad': 2,
                     'ytick.major.size': 3,
                     'ytick.major.pad': 2,
                     'svg.fonttype': 'none',  # path
                     'font.sans-serif': ['Liberation Sans', 'DejaVu Sans'],
                     'font.serif': ['Liberation Serif', 'DejaVu Serif'],
                     'font.monospace': ['Liberation Mono', 'DejaVu Sans Mono']
                     })

#
# Constants/parameters/defaults
#
CONFIG_PATH = os.path.normpath(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), '..', 'paths.cfg'))
config = ConfigParser.RawConfigParser()
config.read(CONFIG_PATH)

data_path = config.get('user', 'data_path')
fig_save_dir = config.get('user', 'save_path')
metadata_path = os.path.join(data_path, 'metadata')
expt_sets_path = os.path.normpath(os.path.join(os.path.dirname(
    os.path.realpath(__file__)), '..', 'expt_sets'))

IMAGING_LABEL = 'soma'

# ROI filters
WT_filter = lambda roi: 'red' not in roi.tags
Df_filter = lambda roi: 'red' not in roi.tags

# Colors
WT_color = sns.xkcd_rgb['dark sky blue']  # sns.xkcd_rgb['dark blue grey']
Df_color = sns.xkcd_rgb['dark red']
colors = [WT_color, Df_color]

WT_marker = 'o'
Df_marker = 's'
markers = (WT_marker, Df_marker)

WT_ls = '-'
Df_ls = '--'  # (0, (5, 1))  # dense dashed line
linestyles = (WT_ls, Df_ls)

WT_hatch = None
Df_hatch = '/'
hatchstyles = (WT_hatch, Df_hatch)

WT_label = r'$WT$'
Df_label = r'$Df(16)A^{+/-}$'
labels = (WT_label, Df_label)

# All experiment sets
WT_behavior_set = os.path.join(expt_sets_path, 'WT_GOL_behavior_expts.json')
WT_imaging_set = os.path.join(expt_sets_path, 'WT_GOL_imaging_expts.json')
Df_behavior_set = os.path.join(expt_sets_path, 'Df_GOL_behavior_expts.json')
Df_imaging_set = os.path.join(expt_sets_path, 'Df_GOL_imaging_expts.json')

WT_RF_set = os.path.join(expt_sets_path, 'WT_RF_expts.json')
Df_RF_set = os.path.join(expt_sets_path, 'Df_RF_expts.json')


def combine_dataframes(data, include_columns=()):
    if isinstance(data, dict):
        data_found = False
        for outer_key in data:
            if isinstance(data[outer_key], dict) and \
                    sorted(data[outer_key].keys()) == ['dataframe', 'shuffle']:
                data_found = True
                break
        if data_found:
            dataframe_dfs, shuffle_dfs, labels = [], [], []

            for label in data:
                # Keep track of keys/labels so we can pop them off later.
                # You can't remove keys from a dict while you are iterating
                # over it.
                labels.append(label)
                data[label]['dataframe']['label'] = label
                dataframe_dfs.append(data[label]['dataframe'])
                try:
                    data[label]['shuffle']['label'] = label
                except:
                    pass  # ?
                else:
                    shuffle_dfs.append(data[label]['shuffle'])
            for key in labels:
                data.pop(key)
            data['dataframe'] = pd.concat(dataframe_dfs, ignore_index=True)
            if len(shuffle_dfs):
                data['shuffle'] = pd.concat(shuffle_dfs, ignore_index=True)
            else:
                data['shuffle'] = None

        for key in data:
            combine_dataframes(data[key], include_columns)
    elif isinstance(data, pd.DataFrame):
        for col in include_columns:
            try:
                plotting.prepare_dataframe(data, [col])
            except (exc.InvalidDataFrame, KeyError):
                pass


def rename_columns(data, name_map):
    if isinstance(data, dict):
        for key in data:
            rename_columns(data[key], name_map)
    elif isinstance(data, pd.DataFrame):
        data.rename(columns=name_map, inplace=True)


def reverse_values(data, cols=()):
    if isinstance(data, dict):
        for key in data:
            reverse_values(data[key], cols)
    elif isinstance(data, pd.DataFrame):
        for col in cols:
            if col in data:
                vals = sorted(data[col].unique())
                data[col + '_rev'] = data[col].map(
                    {val: rev_val for val, rev_val in zip(vals, vals[::-1])})


def alpha_to_num(data, cols=()):
    if isinstance(data, dict):
        for key in data:
            alpha_to_num(data[key], cols)
    elif isinstance(data, pd.DataFrame):
        for col in cols:
            if col in data:
                vals = sorted(data[col].unique())
                data[col + '_num'] = data[col].map(
                    {val: num for num, val in enumerate(vals)})


def label_rois(data):
    if isinstance(data, dict):
        for key in data:
            label_rois(data[key])
    elif isinstance(data, pd.DataFrame):
        for pre in ('', 'first_', 'second_'):
            try:
                data[pre + 'roi_label'] = zip(
                    data[pre + 'mouseID'],
                    data[pre + 'uniqueLocationKey'],
                    data[pre + 'roi_id'])
            except KeyError:
                pass


def loadExptGrps(mouse_set):

    expts = lab.ExperimentSet(
        os.path.join(metadata_path, 'expt_metadata.xml'),
        behaviorDataPath=os.path.join(data_path, 'behavior'),
        dataPath=os.path.join(data_path, 'imaging'))

    mouse_sets = {}
    mouse_sets['GOL'] = ('jz096', 'jz097', 'jz098', 'jz100', 'jz101', 'jz102',
                         'jz106', 'jz113', 'jz114', 'jz121', 'jz135', 'jz136')
    mouse_sets['RF'] = ('jz049', 'jz051', 'jz052', 'jz053', 'jz054', 'jz058',
                        'jz059', 'jz060', 'jz064', 'jz066', 'jz067')

    if mouse_set not in mouse_sets:
        raise ValueError('Unrecognized mouse set')

    exptGrps = {}
    WT = []
    Df = []
    for mouse in [expts.grabMouse(m) for m in mouse_sets[mouse_set]]:
        genotype = mouse.get('genotype').lower()
        if 'df16ap' in genotype:
            Df.append(mouse)
        elif 'df16an' in genotype:
            WT.append(mouse)

    exptGrps['WT_mice'] = WT
    exptGrps['Df_mice'] = Df

    if mouse_set == 'GOL':
        exptGrps['WT_hidden_behavior_set'] = \
            lab.classes.HiddenRewardExperimentGroup.from_json(
                WT_behavior_set, expts, label=WT_label)
        exptGrps['Df_hidden_behavior_set'] = \
            lab.classes.HiddenRewardExperimentGroup.from_json(
                Df_behavior_set, expts, label=Df_label)

        exptGrps['WT_place_set'] = \
            lab.classes.pcExperimentGroup.from_json(
                WT_imaging_set, expts, imaging_label=IMAGING_LABEL,
                label=WT_label)
        exptGrps['Df_place_set'] = \
            lab.classes.pcExperimentGroup.from_json(
                Df_imaging_set, expts, imaging_label=IMAGING_LABEL,
                label=Df_label)

    elif mouse_set == 'RF':
        exptGrps['WT_place_set'] = \
            lab.classes.pcExperimentGroup.from_json(
                WT_RF_set, expts, imaging_label=IMAGING_LABEL,
                label=WT_label).pair()
        exptGrps['Df_place_set'] = \
            lab.classes.pcExperimentGroup.from_json(
                Df_RF_set, expts, imaging_label=IMAGING_LABEL,
                label=Df_label).pair()

    return exptGrps
