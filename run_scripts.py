"""Helper script to configure paths and run individual figure scripts.

All figures are from Zaremba et al. Nature Neuroscience 2017.

"""

import subprocess
import os
import sys
import ConfigParser
from collections import OrderedDict

CONFIG_PATH = 'paths.cfg'

fig_text = """
Main figures
============
1  - Goal-oriented learning (GOL) task performance
2  - Place cell intro
3  - Imaging and performance correlation
4  - Context change and random foraging (RF) task
5  - Goal zone place cell enrichment
6  - Wild-type place field shift parameter fits
7  - Wild-type enrichment model
8  - Df(16)A parameter fits and enrichment model

Supplemental figures
====================
S1  - Task performance for each mouse
S2  - Baseline behavior comparison
S4  - Additional place cell metrics
S6  - Performance and stability by task Condition
S7  - Muscimol inactivation of hippocampus
S8  - Spatial tuning on cue-free belt
S10 - Possible enrichment methods
S11 - Evidence for latent spatial tuning
S12 - Enrichment model for all Conditions
S13 - Enrichment model parameter swap
"""

number_filename_map = OrderedDict([
    ('1', 'Fig1_GOL_task_performance.py'),
    ('2', 'Fig2_place_cell_intro.py'),
    ('3', 'Fig3_imaging_performance_correlation.py'),
    ('4', 'Fig4_context_change_and_random_foraging.py'),
    ('5', 'Fig5_goal_enrichment.py'),
    ('6', 'Fig6_WT_parameter_fits.py'),
    ('7', 'Fig7_WT_enrichment_model.py'),
    ('8', 'Fig8_Df_model.py'),
    ('S1', 'FigS1_performance_by_mouse.py'),
    ('S2', 'FigS2_behavior_compare.py'),
    ('S4', 'FigS4_additional_place_cell_metrics.py'),
    ('S6', 'FigS6_stability_by_condition.py'),
    ('S7', 'FigS7_muscimol_inactivation.py'),
    ('S8', 'FigS8_cue_free_spatial_tuning.py'),
    ('S10', 'FigS10_possible_enrichment_methods.py'),
    ('S11', 'FigS11_latent_tuning.py'),
    ('S12', 'FigS12_all_conditions_enrichment_model.py'),
    ('S13', 'FigS13_model_swap_parameters.py')
])


def check_lab():
    try:
        import lab
    except ImportError:
        print('Unable to import LAB module, attempting to install.')
        old_cwd = os.getcwd()
        os.chdir('losonczy_analysis_bundle')
        subprocess.call(
            ['python', 'setup.py', 'install', '--user'])
        os.chdir(old_cwd)
    else:
        print('LAB module already installed.')


def check_config():
    config = ConfigParser.RawConfigParser()
    config.read(CONFIG_PATH)

    if not config.has_section('user'):
        reconfigure()


def reconfigure():
    config = ConfigParser.RawConfigParser()
    config.read(CONFIG_PATH)

    if 'user' not in config.sections():
        config.add_section('user')

    data_path = raw_input(
        'Enter path to data: [{}] '.format(config.get('user', 'data_path')))
    config.set(
        'user', 'data_path',
        data_path if len(data_path) else config.get('user', 'data_path'))

    save_path = raw_input(
        'Enter figure save path: [{}] '.format(
            config.get('user', 'save_path')))
    config.set(
        'user', 'save_path',
        save_path if len(save_path) else config.get('user', 'save_path'))

    config.write(open(CONFIG_PATH, 'w'))


def run_script(fig_number):
    print('Running script for Figure {}: {}'.format(
        fig_number, number_filename_map[fig_number]))
    subprocess.call(
        [sys.executable,
         os.path.join('scripts', number_filename_map[fig_number])])


def main():
    check_lab()
    check_config()
    while True:
        print(fig_text)
        input_text = raw_input(
            'Enter figure number, (a)ll, (r)econfigure, or (q)uit: ')
        if input_text == 'r':
            reconfigure()
        elif input_text == 'q':
            break
        elif input_text == 'a':
            print('Generating all figures, this will take a long time ' +
                  '(several hours).')
            for fig in number_filename_map.keys():
                run_script(fig)
        elif input_text in number_filename_map:
            run_script(input_text)


if __name__ == '__main__':
    main()
