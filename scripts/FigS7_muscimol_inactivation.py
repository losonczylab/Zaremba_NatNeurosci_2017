"""Figure S7 - Muscimol inactivation"""

FIG_FORMAT = 'svg'

import matplotlib as mpl
if FIG_FORMAT == 'svg':
    mpl.use('agg')
elif FIG_FORMAT == 'pdf':
    mpl.use('pdf')
elif FIG_FORMAT == 'interactive':
    mpl.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import os

import lab
import lab.analysis.reward_analysis as ra
import lab.plotting as plotting
import lab.misc as misc

import Df16a_analysis as df

save_dir = df.fig_save_dir
filename = 'FigS7_muscimol_inactivation.{}'.format(FIG_FORMAT)

mus_json = os.path.join(df.expt_sets_path, 'muscimol_expts.json')
sal_json = os.path.join(df.expt_sets_path, 'saline_expts.json')


def main():
    expts = lab.ExperimentSet(
        os.path.join(df.metadata_path, 'expt_metadata.xml'),
        behaviorDataPath=os.path.join(df.data_path, 'behavior'),
        dataPath=os.path.join(df.data_path, 'imaging'))

    sal_grp = lab.classes.HiddenRewardExperimentGroup.from_json(
        sal_json, expts, label='saline to muscimol')
    mus_grp = lab.classes.HiddenRewardExperimentGroup.from_json(
        mus_json, expts, label='muscimol to saline')

    fig = plt.figure(figsize=(8.5, 11))
    gs = plt.GridSpec(1, 1, top=0.9, bottom=0.7, left=0.1, right=0.4)
    ax = fig.add_subplot(gs[0, 0])

    for expt in mus_grp:
        if 'saline' in expt.get('drug'):
            expt.attrib['drug_condition'] = 'reversal'
        elif 'muscimol' in expt.get('drug'):
            expt.attrib['drug_condition'] = 'learning'
    for expt in sal_grp:
        if 'saline' in expt.get('drug'):
            expt.attrib['drug_condition'] = 'learning'
        elif 'muscimol' in expt.get('drug'):
            expt.attrib['drug_condition'] = 'reversal'

    plotting.plot_metric(
        ax, [sal_grp, mus_grp], metric_fn=ra.fraction_licks_in_reward_zone,
        label_groupby=False, plotby=['X_drug_condition'],
        plot_method='swarm', rotate_labels=False,
        activity_label='Fraction of licks in reward zone',
        colors=sns.color_palette('deep'), plot_bar=True)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_ylim(top=0.4)
    ax.set_xticklabels(['Days 1-3', 'Day 4'])

    sns.despine(fig)
    ax.set_title('')
    ax.set_xlabel('')

    misc.save_figure(
        fig, filename, save_dir=save_dir)

    plt.close('all')

if __name__ == '__main__':
    main()
