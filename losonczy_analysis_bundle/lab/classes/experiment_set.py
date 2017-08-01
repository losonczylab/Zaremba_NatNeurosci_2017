"""ExperimentSet class definitions."""

import os
from copy import deepcopy
import re
from xml.etree import ElementTree

from classes import Mouse, Trial
from lab.classes import database

import experiment


class ExperimentSet:
    """The set of experiment metadata extracted from an XML file.

    TODO: Jack -- Please update the docs here w/ SQL initialization

    Example
    -------

    >>> from lab import ExperimentSet
    >>> expt_set = ExperimentSet(
        '/analysis/experimentSummaries/.clean_code/experiments/behavior_jeff.xml')

    >>> for mouse in expt_set:
    ...    for injection in mouse.findall('injection'):
    ...        for virus in injection.findall('virus'):
    ...            pass
    ...    for expt in mouse.findall('experiment'):
    ...        for drug in expt.findall('drug'):
    ...            pass
    ...        for trial in expt.findall('trial'):
    ...            pass

    Parameters
    ----------
    filename : str
        Path to the xml to parse and use for initialization

    behaviorDataPath : str
        Path to the root where all raw behavior data is stored.
        os.path.join(behaviorDataPath, trial.get('filename')) should point to
        the behavior data for a trial.

    dataPath : str
        A root to prepend to the tSeriesDirectory if you are mounting
        drives remotely. os.path.join(dataPath, expt.get('tSeriesDirectory'))
        should point to the imaging data for an expt.

    removeBad : bool
        Remove experiments that have been marked as "bad" from the experiment
        set. Removes experiments if ERROR, EXCLUDE, or bad-image are in
        expt.get('experimentType'). Removes trials with a missing or empty
        startTime or filename attribute.
    """

    def __init__(
            self, filename, behaviorDataPath='/data/BehaviorData',
            dataPath="/", removeBad=True):
        """Initialize the ExperimentSet."""
        if (re.match('.*sql$', filename)):
            # find set in sql database
            self.doc = database.createExperimentSetRoot(
                experiment_group=filename.rstrip('.sql'))
        else:
            # parse the xml file
            with open(filename) as f:
                self.doc = ElementTree.parse(f)
        self.root = self.doc.getroot()

        # modify the classes of the nodes to add extra functionality
        for node in self.root.iter():
            if node.tag == 'mouse':
                node.__class__ = Mouse
            elif node.tag == 'experiment':
                if node.get('experimentType') == 'salience':
                    node.__class__ = experiment.SalienceExperiment
                elif node.get('experimentType') == 'doubleStimulus':
                    node.__class__ = experiment.DoubleStimulusExperiment
                elif node.get('experimentType') == 'intensityRanges':
                    node.__class__ = experiment.IntensityRangeExperiment
                elif node.get('experimentType') == \
                        'contextualFearConditioning':
                    node.__class__ = experiment.FearConditioningExperiment
                elif node.get('experimentType') == 'toneTraceConditioning':
                    node.__class__ = experiment.toneTraceConditioningExperiment
                elif node.get('experimentType') == 'runTraining':
                    node.__class__ = experiment.RunTrainingExperiment
                elif node.get('experimentType') == 'hiddenRewards':
                    node.__class__ = experiment.HiddenRewardExperiment
                else:
                    node.__class__ = experiment.Experiment
                # initialize the experiment class
                experiment.Experiment.__init__(node)
            elif node.tag == 'trial':
                node.__class__ = Trial

        for p in self.doc.iter():  # add parent pointers
            for c in p:
                c.parent = p

        # add path to the behavior data
        self.root.behaviorDataPath = behaviorDataPath

        # add path to the imaging data
        self.root.dataPath = dataPath

        # add path to the belt xml
        if (not hasattr(self.root, 'beltXmlPath')) or \
                self.root.beltXmlPath is None:
            self.root.beltXmlPath = os.path.dirname(filename) + '/belts.xml'

        if removeBad:
            # Remove experiments that are marked as bad or trials with no
            # data
            for mouse in self.root.findall('mouse'):
                for expt in mouse.findall('experiment'):
                    if expt.get('experimentType').count('ERROR'):
                        mouse.remove(expt)
                    elif expt.get('experimentType').count('EXCLUDE'):
                        mouse.remove(expt)
                    elif expt.get('experimentType').count('bad-image'):
                        mouse.remove(expt)
                    else:
                        for trial in expt.findall('trial'):
                            if not trial.get('time') or not \
                                    trial.get('filename'):
                                expt.remove(trial)
                        if len(expt.findall('trial')) == 0:
                            mouse.remove(expt)

    def __iter__(self):
        """Iterate over the mice in the ExperimentSet."""
        return self.root.iterfind('mouse')

    def grabMouse(self, mouseID):
        """Return the mouse within matching mouseID.

        Parameters
        ----------
        mouseID : str
            mouseID to locate mouse

        Returns
        -------
        mouse : Mouse

        """
        return self.root.find("./*[@mouseID='" + mouseID + "']")

    def grabExpt(self, mouseID, startTime):
        """Return the experiment with matching mouseID and start time.

        Parameters
        ----------
        mouseID : str
            mouseID of corresponding mouse
        starTime : str
            startTime of experiment

        Returns
        -------
        expt : Experiment

        """
        mouse = self.grabMouse(mouseID)
        return mouse.find("./*[@startTime='" + startTime + "']")

    def grabExptByPath(self, path):
        """Return the experiment with matching imaging data location.

        Parameters
        ----------
        path : str
            Match experiment tSeriesDirectory

        Returns
        -------
        expt : Experiment

        """
        return self.root.find(
            "./mouse/experiment[@tSeriesDirectory='" + path + "']")

    def merge(self, other):
        """Merge 2 ExperimentSets together from multiple xml files.

        Parameters
        ----------
        other : ExperimentSet
            Other ExperimentSet to merge in to current one.

        Returns
        -------
        merged : ExperimentSet

        Note
        ----
        Returns a new ExperimentSet, does not merge in place.

        """
        if self.root.dataPath != other.root.dataPath:
            raise ValueError('Different dataPaths, unable to merge')
        if self.root.behaviorDataPath != other.root.behaviorDataPath:
            raise ValueError('Different behaviorDataPaths, unable to merge')
        if self.root.beltXmlPath != other.root.beltXmlPath:
            raise ValueError('Different belt XML paths, unable to merge')

        new_experiment_set = deepcopy(self)
        new_experiment_set.root.extend(other.root)

        return new_experiment_set
