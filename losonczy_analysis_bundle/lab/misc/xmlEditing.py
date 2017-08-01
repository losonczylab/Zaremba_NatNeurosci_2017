import os
import shutil
import csv
from lxml import etree
from numpy import around, mean, diff

"""
General xml formatting functions
"""


def writeFormattedXml(filename, root):
    output = etree.tostring(root, pretty_print=True).replace(
        '&#10', '&#xa').replace('&#13', '&#xd').replace(
        '/><', '/>\n  <').replace('&#9;', '&#x9;').replace('&gt;', '>')
    with open(filename, 'w') as f:
        f.write(output)


def reformatXml(filename):
    with open(filename) as f:
        root = etree.parse(f)
    writeFormattedXml(filename, root)

"""
Xml modification functions
"""


def copyMice(source_filename, target_filename, mousetags):
    """Copies all mice with mouseID containing any of 'mousetags' from
    source_filename to new xml 'target_filename'

    """
    if os.path.isfile(target_filename):
        raise ValueError('Target filename already exists')

    with open(source_filename) as f:
        source = etree.parse(f)

    new_xml = etree.Element("BehaviorExperiments")

    for mouse in source.getroot().findall('mouse'):
        if any([tag.lower() in mouse.get('mouseID').lower()
                for tag in mousetags]):
            new_xml.append(mouse)

    writeFormattedXml(target_filename, new_xml)


def normTSeriesFilename(filename, mousetag):
    """Normalize the tSeriesFilenames for all mice matching mousetag"""
    with open(filename) as f:
        root = etree.parse(f)
    for mouse in root.getroot().findall('mouse'):
        if mousetag in mouse.get('mouseID'):
            for expt in mouse.findall('experiment'):
                if expt.get('tSeriesDirectory'):
                    expt.set('tSeriesDirectory',
                             os.path.normpath(expt.get('tSeriesDirectory')))
    writeFormattedXml(filename, root)


def setGenotype(filename, genotype, mouseList):
    """Set the genotype of each mouse whose mouseID is in the list"""
    with open(filename) as f:
        root = etree.parse(f)
    for mouse in root.getroot().getchildren():
        if mouse.get('mouseID') in mouseList:
            mouse.set('genotype', genotype)
    writeFormattedXml(filename, root)


def setExperimentSubtype(filename, subtype, exptList):
    """Set the genotype of each mouse whose mouseID is in the list"""
    with open(filename) as f:
        root = etree.parse(f)

    start_times = [expt.get('startTime') for expt in exptList]
    mouse_list = list(set([expt.parent.get('mouseID') for expt in exptList]))

    for mouse in root.getroot().getchildren():
        if mouse.get('mouseID') in mouse_list:
            for expt in mouse.findall('experiment'):
                if expt.get('startTime') in start_times:
                    if expt.get('experiment_subtype'):
                        f = expt.get('experiment_subtype') + ', ' + subtype
                        expt.set('experiment_subtype', f)
                    else:
                        expt.set('experiment_subtype', subtype)
    writeFormattedXml(filename, root)


def addInjection(filename, virusList, location, date, mouseList):
    """Add a record for the injection of a given virus at a given location for
    each mice in the list

    """
    with open(filename) as f:
        root = etree.parse(f)

    for mouse in root.getroot().getchildren():
        if mouse.get('mouseID') in mouseList:
            injection = etree.Element('injection')
            injection.set('date', date)
            injection.set('location', location)

            for bug in virusList:
                virus = etree.Element('virus')
                virus.set('name', bug)
                injection.insert(0, virus)
            mouse.insert(0, injection)

    writeFormattedXml(filename, root)


def correctInjectionFormat(filename, mouseID):
    """take an entries in the old injection format and convert them to the new
    format for injections

    """
    with open(filename) as f:
        root = etree.parse(f)

    mouse = grabMouse(root, mouseID)
    inj = mouse.findall('injection')

    virusName = inj[0].get('virus')
    injLocation = inj[0].get('location')
    inj[0].getparent().remove(inj[0])

    writeFormattedXml(filename, root)
    addInjection(filename, [virusName], injLocation, '', mouse.get('mouseID'))


def correctAttribute(filename, objectTag, attribute, termsList, correctTerm,
                     startTime=None):
    """'startTime' argument will only work with objectTag==experiment"""
    with open(filename) as f:
        root = etree.parse(f)
    for obj in root.iter(objectTag):
        if obj.get(attribute) in termsList \
                and (startTime is None or obj.get('startTime') > startTime):
            obj.set(attribute, correctTerm)
    writeFormattedXml(filename, root)


def correctLayerFormat(filename, termsList, correctTerm):
    correctAttribute(filename, 'experiment', 'imagingLayer', termsList,
                     correctTerm, None)
    # with open(filename) as f:
    #     root = etree.parse(f)
    # for exp in root.findall("//experiment"):
    #     if exp.get('imagingLocation') in termsList:
    #         exp.set('imagingLocation', correctTerm)
    # writeFormattedXml(filename, root)


def updateDoubleStim(filename):
    with open(filename) as f:
        root = etree.parse(f)

    for mouse in root.getroot().findall('mouse'):
        for experiment in mouse.findall('experiment'):
            if experiment.get('experimentType') == 'doubleStimulus':
                if 'comments' in experiment.attrib.keys():
                    if experiment.get('comments').count('200ms') \
                            and not experiment.get('comments').count('50ms'):
                        experiment.set('stimulusDuration', '0.2')
                    elif experiment.get('comments').count('50ms') \
                            and not experiment.get('comments').count('200ms'):
                        experiment.set('stimulusDuration', '0.05')
                    else:
                        print experiment.get('comments')

    writeFormattedXml(filename, root)


def addStimDurations(filename):
    with open(filename) as f:
        root = etree.parse(f)
    for mouse in root.getroot().findall('mouse'):
        for experiment in mouse.findall('experiment'):
            if experiment.get('experimentType') == 'salience':
                if 'comments' in experiment.attrib.keys():
                    d = experiment.get('airpuffDuration')
                    if d:
                        for trial in experiment.findall('trial'):
                            if trial.get('stimulus') == 'air':
                                trial.set('duration', d)
    writeFormattedXml(filename, root)


def batchAttributeChange(filename, objectTag, attribute, oldValue, newValue):
    """For every object with tag objectTag in the xml file, if the attribute has
    value oldValue, then change it to newValue"""
    with open(filename) as f:
        root = etree.parse(f)
    for obj in root.iter(objectTag):
        if obj.get(attribute) == oldValue:
            obj.set(attribute, newValue)
    writeFormattedXml(filename, root)


def grabMouse(root, mouseID):
    return root.find("./*[@mouseID='" + mouseID + "']")


def grabExpt(root, mouseID, startTime):
    mouse = grabMouse(root, mouseID)
    return mouse.find("./*[@startTime='" + startTime + "']")


def mergeBehaviorData(first_file, second_file):
    """Helper function to merge two behaviorData files. Does not modify the XML."""

    if not os.path.isfile(first_file):
        print 'Unable to locate {0}, exiting'.format(first_file)
        return
    if not os.path.isfile(second_file):
        print 'Unable to locate {0}, exiting'.format(second_file)
        return
    final_file = first_file.replace('.csv', '') + '_merged.csv'
    # Determine frame rate and end time
    time = []
    with open(first_file, 'r') as first:
        reader = csv.reader(first, delimiter='\t')
        reader.next()
        for line in reader:
            time.append(float(line[0]))
    timeStep = around(mean(diff(time)), 7)
    lastTime = around(time[-1], 7)
    count = 1
    shutil.copyfile(first_file, final_file)
    with open(final_file, 'a') as final:
        writer = csv.writer(final, delimiter='\t')
        with open(second_file, 'r') as second:
            reader = csv.reader(second, delimiter='\t')
            reader.next()
            for line in reader:
                line[0] = str(lastTime + count * timeStep)
                writer.writerow(line)
                count += 1
    print 'Merged {0} and {1} into {2}.'.format(
        os.path.basename(first_file), os.path.basename(second_file),
        os.path.basename(final_file))
    shutil.move(first_file, first_file.replace('csv', 'orig'))
    print 'Renamed {0} to {1}'.format(
        os.path.basename(first_file),
        os.path.basename(first_file).replace('csv', 'orig'))
    shutil.move(second_file, second_file.replace('csv', 'orig'))
    print 'Renamed {0} to {1}'.format(os.path.basename(
        second_file), os.path.basename(second_file).replace('csv', 'orig'))

    return final_file


def mergeOrphanedBehaviorData(filename, path, directory='/data/BehaviorData', merge_forward=True):
    """Takes a relative file path (as output from printOrphanedBehaviorData) and merges it with an appropriate trial
    If merge_forward is True, merges file with the next trial in time, if false with the previous trial (only forward is implemented at the moment)"""
    with open(filename) as f:
        root = etree.parse(f)
    mouseID = os.path.dirname(path)
    file_time = os.path.basename(path).replace(
        '.csv', '').replace(mouseID, '').strip('_')
    mouse = grabMouse(root, mouseID)
    foundTrial = None
    for expt in mouse.findall('experiment'):
        if expt.get('startTime') <= file_time \
                and expt.get('stopTime') >= file_time:
            for trial in expt.findall('trial'):
                if merge_forward:
                    # Find the first trial at a later time than file_time
                    if trial.get('time') > file_time:
                        foundTrial = trial
                        break
                else:
                    raise NotImplemented()
            if foundTrial is not None:
                break
    if foundTrial is None:
        print 'No valid trial found for {0}'.format(os.path.basename(path))
        return
    else:
        first_file = os.path.join(directory, path)
        second_file = os.path.join(directory, foundTrial.get('filename'))
        final_file = mergeBehaviorData(first_file, second_file)

        trial.set('filename', os.path.join(mouseID, os.path.basename(final_file)))
        trial.set('time', os.path.basename(final_file).replace('_merged.csv', '').replace(mouseID, '').strip('_'))
        writeFormattedXml(filename, root)
        print 'Updated xml, modified {0}: experiment {1}'.format(
            mouseID, trial.getparent().get('startTime'))


def mergeSplitTrials(filename, mouseID, startTime, directory='/data/BehaviorData'):
    """Merge two trials for the give experiment"""
    with open(filename) as f:
        root = etree.parse(f)
    expt = grabExpt(root, mouseID, startTime)
    trials = expt.findall('trial')
    if len(trials) != 2:
        print 'Number of trials must be exactly 2, found {}'.format(len(trials))
        return
    first_file = os.path.join(directory, trials[0].get('filename'))
    second_file = os.path.join(directory, trials[1].get('filename'))

    final_file = mergeBehaviorData(first_file, second_file)

    if not final_file:
        print "Merge failed, exiting. XML not modified"
        return

    expt.remove(trials[1])
    trials[0].set('filename', os.path.join(mouseID, os.path.basename(final_file)))
    trials[0].set('time', os.path.basename(final_file).replace(
        '_merged.csv', '').replace(mouseID, '').strip('_'))
    writeFormattedXml(filename, root)
    print 'Updated xml, modified {0}: experiment {1}'.format(mouseID, startTime)


def addBeltTag(filename, value, mouseID, start_date=None, stop_date=None):
    """
    Adds/replace the belt tag on all experiments for the given mouse within the given date range.
    start_date or stop_date can be None, and will not be used as a limit if so
    """
    with open(filename) as f:
        root = etree.parse(f)
    mouse = grabMouse(root, mouseID)
    if start_date is None:
        start_date = ''
    if stop_date is None:
        stop_date = 'zzzzzzzzzzzzz'
    for experiment in mouse.findall('experiment'):
        if experiment.get('startTime') >= start_date and experiment.get('startTime') <= stop_date:
            experiment.set('belt', value)
    writeFormattedXml(filename, root)

"""
Editing functions for belts.xml
"""


def addBeltToXML(filename, beltID, imageLocation=""):

    with open(filename) as f:
        root = etree.parse(f)

    belt = etree.Element('belt')
    belt.set('beltID', beltID)
    belt.set('imageLocation', imageLocation)

    root.getroot().append(belt)
    writeFormattedXml(filename, root)


def addSegmentToBelt(filename, beltID, material, length, segmentIndex):

    with open(filename) as f:
        root = etree.parse(f)

    segment = etree.Element('segment')
    segment.set('material', str(material))
    segment.set('length_cm', str(length))
    segment.set('ordering', str(segmentIndex))

    #find belt element with this beltID
    for belt in root.getroot().getchildren():
        if belt.get('beltID') == beltID:
            belt.append(segment)

    writeFormattedXml(filename, root)


def addRFIDtagToSegment(filename, beltID, segmentIndex, distFromSegStart=0, lapStart=False):

    with open(filename) as f:
        root = etree.parse(f)

    tag = etree.Element('RFIDtag')
    tag.set('distanceFromSegmentStart', str(distFromSegStart))
    tag.set('lapStart', str(lapStart))

    for belt in root.getroot().getchildren():
        if belt.get('beltID') == beltID:
            for seg in belt.findall('segment'):
                if seg.get('ordering') == str(segmentIndex):
                    seg.append(tag)

    writeFormattedXml(filename, root)

"""
functions for analyzing accuracy of xml file
"""


def getMiceWithPreviousInjections(filename):
    """list all the mice that have injection info listed"""
    with open(filename) as f:
        root = etree.parse(f)
    mouseList = []
    for inj in root.findall("//injection"):
        mouseList.append(inj.getparent().get('mouseID'))
    return list(set(mouseList))


def printMiceByGenotype(filename):
    with open(filename) as f:
        root = etree.parse(f)
    genotypeDict = {}
    for mouse in root.getroot().findall('mouse'):
        try:
            genotypeDict[mouse.get('genotype')].append(mouse.get('mouseID'))
        except KeyError:
            genotypeDict[mouse.get('genotype')] = [mouse.get('mouseID')]
    for genotype in sorted(genotypeDict.keys()):
        print genotype
        for m in sorted(genotypeDict[genotype]):
            print '\t', m


def printNoGenotypeNoInjectionMice(filename, imagedOnly=False):
    with open(filename) as f:
        root = etree.parse(f)
    for mouse in root.getroot().findall('mouse'):
        if (not imagedOnly) or any([e.get('tSeriesDirectory') for e in mouse.findall('experiment')]):
            if not (mouse.get('genotype') and len(mouse.findall('injection'))):
                print mouse.get('mouseID')


def printAllVariants(filename, objectTag, attribute):
    with open(filename) as f:
        root = etree.parse(f)
    variants = set()
    for obj in root.iter(objectTag):
        variants.add(obj.get(attribute))
    print sorted(variants)


def printOrphanedBehaviorData(filename, directory='/data/BehaviorData', tag=''):
    """Identifies behavior data files not found in the xml, filtered by 'tag'"""
    paths_in_xml = set()
    with open(filename) as f:
        root = etree.parse(f)
    for mouse in root.getroot().findall('mouse'):
        for expt in mouse.findall('experiment'):
            for trial in expt.findall('trial'):
                if 'filename' in trial.keys():
                    if tag in trial.get('filename'):
                        paths_in_xml.add(trial.get('filename'))
    paths_in_directory = set()
    for dirpath, _, filenames in os.walk(directory):
        for file in filenames:
            if 'csv' in file and tag in file:
                paths_in_directory.add(os.path.join(os.path.basename(dirpath), file))
    for orphan in paths_in_directory.difference(paths_in_xml):
        print orphan
    return list(paths_in_directory.difference(paths_in_xml))


def printMultipleTrialExperiments(filename, experimentType, mouseTag=''):
    """Identifies experiments that have multiple trials, when they should only have 1"""
    bad_experiments = []
    with open(filename) as f:
        root = etree.parse(f)
    for mouse in root.getroot().findall('mouse'):
        if mouseTag in mouse.get('mouseID'):
            for expt in mouse.findall('experiment'):
                if expt.get('experimentType') == experimentType \
                        and len(expt.findall('trial')) > 1:
                    bad_experiments.append(
                        (mouse.get('mouseID'), expt.get('startTime')))

    for mouseID, startTime in bad_experiments:
        print '{}: {}'.format(mouseID, startTime)
    return bad_experiments


def tSeriesRepeats(filename):
    """Check for multiple experiments with the same tSeriesDirectory"""
    with open(filename) as f:
        root = etree.parse(f)
    tsdDict = {}
    for mouse in root.getroot().findall('mouse'):
        for expt in mouse.findall('experiment'):
            tsd = expt.get('tSeriesDirectory')
            if tsd:
                try:
                    tsdDict[tsd].append(expt)
                except KeyError:
                    tsdDict[tsd] = [expt]
    multipleList = []
    for k, v in tsdDict.iteritems():
        if len(v) > 1:
            multipleList.append(k)
    return multipleList


def get_all_variants(filename, objectTag, attribute, filter_attrs=None):
    if filter_attrs is None:
        filter_attrs = {}
    with open(filename) as f:
        root = etree.parse(f)
    variants = []
    for obj in root.iter(objectTag):
        for key, value in filter_attrs.iteritems():
            if obj.get(key) != value:
                continue
            variants.append(obj.get(attribute))
    print sorted(variants)
