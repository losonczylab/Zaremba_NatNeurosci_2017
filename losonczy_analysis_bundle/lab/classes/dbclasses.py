import os
import re
import copy
import json
import urllib, urllib2
import warnings

from lab.classes import database
from lab.classes import Experiment, Mouse, ExperimentSet, Trial

class dbExperimentSet(ExperimentSet):
    _keyDict = {
        'project_name': 'experiment_group'
    }

    @staticmethod
    def FetchTrials(*args, **kwargs):
        keyDict = {}
        keyDict.update(dbExperimentSet._keyDict)
        keyDict.update(dbMouse._keyDict)
        keyDict.update(dbExperiment._keyDict)

        args = [keyDict.get(k,k) for k in args]
        kwargs = {keyDict.get(k,k):v for k,v in kwargs.iteritems() if v is not
            None}

        trials=database.fetchTrials(*args, **kwargs)
        return [dbExperiment(e) for e in trials]

    @staticmethod
    def FetchMice(*args, **kwargs):
        keyDict = {}
        keyDict.update(dbMouse._keyDict)
        keyDict.update(dbExperiment._keyDict)

        args = [keyDict.get(k,k) for k in args]
        kwargs = {keyDict.get(k,k):v for k,v in kwargs.iteritems() if v is not
            None}

        mice = database.fetchMice(*args, **kwargs)
        return [dbMouse(mouse) for mouse in mice]

    @staticmethod
    def FetchValues(attr, project_name=None):
        return database.fetchAttributeValues(attr, project_name=project_name)

    def __init__(self, project_name=None):
        self.dataPath = '/'
        self.behaviorDataPath = '/'
        self.beltXmlPath = \
            '/analysis/experimentSummaries/.clean-code/experiments/belts.xml'
        self.root = self

        if project_name is not None and re.search('.sql$', project_name):
            project_name = project_name.split('.sql')[0]
        self._project_name = project_name

    @property
    def parent(self):
        return self

    def fetchTrials(self, *args, **kwargs):
        kwargs['experiment_group'] = self._project_name
        return self.FetchTrials(*args, **kwargs)

    def fetchMice(self, *args, **kwargs):
        kwargs['experiment_group'] = self._project_name
        return self.FetchMice(*args, **kwargs)

    def fetchValues(self, attr):
        return self.FetchValues(attr, project_name=self._project_name)

    def find(self, arg):
        params = re.search('@.+=["|\'].*["|\']', arg)
        if params is not None:
            matches = re.search('(?<=@).*(?=\=)',params.group(0))
            if matches is not None:
                key = matches.group(0)
                value = re.search('(?<=["|\'])[^"]*(?=["|\'])', params.group(0)).group(0)
                if key == 'tSeriesDirectory':
                    trial_id = database.fetchTrialId(tSeries_path=value)
                    if trial_id is not None:
                        return dbExperiment(trial_id)
                elif key =='mouseID':
                    try:
                        return dbMouse(value, project_name=self._project_name)
                    except AttributeError:
                        return None

    def findall(self, arg):
        if arg == 'mouse':
            return [dbMouse(m, project_name=self._project_name) for m in
                database.fetchAllMice(project_name=self._project_name)]


class dbMouse(Mouse):
    _keyDict = {
        'mouseID': 'mouse_name'
    }

    @staticmethod
    def AirTableImport(mouse_name, animal_id, project_name=None, *args):
        if len(args) == 0:
            args = ['Genotype', 'Animal ID', 'Strain', 'Gender', 'Cage Card',
                'Born', 'Father ID', 'Mother ID', 'Date Weaned', 'ID']

        api_key = 'keyGJofA3RMzqITjZ'
        url = 'https://api.airtable.com/v0/appQVthszsOUel8mt/' + \
            'Imported%20Table?'
        query = urllib.urlencode(
            {'filterByFormula': "{Animal ID}='%s'" % animal_id} )
        headers = {'Authorization': 'Bearer {}'.format(api_key)}
        request = urllib2.Request(url+query, headers=headers)

        if len(dbExperimentSet.FetchMice(animal_id=animal_id)):
            raise KeyError(
                "Animal ID {} already in Experiments Database".format(
                    animal_id))

        try:
            response=json.loads(urllib2.urlopen(request).read())
        except:
            raise KeyError(
                "Unable to Fetch Record, Animal ID='{}'".format(animal_id))

        if len(response['records']) != 1:
            raise KeyError(
                "Unable to uniquly identify Animal, Amimal ID='{}'".format(
                    animal_id))

        record = response['records'][0]['fields']
        if 'Partner ID' not in record.keys() and project_name is not None:
            record_url = '/'+response['records'][0]['id']
            headers['Content-type'] = 'application/json'
            data = json.dumps({'fields': {'Partner ID': project_name}})
            request = urllib2.Request(url[:-1]+record_url, headers=headers, data=data)
            request.get_method = lambda: 'PATCH'
            try:
                response=json.loads(urllib2.urlopen(request).read())
            except:
                warnings.warn('Unable to update AirTable Partner ID')
        elif project_name != record['Partner ID']:
            warnings.warn('\nAirTable Partner ID already assigned: {}'.format(
                record['Partner ID']))

        mouse = dbMouse(database.fetchMouseId(mouse_name, create=True,
                                              project_name=project_name))

        for attr in filter(lambda a: a in record.keys(), args):
            dbkey = attr.lower().replace(' ', '_')
            if dbkey == 'id':
                dbkey = 'airtable_id'
            mouse.__setattr__(dbkey, record[attr])

        mouse.save(store=True)
        return mouse

    @staticmethod
    def CreateMouse(mouse_name, project_name=None, **kwargs):
        try:
            database.fetchMouseId(mouse_name, project_name=project_name)
        except:
            pass
        else:
            raise KeyError("mouse_name {} already exists".format(mouse_name))

        mouse = dbMouse(database.fetchMouseId(mouse_name, create=True,
                                              project_name=project_name))
        for attr, val in kwargs.iteritems():
            mouse.__setattr__(attr, val)

        mouse.save(store=True)
        return mouse

    def __init__(self, mouse_id, project_name=None):
        try:
            int(mouse_id)
        except ValueError:
            mouse_id = database.fetchMouseId(mouse_id,
                project_name=project_name)

        self._mouse_id = mouse_id
        self.attrib = database.fetchMouse(mouse_id)
        self.attrib['viruses'] = []
        self.attrib.update(database.fetchAllMouseAttrs(self.mouse_name,
            parse=True))
        self._experiments = [int(r['trial_id']) for r in
            database.fetchMouseTrials(self.mouse_name)]
        self._attrib = copy.deepcopy(self.attrib)

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        return self.mouse_id == other.mouse_id

    def __hash__(self):
        return self.__repr__().__hash__()

    def __iter__(self):
        for i in range(len(self._experiments)):
            yield dbExperiment(self._experiments[i])

    def __len__(self):
        return len(self._experiments)

    def __getitem__(self, key):
        return dbExperiment(self._experiments[key])

    def __getattr__(self, item):
        if item == 'attrib' or item[0] == '_':
            return self.__dict__[item]

        item = self._keyDict.get(item, item)
        if '_'+item in self.__dict__.keys():
            return self.__dict__['_'+item]

        if item == 'viruses':
            return self.getViruses()

        return self.attrib[item]

    def __setattr__(self, item, value):
        if item == 'attrib' or item[0] == '_':
            self.__dict__[item] = value
        else:
            item = self._keyDict.get(item, item)
            if '_' + item in self.__dict__.keys():
                raise Exception('{} is not settable'.format(item))
            if item == 'virus' or item == 'viruses':
                self.addVirus(value)
            else:
                self.attrib[item] = value

    def __delattr__(self, item):
        if item == 'attrib' or item[0] == '_':
            del self.__dict__[item]
        else:
            item = self._keyDict.get(item, item)
            if '_' + item in self.__dict__.keys():
                raise Exception('{} is not settable'.format(item))
            elif item in self._attrib:
                self.attrib[item] = None
            else:
                del self.attrib[item]

    @property
    def parent(self):
        return dbExperimentSet()

    @property
    def mouse_id(self):
        return self._mouse_id

    def removeVirus(self, virus):
        try:
            self.attrib['viruses'].remove(virus)
        except ValueError:
            pass

    def getViruses(self):
        return map(dbVirus, self.attrib['viruses'])

    def addVirus(self, virus_id):
        try:
            dbVirus(virus_id)
        except ValueError:
            dbVirus.AirTableImport(virus_id)

        self.attrib['viruses'].append(virus_id)

    def get(self, arg, default=None):
        try:
            return self.__getattr__(arg)
        except KeyError:
            return default

    def fetchTrials(self, *args, **kwargs):
        return dbExperimentSet.FetchTrials(*args, mouse_id=self.mouse_id, **kwargs)

    def find(self, arg):
        params = re.search('@.+=["|\'].*["|\']', arg)
        if params is not None:
            matches = re.search('(?<=@).*(?=\=)',params.group(0))
            if matches is not None:
                key = matches.group(0)
                value = re.search('(?<=["|\'])[^"]*(?=["|\'])', params.group(0)).group(0)
                if key == 'startTime':
                    trial_id = database.fetchTrialId(mouse_name=self.mouse_name,
                        startTime=value)
                    if trial_id is not None:
                        return dbExperiment(trial_id)

    def findall(self, arg):
        if arg == 'experiment':
            return [dbExperiment(e) for e in self._experiments]

    def save(self, store=False):
        updates = {k:v for k,v in self.attrib.iteritems() if k not in
            self._attrib.keys() or v != self._attrib[k]}

        if not store:
            print 'changes to {}: {}'.format(self.mouse_name, updates)
        else:
            print 'saving changes to {}: {}'.format(self.mouse_name, updates)
            for key, value in updates.iteritems():
                if key in ['mouse_id, mouse_name']:
                    raise Exception(
                        "use database module to modify param {}".format(key))

                length = 1
                try:
                    length = len(value)
                except TypeError:
                    pass

                if value is None or length == 0:
                    database.deleteMouseAttr(self.mouse_name, key)
                else:
                    if isinstance(value, dict) or isinstance(value,list):
                        value = json.dumps(value)
                    database.updateMouseAttr(self.mouse_name, key, value)
            self._attrib = copy.deepcopy(self.attrib)

    def delete(self):
        database.deleteMouse(self.mouse_id)
        self._mouse_id = None
        self.attrib={}


class dbExperiment(Experiment, Trial):
    _keyDict= {
        'startTime': 'start_time',
        'time': 'start_time',
        'stopTime': 'stop_time',
        'tSeriesDirectory': 'tSeries_path',
        'project_name': 'experiment_group'
    }

    def __init__ (self, trial_id):
        self._rois = {}
        self._props = {}
        self.attrib = database.fetchTrial(trial_id)
        self._props['trial_id'] = trial_id

        if self.attrib is None:
            raise KeyError("Trial ID {} does not exist".format(trial_id))

        self._props['start_time'] = self.attrib['start_time'].__format__('%Y-%m-%d-%Hh%Mm%Ss')
        self.attrib['stop_time'] = \
            self.attrib['stop_time'].__format__('%Y-%m-%d-%Hh%Mm%Ss')
        for key in self.attrib.keys():
            if self.attrib[key] is None:
                del self.attrib[key]

        if self.attrib.get('behavior_file') is not None:
            self.attrib['filename'] = \
                os.path.splitext(self.attrib['behavior_file'])[0] + '.pkl'

        self.attrib.update(database.fetchAllTrialAttrs(self.trial_id,
            parse=True))

        self.attrib['experimentType'] = self.attrib.get('experimentType', '')
        self.attrib['imagingLayer'] = self.attrib.get('imagingLayer', 'unk')
        self.attrib['uniqueLocationKey'] = self.attrib.get(
            'uniqueLocationKey', '')
        self.attrib['experimenter'] = self.attrib.get('experimenter', '')
        self.attrib['belt'] = self.attrib.get('belt', 'burlap1')
        self.attrib['belt_length'] = self.attrib.get('track_length', None)

        self._trial = dbTrial(self)
        self._attrib = self.attrib.copy()

    def __str__(self):
        return "<dbExperiemnt: trial_id=%d mouse_id=%d experimentType=%s>" % \
            (self.trial_id, self.mouse_id, self.get('experimentType',''))

    def __repr__(self):
        return "<dbExperiemnt: trial_id=%d mouse_id=%d experimentType=%s>" % \
            (self.trial_id, self.mouse_id, self.get('experimentType',''))

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        return self.trial_id == other.trial_id

    def __hash__(self):
        return self.__repr__().__hash__()

    def __iter__(self):
        yield self._trial

    def __len__(self):
        return 1

    def __getitem__(self, key):
        if key == 0:
            return self._trial
        raise KeyError

    def __getattr__(self, item):
        if item == 'attrib' or item[0] == '_':
            return self.__dict__[item]

        item = self._keyDict.get(item, item)
        if '_'+item in self.__dict__.keys():
            return self.__dict__['_'+item]

        if item == 'filename':
            behavior_file = self.behavior_file
            if behavior_file is None:
                return None
            return os.path.splitext(behavior_file)[0] + '.pkl'

        return self.attrib[item]

    def __setattr__(self, item, value):
        if item == 'attrib' or item[0] == '_':
            self.__dict__[item] = value
        else:
            item = self._keyDict.get(item, item)
            if '_' + item in self.__dict__.keys() or item in self.props.keys():
                raise Exception('{} is not settable'.format(item))

            self.attrib[item] = value

    def __delattr__(self, item):
        if item == 'attrib' or item[0] == '_':
            del self.__dict__[item]
        else:
            item = self._keyDict.get(item, item)
            if '_' + item in self.__dict__.keys():
                raise Exception('{} is not settable'.format(item))
            elif item in self._attrib:
                self.attrib[item] = None
            else:
                del self.attrib[item]

    @property
    def parent(self):
        return dbMouse(self.mouse_id)

    @property
    def trial_id(self):
        return self._props['trial_id']

    @property
    def start_time(self):
        return self._props['start_time']

    def get(self, item, default=None):
        item = self._keyDict.get(item, item)
        if item == 'filename':
            behavior_file = self.behavior_file
            if behavior_file is None:
                return None
            return os.path.splitext(behavior_file)[0] + '.pkl'

        if item in self.props.keys():
            return self.props[item]
        return self.attrib.get(item, default)

    def findall(self, arg):
        if arg == 'trial':
            return [self._trial]

    def find(self, arg):
        if arg == 'trial':
            return self._trial

    def save(self, store=False):
        updates = {k:v for k,v in self.attrib.iteritems() if k not in
            self._attrib.keys() or v != self._attrib[k]}

        update_trial = False
        trial_args = ['behavior_file', 'mouse_name', 'start_time', 'stop_time',
            'experiment_group']

        if not store:
            print 'changes to {}: {}'.format(self.trial_id, updates)
        else:
            print 'saving changes to {}: {}'.format(self.trial_id, updates)
            for key, value in updates.iteritems():
                if key == 'trial_id':
                    raise Exception('changing trial_id')
                elif key == 'tSeries_path':
                    database.pairImagingData(value, trial_id=self.trial_id)
                elif key in trial_args and key != 'mouse_name':
                    update_trial = True
                else:
                    if value is None:
                        database.deleteTrialAttr(self.trial_id, key)
                    else:
                        if isinstance(value, dict) or isinstance(value,list):
                            value = json.dumps(value)
                        database.updateTrialAttr(self.trial_id, key, value)

            self._attrib = self.attrib.copy()
            self.attrib['mouse_name'] = \
                database.fetchMouse(self.mouse_id)['mouse_name']

        if update_trial:
            database.updateTrial(*[self.attrib[k] for k in trial_args],
                trial_id=self.trial_id)

    def delete(self):
        database.deleteTrial(self.trial_id)
        self._trial_id = None
        self.attrib={}
        self._props={}


class dbTrial(Trial):
    def __init__(self, parent):
        self._parent = parent
        self.attrib = parent.attrib

    @property
    def parent(self):
        return self._parent

    def get(self, arg, default=None):
        return self.parent.get(arg, default=default)


class dbVirus():

    @staticmethod
    def AirTableImport(virus_id, *args):
        if len(args) == 0:
            args = ['Status', 'Box #', 'Promoter', 'Activator', 'Fluorophore',
                'FLEX', 'recombinase dependence']

        api_key = 'keyGJofA3RMzqITjZ'
        url = 'https://api.airtable.com/v0/appEhw0nr49Psd3Wy/' + \
            'Imported%20Table?'
        query = urllib.urlencode(
            {'filterByFormula': "{Field 1}='%s'" % virus_id} )
        headers = {'Authorization': 'Bearer {}'.format(api_key)}
        request = urllib2.Request(url+query, headers=headers)

        try:
            response=json.loads(urllib2.urlopen(request).read())
        except:
            raise KeyError(
                "Unable to Fetch Record, virus_id='{}'".format(virus_id))

        try:
            record = response['records'][0]['fields']
        except:
            raise ValueError("unable to import virus {}".format(virus_id))

        database.updateVirus(virus_id, name=record['Name'],
            arrival_date=record['Arrival date'],
            source_code=record['Source Code'])

        virus = dbVirus(virus_id)

        for attr in filter(lambda a: a in record.keys(), args):
            dbkey = attr.lower().replace(' ', '_')
            if dbkey == 'box_#':
                dbkey = 'box'
            virus.__setattr__(dbkey, record[attr])

        virus.save(store=True)
        return virus

    def __init__(self, virus_id):
        self._virus_id = virus_id
        self.attrib = database.fetchVirus(virus_id)
        if self.attrib is None:
            raise ValueError("Virus Not Found")
        self.attrib.update(database.fetchAllVirusAttrs(self._virus_id))
        self._attrib = self.attrib.copy()

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        return self.virus_id == other.virus_id

    def __hash__(self):
        return self.__repr__().__hash__()

    def __repr__(self):
        return "<dbVirus: virus_id=%d name=%s>" % (self.virus_id, self.name)

    def __getattr__(self, item):
        if item == 'attrib' or item[0] == '_':
            return self.__dict__[item]

        if '_'+item in self.__dict__.keys():
            return self.__dict__['_'+item]

        return self.attrib[item]

    def __setattr__(self, item, value):
        if item == 'attrib' or item[0] == '_':
            self.__dict__[item] = value
        else:
            if '_' + item in self.__dict__.keys():
                raise Exception('{} is not settable'.format(item))

            self.attrib[item] = value

    def __delattr__(self, item):
        if item == 'attrib' or item[0] == '_':
            del self.__dict__[item]
        else:
            if '_' + item in self.__dict__.keys():
                raise Exception('{} is not settable'.format(item))
            elif item in self._attrib:
                self.attrib[item] = None
            else:
                del self.attrib[item]

    @property
    def virus_id(self):
        return self._virus_id

    def get(self, arg, default=None):
        try:
            return self.__getattr__(arg)
        except KeyError:
            return default

    def save(self, store=False):
        updates = {k:v for k,v in self.attrib.iteritems() if k not in
            self._attrib.keys() or v != self._attrib[k]}

        if not store:
            print 'changes to virus {}: {}'.format(self._virus_id, updates)
        else:
            print 'saving changes to {}: {}'.format(self._virus_id, updates)
            for key, value in updates.iteritems():
                if key == 'virus_id':
                    raise Exception(
                        "use database module to modify param {}".format(key))
                elif key in ['name', 'arrival_date', 'source_code']:
                    database.updateVirus(self._virus_id, **{key: value})

                if value is None:
                    database.deleteVirusAttr(self._virus_id, key)
                else:
                    if isinstance(value, dict) or isinstance(value,list):
                        value = json.dumps(value)
                    database.updateVirusAttr(self._virus_id, key, value)
            self._attrib = self.attrib.copy()

