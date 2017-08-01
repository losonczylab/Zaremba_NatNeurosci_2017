"""Exceptions"""


class LabException(Exception):
    pass


class NoTSeriesDirectory(LabException):
    pass


class NoSimaPath(LabException):
    pass


class NoSignalsData(LabException):
    pass


class NoTransientsData(LabException):
    pass


class NoDfofTraces(LabException):
    pass


class NoPlaceFields(LabException):
    pass


class NoSpikesData(LabException):
    pass


class NoDemixedImage(LabException):
    pass


class MissingBehaviorData(LabException):
    pass


class MissingTrial(LabException):
    pass


class NoSharedROIs(LabException):
    pass


class No_LFP_Data(LabException):
    pass


class NoBeltInfo(LabException):
    pass


class InvalidDataFrame(LabException):
    pass
