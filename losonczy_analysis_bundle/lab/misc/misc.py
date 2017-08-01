"""General helper functions."""

import numpy as np
import os
import datetime
import cPickle as pickle
import pandas as pd
import csv
import time
import collections
import functools
import matplotlib.pyplot as plt
from os.path import dirname, realpath
import subprocess
from copy import copy
from warnings import warn
import tempfile

from stats import full_anova


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    From:
    http://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions

    """
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):

        key1 = tuple()
        key2 = tuple()
        for arg in args:
            if callable(arg):
                key2 += (arg,)
            else:
                key1 += (repr(arg),)
        for key in sorted(kwargs.keys()):
            if callable(kwargs[key]):
                key2 += (key, kwargs[key])
            else:
                key1 += (repr({key: kwargs[key]}), )
        key = (key1, key2)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return _memcopy(cache[key])
    return memoizer


def _memcopy(obj):
    """Performs a copy that is half-way between a shallow and deep copy.
    The idea is to deepcopy simple iterables (lists, dictionaries), but not
    our customs classes (Experiments, ROIs).

    This is intended to be used by memoize to smartly copy cached data, but
    not Experiments and ROIs that might be keys of a dict or values in columns
    of a DataFrame.

    """

    if isinstance(obj, pd.DataFrame):
        return obj.copy(deep=False)
    if isinstance(obj, collections.Mapping):
        return type(obj)(
            {key: _memcopy(value) for key, value in obj.iteritems()})
    if isinstance(obj, list) or isinstance(obj, tuple):
        return type(obj)([_memcopy(item) for item in obj])
    return copy(obj)


def savefigs(pdf_pages, figs):
    """Save a single figure or list of figures to a multi-page PDF.

    This function is mostly used so that the same call can be used for a single
    page or multiple pages. Will close Figures once they are written.

    Parameters
    ----------
    pdf_pages : matplotlib.backends.backend_pdf.PdfPages
        PdfPage instance that the figures will get written to.
    figs : matplotlib.pyplot.Figure or iterable of Figures

    """
    try:
        for fig in figs:
            pdf_pages.savefig(fig)
            plt.close(fig)
    except TypeError:
        pdf_pages.savefig(figs)
        plt.close(figs)


def save_figure(
        fig, filename, save_dir='', expt_grps=None, stats_data=None,
        ignore_shuffle=True):
    """Helper function to save figure and run stats.

    Parameters
    ----------
    fig : matplotlib.pyplot.Figure
    filename : str
    save_dir : str
    expt_grps : optional, sequence of lab.ExperimentGroup
        If passed in and saving as a pdf, add a page of summary information
        about the experiments used in the analysis.
    stats_data : optional, dict
        If passed in, save data with save_data and also create stat figures
        for all data if writing a pdf. See save_data for details of format.
    ignore_shuffle : bool
        If True, ignore the shuffle data for the ANOVA in any stats figures.

    """
    if not os.path.isdir(os.path.normpath(save_dir)):
        os.makedirs(os.path.normpath(save_dir))

    if filename.endswith('pdf'):
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(os.path.join(save_dir, filename))
        if expt_grps is not None:
            savefigs(pp, summarySheet(expt_grps))
        if stats_data is not None:
            # Create stats summary figures for whatever we can.
            for key in stats_data:
                try:
                    savefigs(pp, stats_fig(
                        stats_data[key], label=key,
                        ignore_shuffle=ignore_shuffle))
                except:
                    pass
        savefigs(pp, fig)
        pp.close()
    elif filename.endswith('svg'):
        fig.savefig(os.path.join(
            save_dir, filename.replace('.pdf', '.svg')), format='svg')
    else:
        # If we don't recognize the file format, drop into a debugger
        warn('Unrecognized file format, dropping into debugger.')
        from pudb import set_trace
        set_trace()


def save_data(
        data, fig=None, method='pkl', label='data',
        save_dir='/analysis/figure_data'):
    """Save data to a file and add the location to the figure.

    Parameters
    ----------
    data : dict
        Dictionary of data to save. Designed to save the output from
        plot_dataframe.
    fig : optional, matplotlib.pyplot.Figure
        Will print the path to the data on the figure.
    format : {'pkl', 'csv', 'dict'}
        Method used to save the data.
    label : optional, str
        Name of either the top-level directory or the data file itself.
        Timestamp and suffix (when appropriate) appended to label.
    save_dir : str
        Path to save data to.

    """
    time_str = datetime.datetime.strftime(
        datetime.datetime.now(), '%Y-%m-%d-%Hh%Mm%Ss')
    basename = os.path.join(save_dir, '{}_{}'.format(label, time_str))

    prep_dataframe_save(data)

    if method == 'pkl':
        pickle.dump(
            data, open(basename + '.pkl', 'w'), pickle.HIGHEST_PROTOCOL)

    elif method == 'csv':
        # When saving as a csv, dictionaries define folders, the keys name the
        # files and the values should be either a numpy array or another
        # dictionary, which will iteratively be processed the same way.
        def parse_data(d, path):
            if isinstance(d, dict):
                os.mkdir(path)
                for key in d:
                    parse_data(d[key], os.path.join(path, key))
            else:
                filename = path + '.csv'
                if d is None:
                    pass
                elif isinstance(d, pd.DataFrame):
                    d.to_csv(filename)
                else:
                    with open(filename, 'wb') as f:
                        writer = csv.writer(f)
                        try:
                            writer.writerows(d)
                        except:
                            writer.writerow(d)

        parse_data(data, basename)

    elif method == 'dict':
        # Expects a dictionary, iteratively parses each key and either creates
        # a new folder if the value is another dictionary, or tries to smartly
        # save the value: npy for numpy arrays, pkl for dataframe, otherwise
        # try pickling whatever is there
        def parse_data(d, path):
            if isinstance(d, dict):
                os.mkdir(path)
                for key in d:
                    parse_data(d[key], os.path.join(path, key))
            else:
                if d is None:
                    pass
                elif isinstance(d, pd.DataFrame):
                    d.to_pickle(path + '.pkl')
                elif isinstance(d, np.ndarray):
                    np.save(path + '.npy', d)
                else:
                    with open(path + '.pkl', 'wb') as f:
                        pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

        parse_data(data, basename)
    else:
        raise ValueError('Unrecognized method argument: ' + method)

    print('Data for {} saved to {}'.format(label, basename))
    if fig is not None:
        try:
            if fig is list:
                for fgr in fig:
                    fgr.text(1, 0, basename, ha='right', va='bottom')
            else:
                fig.text(1, 0, basename, ha='right', va='bottom')
        except:
            pass
    return basename


def stats_fig(data, label='', ignore_shuffle=True, figsize=(8.5, 11)):
    """Create a summary figure of stats.

    Parameters
    ----------
    data : dict or pandas.DataFrame
        A single plot's worth of data.
    label : optional, str
        Label to be put in figure heading.
    figsize : tuple
        Size of figure.
    ignore_shuffle : bool
        If True, ignore shuffle data when running the ANOVA.

    """
    divider = '\n' + '-' * 100 + '\n'

    tmp_file = tempfile.NamedTemporaryFile()
    tmp_path = tmp_file.name
    tmp_file.close()

    full_anova(data, file_path=tmp_path, ignore_shuffle=ignore_shuffle)

    fig = plt.figure(figsize=figsize)
    header = 'Summary Stats: {} - {}'.format(label, time.asctime())
    with open(tmp_path, 'r') as f:
        anova_txt = ''.join(f.readlines())
    fig.text(0.05, 0.97, header + divider + anova_txt,
             va='top', ha='left', fontsize=5)
    os.remove(tmp_path)

    return fig

#
# Filter helper functions
#


def filter_intersection(filters):
    """Return a new ROI filter that is the intersection of all the filters."""
    def intersection(x):
        for f in filters:
            if f is not None and not f(x):
                return False
        return True
    return intersection


def df_filter_intersection(filters):

    def intersection(df):
        filter_results = np.ones(len(df), dtype=bool)
        for f in filters:
            if f is not None:
                filter_results = np.logical_and(filter_results, f(df))
        return filter_results

    return intersection


def filter_union(filters):
    """Returns a new ROI filter that is the union of all the filters passed
    in.

    """

    def union(x):
        for f in filters:
            if f is not None and f(x):
                return True
        return False
    return union


def invert_filter(f):
    """Returns a new inverted ROI filter."""
    def inverted(x):
        return not f(x)
    return inverted


def parseTime(timeStr):
    """Parses a time string from the xml into a datetime object"""
    # Check for sql format
    if ':' in timeStr:
        return datetime.datetime.strptime(timeStr, '%Y-%m-%d %H:%M:%S')
    else:
        return datetime.datetime.strptime(timeStr, '%Y-%m-%d-%Hh%Mm%Ss')


def timestamp():
    """Returns the current time as a timestamp string."""
    return datetime.datetime.strftime(
        datetime.datetime.now(), '%Y-%m-%d-%Hh%Mm%Ss')


def gitRevision():
    """Returns the git SHA1 revision number of the current code"""

    path = os.getcwd()
    os.chdir(dirname(realpath(__file__)))
    sha = subprocess.check_output("git rev-parse HEAD", shell=True)
    os.chdir(path)
    return sha.strip()


def summarySheet(exptGrps, text=None):
    """ Generate a coversheet figure summarizing the exptGrp(s).

    Keyword arguments:
    text -- a string of additional information to include

    """

    nExpts = 0
    for exptGrp in exptGrps:
        nExpts += len(exptGrp)

    height = np.ceil(nExpts / 8.0) + 1

    fig = plt.figure(figsize=(12, height if height >= 8 else 8))

    header = 'Summary sheet - {time} - {sha1}\n\n'.format(
        time=time.asctime(), sha1=gitRevision())

    divider = '---------------------------------------------------------------'

    exptGrp_text = ''
    for grpIdx, exptGrp in enumerate(exptGrps):
        # If exptGrp is a pcExperimentGroup add in a list of parameters
        try:
            exptGrp.args
        except AttributeError:
            exptGrp_text += '{}:\n'.format(
                exptGrp.label() if exptGrp.label() is not None else
                'Group {}'.format(grpIdx + 1))
        else:
            exptGrp_text += '{}:'.format(
                exptGrp.label() if exptGrp.label() is not None else
                'Group {}'.format(grpIdx + 1))
            for arg in exptGrp.args:
                exptGrp_text += ' {key} = {value},'.format(
                    key=arg, value=exptGrp.args[arg])
            exptGrp_text = exptGrp_text[:-1] + '\n'

        # Sort experiments by mouse
        mice = {}
        for expt in exptGrp:
            if expt.parent not in mice:
                mice[expt.parent] = [expt]
            else:
                mice[expt.parent].append(expt)

        for mouse in sorted(mice):
            exptGrp_text += '  ' + str(mouse) + '\n'
            for expt in sorted(mice[mouse]):
                exptGrp_text += '  ' + str(expt) + '\n'
        exptGrp_text += '\n'

    fig_text = header + exptGrp_text

    if text is not None:
        fig_text += divider + '\n' + text

    fig.text(0.05, 0.97, fig_text, va='top', ha='left', fontsize=5)

    return fig


def prep_dataframe_save(data):
    """
    Recursively prepares dataframes for saving by removing custom classes and
    replacing with uniquely determined attribute columns

    data : pd.DataFrame or dict of DataFrames

    """

    def parse_data(d):
        if isinstance(d, dict):
            for key in d:
                parse_data(d[key])
        else:
            if d is None:
                pass
            elif isinstance(d, pd.DataFrame):
                for pre in ('', 'first_', 'second_'):
                    if pre + 'trial' in d.columns:
                        d[pre + 'trial_time'] = d[pre + 'trial'].apply(
                            lambda trial: trial.get('time'))
                        d[pre + 'expt'] = d[pre + 'trial'].apply(
                            lambda trial: trial.parent)
                    if pre + 'roi' in d.columns:
                        d[pre + 'roi_id'] = d[pre + 'roi'].apply(
                            lambda roi: roi.id)
                        d[pre + 'uniqueLocationKey'] = d[pre + 'roi'].apply(
                            lambda roi: roi.expt.get('uniqueLocationKey'))
                        d[pre + 'expt'] = d[pre + 'roi'].apply(
                            lambda roi: roi.expt)
                    if pre + 'expt' in d.columns:
                        d[pre + 'expt_startTime'] = d[pre + 'expt'].apply(
                            lambda expt: expt.get('startTime'))
                        d[pre + 'mouse'] = d[pre + 'expt'].apply(
                            lambda expt: expt.parent)
                    if pre + 'mouse' in d.columns:
                        d[pre + 'mouseID'] = d[pre + 'mouse'].apply(
                            lambda mouse: mouse.get('mouseID'))

                    for col in ['roi', 'trial', 'expt', 'mouse']:
                        try:
                            d.drop(pre + col, 1, inplace=True)
                        except ValueError:
                            pass
    return parse_data(data)


def td_mean(deltas):
    """Returns the mean of datetime.timedelta objects"""
    # First sum is python sum, not numpy and the second argument is the 'zero'
    # of timedeltas
    # First check to see if deltas is a np.timedelta64, item() will convert it
    # to a datetime.timedelta
    try:
        deltas = [delta.item() for delta in deltas]
    except AttributeError:
        pass
    return sum(deltas, datetime.timedelta(0)) / len(deltas)


def norm_to_complex(arr):
    return angle_to_complex(norm_to_angle(arr))


def norm_to_angle(arr):
    return (arr * 2. * np.pi) % (2 * np.pi)


def complex_to_norm(arr):
    return angle_to_norm(complex_to_angle(arr)) % 1.


def complex_to_angle(arr):
    return np.angle(arr) % (2 * np.pi)


def angle_to_norm(arr):
    return (arr / 2. / np.pi) % 1.


def angle_to_complex(arr):
    return np.array([
        np.complex(x, y) for x, y in zip(np.cos(arr), np.sin(arr))])
