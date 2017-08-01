"""This file contains functions to create plots that are nicely formatted for
use in publication figures
"""
from scipy.stats import sem
import matplotlib as mpl
from matplotlib.patches import Ellipse, Polygon
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
from matplotlib.cbook import flatten
import numpy as np
try:
    from bottleneck import nanmean, nanstd
except ImportError:
    from numpy import nanmean, nanstd
from scipy.stats import pearsonr
import itertools as it
import seaborn.apionly as sns

# Import plotting_helpers so you can use them from plotting
from lab.misc import signalsmooth
from plotting_helpers import stackedText, color_cycle


mpl.rcParams['font.size'] = 7
mpl.rcParams['font.sans-serif'] = 'Arial, Bitstream Vera Sans, Lucida Grande, Verdana, Geneva, Lucid, Helvetica, Avant Garde, sans-serif, Georgia'
mpl.rcParams['legend.fontsize'] = 7
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['axes.titlesize'] = 7
mpl.rcParams['xtick.major.size'] = 2
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['xtick.minor.size'] = 1
mpl.rcParams['ytick.minor.size'] = 1
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5
mpl.rcParams['xtick.major.pad'] = 1
mpl.rcParams['ytick.major.pad'] = 1
mpl.rcParams['lines.markersize'] = 2


def ellipsePlot(
        ax, xCentres, yCentres, xRadii, yRadii, boutonGroupLabeling=None,
        color=None, axesCenter=True, zoom_to_data=False, print_stats=False):
    """Create an ellipse scatter plot of one value vs another.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        Axis to plot on

    xCentres, yCentres, xRadii, yRadii : array-like
        The centers and axis lengths of the ellipses. All should be same length
    color : list of matplotlib.colorspec
        Color of the ellipses, should be same length as cetners and radii.
        If None, randomly colors each ellipse.
    axesCentre : bool
        If True, place the axes at 0 rather than at the edge of the plot
    zoom_to_dat : bool
        If True, set x and y lims to include all ellipses
    print_stats : bool
        If True, print correlation and slope on the plot

    """

    ells = [Ellipse(
        xy=[xCentres[i], yCentres[i]], width=2 * xRadii[i],
        height=2 * yRadii[i], lw=0.4) for i in range(len(xCentres)) if
        all([np.isfinite(x) for x in
            [xCentres[i], yCentres[i], xRadii[i], yRadii[i]]])]

    for e in ells:
        ax.add_artist(e)
        e.set_facecolor('none')
        if color:
            e.set_edgecolor(color)
        else:
            e.set_edgecolor(np.random.rand(3) * np.array([1, 1, 1]))
    if boutonGroupLabeling:
        rois = boutonGroupLabeling
        roiGroups, roiGroupNames = BoutonSet(rois).boutonGroups()
        for k, group in enumerate(roiGroups):
            roiIndices = [rois.index(r.name) for r in group]
            ax.plot([xCentres[i] for i in roiIndices], [yCentres[i] for i in roiIndices],
                    groupPointStyle(roiGroupNames[k]))
        a, = ax.plot([], [], 'wo')
        b, = ax.plot([], [], 'w^')
        c, = ax.plot([], [], 'k*')
        ax.legend((a, b, c), ('somatic', 'dendritic', 'unlabeled'), numpoints=1,
                  frameon=False, loc='lower right', borderpad=0, borderaxespad=0,
                  labelspacing=0.1, handletextpad=0)
    elif color is not None:
        ax.plot(xCentres, yCentres, '.', color=color)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if axesCenter:
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.set_xticks([t for t in ax.get_xticks() if t != 0])
        ax.set_yticks([t for t in ax.get_yticks() if t != 0])
    ax.tick_params(axis='x', direction='inout')
    ax.tick_params(axis='y', direction='inout')
    if zoom_to_data:
        min_x = np.amin([x_c - x_r for x_c, x_r in zip(xCentres, xRadii)])
        max_x = np.amax([x_c + x_r for x_c, x_r in zip(xCentres, xRadii)])
        min_y = np.amin([y_c - y_r for y_c, y_r in zip(yCentres, yRadii)])
        max_y = np.amax([y_c + y_r for y_c, y_r in zip(yCentres, yRadii)])
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
    if print_stats:
        finite_vals = np.isfinite(xCentres) & np.isfinite(yCentres)
        correlation = np.corrcoef(
            np.array(xCentres)[finite_vals],
            np.array(yCentres)[finite_vals])[0, 1]
        slope, _ = np.polyfit(np.array(xCentres)[finite_vals],
                              np.array(yCentres)[finite_vals], 1)
        stackedText(ax, ['corr: {:.3f}'.format(correlation),
                         'slope: {:.3f}'.format(slope)],
                    colors=['k', 'k'], loc=2)


def scatterPlot(
        ax, values, conditionNames, colors=None, plotRange=None,
        plotEqualLine=True, print_stats=False, stats_by_color=False,
        color_legend=None, **scatter_kwargs):
    """Create a scatter plot of one value vs another.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        The axis to plot on.
    values : 2xN numpy.ndarray (or list equivalent)
        Contains the x and y values for each of the N data points.
    conditionNames: list of str
        The x and y axis labels
    plotRange : 2-element tuple of floats, optional
        The min and max limits for both axis
    print_stats : bool
        If true, adds the correlation value and slope of the linear fit.
    stats_by_color : bool
        If True and print_stats is True, runs stats on each color
        independently.
    color_legend : dict
        If print_stats and stats_by_color, a dictionary where
        keys are colors and values are a label for that grouping
    **scatter_kwargs
        Additional keyword arguments are passed directly to the scatter
        plotting function.

    """

    if colors:
        assert len(colors) == len(values[0])
        ax.scatter(values[0], values[1], c=colors, **scatter_kwargs)
    else:
        ax.scatter(values[0], values[1], **scatter_kwargs)
    ax.set_xlabel(conditionNames[0])
    ax.set_ylabel(conditionNames[1])
    if plotRange is not None:
        ax.set_xlim(plotRange)
        ax.set_ylim(plotRange)
    if plotEqualLine:
        l = ax.get_xlim()
        ax.plot(l, l, '--k', lw=0.25)
        ax.set_xlim(l)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='x', direction='inout')
    ax.tick_params(axis='y', direction='inout')
    if print_stats:
        finite_vals = np.all(np.isfinite(values), axis=0)
        vals = np.array(values)[:, finite_vals]
        if not stats_by_color:
            r, p = pearsonr(vals[0], vals[1])
            stackedText(ax, ['r: {:.4f}'.format(r), 'p: {:.4f}'.format(p)],
                        colors=['k', 'k'], loc=2)
        else:
            # Keep as a list, since colors can be a string, number or list
            # Converting to an array does not handle all those types the same
            finite_colors = [c for c, f in it.izip(colors, finite_vals) if f]
            text = []
            color_dict = {}
            for color in set(colors):
                color_matches = [
                    i for i, c in enumerate(finite_colors) if c == color]
                color_vals = vals[:, color_matches]
                r, p = pearsonr(color_vals[0], color_vals[1])
                if color_legend:
                    text_str = '{}- '.format(color_legend[color])
                else:
                    text_str = ''
                text_str += 'r: {:.4f}, p: {:.4f}'.format(r, p)
                text.append(text_str)
                color_dict[text_str] = color
            sorted_text = sorted(text)
            all_colors = [color_dict[t] for t in sorted_text]
            stackedText(
                ax, sorted_text, colors=all_colors, loc=2, size='x-small')


def histogram(
        ax, values, bins, range=None, color='k', normed=False, plot_mean=False,
        orientation='vertical', filled=True, mean_kwargs=None, **kwargs):
    """Create a histogram plot of the values.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        The axis to plot on.
    values : array-like
        Data to plot.
    bins, range
        See matplotlib.pyplot.hist documentation
    color : matplotlib.colorspec
        The color of the plot (note the fill will be 50% opacity)
    normed : bool
        If True, plots the probability density.
    plot_mean : bool
        If True, plots the mean of the distribution as a vertical line.
    orientation : str, 'vertical' or 'horizontal'
        Plots bars vertically or horizontally.
    filled : bool
        If True, fill the histogram, otherwise just plot the outline
    mean_kwargs : dict
        Dictionary of keyword pair arguments passed to the plotting of the mean
        line.
    **kwargs
        Additional arguments to pass to the histogram plotting function.

    """

    if mean_kwargs is None:
        mean_kwargs = {}
    else:
        mean_kwargs = dict(mean_kwargs)

    if 'linestyle' not in mean_kwargs and 'ls' not in mean_kwargs:
        mean_kwargs['linestyle'] = '--'

    # Save original ylim, make sure it at least doesn't get shorter
    if len(ax.lines) or len(ax.patches):
        orig_ylim = ax.get_ylim()
    else:
        orig_ylim = (0, 0)

    if filled:
        ax.hist(
            values, bins=bins, range=range, normed=normed, color=color, lw=0,
            histtype='stepfilled', alpha=0.5, orientation=orientation,
            **kwargs)
    hist = ax.hist(
        values, bins=bins, range=range, normed=normed, color=color, lw=1.0,
        histtype='step', orientation=orientation, **kwargs)

    ylim = ax.get_ylim()
    if ylim[1] < orig_ylim[1]:
        ylim = list(ylim)
        ylim[1] = orig_ylim[1]
    if plot_mean:
        value_mean = np.mean(values)
        mean_bin_count = hist[0][np.sum(hist[1] < value_mean) - 1]
        if mean_bin_count == 0:
            mean_bin_count = 1
        if orientation == 'vertical':
            ax.plot([np.mean(values)] * 2, [0, mean_bin_count], color=color,
                    **mean_kwargs)
        elif orientation == 'horizontal':
            ax.plot([0, mean_bin_count], [np.mean(values)] * 2, color=color,
                    **mean_kwargs)

    ax.set_ylim(bottom=0, top=ylim[1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', direction='out')

    return hist


def cdf(ax, values, bins='exact', range=None, **kwargs):
    """Plot the empirical CDF.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        The axis to plot on
    Values : array-like
       The data to be plotted.
    bins
        See matplotlib.pyplot.hist documentation.
        Can also be 'exact' to calculate the exact empirical CDF
    range
        See matplotlib.pyplot.hist documentation.
    **kwargs
        Any additional keyword arguments are passed to the plotting function.

    """
    if bins == 'exact':
        bins = np.unique(np.sort(values))
        if len(bins) == 1:
            return None, None
    hist_counts, hist_bins = np.histogram(values, bins=bins, range=range)

    cum_counts = np.cumsum(hist_counts)
    cdf = cum_counts * 1.0 / cum_counts[-1]

    # Want to plot each value at the right side of the bin, but then also put
    # back in value for the beginning of the first bin
    cdf_zero = np.sum(values <= hist_bins[0]) * 1.0 / cum_counts[-1]
    cdf = np.hstack([cdf_zero, cdf])

    ax.plot(hist_bins, cdf, **kwargs)

    ax.set_ylim((0, 1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', direction='out')
    ax.set_ylabel('Cumulative probability')

    return hist_bins, cdf


def pointPlot(ax, values, labels):
    """Plot all measurements of a value, with one point per measurement and one
    column per condition.
    Means +/- 1.96 standard errors are plotted beside each column.
    Inputs:
        ax: the axis of the plot
        values: the data to be plotted (a list of lists/arrays of values, one
            list/array per condition
        labels: the labels for each condition
    """
    for i, v in enumerate(values):
        v = v[np.isfinite(v)]
        ax.plot([i for x in v], v, 'k.')
        ax.errorbar(i + 0.2, nanmean(v), yerr=1.96 * sem(v), marker='o',
                    color='k', ecolor='k', capsize=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 3))
    ax.set_xlim(-0.1, len(values) - 0.7)
    ax.set_xticks([i for i, _ in enumerate(values)])
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def whiskerPlot(ax, values, conditionNames, color='k', plotMeans=True):
    """
    Inputs:
        values -- CxN numpy array, where C in the number of conditions, N the
            number of observations
    """
    ax.plot(range(len(conditionNames)), values, color=color, lw=0.25)
    if plotMeans:
        m = nanmean(values, axis=1)
        err = sem(values, axis=1)
        ax.errorbar(range(len(conditionNames)), m, yerr=err, color=color)
    ax.set_xlim([-0.05, len(conditionNames) - 0.95])
    ax.set_xticks(range(len(conditionNames)))
    ax.set_xticklabels(conditionNames)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 3))
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def pairedPlot(ax, values, conditionNames, colors=None, plot_means=True):

    # ax.plot(np.arange(len(conditionNames)) + 1, values, lw=0.25, color='k', marker='o')
    assert len(values[0]) == len(values[1])
    m = nanmean(values, axis=1)
    err = sem(values, axis=1)

    for idx in range(len(conditionNames) - 1):
        for v1, v2 in zip(values[idx], values[idx + 1]):
            ax.plot([idx + 1.2, idx + 1.8], [v1, v2], color='k', lw=0.5)

    if plot_means:
        for idx in range(len(conditionNames)):
            c = colors[idx] if colors else 'k'
            ax.errorbar(idx + 1, m[idx], yerr=err[idx], color=c, elinewidth=1,
                        capthick=0, zorder=3, fmt='o', markersize=4, capsize=0,
                        mfc=c, mec=c)
            # ax.plot(idx + 1, m[idx], color='r', marker='o', lw=0.5, markersize=6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', direction='out')

    ax.set_xlim([0, len(conditionNames) + 1])
    ax.set_xticks(np.arange(len(conditionNames)) + 1)
    ax.set_xticklabels(conditionNames)


def tracePlot(ax, data, times, xlabels=[], ylabels=[], stimulusDurations=None,
              showTrials=False, smoothSize=7, shading='stdev', colors=None,
              drugList=['control'], markerDuration=3, yOffsets=None):
    """Plot an array of example traces, with rows for ROIs/PCs, and columns for stimuli/running
    Inputs:
        ax: the plot axis
        data: the traces to be plotted
            This can be formatted as a list of numpy arrays, where each array is of size (N, T, r),
            where N is the number of signals, T the number of time points, and r the number of repeats.
            Each array corresponds to a different stimulus.
            The data can also be organized as a dictionary, where the values are lists of the above
            format and the keys are the drug conditions.
        xlabels: labels for the stimuli
        ylabels: labels for the ROIs / PCs
        stimulusDurations: list of durations for the stimuli (one entry per column)
        showTrials: determines whether individual trials are plotted, or just the trial average
        shading: if set to 'stdev', then the standard deviation is shaded about the mean
        colors: a list of colors through which the traces will cycle (one color per row)
        drugList: list of drug conditions
        markerDuration: the size of the scaleBar (in seconds)
        yOffsets: specified offsets between the columns; calculated automatically if None
    The function could probably be cleaned up a bit, and the input format for data simplified
    """
    if data.__class__ == list:
        data = {'control': data}
    if colors is None:
        colors = ['b', 'g', '#FF8000', 'm', 'r', 'c']
    times = np.array(times)
    offset = times[-1] - times[0] + 1
    assert np.isfinite(offset)
    if ylabels is None:
        ylabels = []
    if smoothSize:
        for x in data.values():
            for d in x:
                for ROI in d:
                    for i in range(ROI.shape[1]):
                        ROI[:, i] = signalsmooth.smooth(
                            ROI[:, i], window_len=smoothSize, window='hanning')
    if yOffsets is None:
        yOffsets = []
        for roiIdx in range(data[drugList[0]][0].shape[0]):
            for drug in drugList:
                if yOffsets == []:
                    yOffsets.append(0)
                elif showTrials:
                    yOffsets.append(
                        yOffsets[-1] - max(
                            [np.nanmax(x[roiIdx, :, :]) - np.nanmin(
                                x[roiIdx - 1, :, :]) for x in data[drug]]))
                elif shading == 'stdev':
                    yOffsets.append(
                        yOffsets[-1] -
                        max([np.nanmax(nanmean(x[roiIdx, :, :], axis=1) +
                            np.isfinite(nanstd(x[roiIdx, :, :], axis=1)) *
                            nanstd(x[roiIdx, :, :], axis=1)) -
                            np.nanmin(
                                nanmean(x[roiIdx - 1, :, :], axis=1) -
                                np.isfinite(nanstd(x[roiIdx, :, :], axis=1)) *
                                nanstd(x[roiIdx - 1, :, :], axis=1))
                            for x in data[drug]]))
                else:
                    yOffsets.append(
                        yOffsets[-1] - max([np.nanmax(x[roiIdx, :, :].mean(
                            axis=1)) - np.nanmin(x[roiIdx - 1, :, :].mean(
                                axis=1)) for x in data[drug]]))
    assert all(np.isfinite(yOffsets))
    ymax = max(
        [np.nanmax(x[0, :, :]) for x in data[drugList[0]]]) + yOffsets[0]
    for dataIdx in range(len(data[drugList[0]])):
        ax.text(offset * dataIdx, ymax + 0.1, xlabels[dataIdx], ha='center')
        if stimulusDurations is not None:
            ax.axvspan(
                offset * dataIdx, offset * dataIdx + stimulusDurations[dataIdx],
                color='k', alpha=0.3, linewidth=0)
        yCount = 0
        for roiIdx in range(data[drugList[0]][0].shape[0]):
            for drug in drugList:
                # if np.all(np.isfinite(data[drug][dataIdx][roiIdx,:,:])):
                mean = nanmean(data[drug][dataIdx][roiIdx, :, :], axis=1)
                ax.plot(times + offset * dataIdx, mean + yOffsets[yCount],
                        colors[yCount % len(colors)], linewidth=0.5)
                if shading == 'stdev' and d.shape[2] > 1:
                    stdev = nanstd(data[drug][dataIdx][roiIdx, :, :], axis=1)
                    valid = [np.isfinite(s) for s in stdev]
                    ax.fill_between(
                        times + offset * dataIdx, mean + stdev + yOffsets[yCount],
                        mean - stdev + yOffsets[yCount], where=valid,
                        color=colors[yCount % len(colors)], linewidth=0,
                        alpha=0.4)
                if showTrials:
                    for i in range(data[drug][dataIdx].shape[2]):
                        ax.plot(
                            times + offset * dataIdx,
                            data[drug][dataIdx][roiIdx, :, i] + yOffsets[yCount],
                            colors[yCount % len(colors)], linewidth=0.1,
                            alpha=0.5)
                yCount += 1
    for yIdx, yLabel in enumerate(ylabels):
        ax.text(np.min(times) - 0.2, yOffsets[yIdx], yLabel,
                va='center', ha='right', color=colors[yIdx % len(colors)])
    xmax = offset * (len(data[drugList[0]]) - 1) + np.max(times)
    ax.set_ylim(
        [min([np.min(x[-1, :, :]) for x in data[drugList[-1]]]) + yOffsets[-1],
         ymax])
    ax.set_xlim([np.min(times), xmax])
    ax.plot(
        [xmax - markerDuration, xmax - markerDuration, xmax],
        [ymax, ymax - 1, ymax - 1], 'k', lw=0.5)  # scale markers
    ax.text(xmax - (markerDuration / 2), ymax - 0.9,
            str(markerDuration) + ' s', ha='center')
    ax.text(xmax - markerDuration - 0.5, ymax - 0.5, '100%', rotation=90,
            ha='right', va='center')
    ax.set_axis_off()


def stackedBar(
        ax, centers, heights, width=0.4, labels=None, colors=None, legend=True,
        separate_bar_colors=False, **kwargs):
    """Plots a stacked bar graph
    Inputs:
        ax: the axis of the plot
        centers: the center of each bar
        heights: the height of each sub-bar, can be a list of lists or a (Nxb) numpy array
            Where N is the number of bars and b is the number of bins/sub-bars in each bar
        width: width of bar
        labels: label for each sub-bar, displayed in top right corner
        colors: 1= or 2-d array/lists of colors for the bars. If 1-d, colors
            are the same for each stacked bar and ordered from bottom to top.
            If 2-d, len(colors) == len(centers) and
            len(colors[i]) == len(heights[i]) and separate_bar_colors should be True
        separate_bar_colors : If True, color each bar separately, colors should be the correct shape
        **kwargs: additional keyword argument pairs will get passed to ax.bar

    """

    assert len(centers) == len(heights)

    if labels is None:
        labels = ['Bin {}'.format(x) for x in np.arange(len(heights[0])) + 1]

    if colors is None:
        cc = color_cycle()
        colors = [cc.next() for _ in range(len(heights[0]))]
    else:
        colors = list(colors)

    if not separate_bar_colors:
        colors = [colors] * len(centers)

    for center, hs, bar_colors in zip(centers, heights, colors):
        bottoms = np.cumsum(hs)
        bottoms = np.hstack((0, bottoms[:-1]))
        for h, b, c in zip(hs, bottoms, bar_colors):
            ax.bar(center - width / 2, h, width, bottom=b, color=c, **kwargs)

    ax.set_xticks(centers)
    center_spacing = np.median(np.diff(centers))
    ax.set_xlim(centers[0] - center_spacing / 2,
                centers[-1] + center_spacing / 2)

    if legend:
        stackedText(ax, labels[::-1], colors[0][::-1], size=7)


def roiDataImageOverlay(
        ax, background, rois, values=None, aspect=2., vmin=None, vmax=None,
        labels=None, cax=None, bg_kwargs=None, **patch_kwargs):
    """Plots ROIs over a background image colored by any value.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        Axis to plot on.
    background : np.ndarray
        2D image onto which ROIs will be drawn.
        Accepts output from Experiment.returnFinalPrototype()
    rois : list of lists of xy coordinates
        Vertices of ROIs
        Accepts output from Experiment.roiVertices()
    values : list, optional
        If not None, used to color each ROI. One value per ROI.
    aspect : float, optional
        Aspect ratio to apply to background image.
    vmin : float, optional
        Minimum val to which values are scaled
    vmax : float, optional
        Maximum val to which values are scaled
    labels : list of str
        List of labels to print in the center of each ROI
    cax : matplotlib.pyplot.axes, optional
        If not None, plot the colorbar on this axis
    alpha : float, optional
        Alpha to apply to coloring of each ROI.
        0.0 is transparent and 1.0 is opaque.
    bg_kwargs : dict
    **patch_kwargs

    """

    if values is None:
        values = np.ones(len(rois))
    if vmin is None:
        vmin = min(values)
    if vmax is None:
        vmax = max(values)

    if bg_kwargs is None:
        bg_kwargs = {'cmap': 'gray'}

    if 'cmap' not in patch_kwargs:
        patch_kwargs['cmap'] = mpl.cm.hsv

    ax.imshow(background, aspect=aspect, **bg_kwargs)

    patches = []
    for roi in rois:
        for poly in roi:
            patches.append(Polygon(poly, closed=True))
    p = PatchCollection(patches, **patch_kwargs)
    p.set_array(values)
    p.set_clim(vmin, vmax)
    ax.add_collection(p)

    if labels is not None:
        for roi, label in zip(rois, labels):
            for poly in roi:
                center_point = np.mean(poly, axis=0)
                ax.text(center_point[0], center_point[1], label, va='center',
                        ha='center', fontsize=6)

    if cax is not None:
        cax.clear()
        cbar_ticks = np.linspace(vmin, vmax, 3)
        mpl.pyplot.colorbar(p, cax, ticks=cbar_ticks)
        mpl.pyplot.setp(cax.get_yticklabels(), fontsize=14)

    ax.set_axis_off()


def scatter_1d(ax, values, group_labels=None, bar_labels=None):
    """Compares values as 'bars' of scattered points on a single axis
    See ba.compareLickRate\compareLapRate for a usage example

    Parameters
    ----------
    ax : axis to plot on
    values : sequence of sequences of sequences
        one bar per outer sequence, one color for next sequence, scatter inner
        sequence
    group_labels : optional, sequence of sequences
        labels for each group within a scatter-bar
    bar_labels : optional, list of strings same length as values, to label
        each scatter-bar

    """
    to_label = group_labels is not None
    if group_labels is None:
        group_labels = [
            [None for color_group in bar_group] for bar_group in values]

    colors = color_cycle()
    color_dict = {}
    for label in set(flatten(group_labels)):
        c = colors.next()
        color_dict[label] = c if c != 'r' else colors.next()
    for idx, (bar, labels) in enumerate(zip(values, group_labels)):
        all_values = []
        for group, group_label in zip(bar, labels):
            all_values.extend(group)
            x = (np.random.rand(len(group)) * 0.4) - 0.2 + idx + 1
            ax.plot(x, group, '.', markersize=7, color=color_dict[group_label])
        ax.plot(idx + 1, np.mean(all_values), 'r*', markersize=10)

    if to_label:
        text_list = color_dict.keys()
        colors = [color_dict[key] for key in text_list]
        stackedText(ax, text_list, colors=colors, loc=1, size=None)

    ax.set_xticks(range(1, len(values) + 1))
    ax.set_xticklabels(bar_labels)
    ax.set_xlim(0, len(values) + 1)


def scatter_bar(ax, values, colors=None, labels=None, jitter_x=False, **plot_kwargs):
    """Compare data as bar with SEM whisker as well as all data points
    scattered within bar.

    Parameters
    ----------
    ax : matplotlib.axes
    values : sequence of sequences
        One bar per first index, averaging across second index for each bar
    labels : list of strings
        Same length as values
    jitter_x : boolean
        If true, jitters the scattered points slightly in x so they are easier
        to visualize.

    """
    x_values = np.arange(len(values)) + 0.5
    mean_values = [np.nanmean(vals) for vals in values]
    sems = [np.nanstd(vals) / np.sqrt(len(vals)) for vals in values]

    ax.bar(x_values - 0.25, mean_values, color='none', width=0.5)
    ax.errorbar(x_values, mean_values, [np.zeros(len(values)), sems],
                fmt='none', ecolor='k', capsize=0)
    for i, (x_val, vals) in enumerate(zip(x_values, values)):
        if jitter_x:
            scatter_x = (np.random.rand(len(vals)) * 0.2) - 0.1 + x_val
        else:
            scatter_x = [x_val] * len(vals)
        if colors:
            for color in set(colors[i]):
                ax.scatter(
                    scatter_x[(np.array(colors[i]) == color)[:, 0]],
                    vals[(np.array(colors[i]) == color)[:, 0]],
                    color=color, **plot_kwargs)
        else:
            ax.scatter(scatter_x, vals, color='k', **plot_kwargs)

    ax.set_xticks(x_values)
    if labels is not None:
        ax.set_xticklabels(labels)
    else:
        ax.tick_params(labelbottom=False)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, len(values))


def grouped_bar(
        ax, values, condition_labels=None, cluster_labels=None,
        bar_colors=None, scatter_points=False, scatterbar_colors=None,
        jitter_x=False, loc='best', s=40, error_bars='sem', group_spacing=0.2,
        **plot_kwargs):
    """Plot a grouped bar graph with sem.

    Parameters
    ----------
    ax : matplotlib.axes
    values : array of arrays
        The actual data to plot; len(values) is the number of conditions or
        bars in each cluster/group and len(values[0]) is the number of clusters
        of bars.
    condition_labels : list of str, optional
    cluster_labels : list of str, optional
    bar_colors : list of colors, optional
    scatter_points : bool
        If True, also scatter the data within each bar.
    scatterbar_colors : list of list of colors, optional
        Color of each point if scattering within bars. Same shape as 'values'.
    jitter_x : bool
        If True, jitter the x coordinate of each point within each bar.
    loc : string or int, optional
        Location of the legend. See matplotlib legend docs for details.
    s : float
        Area of scatter dots in points.
    error_bars : {'sem', 'std'}
        Determines whether to plot standard error or standard deviation error
        bars.
    group_spacing : float, optional
        Space between groups of bars.
    plot_kwargs
        Additional keyword arguments are passed to the plotting function.

    Example
    -------
    If we are plotting data from 3 days for wildtype and mutant mice, 'values'
    might be a list of length 2, where the first element is a list of length 3
    corresponding to data from the wildtype mice for each of the 3 days, and
    the second element of the outside list is the same data for the mutant
    mice.

    This will plot 2 bars close to each other, a larger gap, then 2 more,
    another gap, and finally the last 2 bars. The x-ticks will be labeled
    by the 'cluster_labels' argument; something like 'Day 1', 'Day 2', 'Day 3'.
    The first bar in each cluster is the same color (as are the second bars),
    as determined by 'bar_colors'. The different colors define the different
    conditions (in this example, wildtype vs. mutant mice), which are labeled
    with 'condition_labels' in the legend.

    """
    if condition_labels is None:
        condition_labels = [None] * len(values)

    if cluster_labels is None:
        cluster_labels = ['Cluster {}'.format(idx)
                          for idx in range(len(values[0]))]

    if scatter_points:
        if scatterbar_colors is None:
            scatterbar_colors = [
                [['k'] * len(cluster) for cluster in condition]
                for condition in values]
    if bar_colors is None:
        bar_colors = color_cycle()

    left_edges = np.arange(0, len(values[0]))
    bar_width = (1 - group_spacing) / float(len(values))

    for idx, label, color, data in it.izip(
            it.count(), condition_labels, bar_colors, values):
        means = [np.nanmean(vals) if len(vals) else np.nan for vals in data]
        if error_bars == 'sem':
            err = [np.nanstd(vals) / np.sqrt(np.sum(np.isfinite(vals)))
                   if len(vals) else np.nan for vals in data]
        elif error_bars == 'std':
            err = [np.nanstd(vals) if len(vals) else np.nan for vals in data]

        if scatterbar_colors is None:
            ax.bar(left_edges + (idx + 0.5)* bar_width, means, bar_width,
                   color=color, label=label, align='center', **plot_kwargs)
            if error_bars is not None:
                ax.errorbar(left_edges + (idx + 0.5) * bar_width, means, err,
                            fmt='none', ecolor='k', capsize=0)
        else:
            ax.bar(left_edges + (idx + 0.5) * bar_width, means, bar_width,
                   color='none', edgecolor=color, label=label, align='center',
                   **plot_kwargs)
            if error_bars is not None:
                ax.errorbar(left_edges + (idx + 0.5) * bar_width, means, err,
                            fmt='none', ecolor='k', capsize=0)
            for cluster_idx, left_edge, cluster_name in it.izip(
                    it.count(), left_edges, cluster_labels):
                if cluster_name == 'shuffle':
                    continue
                if jitter_x:
                    scatter_x = (
                        np.random.rand(
                            len(data[cluster_idx])) * bar_width * 0.7) + \
                        left_edge + (idx + 0.15) * bar_width
                else:
                    scatter_x = [
                        left_edge + idx * bar_width + bar_width / 2.] \
                        * len(data[cluster_idx])
                ax.scatter(
                    scatter_x, data[cluster_idx],
                    c=scatterbar_colors[idx][cluster_idx], s=s)

    ax.set_xticks(left_edges + (1 - group_spacing) / 2.0)
    ax.set_xticklabels(cluster_labels)
    ax.tick_params(axis='x', direction='out')
    ax.set_xlim(-group_spacing, len(left_edges))

    if condition_labels[0] is not None:
        ax.legend(frameon=False, loc=loc)


def grouped_line(
        ax, values, condition_labels=None, cluster_labels=None, colors=None,
        loc=1):
    """Similar to 'grouped_bar', but plots lines instead of bars.

    Parameters
    ----------
    ax : matplotlib.axes
    values : array of arrays
        The actual data to plot; len(values) is the number of conditions or
        points in each cluster/group and len(values[0]) is the number of
        discrete x values.
    condition_labels : list of str, optional
    cluster_labels : list of str, optional
    colors : list of colors, optional
    loc : string or int, optional
        Location of the legend. See matplotlib legend docs for details.

    """
    if condition_labels is None:
        condition_labels = [None] * len(values)

    if cluster_labels is None:
        cluster_labels = ['Cluster {}'.format(idx)
                          for idx in range(len(values[0]))]

    if colors is None:
        colors = color_cycle()

    x_axis = np.arange(1, 1 + len(values[0]))
    for label, color, data in it.izip(condition_labels, colors, values):
        means = [np.nanmean(vals) for vals in data]
        sems = [np.nanstd(vals) / np.sqrt(np.sum(np.isfinite(vals)))
                for vals in data]

        ax.plot(x_axis, means, color=color, label=label)
        ax.errorbar(x_axis, means, sems, fmt='none', ecolor='k', capsize=0)

    ax.set_xticks(x_axis)
    ax.set_xticklabels(cluster_labels)
    ax.tick_params(axis='x', direction='out')
    ax.set_xlim(0, len(cluster_labels) + 1)

    if condition_labels[0] is not None:
        ax.legend(frameon=False, loc=loc)


def line_o_gram(ax, values, hist_kwargs=None, **plot_kwargs):
    """Plot a 'line-o-gram'.

    This is basically a histogram with a line connecting what would be the
    middle of the top of each bar.

    Parameters
    ----------
    ax : matplotlib.axes
    values : array-like
    hist_kwargs : dict
        Keyword arguments to pass to the histogram method.
    plot_kwargs are passed to the plotting method

    """
    hist_kwargs = {} if hist_kwargs is None else hist_kwargs

    counts, bins = np.histogram(values, **hist_kwargs)

    bin_means = [
        np.mean([left, right]) for left, right in zip(bins[:-1], bins[1:])]

    ax.plot(bin_means, counts, **plot_kwargs)

    ax.set_xticks(bin_means)
    ax.set_xticklabels(['{:.2f}'.format(x) for x in bin_means])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', direction='out')


def grouped_box(
        ax, values, condition_labels=None, cluster_labels=None,
        box_colors=None, group_spacing=0.2, box_spacing=0.05, notch=True,
        loc='best', **box_kwargs):
    """Plot a grouped box-and-whisker graph.

    See grouped_bar for a detailed example of how the boxes are laid out.

    Parameters
    ----------
    ax : matplotlib.axes
    values : array of arrays
        The actual data to plot; len(values) is the number of conditions or
        boxes in each cluster/group and len(values[0]) is the number of
        clusters of boxes.
    condition_labels : list of str, optional
    cluster_labels : list of str, optional
    box_colors : list of colors, optional
    group_spacing : float, optional
        Space between groups of boxes.
    box_spacing : float, optional
        Space between boxes within each cluster.
    notch : bool
        If True, mark the confidence interval of the median with notches in the
        box. See the matplotlib boxplot documentation for details.
    loc : string or int, optional
        Location of the legend. See matplotlib legend docs for details.
    box_kwargs
        Additional arguments are passed to the box plotting, with a few
        pulled out first. See code.

    """
    n_groups = len(values[0])
    n_conditions = len(values)

    boxprops = box_kwargs.pop('boxprops', {})

    if 'medianprops' not in box_kwargs:
        box_kwargs['medianprops'] = {}
    if 'color' not in box_kwargs['medianprops']:
        box_kwargs['medianprops']['color'] = 'k'

    if condition_labels is None:
        condition_labels = [None] * n_conditions

    if cluster_labels is None:
        cluster_labels = ['Cluster {}'.format(idx)
                          for idx in range(n_groups)]

    if box_colors is None:
        box_colors = color_cycle()

    # Each cluster of boxes will be centered around [0.5, 1.5, 2.5, ...]
    box_width = (1 - group_spacing - (n_conditions - 1) * box_spacing) / \
        float(n_conditions)
    centers = np.arange(n_conditions) * (box_width + box_spacing) + \
        group_spacing / 2. + box_width / 2.

    fake_lines_for_legend = []
    for idx, label, color, data in it.izip(
            it.count(), condition_labels, box_colors, values):
        # Drop NaN's and Inf's
        # Need to casts things as an array to allow for fancy-indexing;
        # can't be a list or a pandas.Series
        data = [np.array(vals)[np.array(np.isfinite(vals))] for vals in data]

        boxprops['color'] = color
        fake_lines_for_legend.append(Line2D([], [], color=color, label=label))

        ax.boxplot(
            data, positions=np.arange(len(data)) + centers[idx],
            boxprops=boxprops, widths=box_width, notch=notch, **box_kwargs)

    ax.set_xticks(np.arange(n_groups) + 0.5)
    ax.set_xticklabels(cluster_labels)
    ax.tick_params(axis='x', direction='out')
    ax.set_xlim(0, n_groups)

    if condition_labels[0] is not None:
        ax.legend(handles=fake_lines_for_legend, frameon=False, loc=loc)


def box_and_line(
        ax, values, condition_labels=None, cluster_labels=None, colors=None,
        box_width=0.4, box_spacing=0.2, notch=True, markers=None,
        line_kwargs=None, linestyles=None, **box_kwargs):
    """Plot a line plot flanked by corresponding box plots.

    Parameters
    ----------
    ax : matplotlib.axes
    values : array of arrays
        The actual data to plot; len(values) is the number of conditions or
        boxes in each cluster/group and len(values[0]) is the number of
        clusters of boxes (must be exactly 2).
    condition_labels : list of str, optional
    cluster_labels : list of str, optional
    colors : list of colors, optional
    box_width : float, optional
        Width of each box.
    box_spacing : float, optional
        Space between each box.
    notch : bool
        If True, mark the confidence interval of the median with notches in the
        box. See the matplotlib boxplot documentation for details.
    markers : list, optional
        List of markers to use for the line plot.
    line_kwargs : dict, optional
        Additional keyword arguments passed to the line/errorbar plot function.
    **box_kwargs
        The rest of the keyword arguments will be passed to the box plotting
        function.

    Notes
    -----

    Must have exactly 2 clusters.

    All the spacing might be a bit hard to follow, but things should layout
    from the y-axis (w/ 2 conditions):
    2s + w + s + w + 2s (line left end) 1 (line right end) 2s + w + s + w + 2s,
    where s is the box_spacing and w is the box_width. The x-ticks will line up
    with lines.

    """
    n_groups = len(values[0])
    n_conditions = len(values)
    assert n_groups == 2

    # Set some default values
    if condition_labels is None:
        condition_labels = [None] * n_conditions
    if cluster_labels is None:
        cluster_labels = ['Cluster {}'.format(idx)
                          for idx in range(n_groups)]
    if colors is None:
        colors = color_cycle()

    if line_kwargs is None:
        line_kwargs = {}
    if 'capsize' not in line_kwargs:
        line_kwargs['capsize'] = 1.5
    if 'markeredgecolor' not in line_kwargs:
        line_kwargs['markeredgecolor'] = 'k'
    if 'markeredgewidth' not in line_kwargs:
        line_kwargs['markeredgewidth'] = 0.5

    if markers is None:
        markers = [None] * len(values)
    if linestyles is None:
        linestyles = ['-'] * len(values)

    boxprops = box_kwargs.pop('boxprops', {})
    legend_loc = box_kwargs.pop('loc', 'best')

    if 'medianprops' not in box_kwargs:
        box_kwargs['medianprops'] = {}
    if 'color' not in box_kwargs['medianprops']:
        box_kwargs['medianprops']['color'] = 'k'

    line_x_values = (1, 2)
    # See the Note for the logic here.
    box_centers_within_clusters = \
        np.arange(n_conditions) * (box_width + box_spacing)
    all_centers = zip(
        1 - (max(box_centers_within_clusters) +
             2. * box_spacing + box_width / 2.) + box_centers_within_clusters,
        2 + 2 * box_spacing + box_width / 2. + box_centers_within_clusters)

    for label, color, marker, ls, centers, data in it.izip(
            condition_labels, colors, markers, linestyles, all_centers,
            values):
        # Drop NaN's
        data = [np.array(vals)[np.array(np.isfinite(vals))] for vals in data]
        means = [np.mean(vals) for vals in data]
        sems = [np.std(vals) / np.sqrt(len(vals)) for vals in data]

        ax.errorbar(
            line_x_values, means, sems, color=color, ecolor=color, label=label,
            marker=marker, ls=ls, **line_kwargs)

        boxprops['color'] = color

        ax.boxplot(
            data, positions=centers, boxprops=boxprops, widths=box_width,
            notch=notch, **box_kwargs)

    ax.set_xticks(line_x_values)
    ax.set_xticklabels(cluster_labels)
    ax.tick_params(axis='x', direction='out')
    ax.set_xlim(
        all_centers[0][0] - 2 * box_spacing - box_width / 2.,
        all_centers[-1][-1] + 2 * box_spacing + box_width / 2.)

    if condition_labels[0] is not None:
        ax.legend(frameon=False, loc=legend_loc)


def swarm_plot(
        ax, values, condition_labels=None, cluster_labels=None, colors=None,
        linewidth=None, edgecolor=None, loc='best', plot_bar=False,
        bar_kwargs=None, **swarm_kwargs):
    """Plot a swarm plot.

    Similar to s a scatter bar, but the plots are laid out smartly to minimize
    overlap.
    See seaborn.swarmplot for more details.

    Parameters
    ----------
    ax : matplotlib.axes
    values : array of arrays
        The actual data to plot; len(values) is the number of conditions or
        boxes in each cluster/group and len(values[0]) is the number of
        clusters of boxes (must be exactly 2).
    condition_labels : list of str, optional
    cluster_labels : list of str, optional
    colors : list of colors, optional
    linewidth : float, optional
        The size of the edge line around the individual points.
    edgecolor : colorspec
        'gray' is a special case (see seaborn.swarmplot) that matches the
        edge color to the fill color.
    loc : string or int, optional
        Location of the legend. See matplotlib legend docs for details.
    plot_bar : bool
        If True, plot a bar around each cluster of points.
    bar_kwargs : dict, optional
        Dictionary of keyword arguments to pass to the bar plot function.
    swarm_kwargs
        Additional arguments are passed to the plotting function directly.

    """
    if plot_bar:
        linewidth = linewidth if linewidth is not None else 0.5
        edgecolor = edgecolor if edgecolor is not None else 'k'
    else:
        linewidth = linewidth if linewidth is not None else 0.2
        edgecolor = edgecolor if edgecolor is not None else 'gray'

    if condition_labels is None:
        condition_labels = [None] * len(values)

    if cluster_labels is None:
        cluster_labels = ['Cluster {}'.format(idx)
                          for idx in range(len(values[0]))]

    if colors is None:
        colors = color_cycle()

    if bar_kwargs is None:
        bar_kwargs = {}

    all_data, x_idxs, hue_idxs = [], [], []
    palette = {}
    for condition_values, color, label in it.izip(
            values, colors, condition_labels):
        for cluster_idx, cluster_values in it.izip(
                it.count(), condition_values):
            all_data.extend(cluster_values)
            x_idxs.extend([cluster_idx] * len(cluster_values))
            hue_idxs.extend([label] * len(cluster_values))
        palette[label] = color

    if plot_bar:
        sns.swarmplot(
            ax=ax, x=x_idxs, y=all_data, hue=hue_idxs,
            palette={label: 'w' for label in condition_labels},
            split=True, linewidth=linewidth, edgecolor=edgecolor,
            **swarm_kwargs)
        sns.barplot(
            ax=ax, x=x_idxs, y=all_data, hue=hue_idxs, palette=palette,
            **bar_kwargs)
    else:
        sns.swarmplot(
            ax=ax, x=x_idxs, y=all_data, hue=hue_idxs, palette=palette,
            split=True, linewidth=linewidth, edgecolor=edgecolor,
            **swarm_kwargs)

    ax.set_xticklabels(cluster_labels)

    if condition_labels[0] is not None:
        if plot_bar:
            # Only plot the bars
            handles, labels = ax.get_legend_handles_labels()
            args = [
                (h, l) for h, l in zip(handles, labels) if
                isinstance(h, mpl.container.BarContainer)]
            ax.legend(*zip(*args), frameon=False, loc=loc)
        else:
            ax.legend(frameon=False, loc=loc)
