"""
Functions that perform statistical analysis and display (print and plot)
the results of multiple datasets.

While some of these functions are specific for the analysis of
presynaptic_stats.py scripts, most are general and can be used for other
scripts.

Parameters needed for printing and plotting have to be provided in a module,
which as passed as argument pp to all relevant functions.

This was previously part of presynaptic_stats.py.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id: multi_dataset_util.py 2140 2024-12-19 14:11:52Z vladan $
"""

__version__ = "$Revision: 2140 $"

import sys
import os
import importlib
import logging
#import itertools
from copy import copy, deepcopy
#import pickle
from types import ModuleType
import pickle

import numpy
import scipy
import scipy.stats
import pandas as pd
try:
    import matplotlib as mpl
    mpl_major_version = mpl.__version__[0]
    import matplotlib.pyplot as plt
except ImportError:
    pass

import pyto
from pyto.util.exceptions import ReferenceError
from pyto.analysis.groups import Groups
from pyto.analysis.observations import Observations


##############################################################
#
# Higher level stats and plot functions
#

def analyze_occupancy(
        layer, bins, bin_names, pixel_size, pp, groups=None, identifiers=None,
        test=None, reference=None, ddof=1, out=sys.stdout, outNames=None,
        plot_=True, plot_type='bar', title='', yerr='sem', confidence='stars',
        y_label=None):
    """
    Statistical analysis of sv occupancy divided in bins according to the
    distance to the AZ.

    Arguments:
      - layer: (Layers) layer data structure
      - bins: (list) distance bins
      - bin_names: (list) names of distance bins, has to correspond to bins
      - pp: module containing print and plot parameters
      - groups: list of group names
      - test: statistical inference test type
      - reference: specifies reference data
      - ddof: differential degrees of freedom used for std
      - out: output stream for printing data and results
      - outNames: list of statistical properties that are printed
      - plot_: flag indicationg whether the data should be plotted
      - title: plot title
      - yerr: name of the statistical property used for y error bars
      - confidence: determines how confidence is plotted
      - y_label: y axis label (default 'occupancy')
    """

    # Note: perhaps should be moved to pyto.analysis.Layers

    # rearange data by bins
    layer_bin = layer.rebin(bins=bins, pixel=pixel_size, categories=groups)

    # make a separate Groups object for each bin
    layer_list = layer_bin.splitIndexed()

    # convert to Groups where group names are distance bins and identifiers
    # are treatment names
    converted = pyto.analysis.Groups.joinExperimentsList(
        groups=groups, identifiers=identifiers,
        list=layer_list, listNames=bin_names, name='occupancy')

    # do statistics and plot
    result, ax = stats(
        data=converted, name='occupancy', join=None, pp=pp, groups=bin_names,
        identifiers=groups, test=test, reference=reference, ddof=ddof,
        out=out, outNames=outNames, plot_=plot_, plot_type=plot_type,
        title=title, yerr=yerr, label='experiment', confidence=confidence,
        y_label=y_label)

    return result, ax

def stats_list(
        data, dataNames, name, pp, join='join', bins=None, fraction=1,
        groups=None, identifiers=None, test=None, reference=None,
        ddof=1, out=sys.stdout, plot_name=None, label=None, outNames=None,
        plot_=True, yerr='sem', confidence='stars', plot_type='bar',
        randomized_x=False, title='', x_label=None, y_label=None):
    """
    Statistical analysis of data specified as a list of Groups objects.

    First, the data from idividual observations of each group are joined. In
    this way each group (of arg data) becomes one observation and the elements
    of data list become groups.

    Arguments:
      - data: (list of Groups) list of data structures
      - dataNames: (list of strs) names corrensponfing to elements of arg data,
      have to be in the same order as the data
      - name: name of the analyzed property
      - pp: module containing print and plot parameters
      - join: 'join' to join experiments, 'mean' to make mean of
      experements or 'mean_bin' to make means within bins (arg bins required)
      - bins: (list) bins for making histogram
      - fraction: bin index for which the fraction is calculated
      - groups: list of group names
      - identifiers: list of identifiers
      - test: statistical inference test type
      - reference: specifies reference data
      - plot_name: name of the statistical property to plot
      - plot_type: plot type, 'bar' or 'boxplot'
      - ddof: differential degrees of freedom used for std
      - out: output stream for printing data and results
      - outNames: list of statistical properties that are printed
      - yerr: name of the statistical property used for y error bars
      - plot_: flag indicating if the result are to be plotted
      - label: determines which color, alpha, ... is used, can be 'group' to
      label by group or 'experiment' to label by experiment
      - confidence: determines how confidence is plotted
      - x_label: x axis label
      - y_label: y axis label, if not specified arg name used instead
      - title: title
    """

    # make one groups object by joining observations
    together = Groups.joinExperimentsList(
        list=data, listNames=dataNames, name=name, mode=join,
        groups=groups, identifiers=identifiers)

    # do stats
    result, ax = stats(
        data=together, name=name, pp=pp, join=None, groups=dataNames,
        identifiers=groups, bins=bins, fraction=fraction, test=test,
        reference=reference, ddof=ddof, out=out, outNames=outNames, yerr=yerr,
        label='experiment', plot_=plot_,
        plot_name=plot_name, plot_type=plot_type, randomized_x=randomized_x,
        confidence=confidence, title=title, x_label=x_label, y_label=y_label)

    return result, ax

def stats_list_pair(
        data, dataNames, name, pp, groups=None, identifiers=None,
        test='t_rel', reference=None,  out=sys.stdout, yerr='sem', ddof=1,
        outNames=None, plot_=True, label=None, confidence='stars',
        title='', x_label=None, y_label=None):
    """
    Statistical analysis of paired data specified as a list of Groups objects.

    Unlike in stats_list(), the data has to be paired so that all Groups
    objects (elements of arg data) have to have the same group names, and the
    same identifiers.

    First, the means of the data (arg name) from all idividual observations of
    each group are calculated. In this way each group (of arg data) becomes one
    observation and the elements of data list become groups.

    Arguments:
      - data: (list of Groups) list of data structures
      - dataNames: (list of strs) names corrensponfing to elements of arg data,
      have to be in the same order as the elements of data
      - name: name of the analyzed property
      - pp: module containing print and plot parameters
      - groups: list of group names
      - identifiers: list of identifiers
      - test: statistical inference test type (default 't_rel')
      - reference: specifies reference data
      - ddof: differential degrees of freedom used for std
      - out: output stream for printing data and results
      - outNames: list of statistical properties that are printed
      - yerr: name of the statistical property used for y error bars
      - plot_: flag indicating if the result are to be plotted
      - label: determines which color, alpha, ... is used, can be 'group' to
      label by group or 'experiment' to label by experiment
      - confidence: determines how confidence is plotted
      - x_label: x axis label
      - y_label: y axis label, if not specified arg name used instead
      - title: title
    """

    # make one groups object by joining observations
    together = Groups.joinExperimentsList(
        list=data, listNames=dataNames, name=name, mode='mean',
        removeEmpty=False)

    # do stats
    result, ax = stats(
        data=together, name=name, pp=pp, join='pair', groups=dataNames,
        identifiers=groups, test=test, reference=reference, plot_=plot_,
        ddof=ddof, out=out, outNames=outNames, yerr=yerr, label='experiment',
        confidence=confidence, title=title, x_label=x_label, y_label=y_label)

    return result, ax

def stats(
        data, name, pp, bins=None, bin_names=None, fraction=None, join=None,
        groups=None, identifiers=None, test=None, test_mean='kruskal',
        reference=None, ddof=1, out=sys.stdout, label=None,
        outNames=None, plot_=True, plot_name=None, yerr='sem',
        plot_type='bar', randomized_x=False, confidence='stars',
        title='', x_label=None, y_label=None):
    """
    Does statistical analysis of data specified by args data and name, prints
    and plots the results as a bar chart.

    Argument join determines how the data is pooled across experiments.
    If join is 'join', data of individual experiments (observations) are
    joined (pooled)  together within a group to be used for further
    analysis. If it is 'mean', the mean value for each experiment is
    calculated and these means are used for further analysis.

    Argument bins determines how the above obtained data is further
    processed. If arg bins is not specified, basic stats (mean, std, sem)
    are calculated for all groups and the data is statistically compared
    between the groups.

    If arg bins is specified, histograms of the data are calculated
    for all groups. A histogram can show the number of events (property
    name 'histogram') or probabilities (number events of each group
    normalized; property name 'probability'). Histograms are
    statistically compared between groups. The probability for bin
    indexed by arg fraction is saved separately as property 'fraction'.

    For example, fraction of connected vesicles is obtained for
    name='n_connection', bins=[0,1,100], fraction=1. More detailed
    explanation for statistics used is given below.

    Histogram normalization (when arg bins is specified) is performed
    based only on values that fall within bins (excluding the values
    outside the bins) when arg join is 'join'. However, when arg join
    is 'mean_bin', all values are taken into account to calculate
    fractions within bins.

    When arg bins is specified and arg fraction is None, arg plot_name
    should be specified.

    Joins 'join_bins' and 'byIndex' are described below. Specifically,
    the following types of analysis are implemented:

      - join is None: a value is printed and a bar is plotted for each
      experiment. This value is either the value of the specified property if
      scalar, or a mean of the property values if indexed. If the data is
      indexed, both significance between groups and between experiments
      are calculated.

      - join='join', bins=None: Data is pooled across experiments of
      the same group, basic stats are calculated within groups and
      statistically compared between groups.

      - join='mean', bins=None: Mean values are calculated for all
      experiments, basic stats are calculated for means within groups
      and statistically compared between groups.

      - join='join', bins specified (not None), fractions not specified:
      Data is pooled across experiments of the same group, histograms
      (acording to arg bins) of the data values are calculated within
      groups. If arg plot_name is 'histogram', number of elements within
      bins are plotted. If arg plot_name is 'probability', normalized
      values (fractions or probabilities of the data elements in all bins)
      are plotted. In both cases, numbers of elements (histograms) are
      used for statistical comparison among groups.

      - join='join', bins and fractions specified: In this case, two
      statistics are calculated. First, the histograms are statistically
      compared as in the case when arg fraction is not specifed, but only
      the bin selected by arg fraction is plotted. The test used is specified
      by arg test. Second, the statistics are also calculated as in the
      join='mean-bin' case and the test used is specified by arg test_mean.
      This case is explained below, in short, the fractions of elements in
      the specified bin is calculated for each experiment, and these
      fractions are statistically compared between groups. The sem values
      from the second statistics are used for the error bars.

      - join='mean', bins specified (not None): Mean values are
      calculated for all experiment, histograms (acording to arg bins)
      of the means are calculated within groups and statistically
      compared between groups.

      - join='mean_bin', bins and fraction have to be specified (not None):
      Histograms of the data values are calculated for each experiment
      (acording to arg bins) and normalized to 1, basic stats are
      calculated for values of the bin specified by arg within groups,
      and statistically compared between groups

      - join='byIndex', bins should not be specified: Basic stats
      (mean, std, sem) are calculated for each index (position)
      separately. Data has to be indexed, and all experiments within
      one group have to have same ids. Arg plot_name has to be None.

    If specified, args groups and identifiers specify the order of groups
    and experiments on the x axis.

    Arg plot_name specifies the statistical property to be plotted. It has
    to be specified only if arg bins is specified and arg fraction is not.
    In this case it can be:
      - 'histogram' (default): plot histogram (number of occurences)
      - 'probability': plot probability (total occurences within a group
      are normalized to 1)

    In other cases, arg plot_name should not be specified, except for
    advanced usage. For a reference, the property to be plotted is
    determined in the following way:
      - 'mean' / 'data', if arg bins are not given for bar / boxplot
      - 'fraction' / 'fraction_data'. if args bins and fraction are specified
      and arg join is 'join'
      - 'mean' / 'data'. if args bins and fraction are specified
      and arg join is 'mean-bin'

    Arguments:
      - data: (Groups or Observations) data structure
      - name: name of the analyzed property
      - pp: module containing print and plot parameters
      - join: 'join' to join experiments, otherwise None
      - bins: (list) bins for making histogram
      - bin_names: bin names (has to have one element less than arg bins)
      - fraction: bin index for which the fraction is calculated
      - groups: list of group names
      - identifiers: list of identifiers
      - test: statistical inference test type
      - test_mean: additional statistical inference test type, used only when
      arg join is 'join', args bins and fractions are specified
      - reference: specifies reference data
      - ddof: differential degrees of freedom used for std
      - out: output stream for printing data and results
      - outNames: list of statistical properties that are printed
      - yerr: name of the statistical property used for y error bars
      - plot_: flag indicating if the result are to be plotted
      - plot_name: name of the calculated property to plot
      - plot_type: plot type, currently:
          * bar: barplot
          * boxplot: Tukey boxplot with whiskers (at 1.5 IQR) and outliers,
            box filled with color
          * boxplot_data: boxplot without outliers, box empty together with
            all data
      - randomized_x: Flag indicating if x values should be randomized for
      boxplot_data, so that data don't overlap
      - confidence: None for not showing confidence, 'stars' or 'number' to
      show confidence as stars or confidence number, respectivly
      - label: determines which color, alpha, ... is used, can be 'group' to
      label by group or 'experiment' to label by experiment
      - confidence: determines how confidence is plotted
      - x_label: x axis label
      - y_label: y axis label, if not specified arg name used instead
      - title: title

    Returns: stats, axes

    ToDo: include stats_x in stats
    """

    # sanity check:
    if (fraction is not None) and (plot_name is None) and (join == 'join'):
        plot_name = 'fraction'
        #print("Arg plot_name was None, set to 'fraction'")

    # prepare for plotting
    if plot_:
        #plt.figure()
        fig, axes = plt.subplots()

    # makes sure elements of bin_names are different from those of
    # category_label (appends space(s) to make bin_names different)
    if bin_names is not None:
        fixed_bin_names = []
        for bin_nam in bin_names:
            while(bin_nam in pp.category_label):
                bin_nam = bin_nam + ' '
            fixed_bin_names.append(bin_nam)
        bin_names = fixed_bin_names

    # determine which property to plot
    if plot_name is None:
        if (bins is None) or (join == 'mean_bin'):
            plot_name = 'mean'
        else:
            if bin_names is None:
                plot_name = 'fraction'
            else:
                # should not really get here, so set a reasonable default
                #print('Should try to avoid getting here')
                plot_name='histogram'

    # change plot_name for boxplot
    if ((plot_type == 'boxplot') or (plot_type == 'boxplot_data')):
        if plot_name == 'fraction':
            plot_name = 'fraction_data'
        elif plot_name == 'mean':
            plot_name = 'data'
        else:
            raise ValueError(
                "Plot_type 'boxplot' requires plot_name 'fraction' or 'mean'.")

    # figure out if indexed
    indexed = name in list(data.values())[0].indexed

    if isinstance(data, Groups):
        if not indexed:

            # Groups, not indexed
            if join is None:

                # Groups, join=None, non-indexed, bins
                # non-implemented argument combination
                if bins is not None:
                    raise ValueError(
                        "Binning is not implemented for scalar data")

                # Groups, join=None, non-indexed
                data.printStats(
                    out=out, names=[name], groups=groups,
                    identifiers=identifiers, format_=pp.print_format,
                    title=title)
                if plot_:
                    plot_stats_dict = plot_stats(
                        stats=data, name=name, pp=pp, groups=groups,
                        plot_type=plot_type, randomized_x=randomized_x,
                        identifiers=identifiers, yerr=None, confidence=None,
                        axes=axes)

            elif join == 'join':

                # Groups, join=join, non-indexed
                stats = data.joinAndStats(
                    name=name, mode=join, groups=groups,
                    identifiers=identifiers, test=test, reference=reference,
                    ddof=ddof, out=out, outNames=outNames,
                    format_=pp.print_format, title=title)
                if plot_:
                    plot_stats_dict = plot_stats(
                        #stats=stats, name=plot_name, pp=pp,
                        # work in progress
                        stats=stats, identifiers=groups, name=plot_name, pp=pp,
                        randomized_x=randomized_x,
                        plot_type=plot_type, yerr=yerr, confidence=confidence,
                        axes=axes)

            else:
                raise ValueError(
                    "For Groups data and non-indexed (scalar) property "
                    + "argument join can be None or 'join'.")

        else:

            # Groups, indexed
            if (join is None) or (join == 'pair'):

                # Groups, join=None or pair, indexed

                # stats between groups and  between observations
                if groups is None:
                    groups = list(data.keys())

                # stats between experiments
                exp_ref = {}
                for categ in groups:
                    # Note: makes exp_ref a dict of dicts, which makes this
                    # function work when called from stats_list()
                    exp_ref[categ] = reference
                if join is None:
                    exp_test = test
                elif join == 'pair':
                    exp_test = None
                stats = data.doStats(
                    name=name, bins=bins, fraction=fraction, groups=groups,
                    test=exp_test, between='experiments',
                    reference=exp_ref, ddof=ddof, identifiers=identifiers,
                    format_=pp.print_format, out=None)

                # stats between groups
                if data.isTransposable() and (len(groups) > 0):
                    group_ref = {}
                    for ident in data[groups[0]].identifiers:
                        group_ref[ident] = reference
                    try:
                        stats_x = data.doStats(
                            name=name, bins=bins, fraction=fraction,
                            groups=groups, identifiers=identifiers, test=test,
                            between='groups', reference=group_ref, ddof=ddof,
                            format_=pp.print_format, out=None)
    # ToDo: include stats_x in stats
                        names_x = ['testValue', 'confidence']
                    except ReferenceError:
                        stats_x = None
                        names_x = None
                else:
                    stats_x = None
                    names_x = None

                # print
                stats.printStats(
                    out=out, groups=groups, identifiers=identifiers,
                    format_=pp.print_format, title=title,
                    other=stats_x, otherNames=names_x)

                # plot: Groups, join=None or pair, indexed
                if ((plot_name != 'histogram')
                        and (plot_name != 'probability')):

                    # plot: Groups, join=None or pair, indexed, not histogram
                    if plot_:
                        plot_stats_dict = plot_stats(
                            stats=stats, name=plot_name, pp=pp, groups=groups,
                            plot_type=plot_type, randomized_x=randomized_x,
                            identifiers=identifiers, yerr=yerr, label=label,
                            confidence=confidence, stats_between=stats_x,
                            axes=axes)

                else:

                    # plot: Groups, join=None or pair, indexed, histogram
                    if fraction is not None:

                        # fraction given so split histogram and plot
                        stats_split = stats.splitIndexed()
                        histo_groups = stats_split[fraction]
                        if plot_:
                            plot_stats_dict = plot_stats(
                                stats=histo_groups, name=plot_name, pp=pp,
                                groups=groups, identifiers=identifiers,
                                yerr=yerr, randomized_x=randomized_x,
                                confidence=confidence, label='experiment',
                                axes=axes)

                    else:

                        # fraction is None
                        if join is None:
                            # should not be here
                            raise ValueError(
                                "Unsupported plotting case: indexed variable, "
                                + " join=None, but bins specified.")

                        else:
                            # Groups, join=pair, indexed, plot_name
                            # fraction is None
                            # not sure if should get here
                            logging.warning(
                                "This plotting case needs to be checked.")
                            if plot_:
                                plot_stats_dict = plot_stats(
                                    stats=histo_groups, name=plot_name, pp=pp,
                                    groups=bin_names, identifiers=groups,
                                    yerr=yerr,
                                    randomized_x=randomized_x,
                                    confidence=confidence, label='experiment',
                                    axes=axes)

            elif (join == 'join') or (join == 'mean') or (join == 'mean_bin'):

                # Groups, join=join, mean or mean_bin, indexed
                stats = data.joinAndStats(
                    name=name, bins=bins, fraction=fraction, mode=join,
                    test=test, reference=reference, groups=groups,
                    identifiers=identifiers,
                    ddof=ddof, out=out, format_=pp.print_format, title=title)

                # include sem of the stats ananlysis obtained by join=mean_bin
                if ((join == 'join') and (bins is not None)
                        and (fraction is not None)):

                    if ((plot_name == 'probability') or
                            (plot_name == 'histogram')):
                        print(
                            "Warning: error bars might be wrong (all the same)")

                    # Groups, join=join, indexed, bins, fraction
                    title_mean_bin = (
                        title + " (statistics for join='mean_bin', sem used "
                        + "for the graph)")
                    stats_mean_bin = data.joinAndStats(
                        name=name, bins=bins, fraction=fraction,
                        mode='mean_bin', test=test_mean, reference=reference,
                        groups=groups, identifiers=identifiers,
                        ddof=ddof, out=out, format_=pp.print_format,
                        title=title_mean_bin)
                    stats.addData(
                        source=stats_mean_bin, names={'sem': 'sem_mean_bin'})
                        #identifiers=groups)
                    yerr = 'sem_mean_bin'

                if ((plot_name != 'histogram')
                        and (plot_name != 'probability')):

                    # Groups, join=join, mean or mean_bin, indexed
                    # just plot
                    if plot_:
                        plot_stats_dict = plot_stats(
                            stats=stats, name=plot_name, pp=pp,
                            identifiers=groups, randomized_x=randomized_x,
                            yerr=yerr, plot_type=plot_type,
                            confidence=confidence, axes=axes)

                else:

                    # Groups, join, mean or mean_bin, plot_name
                    # split histogram and plot
                    stats_split = stats.splitIndexed()
                    histo_groups = Groups()
                    histo_groups.fromList(groups=stats_split, names=bin_names)

                    if plot_:
                        plot_stats_dict = plot_stats(
                            stats=histo_groups, name=plot_name, pp=pp,
                            groups=bin_names, identifiers=groups, yerr=yerr,
                            randomized_x=randomized_x,
                            confidence=confidence, label='experiment',
                            axes=axes)
                    stats = histo_groups

            else:
                raise ValueError(
                    "For Groups data and indexed property argument join "
                    + "can be None, 'join', 'mean' or 'mean_bin'.")

    elif isinstance(data, list):

        # list of groups
        raise ValueError("Please use stats_list() instead.")

    else:
        raise ValueError("Argument data has to be an instance of Groups "
                         + "or a list of Groups objects.")

    # finish plotting
    if plot_:
        axes.set_title(title)
        if y_label is None:
            y_label = name
        axes.set_ylabel(y_label)
        if x_label is not None:
            axes.set_xlabel(x_label)

        if pp.legend:
            # don't plot legend only if plot_stats says 'legend_done'
            try:
                if not plot_stats_dict.get('legend_done', False):
                    axes.legend()
            except (NameError, AttributeError):
                axes.legend()

        #plt.show()  # commented out to alow changing plot in notebook

    if indexed or (join is not None):
        if plot_: 
            return stats, axes
        else:
            return stats, None
    else:
        if plot_:
            return None, axes

def count_histogram(
        data, pp, name='ids', dataNames=None, groups=None, identifiers=None,
        test=None, reference=None, out=sys.stdout, outNames=None,
        plot_=True, label=None, plot_name='fraction', confidence='stars',
        title='', x_label=None, y_label=None):
    """
    Analyses and plots number of data points of a property specified by
    arg name, of multiple data objects.

    If (arg) data is a Groups object, all groups have to have the same
    experiment identifiers. Data from experiments having the same identifier
    are combined accross groups and a histogram of the number of data
    points is obtained for each identifier. The histograms are then compared
    statistically. In this case, the specified property (arg name) has to
    be indexed.

    If (arg) data is a list of Groups objects, all elements of the list
    have to have the same group names. Data from all groups having the
    same name are combined and a histogram of the number of data
    points is obtained for each group name. The histograms are then compared
    statistically.

    This method can be regarded as complementary to stats(join='join')
    because the former puts together all values from experiments
    having the same identifiers accros groups, while the latter
    puts together elements of all experiments within a group.

    Arguments:
      - data: (list of Groups) list of data structures
      - name: name of the analyzed property
      - dataNames: (list of strs) names corrensponfing to elements of arg data,
      have to be in the same order as the elements of data
      - pp: module containing print and plot parameters
      - groups: list of group names
      - identifiers: list of identifiers
      - test: statistical inference test type
      - reference: specifies reference data
      - out: output stream for printing data and results
      - outNames: list of statistical properties that are printed
      - plot_: flag indicating if the result are to be plotted
      - plot_name: determines which values are plotted, can be 'count' for the
      number of elements in each bin, or 'fraction' for the fraction of
      elements in respect to all bins belonging to the same histogram.
      - label: determines which color, alpha, ... is used, can be 'group' to
      label by group or 'experiment' to label by experiment
      - confidence: determines how confidence is plotted
      - x_label: x axis label
      - y_label: y axis label, if not specified arg name used instead
      - title: title
    """

    # make Groups object if data is a list
    if isinstance(data, list):

        # join list
        class_ = data[0].__class__
        groups_data = class_.joinExperimentsList(
            list=data, name=name, listNames=dataNames,  mode='join',
            groups=groups, identifiers=identifiers)

        # do stats
        stats = groups_data.countHistogram(
            name=name, test=test, reference=reference,
            groups=dataNames, identifiers=groups,
            out=out, outNames=outNames, format_=pp.print_format, title=title)

        # adjust group and identifiers
        loc_groups = dataNames
        loc_identifiers = groups

    elif isinstance(data, Groups):

        # check
        if not data.isTransposable():
            raise ValueError(
                "Argument data has to be transposable, that is each group "
                + "has to contain the same experiment identifiers.")

        # do stats
        stats = data.countHistogram(
            #name=name, groups=None, identifiers=groups, test=test,
            name=name, groups=groups, identifiers=identifiers, test=test,
            reference=reference, out=out, outNames=outNames,
            format_=pp.print_format, title=title)

        # groups and identifiers
        loc_groups = None
        loc_identifiers = None

    else:
        raise ValueError(
            "Argument data has to be a Groups instance or a list"
            + " of Groups instances.")

    # plot
    if plot_:

        # prepare for plotting
        #plt.figure()
        fig, axes = plt.subplots()

        # plot
        plot_stats(
            stats=stats, name=plot_name, pp=pp, yerr=None, groups=loc_groups,
            identifiers=loc_identifiers, label=label, confidence=confidence,
            axes=axes)

        # finish plotting
        plt.title(title)
        if y_label is None:
            y_label = name
        plt.ylabel(y_label)
        if x_label is not None:
            plt.xlabel(x_label)
        if pp.legend:
            plt.legend()
        #plt.show()  # commented out to alow changing plot in notebook

    return stats, axes

def correlation(
        xData, xName, yName, pp, yData=None, test=None, regress=True,
        reference=None, groups=None, identifiers=None, join=None, plot_=True,
        out=sys.stdout, format_=None, title='', x_label=None, y_label=None):
    """
    Correlates two properties and plots them as a 2d scatter graph. The
    properties are specified by data objects (args xData and yData) and
    property names (args xName and yName, respectively).

    The specified data (args xData and yData) have to be Groups or 
    Observations instances. They have to be of the same type and have
    the same structures: keys, experiment identifiers and ids. If 
    yData is None, xData is used for both properties (args xName and yName). 

    If the specified data are Groups instances (or xData is Groups and 
    yData is None), the correlation is done separately on each of the 
    Observations instances that form the values of these instances.

        The correlation is done separately on each of the Observations 
        instances that form the values of this instance.

        If mode is None, data from each experiment is analyzed separately.
        
        If mode is 'join', data specified by args xName and yName are 
        data from all experiments are taken together. They are processed 
        depending on whether they are indexed. Specifically:
          - If both xName and yName data are indexed, the number of 
          correlation points equals the total number of indexed values 
          for all experiments. 
          - If both xName and yName data are scalar, the number of 
          correlation points equals the total number of experiments
          - If one is scalar and the other indexed, the same scalar value 
          is used with all indexed values (within one experiemnt. The 
          number of correlation points equals the total number of indexed 
          values for all experiments. 

        If mode is 'mean', the mean value of indexed data is calculated and
        used for the correlation. The scalar values are not changed. The 
        number of correlation points equals the total number of experiments.

        If new is True, a new object is created to hold the data and results.
        But if mode is 'join' or 'mean', arg new is ignored (effectively set 
        to True).

        If new is True, and the resulting object has no identifiers, all
        data properties are initialized to [].

        If specified, arg identifiers determines the order of identifiers 
        in the resulting instance. If None, all existing identifiers are used.

        The results are saved as the following attributes:
          - xData, yData: deepcopied xName, yName data, only if new is True
          - n: number of individual data points
          - testValue: correlation test value
          - testSymbol: currently 'r' or 'tau', depending on the test
          - confidence: confidence
          - aRegress, bRegress: slope and intercept of the regression line 
          (if arg regress is True)

    Arguments:
      - xData, yData: (Groups or Observations) structures containing data
      - xName, yName: names of the correlated properties
      - pp: module containing print and plot parameters
      - test: correlation test type
      - regress: flag indicating if regression (best fit) line is calculated
      - reference: currently not used
      - groups: list of group names
      - identifiers: list of identifiers
      - join: None to correlate data from each experiment separately, or 'join'
      to join experiments belonging to the same group
      - out: output stream for printing data and results
      - title: title
      - x_label, y_label: x and y axis labels, if not specified args xName
      and yName are used instead
    """

    # combine data if needed
    if yData is not None:
        data = deepcopy(xData)
        data.addData(source=yData, names=[yName])
    else:
        data = xData

    # set default
    if format_ is None:
        format_ = pp.print_format

    # set regression paramets
    if regress:
        fit = ['aRegress', 'bRegress']
    else:
        fit = None

    # start plotting
    if plot_:
        #fig = plt.figure()
        fig, axes = plt.subplots()

    if isinstance(data, Groups):

        # do correlation and print
        corr = data.doCorrelation(
            xName=xName, yName=yName, test=test, regress=regress,
            reference=reference, mode=join, groups=groups,
            identifiers=identifiers, out=out, format_=format_,
            title=title)

        # plot
        if plot_:
            plot_2d(
                x_data=corr, x_name='xData', y_name='yData', pp=pp, groups=None,
                identifiers=groups, graph_type='scatter', fit=fit, axes=axes)

    elif isinstance(data, Observations):

        # do correlation and print
        corr = data.doCorrelation(
            xName=xName, yName=yName, test=test,  regress=regress,
            reference=reference, mode=join, out=out,
            identifiers=identifiers, format_=format_, title=title)

        # plot
        if plot_:
            plot_2d(
                x_data=corr, x_name='xData', y_name='yData', pp=pp, axes=axes,
                identifiers=identifiers, graph_type='scatter', fit=fit)

    else:
        raise ValueError("Argument data has to be an instance of "
                         + "pyto.analysis.Groups or Observations.")

    # finish plotting
    if plot_:
        axes.set_title(title)
        if x_label is None:
            x_label = xName
        axes.set_xlabel(x_label)
        if y_label is None:
            y_label = yName
        axes.set_ylabel(y_label)
        if pp.legend:
            axes.legend()
        #plt.show()  # commented out to alow changing plot in notebook

    return corr, axes


##############################################################
#
# Plot functions
#

def plot_layers(
        data, pp, yName='occupancy', xName='distance_nm', yerr=None,
        groups=None, identifiers=None, mode='all', ddof=1, graphType='line',
        x_label='Distance to the AZ [nm]', y_label='Vesicle occupancy',
        plot_=True, title=''):
    """
    Plots values of an indexed property specified by arg yName vs. another
    indexed property specified by arg xName as a line plot. Makes separate
    plots for each group of the arg groups.

    Plots sv occupancy by layer for if Layers object is given as arg data and
    the default values of args xName and yName are used.

    If mode is 'all' or 'all&mean' data from all observations (experiments) of
    one group is plotted on one figure. If mode is 'all&mean' the group mean is
    also plotted. If mode is 'mean' all group means are plotted together.

    Arguments:
      - data: (Groups or Observations) data structure
      - xName, yName: name of the plotted properties
      - yerr: property used for y-error
      - pp: module containing print and plot parameters
      - groups: list of group names
      - identifiers: list of identifiers
      - mode: 'all', 'mean' or 'all&mean'
      - ddof = difference degrees of freedom used for std
      - graphType: 'line' for line-graph or 'scatter' for a scatter-graph
      - x_label, y_label: labels for x and y axes
      - title: title (used only if mode is 'mean')

    Returns:
      - axes (list of axes objects) if data is instance of Groups and
      mode is 'all' or 'all&mean'
      - stats, axes: if data is instance of Groups and mode is 'mean'
      = axes: if data is instance of Observations
    """

    # plot ot not
    if not plot_:
        return

    # if data is Groups, print a separate figure for each group
    if isinstance(data, Groups):
        if groups is None:
            groups = list(data.keys())

        if (mode == 'all') or (mode == 'all&mean'):

            # a separate figure for each group
            axes_list = []
            for group_name in groups:
                title = pp.category_label.get(group_name, group_name)
                axes = plot_layers_one(
                    data=data[group_name], yName=yName, xName=xName, pp=pp,
                    yerr=yerr, identifiers=identifiers, mode=mode,
                    graphType=graphType,
                    x_label=x_label, y_label=y_label, title=title)
                axes_list.append(axes)
            ret = axes_list

        elif mode == 'mean':

            # calculate means, add distance_nm and plot (one graph)
            stats = data.joinAndStats(
                name=yName, mode='byIndex', groups=groups,
                identifiers=identifiers, ddof=ddof, out=None, title=title)
            for group_name in groups:
                dist = data[group_name].getValue(
                    property=xName, identifier=data[group_name].identifiers[0])
                stats.setValue(property=xName, value=dist,
                               identifier=group_name)
            axes = plot_layers_one(
                data=stats, yName='mean', xName=xName, pp=pp, yerr=yerr,
                identifiers=None, mode=mode, graphType=graphType, ddof=ddof,
                x_label=x_label, y_label=y_label, title=title)
            ret = stats, axes

    elif isinstance(data, Observations):

        # Observations: plot one graph
        axes = plot_layers_one(
            data=data, yName=yName, xName=xName, pp=pp, yerr=yerr,
            identifiers=identifiers, mode=mode, graphType=graphType, ddof=ddof,
            x_label=x_label, y_label=y_label)
        ret = axes

    else:
        raise ValueError(
            "Argument 'data' has to be either pyto.analysis.Groups"
            + " or Observations.")

    return ret

def plot_layers_one(
        data, pp, yName='occupancy', xName='distance_nm', yerr=None,
        identifiers=None, mode='all', ddof=1, graphType='line',
        x_label='Distance to the AZ', y_label='Vesicle occupancy', title=''):
    """
    Plots values of an indexed property specified by arg yName vs. another
    indexed property specified by arg xName as a line plot.

    Only one group can be specified as arg data. Data for all observations
    (experiments) of that group are plotted on one graph.

    Arguments:
      - data: (Observations) data structure
      - pp: module containing print and plot parameters
      - xName, yName: name of the plotted properties
      - yerr: property used for y-error
      - groups: list of group names
      - identifiers: list of identifiers
      - mode: 'all', 'mean' or 'all&mean'
      - ddof = difference degrees of freedom used for std
      - graphType: 'line' for line-graph or 'scatter' for a scatter-graph
      - x_label, y_label: labels for x and y axes
      - title: title

    Returns axes
    """
    # from here on plotting an Observations object
    #fig = plt.figure()
    fig, axes = plt.subplots()

    # set identifiers
    if identifiers is None:
        identifiers = data.identifiers
    identifiers = [ident for ident in identifiers if ident in data.identifiers]

    # plot data for each experiment
    for ident in identifiers:

        # plot data for the current experiment
        line = plot_2d(
            x_data=data, x_name=xName, y_name=yName, pp=pp, yerr=yerr,
            identifiers=[ident], graph_type=graphType, axes=axes)

    # calculate and plot mean
    if mode == 'all&mean':
        exp = data.doStatsByIndex(
            name=yName, identifiers=identifiers, identifier='mean', ddof=ddof)
        if len(identifiers) > 0:
            #exp_dist = data.getExperiment(identifier=identifiers[0])

            # set x axis values
            x_values = data.getValue(identifier=identifiers[0], name=xName)
            if len(x_values) > len(exp.mean):
                x_values = x_values[:len(exp.mean)]
            exp.__setattr__(xName, x_values)
            exp.properties.add(xName)
            exp.indexed.add(xName)

            # plot
            line = plot_2d(
                x_data=exp, x_name=xName, y_data=exp, y_name='mean', pp=pp,
                yerr=yerr, graph_type=graphType, line_width_='thick', axes=axes)

    # finish plotting
    axes.set_title(title)
    axes.set_ylabel(y_label)
    axes.set_xlabel(x_label)
    #ends = list(axes.axis())
    axes.axis([0, 250, 0, 0.3])
    if pp.legend:
        axes.legend()
    #plt.show()  # commented out to alow changing plot in notebook

    return axes

def plot_histogram(
        data, name, bins, pp, groups=None, identifiers=None,
        facecolor=None, edgecolor=None, x_label=None, title=None):
    """
    Plots data as a histogram.

    If more than one group is given (arg groups), data from all groups are
    combined. Also data from all experiments are combined.

    Arguments:
      - data: (Groups or Observations) data
      - name: property name
      - bins: histogram bins
      - pp: module containing print and plot parameters
      - groups: list of group names, None for all groups
      - identifiers: experiment identifier names, None for all identifiers
      - facecolor: histogram facecolor
      - edgecolor: histogram edgecolor
      - x_label: x axis label
      - title: title
    """

    # combine data
    if isinstance(data, Groups):
        obs = data.joinExperiments(
            name=name, mode='join', groups=groups, identifiers=identifiers)
    elif isinstance(data, Observations):
        obs = data
    exp = obs.joinExperiments(name=name, mode='join')
    combined_data = getattr(exp, name)

    # color
    if facecolor is None:
        if (groups is not None):
            if len(groups) == 1:
                facecolor = pp.color.get(groups[0], None)
        else:
            if isinstance(groups, basestring):
                facecolor = pp.color.get(groups, None)

    # plot
    plt.hist(
        combined_data, bins=bins, facecolor=facecolor, edgecolor=edgecolor)

    # finish plot
    if title is not None:
        plt.title(title)
    if x_label is None:
        x_label = name
    plt.xlabel(x_label)
    #plt.show()  # commented out to alow changing plot in notebook

def plot_stats(
        stats, name, pp, groups=None, identifiers=None, axes=None, yerr='sem',
        plot_type='bar', randomized_x=False, confidence='stars',
        stats_between=None, label=None, skip_name='_skip'):
    """
    Does main part of plotting property (arg) name of (arg) stats, in the
    form of a bar chart.

    If specified, args groups and identifiers specify the order of groups
    and experiments on the x axis.

    Plots on the current figure.

    Arguments:
      - stats: (Groups, or Observations) object
      containing data
      - name: property name
      - pp: module containing print and plot parameters
      - groups: list of group names, None for all groups
      - identifiers: experiment identifier names, None for all identifiers
      - axes: matplotlib.axes object (experimental)
      - yerr: attribute name of stats that is used to plot y error
      - plot_type: plot type, currently:
          * bar: barplot
          * boxplot: Tukey boxplot showing IQR = Q3 - Q1, whiskers (at
            the last point within Q3 + 1.5 IQR and Q1 - 1.5 IQR) and outliers,
            box filled with color
          * boxplot_data: boxplot without fliers, box empty together with
            all data
      - randomized_x: Flag indicating if x values should be randomized for
      boxplot_data, so that data don't overlap
      - confidence: None for not showing confidence, 'stars' or 'number' to
      show confidence as stars or confidence number, respectivly
      - stats_between: (Groups) Needs to contain confidence between
      (experiments of) different groups having the same identifiers
      - label: determines which color, alpha, ... is used, can be 'group' to
      label by group or 'experiment' to label by experiment
      - skip_name: should not be used - Work in progress
    """

    # stats type
    if isinstance(stats, Groups):
        stats_type = 'groups'
    elif isinstance(stats, Observations):
        stats_type = 'observations'
        stats_obs = stats
        stats = Groups()
        stats[''] = stats_obs
    else:
        raise ValueError(
            "Argument stats has to be an instance of Groups or Observations.")

    # set group order
    if groups is None:
        group_names = list(stats.keys())
    else:
        group_names = groups

    # find rough range of y-axis values (to plot confidence)
    if (confidence is not None):
        y_values = [
            stats[group_nam].getValue(identifier=ident, property=name)
            for group_nam in group_names
            for ident in stats[group_nam].identifiers
            if ((identifiers is None) or (ident in identifiers))]
        if (y_values is not None) and (len(y_values) > 0):
            if isinstance(y_values[0], (list, numpy.ndarray)):
                rough_y_min = min(min(yyy) for yyy in y_values)
                rough_y_max = max(max(yyy) for yyy in y_values)
            else:
                rough_y_min = min(y_values)
                rough_y_max = max(y_values)
        else:
            rough_y_min = 0
            rough_y_max = 1
        rough_y_range = rough_y_max - min(rough_y_min, 0)

    # set bar width if needed
    if pp.bar_arrange == 'uniform':
        bar_width = 0.2
        left = -2 * bar_width
    elif pp.bar_arrange == 'grouped':
        max_identifs = max(
            len(stats[group_nam].identifiers) for group_nam in group_names)
        bar_width = numpy.floor(80 / max_identifs) / 100
    else:
        raise ValueError("pp.bar_arrange has to be 'uniform' or 'grouped'.")

    # initialize vars
    y_min = 0
    y_max = 0
    group_left = []
    label_done = False  # probably not needed anymore
    legend_handles = []
    legend_labels = []

    # loop over groups
    if axes is None:
        axes = plt.gca()
        print("Debug: axes not specified")
    for group_nam, group_ind in zip(
            group_names, list(range(len(group_names)))):
        group = stats[group_nam]

        # move bar position
        if pp.bar_arrange == 'uniform':
            left += bar_width
            group_left.append(left + bar_width)

        # skip if placeholder - Work in progress
        if group_nam == skip_name:
            continue

        # set experiment order
        if identifiers is None:
            loc_identifs = group.identifiers
        elif isinstance(identifiers, (list, tuple)):
            loc_identifs = [
                ident for ident in identifiers
                if ident in group.identifiers or ident == skip_name]
            #loc_identifs = identifiers.copy()
        elif isinstance(identifiers, dict):
            loc_identifs = identifiers[group_nam]

        # loop over experiments
        for ident, exp_ind in zip(
                loc_identifs, list(range(len(loc_identifs)))):

            #
            #if ident not in loc_identifs:
            #    continue
            
            # move plot position
            if pp.bar_arrange == 'uniform':
                left += bar_width
            elif pp.bar_arrange == 'grouped':
                left = group_ind + exp_ind * bar_width
                
            # skip if placeholder - Work in progress
            if ident == skip_name:
                continue
                
            # label
            if label is None:
                if stats_type == 'groups':
                    label_code = group_nam
                elif stats_type == 'observations':
                    label_code = ident
            elif label == 'group':
                label_code = group_nam
            elif label == 'experiment':
                label_code = ident

            # adjust alpha
            loc_alpha = pp.alpha.get(label_code, 1)

            # y values
            value = group.getValue(identifier=ident, property=name)
            if numpy.isnan(value).all(): continue
            #print("value {}".format(value))

            # y error and y limits
            if ((yerr is not None) and (yerr in group.properties)
                    and (plot_type != 'boxplot')
                and (loc_alpha == 1)):
                yerr_num = group.getValue(identifier=ident, property=yerr)
                if numpy.isnan(yerr_num):
                    yerr_num = 0
                    yerr_one = 0
                yerr_one = yerr_num
                if isinstance(value, (list, numpy.ndarray)):
                    y_max = max(y_max, max(value)+yerr_num)
                else:
                    y_max = max(y_max, value+yerr_num)
                if pp.one_side_yerr:
                    yerr_num = ([0], [yerr_num])
            else:
                yerr_num = None
                yerr_one = 0
                if isinstance(value, (list, numpy.ndarray)):
                    y_max = max(y_max, max(value))
                else:
                    y_max = max(y_max, value)
            if isinstance(value, (list, numpy.ndarray)):
                y_min = min(y_min, min(value))
            else:
                y_min = min(y_min, value)

            # plot
            if plot_type == 'bar':
                bar, = axes.bar(
                    x=left, height=value, yerr=yerr_num, width=bar_width,
                    #label=pp.category_label.get(label_code, ''),
                    color=pp.color[label_code], ecolor=pp.color[label_code],
                    alpha=loc_alpha)
                if not label_done:
                    # add label if the same one wasn't added before
                    label_local = pp.category_label.get(label_code, '')
                    if label_local not in axes.get_legend_handles_labels()[1]:
                        bar.set_label(label_local)
                x_confid = bar.get_x() + bar.get_width() / 2.
                y_confid_base = bar.get_height() + yerr_one

            elif plot_type == 'boxplot':
                x_value = left + bar_width / 2.
                bplot = axes.boxplot(
                    value, positions=(x_value,), widths=(bar_width,),
                    #labels=[pp.category_label.get(label_code, '')],
                    patch_artist=True)
                bplot['boxes'][0].set_color(pp.color[label_code])
                x_confid = x_value
                y_confid_base = bplot['whiskers'][1].get_ydata()[1]
                y_confid_base = max(y_confid_base, y_max)
                # for some reason arg labels in boxplot() doesn't add
                # legends to axes when patch_artist=True, so need to add
                # manually
                if not label_done:
                    legend_handles.append(bplot['boxes'][0])
                    legend_labels.append(pp.category_label.get(label_code, ''))

            elif plot_type == 'boxplot_data':
                squeeze = 0.5 # IMPORTANT: Change if needed
                x_value = left + bar_width / 2.
                try:
                    bplot = axes.boxplot(
                        value, positions=(x_value,),
                        widths=(squeeze*bar_width,),
                        labels=pp.category_label.get(label_code, ''),
                        patch_artist=True, showfliers=False)
                except ValueError:
                    bplot = axes.boxplot(
                        value, positions=(x_value,),
                        widths=(squeeze*bar_width,),
                        patch_artist=True, showfliers=False)
                #print bplot['boxes'][0].get_ydata()
                bplot_el_all = (
                    bplot['boxes'] + bplot['caps'] + bplot['whiskers'])
                for bplot_el in bplot_el_all:
                    bplot_el.set_color(pp.color[label_code])
                bplot['boxes'][0].set_facecolor('none')
                x_value_all = [x_value] * len(value)
                # for some reason arg labels in boxplot() doesn't add
                # legends to axes, so need to do manually
                if not label_done:
                    legend_handles.append(bplot['boxes'][0])
                    legend_labels.append(pp.category_label.get(label_code, ''))

                if randomized_x:
                    #box_x = bplot['boxes'][0].get_xdata()
                    x_value_all = numpy.random.normal(
                        loc=x_value, scale=0.5*0.8*bar_width/2.,
                        size=len(x_value_all))
                dataplot = axes.plot(
                    x_value_all, value, marker='o', markerfacecolor='none',
                    markeredgecolor=pp.color[label_code], linestyle=' ')

                x_confid = x_value
                y_confid_base = bplot['whiskers'][1].get_ydata()[1]
                y_confid_base = max(y_confid_base, y_max)

            # should be removed when Matplot lib 1.* not used anymore
            if mpl.__version__[0] == 1:
                plt.errorbar(
                    left+bar_width/2, value, yerr=yerr_num,
                    ecolor=pp.ecolor.get(label_code, 'k'), label='_nolegend_')

            # confidence within group
            if (confidence is not None) and ('confidence' in group.properties):

                # get confidence
                confid_num = group.getValue(
                    identifier=ident, property='confidence')
                # workaround for problem in Observations.getValue()
                if (isinstance(confid_num, (list, numpy.ndarray))
                    and len(confid_num) == 1):
                    confid_num = confid_num[0]
                if confidence == 'number':
                    confid = pp.confidence_plot_format % confid_num
                    conf_size = pp.confidence_plot_font_size
                elif confidence == 'stars':
                    confid = '*' * get_confidence_stars(
                        confid_num, limits=pp.confidence_stars)
                    conf_size = 1.5 * pp.confidence_plot_font_size

                # plot confidence
                y_confid = 0.02 * rough_y_range + y_confid_base
                ref_ident = group.getValue(identifier=ident,
                                           property='reference')
                ref_color = pp.color.get(ref_ident, label_code)
                axes.text(
                    x_confid, y_confid, confid, ha='center', va='bottom',
                    size=conf_size, color=ref_color)

            # confidence between groups
            if ((stats_type == 'groups') and (confidence is not None)
                and (stats_between is not None)
                and ('confidence' in stats_between[group_nam].properties)):
                other_group = stats_between[group_nam]

                # check
                other_ref = other_group.getValue(identifier=ident,
                                                 property='reference')
                if other_ref != ident:
                    logging.warning(
                        "Confidence between groups calculated between "
                        + "experiments with different identifiers: "
                        + ident + " and " + other_ref + ".")

                # get confidence
                confid_num = other_group.getValue(identifier=ident,
                                                  property='confidence')
                if confidence == 'number':
                    confid = pp.confidence_plot_format % confid_num
                    conf_size = pp.confidence_plot_font_size
                elif confidence == 'stars':
                    confid = '*' * get_confidence_stars(
                        confid_num, limits=pp.confidence_stars)
                    conf_size = 1.5 * pp.confidence_plot_font_size

                # plot
                y_confid = 0.04 * rough_y_range + y_confid_base
                ref_color = pp.color[ident]
                axes.text(
                    x_confid, y_confid, confid, ha='center', va='bottom',
                    size=conf_size, color=ref_color)

        # set flag that prevents adding further labels to legend
        #label_done = True  # multiple labels removed above (bar) below (box)

    # adjust axes
    axis_limits = list(axes.axis())
    if pp.bar_arrange == 'uniform':
        x_min = axis_limits[0]-bar_width
    if pp.bar_arrange == 'grouped':
        x_min = -bar_width
    axes.axis([x_min, max(axis_limits[1], 4), y_min, 1.1*y_max])
    if pp.bar_arrange == 'uniform':
        group_left.append(left)
        x_tick_pos = [
            group_left[ind] +
            (group_left[ind+1] - group_left[ind] - bar_width) / 2.
            for ind in range(len(group_left) - 1)]
    elif pp.bar_arrange == 'grouped':
        x_tick_pos = numpy.arange(len(group_names)) + bar_width*(exp_ind+1)/2.
    group_labels = [pp.category_label.get(g_name, g_name)
                    for g_name in group_names]
    axes.set_xticks(x_tick_pos)
    axes.set_xticklabels(group_labels)
    #plt.xticks(x_tick_pos, group_labels) old

    # legend
    legend_done = False
    if (plot_type == 'boxplot') or (plot_type == 'boxplot_data'):

        # boxplot: need to remove double labels from legend
        if len(legend_labels) > 0:
            uni_indices = [
                numpy.where(numpy.array(legend_labels) == ind)[0].min()
                for ind in set(legend_labels)]
            uni_indices = numpy.sort(uni_indices)
            uni_handles = [legend_handles[ind] for ind in uni_indices]
            uni_labels = [legend_labels[ind] for ind in uni_indices]
            if pp.legend:
                axes.legend(uni_handles, uni_labels)
                legend_done = True

    else:

        # bars: labels already cleaned
        if pp.legend:
            axes.legend()
            legend_done = True

    # return some info
    result = {}
    result['legend_done'] = legend_done  # not sure if needed
    return result

def plot_2d(
        x_data, pp, x_name='x_data', y_data=None, y_name='y_data', yerr=None,
        groups=None, identifiers=None, graph_type='scatter',
        line_width_=None, fit=None, axes=None):
    """
    Min part for plottings a 2d graph.

    If specified, args groups and identifiers specify the order of groups
    and experiments on the x axis.

    Plots on the current figure.

    Arguments:
      - pp: module containing print and plot parameters
      - x_data, y_data: data objects, have to be instances of Groups,
      Observations or Experiment and they both have to be instances of the
      same class
      - x_name, y_name: names of properties of x_data and y_data that are
      plotted on x and y axis
      - axes: matplotlib.axes object (experimental)
    """

    # y data
    if y_data is None:
        y_data = x_data

    # determine data type and set group order
    if (isinstance(x_data, Groups) and isinstance(y_data, Groups)):
        data_type = 'groups'
        if groups is None:
            group_names = list(x_data.keys())
        else:
            group_names = groups
    elif (isinstance(x_data, Observations)
          and isinstance(y_data, Observations)):
        data_type = 'observations'
        group_names = ['']
    elif (isinstance(x_data, pyto.analysis.Experiment)
          and isinstance(y_data, pyto.analysis.Experiment)):
        data_type = 'experiment'
        group_names = ['']
    else:
        raise ValueError(
            "Arguments x_data and y_data have to be instances of Groups, "
            + "Observations or Experiment and they need to be instances "
            + "of the same class.")

    # line style and width
    if graph_type == 'scatter':
        loc_line_style = ''
        loc_marker = pp.marker
    elif graph_type == 'line':
        loc_line_style = pp.default_line_style
        loc_marker = ''
    if line_width_ is None:
        loc_line_width = pp.default_line_width
    elif line_width_ == 'thick':
        loc_line_width = pp.thick_line_width

    # loop over groups
    figure = None
    markers_default_copy = copy(pp.markers_default)
    if axes is None:
        axes = plt.gca()
        print("Debug: axes not specified")
    for group_nam, group_ind in zip(
            group_names, list(range(len(group_names)))):

        # get data
        if data_type == 'groups':
            x_group = x_data[group_nam]
            y_group = y_data[group_nam]
        elif data_type == 'observations':
            x_group = x_data
            y_group = y_data
        elif data_type == 'experiment':
            x_group = Observations()
            x_group.addExperiment(experiment=x_data)
            y_group = Observations()
            y_group.addExperiment(experiment=y_data)

        # set experiment order
        if identifiers is None:
            loc_identifs = x_group.identifiers
        elif isinstance(identifiers, list):
            loc_identifs = [ident for ident in identifiers
                            if ident in x_group.identifiers]
        elif isinstance(identifiers, dict):
            loc_identifs = identifiers[group_nam]

        # loop over experiments
        for ident, exp_ind in zip(
                loc_identifs, list(range(len(loc_identifs)))):

            # values
            if (data_type == 'groups') or (data_type == 'observations'):
                x_value = x_group.getValue(identifier=ident, property=x_name)
                y_value = y_group.getValue(identifier=ident, property=y_name)
            elif data_type == 'experiment':
                x_ident = x_data.identifier
                x_value = x_group.getValue(identifier=x_ident, property=x_name)
                y_ident = y_data.identifier
                y_value = y_group.getValue(identifier=y_ident, property=y_name)

            # cut data to min length
            if len(x_value) != len(y_value):
                min_len = min(len(x_value), len(y_value))
                x_value = x_value[:min_len]
                y_value = y_value[:min_len]

            # adjust colors
            loc_alpha = pp.alpha.get(ident, 1)
            if graph_type == 'scatter':
                loc_marker = pp.marker.get(ident, None)
                if loc_marker is None:
                    loc_marker = markers_default_copy.pop(0)
            loc_color = pp.color.get(ident, None)

            # plot data points
            #label = (group_nam + ' ' + ident).strip()
            loc_label = pp.category_label.get(ident, ident)
            if loc_color is not None:
                figure = axes.plot(
                    x_value, y_value, linestyle=loc_line_style, color=loc_color,
                    linewidth=loc_line_width, marker=loc_marker,
                    markersize=pp.marker_size, alpha=loc_alpha, label=loc_label)

            else:
                figure = axes.plot(
                    x_value, y_value, linestyle=loc_line_style,
                    linewidth=loc_line_width, marker=loc_marker,
                    markersize=pp.marker_size, alpha=loc_alpha, label=loc_label)

            # plot eror bars
            if yerr is not None:
                yerr_value = y_group.getValue(identifier=ident, property=yerr)
                # arg color needed otherwise makes line with another color
                axes.errorbar(
                    x_value, y_value, yerr=yerr_value,
                    color=loc_color, ecolor=loc_color, label='_nolegend_')

            # plot fit line
#            if fit is not None:
            if fit is not None:

                # data limits
                x_max = x_value.max()
                x_min = x_value.min()
                y_max = y_value.max()
                y_min = y_value.min()

                # fit line parameters
                a_reg = x_group.getValue(identifier=ident, property=fit[0])
                b_reg = y_group.getValue(identifier=ident, property=fit[1])

                # fit limits
                x_range = numpy.arange(x_min, x_max, (x_max - x_min) / 100.)
                poly = numpy.poly1d([a_reg, b_reg])
                y_range = numpy.polyval(poly, x_range)
                start = False
                x_fit = []
                y_fit = []
                for x, y in zip(x_range, y_range):
                    if (y >= y_min) and (y <= y_max):
                        x_fit.append(x)
                        y_fit.append(y)
                        start = True
                    else:
                        if start:
                            break
                # plot fit
                if loc_color is not None:
                    axes.plot(
                        x_fit, y_fit, linestyle=pp.default_line_style,
                        color=loc_color, linewidth=loc_line_width, marker='',
                        alpha=loc_alpha)
                else:
                    axes.plot(
                        x_fit, y_fit, linestyle=pp.default_line_style,
                        linewidth=loc_line_width, marker='', alpha=loc_alpha)

    return figure

def get_confidence_stars(value, limits):
    """
    Returns number of stars for a given confidence level(s).
    """

    # more than one value
    if isinstance(value, (numpy.ndarray, list)):
        result = [get_confidence_stars(x, limits) for x in value]
        return result

    # one value
    result = 0
    for lim in limits:
        if value <= lim:
            result += 1
        else:
            break

    return result

def get_bin_names(bins, mode='limits', fancy_ends=True, discrete=False):
    """
    Given bin values, returns reasonably formated bin names (strings) to
    be used to present the bins, for example in plots.

    Arguments:
      - bins: list of bin values
      - mode: 'limits' to make bin names show bin intervals, 'lower' to show
      lower and 'higher' to show higher bin values
      - fancy_ends: if True, the lowest and the highest bin names will
      look like '<value', '>value' or similar
      - discrete: should be set to True if bin values are integers (and
      not foats)

    Returns: list of the corresponding bin names.
    """

    if (mode == 'limits') and not discrete:
        bin_names = ['{}-{}'.format(x, y) for x, y in zip(bins[:-1], bins[1:])]
        if fancy_ends:
            bin_names[0] = '<{}'.format(bins[1])
            bin_names[-1] = '>{}'.format(bins[-2])

    elif (mode == 'limits') and discrete:
        bin_values = [(x, y-1) for x, y in zip(bins[:-1], bins[1:])]
        bin_names = ['{}-{}'.format(x, y) for x, y in bin_values]
        for ind, val in enumerate(bin_values):
            if val[0] == val[1]:
                bin_names[ind] = '{}'.format(val[0])
        if fancy_ends and (bin_values[0][0] != bin_values[0][1]):
            bin_names[0] = '\u2264{}'.format(bin_values[0][1])
        if fancy_ends and (bin_values[-1][0] != bin_values[-1][1]):
            bin_names[-1] = '\u2265{}'.format(bin_values[-1][0])

    elif mode == 'lower':
        bin_names = ['{}'.format(x) for x in bins[:-1]]

    elif mode == 'higher':
        bin_names = ['{}'.format(x) for x in bins[1:]]
        bin_names[-1] = '>{}'.format(bins[-2])

    return bin_names

def save_data(object, base, categories, name=['mean', 'sem']):
    """
    Saves indexed data in a file. If more than one property is specified, the
    corresponding values are saved in separate files. Each row contains
    values for one index. Indices are saved in the first coulmn, while each
    other column correspond to one of the identifiers.

    Will probably get depreciated because it is now more straightforward to
    covert data to pandas.DataFrame (Groups.get_indexed_data() and print
    from there.

    Arguments:
      - object: (Observations) object that contains data
      - base: file name is created as base_property_name
      - name: names of properties that need to be saved
      - categories: categories
    """

    # find shortest ids
    if 'ids' in object.indexed:
        ids = object.ids[0]
        for group, group_ind in zip(categories, list(range(len(categories)))):
            current_ids = object.getValue(identifier=group, name='ids')
            if len(current_ids) < len(ids):
                ids = current_ids
        len_ids = len(ids)

    # loop over properties
    if not isinstance(name, (list, tuple)):
        name = [name]
    for one_name in name:

        # initialize results
        if one_name in object.indexed:
            result = numpy.zeros(shape=(len_ids, len(categories)+1))
            result[:, 0] = ids
        else:
            result = numpy.zeros(shape=(1, len(categories)+1))
            result[0,0] = 1

        # make array that contains all values for current property
        for group, group_ind in zip(categories, list(range(len(categories)))):
            values =  object.getValue(identifier=group, name=one_name)

            if one_name in object.indexed:
                len_values = len(values)
                if len_ids <= len_values:
                    result[:, group_ind+1] = values[:len_ids]
                else:
                    result[:len_values, group_ind+1] = values[:]

            else:
                result[0, group_ind+1] = values

        # write current array
        format = ' %i '
        header = 'index'
        for categ in categories:
            format += '      %8.5f'
            header += '  ' + categ
        file_ = base + '_' + one_name
        numpy.savetxt(file_, result, fmt=format, header=header)

##############################################################
#
# Functions that calculate certain properites for cleft analysis
#

def getSpecialThreshold(
        cleft, segments, fraction, groups=None, identifiers=None):
    """
    Return threshold closest to the density level at the specified
    fraction between boundaries (level 0) and cleft (level 1).

    Warning: Not sure if working

    Arguments:
      - cleft: (Groups)
      - segments:
      - fraction: fraction between 0-1
    """

    # get groups
    if groups is None:
        groups = list(cleft.keys())

    # loop over groups
    fract_thresholds = {}
    fract_densities = {}
    for categ in groups:

        # loop over experiments (identifiers)
        categ_identifiers = cleft[categ].identifiers
        for identif in categ_identifiers:

            # skip identifiers that were not passed
            if identifiers is not None:
                if identif not in identifiers:
                    continue

            # get boundary and cleft ids
            bound_ids = cleft[categ].getValue(
                identifier=identif, property='boundIds')
            cleft_ids = cleft[categ].getValue(
                identifier=identif, property='cleftIds')

            # get mean boundary and cleft and fractional densities
            bound_densities = cleft[categ].getValue(
                identifier=identif, property='mean', ids=bound_ids)
            bound_volume = cleft[categ].getValue(
                identifier=identif, property='volume', ids=bound_ids)
            bound_density = (
                numpy.dot(bound_densities, bound_volume) / bound_volume.sum())
            cleft_densities = cleft[categ].getValue(
                identifier=identif, property='mean', ids=cleft_ids)
            cleft_volume = cleft[categ].getValue(
                identifier=identif, property='volume', ids=cleft_ids)
            cleft_density = (
                numpy.dot(cleft_densities, cleft_volume) / cleft_volume.sum())
            fract_density = (
                bound_density + (cleft_density - bound_density) * fraction)

            # get closest threshold
            # ERROR thresholds badly formated in segments
            all_thresh = segments[categ].getValue(identifier=identif,
                                                  property='thresh')
            index = numpy.abs(all_thresh - fract_density).argmin()
            thresh = all_thresh[index]
            thresh_str = "%6.3f" % thresh
            try:
                fract_thresholds[categ][identif] = thresh_str
                fract_densities[categ][identif] = (
                    bound_density, cleft_density, fract_density)
            except KeyError:
                fract_thresholds[categ] = {}
                fract_thresholds[categ][identif] = thresh_str
                fract_densities[categ] = {}
                fract_densities[categ][identif] = (
                    bound_density, cleft_density, fract_density)

    return fract_densities

def get_occupancy(segments, layers, groups, name):
    """
    Occupancy is added to the segments object

    Arguments:
      - segments: (connections)
      - layers: (CleftRegions)
      - groups
      - name: name of the added (occupancy) property
    """

    for categ in groups:
        for ident in segments[categ].identifiers:

            seg_vol = segments[categ].getValue(identifier=ident,
                                               property='volume').sum()
            total_vol = layers[categ].getValue(identifier=ident,
                                               property='volume')
            cleft_ids = layers[categ].getValue(identifier=ident,
                                               property='cleftIds')
            cleft_vol = total_vol[cleft_ids-1].sum()
            occup = seg_vol / float(cleft_vol)
            segments[categ].setValue(identifier=ident, property=name,
                                     value=occup)

def get_cleft_layer_differences(data, name, groups):
    """
    """

    def abs_diff43(x):
        return x[3] - x[2]

    def abs_diff65(x):
        return x[5] - x[4]

    # not good because apply mane the new property indexed
    #data.apply(funct=abs_diff43, args=[name],
    #           name='diffNormalMean43', categories=groups)
    #data.apply(funct=abs_diff65, args=[name],
    #           name='diffNormalMean65', categories=groups)

    for categ in groups:
        for ident in data[categ].identifiers:

            # 4 - 3
            val4 = data[categ].getValue(
                identifier=ident, property=name, ids=[4])[0]
            val3 = data[categ].getValue(
                identifier=ident, property=name, ids=[3])[0]
            diff43 = val4 - val3
            data[categ].setValue(
                identifier=ident, property='diffNormalMean43', value=diff43)

            # 6 - 5
            val6 = data[categ].getValue(
                identifier=ident, property=name, ids=[6])[0]
            val5 = data[categ].getValue(
                identifier=ident, property=name, ids=[5])[0]
            diff65 = val6 - val5
            data[categ].setValue(
                identifier=ident, property='diffNormalMean65', value=diff65)


##############################################################
#
# Functions that calculate certain properites for presynaptic analysis
#

def calculateVesicleProperties(data, layer=None, tether=None, categories=None):
    """
    Calculates additional vesicle related properties.

    The properties calculated are:
      - 'n_vesicle'
      - 'az_surface_um'
      - 'vesicle_per_area_um'
      - 'mean_tether_nm' (for non-tethered vesicles value set to numpy.nan)
    """

    # calculate n vesicles per synapse
    data.getNVesicles(name='n_vesicle', categories=categories)

    # calculate az surface (defined as layer 1) in um
    if layer is not None:
        data.getNVesicles(
            layer=layer, name='az_surface_um', fixed=1, inverse=True,
            layer_name='surface_nm', layer_factor=1.e-6, categories=categories)

    # calculate N vesicles per unit az surface (defined as layer 1) in um
    if layer is not None:
        data.getNVesicles(
            layer=layer, name='vesicle_per_area_um',
            layer_name='surface_nm', layer_factor=1.e-6, categories=categories)

    # calculate mean tether length for each sv
    if tether is not None:
        data.getMeanConnectionLength(conn=tether, name='mean_tether_nm',
                                     categories=categories, value=numpy.nan)

def calculateTetherProperties(data, layer=None, categories=None):
    """
    Calculates additional vesicle related properties.

    The properties calculated are:
      - 'n_tether'
      - 'tether_per_area_um'
    """

    # calculate n tethers per synapse (to be moved up before pickles are made)
    data.getN(name='n_tether', categories=categories)

    # calculate N tethers per unit az surface (defined as layer 1) in um
    if layer is not None:
        data.getN(
            layer=layer, name='tether_per_area_um',
            layer_name='surface_nm', layer_factor=1.e-6, categories=categories)

def calculateConnectivityDistanceRatio(
        vesicles, initial, distances, name='n_tethered_ratio',
        categories=None):
    """
    """

    # calculate connectivity distances
    vesicles.getConnectivityDistance(
        initial=initial, name='conn_distance', distance=1,
        categories=categories)

    # shortcuts
    d0 = [distances[0], distances[0]]
    d1 = [distances[1], distances[1]]

    # find n vesicles at specified distances
    conndist_0_sv = vesicles.split(
        value=d0, name='conn_distance', categories=categories)[0]
    conndist_0_sv.getNVesicles(name='_n_conndist_0', categories=categories)
    vesicles.addData(
        source=conndist_0_sv, names={'_n_conndist_0': '_n_conndist_0'})
    conndist_1_sv = vesicles.split(
        value=d1, name='conn_distance', categories=categories)[0]
    conndist_1_sv.getNVesicles(name='_n_conndist_1', categories=categories)
    vesicles.addData(
        source=conndist_1_sv, names={'_n_conndist_1': '_n_conndist_1'})

    # calculate reatio
    vesicles.apply(
        funct=numpy.true_divide, args=['_n_conndist_1', '_n_conndist_0'],
        name=name, categories=categories, indexed=False)

def convert_synapse_angles(
        data, old_mode, new_name='angle_90', old_name='angle'):
    """
    Converts synapse orientation, defined as the direction of the vector from 
    post- to presynaptic terminal (synapse vector) to an angle between 0 and 90 
    degrees that characterizes the missing wedge effects in z-plane.
    
    If arg old_mode is 'clock', the direction of the synaptic vector is 
    defined as the angle it makes with the y-axis in the clockwise mode 
    and it is specified by the value of the property with name given by 
    the arg old_name. For example, if this angle is:
      - 0, it means the presynaptic terminal is up (positive y axis) and the 
      postsynaptic down
      - 90 degerees, it means that the presynaptic terminal is on the right 
      (positive x axis) and the postsinaptic on the left.
      
    The final angle is also defined by the angle between the synaptic 
    vector and the y-axis, but it takes values from 0-90 degrees as follows:
      - 0: pre- or postsynaptic terminal is up (positive y) and the other 
      is down
      - 90: one terninal is on the right (positive x) and the other on the left 
  
    If the property new_name already exists, it will be overwriten.

    If arg data is a list or tuple of Groups objects, they have to have
    the same group names and identifiers. This is because the (old) angle is 
    read only from the first object and the converted angle property is 
    added to all Groups objects.

    Sets:
      - property new_angle

    Arguments:
      - data: Groups object, or a list (tuple) of Groups objects that 
      contains the old angles
      - old_mode: the way old angles are defined
      - now_name: property name for the new (converted angles)
      - old_name: property name for the old angle
    """

    # get scalars
    if isinstance(data, (list, tuple)):
        data_0 = data[0]
    else:
        data = [data]
        data_0 = data[0]
    sdata = data_0.scalar_data

    # convert angles
    if old_mode == 'clock':
        sdata[new_name] = sdata[old_name].abs()
        sdata[new_name] = sdata[new_name].map(
            lambda x: 180 - x if x > 90 else x)    

    elif old_mode == 'phi':
        sdata[new_name] = (sdata[old_name].abs() - 90).abs()
        #sdata[new_name] = sdata[new_name].map(
        #    lambda x: 180 - x if x > 90 else x)
        
    else:
        raise ValueError(f"Mode {old_mode} is not understood.")
    
    # make a Groups object that contains converted angles
    sdata_small = sdata[['identifiers', 'group', new_name]]
    idata_small = sdata[['identifiers', 'group']]
    groups_new = Groups.from_pandas(indexed=idata_small, scalar=sdata_small)
    
    # add restricted angles
    for da in data:
        da.addData(source=groups_new, names=[new_name], copy=True)


##################################################################
#
# Fanctions that start from multi-dataset analysis to get data from
# individual datasets (tomograms).
#
# Currently applied for the presynaptic analysis
#

def tomo_metadata(work, segment_var, boundary_var, rm_prefix=None):
    """
    Figures out tomogram paths from structure specific pickles.

    This works in the oposite direction from the standard workflow, namely
    the flow here is:

      Multi dataset structure specific pickles -> single dataset pickles
      -> tomo_info files, paths to tomo and label files

    Arg work can be specified in two ways: First, as the loaded and
    preprocessed module. Second, as the path to the work script in which
    case the work script is loaded as a module and preprocessed.

    In the presynaptic workflow, the work file is obtained by editing
    pyto.scripts/presynaptic_stats.py script.

    Arguments:
      - work: Module containing multidataset analysis (preprocessed) or
      the name (path) to the work file.
      - segment_var: Name of the catalog variable that shows path to the
      single dataset connectors pickle (typically 'tethers_file' for
      tethers and 'connectors_file' for connectors
      - boundary_var: Name of the catalog variable that shows path to the
      single dataset vesicles pickle
      - rm_prefix: The common part of all tomogram paths, set to None if
      no common path

    Returns pandas.DataFrame containing the following columns:
        - tomo identifier,
        - experimental group
        - tomogram directory (relative to arg rm_prefix if specified)
        - tomogram file name
        - boundary file path relative to the tomo directory
        - tomo_info file path relative to the tomo directory
    """

    # load and preprocess work module if needed
    if not isinstance(work, ModuleType):
        work_file = work
        work_dir, work_name = os.path.split(work_file)
        work_base, work_ext = os.path.splitext(work_name)
        spec = importlib.util.spec_from_file_location(work_base, work_file)
        work = spec.loader.load_module(spec.name)
        work.main()
    work_path = os.path.normpath(os.path.join(os.getcwd(), work.__file__))

    # get multiple dataset data ready
    particles = pyto.particles.Set(struct=work.near_sv, work_path=work_path)
    labels = pyto.particles.LabelSet(
        struct=work.near_sv, work_path=work_path, catalog_var=segment_var)
    bounds = pyto.particles.BoundarySet(
        struct=work.near_sv, work_path=work_path, catalog_var=boundary_var)

    for ident in work.identifiers:
        group_found = False
        for g_name in work.categories:
            if ident not in particles.struct[g_name].identifiers:
                    continue

            # get tomo and boundary paths
            labels_pickle_path = labels.get_pickle_path(
                work=work, group_name=g_name, identifier=ident)
            tomo_path = particles.get_tomo_path(pickle_path=labels_pickle_path)
            bounds_path = bounds.get_tomo_path(pickle_path=labels_pickle_path)
            tomo_info_path = particles.get_tomo_info_path(
                pickle_path=labels_pickle_path)

            # get relative paths
            tomo_dir, tomo_name = os.path.split(tomo_path)
            bounds_dir, bounds_name = os.path.split(bounds_path)
            bounds_path_rel = os.path.relpath(bounds_path, start=tomo_dir)
            tomo_info_path_rel = os.path.relpath(tomo_info_path, start=tomo_dir)
            if rm_prefix is not None:
                tomo_dir = os.path.relpath(tomo_dir, rm_prefix)

            # make local table
            curr_metadata = pd.DataFrame({
                'identifier' : ident, 'group_name' : g_name,
                'tomo_dir' : tomo_dir, 'tomo_name' : tomo_name,
                'bounds_path_rel' : bounds_path_rel,
                'tomo_info_path_rel' : tomo_info_path_rel},
                index = [0])

            # add local
            try:
                metadata = metadata.append(
                    curr_metadata, ignore_index=True)
            except (AttributeError, NameError):
                metadata = curr_metadata
            #self.metadata.drop_duplicates().reset_index(drop=True)

            group_found = True
            break

    return metadata

def find_synapse_angle(
        data, segment_var, bound_id, segmentation_id, angle_name='angle_phi'):
    """
    Finds synapse angles for all synapses contained in the specified Groups
    object (arg data) and adds them as a new property (name specified by
    arg 'angle_phi').

    Synapse angle is defined as the angle between the vector from the post-
    to the presynaptic terminal (synaptic direction vector, perpendicular 
    to the synaptic membranes and the cleft) and the x-axis. This convention
    is named 'phi' mode in convert_synapse_angles().

    Both theta and phy angles are calculated, but only phi is saved because
    normally theta is 90 degrees.

    Synapse angle is determined as follows:
      - From Groups object data, for each experiment, the path to individual
      dataset analysis pickle is found as the value of the property 
      named segment_var, and the pickles are loaded
      - Attribute boundary of the loaded pickled object is read. It is
      expected to be a Segment object containing synapse boundaries
      (labeled presynaptic membrane and the presynaptic segmentation region),
      otherwise an exception is raised.
      - The synaptic direction vector is determined to point from the 
      presynaptic membrane to the presynaptic cytoplasm (see 
      Segment.findDirection()).

    Assumes that the presynaptic membrane label and the presynaptic 
    cytoplasm are the same (specified by args bound_id and segmentation_id, 
    respectively). 

    ToDo: read the labels for each dataset from tomo_info.py files 
    (tomo_metadata, column 'tomo_info_path_rel', load, get variables)

    Sets:
      - Adds property angle_name to data

    Arguments:
      - data: Groups object containing all synapses
      - segment_var: property of data that contains the part to a pickled
      individual dataset object that contains labeled synapse as attribute
      boundary
      - bound_id: label id of presynaptic membrane
      - segmentation_id: label id of presynaptic membrane (segmentation 
      region) 
      - angle_name: name of the property holding the determined angles 
    """
 
    # get and sort scalar dataframe
    sdata = data.scalar_data
    sdata['file'] = sdata[segment_var]
    #sdata['file'] = sdata[segment_var].apply(lambda x: x.lstrip(rm_prefix))
    sdata.sort_values('identifiers', ascending=True, inplace=True)

    # loop over tomos
    for id_, (ident, gr, pkl_path) in sdata[
            ['identifiers', 'group', 'file']].iterrows():

        # determine the angle
        struct = pickle.load(open(pkl_path, 'rb'), encoding='latin1')
        vector = struct.boundary.findDirection(
            segmentId=bound_id, directionId=segmentation_id, thick=1)
        
        # add the angle to the groups 
        phi_deg =  vector.phi * 180 / numpy.pi
        data[gr].setValue(identifier=ident, name=angle_name, value=phi_deg)
    

##############################################################
#
# Other statistics functions (should be integrated with the rest)
#

def connectivity_factorial(
        data, groups, identifiers=None, name='n_connection', mode='positive'):
    """
    Calculates interaction term for 4 sets of data obtained under two
    conditions.

    Uses property n_connection to calculate fraction connected for each
    experiemnt. In other words, the data points for one condition consist
    of individual values corresponding to experiments.
    """

    # extract values
    #values = [
    #    numpy.array([len(x[x>0]) / float(len(x))
    #                 for x in getattr(data[group], name)])
    #    for group in groups]

    total_conn = []
    for group in groups:
        conn_values = []
        for ident in data[group].identifiers:
            if (identifiers is None) or (ident in identifiers):
                x = data[group].getValue(name=name, identifier=ident)
                if mode is None:
                    conn_values.extend(x)
                elif mode == 'join':
                    conn_values.append(x.sum() / float(len(x)))
                elif mode == 'positive':
                    conn_values.append(len(x[x > 0]) / float(len(x)))
        total_conn.append(numpy.asarray(conn_values))

    # calculate
    anova_factorial(*total_conn)

def anova_factorial(data_11, data_12, data_21, data_22):
    """
    ANOVA analysis of 2x2 factorial experimental design.
    """

    # make sure ndarrays
    data_11 = numpy.asarray(data_11)
    data_12 = numpy.asarray(data_12)
    data_21 = numpy.asarray(data_21)
    data_22 = numpy.asarray(data_22)

    # all data
    tot = numpy.hstack((data_11, data_12, data_21, data_22))
    ss_tot = (tot**2).sum() - tot.sum()**2 / float(len(tot))

    # ss between columns
    ss_col = (
        numpy.hstack((data_11, data_21)).sum()**2 /
        (float(len(data_11) + len(data_21)))
        + numpy.hstack((data_12, data_22)).sum()**2 /
        (float(len(data_12) + len(data_22)))
        - tot.sum()**2 / float(len(tot)))

    # ss between rows
    ss_row = (
        numpy.hstack(
            (data_11, data_12)).sum()**2 / (float(len(data_11) + len(data_12)))
        + numpy.hstack(
            (data_21, data_22)).sum()**2 / (float(len(data_21) + len(data_22)))
        - tot.sum()**2 / float(len(tot)))

    # ss interaction
    ss_int = (
        data_11.sum()**2 / float(len(data_11))
        + data_12.sum()**2 / float(len(data_12))
        + data_21.sum()**2 / float(len(data_21))
        + data_22.sum()**2 / float(len(data_22))
        - tot.sum()**2 / float(len(tot))
        - (ss_col + ss_row))

    # ss error
    ss_err = ss_tot - (ss_col + ss_row + ss_int)
    ms_err = ss_err / float(
        len(data_11) + len(data_12) + len(data_21) + len(data_22) - 4)

    # f values and significances
    f_col = ss_col / ms_err
    p_col = scipy.stats.f.sf(f_col, dfn=1, dfd=len(tot)-4)
    print("Columns (1&3 vs 2&4): f = %f6.2  p = %f7.5" % (f_col, p_col))

    f_row = ss_row / ms_err
    p_row = scipy.stats.f.sf(f_row, dfn=1, dfd=len(tot)-4)
    print("Rows (1&2 vs 3&4):    f = %f6.2  p = %f7.5" % (f_row, p_row))

    f_int = ss_int / ms_err
    p_int = scipy.stats.f.sf(f_int, dfn=1, dfd=len(tot)-4)
    print("Interaction:          f = %f6.2  p = %f7.5" % (f_int, p_int))


##############################################################
#
# Miscelaneous functions
#

def str_attach(string, attach):
    """
    Inserts '_' followed by attach in front of the right-most '.' in string and
    returns the resulting string.

    For example:
      str_attach(string='sv.new.pkl', attach='raw') -> 'sv.new_raw.pkl)
    """

    string_parts = list(string.rpartition('.'))
    string_parts.insert(-2, '_' + attach)
    res = ''.join(string_parts)

    return res
