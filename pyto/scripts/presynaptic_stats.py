#!/usr/bin/env python
"""
Statstical analysis of the presynaptic terminal.

Common usage contains the following steps:

1) Make structure specific pickles. These pickles contain data from all
individual segmentation & analysis pickles. This step needs to be executed
only when a new individual segmentation & analysis pickle is added.

  - Make catalog files for all datasets in ../catalog/
  - Copy this script to a desired directory and (optionally) rename it to
    work.py
  - Adjust parameters in this script
  - cd to this script directory, then in ipython (or similar) execute:
      > import work
      > from work import *
      > work.main(individual=True, save=True)

  This should create few pickle files in this directory (tethers.pkl,
  sv.pkl, ...)

2) (depreciated) Create profile in order to keep profile-specific history.
Not required, can be used only in IPython qtconsole. Doesn't
work with Jupyter. Need to be done only once.

  - Create ipython profile for your project:
      $ ipython profile create <project_name>
  - Create startup file for this profile, it enough to have:
      import work
      work.main()
      from work import *

3a) Statistical analysis in Jupyter notebook or qtconsole (current way)

  - Start jupyter notebook or qtconsole
  - Setup plotting (in notebook /  qtconsole):
      %matplotlib inline
  - Load and preprocess structure dependent pickles (in notebook / qtconsole):
      import work
      work.main()
      from work import *
  - Copy desired individual analysis commands from this script (see SV
  distribution and the following sections), paste them in the notebook /
  qtconsole and execute
  - Save notebook

3b) Statistical analysis in IPython qtconsole (used before Jupyter)

  - Start ipython (qtconsole is optional, profile only if created):
      $ ipython qtconsole  --profile=project_name --pylab=qt
  - In ipython (not needed if profile used):
      > import work
      > work.main()
      > from work import *
  - Run individual analysis commands (from SV distribution section and below)

  - If work.py is edited during an ipython session:
      > reload(work)
      > from work import *


# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
#from builtins import str
from builtins import zip
from builtins import range
#from past.utils import old_div
from past.builtins import basestring

__version__ = "$Revision$"


import sys
import os
import logging
import itertools
from copy import copy, deepcopy
import pickle

import numpy
import scipy
import scipy.stats
try:
    import matplotlib as mpl
    mpl_major_version = mpl.__version__[0]
    import matplotlib.pyplot as plt
except ImportError:
    pass

import pyto
from pyto.analysis.groups import Groups
from pyto.analysis.observations import Observations
import pyto.scripts.multi_dataset_util as util


# to debug replace INFO by DEBUG
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(module)s.%(funcName)s():%(lineno)d %(message)s',
    datefmt='%d %b %Y %H:%M:%S')


##############################################################
#
# Parameters (need to be edited)
#
##############################################################

###########################################################
#
# Categories, identifiers and file names
#

# experiment categories to be used (prints and plots according to this order)
categories = ['ko_1', 'wt_1', 'ko_2', 'wt_2']

# experiment identifiers
identifiers = [
    'ko_1_1', 'ko_1_3', 'ko_1_6',
    'wt_1_1', 'wt_1_2',
    'ko_2_1', 'ko_2_2', 'ko_2_5',
    'wt_2_3', 'wt_2_4']

# select identifiers to remove
identifiers_discard = ['ko_1_3']

# For running this script directly:
#   - identifiers are defined here
# For running a notebook that imports this script:
#   - identifiers variable need only be declared (None, or any value is fine)
#   - identifiers are defined in the notebook
#identifiers = None

# set identifiers
identifiers = [ident for ident in identifiers
               if ident not in identifiers_discard]
#identifiers = None

# reference categories
#  - each key category is statistially compared with its corresponding
#  value category
#  - control-like categories are compared to themselves (kay and value
#  are the same)
reference = {
    'ko_1': 'wt_1',
    'wt_1': 'wt_1',
    'ko_2': 'wt_2',
    'wt_2': 'wt_2'}

# catalog directory
# Important note: Catalog files have to be in a directory parallel to this
# one (because relative paths specified in catalog files are not converted
# to be relative to this file). This isn't needed if paths specified in
# catalog files are absolute, but that is discouraged because it causes
# portability problems.
catalog_directory = '../catalog'

# catalogs file name pattern (can be a list of patterns)
catalog_pattern = r'[^_].*\.py$'   # extension 'py', doesn't start with '_'

# Names of properties defined in catalogs (that define data files)
# The values here have to be the same as the property names
# (variable names) in catalog files
tethers_name = 'tethers_file'
connectors_name = 'connectors_file'
sv_name = 'sv_file'
sv_membrane_name = 'sv_membrane_file'
sv_lumen_name = 'sv_lumen_file'
layers_name = 'layers_file'
clusters_name = 'cluster_file'

# Names of structure spcific pickles generated by this script
# (pyto.analysis.Connections, Vesicles and Layers pickle files;
# names are relative to this directory)
sv_pkl = 'sv.pkl'
tethers_pkl = 'tether.pkl'
connectors_pkl = 'conn.pkl'
layers_pkl = 'layer.pkl'
clusters_pkl = 'cluster.pkl'

###########################################################
#
# Parameters
#
#   - All lengths (radii, distances, lengths) are in nm

# vesicle radius bins (small, medium, large), svs are medium
vesicle_radius_bins = [0, 10, 30, 3000]

# Distance bins for layers
# The largest bin limit, when converted to pixels, has to be smaller
# than the number of layers (as set in the layers.py script)
distance_bins = [0, 45, 75, 150, 250]
layer_distance_bins = [10, 45, 75, 150, 250]
distance_bin_references = {
             'proximal': 'proximal',
             'intermediate': 'proximal',
             'distal_1': 'proximal',
             'distal_2': 'proximal',
             'all': 'all'}
reference.update(distance_bin_references)
distance_bin_names = ['proximal', 'intermediate', 'distal_1', 'distal_2']
#distance_bin_num = ['0-45', '45-75', '75-150', '150-250']
distance_bins_label = 'Distance to the AZ [nm]'

# tether and connector length bins
rough_length_bins = [0, 5, 10, 20, 40]
rough_length_bin_names = ['<5', '5-9', '10-19', '20-40']
fine_length_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40]
fine_length_bin_names = ['<5', '5-9', '10-14', '15-19', '20-24',
                         '25-29', '30-34', '35-40']
tether_length_bins = [0, 5, 500]
tether_length_bin_names = ['short', 'long']

# connected_bins = [0,1,100]
teth_bins_label = ['non-tethered', 'tethered']
conn_bins_label = ['non-connected', 'connected']

# number of connector bins
n_conn_bins = [0,1,3,100]
n_conn_bin_names = ['0', '1-2', '>2']
n_conn_bins_label = 'N connectors'
reference.update(
    {'0': '0',
     '1-2': '0',
     '>2': '0'})

# RRP definition by the number of tethers
rrp_ntether_bins = [0, 3, 300]
rrp_ntether_bin_names = ['non-rrp', 'rrp']

# connected and tethered
reference.update(
    {'near_conn_sv': 'near_conn_sv',
     'near_non_conn_sv': 'near_conn_sv',
     'near_teth_sv': 'near_teth_sv',
     'near_non_teth_sv': 'near_teth_sv',
     'tethered': 'tethered',
     'non_tethered': 'tethered',
     'connected': 'connected',
     'non_connected': 'connected',
     't c': 't c',
     'nt c': 't c',
     't nc': 't c',
     'nt nc': 't c',
     't_c': 't_c',
     'nt_c': 't_c',
     't_nc': 't_c',
     'nt_nc': 't_c',
     'rrp': 'rrp',
     'non_rrp': 'rrp'})

# radius
radius_bins = numpy.arange(10, 30, 0.5)
radius_bin_names = [str(low) + '-' + str(up) for low, up
                    in zip(radius_bins[:-1], radius_bins[1:])]


###########################################################
#
# Printing
#

# print format
print_format = {

    # data
    'nSegments': '   %5d ',
    'surfaceDensityContacts_1': '   %7.4f ',
    'surfaceDensityContacts_2': '   %7.4f ',

    # stats
    'mean': '    %5.2f ',
    'std': '    %5.2f ',
    'sem': '    %5.2f ',
    'diff_bin_mean': '   %6.3f ',
    'n': '    %5d ',
    'testValue': '   %7.4f ',
    'confidence': '  %7.4f ',
    'fraction': '    %5.3f ',
    'count': '    %5d ',
    't_value': '    %5.2f ',
    't': '    %5.2f ',
    'h': '    %5.2f ',
    'u': '    %5.2f ',
    't_bin_value': '    %5.2f ',
    't_bin': '    %5.2f ',
    'chi_square': '    %5.2f ',
    'chi_sq_bin': '    %5.2f ',
    'confid_bin': '  %7.4f '
    }

###########################################################
#
# Plotting
#
# Note: These parameters are used directly in the plotting functions of
# this module, they are not passed as function arguments.
#

# determines if data is plotted
plot_ = True

# legend
legend = True

# data color
color = {
    'wt_1': 'black',
    'ko_1': 'grey',
    'wt_2': 'tan',
    'ko_2': 'orange'
    }

# edge color
ecolor={}

# category labels
category_label = {
    'ko_1': 'KO type 1',
    'wt_1': 'WT for KO 1',
    'ko_2': 'KO type 2',
    'wt_2': 'WT for KO 2',
    'near_conn_sv': 'connected',
    'near_non_conn_sv': 'non-connected',
    'near_teth_sv': 'tethered',
    'near_non_teth_sv': 'non-tethered',
    'near_conn_sv': 'connected',
    'near_non_conn_sv': 'non-connected',
    'non_connected': 'non-connected',
    'connected': 'connected',
    'non_tethered': 'non-tethered',
    'tethered': 'tethered',
    'proximal': '0-45',
    'intermediate': '45-75',
    'distal_1': '75-150',
    'distal_2': '150-250',
    't_c': 'teth conn',
    't_nc': 'teth non-conn',
    'nt_c': 'non-teth conn',
    'nt_nc': 'non-teth non-conn',
    'rrp': 'RRP',
    'non_rrp': 'non-RRP'
    }

# data alpha
alpha = {
    'ko_1': 1,
    'wt_1': 1,
    'ko_2': 1,
    'wt_2': 1
    }

# markers
marker =  {
    'ko_1': 'o',
    'wt_1': '*',
    'ko_2': 'x',
    'wt_2': 'v'
    }

# list of default markers
markers_default = ['o', 'v', 's', '*', '^', 'H', '<', 'D', '>']

# for presentations
#matplotlib.rcParams['font.size'] = 18

# markersize
marker_size = 7

# line width
line_width = {
    'mean': 4
    }

# line width
default_line_width = 2

# thick line width
thick_line_width = 4

# line style
default_line_style = '-'

# bar arrangement
bar_arrange = 'uniform'    # all bars in a row, single bar space between groups
bar_arrange = 'grouped'    # each group takes the same space

# flag indicating if one-sided yerr
one_side_yerr = True

# confidence format
confidence_plot_format = "%5.3f"

# confidence font size
confidence_plot_font_size = 7

# confidence levels for 1, 2, 3, ... stars
confidence_stars = [0.05, 0.01, 0.001]


##############################################################
#
# Functions moved to multi_dataset_util
#
##############################################################


################################################################
#
# Main function (edit to add additional analysis files and plot (calculate)
# other properties)
#
###############################################################

def main(individual=False, save=False, analyze=False):
    """
    Arguments:
      - individual: if True read data for each tomo separately, otherwise
      read from pickles containing partailly analyzed data
      - save: if True save pickles containing partailly analyzed data
      - analyze: do complete analysis
    """

    # read catalogs and make groups
    global catalog
    try:
        curr_dir, base = os.path.split(os.path.abspath(__file__))
    except NameError:
        # needed for running from ipython - perhaps not anymore
        curr_dir = os.getcwd()
    cat_dir = os.path.normpath(os.path.join(curr_dir, catalog_directory))
    catalog = pyto.analysis.Catalog(
        catalog=catalog_pattern, dir=cat_dir, identifiers=identifiers)
    catalog.makeGroups(feature='treatment')

    # make absolute paths for pickle files
    sv_pkl_abs = os.path.join(curr_dir, sv_pkl)
    tethers_pkl_abs = os.path.join(curr_dir, tethers_pkl)
    connectors_pkl_abs = os.path.join(curr_dir, connectors_pkl)
    layers_pkl_abs = os.path.join(curr_dir, layers_pkl)
    clusters_pkl_abs = os.path.join(curr_dir, clusters_pkl)

    # prepare for making or reading data pickles
    global sv, tether, conn, layer, clust

    ##########################################################
    #
    # Read individual tomo data, calculate few things and save this
    # preprocessed data, or just read the preprocessed data
    #

    if individual:

        # read sv data
        sv_files = getattr(catalog, sv_name)
        sv_membrane_files = getattr(catalog, sv_membrane_name)
        sv_lumen_files = getattr(catalog, sv_lumen_name)
        logging.info("Reading sv")
        sv = pyto.analysis.Vesicles.read(
            files=sv_files, catalog=catalog, categories=categories,
            membrane=sv_membrane_files, lumen=sv_lumen_files,
            additional=[])

        # read tether data
        tether_files = getattr(catalog, tethers_name)
        logging.info("Reading tethers")
        tether = pyto.analysis.Connections.read(
            files=tether_files, mode='connectors', catalog=catalog,
            categories=categories, order=sv)

        # read connector data
        conn_files = getattr(catalog, connectors_name)
        logging.info("Reading connectors")
        conn = pyto.analysis.Connections.read(
            files=conn_files, mode='connectors', catalog=catalog,
            categories=categories, order=sv)

        # read layer data
        layer_files = getattr(catalog, layers_name)
        logging.info("Reading layers")
        layer = pyto.analysis.Layers.read(files=layer_files, catalog=catalog,
                                          categories=categories, order=sv)

        # read cluster data
        cluster_files = getattr(catalog, clusters_name, None)
        if cluster_files is not None:
            logging.info("Reading clusters")
            clust = pyto.analysis.Clusters.read(
                files=cluster_files, mode='connectivity', catalog=catalog,
                categories=categories, distances='default', order=sv)
            hi_clust_bound = pyto.analysis.Clusters.read(
                files=cluster_files, mode='hierarchicalBoundaries',
                catalog=catalog, categories=categories, order=sv)
            hi_clust_conn = pyto.analysis.Clusters.read(
                files=cluster_files, mode='hierarchicalConnections',
                catalog=catalog, categories=categories, order=sv)
        else:
            clust = None

        # pickle raw?

        # find sv nearest neighbors
        if clust is not None:
            sv.getNearestNeighbor(
                cluster=clust, dist_name='bound_dist', name='nearest',
                categories=categories)

        # separate svs by size
        [small_sv, sv, big_sv] = sv.splitByRadius(radius=vesicle_radius_bins)

        # remove bad svs from tethers and connections
        tether.removeBoundaries(boundary=small_sv)
        tether.removeBoundaries(boundary=big_sv)
        conn.removeBoundaries(boundary=small_sv)
        conn.removeBoundaries(boundary=big_sv)

        # calculate number of tethers, connections and linked svs for each sv
        sv.getNTethers(tether=tether)
        sv.getNConnections(conn=conn)
        sv.addLinked(files=conn_files)
        sv.getNLinked()
        if clust is not None:
            sv.getClusterSize(clusters=clust)

        # calculate number of items and max cluster fraction
        if clust is not None:
            clust.findNItems()
            clust.findBoundFract()
            clust.findRedundancy()

        # pickle
        if save:
            pickle.dump(sv, open(sv_pkl_abs, 'wb'), -1)
            pickle.dump(tether, open(tethers_pkl_abs, 'wb'), -1)
            pickle.dump(conn, open(connectors_pkl_abs, 'wb'), -1)
            pickle.dump(layer, open(layers_pkl_abs, 'wb'), -1)
            if clust is not None:
                pickle.dump(clust, open(clusters_pkl_abs, 'wb'), -1)

    else:

        # unpickle
        try:
            # python 3
            sv = pickle.load(open(sv_pkl_abs, 'rb'), encoding='latin1')
            tether = pickle.load(
                open(tethers_pkl_abs, 'rb'), encoding='latin1')
            conn = pickle.load(
                open(connectors_pkl_abs, 'rb'), encoding='latin1')
            layer = pickle.load(open(layers_pkl_abs, 'rb'), encoding='latin1')
        except TypeError:
            # python 2
            sv = pickle.load(open(sv_pkl_abs))
            tether = pickle.load(open(tethers_pkl_abs))
            conn = pickle.load(open(connectors_pkl_abs))
            layer = pickle.load(open(layers_pkl_abs))
        try:
            try:
                # python 3
                clust = pickle.load(
                    open(clusters_pkl_abs, 'rb'), encoding='latin1')
            except TypeError:
                # python2
                clust = pickle.load(open(clusters_pkl_abs))
        except IOError:
            clust = None

        # keep only specified groups and identifiers
        if clust is not None:
            for obj in [sv, tether, conn, layer, clust]:
                obj.keep(groups=categories, identifiers=identifiers,
                         removeGroups=True)
        else:
            for obj in [sv, tether, conn, layer]:
                obj.keep(groups=categories, identifiers=identifiers,
                         removeGroups=True)

    ##########################################################
    #
    # Separate data in various categories
    #

    # split svs by distance
    global bulk_sv, sv_bins, near_sv, inter_sv, dist_sv, inter_dist_sv
    bulk_sv = sv.splitByDistance(distance=distance_bins[-1])
    sv_bins = sv.splitByDistance(distance=distance_bins)
    near_sv = sv_bins[0]
    inter_sv = sv_bins[1]
    dist_sv = sv.splitByDistance(distance=[distance_bins[2],
                                           distance_bins[-1]])[0]
    inter_dist_sv = sv.splitByDistance(
        distance=[distance_bins[1], distance_bins[-1]])[0]

    # split layers by distance
    global layer_bin
    layer_bin = layer.rebin(bins=distance_bins, pixel=catalog.pixel_size)

    # extract svs that are near az, tethered, near+tethered, near-tethered
    global teth_sv, non_teth_sv, near_teth_sv, near_non_teth_sv
    teth_sv, non_teth_sv = bulk_sv.extractTethered(other=True)
    near_teth_sv, near_non_teth_sv = near_sv.extractTethered(other=True)

    # extract connected and non-connected svs
    global conn_sv, non_conn_sv, bulk_conn_sv, bulk_non_conn_sv
    global near_conn_sv, near_non_conn_sv, inter_conn_sv, inter_non_conn_sv
    global inter_dist_conn_sv, inter_dist_non_conn_sv
    conn_sv, non_conn_sv = sv.extractConnected(other=True)
    bulk_conn_sv, bulk_non_conn_sv = bulk_sv.extractConnected(other=True)
    near_conn_sv, near_non_conn_sv = near_sv.extractConnected(other=True)
    inter_conn_sv, inter_non_conn_sv = sv_bins[1].extractConnected(other=True)
    inter_dist_conn_sv, inter_dist_non_conn_sv = inter_dist_sv.extractConnected(
        other=True)

    # extract by tethering and connectivity
    global near_teth_conn_sv, near_teth_non_conn_sv
    near_teth_conn_sv, near_teth_non_conn_sv = \
        near_teth_sv.extractConnected(other=True)
    global near_non_teth_conn_sv, near_non_teth_non_conn_sv
    near_non_teth_conn_sv, near_non_teth_non_conn_sv = \
        near_non_teth_sv.extractConnected(other=True)

    # calculate additional properties for different vesicle objects
    for xxx_sv in [near_sv, near_teth_sv, near_non_teth_sv, near_teth_conn_sv,
                   near_non_teth_conn_sv, near_teth_non_conn_sv,
                   near_non_teth_non_conn_sv,
                   teth_sv, non_teth_sv]:
        util.calculateVesicleProperties(
            data=xxx_sv, layer=layer, tether=tether, categories=categories)

    # calculate additional properties for different tether objects
    util.calculateTetherProperties(
        data=tether, layer=layer, categories=categories)

    # split near_sv and tether according to rrp (defined as >2 tethered)
    global sv_non_rrp, sv_rrp, tether_rrp, tether_non_rrp
    sv_non_rrp, sv_rrp = near_sv.split(
        name='n_tether', value=rrp_ntether_bins, categories=categories)
    tether_rrp = tether.extractByVesicles(
        vesicles=sv_rrp, categories=categories, other=False)
    tether_non_rrp = tether.extractByVesicles(
        vesicles=sv_non_rrp, categories=categories, other=False)
    util.calculateVesicleProperties(
        data=sv_rrp, layer=layer, tether=tether_rrp, categories=categories)
    util.calculateVesicleProperties(
        data=sv_non_rrp, layer=layer, tether=tether_non_rrp,
        categories=categories)
    util.calculateTetherProperties(
        data=tether_rrp, layer=layer, categories=categories)
    util.calculateTetherProperties(
        data=tether_non_rrp, layer=layer, categories=categories)

    # split tethers according to their length
    global short_tether, long_tether
    short_tether, long_tether = tether.split(
        name='length_nm', value=tether_length_bins, categories=categories)
    util.calculateTetherProperties(
        data=short_tether, layer=layer, categories=categories)
    util.calculateTetherProperties(
        data=long_tether, layer=layer, categories=categories)

    # calculate n short and long tethers for proximal vesicles
    near_sv.getNConnections(conn=short_tether, name='n_short_tether',
                            categories=categories)
    near_sv.getNConnections(conn=long_tether, name='n_long_tether',
                            categories=categories)

    # stop here if no analysis
    if not analyze: return

    ###########################################################
    #
    # What follows is an extensive list of function calls. Selected
    # functions are ment to be executed somewhere else (e.g. in a
    # Jupyter notebook), rather than all functions executed here.
    #
    # These function calls can be copied to IPython / Jupyter notebook
    # provided that it:
    #   - imports this modules as pp
    #   - imports pyto.script.multi_dataset_util as util (as in this module)

    # this module
    pp = sys.modules[__name__]

    ###########################################################
    #
    # SV distribution
    #

    # plot individual vesicle occupancy
    util.plot_layers(
        data=layer, mode='all', pp=pp, groups=categories,
        identifiers=identifiers)

    # plot individual vesicle occupancy with means
    util.plot_layers(
        data=layer, mode='all&mean', pp=pp,
        groups=categories, identifiers=identifiers)

    # plot individual vesicle occupancy with means
    util.plot_layers(
        data=layer, mode='mean', pp=pp, groups=categories,
        identifiers=identifiers, title="Mean vesicle occupancy")

    # mean occupancy for all vesicles within 250 nm to the AZ
    util.analyze_occupancy(
        layer=layer, bins=[0, 250], bin_names=["all"],
        pixel_size=catalog.pixel_size, pp=pp, groups=categories,
        identifiers=identifiers, test='t', reference=reference, ddof=1,
        out=sys.stdout, outNames=None, yerr='sem', confidence='stars',
        title='SV occupancy', y_label='Fraction of volume occupied by svs')

    # mean occupancy in distance bins
    util.analyze_occupancy(
        layer=layer, bins=distance_bins, bin_names=distance_bin_names,
        pixel_size=catalog.pixel_size, pp=pp, groups=categories,
        identifiers=identifiers, test='t', reference=reference, ddof=1,
        out=sys.stdout, outNames=None, yerr='sem', confidence='stars',
        title='SV occupancy', y_label='Fraction of volume occupied by svs')

    # min distance to the AZ, fine histogram for proximal svs
    util.stats(data=near_sv, name='minDistance_nm', bins=fine_length_bins,
          bin_names=fine_length_bin_names, join='join', groups=categories,
          identifiers=identifiers, test='chi2', reference=reference,
          x_label='Distance to the AZ [nm]', y_label='N vesicles',
          title='Histogram of min distance to the AZ of proximal svs')

    # Min distance to the AZ for near svs
    util.stats(
        data=near_sv, name='minDistance_nm', join='join', groups=categories,
        pp=pp, identifiers=identifiers, test='t', reference=reference,
        ddof=1, out=sys.stdout, y_label='Mean [nm]',
        title='Min distance of proximal svs to the AZ')

    # min sv distance to the AZ dependence on connectivity
    util.stats_list(
        data=[near_conn_sv, near_non_conn_sv], pp=pp,
        dataNames=['connected', 'non_connected'], name='minDistance_nm',
        join='join', groups=categories, identifiers=identifiers,
        test='t', reference=reference,  ddof=1, out=sys.stdout,
        title='Min distance of proximal svs to the AZ by connectivity',
        y_label='Mean min distance to the AZ [nm]')

    # min sv distance to the AZ dependence on tethering
    util.stats_list(
        data=[near_non_teth_sv, near_teth_sv], pp=pp,
        dataNames=['non_tethered', 'tethered'], name='minDistance_nm',
        join='join', groups=categories, identifiers=identifiers,
        test='t', reference=reference,  ddof=1, out=sys.stdout,
        title='Min distance of proximal svs to the AZ by tethering',
        y_label='Mean min distance to the AZ [nm]')

    # ToDo:
    #   - ratio of occupancies between intermediate and proximal
    #   - ratio between min and max occupancies for each tomo

    ###########################################################
    #
    # Analyze sv radii
    #

    # radius of bulk svs
    util.stats(
        data=bulk_sv, name='radius_nm', join='join', pp=pp, groups=categories,
        identifiers=identifiers, reference=reference, test='t',
        y_label='Raduis [nm]', title='Vesicle radius')

    # sv radius dependence on distance to the AZ
    util.stats_list(
        data=sv_bins, dataNames=distance_bin_names, name='radius_nm',
        join='join', pp=pp, groups=categories, identifiers=identifiers,
        reference=reference, test='t', y_label='Radius [nm]',
        title='Vesicle radius dependence on distance to the AZ')

    # sv radius of proximal svs dependence on tethering
    util.stats_list(
        data=[near_non_teth_sv, near_teth_sv],
        dataNames=['non_tethered', 'tethered'],
        name='radius_nm', join='join', pp=pp, groups=categories,
        identifiers=identifiers, reference=reference, test='t',
        y_label='Radius [nm]',
        title='Proximal sv radius dependence on tethering')

    # radius dependence on number of connectors
    util.stats_list(
        data=bulk_sv.split(value=[0,1,3,100], name='n_connection'),
        dataNames=['0', '1-2', '>2'], name='radius_nm', join='join', pp=pp,
        groups=categories, identifiers=identifiers, reference=reference,
        test='t', y_label='Radius [nm]',
        title='Radius dependence on N connectors')

    # ananlyze dependence on both tethering and connectivity
    util.stats_list(
        data=[near_teth_conn_sv, near_teth_non_conn_sv,
              near_non_teth_conn_sv, near_non_teth_non_conn_sv],
        dataNames=['t c', 't nc', 'nt c', 'nt nc'], name='radius_nm',
        join='join', groups=categories, identifiers=identifiers,
        reference=reference, test='t', y_label='Radius [nm]',
        title='Radius dependence on connectivity and tethering')

    # radius histogram of all groups together
    plot_histogram(
        data=bulk_sv, name='radius_nm', bins=radius_bins, pp=pp,
        groups=categories, identifiers=identifiers, x_label='Radius [nm]',
        title='Vesicle radius histogram of all groups together')

    # radius histogram of one group
    plot_histogram(
        data=bulk_sv, name='radius_nm', bins=radius_bins, pp=pp, groups='ko_1',
        identifiers=identifiers, x_label='Radius [nm]',
        title='Vesicle radius histogram of ko_1')


    ###########################################################
    #
    # Vesicle density analysis
    #

    # vesicle lumen density comparison between tethered vs non-tethered, paired
    util.stats_list_pair(
        data=[non_teth_sv, teth_sv], dataNames=['non_tethered', 'tethered'],
        name='lumen_density', pp=pp, groups=categories, identifiers=identifiers,
        reference=reference, test='t_rel', label='experiment',
        y_label='Lumenal density',
        title='Lumenal density comparison between tethered and non-tethered')

    # vesicle lumen density comparison between proximal tethered vs
    # non-tethered, paired
    util.stats_list_pair(
        data=[near_non_teth_sv, near_teth_sv],
        dataNames=['non_tethered', 'tethered'], name='lumen_density',
        pp=pp, groups=categories, identifiers=identifiers,
        reference=reference, test='t_rel', label='experiment',
        y_label='Lumenal density',
        title=('Lumenal density comparison between proximal tethered and '
               + 'non-tethered'))

    # vesicle membrane density comparison between tethered vs non-tethered,
    # paired
    util.stats_list_pair(
        data=[non_teth_sv, teth_sv], dataNames=['non_tethered', 'tethered'],
        name='membrane_density', pp=pp, groups=categories,
        identifiers=identifiers, reference=reference, test='t_rel',
        label='experiment', y_label='Lumenal density',
        title='Membrane density comparison between tethered and non-tethered')

    # vesicle lumen density comparison between connected vs non-connected,
    # paired
    util.stats_list_pair(
        data=[non_conn_sv, conn_sv], dataNames=['non_connected', 'connected'],
        name='lumen_density', pp=pp, groups=categories,
        identifiers=identifiers, reference=reference, test='t_rel',
        label='experiment', y_label='Lumenal density',
        title='Lumenal density comparison between connected and non-connected')

    # vesicle membrane density comparison between connected vs non-connected,
    # paired
    util.stats_list_pair(
        data=[non_teth_sv, teth_sv], dataNames=['non_connected', 'connected'],
        name='membrane_density', pp=pp, groups=categories,
        identifiers=identifiers, reference=reference, test='t_rel',
        label='experiment', y_label='Lumenal density',
        title='Membrane density comparison between connected and non-connected')

    # difference between lumenal and membrane density vs distance
    util.stats_list(
        data=sv_bins, dataNames=distance_bin_names, name='lum_mem_density_diff',
        join='join', pp=pp, groups=categories, identifiers=identifiers,
        reference=reference, test='t',
        y_label='Difference between lumenal and membrane density',
        title='Lumenal - membrane density dependence on distance to the AZ')

    # difference between lumenal and membrane density dependence on
    # connectivity, paired
    util.stats_list_pair(
        data=[non_conn_sv, conn_sv], dataNames=['non_connected', 'connected'],
        name='lum_mem_density_diff', pp=pp, groups=categories,
        identifiers=identifiers, reference=reference, test='t_rel',
        label='experiment',
        y_label='Difference between lumenal and membrane density',
        title='Lumenal and membrane density dependence on connectivity')


    ###########################################################
    #
    # Vesicle clustering analysis
    #

    # fraction of total vesicles in a largest cluster
    util.stats(
        data=clust, name='fract_bound_max', join='join', pp=pp,
        groups=categories, identifiers=identifiers, reference=reference,
        test='kruskal', y_label='Fraction of total vesicles',
        title='Fraction of total vesicles in the largest cluster')

    # histogram of sv cluster sizes
    # to fix stats and x-labels
    util.stats(
        data=clust, name='n_bound_clust', join='join', bins=[1, 2, 5, 50, 3000],
        bin_names=['1', '2-4', '5-49', '50+'], pp=pp, groups=categories,
        identifiers=identifiers, reference=reference, test='chi2',
        y_label='N clusters', title='Histogram of cluster sizes')

    # loops / connections per tomo
    util.stats(
        data=clust, name='redundancy_obs', join='join', pp=pp,
        groups=categories, identifiers=identifiers, reference=reference,
        test='kruskal', y_label='N loops / N connectors',
        title='Redundancy (loops per connector) per observation')

    # loops / links per tomo
    util.stats(
        data=clust, name='redundancy_links_obs', join='join', pp=pp,
        groups=categories, identifiers=identifiers, reference=reference,
        test='kruskal', y_label='N loops / N links',
        title='Redundancy (loops per link) per observation')

    ###########################################################
    #
    # Connectivity analysis of svs
    #

    # fraction of svs that are connected
    util.stats(
        data=bulk_sv, name='n_connection', join='join', bins=[0, 1, 100],
        fraction=1, pp=pp, groups=categories, identifiers=identifiers,
        test='chi2', reference=reference, y_label='Fraction of all vesicles',
        title='Fraction of vesicles that are connected')

    # fraction of svs that are connected per distance bins
    util.stats_list(
        data=sv_bins, dataNames=distance_bin_names,  pp=pp, groups=categories,
        identifiers=identifiers, name='n_connection', bins=[0, 1, 100],
        join='join', test='chi2', reference=reference,
        x_label=distance_bins_label, y_label='Fraction of svs',
        title='Fraction of connected svs')

    # fraction of proximal svs that are connected
    util.stats(
        data=near_sv, name='n_connection', join='join', bins=[0, 1, 100],
        fraction=1, pp=pp, groups=categories, identifiers=identifiers,
        test='chi2', reference=reference, y_label='Fraction of vesicles',
        title='Fraction of proximal vesicles that are connected')

    # connectivity interaction beween wt / tko and w/wo aox
    connectivity_factorial(
        data=near_sv, groups=['snc_wt', 'snc_aox', 'snc_tko', 'snc_aox_tko'],
        identifiers=identifiers)

    # n connections per sv
    util.stats(
        data=bulk_sv, name='n_connection', join='join', groups=categories,
        identifiers=identifiers, reference=reference, test='kruskal',
        y_label='N connectors', title='N connectors per vesicle')

    # n connections per connected sv
    util.stats(
        data=bulk_conn_sv, name='n_connection', join='join', pp=pp,
        groups=categories, identifiers=identifiers, reference=reference,
        test='kruskal',
        y_label='N connectors', title='N connectors per connected vesicle')

    # n connections per sv dependence on distance
    util.stats_list(
        data=sv_bins, dataNames=distance_bin_names, name='n_connection',
        join='join', pp=pp, groups=categories, identifiers=identifiers,
        reference=reference, test='kruskal', y_label='N connectors',
        title='N connectors per vesicle')

    # n connections per connected sv dependence on distance
    util.stats_list(
        data=bulk_conn_sv.splitByDistance(distance_bins),
        dataNames=distance_bin_names, name='n_connection',
        join='join', pp=pp, groups=categories, identifiers=identifiers,
        reference=reference, test='kruskal', y_label='N connectors',
        title='N connectors per connected vesicle')

    # histogram of n connectors for connected svs
    util.stats(data=bulk_conn_sv, name='n_connection', bins=[1,2,3,100],
          bin_names=['1', '2', '>2'], join='join', pp=pp, groups=categories,
          identifiers=identifiers, test='chi2', reference=reference,
          y_label='N svs',
          title='Histogram of number of connectors per connected sv')

    # fraction of svs that are linked per distance bins
    util.stats_list(
        data=sv_bins, dataNames=distance_bin_names,  pp=pp, groups=categories,
        identifiers=identifiers, name='n_linked', bins=[0,1,100],  join='join',
        test='chi2', reference=reference,
        x_label=distance_bins_label, y_label='Fraction of svs',
        title='Fraction of connected svs')

    # n links per sv dependence on distance
    util.stats_list(
        data=sv_bins, dataNames=distance_bin_names, name='n_linked',
        join='join', pp=pp, groups=categories, identifiers=identifiers,
        reference=reference, test='kruskal',
        y_label='N connectors', title='N connectors per vesicle')

    # fraction of near svs that are connected dependence on tethering
    util.stats_list(
        data=[near_teth_sv, near_non_teth_sv],
        dataNames=['tethered', 'non_tethered'], name='n_connection',
        join='join', bins=[0,1,100],  pp=pp, groups=categories,
        identifiers=identifiers, test='chi2', reference=reference,
        y_label='Fraction of vesicles that are connected',
        title='Proximal vesicles connectivity')

    # connector length
    # Q: Shouldn't length use t-test?
    util.stats(
        data=conn, name='length_nm', join='join', pp=pp, groups=categories,
        identifiers=identifiers, test='kruskal', reference=reference,
        y_label='Length [nm]', title='Mean connector length')

    # connector length dependence on distance to the AZ
    util.stats_list(
        data=conn.splitByDistance(distance=distance_bins),
        dataNames=distance_bin_names, name='length_nm', join='join',
        pp=pp, groups=categories, identifiers=identifiers, test='kruskal',
        reference=reference, y_label='Length [nm]',
        title='Mean connector length dependence on distance')

    # connector length of proximal svs
    util.stats(
        data=conn.extractByVesicles(vesicles=near_sv, categories=categories)[0],
        name='length_nm', join='join', pp=pp, groups=categories,
        identifiers=identifiers, test='kruskal', reference=reference,
        y_label='Length [nm]',
        title='Mean connector length of proximal vesicles')

    # connector length of proximal svs dependence on tethering
    util.stats_list(
        data=[conn.extractByVesicles(vesicles=near_non_teth_sv)[0],
              conn.extractByVesicles(vesicles=near_teth_sv)[0]],
        dataNames=['non_teth_sv', 'teth_sv'], name='length_nm',
        join='join', pp=pp, groups=categories, identifiers=identifiers,
        test='kruskal', reference=reference, y_label='Length [nm]',
        title='Mean connector length dependence on tethering')

    # connector length histogram
    util.stats(
        data=conn, name='length_nm', bins=rough_length_bins,
        bin_names=rough_length_bin_names, join='join', pp=pp,
        groups=categories, identifiers=identifiers, test='chi2',
        reference=reference, y_label='Number of connectors',
        x_label='Length [nm]', title='Connectors length histogram')


    ###########################################################
    #
    # Tethering based analysis of svs
    #

    # fraction of near svs that are tethered
    util.stats(
        data=near_sv, name='n_tether', join='join', bins=[0,1,100],
        fraction=1, pp=pp, groups=categories, identifiers=identifiers,
        test='chi2', reference=reference, y_label='Fraction of all vesicles',
        title='Fraction of proximal vesicles that are tethered')

    # n tethers per near sv
    util.stats(
        data=near_sv, name='n_tether', join='join', pp=pp, groups=categories,
        identifiers=identifiers, reference=reference, test='kruskal',
        title='N tethers per proximal sv', y_label='N tethers')

    # n tethers per tethered sv
    util.stats(
        data=near_teth_sv, name='n_tether', join='join', pp=pp,
        groups=categories, identifiers=identifiers, reference=reference,
        test='kruskal',
        title='N tethers per tethered proximal sv', y_label='N tethers')

    # histogram of n tethers for near svs
    util.stats(
        data=near_sv, name='n_tether', bins=[0,1,3,100],
        bin_names=['0', '1-2', '>2'], join='join', groups=categories,
        identifiers=identifiers, test='chi2', reference=reference,
        x_label='N tethers', y_label='N svs',
        title='Histogram of number of tethers per proximal sv')

    # histogram of n tethers for proximal tethered svs
    util.stats(
        data=near_teth_sv, name='n_tether', bins=[1,2,3,100],
        bin_names=['0', '1-2', '>2'], join='join', pp=pp, groups=categories,
        identifiers=identifiers, test='chi2', reference=reference,
        x_label='N tethers', y_label='N svs',
        title='Histogram of number of tethers per proximal sv')

    # correlation between min sv distance to the AZ and n tethers
    correlation(
        xData=near_teth_sv, xName='minDistance_nm', yName='n_tether',
        join='join', pp=pp, groups=categories, identifiers=identifiers,
        test='r', x_label='Min distance to the AZ [nm]', y_label='N tethers',
        title=('Proximal sv correlation between min distance and n tethers'))

    # mean tether length vs n tether (for each sv) correlation for tethered svs
    correlation(
        xData=near_teth_sv, yName='n_tether', xName='mean_tether_nm',
        pp=pp, groups=categories, identifiers=identifiers, join='join',
        test='r',
        x_label='Mean tether length per vesicle [nm]', y_label='N tethers',
        title='Correlation between mean tether length (per sv) and N tethers')

    # tether length
    # Q: Shouldn't length use t-test?
    util.stats(
        data=tether, name='length_nm', join='join', pp=pp, groups=categories,
        identifiers=identifiers, test='kruskal', reference=reference,
        y_label='Length [nm]', title='Mean tether length')

    # tether length dependence on connectivity
    util.stats_list(
        data=[
            tether.extractByVesicles(
                vesicles=near_non_conn_sv, categories=categories)[0],
            tether.extractByVesicles(
                vesicles=near_conn_sv, categories=categories)[0]],
        dataNames=['non_connected', 'connected'], name='length_nm',
        join='join', pp=pp, groups=categories, identifiers=identifiers,
        test='kruskal', reference=reference, y_label='Length [nm]',
        title='Mean tether length dependence on connectivity')

    # min tethered sv distance to the AZ dependence on connectivity
    util.stats_list(
        data=[near_teth_non_conn_sv, near_teth_conn_sv],
        dataNames=['non_connected', 'connected'], name='minDistance_nm',
        join='join', pp=pp, groups=categories, identifiers=identifiers,
        test='kruskal', reference=reference, y_label='Min distance [nm]',
        title='Min tethered sv distance dependence on connectivity')

    # tether length histogram, show number of tethers
    util.stats(
        data=tether, name='length_nm', bins=rough_length_bins,
        bin_names=rough_length_bin_names, join='join', pp=pp,
        groups=categories, identifiers=identifiers, test='chi2',
        reference=reference, y_label='Number of tethers',
        x_label='Length [nm]', title='Tether length histogram')

    # tether length histogram, show probability
    util.stats(
        data=tether, name='length_nm', bins=rough_length_bins,
        bin_names=rough_length_bin_names, join='join', pp=pp,
        groups=categories, identifiers=identifiers, test='chi2',
        reference=reference, plot_name='probability',
        y_label='Fraction of tethers',
        x_label='Length [nm]', title='Tether length histogram')

    # mean tether length vs n tether (for each sv) correlation
    correlation(
        xData=near_teth_sv, yName='n_tether', xName='mean_tether_nm',
        pp=pp, groups=categories, identifiers=identifiers, join='join',
        test='r', x_label='Mean tether length [nm]', y_label='N tethers',
        title='Correlation between mean tether length (per sv) and N tethers')

    # correlation min sv distance to the AZ vs n tether, tethered svs
    correlation(
        xData=near_teth_sv, yName='n_tether', xName='minDistance_nm',
        pp=pp, groups=categories, identifiers=identifiers, join='join',
        test='r', x_label='Min sv distance [nm]', y_label='N tethers',
        title='Correlation between sv distance and N tethers for tethered svs')

    ###########################################################
    #
    # Tethering and connectivity
    #

    # fraction of tethered and connected
    count_histogram(
        data=[near_teth_conn_sv, near_teth_non_conn_sv,
              near_non_teth_conn_sv, near_non_teth_non_conn_sv],
        dataNames=['t_c', 't_nc', 'nt_c', 'nt_nc'], pp=pp, groups=categories,
        identifiers=identifiers, test='chi2', reference=reference,
        label='experiment', plot_name='fraction', y_label='Fraction',
        title='Tethering and connectivity of proximal vesicles')

    # fraction of connected dependence on tethering
    util.stats_list(
        data=[near_teth_sv, near_non_teth_sv],
        dataNames=['tethered', 'non_tethered'], pp=pp, groups=categories,
        identifiers=identifiers, name='n_connection', bins=[0, 1, 100],
        join='join', test='chi2', reference=reference,
        y_label='Fraction of svs',
        title='Fraction of connected proximal svs dependence on tethering')

    # fraction of tethered dependence on connectivity
    util.stats_list(
        data=[near_conn_sv, near_non_conn_sv],
        dataNames=['connected', 'non_connected'], pp=pp, groups=categories,
        identifiers=identifiers, name='n_tether', bins=[0, 1, 100],
        join='join', test='chi2', reference=reference,
        y_label='Fraction of svs',
        title='Fraction of tethered proximal svs dependence on connectivity')

    # n connectors per proximal sv dependence on tethering
    util.stats_list(
        data=[near_teth_sv, near_non_teth_sv],
        dataNames=['tethered', 'non_tethered'], name='n_connection',
        join='join', pp=pp, groups=categories, identifiers=identifiers,
        test='kruskal', reference=reference, y_label='N connectors',
        title='N connectors per proximal sv dependence on tethering')

    # n connectors per connected proximal sv dependence on tethering
    util.stats_list(
        data=[near_teth_conn_sv, near_non_teth_conn_sv],
        dataNames=['tethered', 'non_tethered'], name='n_connection',
        join='join', pp=pp, groups=categories, identifiers=identifiers,
        test='kruskal', reference=reference, y_label='N connectors',
        title='N connectors per connected proximal sv dependence on tethering')

    # fraction of near svs that are tethered dependence on connectivity
    util.stats_list(
        data=[near_conn_sv, near_non_conn_sv],
        dataNames=['connected', 'non_connected'], name='n_tether', join='join',
        bins=[0,1,100], pp=pp, groups=categories, identifiers=identifiers,
        test='chi2', reference=reference,
        y_label='Fraction of vesicles that are tethered',
        title='Proximal vesicles tethering dependence on connectivity')

    # n tethers per sv dependence on connectivity
    util.stats_list(
        data=[near_conn_sv, near_non_conn_sv],
        dataNames=['connected', 'non_connected'], name='n_tether',
        join='join', pp=pp, groups=categories, identifiers=identifiers,
        test='t', reference=reference, y_label='N tethers',
        title='N tethers per proximal svs dependece on connectivity')


    ###########################################################
    #
    # RRP analysis
    #
    # Note: RRP is defined by number of tethers, see rrp_ntether_bins
    # (usually >2)

    # fraction of near svs that are in RRP
    util.stats(
        data=near_sv, name='n_tether', join='join', bins=[0,3,100],
        fraction=1, pp=pp, groups=categories, identifiers=identifiers,
        test='chi2', reference=reference, y_label='Fraction of all vesicles',
        title='Fraction of proximal vesicles that are in RRP')

    # fraction of near tethered svs that are in RRP
    util.stats(
        data=near_teth_sv, name='n_tether', join='join', bins=[0,3,100],
        fraction=1, pp=pp, groups=categories, identifiers=identifiers,
        test='chi2', reference=reference, y_label='Fraction of all vesicles',
        title='Fraction of proximal tethered vesicles that are in RRP')

    # n tether per tethered sv
    util.stats_list(
        data=[sv_rrp, sv_non_rrp], dataNames=['rrp', 'non_rrp'],
        name='n_tether', join='join', pp=pp,  groups=categories,
        identifiers=identifiers, test='kruskal', reference=reference,
        y_label='N tethers', title='N tethers per vesicle')

    # tether length
    util.stats_list(
        data=[tether_rrp, tether_non_rrp], dataNames=['rrp', 'non_rrp'],
        name='length_nm', join='join', groups=categories,
        identifiers=identifiers, test='t', reference=reference,
        y_label='Tether length [nm]',
        title='Tether length for proximal vesicles')

    # fraction of rrp and non-rrp svs
    count_histogram(
        data=[sv_rrp, sv_non_rrp], dataNames=['rrp', 'non_rrp'],
        pp=pp, groups=categories, identifiers=identifiers,
        test='chi2', reference=reference,
        label='experiment', plot_name='fraction', y_label='Fraction',
        title='Fraction of proximal vesicles by number of tethers')

    # fraction connected
    util.stats_list(
        data=[sv_rrp, sv_non_rrp], dataNames=['rrp', 'non_rrp'],
        name='n_connection', join='join', bins=[0,1,100],
        pp=pp, groups=categories, identifiers=identifiers, test='chi2',
        reference=reference, y_label='Fraction of vesicles that are connected',
        title='Proximal vesicles connectivity')

    # n connections per tethered sv
    util.stats_list(
        data=[sv_rrp, sv_non_rrp], dataNames=['rrp', 'non_rrp'],
        name='n_connection', join='join', pp=pp, groups=categories,
        identifiers=identifiers, test='kruskal', reference=reference,
        y_label='N connectors',
        title='N connectors per proximal sv')

    # fraction of short and long tethers
    count_histogram(
        data=[short_tether, long_tether],
        dataNames=['short_tether', 'long_tether'],
        pp=pp, groups=categories, identifiers=identifiers,
        test='chi2', reference=reference,
        label='experiment', plot_name='fraction', y_label='Fraction',
        title='Fraction of short and long tethers')

    # n short tethers per sv
    util.stats(
        data=near_sv, name='n_short_tether', join='join', pp=pp,
        groups=categories, identifiers=identifiers, test='kruskal',
        reference=reference,
        y_label='N tethers', title='N short tethers per proximal vesicle')

    # n long tethers per sv
    util.stats(
        data=near_sv, name='n_long_tether', join='join', pp=pp,
        groups=categories, identifiers=identifiers, test='kruskal',
        reference=reference,
        y_label='N tethers', title='N long tethers per proximal vesicle')

    # n short tethers per short tethered sv


    ###########################################################
    #
    # AZ analysis
    #
    # Note: AZ surface is defined as the layer 1

    # surface of the AZ
    util.stats(
        data=near_sv, name='az_surface_um', join='join', pp=pp,
        groups=categories, identifiers=identifiers, test='t',
        reference=reference,
        y_label=r'AZ area [${\mu}m^2$]', title='AZ surface area')

    # surface of the AZ, individual synapses
    util.stats(
        data=near_sv, name='az_surface_um', join=None, pp=pp,
        groups=categories, identifiers=identifiers, test='t',
        reference=reference,
        y_label=r'AZ area [${\mu}m^2$]', title='AZ surface area')

    # N proximal svs per synapse
    util.stats(
        data=near_sv, name='n_vesicle', join='join', pp=pp,
        groups=categories, identifiers=identifiers, test='t',
        reference=reference, y_label='Number of vesicles',
        title='Number of proximal vesicles per synapse')

    # N proximal vesicles per unit (1 um^2) AZ surface
    util.stats(
        data=near_sv, name='vesicle_per_area_um', join='join',
        pp=pp, groups=categories, identifiers=identifiers, test='t',
        reference=reference, y_label='Number of vesicles',
        title=r'Number of proximal vesicles per unit ($1 {\mu}m^2$) AZ area')

    # N tethers per synapse
    util.stats(
        data=tether, name='n_tether', join='join', pp=pp, groups=categories,
        identifiers=identifiers, test='t', reference=reference,
        y_label='Number of tethers',  title='Number of tethers per synapse')

    # N tethers per unit AZ area (um)
    util.stats(
        data=tether, name='tether_per_area_um', join='join',
        pp=pp, groups=categories, identifiers=identifiers, test='t',
        reference=reference, y_label='Number of tethers',
        title=r'Number of tethers per unit ($1 {\mu}m^2$) AZ area')

    # N tethered and non-tethered proximal svs per synapse
    util.stats_list(
        data=[near_non_teth_sv, near_teth_sv],
        dataNames=['non_tethered', 'tethered'],
        name='n_vesicle', join='join', pp=pp, groups=categories,
        identifiers=identifiers, test='t', reference=reference,
        y_label='Number of vesicles',
        title=('N tethered and non-tethered proximal vesicles per synapse'))

    # N tethered and non-tethered proximal svs per unit (1 um^2) AZ area
    util.stats_list(
        data=[near_non_teth_sv, near_teth_sv],
        dataNames=['non_tethered', 'tethered'],
        name='vesicle_per_area_um', join='join', pp=pp, groups=categories,
        identifiers=identifiers, test='t', reference=reference,
        y_label='Number of vesicles',
        title=('N tethered and non-tethered proximal vesicles per '
               + r'unit ($1 {\mu}m^2$) AZ area'))

    # N tethered /connected proximal svs per synapse
    util.stats_list(
        data=[near_teth_conn_sv, near_teth_non_conn_sv,
              near_non_teth_conn_sv, near_non_teth_non_conn_sv],
        dataNames=['t_c', 't_nc', 'nt_c', 'nt_nc'],
        name='n_vesicle', join='join', pp=pp, groups=categories,
        identifiers=identifiers, test='t', reference=reference,
        y_label='Number of vesicles',
        title='N tethered / connected proximal vesicles per synapse')

    # N tethered /connected proximal svs per unit (1 um^2) AZ area
    util.stats_list(
        data=[near_teth_conn_sv, near_teth_non_conn_sv,
              near_non_teth_conn_sv, near_non_teth_non_conn_sv],
        dataNames=['t_c', 't_nc', 'nt_c', 'nt_nc'], name='vesicle_per_area_um',
        join='join', pp=pp, groups=categories, identifiers=identifiers,
        test='t', reference=reference, y_label='Number of vesicles',
        title=('N tethered / connected proximal vesicles per '
               + r'unit ($1 {\mu}m^2$) AZ area'))

    # N rrp and non-rrp svs per synapse
    util.stats_list(
        data=[sv_rrp, sv_non_rrp], dataNames=['rrp', 'non_rrp'],
        name='n_vesicle', join='join', pp=pp, groups=categories,
        identifiers=identifiers, test='t', reference=reference,
        y_label='Number of vesicles',
        title=('N vesicles per synapse, dependence on RRP'))

    # N rrp and non-rrp proximal svs per unit (1 um^2) AZ area
    util.stats_list(
        data=[sv_rrp, sv_non_rrp], dataNames=['rrp', 'non_rrp'],
        name='vesicle_per_area_um', join='join', pp=pp, groups=categories,
        identifiers=identifiers, test='t', reference=reference,
        y_label='Number of vesicles',
        title=('N proximal vesicles per unit ($1 {\mu}m^2$) AZ area,'
               + ' dependence on RRP'))

    # correlation between number of proximal vesicles and the AZ surface
    util.correlation(
        xData=near_sv, xName='az_surface_um', yData=near_sv, yName='n_vesicle',
        test='r', pp=pp, groups=categories, identifiers=identifiers,
        join='join', x_label='AZ surface area [${\mu}m^2$]',
        y_label='N vesicles',
        title='Correlation between N proximal vesicles and AZ surface')

    # correlation between number of tethered vesicles and the AZ surface
    util.correlation(
        xData=near_sv, xName='az_surface_um', yData=near_teth_sv,
        yName='n_vesicle', test='r', pp=pp, groups=categories,
        identifiers=identifiers, join='join',
        x_label='AZ surface area [${\mu}m^2$]', y_label='N vesicles',
        title='Correlation between N tethered vesicles and AZ surface')


    ###########################################################
    #
    # Numbers of analyzed
    #

    # number of synapses
    [(categ, len(bulk_sv[categ].identifiers)) for categ in categories]

    # number of bulk svs (within 250 nm)
    [(categ, len(pyto.util.nested.flatten(bulk_sv[categ].ids)))
     for categ in categories]

    # number of proximal svs (within 45 nm)
    [(categ, len(pyto.util.nested.flatten(
        near_sv[categ].ids))) for categ in categories]

    # number of intermediate + distal svs (45 - 250 nm)
    [(categ, len(pyto.util.nested.flatten(inter_dist_sv[categ].ids)))
     for categ in categories]

    # number of tethers
    [(categ, len(pyto.util.nested.flatten(tether[categ].ids)))
     for categ in categories]

    # number of connections in bulk
    [(categ, len(pyto.util.nested.flatten(conn[categ].ids)))
     for categ in categories]


# run if standalone
if __name__ == '__main__':
    main()
