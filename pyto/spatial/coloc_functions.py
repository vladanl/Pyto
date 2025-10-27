"""
Contains (some of the) functions used for colocalization analysus
(module coloc_analysis) that act directly on individual colocalization
data tables.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""

__version__ = "$Revision$"


import os
import re
import functools
import pickle
from collections.abc import Iterable

import numpy as np
import scipy as sp
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt

import pyto

    
###########################################################
#
# Functions that manipulate colocalization and particle set names.
#

def get_layers(name, mode='columns_2021'):
    """
    Extracts layer names from the colocalization name (arg name)

    Depreciated: The same as get_names() except here default mode is 
    'columns_2021', left for backcompatibility.

    In mode 'munc13' arg name is simply split by '_'. Mode 'columns_2021'
    does the same but does not split 'ves_ap'. 

    Example valid for modes 'columns_2021' and 'munc13':
      get_layers(name='X_Y_Z') -> ['X', 'Y', 'Z']

    Arguments:
      - name: (str) colocalization name
      - mode: operation mode, currently implemented 'columns_2021' (defalut
      because of backcompatibility) and 'munc13'

    Returns: list of layer names
    """
    return get_names(name=name, mode=mode)

def get_names(name, mode='_'):
    """
    Extracts particle set names from the colocalization name (arg name).

    In mode 'munc13' arg name is simply split by '_'. Mode 'columns_2021'
    does the same but does not split 'ves_ap'. Otherwise, mode can be any 
    string, which is used to split arg name.

    Examples:
      get_names(name='X_Y_Z', mode='_') -> ['X', 'Y', 'Z']
      get_names(name='X_Y_Z', mode='columns_2021') -> ['X', 'Y', 'Z']
      get_names(name='X-1__Y-2__Z-3', mode='__') -> ['X-1', 'Y-2', 'Z-3']

    Arguments:
      - name: (str) colocalization name
      - mode: operation mode, currently implemented '_' (default) and 
      'columns_2021'

    Returns: layer names:
      - if one name is given, returns a list with 2-coloc names
      - if multiple names are given (arg name is a list of strings), returns
      a list of the same length as name, where elements of the returned list
      are lists as above that correspond to elements of arg name 
    """

    if (mode == 'columns_2021'):
        layers = name.split('_')
        if 'ves' in layers:
            ves_ind = layers.index('ves')
            if layers[ves_ind+1] == 'ap':
                layers[ves_ind] = 'ves_ap'
                layers.remove('ap')

    else:
        layers = name.split(mode)
        
    #else:
    #    raise ValueError(
    #        f"Mode {mode} was not undertood, implemented values are "
    #        + "'columns_2021' (default) and '_'.")

    return layers

def make_name(names, suffix, mode='_'):
    """Makes colocalization name.

    For example:
      make_name(names=['X', 'Y', 'Z'], suffix='data') -> 'X_Y_Z_data'
      make_name(names=['X', 'Y', 'Z'], suffix=None) -> 'X_Y_Z'

    Arguments:
      - names: (list) names of individual particle sets
      - suffix: suffix added at the end of the colocalization name

    Returns colocalization name
    """

    if suffix is not None:
        result = mode.join(names + [suffix])
    else:
        result = mode.join(names)
    return result

def make_full_coloc_names(names, suffix, mode='_'):
    """Makes full colocalization names:

    For example:
      make_full_coloc_names(['setX', 'setY', 'setZ'], suffix='data')
      -> 'setX_setY_setZ_data', 'setX_setY_data', 'setX_setZ_data']
      make_full_coloc_names(['setX', 'setY', 'setZ'], suffix=None)
      -> 'setX_setY_setZ', 'setX_setY', 'setX_setZ']

    Arguments:
      - names: (list) point pattern (particle set) names
      - suffix: suffix, it's preceeded by '_' unles None
    """

    name_3 = make_name(names=names, suffix=suffix)
    if len(names) > 2:
        if suffix is not None:
            names_2 = [mode.join([names[0], nam, suffix]) for nam in names[1:]]
        else:
            names_2 = [mode.join([names[0], nam]) for nam in names[1:]]
    else:
        names_2 = []
        
    return [name_3] + names_2

def get_2_names(name, order=None, mode='_', by_order=False):
    """Make 2-colocalization names from a 3- or higher colocalization.

    If arg order is None, the standard 2-colocalizations are made, for example:

      get_2_names(name='X_Y_Z') -> ['X_Y', 'X_Z']

    Setting arg order allows making other pairs, for example:

      get_2_names(name='X_Y_Z', order=((2, 0), (2, 1)) -> ['Z_Y', 'Z_Y'] 

    If multiple 3-colocalization names are given, the order of elemnts of 
    the returned value depends on arg by_order:

      get_2_names(name=['X_Y_Z', '1_2_3'], by_order=False) 
          -> [['X_Y', 'X_Z'], ['1_2', '1_3']]
      get_2_names(name=['X_Y_Z', '1_2_3'], by_order=True) 
          -> [['X_Y', '1_2'], ['X_Z', '1_3']]

    Arguments:
      - name: one or more colocalization names, where each name is composed of
      3 or more particle set names
      - order: (list of lists of ints) defins how the set names are combined
      - mode: (str) separator between particle set names, such as '_' or '__'
      - by_order: (bool, default False) determined the order of returned 
      elements when more than one colocalization name is given
    """

    # figure out name
    if isinstance(name, str):
        name = [name]
        one_name = True
    else:
        one_name = (True if len(name) == 1 else False)
        #one_name = False
        
    results = []
    for na in name:
        current = []
        layers = get_names(na, mode=mode)
        if len(layers) < 3:
            raise ValueError(
                "This function requires a 3- or higher colocalization")
        if order is None:
            order = [(0, ind) for ind in range(1, len(layers))]
        for comb in order:
            pair = [layers[comb[0]], layers[comb[1]]]
            current.append(make_name(names=pair, suffix=None, mode=mode))
        results.append(current)
        
    if one_name:
        results = results[0]
    elif by_order:
        results = np.asarray(results).transpose()
        
    return results


###########################################################
#
# Functions dealing with file names and paths that need to give different
# results for different projects.
#
# For a new project, another mode can be added to these methods, or
# they can be overriden by inheritance
#

#@staticmethod
def get_raw_coloc_bases(
        name, distances, mode, n_sim, n_coloc='', column_radius_factor=2):
    """
    Makes a part of the full path to raw colocalization data and returns
    a list of these paths, one element for each specified distance.

    The paths depend on arg mode and other arguments in the following way:

    mode='method-1':
        3_name/aln_distance/3_name_sim200
        (e.g. 3_set0_set1_set2/aln_15/3_set0_set1_set2_sim200)

    mode='method-1_cr-same':
        3_name/aln_distance_cr_distance/3_name_sim200
        (e.g. 3_set0_set1_set2/aln_15_cr_30/3_set0_set1_set2_sim200)

    mode='method-1_cr-double':
        n_name/aln_distance_cr-coldist/n_name_sim200_wspace.pkl
        (e.g. 3_set0_set1_set2/aln_15_cr-30/3_set0_set1_set2_sim200_wspace.pkl)

    mode='method-1_cr-double_v2':
        n_name/dist-distance_cr-coldist/n_name_sim-n_sim_wspace.pkl
        (e.g. 2_set1_set2/dist-10_cr-20/3_set1_set2_sim-200_wspace.pkl,
        3set0_set1_set2/dist-20_cr-40/3_set0_set1_set2_sim-200_wspace.pkl)

    mode='simple_cr-double' or mode='munc13':
        n_name/dist-distance_cr-coldist/n_name_sim-n_sim_wspace.pkl
        (e.g. 2_set2_set1/dist-15_cr-30/3_set2_set1_sim-200_wspace.pkl,
        3_set0_set1_set2/dist-15_cr-30/3_set0_set1-set2_sim-200_wspace.pkl)

    where:
      - n: arg name, that is the number of colocalization layers (2 or 3)
      - name: specifies the layers that are aligned (e.g. 
      'pre_tether_pst', or 'pre_tether')
      - n_coloc_name examples: '3_pre_tether_pst', '2_pre_tether'
      - distance: distance (e.g. 15), has to be an element of arg distances 
      - coldist: column_radius_factor * dist (e.g. 30).
      - n-sim: arg n_sim, that is number of simulations (e.g. 200)

    Arguments:
      - name: colocalization name (layers that are colocalized), such as 
      'setx_sety_setz'
      - distances: list of distances in nm
      - mode: defines the way the path is made (see get_raw_coloc_bases() doc)
      - n_sim: number of simulations
      - n_coloc: (str) number of sets that are colocalized
      - column_radius_factor: column redius is calculated by multiplying 
      the actual distance (an element of arg distances) by this number
    """

    if mode == 'method-1':
        bases = ['3_{}/aln_{}/3_{}_sim200'.format(
            name, dist, name)
            for dist in distances]   
    elif mode == 'method-1_cr-same':
        bases = ['3_{}/aln_{}_cr{}/3_{}_sim200'.format(
            name, dist, dist, name)
            for dist in distances]     
    elif mode == 'method-1_cr-double':
        bases = ['{}_{}/aln_{}_cr{}/{}_{}_sim200'.format(
            n_coloc, name, dist, column_radius_factor*dist, n_coloc, name)
            for dist in distances]     
    elif mode == 'method-1_cr-double_v2':
        bases = ['{}_{}/dist-{}_cr-{}/{}_{}_sim-{}'.format(
            n_coloc, name, dist, column_radius_factor*dist, n_coloc,
            name, n_sim)
            for dist in distances]     
    elif (mode == 'simple_cr-double') or (mode == 'munc13'):
        bases = ['{}_{}/dist-{}_cr-{}/{}_{}_sim-{}'.format(
            n_coloc, name, dist, column_radius_factor*dist, n_coloc,
            name, n_sim)
            for dist in distances]     
    else:
        raise ValueError(
            "Argument mode {} was not understood".format(mode))

    return bases

#@staticmethod
def get_tomo_id(path, mode):
    """
    Extracts tomo id from tomo file name. 

    Tomo id is extracted as follows:

    mode='simple_cr-double':
      path = 'dir1/dir2/base_name.ext'
      tomo id = base_name

    mode='munc13':
      path = 'dir1/dir2/foo_syn_tomoid_bin-foo.ext'
      tomo id = tomoid

    mode='simple_examples':
      path = 'dir1/dir2/foo_tomo-tomoid_seg-foo.ext'
      tomo id = tomoid

    all other cases ('method-1', 'method-1_cr-same', 'method-1_cr-double'
    and 'method-1_cr-double_v2'):
      path = 'dir1/dir2/foo_tomoid1_tomoid2_foo.ext'
      tomo id = tomoid1_tomoid2

    Arguments:
      - path: tomo path
      - mode: defines how to extract tomo id   
    """

    if ((mode == 'method-1_cr-double') or (mode == 'method-1') 
        or (mode == 'method-1_cr-same') 
        or (mode == 'method-1_cr-double_v2')):
        split_name = os.path.split(path)[1].split('_')
        tomo_id = split_name[1] + '_' + split_name[2]

    elif mode == 'simple_cr-double':
        name = os.path.split(path)[1]
        tomo_id = name.split('.')[0]

    elif mode == 'munc13':
        name = os.path.split(path)[1]
        id_like = re.split('syn_|_bin', name)[1]
        pieces = re.split('_|-', id_like)
        tomo_id = functools.reduce(lambda a, b: a + '_' + b, pieces)

    elif mode == 'simple_examples':
        name = os.path.split(path)[1]
        tomo_id = re.split('tomo-|_seg', name)[1]

    else:
        raise ValueError(f"Mode {mode} was not understood") 
        

    return tomo_id   

    
###########################################################
#
# Functions needed to read and preprocess data
#

def set_read_parameters(name, distances, in_path, mode, n_sim=None):
    """
    Figures out parameters needed to access and interpret data given in 
    workspaces.

    All logic specific to the form of workspaces (pickles) should be here.

    If arg mode is None, returns [] for workspaces.

    Arguments:
      - name: name of the colocalization
      - distances: list of distances
      - in_path: root path of the workspaces (pickles)
      - mode: workspace path mode (see get_raw_coloc_bases() doc)
      - n_sim: n simulations

    Returns:
      - workspaces: list of absolute paths to workspaces (colocalization 
      raw data)
      - columns: dictionary where keys are indices of workspace object and 
      values are the names of the corresponding columns
      - add columns: (list) names of columns where individual tomo values
      should be added to get the value for all tomos together
      - array_columns: (list) names of columns where each element is 
      an array (results of random simulations, for example)
    """

    # parse name
    layers = get_layers(name=name)
    n_layers = len(layers)

    # set pickle paths
    if mode is not None:
        wspaces = [
            os.path.join(in_path, f'{bas}_wspace.pkl')
            for bas in get_raw_coloc_bases(
                    name=name, distances=distances, mode=mode, n_sim=n_sim,
                    n_coloc=n_layers)]
    else:
        wspaces = []

    # set initial meta data 
    # Note: The order given here is used for column tables 
    columns = {
        39: 'n_subcol', 
        21: 'tomos_npc_l1', 22: 'tomos_npc_l2', 23: 'tomos_npc_l3',
        17: 'tomos_np_l1', 18: 'tomos_np_l2', 19: 'tomos_np_l3',
        40: 'n_subcol_random_all', 41: 'n_subcol_random_alt_all',
        25: 'tomos_npc_l1_sim', 26: 'tomos_npc_l2_sim',
        27:'tomos_npc_l3_sim',
        13: 'n_sv', 15: 'n_teth_cent', 28: 'area', 
        14: 'volume', 0: 'n_col', 29: 'area_col'}
    array_columns = ['n_subcol_random_all', 'n_subcol_random_alt_all']

    # remove n_sv and n_teth_cent because they are not expected in
    # munc13 mode
    if (mode is not None) and (mode == 'munc13'):
        columns.pop(13)
        columns.pop(15)

    # rename meta data
    try:
        for index in range(n_layers):
        #for index in [0,1,2]:
            columns[17+index] = 'n_{}_total'.format(layers[0+index])
            columns[21+index] = 'n_{}_subcol'.format(layers[0+index])
            columns[25+index] = 'n_{}_subcol_random'.format(layers[0+index])
    except IndexError: pass

    # columns where the data for all tomos is the sum of the individuals
    add_columns = (
        ['n_subcol']
        + [columns[21+index] for index in list(range(len(layers)))]
        + [columns[17+index] for index in list(range(len(layers)))])
    if (mode is not None) and (mode == 'munc13'):
        add_columns += ['volume', 'n_col', 'area_col', 'area']
    else:
        add_columns += [
            'n_sv', 'volume', 'n_teth_cent','n_col', 'area_col', 'area']

    # add number of particles in subcolumns to array columns
    # Warning: Perhaps breaks backcompatibility prior to munc13 mode (9.2022)
    array_columns += [columns[25+index] for index in list(range(n_layers))]
    
    return wspaces, columns, add_columns, array_columns

def read_data(pkl_path, columns, mode):
    """
    Reads data from a specified workspace (colocalization raw data) and 
    puts it in a table.

    In addition to the columns specified in arg columns, tomo_id is 
    extracted and added to the data table.

    Arg mode defines how to extract tomo id (see get_raw_coloc_bases() doc).
    Also, if mode is 'munc13', a default index is used (ints), while in all
    other cases tomo id (column id) is used as index.

    Arguments:
      - pkl_path: path to the workspace
      - columns: dictionary where keys are indices of workspace object and 
      values are the names of the corresponding columns
      - mode: defines how to extract tomo id (see get_raw_coloc_bases() doc)

    Returns:
      - data: (pandas.DataFrame) data table
    """

    # read pickle
    with open(pkl_path, 'rb') as pkl_fd:
        pkl_data = pickle.load(pkl_fd, encoding='latin1')

    for data_ind, c_name in columns.items():

        # make DataFrame with current data and make synapse id 
        #id_data = [
        #    [self.get_tomo_id(t_name, mode=mode), value] 
        #    for t_name, value in pkl_data[data_ind].items()]

        tomo_names = list(pkl_data[data_ind].keys())
        ids = [get_tomo_id(t_name, mode=mode) for t_name in tomo_names]
        values = [pkl_data[data_ind][t_name] for t_name in tomo_names]
        local_data = pd.DataFrame({'id' : ids, c_name: values})

        # id is not unique, so it shouldn't be index (not unique? 9.2022)
        if (mode is not None) and mode.startswith('method-1'):
            # backcompatibility with the trans-synaptic projet (finished 2021)
            local_data = local_data.set_index('id')

            # add current to total data 
            try:
                data = pd.concat([data, local_data], axis=1, sort=False)
            except NameError:
                data = local_data
                
        elif mode == 'munc13':
            try:
                data = pd.merge(data, local_data, on='id')
            except NameError:
                data = local_data
            
        else:
            raise ValueError(
                "Arg mode can be None, 'munc13' or start with 'method-1'")
        
    return data

def select_rows(data, ids=None, distance=None):
    """
    Select rows from the specified table based on columns given by args ids 
    and distance. Meant for chosing selecting rows corresponding to tomos
    (specified by arg ids) from a table containing tomo-specific rows. 

    If ids or distance is None, the corresponding selection is omitted. 

    If arg ids is not None, there are two possibilities how to chose the 
    column that contains ids:
      - table index has no name, in which case column id is used
      - column id is index, in which case the index column is used

    If arg distance is specified, the table has to contain column labeled
    distance.

    Arguments:
      - data: table, has to contain column labeled 'id', which can but does 
      not have to be indexed
      - ids: list of ids
      - distance: list or a single distance
    """

    # select by specified tomos
    if ids is not None:
        index_name = data.index.name
        if index_name is None:
            cond = data['id'].isin(ids)
        else:
            cond = data.index.isin(ids)
        data = data[cond].copy()

    # select specified distances
    if distance is not None:
        if isinstance(distance, (list, tuple, np.ndarray)):
            data = data[data['distance'].isin(distance)].copy()
        else:
            data = data[data['distance'] == distance].copy()
    else:
        data = data.copy()

    return data

def get_aggregate_columns(columns):
    """Returns columns needed for aggregate() in ColocLite

    Array columns are those that end with 'all'.

    Add columns are 'n_col', 'size_col', 'size_region', and those ending 
    with 'subcol' or 'total'.

    Argument:
      - columns: (list) columns from wich the add and array columns are 
      determined

    Returns (add_columns, array_columns):
      - add columns: (list) names of columns where individual tomo values
      should be added to get the value for all tomos together
      - array_columns: (list) names of columns where each element is 
      an array (results of random simulations, for example)
    """
    array_columns = [col for col in columns if col.endswith('all')]
    add_columns = [
        col for col in columns
        if (col.endswith('subcol') or col.endswith('total') or (col == 'n_col')
            or (col == 'size_col') or (col == 'size_region'))]
    return add_columns, array_columns

def aggregate(
        data, distance, add_columns, array_columns, p_values=True,
        p_func=np.greater, random_stats=True,
        random_suff=['random', 'random_alt', 'random_combined'],
        p_suff=['normal', 'other', 'combined']):
    """
    Calculates data for all tomograms (synapses) together by combining 
    data for individual tomograms.

    The values in columns specified by arg add_columns are added, while 
    the values in columns specified by arg array_columns (expect to have
    multiple values, like lists) are element-wise added.

    The data is combined for the specified distance(s).

    For calculations of p values and random data statistics, it is 
    expected that two types of simulations were run. 

    If arg p_values is True, p-values are calculated for each distance, for
    all tomograms together in the following way:
      - Column 'p_subcol_normal' of the resulting table is calculated from
      n_subcol_random_all values from arg data
      - Column 'p_subcol_other' of the resulting table is calculated from
      n_subcol_random_alt_all values from arg data
      - Column 'p_subcol_combined' of the resulting table is calculated 
      from both of the above together
    In each case, p-value is the number of simulations for which the actual
    number of subcolumns (column n_subcol of data, for all tomograms 
    together and each distance separately) is greater than the simulation
    value.

    If arg random_stats is True, basic stats (mean and std) are calculated 
    for random simulations data. which adds the following columns:
      - n_subcol_random_{mean, std): Stats for the first random type
      - n_subcol_random_alt_{mean, std): Stats for the second random type
      - n_subcol_random_combined_{mean, std): Stats where the two random 
      simulation types are taken together

    Note: Column names mentioned in the above two paragraphs are for the 
    default values of args random_suff and p_suff. Consequently, they are
    changed if the corresponding arguments are set to other values.

    Arguments:
      - data: (pandas.DataFrame) table containing data for each tomogram
      separately
      - distance: single value or a list of distances for which the data 
      is calculated
      - add columns: (list) names of columns where individual tomo values
      should be added to get the value for all tomos together (as returned 
      by set_read_parameters())
      - array_columns: (list) names of columns where each element is 
      an array (results of random simulations, for example) (as returned 
      by set_read_parameters())
      - p_values: flag indication if p_values should be calculated
      - random_stats: flag indicating whether basic stats are calculated
      for random simulations
      - random_suff: suffixes for random data (default 
      ['random', 'random_alt, random_combined']
      - p_suff: suffixes for p_values (default ['normal', 'other', 'combined'])

    Returns: (pandas.DataFrame) data table containg data for all 
    synapses together
    """

    # use all existing distances if None
    if distance is None:
        distance = data['distance'].unique()
        distance.sort()
        
    # multiple distances
    if isinstance(distance, Iterable):
        for dist in distance:
            dist_row = aggregate(
                data=data, distance=dist, add_columns=add_columns,
                array_columns=array_columns, p_values=p_values, p_func=p_func,
                random_stats=random_stats,
                random_suff=random_suff, p_suff=p_suff)
            try:
                result = pd.concat([result, dist_row], ignore_index=True)
            except NameError:
                result = dist_row
        return result
            
    # keep only the specified distance
    data_tomo = data[data['distance'] == distance].copy()
    
    # sum data: simple summation
    for column in add_columns:
        if column in array_columns + ['distance']: continue

        value = data_tomo[column].sum()
        try:
            result[column] = value
        except (NameError, UnboundLocalError):
            result = pd.DataFrame(
                {'distance' : distance, column: value}, index=[-1])
            #result = result.set_index('distance')            

    # array data: element-wise summation 
    for column in array_columns:
        data_np = data_tomo[column].map(lambda x: np.asarray(x)).to_numpy()
        value = np.vstack(data_np).sum(axis=0).tolist()
        result[column] = [value]  # works because result has 1 row only
            
    # calculate p-values
    if p_values:

        # remove rows where experimental data nan
        if len(random_suff) > 1:
            random = data_tomo[
                ['n_subcol', f'n_subcol_{random_suff[0]}_all',
                 f'n_subcol_{random_suff[1]}_all']]
        else:
             random = data_tomo[
                ['n_subcol', f'n_subcol_{random_suff[0]}_all']]
        bad_tomos = [
            index for index, row in random.iterrows() 
            if np.isnan(row['n_subcol'])]
        random = random.drop(bad_tomos)

        # experimental n subcolumns
        exp = random['n_subcol'].sum()

        # calculate fractions standard
        random_column = f'n_subcol_{random_suff[0]}_all'
        random_total = np.vstack(
            np.array(random[random_column])).sum(axis=0)
        n_good = p_func(exp, random_total).sum()
        result[f'p_subcol_{p_suff[0]}'] = n_good / random_total.shape[0]

        # calculate fractions other
        if len(random_suff) > 1:
            random_column = f'n_subcol_{random_suff[1]}_all'
            random_alt_total = np.vstack(
                np.array(random[random_column])).sum(axis=0)
            n_good_alt = p_func(exp, random_alt_total).sum()
            result[f'p_subcol_{p_suff[1]}'] = (
                n_good_alt / random_alt_total.shape[0])

            # calculate fractions combined
            result[f'p_subcol_{p_suff[2]}'] = (
                n_good + n_good_alt) / float(
                    random_total.shape[0] + random_alt_total.shape[0])

    # calculate statistics on random
    if random_stats:
 
        result, _ = get_random_stats(
            data=result, data_syn=data_tomo,
            column=f'n_subcol_{random_suff[0]}_all',
            out_column=f'n_subcol_{random_suff[0]}', combine=False)
        if len(random_suff) > 1:
            result, _ = get_random_stats(
                data=result, data_syn=data_tomo,
                column=f'n_subcol_{random_suff[1]}_all', 
                out_column=f'n_subcol_{random_suff[1]}', combine=False)
            result, _ = get_random_stats(
                data=result, data_syn=data_tomo,
                column=[f'n_subcol_{random_suff[0]}_all',
                        f'n_subcol_{random_suff[1]}_all'],
                out_column=f'n_subcol_{random_suff[2]}', combine=True)

    return result


####################################################################
#
# Functions that calculate properties derived from results
#

def calculate_density(data, density='density', number='n_subcol'):
    """   
    Calculates the ratio of the specified number of colocalizations (or 
    particles) per single subcolumn area.

    Specifically, for default arguments, calculates the number of 
    subcolumns per single colocalization area and saves it as column
    'density'.

    Sets calculated values as a new column in data.

    Arguments: 
      - data: (pd.DataFrame) data table
      - density: name of the column containing the calculated density ratio
      - number: name of the colummn containing the number used for the
      numerator
    """
    data[density] = data[number] / (np.pi * data['distance']**2)

    
####################################################################
#
# Functions that calculate properties of random simulations
#

def get_random_stats(
        data, data_syn, column, out_column, combine=True):
    """
    Calculates basic statistics (mean and std) for random simulations
    for given tables and given columns.

    Modifies arg data but not data_syn.

    Arguments:
      - data: combined data for all tomograms (rows defined by distance)
      - data_syn: data for individual tomograms
      - column: list of column names of random simulation data
      - out_column: root name of the column where the results are added,
      - combine: flag indicating if random simulation types need to be combined

    Returns (data, data_syn): modified combined and individual tomo data
    tables, respectively
    """

    for dist in data.distance:

        # extract individual synapse data for current distance
        data_syn_local = data_syn[data_syn.distance==dist].copy()

        # calculate stats
        data, data_syn_local = get_random_stats_single(
            data=data, data_syn=data_syn_local, 
            column=column, out_column=out_column, combine=combine)

        # append to individual synapse table
        try:
            data_syn_res = pd.concat(
                [data_syn_res, data_syn_local], ignore_index=False)
        except NameError:
            data_syn_res = data_syn_local

    # sort by tomo name and distance
    data = data.sort_values(by='distance')
    data_syn_res = data_syn_res.sort_values(by=['id', 'distance'])

    return data, data_syn_res      

def get_random_stats_single(
        data, data_syn, column, out_column, distance=None, combine=True):
    """
    Calculate basic stats (mean and std) for random simulations. The stats 
    are calculated for both individual and all synapses taken together.

    If arg column is a string, the stats are calculated for that column
    of table data_syn and the results are saved in columns whose name 
    starts with arg out_column and has suffixes '_mean' and '_std'. Arg
    combine has to be False.

    If arg combine is True, random simulations given in multiple columns 
    of table data_syn, specified by elements of list column, are combined
    and saved in columns whose name starts with arg out_column and has 
    suffixes '_mean' and '_std'.

    Modifies args data and data_syn.

    Arguments:
      - data: (pandas.DataFrame) table containing experimental and random 
      simulation values for all synapses together, for a single distance
      (so one row only)
      - data_syn: (pandas.DataFrame) table containing experimental and 
      random 
      simulation values for each synapse separately, for a single distance
      - column: list of column names of random simulation data
      - out_column: root name of the column where the results are added,
      '_mean' is appended for the mean of the data and '_std' to std.
      - distance: Should be None (other values are experimental)
      - combine: Flag indicating if the data from both random simulations
      should be combined to calculate the combined stats

    Return: 
      - data: modfied data table for all synapses together
      - data_syn: modified data table for each synapse separately
    """

    # check args
    if combine:
        if not isinstance(column, (list, tuple)):
            raise ValueError(
                "When argument combine is True, argument column has "
                + "to be a list or tuple")
    else:
        if isinstance(column, (list, tuple)):
            raise ValueError(
                "When argument combine is False, argument column cannot "
                + "be a list or tuple")
        else:
            column = [column]

    # figure out distance
    if distance is not None:
        # should not happen
        #data_syn = data_syn[data_syn.distance==distance]
        pass
    else:
        if len(data_syn.distance) > 0:
            dist_0 = data_syn.distance.iloc[0]
            if (np.array(data_syn.distance) == dist_0).all():
                distance = dist_0
            else:
                raise ValueError(
                    "Table specified by argument data_syn has to "
                    + "contain data for one distance only. ")
        else:
            raise ValueError(
                "Table specified by data_syn argumnet does not have "
                + "data for distance {}".format(distance))

    # get random data
    for col in column:
        current = np.array(
            [data_syn.loc[index, col] for index in data_syn.index])
        try:
            total = np.concatenate([total, current], axis=1)
        except NameError:
            total = current

    # add basic stats to the individual tomo table
    means = total.mean(axis=1)
    #data_syn = data_syn.copy()
    data_syn[out_column+'_mean'] = means
    data_syn[out_column+'_std'] = total.std(axis=1)

    # add basic stats to the all synapses table
    distance_index = data[data.distance==distance].index[0]
    #data = data.copy()
    data.at[distance_index, out_column+'_mean'] = means.sum()
    data.at[distance_index, out_column+'_std'] = total.sum(axis=0).std()

    return data, data_syn

def get_fraction_syn(data):
    """
    Renamed get_fraction_random(), left for backcompatibility
    """
    result = get_fraction_syn(data=data)
    return result
    
def get_fraction_random(
        data, p_func=np.greater, random_suff=['random', 'random_alt'],
        p_suff=['normal', 'other', 'combined']):
    """
    Calculates p-value based on the comparison of experimental and 
    data from multiple simulations.

    The p-values are calculated based on the arg p_func (such as np.greater, 
    np.less_equal, ...) for each row of the specified data table. 

    For example, if arg p_func is np.greater, the calculated p-value would 
    equal the fraction of random simulation where the experimental 
    number of subcolumns (column n_subcol) is greater than the simulated
    number of subcolumns. Therefore, this p-value is 1-p where p is
    the probability to reject the null hypothesis.
    
    Data for two separate types of simulations need to be present, where 
    each simulation type comprises multiple random simulations. 

    The specified data table has to contain the following columns:
      - n_subcol: experimental data (ove value)
      - n_subcol_+random_suff[0]+_all: data from one simulation type (each 
      element has to be a list containing values for different simulations
      - n_subcol_+random_suff[1]+_all: like previous but for the other
      simulation type
    Consequently, this function is meant to be applied to individual tomos
    tables because the combined tomos tables usually do not have these
    columns.

    The following columns are added to the resulting table:
      - p_subcol+p_suff[0]: p-values for the first simulation type
      - p_subcol+p_suff[1]: p-values for the second simulation type
      - p_subcol+p_suff[2]: p-values where the two simylation types are 
      combined

    Arguments:
      - data: (pandas.DataFrame) table containing experimental and random 
      simulation values for each tomogram separately
      - p_func: function that takes two array arguments like np.greater, 
      np.greater-equal, ...  and returns a boolean array.
      - random_suff: suffixes for random data (default ['random', 'random_alt']
      - p_suff: suffixes for p_values (default ['normal', 'other', 'combined'])

     Returns:
      - data: (pandas.DataFrame) table where the fractions are added as 
      additional columns for each tomogram separately
   """

    # standard random
    random, n_good, n_random = get_fraction_single(
        data=data, random_column=f'n_subcol_{random_suff[0]}_all',
        exp_column='n_subcol', fraction_column=f'p_subcol_{p_suff[0]}',
        p_func=p_func)

    # alternative random
    if len(random_suff) > 1:
        random_alt, n_good_alt, n_random_alt = get_fraction_single(
            data=data, random_column=f'n_subcol_{random_suff[1]}_all',
            exp_column='n_subcol', fraction_column=f'p_subcol_{p_suff[1]}',
            p_func=p_func)

    # add fractions to data
    data[f'p_subcol_{p_suff[0]}'] = random[f'p_subcol_{p_suff[0]}'] 
    if len(random_suff) > 1:
        data[f'p_subcol_{p_suff[1]}'] = random_alt[f'p_subcol_{p_suff[1]}'] 

    # combined random       
    if len(random_suff) > 1:
        fraction_good_combined = (
            (n_good + n_good_alt) / float(n_random + n_random_alt))
        random[f'p_subcol_{p_suff[2]}'] = fraction_good_combined
        data[f'p_subcol_{p_suff[2]}'] = random[f'p_subcol_{p_suff[2]}']

    return data

def get_fraction_syn_single(*args, **kwargs):
    """
    Renamed to get_fraction_single(), left for backcompatibility
    """
    result = get_fraction_single(*args, **kwargs)
    return result
    
def get_fraction_single(
        data, random_column, exp_column, fraction_column, p_func=np.greater):
    """
    Calculates the fraction of random simulations for which the experimental
    number of subcolumns (defined by arg exp column) is greater/less ... 
    (specified by arg p_func) than the number of subcolumns obtained by 
    the random simulations.

    The function specified by arg p_func takes experimental number as the 
    first and the simulated as the second argument, for example np.greater,
    np.less_equal, ... .
 
    Acts on each row of the data table separately. It is intended to be 
    applied to data tables where the values for different tomograms are
    specified on separate rows, but it can be also used for data tables
    where tomograms are combined.

    Arguments:
      - data: (pandas.DataFrame) table containing experimental and random 
      simulation values for each synapse separately
      - random_column: name of the column cntaining random simulations data
      - exp_column: name of the column cntaining experimental data
      - fraction_column: name of the column containing calculated fractions
      - p_func: function that takes two array arguments like np.greater, 
      np.greater-equal, ...  and returns a boolean array.

    Returns:
      - data: (pandas.DataFrame) table where the fractions are added as 
      additional columns for each synapse separately
      - n_good: Number of random simulations that yielded a smaller number 
      of subcolumns than the experimental data for each synapse separately
       - n_random: number of random simulations
    """

    # remove rows where exp data nan
    random = data[[exp_column, random_column]]
    bad_tomos = [
        index for index, row in random.iterrows() 
        if np.isnan(row[exp_column])]
    random = random.drop(bad_tomos)

    # calculate fraction
    random_ar = np.vstack(np.array(random[random_column]))
    n_random = random_ar.shape[1]
    actual = np.array(random[exp_column])[:,np.newaxis]
    n_good = p_func(actual, random_ar).sum(axis=1)
    random[fraction_column] = n_good / float(n_random)

    return random, n_good, n_random    


####################################################################
#
# Plotting and related functions
#

def make_nice_label(label, sets):
    """Makes nice looking labels for colocalization names. 

    Important: Moved to coloc_plot.py. Left here for backcompatibility.
    
    Arg label is split in pieces separated by '_' and each piece is 
    substituted by the corresponding value of arg sets.

    Arguments:
      - label: (str) typically a colocalization name (e.g. 'pre_tether_post')
      - sets: (dict) substitution rules in the form of {'old_piece:
    'nice_piece'}

    Returns nice looking label
    """

    print(
        "Depreciation warning: This function has been moved to coloc_plot.py.") 
    from . import coloc_plot
    return coloc_plot.make_nice_label(label=label, sets=sets)

def table_generator(coloc=None, name=None, groups=None, single=False):
    """Generator that makes an iterator over colocalization results.

    Important: Moved to coloc_plot.py. Left here for backcompatibility.
    
    Each element returned by the iterator contains a label and a 
    coloclization table (pandas.DataFrame) where rows correspond to 
    colocalization distances.

    The following cases are implemented:

    1) One or more colocalization names, all tomos together

    Arg coloc specified, name contains one or more colocalization 
    names, single=False: The returned iterator contains a table 
    for each coloclization specified (arg name), for all tomos together.

    2) One colocalization name, each tomogram separately

    Arg coloc specified, name is one colocalization name, single=True:
    The returned iterator contains a table for each tomo (contained in
    colocalization data) separately.

    3) One colocalization name, each group separately, DataFrame version

    Arg coloc is None, arg groups is dict of labels (keys) and 
    colocalization data as DataFrames (values). Args name and single 
    are ignored: The returned iterator contains the specified 
    colocalization tables (values of arg groups).

    4) One colocalization name, each group separately, ColocAnalysis

    Arg coloc is None, arg groups is dict of group names (keys) and 
    colocalization data as ColocAnalysis objects (values). Arg name is 
    a colocalization name, while arg single is ignored: The returned 
    iterator contains the colocalization tables for the specified 
    colocalization names, for each of the colocalizations given in arg 
    groups. The specifeid colocalization name has to be present in all 
    colocalization objects. 

    Aguments:
      - coloc: (ColocAnalysis) colocalization object
      - name: one or more colocalization names
      - groups: dictionary where keys are group names and values are the
      corresponding colocalization tables
      - single: Flag indication if individual tomo data is returned, used 
      only if arg coloc is specified

    Returns iterator that in each iteration returns a label and the 
    corresponding colocalization data.
    """

    print(
        "Depreciation warning: This function has been moved to coloc_plot.py.") 
    from . import coloc_plot
    return coloc_plot.table_generator(
        coloc=coloc, name=name, groups=groups, single=single)

def plot_p(
        coloc=None, name=None, groups=None, single=False,
        y_var='p_subcol_combined', tomos=None, sets={}, pp=None, ax=None):
    """Plots p-values for one colocalization. 

    Important: Moved to coloc_plot.py. Left here for backcompatibility.
    
    """
    print(
        "Depreciation warning: This function has been moved to coloc_plot.py.") 
    from . import coloc_plot
    return coloc_plot.plot_p(
        coloc=coloc, name=name, groups=groups, single=single,
        y_var=y_var, tomos=tomos, sets=sets, pp=pp, ax=ax)
        
def plot_data(
        coloc=None, name=None, groups=None, single=False,
        y_var='n_subcol', tomos=None, simulated={}, normalize=None,
        sets={}, pp=None, ax=None):
    """Plots data for one colocalization. 

    Important: Moved to coloc_plot.py. Left here for backcompatibility.
    
    Args coloc, name, groups and single are used to select colocalization
    data, as explained in table_generator() doc. 

    If arg y_vars contain multiple values, they should be either p-values
    related ('p_subcol_normal', 'p_subcol_other' and 'p_subcol_combined'), 
    or other variables.

    Produces nice looking plots for default values of colocalization 
    parameters related to simulation suffixes, both for the standard 
    simulations ('normal' and 'other') and for all random simulations
    (see ColocLite() arg all_random and method set_simulation_suffixes().
    """

    print(
        "Depreciation warning: This function has been moved to coloc_plot.py.") 
    from . import coloc_plot
    return coloc_plot.plot_data(
        coloc=coloc, name=name, groups=groups, single=single,
        y_var=y_var, tomos=tomos, simulated=simulated, normalize=normalize,
        sets=sets, pp=pp, ax=ax)

def plot_32_p(
        name, coloc=None, groups=None, single=False,
        y_var='p_subcol_combined', tomos=None, sets={},
        ax=None, figsize=(15, 3)):
    """
    Important: Moved to coloc_plot.py. Left here for backcompatibility.
    
    """
    print(
        "Depreciation warning: This function has been moved to coloc_plot.py.") 
    from . import coloc_plot
    return coloc_plot.plot_32_p(
        name=name, coloc=coloc, groups=groups, single=single,
        y_var=y_var, tomos=tomos,
        ax=ax, figsize=figsize)

def plot_32_data(
        name, coloc=None, groups=None, single=False,
        y_var='n_subcol', tomos=None, simulated={}, normalize=None,
        sets={}, ax=None, figsize=(15, 3)):
    """
    Important: Moved to coloc_plot.py. Left here for backcompatibility.
    
    """
    print(
        "Depreciation warning: This function has been moved to coloc_plot.py.") 
    from . import coloc_plot
    return coloc_plot.plot_32_data(
        name=name, coloc=coloc, groups=groups, single=single,
        y_var=y_var, tomos=tomos, simulated=simulated, normalize=normalize,
        sets=sets, ax=ax, figsize=figsize)

