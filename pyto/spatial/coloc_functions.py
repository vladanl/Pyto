"""
Contains (some of the) functions used for colocalization analysus
(module coloc_analysis) that act directly aon individual colocalization
data tables.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""

__version__ = "$Revision$"


import os
import re
import functools
import pickle

import numpy as np
import scipy as sp
import pandas as pd 

import pyto

    
###########################################################
#
# Basic utility functions
#

def get_layers(name):
    """
    Extracts layer names from the colocalization name (arg name)

    Returns: list of layer names
    """
    layers = name.split('_')
    if 'ves' in layers:
        ves_ind = layers.index('ves')
        if layers[ves_ind+1] == 'ap':
            layers[ves_ind] = 'ves_ap'
            layers.remove('ap')
    return layers


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
      - name: colocalization case (layers that are colocalized), such as 
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

    all other cases:
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
    #layers = name.split('_')
    #if 'ves' in layers:
    #    ves_ind = layers.index('ves')
    #    if layers[ves_ind+1] == 'ap':
    #        layers[ves_ind] = 'ves_ap'
    #        layers.remove('ap')
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

def aggregate(
        data, distance, add_columns, array_columns, p_values=True,
        p_func=np.greater, random_stats=True):
    """
    Calculates data for all tomograms (synapses) together by combining the 
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

    Returns: (pandas.DataFrame) data table containg data for all 
    synapses together
    """

    # use all existing distances if None
    if distance is None:
        distance = data['distance'].unique()
        distance.sort()
        
    # multiple distances
    if isinstance(distance, (list, tuple, np.ndarray)):
        for dist in distance:
            dist_row = aggregate(
                data=data, distance=dist, add_columns=add_columns,
                array_columns=array_columns, p_values=p_values, p_func=p_func,
                random_stats=random_stats)
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
        random = data_tomo[
            ['n_subcol', 'n_subcol_random_all', 'n_subcol_random_alt_all']]
        bad_tomos = [
            index for index, row in random.iterrows() 
            if np.isnan(row['n_subcol'])]
        random = random.drop(bad_tomos)

        # experimental n subcolumns
        exp = random['n_subcol'].sum()

        # calculate fractions standard
        random_column = 'n_subcol_random_all'
        random_total = np.vstack(
            np.array(random[random_column])).sum(axis=0)
        n_good = p_func(exp, random_total).sum()
        result['p_subcol_normal'] = n_good / random_total.shape[0]

        # calculate fractions other
        random_column = 'n_subcol_random_alt_all'
        random_alt_total = np.vstack(
            np.array(random[random_column])).sum(axis=0)
        n_good_alt = p_func(exp, random_alt_total).sum()
        result['p_subcol_other'] = n_good_alt / random_alt_total.shape[0]

        # calculate fractions combined
        result['p_subcol_combined'] = (
            n_good + n_good_alt) / float(
            random_total.shape[0] + random_alt_total.shape[0])

    # calculate statistics on random
    if random_stats:
 
        result, _ = get_random_stats(
            data=result, data_syn=data_tomo, column='n_subcol_random_all',
            out_column='n_subcol_random', combine=False)
        result, _ = get_random_stats(
            data=result, data_syn=data_tomo, column='n_subcol_random_alt_all', 
            out_column='n_subcol_random_alt', combine=False)
        result, _ = get_random_stats(
            data=result, data_syn=data_tomo,
            column=['n_subcol_random_all', 'n_subcol_random_alt_all'],
            out_column='n_subcol_random_combined', combine=True)

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
    
def get_fraction_random(data, p_func=np.greater):
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
      - n_subcol_random_all: data from one simulation type (each element
      has to be a list containing values for different simulations
      - n_subcol_random_alt_all: like n_subcol_random_all but for the other
      simulation type
    Consequently, this function is meant to be applied to individual tomos
    tables because the combined tomos tables usually do not have these
    columns.

    The following columns are added to the resulting table:
      - p_subcol_normal: p-values for the first simulation type
      - p_subcol_other: p-values for the second simulation type
      - p_subcol_combined: p-values where the two simylation types are 
      combined

    Arguments:
      - data: (pandas.DataFrame) table containing experimental and random 
      simulation values for each tomogram separately

     Returns:
      - data: (pandas.DataFrame) table where the fractions are added as 
      additional columns for each tomogram separately
      - p_func: function that takes two array arguments like np.greater, 
      np.greater-equal, ...  and returns a boolean array.
   """

    # standard random
    random, n_good, n_random = get_fraction_single(
        data=data, random_column='n_subcol_random_all',
        exp_column='n_subcol', fraction_column='p_subcol_normal',
        p_func=p_func)

    # alternative random
    random_alt, n_good_alt, n_random_alt = get_fraction_single(
        data=data, random_column='n_subcol_random_alt_all',
        exp_column='n_subcol', fraction_column='p_subcol_other',
        p_func=p_func)

    # add fractions to data
    data['p_subcol_normal'] = random['p_subcol_normal'] 
    data['p_subcol_other'] = random_alt['p_subcol_other'] 

    # combined random
    fraction_good_combined = (
        (n_good + n_good_alt) / float(n_random + n_random_alt))
    random['p_subcol_combined'] = fraction_good_combined
    data['p_subcol_combined'] = random['p_subcol_combined']

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

