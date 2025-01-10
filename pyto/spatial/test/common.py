"""

Common stuff for tests

# Author: Vladan Lucic
# $Id$
"""

__version__ = "$Revision$"

import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
import pandas as pd


def make_coloc_tables(random_stats=True, p_func='greater'):
    """
    """

    # tomo specific
    n_subcol_random_all_tomos = [
        [0, 0, 1], [0, 1, 2], [1, 1, 2], [2, 3, 4], [4, 4, 4], [5, 5, 5]]
    n_subcol_random_alt_all_tomos = [
        [0, 1, 1], [0, 1, 2], [2, 1, 2], [2, 3, 4], [4, 4, 4], [3, 4, 5]]
    n_set1_subcol_random_tomos = [
        [0, 0, 1], [0, 1, 2], [1, 1, 2], [2, 3, 4], [4, 4, 4], [5, 5, 5]]
    n_set3_subcol_random_tomos = [
        [0, 0, 1], [0, 1, 2], [1, 1, 2], [2, 3, 4], [4, 4, 4], [5, 5, 5]]
    n_set8_subcol_random_tomos = [
        [0, 0, 1], [0, 1, 2], [1, 1, 2], [2, 3, 4], [4, 4, 4], [5, 5, 5]]
    n_set1_subcol_random_tomos = np.asarray(n_subcol_random_all_tomos) + 1
    n_set3_subcol_random_tomos = np.asarray(n_subcol_random_all_tomos) + 3
    n_set8_subcol_random_tomos = np.asarray(n_subcol_random_all_tomos) + 8
    data_tomos = pd.DataFrame({
        'id': ['alpha', 'alpha', 'bravo', 'bravo', 'charlie', 'charlie'],
        'distance': [5, 10, 5, 10, 5, 10],
        'n_subcol': [0, 2, 1, 4, 3, 3],
        'n_set1_subcol': [0, 2, 1, 5, 4, 4],
        'n_set1_total': [2, 2, 8, 8, 4, 4],
        'n_set3_subcol': [0, 2, 1, 4, 3, 3],
        'n_set3_total': [2, 2, 8, 8, 4, 4],
        'n_set8_subcol': [0, 1, 5, 5, 3, 3],
        'n_set8_total': [7, 8, 7, 8, 7, 8],
        'n_col': [0, 2, 1, 3, 2, 1],
        'area_col': [0, 20, 10, 35, 25, 20],
        'volume': [1000, 1000, 2000, 2000, 3000, 3000],
        'area': [100, 100, 200, 200, 300, 300],
        'tomo_like': ['a', 'a', 'b', 'b', 'c', 'c'],
        'p_subcol_normal': [0, 2/3, 0, 2/3, 0, 0],
        'p_subcol_other': [0, 2/3, 0, 2/3, 0, 0],
        'p_subcol_combined': [0, 2/3, 0, 2/3, 0, 0],
        'n_subcol_random_all': n_subcol_random_all_tomos,
        'n_subcol_random_alt_all': n_subcol_random_alt_all_tomos,
        'n_subcol_random_mean': [1/3, 1, 4/3, 3, 4, 5],
        'n_subcol_random_std': np.sqrt(np.array([2/3, 2, 2/3, 2, 0, 0]) / 3),
        'n_subcol_random_alt_mean': [2/3, 1, 5/3, 3, 4, 4],
        'n_subcol_random_alt_std': np.sqrt(
            np.array([2/3, 2, 2/3, 2, 0, 2]) /3),
        'n_subcol_random_combined_mean': [1/2, 1, 3/2, 3, 4, 4.5],
        'n_subcol_random_combined_std': np.sqrt(
            np.array([3/2, 4, 3/2, 4, 0, 14/4]) / 6),
        'n_set1_subcol_random': n_set1_subcol_random_tomos.tolist(),
        'n_set3_subcol_random': n_set3_subcol_random_tomos.tolist(),
        'n_set8_subcol_random': n_set8_subcol_random_tomos.tolist()
    })

    if p_func == 'less_equal':
        data_tomos['p_subcol_normal'] = [1, 1/3, 1, 1/3, 1, 1]
        data_tomos['p_subcol_other'] = [1, 1/3, 1, 1/3, 1, 1]
        data_tomos['p_subcol_combined'] = [1, 1/3, 1, 1/3, 1, 1]
    #data_tomos = data_tomos.set_index('id')

    # combined
    n_subcol_random_all_5 = (
        np.asarray(n_subcol_random_all_tomos[0])
        + np.asarray(n_subcol_random_all_tomos[2])
        + np.asarray(n_subcol_random_all_tomos[4])).tolist()
    n_subcol_random_alt_all_5 = (
        np.asarray(n_subcol_random_alt_all_tomos[0])
        + np.asarray(n_subcol_random_alt_all_tomos[2])
        + np.asarray(n_subcol_random_alt_all_tomos[4])).tolist()
    n_subcol_random_all_10 = (
        np.asarray(n_subcol_random_all_tomos[1])
        + np.asarray(n_subcol_random_all_tomos[3])
        + np.asarray(n_subcol_random_all_tomos[5])).tolist()
    n_subcol_random_alt_all_10 = (
        np.asarray(n_subcol_random_alt_all_tomos[1])
        + np.asarray(n_subcol_random_alt_all_tomos[3])
        + np.asarray(n_subcol_random_alt_all_tomos[5])).tolist()
    n_subcol_random_all = [n_subcol_random_all_5, n_subcol_random_all_10]
    n_subcol_random_alt_all = [
        n_subcol_random_alt_all_5, n_subcol_random_alt_all_10]
    n_set1_subcol_random = np.asarray(n_subcol_random_all) + 3
    n_set3_subcol_random = np.asarray(n_subcol_random_all) + 9
    n_set8_subcol_random = np.asarray(n_subcol_random_all) + 24
       
    data = pd.DataFrame({
        'distance': [5, 10], 'dist_like': ['five', 'ten'], 'n_subcol': [4, 9],
        'n_set1_subcol': [5, 11], 'n_set1_total': [14, 14],
        'n_set3_subcol': [4, 9], 'n_set3_total': [14, 14],
        'n_set8_subcol': [8, 9], 'n_set8_total': [21, 24],
        'n_col': [3, 6], 'area_col': [35, 75],
        'volume': [6000, 6000], 'area': [600, 600],        
        'p_subcol_normal': [0, 1/3], 'p_subcol_other': [0, 2/3],
        'p_subcol_combined': [0, 3/6],
        'n_subcol_random_all': n_subcol_random_all,
        'n_subcol_random_alt_all': n_subcol_random_alt_all,
        'n_subcol_random_mean': [17/3, 9],
        'n_subcol_random_std': np.sqrt(np.array(
            [99/3 - (17/3)**2, 251/3 - 9**2])),
        'n_subcol_random_alt_mean': [19/3, 8],
        'n_subcol_random_alt_std': np.sqrt(np.array(
            [121/3 - (19/3)**2, 210/3 - 8**2])),         
        'n_subcol_random_combined_mean': [6, 8.5],
        'n_subcol_random_combined_std': np.sqrt(np.array(
            [220/6 - 6**2, 461/6 - (8.5)**2])),
        'n_set1_subcol_random': n_set1_subcol_random.tolist(),
        'n_set3_subcol_random': n_set3_subcol_random.tolist(),
        'n_set8_subcol_random': n_set8_subcol_random.tolist()
        })

    if p_func == 'greater':
        pass
    elif p_func == 'less_equal':
        data['p_subcol_normal'] = [1, 2/3]
        data['p_subcol_other'] = [1, 1/3]
        data['p_subcol_combined'] =  [1, 3/6]
        
    if random_stats is False:
        drop_cols = [
            'n_subcol_random_mean', 'n_subcol_random_std',
            'n_subcol_random_alt_mean', 'n_subcol_random_alt_std',
            'n_subcol_random_combined_mean', 'n_subcol_random_combined_std']
        data_tomos = data_tomos.drop(columns=drop_cols)
        data = data.drop(columns=drop_cols)

    return data, data_tomos

def make_coloc_tables_ac(random_stats=False):
    """
    Returns colocalization tables like those made my make_coloc_tables(),
    but containing only rows for alpha nad charlie tomos
    """

    _, full_tomo = make_coloc_tables(random_stats=random_stats)

    # separate tomos table
    if full_tomo.index.name is None:
        condition = (
            (full_tomo['id'] == 'alpha') | (full_tomo['id'] == 'charlie'))
    else:
        condition = (
            (full_tomo.index == 'alpha') | (full_tomo.index == 'charlie')) 
    data_tomo = full_tomo[condition].copy()

    # joined tomos table
    n_subcol_random_all = [[4, 4, 5], [5, 6, 7]]
    n_set1_subcol_random = (np.asarray(n_subcol_random_all) + 2).tolist()
    n_set3_subcol_random = (np.asarray(n_subcol_random_all) + 6).tolist()
    n_set8_subcol_random = (np.asarray(n_subcol_random_all) + 16).tolist()
    data = pd.DataFrame({
        'distance': [5, 10], 'dist_like': ['five', 'ten'], 'n_subcol': [3, 5],
        'n_set1_subcol': [4, 6], 'n_set1_total': [6, 6],
        'n_set3_subcol': [3, 5], 'n_set3_total': [6, 6],
        'n_set8_subcol': [3, 4], 'n_set8_total': [14, 16],
        'n_col': [2, 3], 'area_col': [25, 40],
        'volume': [4000, 4000], 'area': [400, 400],        
        'p_subcol_normal': [0., 0], 'p_subcol_other': [0, 1/3],
        'p_subcol_combined': [0, 1/6],
        'n_subcol_random_all': n_subcol_random_all,
        'n_subcol_random_alt_all': [[4, 5, 5], [3, 5, 7]],
        'n_set1_subcol_random': n_set1_subcol_random,
        'n_set3_subcol_random': n_set3_subcol_random,
        'n_set8_subcol_random': n_set8_subcol_random
    })

    if random_stats is True:
        raise ValueError("Sorry, not implemented for random_stats=True")

    return data, data_tomo

def make_coloc_tables_1tomo_2d():
    """Makes colocalization tables for 2d data of one tomo.

    The data underlying these tables is in test_coloc_core because
    that is where the creation of these tables is tested in more detail. 
    """

    # particle patterns and distances
    global pattern_0, pattern_1, pattern_2, dist_0_0, dist_0_1, dist_0_2
    global pattern_none, pattern_empty 
    pattern_0 = np.array(
        [[1, 1], [11, 1], [11, 20], [1, 20], [1, 15]])
    pattern_1 = np.array(
        [[1, 2], [6, 1], [11, 2], [11, 4], [18, 20], [4, 20]])
    pattern_2 = np.array(
        [[4, 1], [13, 1], [1, 25]])
    dist_0_0 = cdist(pattern_0, pattern_0)
    dist_0_1 = cdist(pattern_0, pattern_1)
    dist_0_2 = cdist(pattern_0, pattern_2)
    pattern_none = None
    pattern_empty = np.array([[]])

    # d = 2
    global coloc2_d2, particles2_d2, coloc2_indices_d2, particles2_indices_d2
    global coloc2_n_d2, particles2_n_d2
    global coloc3_d2, particles3_d2, coloc3_indices_d2, particles3_indices_d2
    global coloc3_n_d2, particles3_n_d2
    coloc2_d2 = [
        [True, True, False, False, False], [False, False, False, False, False]]
    particles2_d2 = [
        [[True, True, False, False, False],
         [True, False, True, False, False, False]],
        [[False, False, False, False, False], [False, False, False]]]
    coloc2_indices_d2 = [[0, 1], []]
    particles2_indices_d2 = [[[0, 1], [0, 2]], [[], []]]
    coloc2_n_d2 = [2, 0]
    particles2_n_d2 = [[2, 2], [0, 0]]
    coloc3_d2 = np.array([False, False, False, False, False])
    particles3_d2 = [
        np.array([False, False, False, False, False]),
        np.array([False, False, False, False, False, False]),
        np.array([False, False, False])]
    coloc3_indices_d2 = []
    particles3_indices_d2 = [[], [], []]
    coloc3_n_d2 = 0
    particles3_n_d2 = [0, 0, 0]

    # d = 2 less_eq
    global coloc2_d2_le, particles2_d2_le, coloc2_indices_d2_le
    global particles2_indices_d2_le, coloc2_n_d2_le, particles2_n_d2_le
    global coloc3_d2_le, particles3_d2_le, coloc3_indices_d2_le
    global particles3_indices_d2_le, coloc3_n_d2_le, particles3_n_d2_le
    coloc2_d2_le = [
        [True, True, False, False, False], [False, True, False, False, False]]
    particles2_d2_le = [
        [[True, True, False, False, False],
         [True, False, True, False, False, False]],
        [[False, True, False, False, False], [False, True, False]]]
    coloc2_indices_d2_le = [[0, 1], [1]]
    particles2_indices_d2_le = [[[0, 1], [0, 2]], [[1], [1]]]
    coloc2_n_d2_le = [2, 1]
    particles2_n_d2_le = [[2, 2], [1, 1]]
    coloc3_d2_le = np.array([False, True, False, False, False])
    particles3_d2_le = [
        np.array([False, True, False, False, False]),
        np.array([False, False, True, False, False, False]),
        np.array([False, True, False])]
    coloc3_indices_d2_le = [1]
    particles3_indices_d2_le = [[1], [2], [1]]
    coloc3_n_d2_le = 1
    particles3_n_d2_le = [1, 1, 1]

    # d = 4
    global coloc2_d4, particles2_d4, coloc2_indices_d4, particles2_indices_d4
    global coloc2_n_d4, particles2_n_d4
    global coloc3_d4, particles3_d4, coloc3_indices_d4, particles3_indices_d4
    global coloc3_n_d4, particles3_n_d4
    coloc2_d4 = [
        [True, True, False, True, False], [True, True, False, False, False]]
    particles2_d4 = [
        [[True, True, False, True, False],
         [True, False, True, True, False, True]],
        [[True, True, False, False, False], [True, True, False]]]
    coloc2_indices_d4 = [[0, 1, 3], [0, 1]]
    particles2_indices_d4 = [
        [[0, 1, 3], [0, 2, 3, 5]], [[0, 1], [0, 1]]]
    coloc2_n_d4 = [3, 2]
    particles2_n_d4 = [[3, 4], [2, 2]]
    coloc3_d4 = np.array([True, True, False, False, False])
    particles3_d4 = [
        np.array([True, True, False, False, False]),
        np.array([True, False, True, True, False, False]),
        np.array([True, True, False])]
    coloc3_indices_d4 = [0, 1]
    particles3_indices_d4 = [[0, 1], [0, 2, 3], [0, 1]]
    coloc3_n_d4 = 2
    particles3_n_d4 = [2, 3, 2]

    # d = 6
    global coloc2_d6, particles2_d6, coloc2_indices_d6, particles2_indices_d6
    global coloc2_n_d6, particles2_n_d6
    global coloc3_d6, particles3_d6, coloc3_indices_d6, particles3_indices_d6
    global coloc3_n_d6, particles3_n_d6
    coloc2_d6 = [
        [True, True, False, True, True], [True, True, False, True, False]]
    particles2_d6 = [
        [[True, True, False, True, True],
         [True, True, True, True, False, True]],
        [[True, True, False, True, False], [True, True, True]]]
    coloc2_indices_d6 = [[0, 1, 3, 4], [0, 1, 3]]
    particles2_indices_d6 = [
        [[0, 1, 3, 4], [0, 1, 2, 3, 5]], [[0, 1, 3], [0, 1, 2]]]
    coloc2_n_d6 = [4, 3]
    particles2_n_d6 = [[4, 5], [3, 3]]
    coloc3_d6 = np.array([True, True, False, True, False])
    particles3_d6 = [
        np.array([True, True, False, True, True]),
        np.array([True, True, True, True, False, True]),
        np.array([True, True, True])]
    coloc3_indices_d6 = [0, 1, 3]
    particles3_indices_d6 = [[0, 1, 3, 4], [0, 1, 2, 3, 5], [0, 1, 2]]
    coloc3_n_d6 = 3
    particles3_n_d6 = [4, 5, 3]

    # d = 8
    global coloc2_d8, particles2_d8, coloc2_indices_d8, particles2_indices_d8
    global coloc2_n_d8, particles2_n_d8
    global coloc3_d8, particles3_d8, coloc3_indices_d8, particles3_indices_d8
    global coloc3_n_d8, particles3_n_d8
    coloc2_d8 = [
        [True, True, True, True, True], [True, True, False, True, False]]
    particles2_d8 = [
        [[True, True, True, True, True], [True, True, True, True, True, True]],
        [[True, True, False, True, False], [True, True, True]]]
    coloc2_indices_d8 = [[0, 1, 2, 3, 4], [0, 1, 3]]
    particles2_indices_d8 = [
        [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]], [[0, 1, 3], [0, 1, 2]]]
    coloc2_n_d8 = [5, 3]
    particles2_n_d8 = [[5, 6], [3, 3]]
    coloc3_d8 = np.array([True, True, False, True, False])
    particles3_d8 = [
        np.array([True, True, False, True, True]),
        np.array([True, True, True, True, False, True]),
        np.array([True, True, True])]
    coloc3_indices_d8 = [0, 1, 3]
    particles3_indices_d8 = [[0, 1, 3, 4], [0, 1, 2, 3, 5], [0, 1, 2]]
    coloc3_n_d8 = 3
    particles3_n_d8 = [4, 5, 3]

    # region image
    global region, size_region
    region = np.zeros(shape=(50, 50), dtype=int)
    region[:30, :30] = 4
    size_region = 900

    # columns
    global n_columns_3, n_columns_0_1, n_columns_0_2
    global size_col_3, size_col_0_1, size_col_0_2
    n_columns_3 = [0, 2, 2, 2]
    n_columns_0_1 = [2, 3, 2, 2]
    n_columns_0_2 = [0, 2, 2, 2]
    size_col_3 = [0, 57, 183, 279]
    size_col_0_1 = [18, 90, 218, 458]
    size_col_0_2 = [0, 57, 183, 279]
        
    # dataframes
    global pat0_pat1_pat2_data, pat0_pat1_data, pat0_pat2_data
    pat0_pat1_pat2_data = pd.DataFrame({
        'distance': [2, 4, 6, 8], 'id': 'tomo', 'n_subcol': [0, 2, 3, 3],
        'n_pattern0_subcol': [0, 2, 4, 4], 'n_pattern1_subcol': [0, 3, 5, 5],
        'n_pattern2_subcol': [0, 2, 3, 3], 'n_pattern0_total': 5, 
        'n_pattern1_total': [6, 6, 6, 6], 'n_pattern2_total': [3, 3, 3, 3],
        'size_region': 900, 'n_col': [0, 2, 2, 2],
        'size_col': size_col_3})

    pat0_pat1_data = pd.DataFrame({
        'distance': [2, 4, 6, 8], 'id': 'tomo', 'n_subcol': [2, 3, 4, 5],
        'n_pattern0_subcol': [2, 3, 4, 5], 'n_pattern1_subcol': [2, 4, 5, 6],
        'n_pattern0_total': 5,  'n_pattern1_total': [6, 6, 6, 6],
        'size_region': 900, 'n_col': [2, 3, 2, 2],
        'size_col': size_col_0_1})

    pat0_pat2_data = pd.DataFrame({
        'distance': [2, 4, 6, 8], 'id': 'tomo', 'n_subcol': [0, 2, 3, 3],
        'n_pattern0_subcol': [0, 2, 3, 3], 'n_pattern2_subcol': [0, 2, 3, 3],
        'n_pattern0_total': 5, 'n_pattern2_total': [3, 3, 3, 3],
        'size_region': 900, 'n_col': [0, 2, 2, 2],
        'size_col': size_col_0_2})

    return (pattern_0, pattern_1, pattern_2, dist_0_1, dist_0_2,
            pat0_pat1_pat2_data, pat0_pat1_data, pat0_pat2_data)




