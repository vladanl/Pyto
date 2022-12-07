"""

Common stuff for tests

# Author: Vladan Lucic
# $Id$
"""

__version__ = "$Revision$"

import numpy as np
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

