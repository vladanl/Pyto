"""

Tests module coloc_functions

# Author: Vladan Lucic
# $Id$
"""
__version__ = "$Revision$"

import os
import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd

from pyto.spatial.test import common
import pyto.spatial.coloc_functions as col_func

class TestColocFunctions(np_test.TestCase):
    """
    Tests coloc_functions module
    """

    def setUp(self):
        """
        """
        pass
          
    def test_read_data(self):
        """
        Tests read_data() using data from munc13 columns project  
        """

        pkl_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data/3_set0_set1_set2_sim-3_wspace.pkl')
        columns_dict = {
            39: 'n_subcol', 21: 'n_set0_subcol', 22: 'n_set1_subcol',
            23: 'n_set2_subcol', 17: 'n_set0_total', 18: 'n_set1_total',
            19: 'n_set2_total', 40: 'n_subcol_random_all',
            41: 'n_subcol_random_alt_all', 25: 'n_set0_subcol_random',
            26: 'n_set1_subcol_random', 27: 'n_set2_subcol_random',
            28: 'area', 14: 'volume', 0: 'n_col', 29: 'area_col'}
        expected_columns = list(columns_dict.values())
        expected_columns = ['id'] + expected_columns
        actual = col_func.read_data(
            pkl_path=pkl_path, columns=columns_dict, mode='munc13')
        np_test.assert_equal(list(actual.columns), expected_columns)
        np_test.assert_equal(
            actual['n_subcol'].to_list(), [0, 0, 0, 0, 0, 20, 0, 0])
        np_test.assert_equal(
            actual['n_set2_subcol'].to_list(), [0, 0, 0, 0, 0, 6, 0, 0])
        np_test.assert_equal(
            actual['n_set0_total'].to_list(),
            [268, 28, 20, 210, 54, 330, 426, 54])
        np_test.assert_equal(
            actual['n_set1_subcol_random'].to_list(),
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0],
             [0, 0, 0], [2, 1, 1], [4, 2, 1], [0, 0, 0]])
        np_test.assert_equal(
            actual['volume'].to_list(),
            [422208.0, 241472.0, 377064.0, 382328.0,
             79728.0, 526720.0, 396096.0, 184128.0])
        np_test.assert_equal(
            actual['n_col'].to_list(), [0, 0, 0, 0, 0, 2, 0, 0])
        
    def test_select_rows(self):
        """
        Tests selct_rows()
        """

        # select distance
        _, data_tomo = common.make_coloc_tables(random_stats=True)
        actual_tomo = col_func.select_rows(data=data_tomo, distance=10)
        np_test.assert_equal(
            data_tomo.columns.to_list() == actual_tomo.columns.to_list(),
            True)
        np_test.assert_equal(actual_tomo['distance'].to_list(), [10, 10, 10])
        actual_tomo = col_func.select_rows(data=data_tomo, distance=5)
        np_test.assert_equal(
            data_tomo.columns.to_list() == actual_tomo.columns.to_list(),
            True)
        np_test.assert_equal(actual_tomo['distance'].to_list(), [5, 5, 5])
        actual_tomo = col_func.select_rows(data=data_tomo, distance=6)
        np_test.assert_equal(actual_tomo.empty, True)

        # select tomos by id, indexed by id
        actual_tomo = col_func.select_rows(
            data=data_tomo, ids=['alpha', 'charlie'])
        np_test.assert_equal(
            data_tomo.columns.to_list() == actual_tomo.columns.to_list(),
            True)
        if actual_tomo.index.name is None:
            np_test.assert_equal(
                actual_tomo['id'].to_list(),
                ['alpha', 'alpha', 'charlie', 'charlie'])
        else:
            np_test.assert_equal(
                actual_tomo.index.to_list(),
                ['alpha', 'alpha', 'charlie', 'charlie'])

        # select tomos by id, id not index
        data_tomo = data_tomo.reset_index()
        actual_tomo = col_func.select_rows(
            data=data_tomo, ids=['alpha', 'charlie'])
        np_test.assert_equal(
            data_tomo.columns.to_list() == actual_tomo.columns.to_list(),
            True)
        np_test.assert_equal(
            actual_tomo['id'].to_list(),
            ['alpha', 'alpha', 'charlie', 'charlie'])
      
    def test_aggregate(self):
        """
        Tests aggregate()
        """

        # individual distances
        data, data_tomos = common.make_coloc_tables(random_stats=False)
        actual = col_func.aggregate(
            data=data_tomos, distance=5,
            add_columns=['n_subcol', 'n_set1_subcol', 'n_set1_total'],
            array_columns=['n_subcol_random_all', 'n_subcol_random_alt_all'],
            p_values=False, random_stats=False)
        np_test.assert_equal(
            actual.iloc[0].equals(data[actual.columns].iloc[0]), True)
        actual = col_func.aggregate(
            data=data_tomos, distance=10,
            add_columns=['n_subcol', 'n_set1_subcol', 'n_set1_total'],
            array_columns=['n_subcol_random_all', 'n_subcol_random_alt_all'],
            p_values=False, random_stats=False)
        np_test.assert_equal(
            actual.iloc[0].equals(data[actual.columns].iloc[1]), True)
    
        # multiple distances
        data, data_tomos = common.make_coloc_tables(random_stats=False)
        actual = col_func.aggregate(
            data=data_tomos, distance=[5, 10],
            add_columns=['n_subcol', 'n_set1_subcol', 'n_set1_total'],
            array_columns=['n_subcol_random_all', 'n_subcol_random_alt_all'],
            p_values=False, random_stats=False)
        np_test.assert_equal(
            actual.equals(data[actual.columns]), True)
 
        # multiple distances
        data, data_tomos = common.make_coloc_tables(random_stats=False)
        actual = col_func.aggregate(
            data=data_tomos, distance=range(5, 11, 5),
            add_columns=['n_subcol', 'n_set1_subcol', 'n_set1_total'],
            array_columns=['n_subcol_random_all', 'n_subcol_random_alt_all'],
            p_values=False, random_stats=False)
        np_test.assert_equal(
            actual.equals(data[actual.columns]), True)
 
        # multiple distances, p values, random stats
        data, data_tomos = common.make_coloc_tables(random_stats=True)
        actual = col_func.aggregate(
            data=data_tomos, distance=None,
            add_columns=[
                'n_subcol', 'n_set1_subcol', 'n_set1_total', 'n_set3_subcol',
                'n_set3_total', 'n_set8_subcol', 'n_set8_total',
                'n_col', 'area_col', 'volume', 'area'],
            array_columns=['n_subcol_random_all', 'n_subcol_random_alt_all'],
            p_values=True, random_stats=True)
        np_test.assert_equal(
            set(['n_subcol_random_all', 'p_subcol_normal',
                 'n_subcol_random_mean', 'n_subcol_random_std'])
            - set(actual.columns.to_numpy()),
            set([]))
        np_test.assert_array_equal(
            [x in set(actual.columns.to_numpy())
             for x
             in ['n_subcol_random_alt_all', 'p_subcol_other',
                 'n_subcol_random_alt_mean', 'n_subcol_random_alt_std']],
             4 * [True])
        np_test.assert_equal(
            actual.round(12).equals(data[actual.columns].round(12)), True)
 
        # multiple distances, p values, random stats, all random
        data, data_tomos = common.make_coloc_tables(random_stats=True)
        actual = col_func.aggregate(
            data=data_tomos, distance=None,
            add_columns=[
                'n_subcol', 'n_set1_subcol', 'n_set1_total', 'n_set3_subcol',
                'n_set3_total', 'n_set8_subcol', 'n_set8_total',
                'n_col', 'area_col', 'volume', 'area'],
            array_columns=['n_subcol_random_all'],# 'n_subcol_random_alt_all'],
            p_values=True, random_stats=True,
            random_suff=['random'], p_suff=['normal'])
        np_test.assert_equal(
            set(['n_subcol_random_all', 'p_subcol_normal',
                 'n_subcol_random_mean', 'n_subcol_random_std'])
            - set(actual.columns.to_numpy()),
            set([]))
        np_test.assert_array_equal(
            [x in set(actual.columns.to_numpy())
             for x
             in ['n_subcol_random_alt_all', 'p_subcol_other',
                 'n_subcol_random_alt_mean', 'n_subcol_random_alt_std']],
             4 * [False])
        np_test.assert_equal(
            actual.round(12).equals(data[actual.columns].round(12)), True)
 
       # multiple distances, p values <=
        data, data_tomos = common.make_coloc_tables(
            p_func='less_equal', random_stats=True)
        actual = col_func.aggregate(
            data=data_tomos, distance=None,
            add_columns=['n_subcol', 'n_set1_subcol', 'n_set1_total'],
            array_columns=['n_subcol_random_all', 'n_subcol_random_alt_all'],
            p_values=True, p_func=np.less_equal, random_stats=True)
        np_test.assert_equal(
            actual.round(12).equals(data[actual.columns].round(12)), True)

    def test_get_random_stats(self):
        """
        Tests get_random_stats() and implicitly get_random_stats_single()
        """

        # input and expected data
        data, data_tomo = common.make_coloc_tables(random_stats=False)
        expected, expected_tomo = common.make_coloc_tables(random_stats=True)

        # check first random
        actual, actual_tomo = col_func.get_random_stats(
            data=data, data_syn=data_tomo, column='n_subcol_random_all', 
            out_column='n_subcol_random', combine=False)
        cols = ['n_subcol_random_mean', 'n_subcol_random_std']
        np_test.assert_equal(
            actual_tomo[cols].equals(expected_tomo[cols]), True)
        np_test.assert_equal(
            actual[cols].round(12).equals(expected[cols].round(12)), True)

        # check second random
        actual, actual_tomo = col_func.get_random_stats(
            data=actual, data_syn=actual_tomo,
            column='n_subcol_random_alt_all', 
            out_column='n_subcol_random_alt', combine=False)
        cols = ['n_subcol_random_alt_mean', 'n_subcol_random_alt_std']
        np_test.assert_equal(
            actual_tomo[cols].round(12).equals(expected_tomo[cols].round(12)),
            True)
        np_test.assert_equal(
            actual[cols].round(12).equals(expected[cols].round(12)), True)

        # check combined random
        actual, actual_tomo = col_func.get_random_stats(
            data=actual, data_syn=actual_tomo, 
            column=['n_subcol_random_all', 'n_subcol_random_alt_all'], 
            out_column='n_subcol_random_combined', combine=True)         
        cols = [
            'n_subcol_random_combined_mean', 'n_subcol_random_combined_std']
        np_test.assert_equal(
            actual_tomo[cols].round(12).equals(expected_tomo[cols].round(12)),
            True)
        np_test.assert_equal(
            actual[cols].round(12).equals(expected[cols].round(12)), True)
        
    def test_get_fraction_random(self):
        """
        Tests get_fraction_random()
        """

        # use np.greater
        data, data_tomo = common.make_coloc_tables(random_stats=False)
        p_columns = ['p_subcol_normal', 'p_subcol_other', 'p_subcol_combined']
        data_tomo_clean = data_tomo.drop(columns=p_columns)
        actual = col_func.get_fraction_random(data=data_tomo_clean)
        np_test.assert_equal(
            actual[p_columns].equals(data_tomo[p_columns]), True)

        # use np.less_equal
        data, data_tomo = common.make_coloc_tables(
            random_stats=False, p_func='less_equal')
        p_columns = ['p_subcol_normal', 'p_subcol_other', 'p_subcol_combined']
        data_tomo_clean = data_tomo.drop(columns=p_columns)
        actual = col_func.get_fraction_random(
            data=data_tomo_clean, p_func=np.less_equal)
        np_test.assert_equal(
            actual[p_columns].equals(data_tomo[p_columns]), True)
        
    def test_get_fraction_single(self):
        """
        Tests get_fraction_single()
        """
       
        # use np.greater
        data, data_tomo = common.make_coloc_tables(random_stats=False)
        actual, n_good, n_random = col_func.get_fraction_single(
            data_tomo, random_column='n_subcol_random_all',
            exp_column='n_subcol', fraction_column='fraction')
        np_test.assert_equal(
            actual['fraction'].equals(data_tomo['p_subcol_normal']), True)
        actual, n_good, n_random = col_func.get_fraction_single(
            data_tomo, random_column='n_subcol_random_alt_all',
            exp_column='n_subcol', fraction_column='fraction')
        np_test.assert_equal(
            actual['fraction'].equals(data_tomo['p_subcol_other']), True)

        # use np.less_equal
        data, data_tomo = common.make_coloc_tables(
            random_stats=False, p_func='less_equal')
        actual, n_good, n_random = col_func.get_fraction_single(
            data_tomo, random_column='n_subcol_random_all',
            exp_column='n_subcol', fraction_column='fraction',
            p_func=np.less_equal)
        np_test.assert_equal(
            actual['fraction'].equals(data_tomo['p_subcol_normal']), True)
        actual, n_good, n_random = col_func.get_fraction_single(
            data_tomo, random_column='n_subcol_random_alt_all',
            exp_column='n_subcol', fraction_column='fraction',
            p_func=np.less_equal)
        np_test.assert_equal(
            actual['fraction'].equals(data_tomo['p_subcol_other']), True)

    def test_get_names(self):
        """Tests get_names() and get_layers()
        """
        
        np_test.assert_array_equal(
            col_func.get_names(name='setx_sety_setz'), ['setx', 'sety', 'setz'])
        np_test.assert_array_equal(
            col_func.get_names(name='setx_sety_ves_ap'),
            ['setx', 'sety', 'ves', 'ap'])
        np_test.assert_array_equal(
            col_func.get_names(name='setx_sety_ves_ap', mode='columns_2021'),
            ['setx', 'sety', 'ves_ap'])
        np_test.assert_array_equal(
            col_func.get_layers(name='setx_sety_ves_ap', mode='_'),
            ['setx', 'sety', 'ves', 'ap'])

        # other mode
        np_test.assert_array_equal(
            col_func.get_names(name='X-1__Y-2__Z-3', mode='__'),
            ['X-1', 'Y-2', 'Z-3'])
       
    def test_make_name(self):
        """Tests make_name()
        """

        actual = col_func.make_name(
            names=['setx', 'sety', 'setz'], suffix='dat')
        np_test.assert_equal(actual, 'setx_sety_setz_dat')
        actual = col_func.make_name(
            names=['setx', 'sety', 'setz'], suffix=None)
        np_test.assert_equal(actual, 'setx_sety_setz')

    def test_make_full_coloc_names(self):
        """Tests make_full_coloc_names()
        """
        
        actual = col_func.make_full_coloc_names(
            names=['setX', 'setY', 'setZ', 'setW'], suffix='data')
        desired = [
            'setX_setY_setZ_setW_data', 'setX_setY_data', 'setX_setZ_data',
            'setX_setW_data'] 
        np_test.assert_equal(actual, desired)
        
        actual = col_func.make_full_coloc_names(
            names=['setX', 'setY', 'setZ', 'setW'], suffix=None)
        desired = [
            'setX_setY_setZ_setW', 'setX_setY', 'setX_setZ',
            'setX_setW'] 
        np_test.assert_equal(actual, desired)
        
        actual = col_func.make_full_coloc_names(
            names=['setX', 'setY'], suffix='data')
        desired = ['setX_setY_data']
        np_test.assert_equal(actual, desired)
        
    def test_get_2_names(self):
        """Tests get_2_names()
        """

        actual = col_func.get_2_names(
            name='pre_sv_post', order=((0, 1), (0, 2)))
        np_test.assert_equal(actual, ['pre_sv', 'pre_post'])

        actual = col_func.get_2_names(
            name=['pre_sv_post'], order=((0, 1), (2, 1)))
        np_test.assert_equal(actual, ['pre_sv', 'post_sv'])
        actual = col_func.get_2_names(
            name=['pre1_sv1_post1', 'pre2_sv2_post2'],
            order=((0, 1), (1, 2)))
        np_test.assert_equal(
            actual, [['pre1_sv1', 'sv1_post1'], ['pre2_sv2', 'sv2_post2']])

        # order None
        actual = col_func.get_2_names(name=['pre_sv_post_4th'])
        np_test.assert_equal(actual, ['pre_sv', 'pre_post', 'pre_4th'])

        # multiple names, changed mode
        actual = col_func.get_2_names(
            name=['pre__sv__post', 'X__Y__Z'], mode='__')
        np_test.assert_equal(
            actual, [['pre__sv', 'pre__post'], ['X__Y', 'X__Z']])
        
        # multiple names, by_order
        actual = col_func.get_2_names(
            name=['pre__sv__post', 'X_a__Y_b__Z_c', '1__2__3'], mode='__',
            by_order=True)
        np_test.assert_equal(
            actual,
            [['pre__sv', 'X_a__Y_b', '1__2'],
             ['pre__post', 'X_a__Z_c', '1__3']])
        
        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestColocFunctions)
    unittest.TextTestRunner(verbosity=2).run(suite)
