"""

Tests module coloc_functions

# Author: Vladan Lucic
# $Id$
"""
__version__ = "$Revision$"

import os
import pickle
import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd

from pyto.spatial.test import common
from pyto.spatial.coloc_analysis import ColocAnalysis 

class TestColocAnalysis(np_test.TestCase):
    """
    Tests coloc_functions module
    """

    def setUp(self):
        """
        """
        self.table_dir = 'tables'
        self.raw_coloc_dir = 'raw_data/colocalization_pick-1_bin2_25'

    def test_extract_multi(self):
        """
        Tests extract_multi()
        """

        # setup
        names = ['pre_centroids_post']
        distances = range(5, 31, 5)
        n_sim = 3
        in_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), self.raw_coloc_dir)
        table_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), self.table_dir)
        expected = ColocAnalysis(mode='munc13')
        expected_data, expected_data_syn = expected.extract_data(
            name=names[0], distances=distances, in_path=in_path, mode='munc13',
            n_sim=n_sim, p_values=True, random_stats=True)
        col = ColocAnalysis.extract_multi(
            names=names, distances=distances, in_path=in_path, mode='munc13',
            n_sim=n_sim, random_stats=True, p_values=True,
            save=True, force=True, dir_=table_path, verbose=False,
            join_suffix='join', individual_suffix='individual')

        # test tables 
        np_test.assert_equal(
            set(col.pre_centroids_post_join.columns.to_list()),
            set(expected_data.columns.to_list() + ['density']))
        np_test.assert_equal(
            set(col.pre_centroids_post_individual.columns.to_list()),
            set(expected_data_syn.columns.to_list()))
        actual = col.pre_centroids_post_join.drop('density', axis=1)
        np_test.assert_equal(
            (actual == expected_data).all().all(),
            True)
        actual = col.pre_centroids_post_individual
        np_test.assert_equal(
            (actual == expected_data_syn).all().all(), True)

        # test pickles
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            self.table_dir, 'pre_centroids_post_join.pkl')
        actual = pickle.load(open(path, 'rb'), encoding='latin1')
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            self.table_dir, 'pre_centroids_post_individual.pkl')
        actual_syn = pickle.load(open(path, 'rb'), encoding='latin1')
        np_test.assert_equal(
            set(col.pre_centroids_post_join.columns.to_list()),
            set(actual.columns.to_list()))
        np_test.assert_equal(
            set(col.pre_centroids_post_individual.columns.to_list()),
            set(actual_syn.columns.to_list()))
        np_test.assert_equal(
            (actual == col.pre_centroids_post_join).all().all(),
            True)
        np_test.assert_equal(
            (actual_syn == col.pre_centroids_post_individual).all().all(),
            True)
       
    def test_read_table_multi(self):
        """
        Tests read_table_multi()
        """

        # setup
        names = ['pre_centroids_post']
        distances = range(5, 31, 5)
        n_sim = 3
        in_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), self.raw_coloc_dir)
        table_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), self.table_dir)
        expected = ColocAnalysis.extract_multi(
            names=names, distances=distances, in_path=in_path, mode='munc13',
            n_sim=n_sim, random_stats=True, p_values=True,
            save=True, force=True, dir_=table_path, verbose=False,
            join_suffix='join', individual_suffix='individual')
        actual = ColocAnalysis.read_table_multi(
            dir_=table_path, mode='munc13', suffix='_join.pkl', verbose=False,
            join_suffix='join', individual_suffix='individual')

        # test 
        np_test.assert_equal(
            set(actual.pre_centroids_post_join.columns.to_list()),
            set(expected.pre_centroids_post_join.columns.to_list()))
        np_test.assert_equal(
            set(actual.pre_centroids_post_individual.columns.to_list()),
            set(expected.pre_centroids_post_individual.columns.to_list()))
        np_test.assert_equal(
            (actual.pre_centroids_post_join
             == expected.pre_centroids_post_join).all().all(),
            True)
        np_test.assert_equal(
            (actual.pre_centroids_post_individual
             == expected.pre_centroids_post_individual).all().all(),
            True)       
        
    def test_extract_data(self):
        """
        Tests extract_data().

        Uses data from munc13-snap25 project.
        """

        # params
        name = 'pre_centroids_post'
        distances = range(5, 31, 5)
        n_sim = 3
        in_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), self.raw_coloc_dir)

        # p_values=False, random_stats=False
        col = ColocAnalysis(mode='munc13')
        data, data_syn = col.extract_data(
            name=name, distances=distances, in_path=in_path, mode='munc13',
            n_sim=n_sim, p_values=False, random_stats=False)

        # test basic properties of data and data_syn
        expected_data_columns = set([
            'distance', 'n_subcol', 'n_pre_subcol', 'n_centroids_subcol',
            'n_post_subcol', 'n_pre_total', 'n_centroids_total', 'n_post_total',
            'volume', 'n_col', 'area_col', 'area', 'n_subcol_random_all',
            'n_subcol_random_alt_all', 'n_pre_subcol_random',
            'n_centroids_subcol_random', 'n_post_subcol_random'])
        np_test.assert_equal(set(data.columns), expected_data_columns)
        np_test.assert_equal(data['distance'].to_numpy(), distances)
        expected_data_columns.add('id')       
        np_test.assert_equal(set(data_syn.columns), expected_data_columns)
        np_test.assert_equal((data_syn['distance'] == 20).sum(), 23)
        np_test.assert_equal(data_syn.shape[0], 23 * len(distances))
        np_test.assert_equal(
            set(data_syn['distance'].to_numpy()), set(distances))
        np_test.assert_equal(
            (data_syn['id'].to_numpy() == 'm13_ctrl_209').sum(),
            len(distances))

        # test additions from data_syn to data
        expected_n_subcol = [24, 293, 776, 1165, 1425, 1583]
        np_test.assert_equal(data['n_subcol'].to_numpy(), expected_n_subcol)
        np_test.assert_equal(
            data_syn.groupby('distance')['n_subcol'].sum().to_numpy(),
            expected_n_subcol)
        expected_n_pre_subcol = [33, 517, 1170, 1520, 1753, 1871]
        np_test.assert_equal(
            data['n_pre_subcol'].to_numpy(), expected_n_pre_subcol)
        np_test.assert_equal(
            data_syn.groupby('distance')['n_pre_subcol'].sum().to_numpy(),
            expected_n_pre_subcol)        

        # test element-wise additions
        expected_15 = [1552, 1391, 1551]
        cond_15 = (data['distance'] == 15)
        cond_15_syn = (data_syn['distance'] == 15)
        np_test.assert_equal(
            data[cond_15]['n_post_subcol_random'].to_numpy()[0],
            expected_15)
        np_test.assert_equal(
            np.array(
                data_syn[cond_15_syn]['n_post_subcol_random'].tolist()
            ).sum(axis=0),
            expected_15)
        
        # p_values=True, random_stats=True
        col = ColocAnalysis(mode='munc13')
        data, data_syn = col.extract_data(
            name=name, distances=distances, in_path=in_path, mode='munc13',
            n_sim=n_sim, p_values=True, random_stats=True)
 
        # test basic properties of data and data_syn
        expected_data_columns = set([
            'distance', 'n_subcol', 'n_pre_subcol', 'n_centroids_subcol',
            'n_post_subcol', 'n_pre_total', 'n_centroids_total', 'n_post_total',
            'volume', 'n_col', 'area_col', 'area', 'n_subcol_random_all',
            'n_subcol_random_alt_all', 'n_pre_subcol_random',
            'n_centroids_subcol_random', 'n_post_subcol_random',
            'n_subcol_random_alt_mean', 'n_subcol_random_alt_std',
            'n_subcol_random_combined_mean', 'n_subcol_random_combined_std',
            'n_subcol_random_mean', 'n_subcol_random_std',
            'p_subcol_combined', 'p_subcol_normal', 'p_subcol_other'])
        np_test.assert_equal(set(data.columns), expected_data_columns)
        np_test.assert_equal(data['distance'].to_numpy(), distances)
        expected_data_columns.add('id')       
        np_test.assert_equal(set(data_syn.columns), expected_data_columns)
        np_test.assert_equal((data_syn['distance'] == 5).sum(), 23)
        np_test.assert_equal(data_syn.shape[0], 23 * len(distances))
        np_test.assert_equal(
            set(data_syn['distance'].to_numpy()), set(distances))
        np_test.assert_equal(
            (data_syn['id'].to_numpy() == 'snap25_ko_402').sum(),
            len(distances))

        # test additions from data_syn to data
        expected_n_subcol = [24, 293, 776, 1165, 1425, 1583]
        np_test.assert_equal(data['n_subcol'].to_numpy(), expected_n_subcol)
        np_test.assert_equal(
            data_syn.groupby('distance')['n_subcol'].sum().to_numpy(),
            expected_n_subcol)
        expected_n_centroids_subcol = [76, 529, 977, 1231, 1419, 1527]
        np_test.assert_equal(
            data['n_centroids_subcol'].to_numpy(), expected_n_centroids_subcol)
        np_test.assert_equal(
            data_syn.groupby('distance')['n_pre_subcol'].sum().to_numpy(),
            expected_n_pre_subcol)        

        # test element-wise additions
        expected_10 = [205, 201, 226]
        cond_10 = (data['distance'] == 10)
        cond_10_syn = (data_syn['distance'] == 10)
        np_test.assert_equal(
            data[cond_10]['n_pre_subcol_random'].to_numpy()[0],
            expected_10)
        np_test.assert_equal(
            np.array(
                data_syn[cond_10_syn]['n_pre_subcol_random'].tolist()
            ).sum(axis=0),
            expected_10)

        # test random stats
        cond = (data_syn['distance'] == 15)
        calculated = data_syn[cond]['n_subcol_random_alt_all'].map(
            lambda x: np.mean(x))
        np_test.assert_almost_equal(
            data_syn[cond]['n_subcol_random_alt_mean'].to_numpy(), calculated)
        calculated = data_syn[cond]['n_subcol_random_all'].map(
            lambda x: np.std(x))
        np_test.assert_almost_equal(
            data_syn[cond]['n_subcol_random_std'].to_numpy(), calculated)
        calculated = data['n_subcol_random_all'].map(lambda x: np.mean(x))
        np_test.assert_almost_equal(
            data['n_subcol_random_mean'].to_numpy(), calculated)
        calculated = data['n_subcol_random_alt_all'].map(lambda x: np.std(x))
        np_test.assert_almost_equal(
            data['n_subcol_random_alt_std'].to_numpy(), calculated)

        # test p-values
        cond = (data_syn['distance'] == 10)
        expected = [
            1., 0., 1., 0.33333333, 1.,
            1., 0., 0., 0., 0.66666667,
            0.66666667, 0.33333333, 0.66666667, 1., 0.,
            0., 0.66666667, 0.33333333, 0.33333333, 0.33333333,
            1., 1., 1.]
        np_test.assert_almost_equal(
            data_syn[cond]['p_subcol_normal'].to_numpy(), expected)
        cond = (data_syn['distance'] == 15)
        expected = [
            1., 0.66666667, 1., 0.33333333, 1.,
            1., 0.66666667, 0., 0.33333333, 1.,
            1., 1., 1., 0., 0,
            0., 1., 0., 0., 0., 0.66666667, 1., 0.66666667]
        np_test.assert_almost_equal(
            data_syn[cond]['p_subcol_other'].to_numpy(), expected)
        cond = (data_syn['distance'] == 10)
        expected = [
           1., 0, 1., 0.16666667, 0.83333333,
            0.83333333, 0., 0., 0.33333333, 0.83333333,
            0.83333333, 0.5, 0.5, 0.66666667, 0.,
            0., 0.66666667, 0.33333333, 0.66666667, 0.33333333,
            0.83333333, 1., 1.]
        np_test.assert_almost_equal(
            data_syn[cond]['p_subcol_combined'].to_numpy(), expected)

        
    def test_add_random_stats(self):
        """
        Tests add_random_stats
        """

        # input object and expected data
        data, data_tomo = common.make_coloc_tables(random_stats=False)
        coloc = ColocAnalysis()
        name = 'set3_set1_set8'
        coloc.add_data(name=name, data=data, data_syn=data_tomo)
        expected, expected_tomo = common.make_coloc_tables(random_stats=True)

        coloc.add_random_stats()
        actual, actual_tomo = coloc.get_data(name=name)
        cols = [
            'n_subcol_random_mean', 'n_subcol_random_std',
            'n_subcol_random_alt_mean', 'n_subcol_random_alt_std',
            'n_subcol_random_combined_mean', 'n_subcol_random_combined_std']
        np_test.assert_equal(
            actual_tomo[cols].round(12).equals(expected_tomo[cols].round(12)),
            True)
        np_test.assert_equal(
            actual[cols].round(12).equals(expected[cols].round(12)), True)
             
    def test_select(self):
        """
        Tests select()
        """

        # input object and expected data
        data, data_tomo = common.make_coloc_tables(random_stats=False)
        coloc = ColocAnalysis(mode='munc13')
        name = 'set3_set1_set8'
        coloc.add_data(name=name, data=data, data_syn=data_tomo)
 
        # no selection, no random
        actual, actual_tomo = coloc.select(name=name, ids=None)
        random_cols = [
            'n_subcol_random_mean', 'n_subcol_random_std',
            'n_subcol_random_alt_mean', 'n_subcol_random_alt_std',
            'n_subcol_random_combined_mean', 'n_subcol_random_combined_std']
        np_test.assert_equal(
            len(set(actual.columns.to_list()).intersection(random_cols)), 0)
        np_test.assert_equal(
            actual.round(12).equals(data[actual.columns].round(12)), True)
        np_test.assert_equal(actual_tomo.equals(data_tomo), True)
          
        # input object and expected data
        data, data_tomo = common.make_coloc_tables(random_stats=True)
        coloc = ColocAnalysis(mode='munc13')
        name = 'set3_set1_set8'
        coloc.add_data(name=name, data=data, data_syn=data_tomo)
 
       # no selection, random
        actual, actual_tomo = coloc.select(
            name=name, ids=None, random_stats=True)
        np_test.assert_equal(
            set(actual.columns.to_list()), set(data.columns.to_list()))
        np_test.assert_equal(
            actual.round(12).equals(data[actual.columns].round(12)), True)
        np_test.assert_equal(actual_tomo.equals(data_tomo), True)

        # input object and expected data
        data, data_tomo = common.make_coloc_tables(random_stats=False)
        coloc = ColocAnalysis(mode='munc13')
        name = 'set3_set1_set8'
        coloc.add_data(name=name, data=data, data_syn=data_tomo)

        # select existing tomos 
        expected, expected_tomo = common.make_coloc_tables_ac(
            random_stats=False)
        actual, actual_tomo = coloc.select(
            name=name, ids=['alpha', 'charlie'], random_stats=False)
        np_test.assert_equal(
            actual_tomo.round(12).equals(expected_tomo.round(12)), True)
        np_test.assert_equal(
            actual.round(12).equals(expected[actual.columns].round(12)), True)
        
        # select non-existing tomos 
        actual, actual_tomo = coloc.select(
            name=name, ids=['alphaaa', 'charlieee'], random_stats=False)
        np_test.assert_equal(actual is None, True)
        np_test.assert_equal(actual_tomo is None, True)

    def test_select_by_group(self):
        """
        Tests select_by_group()
        """

        # input object
        data, data_tomo = common.make_coloc_tables(random_stats=False)
        coloc = ColocAnalysis(mode='munc13')
        name = 'set3_set1_set8'
        coloc.add_data(name=name, data=data, data_syn=data_tomo)

        # define groups
        id_group = pd.DataFrame({
            'group': ['odd', 'even', 'odd'],
            'identifiers': ['alpha', 'bravo', 'charlie'],
            'something': ['a', 'b', 'c']})
        
        # group alpha, charlie
        expected, expected_tomo = common.make_coloc_tables_ac(
            random_stats=False)
        actual, actual_tomo = coloc.select_by_group(
            name=name, id_group=id_group, group_name='odd',
            group_label='group', id_label='identifiers', random_stats=False)
        np_test.assert_equal(
            actual_tomo.round(12).equals(expected_tomo.round(12)), True)
        np_test.assert_equal(
            actual.round(12).equals(expected[actual.columns].round(12)), True)
        
        # group bravo
        expected, expected_tomo = common.make_coloc_tables(
            random_stats=False)
        expected_tomo = expected_tomo[expected_tomo['id'] == 'bravo'].copy()
        actual, actual_tomo = coloc.select_by_group(
            name=name, id_group=id_group, group_name='even',
            group_label='group', id_label='identifiers', random_stats=False)
        np_test.assert_equal(
            actual_tomo.round(12).equals(expected_tomo.round(12)), True)
        cols = actual.columns.drop('dist_like')
        np_test.assert_equal(
            actual[cols].set_index('distance').round(12).equals(
                expected_tomo[cols].set_index('distance').round(12)),
            True)

        # group with non-existing tomos
        id_group = pd.DataFrame({
            'group': ['odd', 'even', 'odd'],
            'identifiers': ['alphaaa', 'bravo', 'charlieee'],
            'something': ['a', 'b', 'c']})
        actual, actual_tomo = coloc.select_by_group(
            name=name, id_group=id_group, group_name='odd',
            group_label='group', id_label='identifiers', random_stats=False)
        np_test.assert_equal(actual is None, True)
        np_test.assert_equal(actual_tomo is None, True)
        
        # group with some existing and some non-existing tomos
        id_group = pd.DataFrame({
            'group': ['odd', 'even', 'odd', 'odd'],
            'identifiers': ['alpha', 'bravo', 'charlie', 'delta'],
            'something': ['a', 'b', 'c', 'd']})
        expected, expected_tomo = common.make_coloc_tables_ac(
            random_stats=False)
        actual, actual_tomo = coloc.select_by_group(
            name=name, id_group=id_group, group_name='odd',
            group_label='group', id_label='identifiers', random_stats=False)
        np_test.assert_equal(
            actual_tomo.round(12).equals(expected_tomo.round(12)), True)
        np_test.assert_equal(
            actual.round(12).equals(expected[actual.columns].round(12)), True)
 
    def test_split_by_group(self):
        """
        Tests split_by_group()
        """

        # input object
        data, data_tomo = common.make_coloc_tables(random_stats=False)
        coloc = ColocAnalysis(mode='munc13')
        name = 'set3_set1_set8'
        coloc.add_data(name=name, data=data, data_syn=data_tomo)

        # define groups
        id_group = pd.DataFrame({
            'group': ['odd', 'even', 'odd'],
            'identifiers': ['alpha', 'bravo', 'charlie'],
            'something': ['a', 'b', 'c']})

        # expected
        expected_ac, expected_tomo_ac = common.make_coloc_tables_ac(
            random_stats=False)
        expected_b, expected_tomo_b = common.make_coloc_tables(
            random_stats=False)
        expected_tomo_b = expected_tomo_b[
            expected_tomo_b['id'] == 'bravo'].copy()

        # test
        actual, actual_tomo = coloc.split_by_groups(
            name=name, id_group=id_group, 
            group_label='group', id_label='identifiers', random_stats=False)
        np_test.assert_equal(
            actual_tomo['odd'].round(12).equals(
                expected_tomo_ac.round(12)), True)
        np_test.assert_equal(
            actual['odd'].round(12).equals(
                expected_ac[actual['odd'].columns].round(12)), True)
        np_test.assert_equal(
            actual_tomo['even'].round(12).equals(
                expected_tomo_b.round(12)), True)
        cols = actual['even'].columns.drop('dist_like')
        np_test.assert_equal(
            actual['even'][cols].set_index('distance').round(12).equals(
                expected_tomo_b[cols].set_index('distance').round(12)),
            True)
      
    def test_full_split_by_groups(self):
        """
        Tests full_split_by_groups()
        """

        # input object
        coloc = ColocAnalysis(mode='munc13')
        data, data_tomo = common.make_coloc_tables(random_stats=False)
        coloc.add_data(name='set3_set1_set8', data=data, data_syn=data_tomo)
        data, data_tomo = common.make_coloc_tables(random_stats=False)
        data_tomo['area'] = data_tomo['area'] + 1
        coloc.add_data(name='set8_set3_set1', data=data, data_syn=data_tomo)
        
        # define groups
        id_group = pd.DataFrame({
            'group': ['odd', 'even', 'odd', 'undecided'],
            'identifiers': ['alpha', 'bravo', 'charlie', 'delta'],
            'something': ['a', 'b', 'c', 'd']})

        # expected
        expected_318_ac, expected_tomo_318_ac = common.make_coloc_tables_ac(
            random_stats=False)
        expected_318_b, expected_tomo_318_b = common.make_coloc_tables(
            random_stats=False)
        expected_tomo_318_b = expected_tomo_318_b[
            expected_tomo_318_b['id'] == 'bravo'].copy()
        expected_831_ac, expected_tomo_831_ac = common.make_coloc_tables_ac(
            random_stats=False)
        expected_tomo_831_ac['area'] = expected_tomo_831_ac['area'] + 1
        expected_831_ac['area'] = expected_831_ac['area'] + 2
        expected_831_b, expected_tomo_831_b = common.make_coloc_tables(
            random_stats=False)
        expected_tomo_831_b = expected_tomo_831_b[
            expected_tomo_831_b['id'] == 'bravo'].copy()
        expected_tomo_831_b['area'] = expected_tomo_831_b['area'] + 1
        expected_831_b['area'] = expected_831_b['area'] + 1
 
        # full split colocalization
        actual = coloc.full_split_by_groups(
            id_group=id_group, 
            group_label='group', id_label='identifiers', random_stats=False)

        # test set3_set1_set8
        np_test.assert_equal(
            actual['odd'].set3_set1_set8_data_syn.round(12).equals(
                expected_tomo_318_ac.round(12)), True)
        cols = actual['odd'].set3_set1_set8_data.columns
        np_test.assert_equal(
            actual['odd'].set3_set1_set8_data.round(12).equals(
                expected_318_ac[cols].round(12)), True)
        np_test.assert_equal(
            actual['even'].set3_set1_set8_data_syn.round(12).equals(
                expected_tomo_318_b.round(12)), True)
        cols = actual['even'].set3_set1_set8_data.columns.drop('dist_like')
        actual_318_even = actual['even'].set3_set1_set8_data[cols].set_index(
            'distance')
        np_test.assert_equal(
            actual_318_even.round(12).equals(
                expected_tomo_318_b[cols].set_index('distance').round(12)),
            True)
        np_test.assert_equal(actual['undecided'] is None, True)

        # test set8_set3_set1
        np_test.assert_equal(
            actual['odd'].set8_set3_set1_data_syn.round(12).equals(
                expected_tomo_831_ac.round(12)), True)
        cols = actual['odd'].set8_set3_set1_data.columns
        np_test.assert_equal(
            actual['odd'].set8_set3_set1_data.round(12).equals(
                expected_831_ac[cols].round(12)), True)
        np_test.assert_equal(
            actual['even'].set8_set3_set1_data_syn.round(12).equals(
                expected_tomo_831_b.round(12)), True)
        cols = actual['even'].set8_set3_set1_data.columns.drop('dist_like')
        actual_831_even = actual['even'].set8_set3_set1_data[cols].set_index(
            'distance')
        np_test.assert_equal(
            actual_831_even.round(12).equals(
                expected_tomo_831_b[cols].set_index('distance').round(12)),
            True)
        
    def tearDown(self):
        """
        Remove pickled tables generated here
        """

        table_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), self.table_dir)
        try:
            file_list = os.listdir(table_dir)
        except (FileNotFoundError, OSError):
            return
        
        for file_name in file_list:
            try:
                path = os.path.join(table_dir, file_name)
                os.remove(path)
            except (FileNotFoundError, OSError):
                print("Tests fine but could not remove " + str(path))       
          
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestColocAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
