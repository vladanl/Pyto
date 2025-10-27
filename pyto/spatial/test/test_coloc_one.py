"""
Tests module coloc_one

# Author: Vladan Lucic
# $Id$
"""
__version__ = "$Revision$"

from copy import copy, deepcopy
import unittest

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import numpy.testing as np_test
import pandas as pd
from pandas.testing import assert_frame_equal

from pyto.spatial.test import common
from pyto.spatial.coloc_one import ColocOne

class TestColocOne(np_test.TestCase):
    """
    Tests coloc_one module
    """

    def setUp(self):
        """
        """

        # to make accessible vars defined in common.make_coloc_tables_1tomo_2d()
        common.make_coloc_tables_1tomo_2d()

    def test_make_one(self):
        """Tests "real" (non-simulation) parts of make_one()
        """

        # 3-coloc, check columns, real data
        co = ColocOne(keep_dist=True)
        distance = [2, 4, 6, 8]
        rand_columns_common = [
            'n_subcol_random_all', 'n_subcol_random_alt_all',
            'n_subcol_random_mean', 'n_subcol_random_std',
            'n_subcol_random_alt_mean', 'n_subcol_random_alt_std', 
            'n_subcol_random_combined_mean', 'n_subcol_random_combined_std']
        rand_columns_012 = [
            'n_pattern0_subcol_random_all', 'n_pattern0_subcol_random_alt_all',
            'n_pattern1_subcol_random_all', 'n_pattern1_subcol_random_alt_all',
            'n_pattern2_subcol_random_all', 'n_pattern2_subcol_random_alt_all'] 
        rand_columns_01 = [
            'n_pattern0_subcol_random_all', 'n_pattern0_subcol_random_alt_all',
            'n_pattern1_subcol_random_all', 'n_pattern1_subcol_random_alt_all'] 
        rand_columns_02 = [
            'n_pattern0_subcol_random_all', 'n_pattern0_subcol_random_alt_all',
            'n_pattern2_subcol_random_all', 'n_pattern2_subcol_random_alt_all']
        p_columns = ['p_subcol_normal', 'p_subcol_other', 'p_subcol_combined']
        co.make_one(
            patterns=[common.pattern_0, common.pattern_1, common.pattern_2],
            distance=distance, regions=common.region, n_simul=5)
        np_test.assert_array_equal(
            co.pattern0_pattern1_pattern2_data.columns,
            list(common.pat0_pat1_pat2_data.columns) + rand_columns_common
            + rand_columns_012 + p_columns)
        np_test.assert_array_equal(
            co.pattern0_pattern1_pattern2_data.values[:, :11],
            common.pat0_pat1_pat2_data.values[:, :11]), 
        np_test.assert_array_equal(
            co.pattern0_pattern1_data.columns,
            list(common.pat0_pat1_data.columns)  + rand_columns_common
            + rand_columns_01 + p_columns)
        np_test.assert_array_equal(
            co.pattern0_pattern1_data.values[:, :10],
            common.pat0_pat1_data.values),
        np_test.assert_array_equal(
            co.pattern0_pattern2_data.columns,
            list(common.pat0_pat2_data.columns) + rand_columns_common
            + rand_columns_02 + p_columns)
        np_test.assert_array_equal(
            co.pattern0_pattern2_data.values[:, :10],
            common.pat0_pat2_data.values),
        np_test.assert_equal(len(co.dist_nm), 3)
        np_test.assert_array_equal(co.dist_nm[0], common.dist_0_0)
        np_test.assert_array_equal(co.dist_nm[1], common.dist_0_1)
        np_test.assert_array_equal(co.dist_nm[2], common.dist_0_2)
        np_test.assert_array_equal(
            co.data_names,
            ['pattern0_pattern1_pattern2_data', 'pattern0_pattern1_data',
             'pattern0_pattern2_data',
             'pattern0_pattern1_pattern2_simul_normal',
             'pattern0_pattern1_simul_normal', 'pattern0_pattern2_simul_normal',
             'pattern0_pattern1_pattern2_simul_other',
             'pattern0_pattern1_simul_other', 'pattern0_pattern2_simul_other'])
        
        # 2-coloc, check columns, real data
        co = ColocOne(keep_dist=True)
        distance = [2, 4, 6, 8]
        co.make_one(
            patterns=[common.pattern_0, common.pattern_1], distance=distance,
            regions=common.region, n_simul=5)
        np_test.assert_array_equal(
            co.pattern0_pattern1_data.columns,
            list(common.pat0_pat1_data.columns)  + rand_columns_common
            + rand_columns_01 + p_columns)
        np_test.assert_array_equal(
            co.pattern0_pattern1_data.values[:, :10],
            common.pat0_pat1_data.values)
        np_test.assert_equal(len(co.dist_nm), 2)
        np_test.assert_array_equal(co.dist_nm[1], common.dist_0_1)
        np_test.assert_array_equal(
            co.data_names,
            ['pattern0_pattern1_data', 'pattern0_pattern1_simul_normal',
             'pattern0_pattern1_simul_other'])

        # 3-coloc pixel changed distance, names, region
        pixel = 0.6
        distance = np.array([2, 4, 6, 8]) * pixel
        co = ColocOne(pixel_nm=pixel, keep_dist=True)
        co.make_one(
            patterns=[common.pattern_0, common.pattern_1, common.pattern_2],
            distance=distance, regions=common.region, n_simul=5,
            names=['setX', 'setY', 'setZ'])
        columns_common = [
            'distance', 'id', 'n_subcol', 'n_setX_subcol', 'n_setY_subcol',
            'n_setZ_subcol', 'n_setX_total', 'n_setY_total', 'n_setZ_total',
            'size_region', 'n_col', 'size_col']
        rand_columns_common = [
            'n_subcol_random_all', 'n_subcol_random_alt_all',
            'n_subcol_random_mean', 'n_subcol_random_std',
            'n_subcol_random_alt_mean', 'n_subcol_random_alt_std', 
            'n_subcol_random_combined_mean', 'n_subcol_random_combined_std']
        rand_columns_012 = [
            'n_setX_subcol_random_all', 'n_setX_subcol_random_alt_all',
            'n_setY_subcol_random_all', 'n_setY_subcol_random_alt_all',
            'n_setZ_subcol_random_all', 'n_setZ_subcol_random_alt_all'] 
        rand_columns_01 = [
            'n_setX_subcol_random_all', 'n_setX_subcol_random_alt_all',
            'n_setY_subcol_random_all', 'n_setY_subcol_random_alt_all'] 
        rand_columns_02 = [
            'n_setX_subcol_random_all', 'n_setX_subcol_random_alt_all',
            'n_setZ_subcol_random_all', 'n_setZ_subcol_random_alt_all'] 
        np_test.assert_array_equal(
            co.setX_setY_setZ_data.columns,
            columns_common + rand_columns_common + rand_columns_012 + p_columns)
        np_test.assert_array_equal(
            co.setX_setY_setZ_data['distance'].values,
            pixel * common.pat0_pat1_pat2_data['distance'].values)
        np_test.assert_array_equal(
            co.setX_setY_setZ_data.values[:, 1:10],
            common.pat0_pat1_pat2_data.values[:, 1:10])
        np_test.assert_equal(len(co.dist_nm), 3)
        np_test.assert_array_almost_equal(co.dist_nm[0], pixel*common.dist_0_0)
        np_test.assert_array_almost_equal(co.dist_nm[1], pixel*common.dist_0_1)
        np_test.assert_array_almost_equal(co.dist_nm[2], pixel*common.dist_0_2)
        np_test.assert_array_equal(
            co.data_names,
            ['setX_setY_setZ_data', 'setX_setY_data', 'setX_setZ_data',
             'setX_setY_setZ_simul_normal', 'setX_setY_simul_normal',
             'setX_setZ_simul_normal',
             'setX_setY_setZ_simul_other', 'setX_setY_simul_other',
             'setX_setZ_simul_other'])

        # 3-coloc pixel changed pattern, names
        pixel = 0.5
        distance = np.array([2, 4, 6, 8])
        co = ColocOne(pixel_nm=pixel, keep_dist=True)
        co.make_one(
            patterns=[
                2 * common.pattern_0, 2 * common.pattern_1,
                2 * common.pattern_2],
            distance=distance, names=['setX', 'setY', 'setZ'],
            regions=common.region, n_simul=4)
        np_test.assert_array_equal(
            co.setX_setY_setZ_data.columns[:11],
            ['distance', 'id', 'n_subcol', 'n_setX_subcol', 'n_setY_subcol',
             'n_setZ_subcol', 'n_setX_total', 'n_setY_total', 'n_setZ_total',
             'size_region', 'n_col'])
        np_test.assert_array_equal(
            co.setX_setY_setZ_data.values[:, :10],
            common.pat0_pat1_pat2_data.values[:, :10])
        np_test.assert_array_equal(
            co.setX_setY_data.values[:, :8], common.pat0_pat1_data.values[:, :8]),
        np_test.assert_array_equal(
            co.setX_setZ_data.values[:, :8], common.pat0_pat2_data.values[:, :8]),
        np_test.assert_equal(len(co.dist_nm), 3)
        np_test.assert_array_almost_equal(co.dist_nm[0], common.dist_0_0)
        np_test.assert_array_almost_equal(co.dist_nm[1], common.dist_0_1)
        np_test.assert_array_almost_equal(co.dist_nm[2], common.dist_0_2)

        # 3-coloc pixel changed pattern, names, 
        pixel = 0.5
        distance = np.array([2, 4, 6, 8])
        co = ColocOne(pixel_nm=pixel, keep_dist=True)
        co.make_one(
            patterns=[
                2 * common.pattern_0, 2 * common.pattern_1,
                2 * common.pattern_2],
            regions=common.region, distance=distance,
            names=['setX', 'setY', 'setZ'], n_simul=6)
        np_test.assert_array_equal(
            co.setX_setY_setZ_data.columns[:12],
            ['distance', 'id', 'n_subcol', 'n_setX_subcol', 'n_setY_subcol',
             'n_setZ_subcol', 'n_setX_total', 'n_setY_total', 'n_setZ_total',
             'size_region', 'n_col', 'size_col'])
        np_test.assert_array_equal(
            co.setX_setY_setZ_data.values[:, :10],
            common.pat0_pat1_pat2_data.values[:, :10])
        np_test.assert_array_equal(
            co.setX_setY_data.values[:, :8],
            common.pat0_pat1_data.values[:, :8]),
        np_test.assert_array_equal(
            co.setX_setZ_data.values[:, :8],
            common.pat0_pat2_data.values[:, :8]),
        np_test.assert_equal(len(co.dist_nm), 3)
        np_test.assert_array_almost_equal(co.dist_nm[0], common.dist_0_0)
        np_test.assert_array_almost_equal(co.dist_nm[1], common.dist_0_1)
        np_test.assert_array_almost_equal(co.dist_nm[2], common.dist_0_2)

        # 3-coloc, no particles in pattern 1, check columns, real data
        co = ColocOne(keep_dist=True)
        distance = [2, 4, 6, 8]
        rand_columns_common = [
            'n_subcol_random_all', 'n_subcol_random_alt_all',
            'n_subcol_random_mean', 'n_subcol_random_std',
            'n_subcol_random_alt_mean', 'n_subcol_random_alt_std', 
            'n_subcol_random_combined_mean', 'n_subcol_random_combined_std']
        rand_columns_012 = [
            'n_pattern0_subcol_random_all', 'n_pattern0_subcol_random_alt_all',
            'n_pattern1_subcol_random_all', 'n_pattern1_subcol_random_alt_all',
            'n_pattern2_subcol_random_all', 'n_pattern2_subcol_random_alt_all'] 
        rand_columns_01 = [
            'n_pattern0_subcol_random_all', 'n_pattern0_subcol_random_alt_all',
            'n_pattern1_subcol_random_all', 'n_pattern1_subcol_random_alt_all'] 
        rand_columns_02 = [
            'n_pattern0_subcol_random_all', 'n_pattern0_subcol_random_alt_all',
            'n_pattern2_subcol_random_all', 'n_pattern2_subcol_random_alt_all']
        p_columns = ['p_subcol_normal', 'p_subcol_other', 'p_subcol_combined']
        co.make_one(
            patterns=[common.pattern_0, None, common.pattern_2],
            distance=distance, regions=common.region, n_simul=5)
        np_test.assert_array_equal(
            co.pattern0_pattern1_pattern2_data.columns,
            list(common.pat0_pat1_pat2_data.columns) + rand_columns_common
            + rand_columns_012 + p_columns)
        zero_cols = [
            'n_subcol', 'n_pattern0_subcol', 'n_pattern1_subcol',
            'n_pattern2_subcol',
            'n_pattern1_total', 'n_col', 'size_col']
        np_test.assert_array_equal(
            co.pattern0_pattern1_pattern2_data[zero_cols].values,
            np.zeros((len(distance), len(zero_cols))))
        tot_cols = ['n_pattern0_total', 'n_pattern2_total']
        np_test.assert_array_equal(
            co.pattern0_pattern1_pattern2_data[tot_cols].values,
            common.pat0_pat1_pat2_data[tot_cols].values)
        
        np_test.assert_array_equal(
            co.pattern0_pattern1_data.columns,
            list(common.pat0_pat1_data.columns)  + rand_columns_common
            + rand_columns_01 + p_columns)
        zero_cols = [
            'n_subcol', 'n_pattern0_subcol', 'n_pattern1_subcol',
            'n_pattern1_total', 'n_col', 'size_col']
        np_test.assert_array_equal(
            co.pattern0_pattern1_data[zero_cols].values,
            np.zeros((len(distance), len(zero_cols))))
        tot_cols = ['n_pattern0_total']
        np_test.assert_array_equal(
            co.pattern0_pattern1_data[tot_cols].values,
            common.pat0_pat1_data[tot_cols].values)

        np_test.assert_array_equal(
            co.pattern0_pattern2_data.columns,
            list(common.pat0_pat2_data.columns) + rand_columns_common
            + rand_columns_02 + p_columns)
        np_test.assert_array_equal(
            co.pattern0_pattern2_data.values[:, :10],
            common.pat0_pat2_data.values),
        np_test.assert_array_equal(
            co.data_names,
            ['pattern0_pattern1_pattern2_data', 'pattern0_pattern1_data',
             'pattern0_pattern2_data',
             'pattern0_pattern1_pattern2_simul_normal',
             'pattern0_pattern1_simul_normal', 'pattern0_pattern2_simul_normal',
             'pattern0_pattern1_pattern2_simul_other',
             'pattern0_pattern1_simul_other', 'pattern0_pattern2_simul_other'])
        
        # 3-coloc, no particles in pattern 0, check columns, real data
        co = ColocOne(keep_dist=True)
        distance = [2, 4, 6, 8]
        rand_columns_common = [
            'n_subcol_random_all', 'n_subcol_random_alt_all',
            'n_subcol_random_mean', 'n_subcol_random_std',
            'n_subcol_random_alt_mean', 'n_subcol_random_alt_std', 
            'n_subcol_random_combined_mean', 'n_subcol_random_combined_std']
        rand_columns_012 = [
            'n_pattern0_subcol_random_all', 'n_pattern0_subcol_random_alt_all',
            'n_pattern1_subcol_random_all', 'n_pattern1_subcol_random_alt_all',
            'n_pattern2_subcol_random_all', 'n_pattern2_subcol_random_alt_all'] 
        rand_columns_01 = [
            'n_pattern0_subcol_random_all', 'n_pattern0_subcol_random_alt_all',
            'n_pattern1_subcol_random_all', 'n_pattern1_subcol_random_alt_all'] 
        rand_columns_02 = [
            'n_pattern0_subcol_random_all', 'n_pattern0_subcol_random_alt_all',
            'n_pattern2_subcol_random_all', 'n_pattern2_subcol_random_alt_all']
        p_columns = ['p_subcol_normal', 'p_subcol_other', 'p_subcol_combined']
        co.make_one(
            patterns=[np.array([]), common.pattern_1, common.pattern_2],
            distance=distance, regions=common.region, n_simul=5)
        np_test.assert_array_equal(
            co.pattern0_pattern1_pattern2_data.columns,
            list(common.pat0_pat1_pat2_data.columns) + rand_columns_common
            + rand_columns_012 + p_columns)
        zero_cols = [
            'n_subcol', 'n_pattern0_subcol', 'n_pattern1_subcol',
            'n_pattern2_subcol',
            'n_pattern0_total', 'n_col', 'size_col']
        np_test.assert_array_equal(
            co.pattern0_pattern1_pattern2_data[zero_cols].values,
            np.zeros((len(distance), len(zero_cols))))
        tot_cols = ['n_pattern1_total', 'n_pattern2_total']
        np_test.assert_array_equal(
            co.pattern0_pattern1_pattern2_data[tot_cols].values,
            common.pat0_pat1_pat2_data[tot_cols].values)

        np_test.assert_array_equal(
            co.pattern0_pattern1_data.columns,
            list(common.pat0_pat1_data.columns)  + rand_columns_common
            + rand_columns_01 + p_columns)
        zero_cols = [
            'n_subcol', 'n_pattern0_subcol', 'n_pattern1_subcol',
            'n_pattern0_total', 'n_col', 'size_col']
        np_test.assert_array_equal(
            co.pattern0_pattern1_data[zero_cols].values,
            np.zeros((len(distance), len(zero_cols))))
        tot_cols = ['n_pattern1_total']
        np_test.assert_array_equal(
            co.pattern0_pattern1_data[tot_cols].values,
            common.pat0_pat1_data[tot_cols].values)

        np_test.assert_array_equal(
            co.pattern0_pattern2_data.columns,
            list(common.pat0_pat2_data.columns)  + rand_columns_common
            + rand_columns_02 + p_columns)
        zero_cols = [
            'n_subcol', 'n_pattern0_subcol', 'n_pattern2_subcol',
            'n_pattern0_total', 'n_col', 'size_col']
        np_test.assert_array_equal(
            co.pattern0_pattern2_data[zero_cols].values,
            np.zeros((len(distance), len(zero_cols))))
        tot_cols = ['n_pattern2_total']
        np_test.assert_array_equal(
            co.pattern0_pattern2_data[tot_cols].values,
            common.pat0_pat2_data[tot_cols].values)

    def test_make_one_simul(self):
        """Tests simulations part of make_one()
        """

        # 3-coloc
        distance = [2, 4, 6, 8]
        n_simul = 100
        co = ColocOne(keep_dist=False)
        co.make_one(
            patterns=[common.pattern_0, common.pattern_1, common.pattern_2],
            distance=distance, regions=common.region, n_simul=n_simul,
            names=['X', 'Y', 'Z'])

        # normal
        np_test.assert_equal(
            co.X_Y_Z_data[['n_subcol_random_all']].map(
                lambda x: x.shape[0] == n_simul).values.all(), True)
        np_test.assert_array_almost_equal(
            co.X_Y_Z_data[['n_subcol_random_mean']],
            co.X_Y_Z_data[['n_subcol_random_all']].map(
                lambda x: x.mean()).values)
        np_test.assert_array_almost_equal(
            co.X_Y_Z_data[['n_subcol_random_std']],
            co.X_Y_Z_data[['n_subcol_random_all']].map(
                lambda x: x.std(ddof=1)).values)
        np_test.assert_array_almost_equal(
            co.X_Y_Z_data['p_subcol_normal'].values,
            (np.vstack(co.X_Y_Z_data['n_subcol_random_all'].values)
             < np.vstack(co.X_Y_Z_data['n_subcol'].values)
             ).mean(axis=1))

        # other
        np_test.assert_equal(
            co.X_Y_Z_data[['n_subcol_random_alt_all']].map(
                lambda x: x.shape[0] == n_simul).values.all(), True)
        np_test.assert_array_almost_equal(
            co.X_Y_Z_data[['n_subcol_random_alt_mean']],
            co.X_Y_Z_data[['n_subcol_random_alt_all']].map(
                lambda x: x.mean()).values)
        np_test.assert_array_almost_equal(
            co.X_Y_Z_data[['n_subcol_random_alt_std']],
            co.X_Y_Z_data[['n_subcol_random_alt_all']].map(
                lambda x: x.std(ddof=1)).values)
        np_test.assert_array_almost_equal(
            co.X_Y_Z_data['p_subcol_other'].values,
            (np.vstack(co.X_Y_Z_data['n_subcol_random_alt_all'].values)
             < np.vstack(co.X_Y_Z_data['n_subcol'].values)
             ).mean(axis=1))

        # combined
        np_test.assert_array_almost_equal(
            co.X_Y_Z_data['n_subcol_random_combined_mean'].values,
            np.concatenate(
                (np.vstack(co.X_Y_Z_data['n_subcol_random_all'].values),
                 np.vstack(co.X_Y_Z_data['n_subcol_random_alt_all'].values)),
                axis=1).mean(axis=1))
        np_test.assert_array_almost_equal(
            co.X_Y_Z_data['n_subcol_random_combined_std'].values,
            np.concatenate(
                (np.vstack(co.X_Y_Z_data['n_subcol_random_all'].values),
                 np.vstack(co.X_Y_Z_data['n_subcol_random_alt_all'].values)),
                axis=1).std(axis=1, ddof=1))
        np_test.assert_array_almost_equal(
            co.X_Y_Z_data['p_subcol_combined'].values,
            (np.concatenate(
                [np.vstack(co.X_Y_Z_data['n_subcol_random_all'].values),
                 np.vstack(co.X_Y_Z_data['n_subcol_random_alt_all'].values)],
                axis=1)
             < np.vstack(co.X_Y_Z_data['n_subcol'].values)
             ).mean(axis=1))

        # p >=, check p-values
        n_simul = 100
        co = ColocOne(keep_dist=False, p_func=np.greater_equal)
        co.make_one(
            patterns=[common.pattern_0, common.pattern_1, common.pattern_2],
            distance=distance, regions=common.region, n_simul=n_simul,
            names=['X', 'Y', 'Z'])
        np_test.assert_array_almost_equal(
            co.X_Y_Z_data['p_subcol_normal'].values,
            (np.vstack(co.X_Y_Z_data['n_subcol_random_all'].values)
             <= np.vstack(co.X_Y_Z_data['n_subcol'].values)
             ).mean(axis=1))
        np_test.assert_array_almost_equal(
            co.X_Y_Z_data['p_subcol_combined'].values,
            (np.concatenate(
                [np.vstack(co.X_Y_Z_data['n_subcol_random_all'].values),
                 np.vstack(co.X_Y_Z_data['n_subcol_random_alt_all'].values)],
                axis=1)
             <= np.vstack(co.X_Y_Z_data['n_subcol'].values)
             ).mean(axis=1))

        # 3-coloc, no point in pattern 2
        distance = [2, 4, 6, 8]
        n_simul = 100
        co = ColocOne(keep_dist=False)
        co.make_one(
            patterns=[common.pattern_0, common.pattern_1, None],
            distance=distance, regions=common.region, n_simul=n_simul,
            names=['X', 'Y', 'Z'])

        # normal
        np_test.assert_array_equal(
            np.hstack(co.X_Y_Z_data['n_subcol_random_all'].values),
            np.zeros(shape=(len(distance) * n_simul,), dtype=int))
        np_test.assert_array_almost_equal(
            co.X_Y_Z_data['n_subcol_random_mean'].values,
            np.zeros(shape=len(distance), dtype=int))
        np_test.assert_array_almost_equal(
            co.X_Y_Z_data['n_subcol_random_std'].values,
            np.zeros(shape=len(distance), dtype=int))        
        np_test.assert_array_equal(
            np.hstack(co.X_Y_Z_data['n_subcol_random_alt_all'].values),
            np.zeros(shape=(len(distance) * n_simul,), dtype=int))
        np_test.assert_array_almost_equal(
            co.X_Y_Z_data['n_subcol_random_alt_mean'].values,
            np.zeros(shape=len(distance), dtype=int))
        np_test.assert_array_almost_equal(
            co.X_Y_Z_data['n_subcol_random_alt_std'].values,
            np.zeros(shape=len(distance), dtype=int))
        np_test.assert_array_almost_equal(
            co.X_Y_Z_data['n_subcol_random_combined_mean'].values,
            np.zeros(shape=len(distance), dtype=int))
        np_test.assert_array_almost_equal(
            co.X_Y_Z_data['n_subcol_random_combined_std'].values,
            np.zeros(shape=len(distance), dtype=int))

    def test_add_coloc(self):
        """Tests add_coloc()
        """
        
        co = ColocOne()
        co1 = ColocOne()
        co1.pat0_pat1_pat2_data = common.pat0_pat1_pat2_data
        co1.pat0_pat1_data = common.pat0_pat1_data
        co1.pat0_pat2_data = common.pat0_pat2_data
        co1.data_names = [
            'pat0_pat1_pat2_data', 'pat0_pat1_data', 'pat0_pat2_data']
        n_distances = common.pat0_pat1_pat2_data.shape[0]
        co.add_coloc(coloc=co1, extra={'add_id': 0})
        assert_frame_equal(co.pat0_pat1_pat2_data, co1.pat0_pat1_pat2_data)
        assert_frame_equal(co.pat0_pat1_data, co1.pat0_pat1_data)
        assert_frame_equal(co.pat0_pat2_data, co1.pat0_pat2_data)
        np_test.assert_array_equal(
            co.pat0_pat1_pat2_data['add_id'].values,
            n_distances * [0])
       
        co2 = deepcopy(co1)
        co2.pat0_pat1_pat2_data['n_subcol'] = (
            10 * co2.pat0_pat1_pat2_data['n_subcol'])
        co.add_coloc(coloc=co2, extra={'add_id': 1})
        np_test.assert_equal(
            co.pat0_pat1_pat2_data.shape[0],
            2 * common.pat0_pat1_pat2_data.shape[0])
        np_test.assert_equal(
            co.pat0_pat2_data.shape[0], 2 * common.pat0_pat2_data.shape[0])
        np_test.assert_array_equal(
            co.pat0_pat1_pat2_data['n_subcol'].values,
            np.hstack((
                common.pat0_pat1_pat2_data['n_subcol'].values,
                10 * common.pat0_pat1_pat2_data['n_subcol'].values)))
        np_test.assert_array_equal(
            co.pat0_pat1_pat2_data['add_id'].values,
            n_distances * [0] + n_distances * [1])
            
        co3 = deepcopy(co1)
        co3.pat0_pat2_data['n_subcol'] = (
            20 * co2.pat0_pat2_data['n_subcol'])
        co.add_coloc(coloc=co3, extra={'add_id': 2})
        np_test.assert_equal(
            co.pat0_pat1_pat2_data.shape[0],
            3 * common.pat0_pat1_pat2_data.shape[0])
        np_test.assert_array_equal(
            co.pat0_pat2_data['n_subcol'].values,
            np.hstack((
                common.pat0_pat2_data['n_subcol'].values,
                common.pat0_pat2_data['n_subcol'].values,
                20 * common.pat0_pat2_data['n_subcol'].values)))
        np_test.assert_array_equal(
            co.pat0_pat1_pat2_data['add_id'].values,
            n_distances * [0] + n_distances * [1] + n_distances * [2])
         
    def test_combine_simulations(self):
        """Tests combine_simulations()
        """
        
        xyz = pd.DataFrame(
            {'distance': [2, 5], 'n_subcol': 3,
             'n_setX_subcol': 4,  'n_setY_subcol': 5, 'n_setZ_subcol': 6,
             'n_setX_total': 7, 'n_setY_total': 8, 'n_setZ_total': 9})
        xyz_cols = [
            'distance', 'n_subcol',
            'n_setX_subcol', 'n_setY_subcol', 'n_setZ_subcol',
            'n_setX_total', 'n_setY_total', 'n_setZ_total']
        xy_cols = [
            'distance', 'n_subcol', 'n_setX_subcol', 'n_setY_subcol',
            'n_setX_total', 'n_setY_total']
        xz_cols = [
            'distance', 'n_subcol', 'n_setX_subcol', 'n_setZ_subcol',
            'n_setX_total', 'n_setZ_total']
        xyz_sim =  pd.DataFrame(
            {'distance': [2, 5, 2, 5, 2, 5], 'simul_id': [0, 0, 1, 1, 2, 2],
             'n_subcol': 13,
             'n_setX_subcol': 14,  'n_setY_subcol': 15, 'n_setZ_subcol': 16,
             'n_setX_total': 17, 'n_setY_total': 18, 'n_setZ_total': 19})
            
        co = ColocOne(suffix='dat')
        co.simul_suffixes = ['si_nor', 'si_ot']
        co.random_suffixes = ['rand', 'rand_alt', 'rand_comb']
        co.setX_setY_setZ_dat = xyz
        co.setX_setY_dat = xyz[xy_cols]
        co.setX_setZ_dat = xyz[xz_cols]
        co.setX_setY_setZ_si_nor = xyz_sim
        co.setX_setY_si_nor = xyz_sim[xy_cols]
        co.setX_setZ_si_nor = xyz_sim[xz_cols]
        co.setX_setY_setZ_si_ot = xyz_sim
        co.setX_setY_si_ot = xyz_sim[xy_cols]
        co.setX_setZ_si_ot = xyz_sim[xz_cols]
        co.data_names = [
            'setX_setY_setZ_dat', 'setX_setY_dat', 'setX_setZ_dat', 
            'setX_setY_setZ_si_nor', 'setX_setY_si_nor', 'setX_setZ_si_nor', 
            'setX_setY_setZ_si_ot', 'setX_setY_si_ot', 'setX_setZ_si_ot']
        random_cols = [
            'n_subcol_rand_all', 'n_subcol_rand_alt_all',
            'n_subcol_rand_mean', 'n_subcol_rand_std',
            'n_subcol_rand_alt_mean', 'n_subcol_rand_alt_std',
            'n_subcol_rand_comb_mean', 'n_subcol_rand_comb_std']
        random_cols_xyz = [
            'n_setX_subcol_rand_all', 'n_setX_subcol_rand_alt_all',
            'n_setY_subcol_rand_all', 'n_setY_subcol_rand_alt_all',
            'n_setZ_subcol_rand_all', 'n_setZ_subcol_rand_alt_all'] 
        random_cols_xy = [
            'n_setX_subcol_rand_all', 'n_setX_subcol_rand_alt_all',
            'n_setY_subcol_rand_all', 'n_setY_subcol_rand_alt_all'] 
        random_cols_xz = [
            'n_setX_subcol_rand_all', 'n_setX_subcol_rand_alt_all',
            'n_setZ_subcol_rand_all', 'n_setZ_subcol_rand_alt_all'] 
        #desired_xyz_cols = (
        #    xyz_cols +  
            
        co.combine_simulations()
        np_test.assert_array_equal(
            co.setX_setY_setZ_dat.columns,
            xyz_cols + random_cols + random_cols_xyz)
        np_test.assert_array_equal(
            co.setX_setY_dat.columns,
            xy_cols + random_cols + random_cols_xy)
        np_test.assert_array_equal(
            co.setX_setZ_dat.columns,
            xz_cols + random_cols + random_cols_xz)

    def test_combine_simulations_one(self):
        """Tests combine_simulations_one() and using custom suffixes
        """

        data = pd.DataFrame(
            {'distance': [2, 4, 6], 'n_foo': [1, 2, 3], 'x': 9, 'y': 19})
        simul_normal = pd.DataFrame(
            {'distance': [2, 4, 6, 2, 4, 6, 2, 4, 6, 2, 4, 6],
             'n_foo': [0, 10, 20, 1, 12, 21, 3, 14, 23, 4, 16, 24],
             'x': 12, 'y': 23})
        simul_other = pd.DataFrame(
            {'distance': [2, 4, 6, 2, 4, 6, 2, 4, 6, 2, 4, 6, 2, 4, 6],
             'n_foo': [8, 16, 24, 6, 14, 21, 0, 12, 23, 2, 10, 20, 2, 10, 20],
             'x': 15, 'y': 29})
        mean_normal = [2, 13, 22]
        std_normal = np.std(
            simul_normal['n_foo'].values.reshape(4, 3), axis=0, ddof=1)
        mean_other = [18/5, 62/5, 108/5]
        std_other = np.std(
            simul_other['n_foo'].values.reshape(5, 3), axis=0, ddof=1)
        mean_combined = [26/9, 114/9, 196/9]
        std_combined = np.std(
            np.concatenate(
                [simul_normal['n_foo'].values.reshape(4, 3),
                 simul_other['n_foo'].values.reshape(5, 3)],
                axis=0),
            axis=0, ddof=1)
        
        co = ColocOne()
        co.x_y_data = data
        co.x_y_simul_normal = simul_normal
        co.x_y_simul_other = simul_other
        co.data_names = [
            'x_y_data', f'x_y_data_{co.simul_suffixes[0]}',
             f'x_y_data_{co.simul_suffixes[1]}']

        tab = co.combine_simulations_one(
            tab=data, tab_simul_normal=simul_normal,
            tab_simul_other=simul_other, column='n_foo', stats=False)
        np_test.assert_array_equal(
            tab.columns,
            ['distance', 'n_foo', 'x', 'y', 'n_foo_random_all',
             'n_foo_random_alt_all'])
        
        tab = co.combine_simulations_one(
            tab=data, tab_simul_normal=simul_normal,
            tab_simul_other=simul_other, column='n_foo', stats=True)
        np_test.assert_array_equal(
            tab.columns,
            ['distance', 'n_foo', 'x', 'y', 'n_foo_random_all',
             'n_foo_random_alt_all',
             'n_foo_random_mean', 'n_foo_random_std',
             'n_foo_random_alt_mean', 'n_foo_random_alt_std',
             'n_foo_random_combined_mean', 'n_foo_random_combined_std'
             ])
        np_test.assert_array_almost_equal(
            tab['n_foo_random_mean'].values, mean_normal)
        np_test.assert_array_almost_equal(
            tab['n_foo_random_std'].values, std_normal)
        np_test.assert_array_almost_equal(
            tab['n_foo_random_alt_mean'].values, mean_other)
        np_test.assert_array_almost_equal(
             tab['n_foo_random_alt_std'].values, std_other)
        np_test.assert_array_almost_equal(
             tab['n_foo_random_combined_mean'].values, mean_combined)
        np_test.assert_array_almost_equal(
             tab['n_foo_random_combined_std'].values, std_combined)
      


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestColocOne)
    unittest.TextTestRunner(verbosity=2).run(suite)
