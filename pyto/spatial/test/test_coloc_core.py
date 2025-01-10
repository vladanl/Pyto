"""
Tests module coloc_core

# Author: Vladan Lucic
# $Id$
"""
__version__ = "$Revision$"

import unittest

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import numpy.testing as np_test
import pandas as pd
from pandas.testing import assert_frame_equal

from pyto.spatial.test import common
from pyto.spatial.coloc_core import ColocCore

class TestColocCore(np_test.TestCase):
    """
    Tests coloc_core module
    """

    def setUp(self):
        """
        """
        
        # to make accessible vars defined in common.make_coloc_tables_1tomo_2d()
        (self.pattern_0, self.pattern_1, self.pattern_2,
         self.dist_0_1, self.dist_0_2,
         self.pat0_pat1_pat2_data, self.pat0_pat1_data,
         self.pat0_pat2_data) = common.make_coloc_tables_1tomo_2d()
         
    def test_make(self):
        """Tests make()
        """

        # 3-coloc
        cc = ColocCore(keep_dist=True)
        distance = [2, 4, 6, 8]
        cc.make(
            patterns=[common.pattern_0, common.pattern_1, common.pattern_2],
            distance=distance, region=common.region)
        np_test.assert_array_equal(
            cc.pattern0_pattern1_pattern2_data.columns,
            common.pat0_pat1_pat2_data.columns)
        assert_frame_equal(
            cc.pattern0_pattern1_pattern2_data, common.pat0_pat1_pat2_data,
            check_dtype=False)
        np_test.assert_array_equal(
            cc.pattern0_pattern1_data.columns, common.pat0_pat1_data.columns)
        assert_frame_equal(
            cc.pattern0_pattern1_data, common.pat0_pat1_data, check_dtype=False)
        np_test.assert_array_equal(
            cc.pattern0_pattern2_data.columns, common.pat0_pat2_data.columns)
        assert_frame_equal(
            cc.pattern0_pattern2_data, common.pat0_pat2_data, check_dtype=False)
        np_test.assert_equal(len(cc.dist_nm), 3)
        np_test.assert_array_equal(cc.dist_nm[0], common.dist_0_0)
        np_test.assert_array_equal(cc.dist_nm[1], common.dist_0_1)
        np_test.assert_array_equal(cc.dist_nm[2], common.dist_0_2)
        np_test.assert_array_equal(
            cc.data_names,
            ['pattern0_pattern1_pattern2_data', 'pattern0_pattern1_data',
             'pattern0_pattern2_data'])
        
        # 2-coloc, region
        cc = ColocCore(keep_dist=True)
        distance = [2, 4, 6, 8]
        cc.make(
            patterns=[common.pattern_0, common.pattern_1], distance=distance,
            region=common.region)
        np_test.assert_array_equal(
            cc.pattern0_pattern1_data.columns, common.pat0_pat1_data.columns)
        assert_frame_equal(
            cc.pattern0_pattern1_data, common.pat0_pat1_data, check_dtype=False)
        np_test.assert_equal(len(cc.dist_nm), 2)
        np_test.assert_array_equal(cc.dist_nm[0], common.dist_0_0)
        np_test.assert_array_equal(cc.dist_nm[1], common.dist_0_1)
        np_test.assert_array_equal(cc.data_names, ['pattern0_pattern1_data'])

        # 2-coloc, names, no region
        cc = ColocCore()
        distance = [2, 4, 6, 8]
        cc.make(
            patterns=[common.pattern_0, common.pattern_2],
            distance=distance, names=['setX', 'setY'], region=None)
        np_test.assert_array_equal(
            cc.setX_setY_data.columns,
            ['distance', 'id', 'n_subcol', 'n_setX_subcol', 'n_setY_subcol',
             'n_setX_total', 'n_setY_total', 'n_col'])
        np_test.assert_array_equal(
            cc.setX_setY_data.values[:, :7], common.pat0_pat2_data.values[:, :7])
        np_test.assert_array_equal(
            cc.setX_setY_data.values[:, 7], common.pat0_pat2_data.values[:, 8])
        np_test.assert_array_equal(cc.data_names, ['setX_setY_data'])

        # 3-coloc pixel changed distance, names, region
        pixel = 0.6
        distance = np.array([2, 4, 6, 8]) * pixel
        cc = ColocCore(pixel_nm=pixel, keep_dist=True)
        cc.make(
            patterns=[common.pattern_0, common.pattern_1, common.pattern_2],
            distance=distance, names=['setX', 'setY', 'setZ'],
            region=common.region)
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data.columns,
            ['distance', 'id', 'n_subcol', 'n_setX_subcol', 'n_setY_subcol',
             'n_setZ_subcol', 'n_setX_total', 'n_setY_total', 'n_setZ_total',
             'size_region', 'n_col', 'size_col'])
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data['distance'].values,
            pixel * common.pat0_pat1_pat2_data['distance'].values)
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data.values[:, 1:],
            common.pat0_pat1_pat2_data.values[:, 1:])
        np_test.assert_equal(len(cc.dist_nm), 3)
        np_test.assert_array_almost_equal(cc.dist_nm[0], pixel*common.dist_0_0)
        np_test.assert_array_almost_equal(cc.dist_nm[1], pixel*common.dist_0_1)
        np_test.assert_array_almost_equal(cc.dist_nm[2], pixel*common.dist_0_2)
        np_test.assert_array_equal(
            cc.data_names,
            ['setX_setY_setZ_data', 'setX_setY_data', 'setX_setZ_data'])

        # 3-coloc pixel changed pattern, names, no region
        pixel = 0.5
        distance = np.array([2, 4, 6, 8])
        cc = ColocCore(pixel_nm=pixel, keep_dist=True)
        cc.make(
            patterns=[
                2 * common.pattern_0, 2 * common.pattern_1,
                2 * common.pattern_2],
            distance=distance, names=['setX', 'setY', 'setZ'], region=None)
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data.columns,
            ['distance', 'id', 'n_subcol', 'n_setX_subcol', 'n_setY_subcol',
             'n_setZ_subcol', 'n_setX_total', 'n_setY_total', 'n_setZ_total',
             'n_col'])
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data.values[:, :9],
            common.pat0_pat1_pat2_data.values[:, :9])
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data.values[:, 9],
            common.pat0_pat1_pat2_data.values[:, 10])
        np_test.assert_array_equal(
            cc.setX_setY_data.values[:, :7], common.pat0_pat1_data.values[:, :7])
        np_test.assert_array_equal(
            cc.setX_setY_data.values[:, 7], common.pat0_pat1_data.values[:, 8])
        np_test.assert_array_equal(
            cc.setX_setZ_data.values[:, :7], common.pat0_pat2_data.values[:, :7])
        np_test.assert_array_equal(
            cc.setX_setZ_data.values[:, 7], common.pat0_pat2_data.values[:, 8])
        np_test.assert_equal(len(cc.dist_nm), 3)
        np_test.assert_array_almost_equal(cc.dist_nm[0], common.dist_0_0)
        np_test.assert_array_almost_equal(cc.dist_nm[1], common.dist_0_1)
        np_test.assert_array_almost_equal(cc.dist_nm[2], common.dist_0_2)
        np_test.assert_array_equal(
            cc.data_names,
            ['setX_setY_setZ_data', 'setX_setY_data', 'setX_setZ_data'])

        # 3-coloc pixel changed pattern, names, no region
        pixel = 0.5
        distance = np.array([2, 4, 6, 8])
        cc = ColocCore(pixel_nm=pixel, keep_dist=True)
        cc.make(
            patterns=[
                2 * common.pattern_0, 2 * common.pattern_1,
                2 * common.pattern_2],
            distance=distance, names=['setX', 'setY', 'setZ'],
            region=common.region)
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data.columns,
            ['distance', 'id', 'n_subcol', 'n_setX_subcol', 'n_setY_subcol',
             'n_setZ_subcol', 'n_setX_total', 'n_setY_total', 'n_setZ_total',
             'size_region', 'n_col', 'size_col'])
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data.values[:, :-1],
            common.pat0_pat1_pat2_data.values[:, :-1])
        np_test.assert_array_equal(
            cc.setX_setY_data.values[:, :-1],
            common.pat0_pat1_data.values[:, :-1]),
        np_test.assert_array_equal(
            cc.setX_setZ_data.values[:, :-1],
            common.pat0_pat2_data.values[:, :-1]),
        np_test.assert_equal(len(cc.dist_nm), 3)
        np_test.assert_array_almost_equal(cc.dist_nm[0], common.dist_0_0)
        np_test.assert_array_almost_equal(cc.dist_nm[1], common.dist_0_1)
        np_test.assert_array_almost_equal(cc.dist_nm[2], common.dist_0_2)
        np_test.assert_array_equal(
            cc.data_names,
            ['setX_setY_setZ_data', 'setX_setY_data', 'setX_setZ_data'])

        # zero points in pattern 1
        pixel = 0.5
        distance = np.array([2, 4, 6, 8])
        cc = ColocCore(pixel_nm=pixel, keep_dist=False)
        cc.make(
            patterns=[2 * common.pattern_0, None, 2 * common.pattern_2],
            names=['setX', 'setY', 'setZ'],
            distance=distance, region=common.region)
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data.columns,
            ['distance', 'id', 'n_subcol', 'n_setX_subcol', 'n_setY_subcol',
             'n_setZ_subcol', 'n_setX_total', 'n_setY_total', 'n_setZ_total',
             'size_region', 'n_col', 'size_col'])
        zero_columns = [
            'n_subcol', 'n_setX_subcol', 'n_setY_subcol', 'n_setZ_subcol',
            'n_setY_total', 'n_col', 'size_col']
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data[zero_columns].values,
            np.zeros((len(distance), len(zero_columns))))
        tot_columns = ['n_setX_total', 'n_setZ_total']
        tot_columns_desired = ['n_pattern0_total', 'n_pattern2_total']
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data[tot_columns].values,
            common.pat0_pat1_pat2_data[tot_columns_desired].values)
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data['size_region'].values,
            np.zeros_like(distance, dtype=int) + common.size_region)       
        zero_columns = [
            'n_subcol', 'n_setX_subcol', 'n_setY_subcol', 
            'n_setY_total', 'n_col', 'size_col']
        np_test.assert_array_equal(
            cc.setX_setY_data[zero_columns].values,
            np.zeros((len(distance), len(zero_columns))))
        tot_columns = ['n_setX_total']
        tot_columns_desired = ['n_pattern0_total']
        np_test.assert_array_equal(
            cc.setX_setY_data[tot_columns].values,
            common.pat0_pat1_data[tot_columns_desired].values)        
        np_test.assert_array_equal(
            cc.setX_setY_data['size_region'].values,
            np.zeros_like(distance, dtype=int) + common.size_region)       
        np_test.assert_array_equal(
            cc.setX_setZ_data.values[:, :-1],
            common.pat0_pat2_data.values[:, :-1])
        np_test.assert_array_equal(
            cc.setX_setZ_data['size_region'].values,
            np.zeros_like(distance, dtype=int) + common.size_region)       
       
        # zero points in pattern 0
        pixel = 0.5
        distance = np.array([2, 4, 6, 8])
        cc = ColocCore(pixel_nm=pixel, keep_dist=False)
        cc.make(
            patterns=[None, 2 * common.pattern_1, 2 * common.pattern_2],
            names=['setX', 'setY', 'setZ'],
            distance=distance, region=common.region)
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data.columns,
            ['distance', 'id', 'n_subcol', 'n_setX_subcol', 'n_setY_subcol',
             'n_setZ_subcol', 'n_setX_total', 'n_setY_total', 'n_setZ_total',
             'size_region', 'n_col', 'size_col'])
        zero_columns = [
            'n_subcol', 'n_setX_subcol', 'n_setY_subcol', 'n_setZ_subcol',
            'n_setX_total', 'n_col', 'size_col']
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data[zero_columns].values,
            np.zeros((len(distance), len(zero_columns))))
        tot_columns = ['n_setY_total', 'n_setZ_total']
        tot_columns_desired = ['n_pattern1_total', 'n_pattern2_total']
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data[tot_columns].values,
            common.pat0_pat1_pat2_data[tot_columns_desired].values)
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data['size_region'].values,
            np.zeros_like(distance, dtype=int) + common.size_region)       
        zero_columns = [
            'n_subcol', 'n_setX_subcol', 'n_setY_subcol', 
            'n_setX_total', 'n_col', 'size_col']
        np_test.assert_array_equal(
            cc.setX_setY_data[zero_columns].values,
            np.zeros((len(distance), len(zero_columns))))
        tot_columns = ['n_setY_total']
        tot_columns_desired = ['n_pattern1_total']
        np_test.assert_array_equal(
            cc.setX_setY_data[tot_columns].values,
            common.pat0_pat1_data[tot_columns_desired].values)        
        np_test.assert_array_equal(
            cc.setX_setY_data['size_region'].values,
            np.zeros_like(distance, dtype=int) + common.size_region)       
        zero_columns = [
            'n_subcol', 'n_setX_subcol', 'n_setZ_subcol', 
            'n_setX_total', 'n_col', 'size_col']
        np_test.assert_array_equal(
            cc.setX_setZ_data[zero_columns].values,
            np.zeros((len(distance), len(zero_columns))))
        tot_columns = ['n_setZ_total']
        tot_columns_desired = ['n_pattern2_total']
        np_test.assert_array_equal(
            cc.setX_setZ_data[tot_columns].values,
            common.pat0_pat2_data[tot_columns_desired].values)        
        np_test.assert_array_equal(
            cc.setX_setZ_data['size_region'].values,
            np.zeros_like(distance, dtype=int) + common.size_region)       

        # 0 points in patterns 0 and 2
        pixel = 0.5
        distance = np.array([2, 4, 6, 8])
        cc = ColocCore(pixel_nm=pixel, keep_dist=False)
        cc.make(
            patterns=[np.array([]), 2 * common.pattern_1, np.array([])],
            names=['setX', 'setY', 'setZ'],
            distance=distance, region=common.region)
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data.columns,
            ['distance', 'id', 'n_subcol', 'n_setX_subcol', 'n_setY_subcol',
             'n_setZ_subcol', 'n_setX_total', 'n_setY_total', 'n_setZ_total',
             'size_region', 'n_col', 'size_col'])
        zero_columns = [
            'n_subcol', 'n_setX_subcol', 'n_setY_subcol', 'n_setZ_subcol',
            'n_setX_total', 'n_setZ_total', 'n_col', 'size_col']
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data[zero_columns].values,
            np.zeros((len(distance), len(zero_columns))))
        tot_columns = ['n_setY_total']
        tot_columns_desired = ['n_pattern1_total']
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data[tot_columns].values,
            common.pat0_pat1_pat2_data[tot_columns_desired].values)
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data['size_region'].values,
            np.zeros_like(distance, dtype=int) + common.size_region)       
        zero_columns = [
            'n_subcol', 'n_setX_subcol', 'n_setY_subcol', 
            'n_setX_total', 'n_col', 'size_col']
        np_test.assert_array_equal(
            cc.setX_setY_data[zero_columns].values,
            np.zeros((len(distance), len(zero_columns))))
        tot_columns = ['n_setY_total']
        tot_columns_desired = ['n_pattern1_total']
        np_test.assert_array_equal(
            cc.setX_setY_data[tot_columns].values,
            common.pat0_pat1_data[tot_columns_desired].values)        
        np_test.assert_array_equal(
            cc.setX_setY_data['size_region'].values,
            np.zeros_like(distance, dtype=int) + common.size_region)       
        zero_columns = [
            'n_subcol', 'n_setX_subcol', 'n_setZ_subcol', 
            'n_setX_total', 'n_setZ_total', 'n_col', 'size_col']
        np_test.assert_array_equal(
            cc.setX_setZ_data[zero_columns].values,
            np.zeros((len(distance), len(zero_columns))))
        np_test.assert_array_equal(
            cc.setX_setZ_data['size_region'].values,
            np.zeros_like(distance, dtype=int) + common.size_region)       

        # 0 points in all three patterns
        pixel = 0.5
        distance = np.array([2, 4, 6, 8])
        cc = ColocCore(pixel_nm=pixel, keep_dist=False)
        cc.make(
            patterns=[np.array([]), None, np.array([])],
            names=['setX', 'setY', 'setZ'],
            distance=distance, region=common.region)
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data.columns,
            ['distance', 'id', 'n_subcol', 'n_setX_subcol', 'n_setY_subcol',
             'n_setZ_subcol', 'n_setX_total', 'n_setY_total', 'n_setZ_total',
             'size_region', 'n_col', 'size_col'])
        zero_columns = [
            'n_subcol', 'n_setX_subcol', 'n_setY_subcol', 'n_setZ_subcol',
            'n_setX_total', 'n_setY_total', 'n_setZ_total', 'n_col', 'size_col']
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data[zero_columns].values,
            np.zeros((len(distance), len(zero_columns))))
        np_test.assert_array_equal(
            cc.setX_setY_setZ_data['size_region'].values,
            np.zeros_like(distance, dtype=int) + common.size_region)       
        zero_columns = [
            'n_subcol', 'n_setX_subcol', 'n_setY_subcol', 
            'n_setX_total', 'n_setY_total', 'n_col', 'size_col']
        np_test.assert_array_equal(
            cc.setX_setY_data[zero_columns].values,
            np.zeros((len(distance), len(zero_columns))))
        np_test.assert_array_equal(
            cc.setX_setY_data['size_region'].values,
            np.zeros_like(distance, dtype=int) + common.size_region)       
        zero_columns = [
            'n_subcol', 'n_setX_subcol', 'n_setZ_subcol', 
            'n_setX_total', 'n_setZ_total', 'n_col', 'size_col']
        np_test.assert_array_equal(
            cc.setX_setZ_data[zero_columns].values,
            np.zeros((len(distance), len(zero_columns))))
        np_test.assert_array_equal(
            cc.setX_setZ_data['size_region'].values,
            np.zeros_like(distance, dtype=int) + common.size_region)       
                        
    def test_find_columns(self):
        """Tests find_columns
        """

        distance = [2, 4, 6, 8]

        # 3-columns no region
        coloc = np.stack(
            (common.coloc3_d2, common.coloc3_d4, common.coloc3_d6,
             common.coloc3_d8), axis=0)
        cc = ColocCore()
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc3_d2, distance=2,
            col_distance=4)
        np_test.assert_equal(n_columns, common.n_columns_3[0])
        np_test.assert_equal(area is None, True)
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc3_d4, distance=4,
            col_distance=8)
        np_test.assert_equal(n_columns, common.n_columns_3[1])
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc3_d6, distance=6,
            col_distance=12)
        np_test.assert_equal(n_columns, common.n_columns_3[2])
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc3_d8, distance=8,
            col_distance=16)
        np_test.assert_equal(n_columns, common.n_columns_3[3])

        # 3-columns region
        coloc = np.stack(
            (common.coloc3_d2, common.coloc3_d4, common.coloc3_d6,
             common.coloc3_d8), axis=0)
        cc = ColocCore()
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc3_d2, distance=2,
            col_distance=4, region=common.region)
        np_test.assert_equal(n_columns, common.n_columns_3[0])
        np_test.assert_equal(area, common.size_col_3[0])
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc3_d4, distance=4,
            col_distance=8, region=common.region)
        np_test.assert_equal(n_columns, common.n_columns_3[1])
        np_test.assert_equal(area, common.size_col_3[1])
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc3_d6, distance=6,
            col_distance=12, region=common.region)
        np_test.assert_equal(n_columns, common.n_columns_3[2])
        np_test.assert_equal(area, common.size_col_3[2])
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc3_d8, distance=8,
            col_distance=16, region=common.region)
        np_test.assert_equal(n_columns, common.n_columns_3[3])
        np_test.assert_equal(area, common.size_col_3[3])

        # 3-columns region, mode 'disjoint'
        coloc = np.stack(
            (common.coloc3_d2, common.coloc3_d4, common.coloc3_d6,
             common.coloc3_d8), axis=0)
        cc = ColocCore(n_columns_mode='disjoint')
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc3_d2, distance=2,
            col_distance=4, region=common.region)
        np_test.assert_equal(n_columns, common.n_columns_3[0])
        np_test.assert_equal(area, common.size_col_3[0])
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc3_d4, distance=4,
            col_distance=8, region=common.region)
        np_test.assert_equal(n_columns, common.n_columns_3[1])
        np_test.assert_equal(area, common.size_col_3[1])
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc3_d6, distance=6,
            col_distance=12, region=common.region)
        np_test.assert_equal(n_columns, common.n_columns_3[2])
        np_test.assert_equal(area, common.size_col_3[2])
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc3_d8, distance=8,
            col_distance=16, region=common.region)
        np_test.assert_equal(n_columns, common.n_columns_3[3])
        np_test.assert_equal(area, common.size_col_3[3])

        # 3-columns region, mode 'image'
        coloc = np.stack(
            (common.coloc3_d2, common.coloc3_d4, common.coloc3_d6,
             common.coloc3_d8),
            axis=0)
        cc = ColocCore(n_columns_mode='image')
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc3_d2, distance=2,
            col_distance=4, region=common.region)
        np_test.assert_equal(n_columns, common.n_columns_3[0])
        np_test.assert_equal(area, common.size_col_3[0])
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc3_d4, distance=4,
            col_distance=8, region=common.region)
        np_test.assert_equal(n_columns, common.n_columns_3[1])
        np_test.assert_equal(area, common.size_col_3[1])
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc3_d6, distance=6,
            col_distance=12, region=common.region)
        np_test.assert_equal(n_columns, common.n_columns_3[2])
        np_test.assert_equal(area, common.size_col_3[2])
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc3_d8, distance=8,
            col_distance=16, region=common.region)
        np_test.assert_equal(n_columns, common.n_columns_3[3])
        np_test.assert_equal(area, common.size_col_3[3])

        # 2_columns from 3-columns, region
        cc = ColocCore()
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc2_d2, distance=2,
            col_distance=4, region=common.region)
        np_test.assert_equal(
            n_columns, [common.n_columns_0_1[0], common.n_columns_0_2[0]])
        np_test.assert_equal(
            area, [common.size_col_0_1[0], common.size_col_0_2[0]])
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc2_d4, distance=4,
            col_distance=8, region=common.region)
        np_test.assert_equal(
            n_columns, [common.n_columns_0_1[1], common.n_columns_0_2[1]])
        np_test.assert_equal(
            area, [common.size_col_0_1[1], common.size_col_0_2[1]])
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc2_d6, distance=6,
            col_distance=12, region=common.region)
        np_test.assert_equal(
            n_columns, [common.n_columns_0_1[2], common.n_columns_0_2[2]])
        np_test.assert_equal(
            area, [common.size_col_0_1[2], common.size_col_0_2[2]])
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc2_d8, distance=8,
            col_distance=16, region=common.region)
        np_test.assert_equal(
            n_columns, [common.n_columns_0_1[3], common.n_columns_0_2[3]])
        np_test.assert_equal(
            area, [common.size_col_0_1[3], common.size_col_0_2[3]])

        # 2_columns from 3-columns, no region
        cc = ColocCore()
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc2_d2, distance=2,
            col_distance=4)
        np_test.assert_equal(
            n_columns, [common.n_columns_0_1[0], common.n_columns_0_2[0]])
        np_test.assert_equal(area is None, True) 
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc2_d4, distance=4,
            col_distance=8)
        np_test.assert_equal(
            n_columns, [common.n_columns_0_1[1], common.n_columns_0_2[1]])
        np_test.assert_equal(area is None, True) 
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc2_d6, distance=6,
            col_distance=12)
        np_test.assert_equal(
            n_columns, [common.n_columns_0_1[2], common.n_columns_0_2[2]])
        np_test.assert_equal(area is None, True) 
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc2_d8, distance=8,
            col_distance=16)
        np_test.assert_equal(
            n_columns, [common.n_columns_0_1[3], common.n_columns_0_2[3]])
        np_test.assert_equal(area is None, True) 

        # 2-columns
        cc = ColocCore()
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc2_d2[0], distance=2,
            col_distance=4, region=common.region)
        np_test.assert_equal(n_columns, common.n_columns_0_1[0])
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc2_d4[1], distance=4,
            col_distance=8, region=common.region)
        np_test.assert_equal(n_columns, common.n_columns_0_2[1])
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc2_d6[0], distance=6,
            col_distance=12, region=common.region)
        np_test.assert_equal(n_columns, common.n_columns_0_1[2])
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=common.coloc2_d8[1], distance=8,
            col_distance=16, region=common.region)
        np_test.assert_equal(n_columns, common.n_columns_0_2[3])

        # 2-columns, no points in set 0
        cc = ColocCore()
        p0_len = len(common.pattern_0)
        coloc = np.array(p0_len * [False])
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0, coloc=coloc, distance=2,
            col_distance=4, region=common.region)
        np_test.assert_equal(n_columns, 0)
        np_test.assert_equal(area, 0)
       
        # 2-columns, no points in set 1
        cc = ColocCore()
        p0_len = len(common.pattern_0)
        coloc = np.array(p0_len * [False])
        n_columns, area = cc.find_columns(
            pattern=np.array([]), coloc=np.array([]), distance=2,
            col_distance=4, region=common.region)
        np_test.assert_equal(n_columns, 0)
        np_test.assert_equal(area, 0)

        # 3-columns no points in set 0
        cc = ColocCore()
        n_columns, area = cc.find_columns(
            pattern=np.array([]), coloc=np.array(p0_len * [False]),
            distance=2, col_distance=4, region=common.region)
        np_test.assert_equal(n_columns, 0)
        np_test.assert_equal(area, 0)

        # 3-columns no points in set 2
        cc = ColocCore()
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0,
            coloc=np.array(p0_len * [False]), distance=2,
            col_distance=4, region=common.region)
        np_test.assert_equal(n_columns, 0)
        np_test.assert_equal(area, 0)

        # 2_columns from 3-columns, no points in set 0
        # n points in sets 1 and 2 are irelevant
        cc = ColocCore()
        n_columns, area = cc.find_columns(
            pattern=np.array([]), coloc=np.array([]).reshape(2, 0),
            distance=2, col_distance=4, region=common.region)
        np_test.assert_equal(n_columns, [0, 0])
        np_test.assert_equal(area, [0, 0])

        # 2_columns from 3-columns, no points in set 1
        cc = ColocCore()
        p0_len = len(common.pattern_0)
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0,
            coloc=np.vstack((p0_len * [False], common.coloc2_d6[1])),
            distance=6, col_distance=12, region=common.region)
        np_test.assert_equal(n_columns, [0, common.n_columns_0_2[2]])
        np_test.assert_equal(area, [0, common.size_col_0_2[2]])
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0,
            coloc=np.vstack((p0_len * [False], common.coloc2_d6[1])),
            distance=8, col_distance=16, region=common.region)
        np_test.assert_equal(n_columns, [0, common.n_columns_0_2[3]])
        np_test.assert_equal(area, [0, common.size_col_0_2[3]])

        # 2_columns from 3-columns, no points in set 2
        cc = ColocCore()
        p0_len = len(common.pattern_0)
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0,
            coloc=np.vstack((common.coloc2_d6[1], p0_len * [False])),
            distance=4, col_distance=8, region=common.region)
        np_test.assert_equal(n_columns, [common.n_columns_0_1[1], 0])
        np_test.assert_equal(area, [common.size_col_0_1[1], 0])
            
        # 2_columns from 3-columns, no points in sets 1, 2
        cc = ColocCore()
        p0_len = len(common.pattern_0)
        n_columns, area = cc.find_columns(
            pattern=common.pattern_0,
            coloc=np.vstack((p0_len * [False], p0_len * [False])),
            distance=4, col_distance=8, region=common.region)
        np_test.assert_equal(n_columns, [0, 0])
        np_test.assert_equal(area, [0, 0])
            
    def test_get_column_size(self):
        """Tests get_column_size()
        """

        # 3-coloc
        cc = ColocCore()
        col_size = cc.get_column_size(
            pattern=common.pattern_0[common.coloc3_d2], distance=2,
            region=common.region)
        np_test.assert_equal(col_size, common.size_col_3[0])
        col_size = cc.get_column_size(
            pattern=common.pattern_0[common.coloc3_d4], distance=4,
            region=common.region)
        np_test.assert_equal(col_size, common.size_col_3[1])
        col_size = cc.get_column_size(
            pattern=common.pattern_0[common.coloc3_d6], distance=6,
            region=common.region)
        np_test.assert_equal(col_size, common.size_col_3[2])
        col_size = cc.get_column_size(
            pattern=common.pattern_0[common.coloc3_d8], distance=8,
            region=common.region)
        np_test.assert_equal(col_size, common.size_col_3[3])
            
        # 2-coloc
        cc = ColocCore()
        col_size = cc.get_column_size(
            pattern=common.pattern_0[common.coloc2_d2[0]], distance=2,
            region=common.region)
        np_test.assert_equal(col_size, common.size_col_0_1[0])
        col_size = cc.get_column_size(
            pattern=common.pattern_0[common.coloc2_d4[0]], distance=4,
            region=common.region)
        np_test.assert_equal(col_size, common.size_col_0_1[1])
        col_size = cc.get_column_size(
            pattern=common.pattern_0[common.coloc2_d6[0]], distance=6,
            region=common.region)
        np_test.assert_equal(col_size, common.size_col_0_1[2])
        col_size = cc.get_column_size(
            pattern=common.pattern_0[common.coloc2_d8[0]], distance=8,
            region=common.region)
        np_test.assert_equal(col_size, common.size_col_0_1[3])
            
        # 2-coloc
        cc = ColocCore()
        col_size = cc.get_column_size(
            pattern=common.pattern_0[common.coloc2_d2[1]], distance=2,
            region=common.region)
        np_test.assert_equal(col_size, 0)
        col_size = cc.get_column_size(
            pattern=common.pattern_0[common.coloc2_d4[1]], distance=4,
            region=common.region)
        np_test.assert_equal(col_size, 57)
        col_size = cc.get_column_size(
            pattern=common.pattern_0[common.coloc2_d6[1]], distance=6,
            region=common.region)
        np_test.assert_equal(col_size, 183)
        col_size = cc.get_column_size(
            pattern=common.pattern_0[common.coloc2_d8[1]], distance=8,
            region=common.region)
        np_test.assert_equal(col_size, 279)
            
        # pixel, avoid boundary, no overlap 
        pixel = 1
        cc = ColocCore(pixel_nm=pixel)
        col_size = cc.get_column_size(
            pattern=np.array([[15, 15]]), distance=2, region=common.region)
        np_test.assert_equal(col_size, 9)
        col_size = cc.get_column_size(
            pattern=np.array([[10, 10], [20, 20]]), distance=2,
            region=common.region)
        np_test.assert_equal(col_size, 2*9)
        col_size = cc.get_column_size(
            pattern=np.array([[10, 10], [20, 20]]), distance=4,
            region=common.region)
        np_test.assert_equal(col_size, 2*45)

        pixel = 2
        cc = ColocCore(pixel_nm=pixel)
        col_size = cc.get_column_size(
            pattern=np.array([[10, 10], [20, 20]]), distance=2,
            region=common.region)
        np_test.assert_equal(col_size, 2)
        col_size = cc.get_column_size(
            pattern=np.array([[10, 10], [20, 20]]), distance=4,
            region=common.region)
        np_test.assert_equal(col_size, 2*9)

        pixel = 0.6
        cc = ColocCore(pixel_nm=pixel)
        col_size = cc.get_column_size(
            pattern=np.array([[10, 10], [20, 20]]), distance=2,
            region=common.region)
        np_test.assert_equal(col_size, 2*(49 - 12))

        # less_eq
        pixel = 2
        cc = ColocCore(pixel_nm=pixel, mode='less_eq')
        col_size = cc.get_column_size(
            pattern=np.array([[10, 10], [20, 20]]), distance=2,
            region=common.region)
        np_test.assert_equal(col_size, 2*5)
        col_size = cc.get_column_size(
            pattern=np.array([[10, 10], [20, 20]]), distance=4,
            region=common.region)
        np_test.assert_equal(col_size, 2*13)

        pixel = 1.9
        cc = ColocCore(pixel_nm=pixel, mode='less')
        col_size = cc.get_column_size(
            pattern=np.array([[10, 10], [20, 20]]), distance=2,
            region=common.region)
        np_test.assert_equal(col_size, 2*5)
        col_size = cc.get_column_size(
            pattern=np.array([[10, 10], [20, 20]]), distance=4,
            region=common.region)
        np_test.assert_equal(col_size, 2*13)

        # avoid boundary, overlap
        pixel = 1
        cc = ColocCore(pixel_nm=pixel, mode='less')
        col_size = cc.get_column_size(
            pattern=np.array([[10, 10], [10, 12]]), distance=2,
            region=common.region)
        np_test.assert_equal(col_size, 2*9-3)
        col_size = cc.get_column_size(
            pattern=np.array([[10, 10], [10, 16]]), distance=4,
            region=common.region)
        np_test.assert_equal(col_size, 2*45-5)

        # overlap
        pixel = 2
        cc = ColocCore(pixel_nm=pixel, mode='less')
        col_size = cc.get_column_size(
            pattern=np.array([[10, 10], [10, 12]]), distance=4,
            region=common.region)
        np_test.assert_equal(col_size, 2*9-3)
        col_size = cc.get_column_size(
            pattern=np.array([[10, 10], [10, 16]]), distance=8,
            region=common.region)
        np_test.assert_equal(col_size, 2*45-5)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestColocCore)
    unittest.TextTestRunner(verbosity=2).run(suite)
