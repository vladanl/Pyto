"""
Tests module mps_analysis

# Author: Vladan Lucic
# $Id$
"""
__version__ = "$Revision$"

import os
import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from pyto.spatial.test import common
from pyto.spatial.mps_analysis import MPSAnalysis
from pyto.spatial.multi_particle_sets import MultiParticleSets
from pyto.spatial.test.test_particle_sets import TestParticleSets
from pyto.spatial.test.test_multi_particle_sets import TestMultiParticleSets


class TestMPSAnalysis(np_test.TestCase):
    """
    Tests mps_analysis module. Other tests are in test_multi_particle_sets.
    """

    def setUp(self):
        """
        """
        self.test_mps = TestMultiParticleSets()
        self.test_mps.setUp()

    def test_abstract_class(self):
        """Tests that MPSAnalysis is an abstract class
        """

        with np_test.assert_raises(TypeError):
            MPSAnalysis()
    
    def test_add_classification(self):
        """Tests add_classification
        """

        mps = MultiParticleSets()
        
        # default unique_cols
        parts = self.test_mps.particle_final.copy()
        cfs_1 = pd.DataFrame(
            {'tomo_id': ['alpha'], 'particle_id': [12],
             'x_orig': [131], 'class': 'three'})
        desired = parts.copy()
        desired['cl_1'] = [np.nan, 'three', np.nan, np.nan]
        parts = mps.add_classification(
            particles=parts, classification=cfs_1, class_col='cl_1',
            class_label='class', check=True, check_cols=['x_orig'])
        assert_frame_equal(parts, desired)        
        cfs_2 = pd.DataFrame(
            {'tomo_id': ['alpha', 'charlie'], 'particle_id': [13, 31]})
        desired = parts.copy()
        desired['cl_1'] = [np.nan, 'three', 'four', 'four']
        parts = mps.add_classification(
            particles=parts, classification=cfs_2, class_col='cl_1',
            class_value='four', check=False)
        assert_frame_equal(parts, desired)        
        
        # specified unique_cols
        parts = self.test_mps.particle_final.copy()
        cfs_1 = pd.DataFrame(
            {'tomo_id': ['alpha'], 'particle_id': [12],
             'x_orig': [131], 'class': 3})
        desired = parts.copy()
        desired['cl_1'] = [np.nan, 3, np.nan, np.nan]
        parts = mps.add_classification(
            particles=parts, classification=cfs_1, class_col='cl_1',
            unique_cols=['tomo_id', 'particle_id'],
            class_label='class', check=True, check_cols=['x_orig'])
        assert_frame_equal(parts, desired)        
        cfs_2 = pd.DataFrame(
            {'tomo_id': ['alpha', 'charlie'], 'particle_id': [13, 31],
             'x_orig': [137, 84]})
        desired = parts.copy()
        desired['cl_1'] = [np.nan, 3, 4, 4]
        parts = mps.add_classification(
            particles=parts, classification=cfs_2, class_col='cl_1',
            unique_cols=['tomo_id', 'particle_id'],
            class_value=4, check=True, check_cols=['x_orig'])
        assert_frame_equal(parts, desired)        
        
        # left and right unique_cols, format
        parts = self.test_mps.particle_final.copy()
        mps.particles = parts
        cfs_1 = pd.DataFrame(
            {'tomo_id_xxx': ['alpha'], 'particle_id_xxx': [12],
             'x_orig': [131], 'class': 3})
        desired = parts.copy()
        desired['cl_1'] = [np.nan, 'group_3', np.nan, np.nan]
        parts = mps.add_classification(
            classification=cfs_1, class_col='cl_1',
            left_unique_cols=['tomo_id', 'particle_id'],
            right_unique_cols=['tomo_id_xxx', 'particle_id_xxx'],
            class_label='class', class_fmt='group_{}',
            check=True, check_cols=['x_orig'], update=False)
        assert_frame_equal(parts, desired)
        mps.particles = parts
        cfs_2 = pd.DataFrame(
            {'tomo_id_xxx': ['alpha', 'charlie'], 'particle_id_xxx': [13, 31],
             'x_orig': [137, 84]})
        desired = parts.copy()
        desired['cl_1'] = [np.nan, 'group_3', 'group_4', 'group_4']
        mps.add_classification(
            classification=cfs_2, class_col='cl_1',
            left_unique_cols=['tomo_id', 'particle_id'],
            right_unique_cols=['tomo_id_xxx', 'particle_id_xxx'],
            class_value='group_4', check=True, check_cols=['x_orig'],
            update=True)
        assert_frame_equal(mps.particles, desired)        
       
    def test_reslassify(self):
        """Tests reclassify()
        """
        
        mps = MultiParticleSets()
        
        # 
        parts = self.test_mps.particle_final.copy()
        parts['class'] = [2, 4, 2, 3]
        cf = {'aa': [2], 'bb': [3, 4]}
        desired = parts.copy()
        desired['reclass'] = ['aa', 'bb', 'aa', 'bb']
        actual = mps.reclassify(
            particles=parts, classification=cf, class_col='class',
            reclass_col=['reclass'], update=False)
        assert_frame_equal(actual, desired)        
  
        # 
        parts = self.test_mps.particle_final.copy()
        parts['class'] = [2, 4, 2, 3]
        mps.particles = parts
        cf = {'aa': [2], 'bb': [4]}
        desired = parts.copy()
        desired['reclass'] = ['aa', 'bb', 'aa', np.nan]
        mps.reclassify(
            classification=cf, class_col='class',
            reclass_col=['reclass'], update=True)
        assert_frame_equal(mps.particles, desired)        

        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMPSAnalysis)
    unittest.TextTestRunner(verbosity=2).run(suite)
