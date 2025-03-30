"""
Tests module extract_mps

# Author: Vladan Lucic
# $Id:$
"""
__version__ = "$Revision:$"

import os
import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from pyto.particles.extract_mps_filter import ExtractMPSFilter
from pyto.particles.extract_mps import ExtractMPS
from pyto.spatial.multi_particle_sets import MultiParticleSets


class TestExtractMPS(np_test.TestCase):
    """
    Tests extract_mps module. 
    """

    def setUp(self):
        """
        """
        self.ex_mps = ExtractMPS(
            normal_source_index_col='index_src', normal_source_suffix='_src')

        self.mps = MultiParticleSets()
        self.mps.tomos = pd.DataFrame()
        self.mps.particles = pd.DataFrame({
            'group': 'group_1', 'tomo_id': ['alfa', 'bravo'],
            'particle_id': [1, 2],
            'x': [48, 70], 'y': [140, 170], 'z': [240, 263]}, index=[3, 4])  
        self.source = MultiParticleSets()
        self.source.tomos = pd.DataFrame()
        self.source.particles = pd.DataFrame({
            'group': 'group_1', 'tomo_id': ['alfa', 'alpha', 'bravo', 'bravo'],
            'particle_id': [11, 12, 21, 22],
            'xx': [40, 60, 50, 70], 'yy': [140, 160, 150, 170],
            'zz': [240, 260, 250, 270],
            'rlnAngleTilt': [10, 20, 30, 40],
            'rlnAngleTiltPrior': [15, 25, 35, 45],
            'rlnAnglePsi': [50, 60, 70, 80],
            'rlnAnglePsiPrior': [54, 64, 74, 84],
            'rlnAngleRot': [110, 120, 130, 140]}, index=[100, 101, 102, 103])  

    def test_set_normals(self):
        """Test set_normals()
        """

        # no reverse, no priors
        desired_source = pd.DataFrame(
            {'index_src': [100, 103], 'distance': [8., 7],
             'rlnAngleTilt': [10, 40], 'rlnAngleTiltPrior': [15., 45],
             'rlnAnglePsi': [50., 80], 'rlnAnglePsiPrior': [54., 84],
             'rlnAngleRot': [110, 140.], 'particle_id_src': [11, 22],
             'normal_theta': [10, 40.], 'normal_phi': [130, 100.]},
            index=[3, 4])
        desired = pd.concat([self.mps.particles, desired_source], axis=1)
        part_2 = self.ex_mps.set_normals(
            mps=self.mps, source=self.source, mps_coord_cols=['x', 'y', 'z'],
            source_coord_cols=['xx', 'yy', 'zz'], reverse=False,
            use_priors=False)
        assert_frame_equal(part_2, desired, check_dtype=False)
        
        # no reverse, priors
        desired_source['normal_theta'] = [15, 45.]
        desired_source['normal_phi'] = [126, 96.]
        desired = pd.concat([self.mps.particles, desired_source], axis=1)
        part_2 = self.ex_mps.set_normals(
            mps=self.mps, source=self.source, mps_coord_cols=['x', 'y', 'z'],
            source_coord_cols=['xx', 'yy', 'zz'], reverse=False,
            use_priors=True)
        assert_frame_equal(part_2, desired, check_dtype=False)

        # reverse, no priors
        desired_source = pd.DataFrame(
            {'index_src': [100, 103], 'distance': [8., 7],
             'rlnAngleTilt': [170, 140], 'rlnAngleTiltPrior': [165., 135],
             'rlnAnglePsi': [230., 260], 'rlnAnglePsiPrior': [234., 264],
             'rlnAngleRot': [290, 320.], 'particle_id_src': [11, 22],
             'normal_theta': [170, 140.], 'normal_phi': [310, 280.]},
            index=[3, 4])
        desired = pd.concat([self.mps.particles, desired_source], axis=1)
        part_2 = self.ex_mps.set_normals(
            mps=self.mps, source=self.source, mps_coord_cols=['x', 'y', 'z'],
            source_coord_cols=['xx', 'yy', 'zz'], reverse=True,
            use_priors=False)
        assert_frame_equal(part_2, desired, check_dtype=False)
        
        # reverse, priors
        desired_source['normal_theta'] = [165, 135.]
        desired_source['normal_phi'] = [306, 276.]
        desired = pd.concat([self.mps.particles, desired_source], axis=1)
        part_2 = self.ex_mps.set_normals(
            mps=self.mps, source=self.source, mps_coord_cols=['x', 'y', 'z'],
            source_coord_cols=['xx', 'yy', 'zz'], reverse=True,
            use_priors=True)
        assert_frame_equal(part_2, desired, check_dtype=False)

    def test_normalize_bound_ids(self):
        """Tests normalize_bound_ids.
        """

        data = np.zeros((2, 10), dtype=int)
        data[:, 1] = 2
        data[:, 3] = 4
        data[0, 5] = 6
        data[1, 5] = 8
        data[0, 7] = 10
        data[1, 7] = 12

        data_cp = data.copy()
        desired = np.where(data_cp>0, data_cp + 1, 0)
        desired[desired > 7] = 7
        actual = ExtractMPS.normalize_bound_ids(
            data=data_cp, min_id_old=6, id_new=7, id_conversion={2: 3, 4: 5})
        np_test.assert_array_equal(actual, desired)

        data_cp = data.copy()
        desired = np.where((data_cp>0) & (data_cp<6), data_cp + 1, 0)
        actual = ExtractMPS.normalize_bound_ids(
            data=data_cp, id_conversion={2: 3, 4: 5})
        np_test.assert_array_equal(actual, desired)
        
        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestExtractMPS)
    unittest.TextTestRunner(verbosity=2).run(suite)
