"""
Tests methods of mps_conversion that are not tested in 
test_multi_particle_sets.py.

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
from pyto.spatial.mps_conversion import MPSConversion
from pyto.spatial.multi_particle_sets import MultiParticleSets
from pyto.spatial.test.test_particle_sets import TestParticleSets
from pyto.spatial.test.test_multi_particle_sets import TestMultiParticleSets


class TestMPSConversion(np_test.TestCase):
    """
    Tests mps_conversion module. Other tests are in test_multi_particle_sets.
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
            MPSConversion()

    def test_from_patterns(self):
        """Tests from_patterns.
        """

        # update False
        mps = MultiParticleSets()
        expected_alpha = pd.DataFrame(
            {mps.tomo_id_col: 'alpha', mps.pixel_nm_col: 1.2,
             mps.class_name_col: ['pattern_P', 'pattern_P', 'pattern_P',
                                  'pattern_Q', 'pattern_Q'],
             mps.subclass_col: ['pattern_P', 'pattern_P', 'pattern_P',
                                'pattern_Q', 'pattern_Q'],
             'x_coords': [100, 101, 102., 103, 104],
             'y_coords': [200, 201, 202., 203, 204],
             'z_coords': [300, 301, 302., 303, 304], mps.keep_col: True})
        coord_cols = ['x_coords', 'y_coords', 'z_coords']
        patterns_alpha = {
            'pattern_P': expected_alpha[coord_cols].to_numpy()[:3, :],
            'pattern_Q': expected_alpha[coord_cols].to_numpy()[3:, :]}
        actual = mps.from_patterns(
            patterns=patterns_alpha, coord_cols=coord_cols, tomo_id='alpha',
            pixel_size_nm=1.2)
        assert_frame_equal(actual, expected_alpha)

        # update True on no mps.particles
        mps.from_patterns(
            patterns=patterns_alpha, coord_cols=coord_cols, tomo_id='alpha',
            pixel_size_nm=1.2, update=True)
        assert_frame_equal(mps.particles, expected_alpha)

        # update
        expected_bravo = pd.DataFrame(
            {mps.tomo_id_col: 'bravo', mps.pixel_nm_col: 1.2,
             mps.class_name_col: ['pattern_P', 'pattern_P'],
             mps.subclass_col: ['pattern_P', 'pattern_P'],
             'x_coords': [10, 11.],
             'y_coords': [20, 21.],
             'z_coords': [30, 31.], mps.keep_col: True})
        expected = pd.concat(
            [expected_alpha, expected_bravo], ignore_index=True)
        patterns_bravo = {
            'pattern_P': expected_bravo[coord_cols].to_numpy()}
        mps.from_patterns(
            patterns=patterns_bravo, coord_cols=coord_cols, tomo_id='bravo',
            pixel_size_nm=1.2, update=True)
        assert_frame_equal(mps.particles, expected)

        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMPSConversion)
    unittest.TextTestRunner(verbosity=2).run(suite)
