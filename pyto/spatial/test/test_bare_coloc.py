"""
Tests module bare_coloc

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
from pyto.spatial.bare_coloc import BareColoc
from pyto.spatial.coloc_core import ColocCore

class TestBareColoc(np_test.TestCase):
    """
    Tests bare_coloc module
    """

    def setUp(self):
        """
        """
        # needed to make variables defined in this function available here
        common.make_coloc_tables_1tomo_2d()
         
    def test_calculate_distances(self):
        """Tests calculate_distances
        """

        # 3-coloc
        bc = BareColoc()
        patterns=[common.pattern_0, common.pattern_1, common.pattern_2]
        bc.calculate_distances(patterns=patterns)
        np_test.assert_equal(isinstance(bc.dist_nm_full, list), True)
        np_test.assert_equal(len(bc.dist_nm_full), 3)
        np_test.assert_almost_equal(bc.dist_nm_full[0], common.dist_0_0)
        np_test.assert_almost_equal(bc.dist_nm_full[1], common.dist_0_1)
        np_test.assert_almost_equal(bc.dist_nm_full[2], common.dist_0_2)

        # 2-coloc
        bc = BareColoc()
        patterns=[common.pattern_0, common.pattern_2]
        bc.calculate_distances(patterns=patterns)
        np_test.assert_equal(isinstance(bc.dist_nm_full, list), True)
        np_test.assert_equal(len(bc.dist_nm_full), 2)
        np_test.assert_almost_equal(bc.dist_nm_full[0], common.dist_0_0)
        np_test.assert_almost_equal(bc.dist_nm_full[1], common.dist_0_2)

        # 2-coloc, no pattern 0
        bc = BareColoc()
        patterns=[np.array([]), common.pattern_1]
        bc.calculate_distances(patterns=patterns)
        np_test.assert_equal(isinstance(bc.dist_nm_full, list), True)
        np_test.assert_equal(len(bc.dist_nm_full), 2)
        np_test.assert_almost_equal(
            bc.dist_nm_full[0], np.array([]).reshape(0, 0))
        np_test.assert_almost_equal(
            bc.dist_nm_full[1], np.array([]).reshape(0, len(common.pattern_1)))

        # 2-coloc, no pattern 1
        bc = BareColoc()
        patterns=[common.pattern_0, None]
        bc.calculate_distances(patterns=patterns)
        np_test.assert_equal(isinstance(bc.dist_nm_full, list), True)
        np_test.assert_equal(len(bc.dist_nm_full), 2)
        np_test.assert_almost_equal(
            bc.dist_nm_full[0], common.dist_0_0)
        np_test.assert_almost_equal(
            bc.dist_nm_full[1], np.array([]).reshape(len(common.pattern_0), 0))

        # 2-coloc, no pattern 0, 1
        bc = BareColoc()
        patterns=[np.array([]), None]
        bc.calculate_distances(patterns=patterns)
        np_test.assert_equal(isinstance(bc.dist_nm_full, list), True)
        np_test.assert_equal(len(bc.dist_nm_full), 2)
        np_test.assert_almost_equal(
            bc.dist_nm_full[1], np.array([]).reshape(0, 0))

        # 3-coloc, no pattern 0
        bc = BareColoc()
        patterns=[None, common.pattern_1, common.pattern_2]
        bc.calculate_distances(patterns=patterns)
        np_test.assert_equal(isinstance(bc.dist_nm_full, list), True)
        np_test.assert_equal(len(bc.dist_nm_full), 3)
        np_test.assert_almost_equal(
            bc.dist_nm_full[0], np.array([]).reshape(0, 0))
        np_test.assert_almost_equal(
            bc.dist_nm_full[1], np.array([]).reshape(0, len(common.pattern_1)))
        np_test.assert_almost_equal(
            bc.dist_nm_full[2], np.array([]).reshape(0, len(common.pattern_2)))

        # 3-coloc, no pattern 2
        bc = BareColoc()
        patterns=[common.pattern_0, common.pattern_1, None]
        bc.calculate_distances(patterns=patterns)
        np_test.assert_equal(isinstance(bc.dist_nm_full, list), True)
        np_test.assert_equal(len(bc.dist_nm_full), 3)
        np_test.assert_almost_equal(bc.dist_nm_full[0], common.dist_0_0)
        np_test.assert_almost_equal(bc.dist_nm_full[1], common.dist_0_1)
        np_test.assert_almost_equal(
            bc.dist_nm_full[2], np.array([]).reshape(len(common.pattern_0), 0))

        # 3-coloc, no patterns
        bc = BareColoc()
        patterns=[np.array([]), np.array([]), None]
        bc.calculate_distances(patterns=patterns)
        np_test.assert_equal(isinstance(bc.dist_nm_full, list), True)
        np_test.assert_equal(len(bc.dist_nm_full), 3)
        np_test.assert_almost_equal(
            bc.dist_nm_full[0], np.array([]).reshape(0, 0))
        np_test.assert_almost_equal(
            bc.dist_nm_full[1], np.array([]).reshape(0, 0))
        np_test.assert_almost_equal(
            bc.dist_nm_full[2], np.array([]).reshape(0, 0))
       
    def test_calculate_coloc(self):
        """Tests calculate_coloc()
        """

        distance = [2, 4, 6, 8]
        
        # d = 2
        di = 2
        bc = BareColoc()
        bc.dist_nm_full = (common.dist_0_0, common.dist_0_1, common.dist_0_2)
        bc.calculate_coloc(distance=di)
        np_test.assert_equal(bc.coloc3, common.coloc3_d2)
        np_test.assert_equal(isinstance(bc.particles3, list), True)
        np_test.assert_equal(bc.particles3, common.particles3_d2)
        np_test.assert_equal(bc.coloc3_indices, common.coloc3_indices_d2)
        np_test.assert_equal(
            bc.particles3_indices, common.particles3_indices_d2)
        np_test.assert_equal(bc.coloc3_n, common.coloc3_n_d2)
        np_test.assert_equal(bc.particles3_n, common.particles3_n_d2)
        
        np_test.assert_equal(isinstance(bc.coloc2, list), True)
        np_test.assert_equal(isinstance(bc.coloc2[0], np.ndarray), True)
        np_test.assert_equal(bc.coloc2, common.coloc2_d2)
        np_test.assert_equal(bc.particles2, common.particles2_d2)
        np_test.assert_equal(isinstance(bc.coloc2_indices, list), True)
        np_test.assert_equal(
            isinstance(bc.coloc2_indices[0], np.ndarray), True)
        np_test.assert_equal(bc.coloc2_indices, common.coloc2_indices_d2)
        np_test.assert_equal(
            bc.particles2_indices, common.particles2_indices_d2)
        np_test.assert_equal(bc.coloc2_n, common.coloc2_n_d2)
        np_test.assert_equal(bc.particles2_n, common.particles2_n_d2)

        # d = 2, less_eq
        di = 2
        bc = BareColoc(mode='less_eq')
        bc.dist_nm_full = (common.dist_0_0, common.dist_0_1, common.dist_0_2)
        bc.calculate_coloc(distance=di)        
        np_test.assert_equal(bc.coloc3, common.coloc3_d2_le)
        np_test.assert_equal(bc.particles3, common.particles3_d2_le)
        np_test.assert_equal(bc.coloc3_indices, common.coloc3_indices_d2_le)
        np_test.assert_equal(
            bc.particles3_indices, common.particles3_indices_d2_le)
        np_test.assert_equal(bc.coloc3_n, common.coloc3_n_d2_le)
        np_test.assert_equal(bc.particles3_n, common.particles3_n_d2_le)
        
        np_test.assert_equal(bc.coloc2, common.coloc2_d2_le)
        np_test.assert_equal(bc.particles2, common.particles2_d2_le)
        np_test.assert_equal(bc.coloc2_indices, common.coloc2_indices_d2_le)
        np_test.assert_equal(
            bc.particles2_indices, common.particles2_indices_d2_le)
        np_test.assert_equal(bc.coloc2_n, common.coloc2_n_d2_le)
        np_test.assert_equal(bc.particles2_n, common.particles2_n_d2_le)

        # d = 4
        di = 4
        bc = BareColoc()
        bc.dist_nm_full = (common.dist_0_0, common.dist_0_1, common.dist_0_2)
        bc.calculate_coloc(distance=di)
        np_test.assert_equal(bc.coloc3, common.coloc3_d4)
        np_test.assert_equal(bc.particles3, common.particles3_d4)
        np_test.assert_equal(bc.coloc3_indices, common.coloc3_indices_d4)
        np_test.assert_equal(
            bc.particles3_indices, common.particles3_indices_d4)
        np_test.assert_equal(bc.coloc3_n, common.coloc3_n_d4)
        np_test.assert_equal(bc.particles3_n, common.particles3_n_d4)
        
        np_test.assert_equal(bc.coloc2, common.coloc2_d4)
        np_test.assert_equal(bc.particles2, common.particles2_d4)
        np_test.assert_equal(bc.coloc2_indices, common.coloc2_indices_d4)
        np_test.assert_equal(
            bc.particles2_indices, common.particles2_indices_d4)
        np_test.assert_equal(bc.coloc2_n, common.coloc2_n_d4)
        np_test.assert_equal(bc.particles2_n, common.particles2_n_d4)

        # d = 6
        di = 6
        bc = BareColoc()
        bc.dist_nm_full = (common.dist_0_0, common.dist_0_1, common.dist_0_2)
        bc.calculate_coloc(distance=di)
        np_test.assert_equal(bc.coloc3, common.coloc3_d6)
        np_test.assert_equal(bc.particles3, common.particles3_d6)
        np_test.assert_equal(bc.coloc3_indices, common.coloc3_indices_d6)
        np_test.assert_equal(
            bc.particles3_indices, common.particles3_indices_d6)
        np_test.assert_equal(bc.coloc3_n, common.coloc3_n_d6)
        np_test.assert_equal(bc.particles3_n, common.particles3_n_d6)
        
        np_test.assert_equal(bc.coloc2, common.coloc2_d6)
        np_test.assert_equal(bc.particles2, common.particles2_d6)
        np_test.assert_equal(bc.coloc2_indices, common.coloc2_indices_d6)
        np_test.assert_equal(
            bc.particles2_indices, common.particles2_indices_d6)
        np_test.assert_equal(bc.coloc2_n, common.coloc2_n_d6)
        np_test.assert_equal(bc.particles2_n, common.particles2_n_d6)

        # d = 8
        di = 8
        bc = BareColoc()
        bc.dist_nm_full = (common.dist_0_0, common.dist_0_1, common.dist_0_2)
        bc.calculate_coloc(distance=di)
        np_test.assert_equal(bc.coloc3, common.coloc3_d8)
        np_test.assert_equal(bc.particles3, common.particles3_d8)
        np_test.assert_equal(bc.coloc3_indices, common.coloc3_indices_d8)
        np_test.assert_equal(
            bc.particles3_indices, common.particles3_indices_d8)
        np_test.assert_equal(bc.coloc3_n, common.coloc3_n_d8)
        np_test.assert_equal(bc.particles3_n, common.particles3_n_d8)
        
        np_test.assert_equal(bc.coloc2, common.coloc2_d8)
        np_test.assert_equal(bc.particles2, common.particles2_d8)
        np_test.assert_equal(bc.coloc2_indices, common.coloc2_indices_d8)
        np_test.assert_equal(
            bc.particles2_indices, common.particles2_indices_d8)
        np_test.assert_equal(bc.coloc2_n, common.coloc2_n_d8)
        np_test.assert_equal(bc.particles2_n, common.particles2_n_d8)

        # 2-coloc, d = 4
        di = 4
        bc = BareColoc()
        bc.dist_nm_full = [common.dist_0_0, common.dist_0_1]
        bc.calculate_coloc(distance=di)
        np_test.assert_equal(bc.coloc2, [common.coloc2_d4[0]])
        np_test.assert_equal(bc.particles2, [common.particles2_d4[0]])
        np_test.assert_equal(
            bc.coloc2_indices, [common.coloc2_indices_d4[0]])
        np_test.assert_equal(
            bc.particles2_indices, [common.particles2_indices_d4[0]])
        np_test.assert_equal(bc.coloc2_n, [common.coloc2_n_d4[0]])
        np_test.assert_equal(bc.particles2_n, [common.particles2_n_d4[0]])

        # 2-coloc, d = 6
        di = 6
        bc = BareColoc()
        bc.dist_nm_full = [common.dist_0_0, common.dist_0_2]
        bc.calculate_coloc(distance=di)
        np_test.assert_equal(bc.coloc2, [common.coloc2_d6[1]])
        np_test.assert_equal(bc.particles2, [common.particles2_d6[1]])
        np_test.assert_equal(
            bc.coloc2_indices, [common.coloc2_indices_d6[1]])
        np_test.assert_equal(
            bc.particles2_indices, [common.particles2_indices_d6[1]])
        np_test.assert_equal(bc.coloc2_n, [common.coloc2_n_d6[1]])
        np_test.assert_equal(bc.particles2_n, [common.particles2_n_d6[1]])

        # zero points in pattern 0, 2-coloc
        di = 6
        bc = BareColoc()
        bc.dist_nm_full = [
            np.array([]).reshape(0, 0), np.array([]).reshape(0, 5)]
        bc.calculate_coloc(distance=di)
        np_test.assert_equal(isinstance(bc.coloc2, list), True)
        np_test.assert_equal(len(bc.coloc2), 1)
        np_test.assert_equal(bc.coloc2, [[]])
        np_test.assert_equal(isinstance(bc.particles2, list), True)
        np_test.assert_equal(bc.particles2, [[[], 5 * [False]]])
        np_test.assert_equal(bc.coloc2_indices, [[]])
        np_test.assert_equal(bc.particles2_indices, [[[], []]])
        np_test.assert_equal(bc.coloc2_n, [0])
        np_test.assert_equal(bc.particles2_n, [[0, 0]])
        
        # zero points in pattern 1, 2-coloc
        di = 6
        bc = BareColoc()
        bc.dist_nm_full = [common.dist_0_0, np.array([]).reshape(5, 0)]
        bc.calculate_coloc(distance=di)
        np_test.assert_equal(isinstance(bc.coloc2, list), True)
        np_test.assert_equal(len(bc.coloc2), 1)
        np_test.assert_equal(bc.coloc2, [5 * [False]])
        np_test.assert_equal(bc.particles2, [[5 * [False], []]])
        np_test.assert_equal(isinstance(bc.coloc2_indices, list), True)
        np_test.assert_equal(bc.coloc2_indices, [[]])
        np_test.assert_equal(bc.particles2_indices, [[[], []]])
        np_test.assert_equal(bc.coloc2_n, [0])
        np_test.assert_equal(bc.particles2_n, [[0, 0]])
        
        # 3-coloc, zero points in pattern 0
        di = 6
        bc = BareColoc()
        bc.dist_nm_full = [
            np.array([]).reshape(0, 0), np.array([]).reshape(0, 5),
            np.array([]).reshape(0, 3)]
        bc.calculate_coloc(distance=di)
        np_test.assert_equal(isinstance(bc.coloc3, np.ndarray), True)
        np_test.assert_equal(bc.coloc3, [])
        np_test.assert_equal(bc.coloc3_indices, [])
        np_test.assert_equal(bc.coloc3_n, 0)
        np_test.assert_equal(isinstance(bc.particles3, list), True)
        np_test.assert_equal(bc.particles3, [[], 5 * [False],3 * [False]])
        np_test.assert_equal(bc.particles3_indices, [[], [], []])
        np_test.assert_equal(bc.particles3_n, [0, 0, 0])
      
        np_test.assert_equal(isinstance(bc.coloc2, list), True)
        np_test.assert_equal(isinstance(bc.coloc2[0], np.ndarray), True)
        np_test.assert_equal(bc.coloc2, [[], []])
        np_test.assert_equal(bc.coloc2_indices, [[], []])
        np_test.assert_equal(bc.coloc2_n, [0, 0])
        np_test.assert_equal(isinstance(bc.particles2, list), True)
        np_test.assert_equal(isinstance(bc.particles2[0], list), True)
        np_test.assert_equal(isinstance(bc.particles2[0][0], np.ndarray), True)
        np_test.assert_equal(
            bc.particles2, [[[], 5 * [False]], [[], 3 * [False]]])
        np_test.assert_equal(bc.particles2_indices, [[[], []], [[], []]])
        np_test.assert_equal(bc.particles2_n, [[0, 0], [0, 0]])
              
        # 3-coloc, zero points in pattern 1
        di = 6
        p0_len = common.dist_0_0.shape[0]
        p2_len = common.dist_0_2.shape[1]
        bc = BareColoc()
        bc.dist_nm_full = [
            common.dist_0_0, np.array([]).reshape(p0_len, 0), common.dist_0_2]
        bc.calculate_coloc(distance=di)
        np_test.assert_equal(isinstance(bc.coloc3, np.ndarray), True)
        np_test.assert_equal(bc.coloc3, p0_len * [False])
        np_test.assert_equal(bc.coloc3_indices, [])
        np_test.assert_equal(bc.coloc3_n, 0)
        np_test.assert_equal(
            bc.particles3, [p0_len * [False], [], p2_len * [False]])
        np_test.assert_equal(bc.particles3_indices, [[], [], []])
        np_test.assert_equal(bc.particles3_n, [0, 0, 0])
 
        np_test.assert_equal(isinstance(bc.coloc2, list), True)
        np_test.assert_equal(isinstance(bc.coloc2[0], np.ndarray), True)
        np_test.assert_equal(bc.coloc2, [p0_len * [False], common.coloc2_d6[1]])
        np_test.assert_equal(
            bc.coloc2_indices, [[], common.coloc2_indices_d6[1]])
        np_test.assert_equal(bc.coloc2_n, [0, common.coloc2_n_d6[1]])
        np_test.assert_equal(isinstance(bc.particles2, list), True)
        np_test.assert_equal(isinstance(bc.particles2[0], list), True)
        np_test.assert_equal(isinstance(bc.particles2[0][0], np.ndarray), True)
        np_test.assert_equal(
            bc.particles2,
            [[p0_len * [False], []], common.particles2_d6[1]])
        np_test.assert_equal(
            bc.particles2_indices,
            [[[], []], common.particles2_indices_d6[1]])
        np_test.assert_equal(
            bc.particles2_n, [[0, 0], common.particles2_n_d6[1]])
       
        # 3-coloc, zero points 
        di = 6
        bc = BareColoc()
        bc.dist_nm_full = [
            np.array([]).reshape(0, 0), np.array([]).reshape(0, 0),
            np.array([]).reshape(0, 0)]
        bc.calculate_coloc(distance=di)
        np_test.assert_equal(isinstance(bc.coloc3, np.ndarray), True)
        np_test.assert_equal(bc.coloc3, [])
        np_test.assert_equal(bc.coloc3_indices, [])
        np_test.assert_equal(bc.coloc3_n, 0)
        np_test.assert_equal(isinstance(bc.particles3, list), True)
        np_test.assert_equal(bc.particles3, [[], [], []])
        np_test.assert_equal(bc.particles3_indices, [[], [], []])
        np_test.assert_equal(bc.particles3_n, [0, 0, 0])

        np_test.assert_equal(isinstance(bc.coloc2, list), True)
        np_test.assert_equal(isinstance(bc.coloc2[0], np.ndarray), True)
        np_test.assert_equal(bc.coloc2, [[], []])
        np_test.assert_equal(bc.coloc2_indices, [[], []])
        np_test.assert_equal(bc.coloc2_n, [0, 0])
        np_test.assert_equal(isinstance(bc.particles2, list), True)
        np_test.assert_equal(isinstance(bc.particles2[0], list), True)
        np_test.assert_equal(isinstance(bc.particles2[0][0], np.ndarray), True)
        np_test.assert_equal(
            bc.particles2, [[[], []], [[], []]])
        np_test.assert_equal(bc.particles2_indices, [[[], []], [[], []]])
        np_test.assert_equal(bc.particles2_n, [[0, 0], [0, 0]])

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBareColoc)
    unittest.TextTestRunner(verbosity=2).run(suite)
