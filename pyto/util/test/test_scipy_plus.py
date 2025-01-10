"""

Tests module pyto.util.scipy_plus.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals

__version__ = "$Revision$"

import unittest

import numpy as np
import numpy.testing as np_test
import scipy as sp
import scipy.stats
import pandas as pd

from pyto.util.scipy_plus import chisquare_2, anova_two_level
#from ..scipy_plus import *  # no relative import when this file run directly


class TestScipyPlus(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        pass

    def test_chisquare_2(self):
        """
        Tests chisquare2()
        """

        # Statistics for the biological sciences, pg 110
        chisq, p = chisquare_2(f_obs_1=np.array([20, 30]),
                               f_obs_2=np.array([24, 26]),
                               yates=True)
        np_test.assert_almost_equal(chisq, 0.364, decimal=2)
        np_test.assert_equal(p>0.25, True)
        np_test.assert_almost_equal(p, 0.546, decimal=3)

        # Statistics for the biological sciences, pg 111
        chisq, p = chisquare_2(f_obs_1=np.array([60, 32, 28]),
                               f_obs_2=np.array([28, 17, 45]))
        np_test.assert_almost_equal(chisq, 16.23, decimal=2)
        np_test.assert_equal(p<0.005, True)
        np_test.assert_almost_equal(p, 0.0003, decimal=4)
        desired = scipy.stats.chi2.sf(chisq, 2)
        np_test.assert_almost_equal(p, desired, decimal=4)

        # check that 0-bins are correctly removed
        chisq, p = chisquare_2(f_obs_1=[2,3,0,4], f_obs_2=[6,5,0,4])
        np_test.assert_almost_equal(
            chisquare_2(f_obs_1=[2,3,4], f_obs_2=[6,5,4]), (chisq, p))

    def test_anova_two_level(self):
        """
        Tests anova_two_level()
        """

        # data from Biometry by Sokal and Rohlf, table 10.6
        y_106_list = []
        y_106_list.append(pd.DataFrame(
            {'dam': 1, 'sire': 1, 'ph':[48, 48, 52, 54]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 1, 'sire': 2, 'ph': [48, 53, 43, 39]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 2, 'sire': 1, 'ph': [45, 43, 49, 40, 40]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 2, 'sire': 2, 'ph': [50, 45, 43, 36]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 3, 'sire': 1, 'ph': [40, 45, 42, 48]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 3, 'sire': 2, 'ph': [45, 33, 40, 46]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 3, 'sire': 3, 'ph': [40, 47, 40, 47, 47]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 4, 'sire': 1, 'ph': [38, 48, 46]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 4, 'sire': 2, 'ph': [37, 31, 45, 41]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 5, 'sire': 1, 'ph': [44, 51, 49, 51, 52]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 5, 'sire': 2, 'ph': [49, 49, 49, 50]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 5, 'sire': 3, 'ph': [48, 59, 59]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 6, 'sire': 1, 'ph': [54, 36, 36, 40]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 6, 'sire': 2, 'ph': [44, 47, 48, 48]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 6, 'sire': 3, 'ph': [43, 52, 50, 46, 39]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 7, 'sire': 1, 'ph': [41, 42, 36, 47]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 7, 'sire': 2, 'ph': [47, 36, 43, 38, 41]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 7, 'sire': 3, 'ph': [53, 40, 44, 40, 45]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 8, 'sire': 1, 'ph': [52, 53, 48]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 8, 'sire': 2, 'ph': [40, 48, 50, 40, 51]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 9, 'sire': 1, 'ph': [40, 34, 37, 45]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 9, 'sire': 2, 'ph': [42, 37, 46, 40]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 10, 'sire': 1, 'ph': [39, 31, 30, 41, 48]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 10, 'sire': 2, 'ph': [50, 44, 40, 45]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 11, 'sire': 1, 'ph': [52, 54, 52, 56, 53]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 11, 'sire': 2, 'ph': [56, 39, 52, 49, 48]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 12, 'sire': 1, 'ph': [50, 45, 43, 44, 49]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 12, 'sire': 2, 'ph': [52, 43, 38, 33]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 13, 'sire': 1, 'ph': [39, 37, 33, 43, 42]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 13, 'sire': 2, 'ph': [43, 38, 44]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 13, 'sire': 3, 'ph': [46, 44, 37, 54]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 14, 'sire': 1, 'ph': [50, 53, 51, 43]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 14, 'sire': 2, 'ph': [44, 45, 39, 52]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 14, 'sire': 3, 'ph': [42, 48, 45, 51, 48]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 15, 'sire': 1, 'ph': [47, 49, 45, 43, 42]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 15, 'sire': 2, 'ph': [45, 42, 52, 51, 32]}))
        y_106_list.append(pd.DataFrame(
            {'dam': 15, 'sire': 3, 'ph': [51, 51, 53, 45, 51]}))
        y_106 = pd.concat(y_106_list, ignore_index=True)
        y_106.set_index(keys=['dam', 'sire'], inplace=True)

        # output struct
        res = anova_two_level(
            data=y_106, group_label='dam', subgroup_label='sire',
            value_label='ph', output='struct')
        np_test.assert_almost_equal(
            res.within_subgroups.sum_squares, 3042.533, decimal=3)
        np_test.assert_almost_equal(
            res.within_subgroups.mean_squares, 24.736, decimal=3)
        np_test.assert_equal(res.within_subgroups.deg_freedom, 123)
        np_test.assert_almost_equal(
            res.between_subgroups.sum_squares, 800.237, decimal=3)
        np_test.assert_almost_equal(
            res.between_subgroups.mean_squares, 36.374, decimal=3)
        np_test.assert_equal(res.between_subgroups.deg_freedom, 22)
        np_test.assert_almost_equal(
            res.between_groups.sum_squares, 1780.174, decimal=3)
        np_test.assert_almost_equal(
            res.between_groups.mean_squares, 127.155, decimal=3)
        np_test.assert_equal(res.between_groups.deg_freedom, 14)
        np_test.assert_almost_equal(
            res.between_subgroups.f, 1.470, decimal=3)
        np_test.assert_almost_equal(
            res.between_groups.f, 3.496, decimal=3)
        np_test.assert_almost_equal(
            res.between_subgroups.p, 0.09662, decimal=5)
        np_test.assert_almost_equal(
            res.between_groups.p, 0.00432, decimal=5)

        # output dataframe
        res = anova_two_level(
            data=y_106, group_label='dam', subgroup_label='sire',
            value_label='ph', output='dataframe')
        np_test.assert_almost_equal(
            res[res['Level'] == 'Between subgroups']['F'].to_numpy()[0],
            1.470, decimal=3)
        np_test.assert_almost_equal(
            res[res['Level'] == 'Between groups']['p'].to_numpy()[0],
            0.00432, decimal=5)
        np_test.assert_almost_equal(
            res[res['Level'] == 'Within subgroups'][
                'sum squares'].to_numpy()[0],
            3042.533, decimal=3)

        # output dataframe_line
        res = anova_two_level(
            data=y_106, group_label='dam', subgroup_label='sire',
            value_label='ph', output='dataframe_line')
        np_test.assert_almost_equal(
            res['Between groups F'].to_numpy()[0], 3.496, decimal=3)
        np_test.assert_almost_equal(
            res['Between subgroups p'].to_numpy()[0], 0.09662, decimal=5)

        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestScipyPlus)
    unittest.TextTestRunner(verbosity=2).run(suite)
