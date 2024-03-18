"""
Tests module pandas_plus

# Author: Vladan Lucic
# $Id$
"""
__version__ = "$Revision$"

import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from pyto.util import pandas_plus


class TestPandasPlus(np_test.TestCase):
    """
    Tests pandas_plus module
    """

    def test_merge_left_keep_ondex(self):
        """Tests merge_left_keep_ondex
        """

        left = pd.DataFrame({'a': [4, 6, 5], 'b': [1, 2, 3]}, index=[4, 6, 5])
        right = pd.DataFrame(
            {'a': [4, 6, 5], 'c': [14, 16, 15]}, index=[20, 21, 22])
        desired = pd.DataFrame(
            {'a': [4, 6, 5], 'b': [1, 2, 3], 'c': [14, 16, 15]},
            index=[4, 6, 5])
        actual = pandas_plus.merge_left_keep_index(
            left=left, right=right, on='a', sort=False)
        assert_frame_equal(actual, desired)

        left = pd.DataFrame({'a': [4, 6, 5], 'b': [1, 2, 3]}, index=[4, 6, 5])
        left.index.name = 'left_ind'
        right = pd.DataFrame(
            {'a': [4, 4, 6, 5], 'c': [14, 140, 16, 15]},
            index=[20, 200, 21, 22])
        desired = pd.DataFrame(
            {'a': [4, 4, 6, 5], 'b': [1, 1, 2, 3], 'c': [14, 140, 16, 15]},
            index=[4, 4, 6, 5])
        desired.index.name = 'left_ind'
        actual = pandas_plus.merge_left_keep_index(
            left=left, right=right, on='a', validate='one_to_many')
        assert_frame_equal(actual, desired)
        
        left = pd.DataFrame(
            {'a': [4, 6, 5, 6], 'b': [1, 2, 3, 4]}, index=[4, 6, 5, 1])
        left.index.name = 'left_ind'
        right = pd.DataFrame(
            {'a': [4, 6, 5], 'c': [14, 16, 15]},
            index=[20, 21, 22])
        desired = pd.DataFrame(
            {'a': [4, 6, 5, 6], 'b': [1, 2, 3, 4], 'c': [14, 16, 15, 16]},
            index=[4, 6, 5, 1])
        desired.index.name = 'left_ind'
        actual = pandas_plus.merge_left_keep_index(
            left=left, right=right, on='a', validate='many_to_one')
        assert_frame_equal(actual, desired)
        
        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPandasPlus)
    unittest.TextTestRunner(verbosity=2).run(suite)
