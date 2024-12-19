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


class TestExtractMPSFilter(np_test.TestCase):
    """
    Tests extract_mps_filter module. 
    """

    def setUp(self):
        """
        """
        self.ex_mps = ExtractMPS()

    def test_abstract_class(self):
        """Tests that ExtractMPSFilter is an abstract class
        """

        with np_test.assert_raises(TypeError):
            ExtractMPSFilter()
    

        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestExtractMPSFilter)
    unittest.TextTestRunner(verbosity=2).run(suite)
