"""
Tests module cylinder

# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals
from __future__ import division
from builtins import range
#from past.utils import old_div

__version__ = "$Revision$"

import unittest

import numpy as np
import numpy.testing as np_test 

import pyto
from pyto.geometry.cylinder import Cylinder


class TestCylinder(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        pass

    def test_make(self):
        """Tests make()
        """

        cyl = Cylinder(
            z_min=5, z_max=15, rho=1, shape=[5, 5, 25], axis_xy='center')
        cyl.make()
        actual = cyl.data
        expected_in = np.zeros((5, 5), dtype=bool)
        expected_in[1:4, 2] = True
        expected_in[2, 1:4] = True
        expected_out = np.zeros((5, 5), dtype=bool)
        np_test.assert_array_equal(actual[:, :, 4], expected_out)
        np_test.assert_array_equal(actual[:, :, 5], expected_in)
        np_test.assert_array_equal(actual[:, :, 15], expected_in)
        np_test.assert_array_equal(actual[:, :, 16], expected_out)



            
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCylinder)
    unittest.TextTestRunner(verbosity=2).run(suite)
