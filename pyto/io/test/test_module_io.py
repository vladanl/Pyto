"""
Tests class ModuleIO

# Author: Vladan Lucic
# $Id$
"""
__version__ = "$Revision$"

import os
import shutil
import pickle
import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd

from pyto.io.module_io import ModuleIO


class TestModuleIO(np_test.TestCase):
    """
    Tests ModuleIO
    """

    def setUp(self):
        """
        """
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.module_a_path = 'dir_a/module_a.py'

    def test_load(self):
        """Tests load()
        """

        # Note: arg calling_dir has to be the dir of this file because the
        # module path is relative to this file's dir
        
        mio = ModuleIO(calling_dir=self.current_dir)
        mod_a = mio.load(path=self.module_a_path, preprocess=True)
        np_test.assert_equal(mod_a.a, 1)
        np_test.assert_equal(mod_a.b, 2)
        np_test.assert_equal(mod_a.x, 3.3)

        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModuleIO)
    unittest.TextTestRunner(verbosity=2).run(suite)
       
        

    
