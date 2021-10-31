"""

Tests module set_path

# Author: Vladan Lucic
# $Id$
"""
#from __future__ import unicode_literals
#from __future__ import division
#from builtins import range
#from past.utils import old_div

__version__ = "$Revision$"

#from copy import copy, deepcopy
import importlib
import pickle
import os
import unittest
import shutil

import numpy as np
import numpy.testing as np_test 
#import scipy as sp

import pyto
from pyto.segmentation.labels import Labels
from pyto.particles.set_path import SetPath
from pyto.particles.set import Set
from pyto.particles.test import common

class TestSetPath(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        Sets paths and imports analysis work module.
        """
        
        # to avoid problems when running multiple tests
        importlib.reload(common) 
        common.setup()

        self.label_particle_paths = []
       
    def test_get_pickle_path(self):
        """
        Tests get_pickle_path()
        """

        # using structure dependent pickle
        labels = SetPath(catalog_var=common.catalogs_var)
        pickle_path = labels.get_pickle_path(
            group_name='group_x', identifier='exp_1', struct=common.tethers,
            catalog_dir=common.catalog_dir_abs)
        np_test.assert_equal(os.path.split(pickle_path)[1], 'tethers_exp-1.pkl')
        np_test.assert_equal(os.path.exists(pickle_path), True)
        np_test.assert_equal(pickle_path, common.pickle_path_1)

        # using structure dependent pickle with substitution
        labels = SetPath(catalog_var=common.catalogs_var)
        good_pickle_path = pickle_path
        pickle_path = labels.get_pickle_path(
            group_name='group_x', identifier='exp_1', struct=common.tethers,
            catalog_dir=common.catalog_dir_abs,
            old_dir='/home/vladan/tools/pyto_dev/pyto/', new_dir='/foo')
        np_test.assert_equal(os.path.split(pickle_path)[1], 'tethers_exp-1.pkl')
        np_test.assert_equal(pickle_path.startswith('/foo/'), True)

        # using work module
        labels = SetPath(catalog_var=common.catalogs_var)
        pickle_path = labels.get_pickle_path(
            work=common.work, work_path=common.work_path_abs,
            group_name='group_x', identifier='exp_1')
        np_test.assert_equal(os.path.split(pickle_path)[1], 'tethers_exp-1.pkl')
        np_test.assert_equal(os.path.exists(pickle_path), True)
        np_test.assert_equal(pickle_path, common.pickle_path_1)

    def test_get_tomo_info_path(self):
        """
        Tests get_tomo_info_path()
        """
        
        sp = SetPath(catalog_var=common.catalogs_var)
        pickle_path = sp.get_pickle_path(
            group_name='group_x', identifier='exp_1', struct=common.tethers,
            catalog_dir=common.catalog_dir_abs)
        tomo_info_path = sp.get_tomo_info_path(pickle_path)
        np_test.assert_equal(os.path.exists(tomo_info_path), True)
        np_test.assert_equal(tomo_info_path.endswith('tomo_info.py'), True)
        
    def test_import_tomo_info(self):
        """
        Tests import_tomo_info()
        """
        
        sp = SetPath(catalog_var=common.catalogs_var)
        pickle_path = sp.get_pickle_path(
            group_name='group_x', identifier='exp_1', struct=common.tethers,
            catalog_dir=common.catalog_dir_abs)
        tomo_info_path = sp.get_tomo_info_path(pickle_path)
        tomo_info = sp.import_tomo_info(tomo_info_path=tomo_info_path)
        np_test.assert_equal(tomo_info.image_file_name, "../3d/tomo-1.mrc")
         
    def test_get_tomo_path(self):
        """
        Tests get_tomo_path()
        """
        
        sp = SetPath(catalog_var=common.catalogs_var)
        pickle_path = sp.get_pickle_path(
            group_name='group_x', identifier='exp_1', struct=common.tethers,
            catalog_dir=common.catalog_dir_abs)
        tomo_path = sp.get_tomo_path(pickle_path=pickle_path)
        np_test.assert_equal(os.path.split(tomo_path)[1], 'tomo-1.mrc')
        np_test.assert_equal(os.path.exists(tomo_path), True)
        np_test.assert_equal(tomo_path, common.tomo_path_1)

    def tearDown(self):
        """
        Removes all label particle image files
        """

        # too complicated to remove only the files created by the tests
        try:
            shutil.rmtree(common.particle_dir_abs)
        except FileNotFoundError:
            pass
            
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSetPath)
    unittest.TextTestRunner(verbosity=2).run(suite)
