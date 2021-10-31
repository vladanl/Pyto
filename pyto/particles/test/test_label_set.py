"""

Tests module label_set

# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals
from __future__ import division
from builtins import range
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
from pyto.particles.label_set import LabelSet
from pyto.particles.set import Set
from pyto.particles.test import common

class TestLabelSet(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        Sets paths and imports analysis work module.
        """

        importlib.reload(common) # to avoid problems when running multiple tests
        common.setup()

        self.label_particle_paths = []
       
    def test_get_pickle_path(self):
        """
        Moving to SetPath

        Tests get_pickle_path()
        """

        #print("common.work_path_abs: {}".format(common.work_path_abs))
        labels = LabelSet(
            work_path=common.work_path_abs, catalog_var=common.catalogs_var)
        pickle_path = labels.get_pickle_path(
            work=common.work, group_name='group_x', identifier='exp_1')
        np_test.assert_equal(os.path.split(pickle_path)[1], 'tethers_exp-1.pkl')
        np_test.assert_equal(os.path.exists(pickle_path), True)
        np_test.assert_equal(pickle_path, common.pickle_path_1)

    def test_load_pickle(self):
        """
        Tests load_pickle()
        """

        labels = LabelSet(
            work_path=common.work_path_abs, catalog_var=common.catalogs_var)
        pickle_path = labels.get_pickle_path(
            work=common.work, group_name='group_x', identifier='exp_1')
        labels.load_pickle(path=pickle_path)
        np_test.assert_equal(hasattr(labels, '_labels'), True)
        np_test.assert_equal(labels._labels.data, common.sa_1.labels.data)
        np_test.assert_equal(hasattr(labels, '_morphology'), True)

    def test_get_label_box_centers(self):
        """
        Tests get_label_box_centers()
        """

        labels = LabelSet(
            work_path=common.work_path_abs, catalog_var=common.catalogs_var,
            box_size=6)
        pickle_path = labels.get_pickle_path(
            work=common.work, group_name='group_x', identifier='exp_1')
        labels.load_pickle(path=pickle_path)
        labels.get_label_box_centers(ids=[2,4,5])
        np_test.assert_equal(
            labels._centers_rel, common.coords_1.centers_labels_init_rel)
        np_test.assert_equal(
            labels._centers_abs, common.coords_1.centers_init_abs)

    def test_set_box_coords(self):
        """
        Tests set_box_coords() and get_relative_label_coords()
        """
            
        labels = LabelSet(
            work_path=common.work_path_abs, catalog_var=common.catalogs_var,
            box_size=6)
        labels._labels = common.sa_1.labels

        labels.set_box_coords(left_corners=common.coords_1.left_corners_6_adj_abs)
        np_test.assert_equal(
            labels._left_corners_abs, common.coords_1.left_corners_6_adj_abs)
        np_test.assert_equal(
            labels._left_corners_rel, common.coords_1.left_corners_labels_6_adj_rel)

        labels.set_box_coords(centers=common.coords_1.centers_6_adj_abs)
        np_test.assert_equal(
            labels._centers_abs, common.coords_1.centers_6_adj_abs)
        np_test.assert_equal(
            labels._centers_rel, common.coords_1.centers_labels_6_adj_rel)

    def test_get_coordinates_single(self):
        """
        Test test_get_coordinates_single()
        """

        # exp 1
        print("watch now")
        labels = LabelSet(
            struct=common.tethers, work_path=common.work_path_abs,
            catalog_var=common.catalogs_var, box_size=6)
        print("end")
        labels.get_coordinates_single(
            work=common.work, group_name='group_x', identifier='exp_1')
        np_test.assert_equal(labels._pickle_path, common.pickle_path_1)
        np_test.assert_equal(labels._labels.data, common.sa_1.labels.data)
        np_test.assert_equal(labels._ids, [2,4,5])
        np_test.assert_equal(
            labels._centers_rel, common.coords_1.centers_labels_init_rel)
        np_test.assert_equal(
            labels._centers_abs, common.coords_1.centers_init_abs)

        # exp 3
        labels = LabelSet(
            struct=common.tethers, work_path=common.work_path_abs,
            catalog_var=common.catalogs_var, box_size=6)
        labels.get_coordinates_single(
            work=common.work, group_name='group_y', identifier='exp_3')
        np_test.assert_equal(labels._pickle_path, common.pickle_path_3)
        np_test.assert_equal(labels._labels.data, common.sa_3.labels.data)
        np_test.assert_equal(labels._ids, [6,7])
        np_test.assert_equal(
            labels._centers_rel, common.coords_3.centers_labels_init_rel)
        np_test.assert_equal(
            labels._centers_abs, common.coords_3.centers_init_abs)

        # exp 2
        labels = LabelSet(
            struct=common.tethers, work_path=common.work_path_abs,
            catalog_var=common.catalogs_var, box_size=6)
        labels.get_coordinates_single(
            work=common.work, group_name='group_x', identifier='exp_2')
        np_test.assert_equal(labels._pickle_path, common.pickle_path_2)
        np_test.assert_equal(labels._labels.data, common.sa_2.labels.data)
        np_test.assert_equal(labels._ids, [])
        np_test.assert_equal(
            labels._centers_rel, common.coords_2.centers_labels_init_rel)
        np_test.assert_equal(
            labels._centers_abs, common.coords_2.centers_init_abs)

    def test_write_particles(self):
        """
        Tests write_particles()
        """

        # make label set
        labels = LabelSet(
            struct=common.tethers, work_path=common.work_path_abs,
            catalog_var=common.catalogs_var,
            box_size=6, particle_dir=common.particle_dir_abs)
        labels.get_coordinates_single(
            work=common.work, group_name='group_x', identifier='exp_1')

        # make particle set
        particles = Set(
            struct=common.tethers, work_path=common.work_path_abs,
            catalog_var=common.catalogs_var,
            box_size=6, particle_dir=common.particle_dir_abs)
        particles.get_coordinates_single(
            work=common.work, group_name='group_x', identifier='exp_1',
            label_set=labels)
        labels.set_box_coords(
            left_corners=particles._left_corners,
            centers=particles._centers)

        # test
        labels.dtype = 'int16'
        labels.fg_value = 1
        labels.bkg_value = 0
        labels.write_particles(identifier='exp_1', ids=None, keep_ids=True)
        self.label_particle_paths += labels._particle_paths
        desired_paths = [
            os.path.join(
                common.particle_dir_abs,'exp_1_id-{}_label.mrc'.format(id_))
            for id_ in [2, 4, 5]]
        np_test.assert_equal(
            labels._particle_paths, desired_paths) 
        images = [Labels.read(file=path) for path in desired_paths]
        for ind in list(range(3)):
            np_test.assert_equal(
                images[ind].data==0, common.label_particles_1[ind].data==0)
        for ind in list(range(3)):
            np_test.assert_equal(
                images[ind].data==1, common.label_particles_1[ind].data>0)

        # test
        labels.dtype = 'uint8'
        labels.fg_value = 3
        labels.bkg_value = 2
        labels.write_particles(identifier='exp_1', ids=None, keep_ids=True)
        self.label_particle_paths += labels._particle_paths
        desired_paths = [
            os.path.join(
                common.particle_dir_abs,'exp_1_id-{}_label.mrc'.format(id_))
            for id_ in [2, 4, 5]]
        np_test.assert_equal(
            labels._particle_paths, desired_paths) 
        images = [Labels.read(file=path) for path in desired_paths]
        for ind in list(range(3)):
            np_test.assert_equal(
                images[ind].data==2, common.label_particles_1[ind].data==0)
        for ind in list(range(3)):
            np_test.assert_equal(
                images[ind].data==3, common.label_particles_1[ind].data>0)

    def test_add_data(self):
        """
        Tests add_data()
        """

        # make label set
        labels = LabelSet(
            struct=common.tethers, work_path=common.work_path_abs,
            catalog_var=common.catalogs_var,
            box_size=6, particle_dir=common.particle_dir_abs,
            dtype='int16', fg_value=1, bkg_value=0)
        labels.get_coordinates_single(
            work=common.work, group_name='group_x', identifier='exp_1')

        # make particle set
        particles = Set(
            struct=common.tethers, work_path=common.work_path_abs,
            catalog_var=common.catalogs_var,
            box_size=6, particle_dir=common.particle_dir_abs)
        particles.get_coordinates_single(
            work=common.work, group_name='group_x', identifier='exp_1',
            label_set=labels)
        labels.set_box_coords(
            left_corners=particles._left_corners,
            centers=particles._centers)
        labels.write_particles(
            identifier='exp_1', ids=None, keep_ids=True, test=True)

        # test
        labels.add_data(group_name='group_x', identifier='exp_1')
        desired_columns = [
            'identifier', 'group_name', 'id', 'tomo_path', 'particle_path',
            'left_corner_x', 'left_corner_y', 'left_corner_z']
        np_test.assert_equal(
            (labels.data.columns == desired_columns).all(), True)
        np_test.assert_equal(
            (labels.data.identifier == 3*['exp_1']).all(), True) 
        np_test.assert_equal(
            (labels.data.group_name == 3*['group_x']).all(), True) 
        np_test.assert_equal(
            (labels.data.tomo_path == 3*[common.pickle_path_1]).all(), True) 
        np_test.assert_equal(
            (labels.data.left_corner_x ==
             [x[0] for x
              in common.coords_1.left_corners_labels_6_adj_rel]).all(),
            True) 
        np_test.assert_equal(
            (labels.data.left_corner_y == [
                x[1] for x
                in common.coords_1.left_corners_labels_6_adj_rel]).all(),
            True) 
        np_test.assert_equal(
            (labels.data.left_corner_z == [
             x[2] for x
                in common.coords_1.left_corners_labels_6_adj_rel]).all(),
            True)      
            
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
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLabelSet)
    unittest.TextTestRunner(verbosity=2).run(suite)
