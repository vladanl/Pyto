"""

Tests module set

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
from pyto.grey.image import Image
from pyto.particles.set import Set
from pyto.particles.label_set import LabelSet
from pyto.particles.test import common

class TestSet(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        Sets paths and imports analysis work module.
        """

        common.setup()
        
        self.tomo_particle_paths = []

    def test_get_tomo_path(self):
        """
        Tests get_tomo_path()
        """
        
        labels = LabelSet(
            work_path=common.work_path_abs, catalog_var=common.catalogs_var)
        pickle_path = labels.get_pickle_path(
            work=common.work, group_name='group_x', identifier='exp_1')
        particles = Set(
            work_path=common.work_path_abs, catalog_var=common.catalogs_var)
        tomo_path = particles.get_tomo_path(pickle_path=pickle_path)
        np_test.assert_equal(os.path.split(tomo_path)[1], 'tomo-1.mrc')
        np_test.assert_equal(os.path.exists(tomo_path), True)
        np_test.assert_equal(tomo_path, common.tomo_path_1)

    def test_adjust_box_coordinates(self):
        """
        Tests adjust_box_coordinates
        """

        # get centers from labels
        labels = LabelSet(
            work_path=common.work_path_abs, catalog_var=common.catalogs_var,
            box_size=6)
        pickle_path = labels.get_pickle_path(
            work=common.work, group_name='group_x', identifier='exp_1')
        labels.load_pickle(path=pickle_path)
        labels.get_label_box_centers(ids=[2,4,5])

        # test
        particles = Set(
            work_path=common.work_path_abs, catalog_var=common.catalogs_var,
            box_size=6)
        particles._tomo = common.tomo_1
        particles.adjust_box_coords(centers=labels._centers_abs)
        np_test.assert_equal(
            particles._left_corners, common.coords_1.left_corners_6_adj_abs)
        np_test.assert_equal(
            particles._centers, common.coords_1.centers_6_adj_abs)

        # get centers from labels
        labels = LabelSet(
            work_path=common.work_path_abs, catalog_var=common.catalogs_var,
            box_size=6)
        pickle_path = labels.get_pickle_path(
            work=common.work, group_name='group_y', identifier='exp_3')
        labels.load_pickle(path=pickle_path)
        labels.get_label_box_centers(ids=[6, 7])

        # test
        particles = Set(
            struct=common.tethers, work_path=common.work_path_abs,
            catalog_var=common.catalogs_var, box_size=6)
        particles._tomo = common.tomo_3
        particles.adjust_box_coords(centers=labels._centers_abs)
        np_test.assert_equal(
            particles._left_corners, common.coords_3.left_corners_6_adj_abs)
        np_test.assert_equal(
            particles._centers, common.coords_3.centers_6_adj_abs)

    def test_get_coordinates_single(self):
        """
        Test test_get_coordinates_single()
        """

        #
        # Experiment 1
        #
        
        # make labels
        labels = LabelSet(
            struct=common.tethers, work_path=common.work_path_abs,
            catalog_var=common.catalogs_var, box_size=6)
        labels.get_coordinates_single(
            work=common.work, group_name='group_x', identifier='exp_1')

        # make particles
        particles = Set(
            struct=common.tethers, work_path=common.work_path_abs,
            catalog_var=common.catalogs_var, box_size=6)

        # test
        particles.get_coordinates_single(
            work=common.work, group_name='group_x', identifier='exp_1',
            label_set=labels)
        np_test.assert_equal(particles._tomo_path, common.tomo_path_1)
        np_test.assert_equal(particles._ids, [2, 4,5])
        np_test.assert_equal(
            particles._left_corners, common.coords_1.left_corners_6_adj_abs)
        np_test.assert_equal(
            particles._centers, common.coords_1.centers_6_adj_abs)
       
        #
        # experiment 2
        #
        
        # make labels
        labels = LabelSet(
            struct=common.tethers, work_path=common.work_path_abs,
            catalog_var=common.catalogs_var, box_size=6)
        labels.get_coordinates_single(
            work=common.work, group_name='group_x', identifier='exp_2')

        # make particles
        particles = Set(
            struct=common.tethers, work_path=common.work_path_abs,
            catalog_var=common.catalogs_var, box_size=6)

        # test
        particles.get_coordinates_single(
            work=common.work, group_name='group_x', identifier='exp_2',
            label_set=labels)
        np_test.assert_equal(particles._tomo_path, common.tomo_path_2)
        np_test.assert_equal(particles._ids, [])
        np_test.assert_equal(
            particles._left_corners, common.coords_2.left_corners_6_adj_abs)
        np_test.assert_equal(
            particles._centers, common.coords_2.centers_6_adj_abs)
       
        #
        # experiment 3
        #
        
        # make labels
        labels = LabelSet(
            struct=common.tethers, work_path=common.work_path_abs,
            catalog_var=common.catalogs_var, box_size=6)
        labels.get_coordinates_single(
            work=common.work, group_name='group_y', identifier='exp_3')

        # make particles
        particles = Set(
            struct=common.tethers, work_path=common.work_path_abs,
            catalog_var=common.catalogs_var, box_size=6)

        # test
        particles.get_coordinates_single(
            work=common.work, group_name='group_y', identifier='exp_3',
            label_set=labels)
        np_test.assert_equal(particles._tomo_path, common.tomo_path_3)
        np_test.assert_equal(particles._ids, [6, 7])
        np_test.assert_equal(
            particles._left_corners, common.coords_3.left_corners_6_adj_abs)
        np_test.assert_equal(
            particles._centers, common.coords_3.centers_6_adj_abs)
       
    def test_write_particles(self):
        """
        Tests write_particles()
        """

        # make label set
        labels = LabelSet(
            struct=common.tethers, work_path=common.work_path_abs,
            catalog_var=common.catalogs_var, box_size=6)
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

        # test
        particles.dtype = 'float32'
        particles.write_particles(identifier='exp_1')
        desired_paths = [
            os.path.join(common.particle_dir_abs,'exp_1_id-{}.mrc'.format(id_))
            for id_ in [2, 4, 5]]
        np_test.assert_equal(
            particles._particle_paths, desired_paths) 
        images = [Image.read(file=path) for path in desired_paths]
        for ind in list(range(3)):
            np_test.assert_equal(
                images[ind].data, common.tomo_particles_1[ind].data)

        # test mean and std
        particles.dtype = 'float32'
        particles.mean = 0
        particles.std = 5
        particles.write_particles(identifier='exp_1')
        self.tomo_particle_paths += particles._particle_paths
        desired_paths = [
            os.path.join(common.particle_dir_abs,'exp_1_id-{}.mrc'.format(id_))
            for id_ in [2, 4, 5]]
        np_test.assert_equal(
            particles._particle_paths, desired_paths) 
        images = [Image.read(file=path) for path in desired_paths]
        for ind in list(range(3)):
            np_test.assert_almost_equal(
                images[ind].data.mean(), particles.mean, decimal=6)
        for ind in list(range(3)):
            np_test.assert_almost_equal(
                images[ind].data.std(), particles.std, decimal=6)

        # test int dtype
        particles = Set(
            struct=common.tethers, work_path=common.work_path_abs,
            catalog_var=common.catalogs_var,
            box_size=6, particle_dir=common.particle_dir_abs,
            dtype='uint8', mean=128., std=5.)
        print('now')
        particles.get_coordinates_single(
            work=common.work, group_name='group_x', identifier='exp_1',
            label_set=labels)
        particles.write_particles(identifier='exp_1')
        images = [Image.read(file=path) for path in desired_paths]
        for ind in list(range(3)):
            np_test.assert_almost_equal(
                images[ind].data.mean(), 128, decimal=-1)
        for ind in list(range(3)):
            np_test.assert_almost_equal(
                images[ind].data.std(), 5, decimal=1)       
            
    def test_add_data(self):
        """
        Tests add_data()
        """

        # make label set
        labels = LabelSet(
            struct=common.tethers, work_path=common.work_path_abs,
            catalog_var=common.catalogs_var, box_size=6)
        labels.get_coordinates_single(
            work=common.work, group_name='group_x', identifier='exp_1')

        # make particle set
        particles = Set(
            struct=common.tethers, work_path=common.work_path_abs,
            catalog_var=common.catalogs_var, dtype='float32',
            box_size=6, particle_dir=common.particle_dir_abs)
        particles.get_coordinates_single(
            work=common.work, group_name='group_x', identifier='exp_1',
            label_set=labels)
        particles.write_particles(identifier='exp_1', test=True)

        # test
        particles.add_data(group_name='group_x', identifier='exp_1')
        desired_columns = [
            'identifier', 'group_name', 'id', 'tomo_path', 'particle_path',
            'left_corner_x', 'left_corner_y', 'left_corner_z']
        np_test.assert_equal(
            (particles.data.columns == desired_columns).all(), True)
        np_test.assert_equal(
            (particles.data.identifier == 3*['exp_1']).all(), True) 
        np_test.assert_equal(
            (particles.data.group_name == 3*['group_x']).all(), True) 
        np_test.assert_equal(
            (particles.data.tomo_path == 3*[common.tomo_path_1]).all(), True) 
        np_test.assert_equal(
            (particles.data.left_corner_x ==
             [x[0] for x in common.coords_1.left_corners_6_adj_abs]).all(),
            True) 
        np_test.assert_equal(
            (particles.data.left_corner_y ==
             [x[1] for x in common.coords_1.left_corners_6_adj_abs]).all(),
            True) 
        np_test.assert_equal(
            (particles.data.left_corner_z ==
             [x[2] for x in common.coords_1.left_corners_6_adj_abs]).all(),
            True)
            
    def test_extract_particles(self):
        """
        Tests extract_particles()
        """

        particles, labels = Set().extract_particles(
            struct=common.tethers, work=common.work,
            work_path=common.work_path_abs, catalog_var=common.catalogs_var,
            group_names=['group_x', 'group_y'],
            identifiers=['exp_1', 'exp_2', 'exp_3'], 
            box_size=6, particle_dir=common.particle_dir_abs)

        # test tomo particle data
        desired_columns = [
            'identifier', 'group_name', 'id', 'tomo_path', 'particle_path',
            'left_corner_x', 'left_corner_y', 'left_corner_z']
        np_test.assert_equal(
            (particles.data.columns == desired_columns).all(), True)
        desired = 3*['exp_1'] + 2*['exp_3']
        np_test.assert_equal(
            (particles.data.identifier == desired).all(), True) 
        desired = 3*['group_x'] + 2*['group_y']
        np_test.assert_equal(
            (particles.data.group_name == desired).all(), True)
        desired = 3*[common.tomo_path_1] + 2*[common.tomo_path_3]
        np_test.assert_equal(
            (particles.data.tomo_path == desired).all(), True) 
        desired = (
            [x[0] for x in common.coords_1.left_corners_6_adj_abs] +
            [x[0] for x in common.coords_3.left_corners_6_adj_abs])
        np_test.assert_equal(
            (particles.data.left_corner_x == desired).all(), True) 
        desired = (
            [x[1] for x in common.coords_1.left_corners_6_adj_abs] +
            [x[1] for x in common.coords_3.left_corners_6_adj_abs])
        np_test.assert_equal(
            (particles.data.left_corner_y == desired).all(), True)
        desired = (
            [x[2] for x in common.coords_1.left_corners_6_adj_abs] +
            [x[2] for x in common.coords_3.left_corners_6_adj_abs])
        np_test.assert_equal(
            (particles.data.left_corner_z == desired).all(), True)
        
        # test label particles data
        desired_columns = [
            'identifier', 'group_name', 'id', 'tomo_path', 'particle_path',
            'left_corner_x', 'left_corner_y', 'left_corner_z']
        np_test.assert_equal(
            (labels.data.columns == desired_columns).all(), True)
        desired = 3*['exp_1'] + 2*['exp_3']
        np_test.assert_equal(
            (labels.data.identifier == desired).all(), True) 
        desired = 3*['group_x'] + 2*['group_y']
        np_test.assert_equal(
            (labels.data.group_name == desired).all(), True)
        desired = 3*[common.pickle_path_1] + 2*[common.pickle_path_3]
        np_test.assert_equal(
            (labels.data.tomo_path == desired).all(), True) 
        desired = (
            [x[0] for x in common.coords_1.left_corners_labels_6_adj_rel] +
            [x[0] for x in common.coords_3.left_corners_labels_6_adj_rel])
        np_test.assert_equal(
            (labels.data.left_corner_x == desired).all(), True) 
        desired = (
            [x[1] for x in common.coords_1.left_corners_labels_6_adj_rel] +
            [x[1] for x in common.coords_3.left_corners_labels_6_adj_rel])
        np_test.assert_equal(
            (labels.data.left_corner_y == desired).all(), True) 
        desired = (
            [x[2] for x in common.coords_1.left_corners_labels_6_adj_rel] +
            [x[2] for x in common.coords_3.left_corners_labels_6_adj_rel])
        np_test.assert_equal(
            (labels.data.left_corner_z == desired).all(), True) 

        # check if the order of groups is irrelevant
        particles, labels = Set().extract_particles(
            struct=common.tethers, work=common.work,
            work_path=common.work_path_abs, catalog_var=common.catalogs_var,
            group_names=['group_y', 'group_x'],
            identifiers=['exp_1', 'exp_2', 'exp_3'], 
            box_size=6, particle_dir=common.particle_dir_abs)
        desired = 3*['exp_1'] + 2*['exp_3']
        np_test.assert_equal(
            (particles.data.identifier == desired).all(), True) 

        # check the order of experiments
        particles, labels = Set().extract_particles(
            struct=common.tethers, work=common.work,
            work_path=common.work_path_abs, catalog_var=common.catalogs_var,
            group_names=['group_x', 'group_y'],
            identifiers=['exp_3', 'exp_1', 'exp_2'], 
            box_size=6, particle_dir=common.particle_dir_abs)
        desired = 2*['exp_3'] + 3*['exp_1']
        np_test.assert_equal(
            (particles.data.identifier == desired).all(), True) 

        # check exception for non-existing group 
        with np_test.assert_raises(ValueError):
            particles, labels = Set().extract_particles(
                struct=common.tethers, work=common.work,
                work_path=common.work_path_abs, catalog_var=common.catalogs_var,
                group_names=['group_x', 'group_y'],
                identifiers=['exp_3', 'exp_17', 'exp_2'], 
                box_size=6, particle_dir=common.particle_dir_abs)

    def tearDown(self):
        """
         Removes all particle image files
        """

        # too complicated to remove only the files created by the tests
        try:
            shutil.rmtree(common.particle_dir_abs)
        except FileNotFoundError:
            pass

        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSet)
    unittest.TextTestRunner(verbosity=2).run(suite)
