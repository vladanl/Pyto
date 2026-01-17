"""
Tests class BoundaryNormal

# Author: Vladan Lucic
# $Id$
"""
__version__ = "$Revision$"

import os
import unittest

import numpy as np
import numpy.testing as np_test
import scipy as sp

from pyto.spatial.boundary import BoundaryNormal 


class TestBoundaryNormal(np_test.TestCase):
    """
    Tests BoundaryNormal
    """

    def setUp(self):
        """
        """

        # 2d region
        self.segment_id = 2
        self.external_id = 1
        self.bkg_id = 0
        image = np.zeros((10, 10), dtype=int)
        image[:9, 1:9] = self.external_id
        image[:3, 1:8] = self.segment_id
        image[3:5, 1:7] = self.segment_id
        image[5, 1:6] = self.segment_id
        image[6, 1:5] = self.segment_id
        image[7, 2:4] = self.segment_id
        self.image_2d = image.copy()

        # 2d region boundary thicknes 1
        image[:3, 1:7] = self.bkg_id
        image[3:5, 1:6] = self.bkg_id
        image[5, 1:5] = self.bkg_id
        image[6, 2:4] = self.bkg_id
        self.image_2d_boundary_1 = image.copy()
        self.image_2d_raw_phi = np.array([
            71.565,  90, 63.434, 71.565, 63.435, 45, 0., 45, -26.5650, 26.5650])
        self.image_2d_raw_normals = [
            [ 0.3162,  0.9486 ],
            [ 0.        ,  1.        ],
            [ 0.4472 , 0.8944],
            [ 0.3162,  0.9486 ],
            [ 0.4472,  0.8944],
            [ 0.7071,  0.7071],
            [ 1.        ,  0.        ],
            [ 0.7071,  0.7071],
            [ 0.8944, -0.4472 ],
            [ 0.8944,  0.4472 ]]
        self.image_2d_4_phi = np.array([
            74.1063, 71.9531, 67.5, 64.2606, 63.0007,
            43.2481, 11.4346, 33.2591, 18.5966, 18.5966])
        self.image_2d_10_phi = 46.7937
        self.image_2d_10_normal = [0.6846, 0.7288]

        # 2d region boundary thicknes 2
        image[2, 6] = self.segment_id
        image[4, 5] = self.segment_id
        image[5, 4] = self.segment_id
        image[6, 2:4] = self.segment_id        
        self.image_2d_boundary_2 = image.copy()

        # 3d sphere
        self.sphere_center = [20, 20, 20]
        self.sphere_radius = 12
        self.circle = np.zeros((40, 40, 40), dtype=int) + self.external_id
        center_mask = np.ones(self.circle.shape)
        center_mask[*self.sphere_center] = 0
        center_dist = sp.ndimage.distance_transform_edt(center_mask)
        self.circle[center_dist <= self.sphere_radius + 0.5] = self.segment_id
        
    def test_extract_boundary(self):
        """Tests extract_boundary()
        """

        bound = BoundaryNormal(
            image=self.image_2d, segment_id=self.segment_id,
            external_id=self.external_id, bound_thickness=None,
            dist_max_external=None)
        actual = bound.extract_boundary(image=self.image_2d)
        np_test.assert_array_equal(actual, self.image_2d_boundary_1)
        
        bound = BoundaryNormal(
            image=self.image_2d, segment_id=self.segment_id,
            external_id=self.external_id, bound_thickness=1.5,
            dist_max_external=None)
        actual = bound.extract_boundary(image=self.image_2d)
        np_test.assert_array_equal(actual, self.image_2d_boundary_2)
        
    def test_find_normals_raw(self):
        """Tests find_normals_raw.
        """

        bound = BoundaryNormal(
            image=self.image_2d, segment_id=self.segment_id,
            external_id=self.external_id)
        bound.find_normals_raw()
        np_test.assert_array_almost_equal(
            bound.spherical_phi_deg, self.image_2d_raw_phi, decimal=3) 
        np_test.assert_array_almost_equal(
            bound.normals, self.image_2d_raw_normals, decimal=3) 
        
    def test_find_normals(self):
        """Tests find_normals.
        """

        # 2d image, segment dist 0
        bound = BoundaryNormal(
            image=self.image_2d, segment_id=self.segment_id,
            external_id=self.external_id, dist_max_segment=0)
        bound.find_normals()
        np_test.assert_array_almost_equal(
            bound.spherical_phi_deg, self.image_2d_raw_phi, decimal=3) 
        np_test.assert_array_almost_equal(
            bound.normals, self.image_2d_raw_normals, decimal=3) 
        
        # 2d image, segment dist 4
        bound = BoundaryNormal(
            image=self.image_2d, segment_id=self.segment_id,
            external_id=self.external_id, dist_max_segment=4)
        bound.find_normals()
        np_test.assert_array_almost_equal(
            bound.spherical_phi_deg, self.image_2d_4_phi, decimal=3) 
        
        # 2d image, segment dist 4, points, raw_all_points=True
        points = [[2, 7], [7, 3]]
        bound = BoundaryNormal(
            image=self.image_2d, segment_id=self.segment_id,
            external_id=self.external_id, dist_max_segment=4)
        bound.find_normals(points=points, raw_all_points=True)
        np_test.assert_array_almost_equal(
            bound.spherical_phi_deg, self.image_2d_4_phi[[2, 9]], decimal=3) 
        
       # 2d image, points, raw_all_points=False
        points = [[2, 7], [7, 3]]
        bound = BoundaryNormal(
            image=self.image_2d, segment_id=self.segment_id,
            external_id=self.external_id, dist_max_segment=2)
        bound.find_normals(points=points, raw_all_points=False)
        np_test.assert_array_almost_equal(
            bound.spherical_phi_deg, self.image_2d_raw_phi[[2, 9]], decimal=3) 
        
        # 2d image, segment dist 10
        bound = BoundaryNormal(
            image=self.image_2d, segment_id=self.segment_id,
            external_id=self.external_id, dist_max_segment=10)
        bound.find_normals()
        n_bound = len(bound.boundary == self.segment_id)
        np_test.assert_array_almost_equal(
            bound.spherical_phi_deg, n_bound*[self.image_2d_10_phi], decimal=3) 
        np_test.assert_array_almost_equal(
            bound.normals, n_bound*[self.image_2d_10_normal], decimal=3) 

        # 3d sphere
        bound = BoundaryNormal(
            image=self.circle, segment_id=self.segment_id,
            external_id=self.external_id, dist_max_segment=4)
        bound.find_normals()
        points = [
            [32, 20, 20], [20, 32, 20], [8, 20, 20], [20, 8, 20],
            [20, 20, 32], [20, 20, 8],
            [29, 28, 20], [28, 20, 11], [20, 11, 12]]
        sq2 = np.sqrt(2) / 2
        desired_normals = np.array([
            [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
            [sq2, sq2, 0], [sq2, 0, -sq2], [0, -sq2, -sq2]])
        actual = [
            bound.normals[np.asarray(
                [(bp == po).all() for bp in bound.points])].squeeze()
            for po in points]
        np_test.assert_array_almost_equal(actual[:6], desired_normals[:6])
        np_test.assert_array_almost_equal(
            actual, desired_normals, decimal=1)
        actual = [
            bound.spherical_phi_deg[np.asarray(
                [(bp == po).all() for bp in bound.points])].squeeze()
            for po in points]
        desired = [0, 90, 180, -90, -1, -1, 45, 0, -90]
        np_test.assert_array_almost_equal(actual[:4], desired[:4])
        np_test.assert_allclose(actual[6], desired[6], atol=5)
        np_test.assert_array_almost_equal(actual[7:9], desired[7:9])
        actual = [
            bound.spherical_theta_deg[np.asarray(
                [(bp == po).all() for bp in bound.points])].squeeze()
            for po in points]
        desired = [90, 90, 90, 90, 0, 180, 90, 135, 135]
        np_test.assert_array_almost_equal(actual[:7], desired[:7])
        np_test.assert_allclose(actual[7:], desired[7:], atol=5)

    def test_find_normal_global(self):
        """Tests find_normal_global()
        """

        # 2d image, segment dist 10
        bound = BoundaryNormal(
            image=self.image_2d, segment_id=self.segment_id,
            external_id=self.external_id, dist_max_segment=10)
        bound.find_normal_global()
        np_test.assert_array_almost_equal(
            bound.spherical_phi_deg, self.image_2d_10_phi, decimal=3) 
        np_test.assert_array_almost_equal(
            bound.normals[0], self.image_2d_10_normal, decimal=3) 

        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBoundaryNormal)
    unittest.TextTestRunner(verbosity=2).run(suite)
