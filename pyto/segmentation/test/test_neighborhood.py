"""Tests module neighborhood.

# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
#from past.utils import old_div

__version__ = "$Revision$"

from copy import copy, deepcopy
import importlib
import unittest

import numpy as np
import numpy.testing as np_test 
import scipy as sp

from pyto.segmentation.test import common
from pyto.segmentation.labels import Labels
from pyto.segmentation.segment import Segment
from pyto.segmentation.neighborhood import Neighborhood


class TestNeighborhood(np_test.TestCase):
    """
    Tests Neighborhood class
    """

    def setUp(self):
        """
        """
        
        bound_data = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 2, 2, 2, 2, 2, 2, 2, 2, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
             [0, 0, 0, 0, 0, 3, 3, 0, 0, 0],
             [0, 0, 0, 3, 3, 3, 0, 0, 0, 0],
             [0, 0, 3, 3, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        seg_data = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 4, 4, 4, 0, 0, 0],
             [0, 0, 1, 0, 4, 4, 0, 0, 0, 0],
             [0, 0, 1, 0, 4, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.desired_contacts_2 = np.array(
            [[0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 4, 4, 4, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0]])
        self.desired_inset_2 = [slice(2, 7), slice(1, 9)]
        self.desired_inset_2_extend = [slice(1, 8), slice(0, 10)]
        self.cm_2_rel = [[1, 1], [1, 4]]
        self.cm_2_rel_extend = [[2, 2], [2, 5]]
        self.cm_2_abs = [[3, 2], [3, 5]]
        self.cm_2_segments = [[2, 1], [2, 4]]
        self.cm_2_regions = [[1, 2], [1, 5]]
        self.desired_contacts_3 = np.array(
            [[0, 0, 0, 0, 4],
             [0, 0, 0, 4, 0],
             [0, 0, 4, 0, 0],
             [1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]])
        self.desired_inset_3 = [slice(3, 8), slice(2, 7)]
        self.cm_3_rel = [[3, 0], [1, 3]]
        self.cm_3_abs = [[6, 2], [4, 5]]
        self.cm_3_segments = [[5, 1], [3, 4]]
        self.cm_3_regions = [[4, 2], [2, 5]]
             
        self.bound = Segment(data=bound_data)
        bound_2_inset = [slice(2, 10), slice(0, 10)]
        self.bound_2 = Segment(data=(self.bound.data[tuple(bound_2_inset)]).copy())
        self.bound_2.inset = bound_2_inset
        self.seg = Segment(data=seg_data)
        seg_2_inset = [slice(1, 9), slice(1, 8)]
        self.seg_2 = Segment(data=(self.seg.data[tuple(seg_2_inset)]).copy())
        self.seg_2.inset = seg_2_inset

    def test_make_contacts(self):
        """Tests make_contacts().
        """

        # full size
        seg_data_cp = self.seg.data.copy()
        regions_data_cp = self.bound.data.copy()
        hood = Neighborhood(segments=self.seg)
        contacts = hood.make_contacts(regions=self.bound, region_id=2)
        np_test.assert_equal(contacts.inset, self.desired_inset_2)
        np_test.assert_equal(contacts.data, self.desired_contacts_2.data)
        contacts = hood.make_contacts(regions=self.bound, region_id=3)
        np_test.assert_equal(contacts.inset, self.desired_inset_3)
        np_test.assert_equal(contacts.data, self.desired_contacts_3.data)

        # check input images not modified 
        np_test.assert_equal(self.seg.data, seg_data_cp)
        np_test.assert_equal(self.bound.data, regions_data_cp)

        # extend
        hood = Neighborhood(segments=self.seg)
        contacts = hood.make_contacts(regions=self.bound, region_id=2, extend=1)
        np_test.assert_equal(contacts.inset, self.desired_inset_2_extend)
       
        # inset
        hood = Neighborhood()
        contacts = hood.make_contacts(
            segments=self.seg_2, regions=self.bound_2, region_id=2)
        np_test.assert_equal(contacts.inset, self.desired_inset_2)
        np_test.assert_equal(contacts.data, self.desired_contacts_2.data)
        contacts = hood.make_contacts(
            segments=self.seg_2, regions=self.bound_2, region_id=3)
        np_test.assert_equal(contacts.inset, self.desired_inset_3)
        np_test.assert_equal(contacts.data, self.desired_contacts_3.data)
 
        # check input images not modified 
        np_test.assert_equal(self.seg.data, seg_data_cp)
        np_test.assert_equal(self.bound.data, regions_data_cp)
        
    def test_find_contact_cm(self):
        """Tests find_contact_cm().
        """

        # whole arrays
        hood = Neighborhood(segments=self.seg)
        cm, _ = hood.find_contact_cm(regions=self.bound, region_id=2, frame='rel')
        np_test.assert_equal(cm, self.cm_2_rel)
        cm, _ = hood.find_contact_cm(regions=self.bound, region_id=2, frame='abs')
        np_test.assert_equal(cm, self.cm_2_abs)
        cm, _ = hood.find_contact_cm(regions=self.bound, region_id=3, frame='rel')
        np_test.assert_equal(cm, self.cm_3_rel)
        cm, _ = hood.find_contact_cm(regions=self.bound, region_id=3, frame='abs')
        np_test.assert_equal(cm, self.cm_3_abs)
 
        # arrays with insets, reg_id = 2
        hood = Neighborhood(segments=self.seg_2)
        cm, _ = hood.find_contact_cm(
            regions=self.bound_2, region_id=2, frame='rel')
        np_test.assert_equal(cm, self.cm_2_rel)
        cm, _ = hood.find_contact_cm(
            regions=self.bound_2, region_id=2, frame='abs')
        np_test.assert_equal(cm, self.cm_2_abs)
        cm, _ = hood.find_contact_cm(
            regions=self.bound_2, region_id=2, frame='segments')
        np_test.assert_equal(cm, self.cm_2_segments)
        cm, _ = hood.find_contact_cm(
            regions=self.bound_2, region_id=2, frame='regions')
        np_test.assert_equal(cm, self.cm_2_regions)
 
        # arrays with insets, reg_id = 2, extend = 1
        hood = Neighborhood(segments=self.seg_2)
        cm, _ = hood.find_contact_cm(
            regions=self.bound_2, region_id=2, frame='rel', extend=1)
        np_test.assert_equal(cm, self.cm_2_rel_extend)

        # arrays with insets, reg_id = 3
        cm, _ = hood.find_contact_cm(
            regions=self.bound_2, region_id=3, frame='rel')
        np_test.assert_equal(cm, self.cm_3_rel)
        cm, _ = hood.find_contact_cm(
            regions=self.bound_2, region_id=3, frame='abs')
        np_test.assert_equal(cm, self.cm_3_abs)
        cm, _ = hood.find_contact_cm(
            regions=self.bound_2, region_id=3, frame='segments')
        np_test.assert_equal(cm, self.cm_3_segments)
        cm, _ = hood.find_contact_cm(
            regions=self.bound_2, region_id=3, frame='regions')
        np_test.assert_equal(cm, self.cm_3_regions)
       
    def test_generate_neighborhoods(self):
        """Tests generate_neighborhood()
        """

        # setup
        bound_data = np.zeros((10, 10), dtype=int)
        bound_data[3, 1:9] = 2
        bound = Segment(data=bound_data)
        bound_2_inset = [slice(2, 10), slice(0, 10)]
        bound_2 = Segment(data=(bound.data[tuple(bound_2_inset)]).copy())
        bound_2.inset = bound_2_inset
        seg_data = np.zeros((10, 10), dtype=int)
        seg_data[4:8, 2] = 1
        seg_data[4:6, 4:7] = 4
        seg = Segment(data=seg_data)
        seg_2_inset = [slice(1, 9), slice(1, 8)]
        seg_2 = Segment(data=(seg.data[tuple(seg_2_inset)]).copy())
        seg_2.inset = seg_2_inset

        # no size, no max_distance
        nbgd = Neighborhood(segments=seg, ids=[1, 4])
        for reg_id, hood, all_hoods in nbgd.generate_neighborhoods(
                regions=bound, region_ids=[2], size=None,
                max_distance=None):
            desired_hood_data = seg.data[4:8, 2:7]
            desired_hood_inset = [slice(4, 8), slice(2, 7)]
            np_test.assert_equal(reg_id, 2)
            np_test.assert_equal(hood.data, desired_hood_data)
            np_test.assert_equal(hood.inset, desired_hood_inset)

        # size, no max_distance
        nbgd = Neighborhood(segments=seg, ids=[1, 4])
        for reg_id, hood, all_hoods in nbgd.generate_neighborhoods(
                regions=bound, region_ids=[2], size=1,
                max_distance=None):
            desired_hood_inset = [slice(4, 8), slice(2, 7)]
            np_test.assert_equal(reg_id, 2)
            np_test.assert_equal(hood.data[0, 0], 1)
            np_test.assert_equal((hood.data[0, 2:5] == 4).any(), True)
            np_test.assert_equal(hood.inset, desired_hood_inset)

        # two regions, no max_distance
        bound.data[8, 1:9] = 3
        seg_data_cp = seg.data.copy()
        nbgd = Neighborhood(segments=seg, ids=[1, 4])
        for reg_id, hood, all_hoods in nbgd.generate_neighborhoods(
                regions=bound, region_ids=[2, 3], size=None,
                max_distance=None):

            np_test.assert_equal(reg_id in [2, 3], True)
            if reg_id == 2:
                desired_hood_inset = [slice(4, 8), slice(2, 7)]
                np_test.assert_equal(hood.inset, desired_hood_inset)
                np_test.assert_equal(hood.data[0, 0], 1)
                np_test.assert_equal(hood.data[0, 2:5], 4)

                # to check seg.data not changed
                all_hoods.data = np.zeros_like(all_hoods.data) - 1
                hood.data = np.zeros_like(hood.data) - 100
                np_test.assert_equal(seg.data, seg_data_cp)
                
            elif reg_id == 3:
                desired_hood_inset = [slice(4, 8), slice(2, 7)]
                np_test.assert_equal(hood.inset, desired_hood_inset)
                np_test.assert_equal(hood.data[3, 0], 1)
                np_test.assert_equal(hood.data[0, 2:5], 4)
                np_test.assert_equal(seg.data, seg_data_cp)
            else:
                np_test.assert_equal(True, False)
    
        # two regions, max_distance
        bound.data[8, 1:9] = 3
        seg_data_cp = seg.data.copy()
        bound_data_cp = bound.data.copy()
        nbgd = Neighborhood(segments=seg, ids=[1, 4])
        for reg_id, hood, all_hoods in nbgd.generate_neighborhoods(
                regions=bound, region_ids=[2, 3], size=None,
                max_distance=1):

            np_test.assert_equal(reg_id in [2, 3], True)
            if reg_id == 2:
                desired_hood_inset = [slice(4, 8), slice(2, 7)]
                np_test.assert_equal(hood.inset, desired_hood_inset)
                np_test.assert_equal(hood.data[0, 0], 1)
                np_test.assert_equal(hood.data[0, 2:5], 4)
                
                # to check seg.data not changed
                all_hoods.data = np.zeros_like(all_hoods.data) - 1
                hood.data = np.zeros_like(hood.data) - 100
                np_test.assert_equal(seg.data, seg_data_cp)
                np_test.assert_equal(bound.data, bound_data_cp)
                
            elif reg_id == 3:
                desired_hood_inset = [slice(4, 8), slice(2, 7)]
                np_test.assert_equal(hood.inset, desired_hood_inset)
                np_test.assert_equal(hood.data[3, 0], 1)
                np_test.assert_equal(hood.data[0, 2:5], 4)

                # to check seg.data not changed
                all_hoods.data = np.zeros_like(all_hoods.data) - 1
                hood.data = np.zeros_like(hood.data) - 100
                np_test.assert_equal(seg.data, seg_data_cp)
                np_test.assert_equal(bound.data, bound_data_cp)
                
            else:
                np_test.assert_equal(True, False)
       
        # seg and bound given as insets, no size, no max_distance
        nbgd = Neighborhood(segments=seg_2, ids=[1, 4])
        for reg_id, hood, all_hoods in nbgd.generate_neighborhoods(
                regions=bound_2, region_ids=[2], ids=[1, 4], size=None,
                max_distance=None):
            desired_hood_data = seg.data[4:8, 2:7]
            desired_hood_inset = [slice(4, 8), slice(2, 7)]
            np_test.assert_equal(reg_id, 2)
            np_test.assert_equal(hood.data, desired_hood_data)
            np_test.assert_equal(hood.inset, desired_hood_inset)

        # seg and bound given as insets, two regions, max_distance
        bound_2.data[6, 1:9] = 3
        seg_2_data_cp = seg_2.data.copy()
        bound_2_data_cp = bound_2.data.copy()
        nbgd = Neighborhood(segments=seg_2, ids=[1, 4])
        for reg_id, hood, all_hoods in nbgd.generate_neighborhoods(
                regions=bound_2, region_ids=[2, 3], size=None,
                max_distance=1):

            np_test.assert_equal(reg_id in [2, 3], True)
            if reg_id == 2:
                desired_hood_inset = [slice(4, 8), slice(2, 7)]
                np_test.assert_equal(hood.inset, desired_hood_inset)
                np_test.assert_equal(hood.data[0, 0], 1)
                np_test.assert_equal(hood.data[0, 2:5], 4)
                
                # to check seg_2.data not changed
                all_hoods.data = np.zeros_like(all_hoods.data) - 1
                hood.data = np.zeros_like(hood.data) - 100
                np_test.assert_equal(seg_2.data, seg_2_data_cp)
                np_test.assert_equal(bound_2.data, bound_2_data_cp)
                
            elif reg_id == 3:
                desired_hood_inset = [slice(4, 8), slice(2, 7)]
                np_test.assert_equal(hood.inset, desired_hood_inset)
                np_test.assert_equal(hood.data[3, 0], 1)
                np_test.assert_equal(hood.data[0, 2:5], 4)

                # to check seg_2.data not changed
                all_hoods.data = np.zeros_like(all_hoods.data) - 1
                hood.data = np.zeros_like(hood.data) - 100
                np_test.assert_equal(seg_2.data, seg_2_data_cp)
                np_test.assert_equal(bound_2.data, bound_2_data_cp)
                
            else:
                np_test.assert_equal(True, False)
       
       
            
if __name__ == '__main__':
    suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestNeighborhood)
    unittest.TextTestRunner(verbosity=2).run(suite)
