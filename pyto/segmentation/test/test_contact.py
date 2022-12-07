"""

Tests module contact. Need to add more tests.

# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals
from __future__ import absolute_import

__version__ = "$Revision$"

from copy import copy, deepcopy
import importlib
import unittest

import numpy
import numpy.testing as np_test 
import scipy

from pyto.segmentation.contact import Contact
from pyto.segmentation.segment import Segment
from pyto.segmentation.test import common 


class TestContact(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        importlib.reload(common) # to avoid problems when running multiple tests

    def testGetSetN(self):
        """
        Tests methods getN() and setN() and also getSegments() and 
        setSegments().
        """

        # test empty contact
        con = Contact()
        np_test.assert_equal(con._n.shape, [1,1])
        np_test.assert_equal(con.getN(boundaryId=2), [])
        np_test.assert_equal(con.getN(segmentId=2), [])
        np_test.assert_equal(con.getSegments(), [])
        np_test.assert_equal(con.getBoundaries(), [])

        # test setting and getting contacts 
        con.setN(boundaryId=2, segmentId=21, nContacts=1)
        np_test.assert_equal(con.getSegments(), [21])
        np_test.assert_equal(con.getBoundaries(), [2])
        con.setN(boundaryId=2, segmentId=23, nContacts=3)
        np_test.assert_equal(con.getSegments(), [21, 23])
        np_test.assert_equal(con.getBoundaries(), [2])
        con.setN(boundaryId=3, segmentId=32, nContacts=2)
        con.setN(boundaryId=3, segmentId=21, nContacts=1)
        np_test.assert_equal(con.getSegments(), [21, 23, 32])
        np_test.assert_equal(con.getBoundaries(), [2, 3])
        np_test.assert_equal(con.getN(boundaryId=2, segmentId=23), 3)
        np_test.assert_equal(con.getN(boundaryId=2)[[21, 23]], [1, 3])
        np_test.assert_equal(con.getN(boundaryId=2, segmentId=[21, 23]), [1, 3])
        np_test.assert_equal(con.getN(segmentId=21)[[2, 3]], [1, 1])

    def testSegmentsBoundaries(self):
        """
        Tests getSegments() and getBoundries().
        """

        contacts = Contact()
        contacts._n = numpy.ma.array(
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 1, 0], 
             [0, 0, 0, 0, 0],
             [0, 0, 1, 0, 1]])
        contacts._n._mask = numpy.zeros(shape=(4,5), dtype=bool)
        contacts._n._mask[0:1,0:1] = True
        np_test.assert_equal(contacts.segments, [1, 2, 3, 4])
        np_test.assert_equal(contacts.boundaries, [1, 2, 3])
        # prehaps this should be the correct behavior
        #np_test.assert_equal(contacts.segments, [2, 3, 4])
        #np_test.assert_equal(contacts.boundaries, [1, 3])

    def testAddBoundary(self):
        """
        Tests addBoundary()
        """

        # start from empty contacts
        contacts = Contact()
        contacts.addBoundary(id=2, nContacts=[-1, 1, 0, 3])
        np_test.assert_equal(contacts.maxBoundary, 2)
        np_test.assert_equal(contacts.maxSegment, 3)
        np_test.assert_equal(contacts.getN(boundaryId=2), [-2, 1, 0, 3])
        np_test.assert_equal(contacts.getN(segmentId=3), [-99, -99, 3])
        
        # add to extend 
        contacts.addBoundary(id=4, nContacts=[-1, 11, 0, 33, 44])
        np_test.assert_equal(contacts.maxBoundary, 4)
        np_test.assert_equal(contacts.maxSegment, 4)
        np_test.assert_equal(contacts.getN(boundaryId=2), [-99, 1, 0, 3, -99])
        np_test.assert_equal(contacts.getN(boundaryId=4), [-99, 11, 0, 33, 44])
        np_test.assert_equal(contacts.getN(segmentId=3), [-99, -99, 3, 0, 33])

        # add wo extending 
        contacts.addBoundary(id=1, nContacts=[-1, 7])
        np_test.assert_equal(contacts.maxBoundary, 4)
        np_test.assert_equal(contacts.maxSegment, 4)
        np_test.assert_equal(contacts.getN(boundaryId=1), [-99, 7, 0, 0, 0])
        np_test.assert_equal(contacts.getN(boundaryId=2), [-99, 1, 0, 3, -99])
        np_test.assert_equal(contacts.getN(boundaryId=4), [-99, 11, 0, 33, 44])
        np_test.assert_equal(contacts.getN(segmentId=3), [-99, 0, 3, 0, 33])

    def testCountContacted(self):
        """
        Tests contContactedBoundaries() and countCountedSegments()
        """

        # empty contacts
        contacts = Contact()
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[5]).mask, [True])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[6]).mask, [True])

        # one boundary
        contacts.addBoundary(id=2, nContacts=[-1, 1, 0, 3])
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[2]), [-99, 1, 0, 1])
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[]).mask, 
            [True, True, True, True])
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[1]).mask, 
            [True, True, True, True])
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[77]).mask, 
            [True, True, True, True])
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[2,77]), [-99, 1, 0, 1])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[1]), [-99, -99, 1])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[2]), [-99, -99, 0])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[3]), [-99, -99, 1])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[1,3]), [-99, -99, 2])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[]).mask, [True, True, True])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[77]).mask, [True, True, True])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[77,3]), [-99, -99, 1])

        # another boundary added
        contacts.addBoundary(id=4, nContacts=[-1, 11, 0, 33, 44])
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[2]), [-99, 1, 0, 1, 0])
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[4]), [-99, 1, 0, 1, 1])
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[2,4]), [-99, 2, 0, 2, 1])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[1]), [-99, -99, 1, -99, 1])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[3]), [-99, -99, 1, -99, 1])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[4]), [-99, -99, 0, -99, 1])
        np_test.assert_equal(
            contacts.countContactedSegments(ids=[3,4]), [-99, -99, 1, -99, 2])

        # test nested
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[[2,4]]), [-99, 1, 0, 1, 1])
        np_test.assert_equal(
            contacts.countContactedBoundaries(ids=[[2,4], 2]), 
            [-99, 2, 0, 2, 1])

    def testFindContacts(self):
        """
        Tests findContacts()
        """

        # no boundaries
        con = Contact()
        seg_data = common.image_1.data.round().astype(int)
        seg = Segment(
            data=seg_data, ids=list(range(1, 7)), clean=True)
        bound = Segment(data=numpy.zeros_like(seg_data))
        con.findContacts(
            segment=seg, boundary=bound)
        np_test.assert_equal(con.segments, [])
        np_test.assert_equal(con.boundaries, [])
        np_test.assert_equal(con._n.shape, [1, 7])

        # all segments and boundaries
        con = Contact()
        seg_data = common.image_1.data.round().astype(int)
        seg = Segment(
            data=seg_data, ids=list(range(1, 7)), clean=True)
        bound = Segment(
            data=common.bound_1.data, ids=[3, 4], clean=True)
        con.findContacts(segment=seg, boundary=bound, count=True)
        np_test.assert_equal(con.segments, list(range(1, 7)))
        np_test.assert_equal(con.boundaries, [3, 4])
        np_test.assert_equal(con.getN(boundaryId=3, segmentId=1), 2)
        np_test.assert_equal(con.getN(boundaryId=4, segmentId=1), 0)
        np_test.assert_equal(con.getN(boundaryId=3, segmentId=2), 0)
        np_test.assert_equal(con.getN(boundaryId=4, segmentId=2), 2)
        np_test.assert_equal(con.getN(boundaryId=3, segmentId=3), 0)
        np_test.assert_equal(con.getN(boundaryId=4, segmentId=3), 2)
        np_test.assert_equal(con.getN(boundaryId=3, segmentId=4), 1)
        np_test.assert_equal(con.getN(boundaryId=4, segmentId=4), 0)
        np_test.assert_equal(con.getN(boundaryId=3, segmentId=5), 0)
        np_test.assert_equal(con.getN(boundaryId=4, segmentId=5), 1)
        np_test.assert_equal(con.getN(boundaryId=3, segmentId=6), 0)
        np_test.assert_equal(con.getN(boundaryId=4, segmentId=6), 0)
        np_test.assert_equal(con._n.shape, [5, 7])
        np_test.assert_equal(con.findSegments(), [1, 2, 3, 4, 5, 6])
        np_test.assert_equal(con.findSegments(nBoundary=1), [1, 2, 3, 4, 5])
        np_test.assert_equal(con.findSegments(boundaryIds=[3]), [1, 4])
        np_test.assert_equal(con.findSegments(boundaryIds=[4]), [2, 3, 5])
        np_test.assert_equal(
            con.findSegments(nBoundary=2, mode='exact'), [])
        np_test.assert_equal(con.findBoundaries(segmentIds=[1, 4]), [3])
        np_test.assert_equal(
            con.findBoundaries(nSegment=3, mode='exact'), [4])

        # select segments, all boundaries
        seg_data = common.image_1.data.round().astype(int)
        seg = Segment(
            data=seg_data, ids=list(range(1, 7)), clean=True)
        bound = Segment(
            data=common.bound_1.data, ids=[3, 4], clean=True)
        #print(seg.data)
        #print(bound.data)
        con = Contact()
        con.findContacts(
            segment=seg, segmentIds=[1, 4, 5], boundary=bound, count=True)
        np_test.assert_equal(con.boundaries, [3, 4])
        np_test.assert_equal(con.getN(boundaryId=3, segmentId=1), 2)
        np_test.assert_equal(con.getN(boundaryId=4, segmentId=1), 0)
        np_test.assert_equal(con.getN(boundaryId=3, segmentId=2), 0)
        np_test.assert_equal(con.getN(boundaryId=4, segmentId=2), 2)
        np_test.assert_equal(con.getN(boundaryId=3, segmentId=4), 1)
        np_test.assert_equal(con.getN(boundaryId=4, segmentId=4), 0)
        np_test.assert_equal(con.getN(boundaryId=3, segmentId=5), 0)
        np_test.assert_equal(con.getN(boundaryId=4, segmentId=5), 1)
        # Note: segment id 6 is not included because it makes not contacts 
        np_test.assert_equal(con.segments, list(range(1, 6)))

        # select segments and boundaries
        seg_data = common.image_1.data.round().astype(int)
        seg = Segment(
            data=seg_data, ids=list(range(1, 7)), clean=True)
        bound = Segment(
            data=common.bound_1.data, ids=[3, 4], clean=True)
        con = Contact()
        con.findContacts(
            segment=seg, segmentIds=[4, 5], boundary=bound,
            boundaryIds=[3], count=True)
        np_test.assert_equal(con.boundaries, [3])
        # Note: contact to boundary 4 are not determined
        np_test.assert_equal(con.getN(boundaryId=3, segmentId=1), 2)
        np_test.assert_equal(con.getN(boundaryId=3, segmentId=2), 0)
        np_test.assert_equal(con.getN(boundaryId=3, segmentId=4), 1)
        np_test.assert_equal(con.getN(boundaryId=3, segmentId=5), 0)
        # Note: segment id 6 is not included because it makes not contacts 
        np_test.assert_equal(con.segments, list(range(1, 6))) 
        
        # select segments and boundaries
        seg_data = common.image_1.data.round().astype(int)
        seg = Segment(
            data=seg_data, ids=list(range(1, 7)), clean=True)
        bound = Segment(
            data=common.bound_1.data, ids=[3, 4], clean=True)
        con = Contact()
        con.findContacts(
            segment=seg, segmentIds=[4, 5], boundary=bound,
            boundaryIds=[4], count=True)
        np_test.assert_equal(con.boundaries, [4])
        # Note: contact to boundary 3 are not determined
        np_test.assert_equal(con.getN(boundaryId=4, segmentId=1), 0)
        np_test.assert_equal(con.getN(boundaryId=4, segmentId=2), 2)
        np_test.assert_equal(con.getN(boundaryId=4, segmentId=4), 0)
        np_test.assert_equal(con.getN(boundaryId=4, segmentId=5), 1)
        # Note: segment id 6 is not included because it makes not contacts 
        np_test.assert_equal(con.segments, list(range(1, 6)))
        

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestContact)
    unittest.TextTestRunner(verbosity=2).run(suite)
