"""

Tests module experiments
 
# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals

__version__ = "$Revision$"

import sys
from copy import copy, deepcopy
import pickle
import os.path
import unittest

import numpy as np
import numpy.testing as np_test 
import scipy 

from pyto.analysis.experiment import Experiment


class TestExperiment(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        self.exp = self.makeInstance()

    @classmethod
    def makeInstance(cls):
        """
        """
        exp = Experiment()
        exp.identifier = 'experiement'
        exp.idNames = ['id_1', 'id_2', 'id_5', 'id_6']
        exp.properties = set(
            ['identifier', 'scalar', 'idNames', 'ids', 'vector'])
        exp.indexed = set(['idNames', 'ids', 'vector'])
        exp.ids = [1, 2, 3, 4]
        exp.vector = np.asarray([4, 6, 8, 9])
        exp.scalar = 10
        return exp

    def test_getValue(self):
        """
        Test getValue()
        """

        np_test.assert_equal(
            self.exp.getValue(name='vector'), [4, 6, 8, 9])
        np_test.assert_equal(
            isinstance(self.exp.getValue(name='vector'), np.ndarray), True)
        np_test.assert_equal(self.exp.getValue(name='scalar'), 10)

    def test_setValue(self):
        """
        Test setValue()
        """

        exp_2 = self.makeInstance()

        exp_2.setValue(name='scalar_2', value=20)
        np_test.assert_equal(exp_2.getValue('scalar_2'), 20)
        np_test.assert_equal('scalar_2' not in exp_2.indexed, True)
        np_test.assert_equal('scalar_2' in exp_2.indexed, False)
        np_test.assert_equal('scalar_2' in exp_2.properties, True)
        
        exp_2.setValue(name='vector_2', value=[1, 2, 3, 4], indexed=True)
        np_test.assert_equal(exp_2.getValue(name='vector_2'), [1, 2, 3, 4])
        np_test.assert_equal(
            isinstance(exp_2.getValue(name='vector_2'), np.ndarray), True)
        np_test.assert_equal('vector_2' in exp_2.indexed, True)
        np_test.assert_equal('vector_2' in exp_2.properties, True)
            
    def test_doCorrelation(self):
        """
        Tests doCorrelation()

        Implicitly tested by test_observations.test_doCorrelation()
        """

        # empty data
        exp = self.makeInstance()
        exp.empty_1 = np.array([])
        exp.properties.add('empty_1')
        exp.indexed.add('empty_1')
        exp.empty_2 = np.array([])
        exp.properties.add('empty_2')
        exp.indexed.add('empty_2')
        corr = exp.doCorrelation(
            xName='empty_1', yName='empty_2', test='r', regress=True, out=None)
        np_test.assert_equal(corr.xData, np.array([]))
        np_test.assert_equal(corr.yData, np.array([]))
        np_test.assert_equal(corr.n, 0)
        np_test.assert_equal(corr.testValue, np.NaN)
        np_test.assert_equal(corr.confidence, np.NaN)
        np_test.assert_equal(corr.aRegress, np.NaN)
        np_test.assert_equal(corr.bRegress, np.NaN)
        
    def test_choose(self):
        """
        Test chose()
        """

        chosen = self.exp.choose(
            name='vector', idNames=['id_1', 'id_5', 'id_6'])
        np_test.assert_equal(chosen, [4, 8, 9])
        
    def testTransformByIds(self):
        """
        Tests transformByIds()
        """
        
        # same ids
        old_ids = [2, 1, 4, 3]
        old_values = [20, 10, 40, 30]
        new_ids = [3, 2, 1, 4]
        exp = Experiment()
        new_values = exp.transformByIds(
            ids=old_ids, new_ids=new_ids, values=old_values, default=100)
        np_test.assert_equal(new_values, [30, 20, 10, 40])

        # different ids
        old_ids = [4, 1, 7, 6]
        old_values = [40, 10, 70, 60]
        new_ids = [4, 2, 1, 5, 7]
        exp = Experiment()
        new_values = exp.transformByIds(
            ids=old_ids, new_ids=new_ids, values=old_values, default=100)
        np_test.assert_equal(new_values, [40, 100, 10, 100, 70])

        # same ids square form
        old_ids = [2, 1, 5, 3]
        new_ids = [3, 2, 1, 5]
        old_values = np.array([[22, 21, 25, 23],
                                  [12, 11, 15, 13],
                                  [52, 51, 55, 53],
                                  [32, 31, 35, 33]])
        exp = Experiment()
        new_values = exp.transformByIds(
            ids=old_ids, new_ids=new_ids, values=old_values, default=100, 
            mode='square')
        np_test.assert_equal(new_values, 
                             np.array([[33, 32, 31, 35],
                                          [23, 22, 21, 25],
                                          [13, 12, 11, 15],
                                          [53, 52, 51, 55]]))

        # different ids square form
        old_ids = [2, 1, 5, 3]
        new_ids = [3, 4, 1]
        old_values = np.array([[22, 21, 25, 23],
                                  [12, 11, 15, 13],
                                  [52, 51, 55, 53],
                                  [32, 31, 35, 33]])
        exp = Experiment()
        new_values = exp.transformByIds(
            ids=old_ids, new_ids=new_ids, values=old_values, default=100, 
            mode='square')
        np_test.assert_equal(new_values, 
                             np.array([[33, 100, 31],
                                          [100, 100, 100],
                                          [13, 100, 11]]))

        # different ids square form
        old_ids = [2, 1, 5, 3]
        new_ids = [3, 6, 1]
        old_values = np.array([[22, 21, 25, 23],
                                  [12, 11, 15, 13],
                                  [52, 51, 55, 53],
                                  [32, 31, 35, 33]])
        exp = Experiment()
        new_values = exp.transformByIds(
            ids=old_ids, new_ids=new_ids, values=old_values, default=100, 
            mode='square')
        np_test.assert_equal(new_values, 
                             np.array([[33, 100, 31],
                                          [100, 100, 100],
                                          [13, 100, 11]]))

        # same ids vector_pair form
        old_ids = [2, 1, 5, 3]
        new_ids = [3, 2, 1, 5]
        old_values = np.array([21, 25, 23, 15, 13, 53])
        exp = Experiment()
        new_values = exp.transformByIds(
            ids=old_ids, new_ids=new_ids, values=old_values, 
            mode='vector_pair')
        np_test.assert_equal(new_values, [23, 13, 53, 21, 25, 15]) 

        # different ids vector_pair form
        old_ids = [2, 1, 7, 4]
        new_ids = [4, 3, 2, 1]
        old_values = np.array([21, 72, 24, 17, 14, 74])
        exp = Experiment()
        new_values = exp.transformByIds(
            ids=old_ids, new_ids=new_ids, values=old_values, 
            mode='vector_pair')
        np_test.assert_equal(new_values, [-1, 24, 14, -1, -1, 21]) 

        # different ids vector_pair form
        old_ids = [2, 1, 7, 4]
        new_ids = [4, 8, 2, 1]
        old_values = np.array([21, 72, 24, 17, 14, 74])
        exp = Experiment()
        new_values = exp.transformByIds(
            ids=old_ids, new_ids=new_ids, values=old_values, 
            mode='vector_pair')
        np_test.assert_equal(new_values, [-1, 24, 14, -1, -1, 21]) 

        # different ids vector_pair form
        old_ids = [2, 1, 7, 4]
        new_ids = [5, 8, 21]
        old_values = np.array([21, 72, 24, 17, 14, 74])
        exp = Experiment()
        new_values = exp.transformByIds(
            ids=old_ids, new_ids=new_ids, values=old_values, default=100,
            mode='vector_pair')
        np_test.assert_equal(new_values, [100, 100, 100]) 



if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestExperiment)
    unittest.TextTestRunner(verbosity=2).run(suite)
