"""
Tests class ClassificationResults

# Author: Vladan Lucic
# $Id$
"""

__version__ = "$Revision$"


import warnings
import unittest

import numpy as np
import numpy.testing as np_test
import scipy as sp
import pandas as pd

from pyto.clustering.classification_results import ClassificationResults


class TestClassificationResults(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        Make instances
        """
        df = pd.DataFrame(
            {'true': [0, 0, 1, 1, 2], 'cl1': [0, 1, 1, 2, 2],
             'cl2': [2, 2, 1, 1, 0], 'cl3': [0, 0, 0, 0, 2],
             'cl4': [0, 0, 1, 1, 2]})
        self.cr_1 = ClassificationResults(data=df, true_label='true')
        self.accuracy_true_1 = [3/5, 2/5, 3/5, 1]
        self.accuracy_pairs_1 = pd.DataFrame(
            {'cl1': [1, 1/5, 2/5, 3/5], 'cl2': [1/5, 1, 0, 2/5],
             'cl3': [2/5, 0, 1, 3/5], 'cl4': [3/5, 2/5, 3/5, 1]},
            index=['cl1', 'cl2', 'cl3', 'cl4'])

    def test_add_classifications(self):
        """Tests add_classifications
        """

        cr = ClassificationResults()
        cr.add_classification(
            id_='cl1', true=self.cr_1.data['true'],
            predict=self.cr_1.data['cl1'])
        cr.add_classification(
            id_='cl2', true=self.cr_1.data['true'],
            predict=self.cr_1.data['cl2'])
        cr.add_classification(
            id_='cl3', true=self.cr_1.data['true'],
            predict=self.cr_1.data['cl3'])
        cr.add_classification(
            id_='cl4', true=self.cr_1.data['true'],
            predict=self.cr_1.data['cl4'])
        np_test.assert_equal(cr.data.equals(self.cr_1.data), True)
        
    def test_accuracy_score(self):
        """Tests accuracy_score()
        """

        # single comparison
        np_test.assert_almost_equal(
            self.cr_1.accuracy_score(id_='cl1', id_ref='true'),
            self.accuracy_true_1[0])
        np_test.assert_almost_equal(
            self.cr_1.accuracy_score(id_='cl3'), self.accuracy_true_1[2])

        # multi vs one
        np_test.assert_almost_equal(
            self.cr_1.accuracy_score(), self.accuracy_true_1)
        
        # pairs
        np_test.assert_equal(
            self.cr_1.accuracy_score(pairs=True).equals(self.accuracy_pairs_1),
            True)
        
        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(
        TestClassificationResults)
    unittest.TextTestRunner(verbosity=2).run(suite)
