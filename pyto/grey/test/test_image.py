"""

Tests module image

# Author: Vladan Lucic
# $Id$
"""
from __future__ import unicode_literals

__version__ = "$Revision$"

from copy import copy, deepcopy
import unittest

import numpy as np
import numpy.testing as np_test
import scipy

from pyto.grey.image import Image
#import common


class TestImage(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        pass

    def testLimit(self):
        """
        Test limit()
        """

        image = self.makeImage()
        image.limit(limit=2, mode='std', size=3)
        np_test.assert_almost_equal(image.data[5, 5], 5)
        np_test.assert_almost_equal(image.data[7, 7], 0)
        desired = np.array([
                [10, 10, 0],
                [10, 10, 0],
                [0, 0, 0]])
        np_test.assert_almost_equal(image.data[0:3, 0:3], desired)

    def makeImage(self):
        """
        Returns an image
        """

        data = np.zeros(100).reshape(10, 10)
        data[0:3, 0:3] = 10
        data[5, 5] = 5
        data[7, 7] = 7

        image = Image(data=data)
        return image

    def test_normalize(self):
        """
        Tests normalize()
        """

        # uniform, no limits
        mean = 5.6
        std = 7.8
        data = np.random.random((10, 10, 10))
        im = Image(data=data)
        im.normalize(mean=mean, std=std)
        np_test.assert_almost_equal(im.data.mean(), mean)
        np_test.assert_almost_equal(im.data.std(), std)

        # beta, no limits
        mean = 5.6
        std = 7.8
        data = np.random.beta(a=2, b=3, size=(10, 10, 10))
        im = Image(data=data)
        im.normalize(mean=mean, std=std)
        np_test.assert_almost_equal(im.data.mean(), mean)
        np_test.assert_almost_equal(im.data.std(), std)

        # mean
        mean = 0
        std = None
        data = np.arange(5)
        im = Image(data=data)
        im.normalize(mean=mean, std=std)
        np_test.assert_almost_equal(im.data, [-2, -1, 0, 1, 2])

        # mean, std
        mean = 0
        std = 1
        data = np.arange(5)
        im = Image(data=data)
        im.normalize(mean=mean, std=std)
        np_test.assert_almost_equal(
            im.data, np.sqrt(2) * np.array([-1, -0.5, 0, 0.5, 1]))

        # mean, std, max
        mean = 2
        std = 1 / np.sqrt(2)
        max_limit = 2
        data = np.arange(5)
        im = Image(data=data)
        im.normalize(mean=mean, std=std, max_limit=max_limit)
        np_test.assert_almost_equal(
            im.data, np.array([1, 1.5, 2, 2, 2]))

        # mean, std, min, max
        mean = 2
        std = 1 / np.sqrt(2)
        min_limit = 1.2
        max_limit = 3
        data = np.arange(5)
        im = Image(data=data)
        im.normalize(
            mean=mean, std=std, min_limit=min_limit, max_limit=max_limit)
        np_test.assert_almost_equal(
            im.data, np.array([1.2, 1.5, 2, 2.5, 3]))


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestImage)
    unittest.TextTestRunner(verbosity=2).run(suite)
