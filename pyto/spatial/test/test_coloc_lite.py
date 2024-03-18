"""
Tests module coloc_lite. 

Further tests are in simple_examples.

# Author: Vladan Lucic
# $Id$
"""
__version__ = "$Revision$"

import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd
from pandas.testing import assert_frame_equal

from pyto.spatial.test import common
from pyto.spatial.particle_sets import ParticleSets
from pyto.spatial.bare_coloc import BareColoc
from pyto.spatial.coloc_lite import ColocLite

class TestColocLite(np_test.TestCase):
    """
    Tests coloc_Lite module
    """

    def setUp(self):
        """
        """
        # needed to make variables defined in this function available here
        common.make_coloc_tables_1tomo_2d()

    def test_make_coord_tabs(self):
        """Tests make_coord_tabs()
        """

        # no particles in set1
        bare = BareColoc()
        bare.coloc3 = [False, False, False]
        bare.particles3 = [[False, False, False], [], [False, False]]
        bare.coloc2 = [[False, False, False], [False, False, True]]
        bare.particles2 = [
            [[False, False, False], []], [[True, False, True], [False, True]]]
        particles = ParticleSets()
        particles.set_coords(
            tomo='alpha', set_name='set0', value=np.arange(6).reshape(3, 2))
        particles.set_coords(tomo='alpha', set_name='set1', value=np.array([]))
        particles.set_coords(
            tomo='alpha', set_name='set2', value=np.arange(4).reshape(2, 2))
        cl = ColocLite()
        actual = cl.make_coord_tabs(
            bare=bare, particles=particles, set_names=['set0', 'set1', 'set2'],
            tomo='alpha', distance=1)
        desired = pd.DataFrame(
            columns=['distance', 'tomo_id', 'set', 'x', 'y', 'pixel_nm'])
        np_test.assert_array_equal(actual['coloc3'].columns, desired.columns) 
        np_test.assert_equal(actual['coloc3'].size, 0) 
        np_test.assert_array_equal(
            actual['particles3'].columns, desired.columns)
        np_test.assert_equal(actual['particles3'].size, 0) 
        np_test.assert_array_equal(
            actual['coloc2'][0].columns, desired.columns)
        np_test.assert_equal(actual['coloc2'][0].size, 0) 
        np_test.assert_array_equal(
            actual['particles2'][0].columns, desired.columns)
        np_test.assert_equal(actual['particles2'][0].size, 0) 
        desired = pd.DataFrame(
            {'distance': 1, 'tomo_id': 'alpha',
             'set': 'coloc2', 'x': [4], 'y': [5], 'pixel_nm': None})
        np_test.assert_array_equal(
            actual['coloc2'][1].columns, desired.columns) 
        np_test.assert_equal(
            actual['coloc2'][1].values, desired.values) 
        desired = pd.DataFrame(
            {'distance': 1, 'tomo_id': 'alpha',
             'set': ['set0', 'set0', 'set2'], 'x': [0, 4, 2], 'y': [1, 5, 3],
             'pixel_nm': None})
        np_test.assert_array_equal(
            actual['particles2'][1].columns, desired.columns) 
        np_test.assert_equal(
            actual['particles2'][1].sort_values(by=['set', 'tomo_id']).values,
            desired.values) 
        
        # no particles, set to []
        bare = BareColoc()
        bare.coloc3 = []
        bare.particles3 = [[], [], []]
        bare.coloc2 = [[], []]
        bare.particles2 = [[[], []], [[], []]]
        particles = ParticleSets()
        particles.set_coords(tomo='alpha', set_name='set0', value=np.array([]))
        particles.set_coords(tomo='alpha', set_name='set1', value=np.array([]))
        particles.set_coords(tomo='alpha', set_name='set2', value=np.array([]))
        cl = ColocLite()
        actual = cl.make_coord_tabs(
            bare=bare, particles=particles, set_names=['set0', 'set1', 'set2'],
            tomo='alpha', distance=1)
        np_test.assert_equal(actual['coloc3'], None) 
        np_test.assert_equal(actual['particles3'], None) 
        np_test.assert_equal(actual['coloc2'], [None, None]) 
        np_test.assert_equal(actual['particles2'], [None, None]) 
        
        # no particles, set to None
        bare = BareColoc()
        bare.coloc3 = []
        bare.particles3 = [[], [], []]
        bare.coloc2 = [[], []]
        bare.particles2 = [[[], []], [[], []]]
        particles = ParticleSets()
        particles.set_coords(tomo='alpha', set_name='set0', value=None)
        particles.set_coords(tomo='alpha', set_name='set1', value=np.array([]))
        particles.set_coords(tomo='alpha', set_name='set2', value=None)
        cl = ColocLite()
        actual = cl.make_coord_tabs(
            bare=bare, particles=particles, set_names=['set0', 'set1', 'set2'],
            tomo='alpha', distance=1)
        np_test.assert_equal(actual['coloc3'], None) 
        np_test.assert_equal(actual['particles3'], None) 
        np_test.assert_equal(actual['coloc2'], [None, None]) 
        np_test.assert_equal(actual['particles2'], [None, None]) 
        
        # no particles, not even set
        bare = BareColoc()
        bare.coloc3 = []
        bare.particles3 = [[], [], []]
        bare.coloc2 = [[], []]
        bare.particles2 = [[[], []], [[], []]]
        particles = ParticleSets()
        particles.set_coords(
            tomo='alpha', set_name='set5', value=np.arange(6).reshape(3,2))
        particles.set_coords(
            tomo='bravo', set_name='set0', value=np.arange(6).reshape(3,2))
        particles.set_coords(
            tomo='bravo', set_name='set1', value=np.arange(6).reshape(3,2))
        particles.set_coords(
            tomo='bravo', set_name='set2', value=np.arange(6).reshape(3,2))
        cl = ColocLite()
        actual = cl.make_coord_tabs(
            bare=bare, particles=particles, set_names=['set0', 'set1', 'set2'],
            tomo='alpha', distance=1)
        np_test.assert_equal(actual['coloc3'], None) 
        np_test.assert_equal(actual['particles3'], None) 
        np_test.assert_equal(actual['coloc2'], [None, None]) 
        np_test.assert_equal(actual['particles2'], [None, None]) 
        
    def test_make_coord_tabs_multid(self):
        """Tests make_coord_tabs_multid()
        """

        bare_empty = BareColoc()
        bare_empty.coloc3 = []
        bare_empty.particles3 = [[], [], []]
        bare_empty.coloc2 = [[], []]
        bare_empty.particles2 = [[[], []], [[], []]]
        bare_multid = {2: bare_empty, 4: bare_empty}
        particles = ParticleSets()
        particles.set_coords(tomo='alpha', set_name='set0', value=None)
        particles.set_coords(tomo='alpha', set_name='set1', value=np.array([]))
        particles.set_coords(tomo='alpha', set_name='set2', value=np.array([]))
        cl = ColocLite()
        actual = cl.make_coord_tabs_multid(
            bare_multid=bare_multid, particles=particles,
            set_names=['set0', 'set1', 'set2'], tomo='alpha')
        np_test.assert_equal(actual['coloc3'], None) 
        np_test.assert_equal(actual['particles3'], None) 
        np_test.assert_equal(actual['coloc2'], [None, None]) 
        np_test.assert_equal(actual['particles2'], [None, None]) 
         

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestColocLite)
    unittest.TextTestRunner(verbosity=2).run(suite)
