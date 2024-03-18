"""
Tests module particle_sets

# Author: Vladan Lucic
# $Id$
"""
__version__ = "$Revision$"

import os
import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd
from pandas.testing import assert_frame_equal

from pyto.spatial.particle_sets import ParticleSets
from pyto.segmentation.labels import Labels

class TestParticleSets(np_test.TestCase):
    """
    Tests particle_sets module
    """

    def setUp(self):
        """
        """
        
        self.dir = os.path.dirname(os.path.realpath(__file__))

        self.ps = ParticleSets()
        self.alpha_u_coords = np.array([[1, 12], [2, 13], [3, 14]])
        self.bravo_u_coords = np.arange(8, dtype='int').reshape(4, 2)
        self.bravo_v_coords = np.arange(10, 14, dtype='int').reshape(2, 2)
        self.ps.set_coords(
            tomo='alpha', set_name='setU', value=self.alpha_u_coords)
        self.ps.set_coords(
            tomo='bravo', set_name='setU', value=self.bravo_u_coords)
        self.ps.set_coords(
            tomo='bravo', set_name='setV', value=self.bravo_v_coords)
        
        self.alpha_u_reg = np.ones((5, 10), dtype=int)
        self.bravo_u_reg = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
        self.bravo_v_reg = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
        self.alpha_pixel = 2.01
        self.bravo_pixel = 3.23
        self.alpha_u_reg_file = 'alpha_x_image.mrc'
        self.bravo_u_reg_file = 'bravo_y_image.mrc'
        self.bravo_v_reg_file = 'bravo_y_image.mrc'

        self.df_alpha_u = pd.DataFrame({
            self.ps.tomo_col: 'alpha', self.ps.set_name_col: 'setU',
            'x': self.alpha_u_coords[:, 0],
            'y': self.alpha_u_coords[:, 1], 
            self.ps.region_path_col: self.alpha_u_reg_file,
            self.ps.pixel_col: self.alpha_pixel})
        self.df_bravo_u = pd.DataFrame({
            self.ps.tomo_col: 'bravo', self.ps.set_name_col: 'setU',
            'x': self.bravo_u_coords[:, 0],
            'y': self.bravo_u_coords[:, 1], 
            self.ps.region_path_col: self.bravo_u_reg_file,
            self.ps.pixel_col: self.bravo_pixel})
        self.df_bravo_v = pd.DataFrame({
            self.ps.tomo_col: 'bravo', self.ps.set_name_col: 'setV',
            'x': self.bravo_v_coords[:, 0],
            'y': self.bravo_v_coords[:, 1], 
            self.ps.region_path_col: self.bravo_v_reg_file,
            self.ps.pixel_col: self.bravo_pixel})
        self.df_full = pd.concat(
            [self.df_alpha_u, self.df_bravo_u, self.df_bravo_v],
            ignore_index=True)

        self.n_particles = pd.DataFrame({
            self.ps.tomo_col: ['alpha', 'bravo'],
            'setU': [len(self.alpha_u_coords), len(self.bravo_u_coords)],
            'setV': [0, len(self.bravo_v_coords)]})

        # index
        self.alpha_u_index = [41, 51, 61]
        self.bravo_u_index = [31, 32, 33, 34]
        self.bravo_v_index = [25, 26]
        index = self.alpha_u_index + self.bravo_u_index + self.bravo_v_index
        self.df_full_index = self.df_full.copy()
        self.df_full_index['index'] = index
        self.df_full_index = self.df_full_index.set_index('index')
        
    def test_coords(self):
        """Test get_coords() and set_coords()
        """

        # existing
        np_test.assert_array_equal(
            self.ps.get_coords(tomo='alpha', set_name='setU'),
            self.alpha_u_coords)
        np_test.assert_equal(
            isinstance(
                self.ps.get_coords(tomo='alpha', set_name='setU'), np.ndarray),
            True)
        np_test.assert_array_equal(
            self.ps.get_coords(tomo='bravo', set_name='setU'),
            self.bravo_u_coords)
        np_test.assert_array_equal(
            self.ps.get_coords(tomo='bravo', set_name='setV'),
            self.bravo_v_coords)

        # tomo set name combination does not exist
        np_test.assert_equal(
            self.ps.get_coords(tomo='alpha', set_name='setV') is None,
            True)

        # tomo or set name does not exist
        with np_test.assert_raises(ValueError):
            self.ps.get_coords(tomo='charlie', set_name='setV')
        with np_test.assert_raises(ValueError):
            self.ps.get_coords(tomo='alpha', set_name='Z')
        np_test.assert_array_equal(
            (self.ps.get_coords(tomo='charlie', set_name='setV', catch=True)
             is None), True)
        np_test.assert_array_equal(
            (self.ps.get_coords(tomo='alpha', set_name='Z', catch=True)
             is None), True)
        
        # all tomos
        desired = {'alpha': self.alpha_u_coords, 'bravo': self.bravo_u_coords}
        np_test.assert_array_equal(
            self.ps.get_coords(tomo=None, set_name='setU'), desired)
        desired = {'alpha': None, 'bravo': self.bravo_v_coords}
        np_test.assert_array_equal(
            self.ps.get_coords(tomo=None, set_name='setV'), desired)

        # all set names
        desired = {'setU': self.alpha_u_coords, 'setV': None}
        np_test.assert_array_equal(
            self.ps.get_coords(tomo='alpha', set_name=None), desired)
        desired = {'setV': self.bravo_v_coords, 'setU': self.bravo_u_coords}
        np_test.assert_array_equal(
            self.ps.get_coords(tomo='bravo', set_name=None), desired)

        # different order
        desired = {'bravo': self.bravo_u_coords, 'alpha': self.alpha_u_coords}
        np_test.assert_array_equal(
            self.ps.get_coords(tomo=['bravo', 'alpha'],
                               set_name='setU'), desired)
        desired = {'bravo': self.bravo_v_coords, 'alpha': None}
        np_test.assert_array_equal(
            self.ps.get_coords(tomo=('bravo', 'alpha'), set_name='setV'),
            desired)
        desired = {'setV': None, 'setU': self.alpha_u_coords}
        np_test.assert_array_equal(
            self.ps.get_coords(tomo='alpha', set_name=['setV', 'setU']),
            desired)
        desired = {'setV': self.bravo_v_coords, 'setU': self.bravo_u_coords}
        np_test.assert_array_equal(
            self.ps.get_coords(tomo='bravo', set_name=('setV', 'setU')),
            desired)     
        
    def test_index(self):
        """Test get_index() and set_index()
        """

        # table wo index
        np_test.assert_equal(
            self.ps.get_index(tomo='alpha', set_name='setU') is None, True)

        # set index and check
        self.ps.set_index(
            tomo='alpha', set_name='setU', value=self.alpha_u_index)
        np_test.assert_array_equal(
            self.ps.get_index(
                tomo='alpha', set_name='setU'), self.alpha_u_index)

        # set index and check
        self.ps.set_index(
            tomo='bravo', set_name='setV', value=self.bravo_v_index)
        np_test.assert_array_equal(
            self.ps.get_index(
                tomo='bravo', set_name='setV'), self.bravo_v_index)
        np_test.assert_equal(
            self.ps.get_index(tomo='bravo', set_name='setU') is None, True)
        np_test.assert_array_equal(
            self.ps.get_index(
                tomo='alpha', set_name='setU'), self.alpha_u_index)
        
    def test_n_points(self):
        """Tests get_n_points()
        """

        # individual
        np_test.assert_array_equal(
            self.ps.get_n_points(tomo='alpha', set_name='setU'),
            self.alpha_u_coords.shape[0])
        np_test.assert_array_equal(
            self.ps.get_n_points(tomo='bravo', set_name='setU'),
            self.bravo_u_coords.shape[0])
        np_test.assert_array_equal(
            self.ps.get_n_points(tomo='bravo', set_name='setV'),
            self.bravo_v_coords.shape[0])

        # multiple
        np_test.assert_array_equal(
            self.ps.get_n_points(tomo='alpha', set_name=None),
            self.alpha_u_coords.shape[0])
        np_test.assert_array_equal(
            self.ps.get_n_points(tomo='bravo', set_name=None),
            self.bravo_u_coords.shape[0] + self.bravo_v_coords.shape[0])
        np_test.assert_array_equal(
            self.ps.get_n_points(tomo=['alpha', 'bravo'], set_name='setU'),
            (self.alpha_u_coords.shape[0] + self.bravo_u_coords.shape[0]))
        np_test.assert_array_equal(
            self.ps.get_n_points(tomo=['alpha', 'bravo'], set_name=None),
            (self.alpha_u_coords.shape[0] + self.bravo_u_coords.shape[0]
             + self.bravo_v_coords.shape[0]))
        np_test.assert_array_equal(
            self.ps.get_n_points(tomo=['bravo', 'charlie'], set_name=None),
            self.bravo_u_coords.shape[0] + self.bravo_v_coords.shape[0])
        np_test.assert_array_equal(
            self.ps.get_n_points(tomo=None, set_name='setU'),
            self.alpha_u_coords.shape[0] + self.bravo_u_coords.shape[0])
        np_test.assert_array_equal(
            self.ps.get_n_points(tomo=None, set_name='setV'),
            self.bravo_v_coords.shape[0])
        np_test.assert_array_equal(
            self.ps.get_n_points(tomo=None, set_name=None),
            (self.alpha_u_coords.shape[0] + self.bravo_u_coords.shape[0]
             + self.bravo_v_coords.shape[0]))
        np_test.assert_array_equal(
            self.ps.get_n_points(
                tomo=['bravo', 'charlie'], set_name=['setU', 'Z']),
            self.bravo_u_coords.shape[0] + self.bravo_v_coords.shape[0])

        # changed order
        np_test.assert_array_equal(
            self.ps.get_n_points(tomo='bravo', set_name=['setV', 'setU']),
            self.bravo_u_coords.shape[0] + self.bravo_v_coords.shape[0])
        np_test.assert_array_equal(
            self.ps.get_n_points(tomo=('bravo', 'alpha'), set_name='setU'),
            self.alpha_u_coords.shape[0] + self.bravo_u_coords.shape[0])
         
    def test_region_path(self):
        """Tests set_region_path() and get_region_path()
        """

        # non-existing combination before setting region_path
        np_test.assert_array_equal(
            self.ps.get_region_path(tomo='bravo', set_name='setU') is None,
            True)

        self.ps.set_region_path(
            tomo='alpha', set_name='setU', value=self.alpha_u_reg_file)
        self.ps.set_region_path(
            tomo='bravo', set_name='setV', value=self.bravo_v_reg_file)

        # existing
        np_test.assert_array_equal(
            self.ps.get_region_path(tomo='alpha', set_name='setU'),
            self.alpha_u_reg_file)
        np_test.assert_array_equal(
            self.ps.get_region_path(tomo='bravo', set_name='setV'),
            self.bravo_v_reg_file)

        # non-existing combination
        np_test.assert_array_equal(
            self.ps.get_region_path(tomo='bravo', set_name='setU') is None,
            True)

        # non existing tomo or set
        with np_test.assert_raises(ValueError):
            self.ps.get_region_path(tomo='charlie', set_name='setV')
        with np_test.assert_raises(ValueError):
            self.ps.get_region_path(tomo='alpha', set_name='Z')
      
    def test_region(self):
        """Tests set_region() and get_region()
        """

        # non-existing combination of tomo and set_name, before setting region
        np_test.assert_array_equal(
            self.ps.get_region(tomo='bravo', set_name='setU') is None, True)

        # set region
        self.ps.set_region(
            tomo='alpha', set_name='setU', value=self.alpha_u_reg)
        self.ps.set_region(
            tomo='bravo', set_name='setV', value=self.bravo_v_reg)

        # test existing
        np_test.assert_array_equal(
            self.ps.get_region(tomo='alpha', set_name='setU'), self.alpha_u_reg)
        np_test.assert_array_equal(
            self.ps.get_region(tomo='bravo', set_name='setV'), self.bravo_v_reg)

        # test non-existing combination of tomo and set_name
        np_test.assert_array_equal(
            self.ps.get_region(tomo='bravo', set_name='setU') is None, True)

        # test non existing tomo or set
        with np_test.assert_raises(ValueError):
            self.ps.get_region(tomo='charlie', set_name='setV')
        with np_test.assert_raises(ValueError):
            self.ps.get_region(tomo='alpha', set_name='Z')

    def test_region_from_file(self):
        """Tests get_region() when region does not exist by region_path does
        """
        
        # non-existing combination of tomo and set_name, before setting region
        np_test.assert_array_equal(
            self.ps.get_region(tomo='bravo', set_name='setU') is None, True)

        # read witout saving (need to make abs path for testing)
        corrected_path = os.path.join(
            os.path.dirname(__file__), 'data', self.bravo_v_reg_file)
        self.ps.set_region_path(
            tomo='bravo', set_name='setV', value=corrected_path)
        desired = self.ps.get_region(tomo='bravo', set_name='setV')
        np_test.assert_array_equal(desired, self.bravo_v_reg)
        np_test.assert_array_equal(
            self.ps.get_pixel_nm(tomo='bravo') is None, True)

        # read with saving
        desired = self.ps.get_region(
            tomo='bravo', set_name='setV', save_from_file=True)
        np_test.assert_almost_equal(
            self.ps.get_pixel_nm(tomo='bravo'), self.bravo_pixel)

        # non existing tomo
        with np_test.assert_raises(ValueError):
            self.ps.get_pixel_nm(tomo='charlie')
                
    def test_pixel_nm(self):
        """Tests pixel_nm
        """

        self.ps.set_pixel_nm(tomo='alpha', value=1.23)
        np_test.assert_equal(
            self.ps.pixel_nm, {'alpha': 1.23, 'bravo': None})
        self.ps.set_pixel_nm(tomo='bravo', value=3.23)
        np_test.assert_equal(
            self.ps.pixel_nm, {'alpha': 1.23, 'bravo': 3.23})

    def test_get_pixel_nm(self):
        """Tests set_pixel_nm() and get_pixel_nm()
        """

        self.ps.set_pixel_nm(tomo='alpha', value=1.23)
        np_test.assert_equal(self.ps.get_pixel_nm(tomo='alpha'), 1.23)
        np_test.assert_equal(self.ps.get_pixel_nm(tomo='bravo') is None, True)
        with np_test.assert_raises(ValueError):
            self.ps.get_pixel_nm(tomo='charlie')        
            
    def test_tomos(self):
        """Tests tomos
        """

        self.ps.set_coords(
            tomo='alpha', set_name='setU', value=self.alpha_u_coords)
        self.ps.set_coords(
            tomo='bravo', set_name='setU', value=self.bravo_u_coords)
        self.ps.set_coords(
            tomo='bravo', set_name='setV', value=self.bravo_v_coords)
        np_test.assert_array_equal(self.ps.tomos, ['alpha', 'bravo'])
        
    def test_set_names(self):
        """Tests set_names
        """

        actual = self.ps.set_names
        actual.sort()
        np_test.assert_array_equal(actual, ['setU', 'setV'])

    def test_data_df(self):
        """Tests data_df getter and setter
        """

        # to dataframe
        self.ps.set_pixel_nm(tomo='alpha', value=self.alpha_pixel)
        self.ps.set_pixel_nm(tomo='bravo', value=self.bravo_pixel)
        self.ps.set_region_path(tomo='alpha', value=self.alpha_u_reg_file)
        self.ps.set_region_path(tomo='bravo', value=self.bravo_u_reg_file)
        actual = self.ps.data_df.sort_values(
            by=['tomo_id', 'set'], ignore_index=True)
        desired = self.df_full.sort_values(
            by=['tomo_id', 'set'], ignore_index=True)
        assert_frame_equal(
            actual, desired, check_index_type=False, check_like=True)

        # to dataframe, with index
        self.ps.set_index(
            tomo='alpha', set_name='setU', value=self.alpha_u_index)
        with np_test.assert_raises(ValueError):
            self.ps.data_df
        self.ps.set_index(
            tomo='bravo', set_name='setV', value=self.bravo_v_index)
        with np_test.assert_raises(ValueError):
            self.ps.data_df
        self.ps.set_index(
            tomo='bravo', set_name='setU', value=self.bravo_u_index)
        actual = self.ps.data_df.sort_index()
        desired = self.df_full_index.sort_index()
        assert_frame_equal(
            actual, desired, check_index_type=False, check_like=True)
        np_test.assert_array_equal(actual, desired)
        
        # from dataframe, no index
        ps = ParticleSets(index=False)
        ps.data_df = self.df_full
        np_test.assert_equal(
            ps.get_coords(tomo='alpha', set_name='setU'), self.alpha_u_coords) 
        np_test.assert_equal(
            ps.get_region_path(tomo='alpha', set_name='setU'),
            self.alpha_u_reg_file) 
        np_test.assert_equal(
            ps.get_pixel_nm(tomo='alpha'), self.alpha_pixel) 
        np_test.assert_equal(
            ps.get_coords(tomo='bravo', set_name='setU'), self.bravo_u_coords) 
        np_test.assert_equal(
            ps.get_region_path(tomo='bravo', set_name='setU'),
            self.bravo_u_reg_file) 
        np_test.assert_equal(
            type(ps.get_region_path(tomo='bravo', set_name='setU')),
            type(self.bravo_u_reg_file)) 
        np_test.assert_equal(
            ps.get_pixel_nm(tomo='bravo'), self.bravo_pixel) 
        np_test.assert_equal(
            isinstance(ps.get_pixel_nm(tomo='bravo'),
                       (int, float, np.number)), True)
        np_test.assert_equal(
            ps.get_coords(tomo='bravo', set_name='setV'), self.bravo_v_coords) 
        np_test.assert_equal(
            ps.get_region_path(tomo='bravo', set_name='setV'),
            self.bravo_v_reg_file) 
        np_test.assert_equal(
            ps.get_pixel_nm(tomo='bravo'), self.bravo_pixel)

        # from dataframe, with index
        ps = ParticleSets()
        ps.data_df = self.df_full_index
        np_test.assert_equal(
            ps.get_coords(tomo='alpha', set_name='setU'), self.alpha_u_coords) 
        np_test.assert_equal(
            ps.get_index(tomo='alpha', set_name='setU'), self.alpha_u_index) 
        np_test.assert_equal(
            ps.get_region_path(tomo='alpha', set_name='setU'),
            self.alpha_u_reg_file) 
        np_test.assert_equal(
            ps.get_pixel_nm(tomo='alpha'), self.alpha_pixel) 
        np_test.assert_equal(
            ps.get_coords(tomo='bravo', set_name='setU'), self.bravo_u_coords) 
        np_test.assert_equal(
            ps.get_index(tomo='bravo', set_name='setU'), self.bravo_u_index) 
        np_test.assert_equal(
            ps.get_region_path(tomo='bravo', set_name='setU'),
            self.bravo_u_reg_file) 
        np_test.assert_equal(
            type(ps.get_region_path(tomo='bravo', set_name='setU')),
            type(self.bravo_u_reg_file)) 
        np_test.assert_equal(
            ps.get_pixel_nm(tomo='bravo'), self.bravo_pixel) 
        np_test.assert_equal(
            isinstance(ps.get_pixel_nm(tomo='bravo'),
                       (int, float, np.number)), True)
        np_test.assert_equal(
            ps.get_coords(tomo='bravo', set_name='setV'), self.bravo_v_coords) 
        np_test.assert_equal(
            ps.get_index(tomo='bravo', set_name='setV'), self.bravo_v_index) 
        np_test.assert_equal(
            ps.get_region_path(tomo='bravo', set_name='setV'),
            self.bravo_v_reg_file) 
        np_test.assert_equal(
            ps.get_pixel_nm(tomo='bravo'), self.bravo_pixel) 

        # no points
        ps = ParticleSets()
        ps.set_coords(
            tomo='charlie', set_name='empty',
            value=np.array([]).reshape(0, 3))
        desired_cols = [
            'tomo_id', 'set', 'x', 'y', 'z', 'region_path', 'pixel_nm']
        np_test.assert_array_equal(
            ps.data_df.columns, desired_cols)
        
        # no points in two sets
        ps = ParticleSets()
        ps.set_coords(
            tomo='charlie', set_name='empty_1',
            value=np.array([]).reshape(0, 3))
        ps.set_coords(
            tomo='charlie', set_name='empty_2',
            value=np.array([]))
        desired_cols = [
            'tomo_id', 'set', 'x', 'y', 'z', 'region_path', 'pixel_nm']
        np_test.assert_array_equal(
            ps.data_df.columns, desired_cols)
        
        # no points
        ps = ParticleSets()
        ps.set_coords(
            tomo='charlie', set_name='empty',
            value=np.array([]))
        desired_cols = [
            'tomo_id', 'set', 'x', 'y', 'z', 'region_path', 'pixel_nm']
        np_test.assert_array_equal(ps.data_df is None, True)
        
    def test_add_data_df(self):
        """Tests add_data_df()
        """

        # add to empty
        ps = ParticleSets()
        ps.add_data_df(self.df_alpha_u)
        assert_frame_equal(ps.data_df, self.df_alpha_u)

        # add further
        ps = ParticleSets()
        ps.add_data_df(self.df_alpha_u)
        ps.add_data_df(self.df_bravo_u)
        ps.add_data_df(self.df_bravo_v)
        ps_data_df = ps.data_df.sort_values(
            by=['tomo_id', 'set']).reset_index(drop=True)
        assert_frame_equal(ps_data_df, self.df_full)       
        #assert_frame_equal(ps.data_df, self.df_full)       
       
    def test_get_coord_cols(self):
        """Tests get_coord_cols()
        """

        n_dim, coord_cols = self.ps.get_coord_cols(
            coords=self.alpha_u_coords)
        np_test.assert_equal(n_dim, 2)
        np_test.assert_equal(coord_cols, ['x', 'y'])

        n_dim, coord_cols = self.ps.get_coord_cols(
            coords=np.arange(20).reshape(5, 4))
        np_test.assert_equal(n_dim, 4)
        np_test.assert_equal(
            coord_cols, ['coord_0', 'coord_1', 'coord_2', 'coord_3'])

        # empty
        ps = ParticleSets()
        n_dim, coord_cols = ps.get_coord_cols(
            coords=np.array([]).reshape(0, 3))
        np_test.assert_equal(n_dim, 3)
        np_test.assert_equal(coord_cols, ['x', 'y', 'z'])

        # empty np.array([])
        ps = ParticleSets()
        n_dim, coord_cols = ps.get_coord_cols(
            coords=np.array([]))
        np_test.assert_equal(n_dim, None)
        np_test.assert_equal(coord_cols, None)

        # empty None
        ps = ParticleSets()
        n_dim, coord_cols = ps.get_coord_cols(coords=None)
        np_test.assert_equal(n_dim, None)
        np_test.assert_equal(coord_cols, None)

    def test_get_n_particles(self):
        """Tests get_n_particles()
        """

        actual = self.ps.get_n_particles()
        assert_frame_equal(actual, self.n_particles)
        
        actual = self.ps.get_n_particles(set_names=['setV'])
        assert_frame_equal(actual, self.n_particles[[self.ps.tomo_col, 'setV']])

    def test_in_region(self):
        """Tests in_region()
        """

        # use particles and regions from TestMultiParticleSets
        from pyto.spatial.multi_particle_sets import MultiParticleSets
        from pyto.spatial.test.test_multi_particle_sets \
            import TestMultiParticleSets
        tmps = TestMultiParticleSets()
        tmps.setUp()
        mps = MultiParticleSets()
        mps.tomos = tmps.tomo_init.copy()
        mps.tomos[mps.region_col] = mps.tomos[mps.region_col].map(
            lambda x: os.path.join(self.dir, 'particles', 'regions', x))
        mps.particles = tmps.particle_final

        # all particles in
        pset = mps.to_particle_sets()
        actual = pset.in_region()
        np_test.assert_array_equal(
            actual[pset.in_region_col], mps.particles.shape[0]*[True])
        
        
         
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestParticleSets)
    unittest.TextTestRunner(verbosity=2).run(suite)
