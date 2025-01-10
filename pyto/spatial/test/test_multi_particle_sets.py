"""
Tests module multi_particle_sets

# Author: Vladan Lucic
# $Id$
"""
__version__ = "$Revision$"

import os
import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from pyto.spatial.test import common
from pyto.spatial.multi_particle_sets import MultiParticleSets
from pyto.spatial.test.test_particle_sets import TestParticleSets


class TestMultiParticleSets(np_test.TestCase):
    """
    Tests multi_particle_sets module and sume of MUltiParticleSets supers
    """

    def setUp(self):
        """
        """
        #
        self.dir = os.path.dirname(os.path.realpath(__file__))
        
        self.mps = MultiParticleSets()
        self.tomo_ids = ['alpha', 'bravo', 'charlie']
        self.pixel_size_nm = [2, 1, 0.5]
        self.coord_bin = [4, 1, 1.]
        self.coord_bin_dict = dict(zip(self.tomo_ids, self.coord_bin))
        tomo_star_dict = {
            self.mps.tomo_id_col: self.tomo_ids,
            self.mps.region_col: [
                os.path.join(
                    self.dir, 'particles/regions/syn_alpha_bin2_crop_seg.mrc'),
                #'../regions/syn_alpha_bin2_crop_seg.mrc',
                os.path.join(
                    self.dir, 'particles/regions/syn_bravo_bin2_crop_seg.mrc'),
                os.path.join(
                    self.dir,
                    'particles/regions/syn_charlie_bin2_crop_seg.mrc')],
            self.mps.region_origin_labels[0]: [100, 10, 60],
            self.mps.region_origin_labels[1]: [60, 20, 50],
            self.mps.region_origin_labels[2]: [40, 30, 40], 
            self.mps.pixel_nm_col: self.pixel_size_nm,
            self.mps.coord_bin_col: self.coord_bin,
            'psSegRot': 0, 'psSegTilt': 0, 'psSegPsi': 0}
        self.tomo_star = pd.DataFrame(tomo_star_dict)

        self.tomo_star_rel = self.tomo_star.copy()
        self.tomo_star_rel[self.mps.region_col] = [
            '../regions/syn_alpha_bin2_crop_seg.mrc',
            '../regions/syn_bravo_bin2_crop_seg.mrc',
            '../regions/syn_charlie_bin2_crop_seg.mrc']            
        
        col_rename = dict([(old, new) for old, new in zip(
                self.mps.region_origin_labels, self.mps.region_offset_cols)])
        self.tomo_init = self.tomo_star.rename(columns=col_rename)
        self.tomo_init_rel = self.tomo_star_rel.rename(columns=col_rename)

        self.reg_shapes = pd.DataFrame({'tomo_id': self.tomo_ids})
        self.reg_shapes[self.mps.region_shape_cols] = np.array(
            [[200, 150, 100],
             [200, 150, 100],
             [150, 100, 50]])
        
        tomo_off_0_dict = tomo_star_dict.copy()
        tomo_off_0_dict['region_offset_x'] = 0.
        tomo_off_0_dict['region_offset_y'] = 0.
        tomo_off_0_dict['region_offset_z'] = 0.
        self.tomo_off_0 = pd.DataFrame(tomo_off_0_dict)
        
        tomo_off_dict = tomo_star_dict.copy()
        tomo_off_dict['region_offset_x'] = [100, 10, 60]
        tomo_off_dict['region_offset_y'] = [60, 20, 50]
        tomo_off_dict['region_offset_z'] = [40, 30, 40]
        self.tomo_off = pd.DataFrame(tomo_off_dict)

        self.region_bins = [2, 1, 4]
        tomo_bin_dict = tomo_star_dict.copy()
        tomo_bin_dict[self.mps.region_bin_col] = self.region_bins
        self.tomo_bin = pd.DataFrame(tomo_bin_dict)

        self.region_id = [1, 2, 3]
        tomo_regid_dict = tomo_star_dict.copy()
        tomo_regid_dict['region_id'] = self.region_id
        self.tomo_off_regid = pd.DataFrame(tomo_regid_dict)

        # full tomos table
        tomos_dict = tomo_star_dict.copy()
        tomos_dict.update(tomo_off_dict)
        tomos_dict.update(tomo_bin_dict)
        tomos_dict.update(tomo_regid_dict)
        self.tomos = pd.DataFrame(tomos_dict)
        
        particle_star_origin_dict = {
            self.mps.tomo_id_col: ['alpha', 'alpha', 'alpha', 'charlie'],
            self.mps.particle_id_col: [11, 12, 13, 31],
            self.mps.coord_labels[0]: [120, 130, 140, 80],
            self.mps.coord_labels[1]: [70, 80, 90, 60],
            self.mps.coord_labels[2]: [40, 50, 60, 41],
            self.mps.particle_rotation_labels[0]: [99, 9, 79, 59],
            self.mps.particle_rotation_labels[1]: [-99, -9, -79, -59],
            self.mps.particle_rotation_labels[2]: [98, 8, 78, 58],
            self.mps.particle_rotation_labels[3]: [-98, -8, -78, -58],
            self.mps.particle_rotation_labels[4]: [97, 7, 77, 57],
            self.mps.origin_labels[0]: [2, -1, 3, -4],
            self.mps.origin_labels[1]: [1, -2, 2, 4],
            self.mps.origin_labels[2]: [-1, 2, 3, 2],
            self.mps.class_name_col: 'Class A',
            self.mps.class_number_col: [11, 12, 13, 11],
            self.mps.pixel_nm_col: [2., 2, 2, 0.5]}
        self.particle_star_origin = pd.DataFrame(particle_star_origin_dict)

        particle_star_angst_dict = {
            self.mps.tomo_id_col: ['alpha', 'alpha', 'alpha', 'charlie'],
            self.mps.particle_id_col: [11, 12, 13, 31],
            self.mps.coord_labels[0]: [120, 130, 140, 80],
            self.mps.coord_labels[1]: [70, 80, 90, 60],
            self.mps.coord_labels[2]: [40, 50, 60, 41],
            self.mps.angst_labels[0]: [40, -20, 60, -20],
            self.mps.angst_labels[1]: [20, -40, 40, 20],
            self.mps.angst_labels[2]: [-20, 40, 60, 10],
            self.mps.class_name_col: 'Class A',
            self.mps.class_number_col: [11, 12, 13, 11],
            self.mps.pixel_nm_col: [2., 2, 2, 0.5]}
        self.particle_star_angst = pd.DataFrame(particle_star_angst_dict)

        particle_orig_dict = {
            self.mps.orig_coord_cols[0]: [118, 131, 137, 84],
            self.mps.orig_coord_cols[1]: [69, 82, 88, 56],
            self.mps.orig_coord_cols[2]: [41, 48, 57, 39]}
        
        particle_origin_dict = particle_star_origin_dict.copy()
        particle_origin_dict.update(particle_orig_dict)
        self.particle_origin = pd.DataFrame(particle_origin_dict)        
        particle_angst_dict = particle_star_angst_dict.copy()
        particle_angst_dict.update(particle_orig_dict)
        self.particle_angst = pd.DataFrame(particle_angst_dict)
       
        #     
        particle_regframe_dict = {
            self.mps.orig_coord_reg_frame_cols[0]: [136, 162, 174, -39],
            self.mps.orig_coord_reg_frame_cols[1]: [78, 104, 116, -36],
            self.mps.orig_coord_reg_frame_cols[2]: [42, 56, 74, -30]}
        self.particle_regframe = self.particle_origin.join(
            pd.DataFrame(particle_regframe_dict))

        particle_proj_dict = {
            self.mps.coord_reg_frame_cols[0]: [174, 174, 174, 20],
            self.mps.coord_reg_frame_cols[1]: [78, 104, 116, 50],
            self.mps.coord_reg_frame_cols[2]: [74, 74, 74, 40],
            self.mps.coord_init_frame_cols[0]: [137, 137, 137, 320],
            self.mps.coord_init_frame_cols[1]: [69, 82, 88, 400],
            self.mps.coord_init_frame_cols[2]: [57, 57, 57, 320],
            self.mps.keep_col: True}
        self.particle_final = self.particle_regframe.join(
            pd.DataFrame(particle_proj_dict))
           
        self.particle_final_excl_30nm = self.particle_final.copy()
        self.particle_final_excl_30nm[self.mps.keep_col] = [
            True, True, False, True]
        self.particle_final_excl_30nm_before = self.particle_final.copy()
        self.particle_final_excl_30nm_before[self.mps.keep_col] = [
            True, True, True, True]

        # particles for exclude() and group_n()
        mps = MultiParticleSets()
        data = {
            mps.tomo_id_col: [
                'alpha', 'alpha', 'alpha', 'alpha', 'alpha', 
                'bravo', 'bravo', 'charlie', 'charlie', 'charlie'],
            mps.particle_id_col: list(range(100, 110)),
            mps.class_name_col: [
                'U', 'U', 'U', 'V', 'V', 'U', 'U', 'U', 'U', 'V'],
            mps.coord_reg_frame_cols[0]: [5, 5, 5, 5, 22,
                                          22, 23, 33, 33, 33],
            mps.coord_reg_frame_cols[1]: [5, 6, 9, 5, 22,
                                          22, 23, 35, 36, 36]}
        self.part_index = [42, 43, 44, 45, 46, 21, 22, 13, 14, 15]
        self.part_keep = [
            False, True, True, False, True, False, True, True, True, True]
        self.particles = pd.DataFrame(data, index=self.part_index)
        self.particles_desired = [
            True, False, True, True, True, True, False, True, False, True]
        self.particles_keep_desired = [
            False, True, True, False, True, False, True, True, False, True]
        self.particles_together_desired = [
            True, False, True, False, True, False, False, True, False, False]
        
    def test_copy(self):
        """Tests copy()
        """

        # make a copy and test
        mps = MultiParticleSets()
        mps.tomos = self.tomos
        mps.particles = self.particle_final
        mps.region_col = 'regionnnn'
        mps_cp = mps.copy()
        np_test.assert_equal(mps_cp.region_col, 'regionnnn')
        assert_frame_equal(mps_cp.tomos, mps.tomos)
        assert_frame_equal(mps_cp.particles, mps.particles)

        # check modifications not passed
        mps_cp.tomos.loc[1, mps_cp.region_bin_col] = 11
        desired = self.region_bins.copy()
        desired[1] = 11
        np_test.assert_array_equal(
            mps_cp.tomos[mps_cp.region_bin_col].to_numpy(), desired)
        np_test.assert_array_equal(
            mps.tomos[mps.region_bin_col].to_numpy(), self.region_bins)
        
    def test_read_star_tomo(self):
        """
        Tests read_star(mode='tomo')
        """

        # tomo star, all tomos
        loc_pixel_size_nm = dict(list(zip(self.tomo_ids, self.pixel_size_nm)))
        actual = self.mps.read_star(
            path=os.path.join(self.dir, 'particles/in_stars/in_regions.star'),
            mode='tomo', pixel_size_nm=loc_pixel_size_nm,
            coord_bin=self.coord_bin_dict)
        assert_frame_equal(actual, self.tomo_init_rel)

        # tomo star, some tomos
        loc_pixel_size_nm = {'bravo': 1, 'charlie': 0.5}
        actual = self.mps.read_star(
            path=os.path.join(self.dir, 'particles/in_stars/in_regions.star'),
            mode='tomo', pixel_size_nm=loc_pixel_size_nm,
            tomo_ids=['bravo', 'charlie'])
        desired_cond = self.tomo_init[self.mps.tomo_id_col].apply(
            lambda x: x in ['bravo', 'charlie'])
        desired = self.tomo_init_rel[desired_cond]
        assert_frame_equal(actual, desired)
        
    def test_read_star_particle(self):
        """
        Tests read_star(mode='particle')
        """

        # particle star, origin, all tomos
        actual = self.mps.read_star(
            path=os.path.join(
                self.dir, 'particles/in_stars/in_particles_origins.star'),
            mode='particle', class_name='Class A', keep_star_coords=False)
        np_test.assert_array_equal(
            self.mps.particle_cols,
            ['tomo_id', 'particle_id', 'x_orig', 'y_orig', 'z_orig',
             'class_name', 'class_number'])
        assert_frame_equal(
            actual[self.mps.particle_cols],
            self.particle_origin[self.mps.particle_cols])

        # particle star, origin, all tomos
        actual = self.mps.read_star(
            path=os.path.join(
                self.dir, 'particles/in_stars/in_particles_origins.star'),
            mode='particle', class_name='Class A', keep_star_coords=True)
        np_test.assert_array_equal(
            self.mps.particle_cols,
            ['tomo_id', 'particle_id',
             'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ',
             'rlnAngleTilt', 'rlnAngleTiltPrior', 'rlnAnglePsi',
             'rlnAnglePsiPrior', 'rlnAngleRot',
             'rlnOriginX', 'rlnOriginY', 'rlnOriginZ',
             'x_orig', 'y_orig', 'z_orig',
             'class_name', 'class_number'])
        assert_frame_equal(
            actual[self.mps.particle_cols],
            self.particle_origin[self.mps.particle_cols],
            check_dtype=False)

        # particle star, origin all tomos, class number
        actual = self.mps.read_star(
            path=os.path.join(
                self.dir, 'particles/in_stars/in_particles_origins.star'),
            mode='particle', class_name='Class A', class_number=5)
        np_test.assert_array_equal(
            actual[self.mps.class_number_col], actual.shape[0] * [5])

        # particle star, origin, selected tomos
        actual = self.mps.read_star(
            path=os.path.join(
                self.dir, 'particles/in_stars/in_particles_origins.star'),
            tomo_ids=['alpha', 'bravo'],
            mode='particle', class_name='Class A', keep_star_coords=False)
        desired_cond = self.particle_origin[self.mps.tomo_id_col].apply(
            lambda x: x in ['alpha', 'bravo'])
        desired = self.particle_origin[desired_cond]
        assert_frame_equal(
            actual[self.mps.particle_cols],
            desired[self.mps.particle_cols], check_dtype=False) 

        # particle star, origin, all tomos
        actual = self.mps.read_star(
            path=os.path.join(
                self.dir, 'particles/in_stars/in_particles_angst.star'),
            mode='particle', class_name='Class A', tomos=self.tomo_off,
            keep_star_coords=False)
        np_test.assert_array_equal(
            self.mps.particle_cols,
            ['tomo_id', 'particle_id', 'pixel_nm', 'x_orig', 'y_orig',
             'z_orig', 'class_name', 'class_number'])
        assert_frame_equal(
            actual[self.mps.particle_cols],
            self.particle_angst[self.mps.particle_cols]) 

    def test_set_region_offset(self):
        """Tests set_region_offset()
        """

        # no offset
        desired = np.zeros(
            (self.tomo_star.shape[0], len(self.mps.region_offset_cols)))
        actual = self.mps.set_region_offset(tomos=self.tomo_star, offset=0)
        np_test.assert_almost_equal(
            actual[self.mps.region_offset_cols].values, desired)

        actual = self.mps.set_region_offset(
            tomos=self.tomo_star, offset=0, update=True)
        assert_frame_equal(actual, self.tomo_off_0)

        # csv file
        desired = self.tomo_star[
            ['psSegOffX', 'psSegOffY', 'psSegOffZ']].values 
        actual = self.mps.set_region_offset(
            offset=os.path.join(self.dir, 'particles/in_stars/offsets.csv'))
        np_test.assert_almost_equal(
            actual[self.mps.region_offset_cols].values, desired)

        # csv file, update
        actual = self.mps.set_region_offset(
            tomos=self.tomo_star,
            offset=os.path.join(self.dir, 'particles/in_stars/offsets.csv'),
            update=True)
        assert_frame_equal(actual, self.tomo_off)

        # csv file, update, tomos contain offsets
        tomos_loc = self.tomo_star.copy()
        tomos_loc[self.mps.region_offset_cols[0]] = -999
        tomos_loc[self.mps.region_offset_cols[1]] = -999
        tomos_loc[self.mps.region_offset_cols[2]] = -999
        actual = self.mps.set_region_offset(
            tomos=tomos_loc,
            offset=os.path.join(self.dir, 'particles/in_stars/offsets.csv'),
            update=True)
        assert_frame_equal(actual, self.tomo_off)

        # tomo file
        desired = self.tomo_off[
            ['psSegOffX', 'psSegOffY', 'psSegOffZ']].values 
        actual = self.mps.set_region_offset(tomos=self.tomo_star)
        np_test.assert_almost_equal(
            actual[self.mps.region_offset_cols].values, desired)
        
        actual = self.mps.set_region_offset(tomos=self.tomo_star, update=True)
        assert_frame_equal(actual, self.tomo_off)

    def test_set_region_id(self):
        """Tests set_region_id() and implicitly set_column()
        """

        # single id
        actual = self.mps.set_region_id(
            tomos=self.tomo_star, region_id=2, update=True)
        assert_frame_equal(actual[self.tomo_star.columns], self.tomo_star)
        np_test.assert_array_almost_equal(
            actual[self.mps.region_id_col].values, [2, 2, 2])

        # none
        actual = self.mps.set_region_id(
            tomos=self.tomo_star, region_id=None, update=True)
        assert_frame_equal(actual, self.tomo_star)

        # in tomos
        desired = pd.DataFrame(
            {self.mps.tomo_id_col: self.tomo_star[self.mps.tomo_id_col].values,
             self.mps.region_id_col: [1, 2, 3]})
        actual = self.mps.set_region_id(tomos=self.tomo_off_regid, update=False)
        assert_frame_equal(actual, desired)
        actual = self.mps.set_region_id(tomos=self.tomo_off_regid, update=True)
        assert_frame_equal(actual, self.tomo_off_regid)

    def test_set_region_bin(self):
        """Tests set_region_bin() and implicitly set_column()
        """

        # dict
        region_bin_dict = dict(
            [(tomo_id, bin) for tomo_id, bin
             in zip(self.tomo_ids, self.region_bins)])
        desired = pd.DataFrame({
            self.mps.tomo_id_col: self.tomo_ids,
            self.mps.region_bin_col: self.region_bins})
        actual = self.mps.set_region_bin(
            tomos=self.tomo_star, region_bin=region_bin_dict, update=False)
        assert_frame_equal(actual, desired)
        
        actual = self.mps.set_region_bin(
            tomos=self.tomo_star, region_bin=region_bin_dict, update=True)
        assert_frame_equal(actual, self.tomo_bin)

    def test_set_column(self):
        """Tests set_column()
        """

        mps = MultiParticleSets()

        # single number
        actual = mps.set_column(
            column='foo', tomos=self.tomo_star, value=2, update=False)
        desired = pd.DataFrame(
            {mps.tomo_id_col: ['alpha', 'bravo', 'charlie'], 'foo': [2, 2, 2]})
        assert_frame_equal(actual, desired)
        
        # single number update
        tomos = self.tomo_star.copy()
        desired = self.tomo_star.copy()
        actual = mps.set_column(column='foo', tomos=tomos, value=2, update=True)
        desired['foo'] = [2, 2, 2]
        assert_frame_equal(actual, desired)
        assert_frame_equal(tomos, self.tomo_star)
        
        # dict
        value = {'charlie': 'c', 'bravo': 'b', 'alpha': 'a'}
        actual = mps.set_column(
            column='foo', tomos=self.tomo_star, value=value, update=False)
        desired = pd.DataFrame(
            {mps.tomo_id_col: ['charlie', 'bravo', 'alpha'],
             'foo': ['c', 'b', 'a']})
        assert_frame_equal(actual, desired)
        
        # dict update
        tomos = self.tomo_star.copy()
        desired = self.tomo_star.copy()
        value = {'charlie': 'c', 'bravo': 'b', 'alpha': 'a'}
        actual = mps.set_column(
            column='foo', tomos=tomos, value=value, update=True)
        desired['foo'] = ['a', 'b', 'c']
        assert_frame_equal(actual, desired)
        assert_frame_equal(tomos, self.tomo_star)

        # dataframe
        value = pd.DataFrame(
            {mps.tomo_id_col: ['charlie', 'bravo', 'alpha'],
             'foo': ['c', 'b', 'a']})
        actual = mps.set_column(
            column='foo', tomos=self.tomo_star, value=value, update=False)
        desired = pd.DataFrame(
            {mps.tomo_id_col: ['charlie', 'bravo', 'alpha'],
             'foo': ['c', 'b', 'a']})
        assert_frame_equal(actual, desired)
        
        # dataframe update
        tomos = self.tomo_star.copy()
        desired = self.tomo_star.copy()
        value = pd.DataFrame(
            {mps.tomo_id_col: ['charlie', 'bravo', 'alpha'],
             'foo': ['c', 'b', 'a']})
        actual = mps.set_column(
            column='foo', tomos=tomos, value=value, update=True)
        desired['foo'] = ['a', 'b', 'c']
        assert_frame_equal(actual, desired)
        assert_frame_equal(tomos, self.tomo_star)
        
    def test_make_bare_tomos(self):
        """Test make_bare_tomos()
        """

        mps = MultiParticleSets()

        # particles
        actual = mps.make_bare_tomos(particles=self.particle_final)
        desired = pd.DataFrame({mps.tomo_id_col: ['alpha', 'charlie']})
        assert_frame_equal(actual, desired)

        # tomo_ids
        tomo_ids = ['alpha', 'charlie', 'bravo']
        actual = mps.make_bare_tomos(tomo_ids=tomo_ids)
        desired = pd.DataFrame({mps.tomo_id_col: tomo_ids})
        assert_frame_equal(actual, desired)

        # particlesand pixel
        pixel = {'charlie': 4., 'alpha': 2}
        actual = mps.make_bare_tomos(
            particles=self.particle_final, pixel_size_nm=pixel)
        desired = pd.DataFrame(
            {mps.tomo_id_col: ['alpha', 'charlie'],
             mps.pixel_nm_col: [2, 4.]})
        assert_frame_equal(actual, desired)

        
    def test_get_original_coords(self):
        """Test get_original_coords()
        """

        # origin
        actual = self.mps.get_original_coords(table=self.particle_star_origin)
        assert_frame_equal(actual, self.particle_origin)

        # angst
        actual = self.mps.get_original_coords(table=self.particle_star_angst)
        assert_frame_equal(actual, self.particle_angst)

    def test_convert_one(self):
        """Tests convert_one()
        """

        #
        tomo_id = 'alpha'
        tom = self.tomos[self.tomos[self.mps.tomo_id_col] == tomo_id]
        part = self.particle_origin[
            self.particle_origin[self.mps.tomo_id_col] == tomo_id].copy()
        region_shape = (180, 100, 80)
        desired_keep = [True, False, False]
        desired = self.particle_final[
            self.particle_final[self.mps.tomo_id_col] == tomo_id].copy()
        actual = self.mps.convert_one(
            particles=part, class_name_col=self.mps.class_name_col,
            bin=(tom[self.mps.coord_bin_col].values
                 / tom[self.mps.region_bin_col].values),
            offsets=tom[self.mps.region_offset_cols].values,
            region_path=os.path.join(
                self.dir, 'particles/in_stars',
                tom[self.mps.region_col].values[0]),
            region_id=tom[self.mps.region_id_col].values[0],
            pixel_nm=tom[self.mps.pixel_nm_col].values[0],
            region_shape=region_shape)
        assert_frame_equal(
            actual[self.mps.orig_coord_reg_frame_cols],
            desired[self.mps.orig_coord_reg_frame_cols])
        assert_frame_equal(
            actual[self.mps.coord_reg_frame_cols],
            desired[self.mps.coord_reg_frame_cols])
        assert_frame_equal(
            actual[self.mps.coord_init_frame_cols],
            desired[self.mps.coord_init_frame_cols])
        np_test.assert_array_equal(
            actual[self.mps.keep_col].to_numpy(), desired_keep)
        desired[self.mps.keep_col] = desired_keep
        assert_frame_equal(actual, desired)
                           
        tomo_id = 'bravo'
        tom = self.tomos[self.tomos[self.mps.tomo_id_col] == tomo_id]
        part = self.particle_origin[
            self.particle_origin[self.mps.tomo_id_col] == tomo_id].copy()
        desired = self.particle_final[
            self.particle_final[self.mps.tomo_id_col] == tomo_id]
        actual = self.mps.convert_one(
            particles=part,
            bin=(tom[self.mps.coord_bin_col].values
                 / tom[self.mps.region_bin_col].values),
            offsets=tom[self.mps.region_offset_cols].values,
            region_path=os.path.join(
                self.dir, 'particles/in_stars',
                tom[self.mps.region_col].values[0]),
            region_id=tom[self.mps.region_id_col].values[0],
            pixel_nm=tom[self.mps.pixel_nm_col].values[0])
        assert_frame_equal(
            actual[self.mps.orig_coord_reg_frame_cols],
            desired[self.mps.orig_coord_reg_frame_cols])
        assert_frame_equal(actual, desired, check_dtype=False)
       
        tomo_id = 'charlie'
        tom = self.tomos[self.tomos[self.mps.tomo_id_col] == tomo_id]
        part = self.particle_origin[
            self.particle_origin[self.mps.tomo_id_col] == tomo_id].copy()
        desired = self.particle_final[
            self.particle_final[self.mps.tomo_id_col] == tomo_id]
        actual = self.mps.convert_one(
            particles=part,
            bin=(tom[self.mps.coord_bin_col].values
                 / tom[self.mps.region_bin_col].values),
            offsets=tom[self.mps.region_offset_cols].values,
            region_path=os.path.join(
                self.dir, 'particles/in_stars',
                tom[self.mps.region_col].values[0]),
            region_id=tom[self.mps.region_id_col].values[0],
            pixel_nm=tom[self.mps.pixel_nm_col].values[0])
        assert_frame_equal(
            actual[self.mps.orig_coord_reg_frame_cols],
            desired[self.mps.orig_coord_reg_frame_cols])
        assert_frame_equal(actual, desired)

    def test_convert(self):
        """Tests convert()
        """

        # no exclusion
        local_tomos = self.tomos.copy()
        local_tomos[self.mps.region_col] = \
            local_tomos[self.mps.region_col].apply(
                lambda x: os.path.join(self.dir, 'particles/in_stars', x))
        actual = self.mps.convert(
            tomos=local_tomos, particles=self.particle_origin)
        assert_frame_equal(actual, self.particle_final)        

        # exclusion 20 nm
        actual = self.mps.convert(
            tomos=local_tomos, particles=self.particle_origin, exclusion=20,
            exclusion_mode='before_projection')
        assert_frame_equal(actual, self.particle_final)
        actual = self.mps.convert(
            tomos=local_tomos, particles=self.particle_origin, exclusion=20,
            exclusion_mode='after_projection')
        assert_frame_equal(actual, self.particle_final)

        # exclusion 20 nm with shape
        local_tomos_shape = local_tomos.copy()
        local_tomos_shape[self.mps.region_shape_cols] = np.array(
            [[180, 110, 80],
             [0, 0, 0],
             [50, 50, 50]])
        desired_shape = np.array([True, True, False, False])
        actual = self.mps.convert(
            tomos=local_tomos_shape, particles=self.particle_origin,
            exclusion=20, exclusion_mode='before_projection',
            remove_outside_region=True)
        desired = self.particle_final.copy()
        desired[self.mps.keep_col] = desired[self.mps.keep_col] & desired_shape
        assert_frame_equal(actual, desired)
        actual = self.mps.convert(
            tomos=local_tomos_shape, particles=self.particle_origin,
            exclusion=20, exclusion_mode='after_projection',
            remove_outside_region=True)
        desired = self.particle_final.copy()
        desired[self.mps.keep_col] = desired[self.mps.keep_col] & desired_shape
        assert_frame_equal(actual, desired)

        # exclusion 20 nm with shape, inside_coord_cols
        local_tomos_shape = local_tomos.copy()
        local_tomos_shape[self.mps.region_shape_cols] = np.array(
            [[180, 110, 80],
             [0, 0, 0],
             [50, 50, 50]])
        desired_shape = np.array([True, True, False, False])
        actual = self.mps.convert(
            tomos=local_tomos_shape, particles=self.particle_origin,
            exclusion=20, exclusion_mode='before_projection',
            remove_outside_region=True,
            inside_coord_cols=self.mps.orig_coord_reg_frame_cols)
        desired = self.particle_final.copy()
        desired[self.mps.keep_col] = desired[self.mps.keep_col] & desired_shape
        assert_frame_equal(actual, desired)
        actual = self.mps.convert(
            tomos=local_tomos_shape, particles=self.particle_origin,
            exclusion=20, exclusion_mode='after_projection',
            remove_outside_region=True,
            inside_coord_cols=self.mps.orig_coord_reg_frame_cols)
        desired = self.particle_final.copy()
        desired[self.mps.keep_col] = desired[self.mps.keep_col] & desired_shape
        assert_frame_equal(actual, desired)

        # exclusion None with shape, inside_coord_cols=projected
        local_tomos_shape = self.tomos.copy()
        local_tomos_shape[self.mps.region_shape_cols] = np.array(
            [[180, 100, 80],
             [0, 0, 0],
             [50, 50, 50]])
        desired_shape = np.array([True, False, False, False])
        actual = self.mps.convert(
            tomos=local_tomos_shape, particles=self.particle_origin,
            exclusion=20, exclusion_mode='before_projection',
            remove_outside_region=True,
            inside_coord_cols=self.mps.coord_reg_frame_cols)
        desired = self.particle_final.copy()
        desired[self.mps.keep_col] = desired[self.mps.keep_col] & desired_shape
        assert_frame_equal(actual, desired)
        actual = self.mps.convert(
            tomos=local_tomos_shape, particles=self.particle_origin,
            exclusion=20, exclusion_mode='after_projection',
            remove_outside_region=True,
            inside_coord_cols=self.mps.coord_reg_frame_cols)
        desired = self.particle_final.copy()
        desired[self.mps.keep_col] = desired[self.mps.keep_col] & desired_shape
        assert_frame_equal(actual, desired)

        # exclusion 20 nm with shape, inside_coord_cols=projected
        local_tomos_shape = self.tomos.copy()
        local_tomos_shape[self.mps.region_shape_cols] = np.array(
            [[180, 100, 80],
             [0, 0, 0],
             [50, 50, 50]])
        desired_shape = np.array([True, False, False, False])
        actual = self.mps.convert(
            tomos=local_tomos_shape, particles=self.particle_origin,
            exclusion=20, exclusion_mode='before_projection',
            remove_outside_region=True,
            inside_coord_cols=self.mps.coord_reg_frame_cols)
        desired = self.particle_final.copy()
        desired[self.mps.keep_col] = desired[self.mps.keep_col] & desired_shape
        assert_frame_equal(actual, desired)
        actual = self.mps.convert(
            tomos=local_tomos_shape, particles=self.particle_origin,
            exclusion=20, exclusion_mode='after_projection',
            remove_outside_region=True,
            inside_coord_cols=self.mps.coord_reg_frame_cols)
        desired = self.particle_final.copy()
        desired[self.mps.keep_col] = desired[self.mps.keep_col] & desired_shape
        assert_frame_equal(actual, desired)

        # exclusion 30 nm
        actual = self.mps.convert(
            tomos=local_tomos, particles=self.particle_origin, exclusion=30,
            exclusion_mode='before_projection')
        assert_frame_equal(actual, self.particle_final_excl_30nm_before)
        actual = self.mps.convert(
            tomos=local_tomos, particles=self.particle_origin, exclusion=30,
            exclusion_mode='after_projection')
        assert_frame_equal(actual, self.particle_final_excl_30nm)
       
        # exclusion 30 nm with shape
        local_tomos_shape = local_tomos.copy()
        local_tomos_shape[self.mps.region_shape_cols] = np.array(
            [[180, 100, 80],
             [0, 0, 0],
             [50, 60, 50]])
        desired_shape = np.array([True, False, False, False])
        actual = self.mps.convert(
            tomos=local_tomos_shape, particles=self.particle_origin,
            exclusion=30, exclusion_mode='before_projection',
            remove_outside_region=True)
        desired = self.particle_final_excl_30nm_before.copy()
        desired[self.mps.keep_col] = desired[self.mps.keep_col] & desired_shape
        assert_frame_equal(actual, desired)
        actual = self.mps.convert(
            tomos=local_tomos_shape, particles=self.particle_origin,
            exclusion=30, exclusion_mode='after_projection',
            remove_outside_region=True)
        desired = self.particle_final_excl_30nm.copy()
        desired[self.mps.keep_col] = desired[self.mps.keep_col] & desired_shape
        assert_frame_equal(actual, desired)
       
        # exclusion 50 nm
        actual = self.mps.convert(
            tomos=local_tomos, particles=self.particle_origin, exclusion=50,
            exclusion_mode='before_projection')
        assert_frame_equal(actual, self.particle_final_excl_30nm)

        # all args, exclusion 30 nm
        region_id = dict([
            (key, value) for key, value in zip(self.tomo_ids, self.region_id)])
        region_bins = dict([
            (key, value) for key, value in zip(
                self.tomo_ids, self.region_bins)])
        local_tomo_star = self.tomo_star.copy()
        local_tomo_star[self.mps.region_col] = local_tomo_star[
            self.mps.region_col].apply(
            lambda x: os.path.join(self.dir, 'particles/in_stars', x))
        actual = self.mps.convert(
            tomos=local_tomo_star, particles=self.particle_origin,
            region_bin=region_bins,
            region_offset=self.tomo_off, region_id=region_id,
            exclusion=30, exclusion_mode='before_projection')
        assert_frame_equal(actual, self.particle_final_excl_30nm_before)
        actual = self.mps.convert(
            tomos=local_tomo_star, particles=self.particle_origin,
            region_bin=region_bins,
            region_offset=self.tomo_off, region_id=region_id,
            exclusion=30, exclusion_mode='after_projection')
        assert_frame_equal(actual, self.particle_final_excl_30nm)

    def test_convert_frame(self):
        """Tests convert_frame().
        """

        mps = MultiParticleSets()
        mps.tomos = self.tomos.copy()
        shift_final_col = 'shift_final'
        shift_final_cols = [
            'region_offset_x', 'region_offset_y', 'region_offset_z']
        mps.particles = self.particle_origin.copy()
        final_coord_cols =  ['xx_orig', 'yy_orig', 'zz_orig']
        desired = np.array([
            [136, 78, 42],
            [162, 104, 56],
            [174, 116, 74],
            [-39, -36, -30]])
        mps.convert_frame(
            init_coord_cols=['x_orig', 'y_orig', 'z_orig'],
            final_coord_cols=final_coord_cols,
            shift_final_cols=shift_final_cols,
            init_bin_col='coord_bin', final_bin_col='region_bin',
            overwrite=False)
        np_test.assert_array_equal(
            mps.particles[final_coord_cols].to_numpy(), desired)
        np_test.assert_equal(
            mps.particles[final_coord_cols[0]].dtype, np.dtype(int))

        # no int conversion, overwrite, continues from above
        desired = np.array([
            [136, 78, 42],
            [162, 104, 56],
            [174, 116, 74],
            [-39, -36, -30.25]])
        mps.convert_frame(
            init_coord_cols=['x_orig', 'y_orig', 'z_orig'],
            final_coord_cols=final_coord_cols,
            shift_final_cols=shift_final_cols,
            init_bin_col='coord_bin', final_bin_col='region_bin',
            to_int=False, overwrite=True)
        np_test.assert_array_equal(
            mps.particles[final_coord_cols].to_numpy(), desired)

        # no int conversion
        mps = MultiParticleSets()
        mps.tomos = self.tomos.copy()
        shift_final_col = 'shift_final'
        shift_final_cols = [
            'region_offset_x', 'region_offset_y', 'region_offset_z']
        mps.particles = self.particle_origin.copy()
        final_coord_cols =  ['xx_orig', 'yy_orig', 'zz_orig']
        desired = np.array([
            [136, 78, 42],
            [162, 104, 56],
            [174, 116, 74],
            [-39, -36, -30.25]])
        mps.convert_frame(
            init_coord_cols=['x_orig', 'y_orig', 'z_orig'],
            final_coord_cols=final_coord_cols,
            shift_final_cols=shift_final_cols,
            init_bin_col='coord_bin', final_bin_col='region_bin',
            overwrite=False, to_int=False)
        np_test.assert_array_almost_equal(
            mps.particles[final_coord_cols].to_numpy(), desired)
        np_test.assert_equal(
            mps.particles[final_coord_cols[0]].dtype, np.dtype(float))
            
    def test_from_particle_sets(self):
        """Tests from_particle_sets()
        """

        # setup
        test_psets = TestParticleSets()
        test_psets.setUp()
        mps = MultiParticleSets()
        desired_tomos = pd.DataFrame({
            mps.tomo_id_col: ['alpha', 'bravo'],
            mps.region_col: ['alpha_path', 'bravo_path'],
            mps.pixel_nm_col: [0.6, 1.2]})
        desired_particles = pd.DataFrame({
            mps.tomo_id_col: 3*['alpha'] + 6*['bravo'],
            mps.class_name_col: 7*['setU'] + 2*['setV'],
            mps.class_number_col: -1,
            mps.coord_reg_frame_cols[0]: [1, 2, 3, 0, 2, 4, 6, 10, 12],
            mps.coord_reg_frame_cols[1]: [12, 13, 14, 1, 3, 5, 7, 11, 13],
            mps.keep_col: True})
        desired_particles = desired_particles.sort_values(
            by=[mps.tomo_id_col, mps.class_name_col], ignore_index=True)

        # from ParticleSets instance
        test_psets.setUp()
        test_psets.ps.set_region_path(tomo='alpha', value='alpha_path')
        test_psets.ps.set_region_path(tomo='bravo', value='bravo_path')
        test_psets.ps.set_pixel_nm(tomo='alpha', value=0.6)
        test_psets.ps.set_pixel_nm(tomo='bravo', value=1.2)
        mps.from_particle_sets(particle_sets=test_psets.ps)
        assert_frame_equal(mps.tomos, desired_tomos)
        actual = mps.particles.sort_values(
            by=[mps.tomo_id_col, mps.class_name_col], ignore_index=True)
        assert_frame_equal(actual, desired_particles)

        # from ParticleSets instance, add pixel to particles
        mps.from_particle_sets(
             particle_sets=test_psets.ps, pixel_to_particles=True)
        assert_frame_equal(mps.tomos, desired_tomos)
        actual = mps.particles.sort_values(
            by=[mps.tomo_id_col, mps.class_name_col], ignore_index=True)
        desired_particles_pixel = desired_particles.copy()
        desired_particles_pixel[mps.pixel_nm_col] = 3*[0.6] + 6*[1.2]
        assert_frame_equal(actual, desired_particles_pixel[actual.columns])

        # from ParticleSets instance with region_id
        test_psets.setUp()
        test_psets.ps.set_region_path(tomo='alpha', value='alpha_path')
        test_psets.ps.set_region_path(tomo='bravo', value='bravo_path')
        test_psets.ps.set_pixel_nm(tomo='alpha', value=0.6)
        test_psets.ps.set_pixel_nm(tomo='bravo', value=1.2)
        mps.from_particle_sets(particle_sets=test_psets.ps, region_id=3)
        np_test.assert_array_equal(
            mps.tomos[mps.region_id_col], mps.tomos.shape[0]*[3])
        assert_frame_equal(
            mps.tomos.drop(mps.region_id_col, axis=1), desired_tomos)
        actual = mps.particles.sort_values(
            by=[mps.tomo_id_col, mps.class_name_col], ignore_index=True)
        assert_frame_equal(actual, desired_particles)

        # from ParticleSets dataframe
        #test_psets.setUp()
        ps_df = test_psets.ps.data_df
        ps_df.loc[ps_df['tomo_id'] == 'alpha', 'region_path'] = 'alpha_path'
        ps_df.loc[ps_df['tomo_id'] == 'bravo', 'region_path'] = 'bravo_path'
        ps_df.loc[ps_df['tomo_id'] == 'alpha', 'pixel_nm'] = 0.6
        ps_df.loc[ps_df['tomo_id'] == 'bravo', 'pixel_nm'] = 1.2
        mps.from_particle_sets(particle_sets=ps_df)
        assert_frame_equal(mps.tomos, desired_tomos, check_dtype=False)
        actual = mps.particles.sort_values(
            by=[mps.tomo_id_col, mps.class_name_col], ignore_index=True)
        assert_frame_equal(actual, desired_particles)

    def test_to_particle_sets(self):
        """Tests to_particle_sets()
        """

        # ignore keep
        mps = MultiParticleSets()
        mps.tomos = self.tomo_star.copy()
        mps.particles = self.particle_final.copy()
        psets = mps.to_particle_sets(ignore_keep=True)
        desired = pd.DataFrame(
            {'tomo_id': ['alpha', 'alpha', 'alpha', 'charlie'],
             'set': 'Class A', #'x': [174, 174, 174, 20],
             #'y': [78, 104, 116, 50], 'z': [74, 74, 74, 40],
             'x': self.particle_final[
                 self.mps.coord_reg_frame_cols[0]].to_numpy(),
             'y': self.particle_final[
                 self.mps.coord_reg_frame_cols[1]].to_numpy(),
             'z': self.particle_final[
                 self.mps.coord_reg_frame_cols[2]].to_numpy(),
             'region_path':
             (3*[os.path.join(
                 self.dir, 'particles/regions/syn_alpha_bin2_crop_seg.mrc')] 
              + [os.path.join(
                  self.dir, 'particles/regions/syn_charlie_bin2_crop_seg.mrc')]),
             'pixel_nm': [2.0, 2.0, 2.0, 0.5]})
        assert_frame_equal(psets.data_df, desired)

        # ignore keep, check indices
        mps.particles = self.particle_final.copy()
        mps.particles = mps.particles.sort_index(ascending=False)
        psets = mps.to_particle_sets(ignore_keep=True)
        assert_frame_equal(psets.data_df.sort_index(), desired)
        
        # ignore keep, check indices, different index
        mps.particles = self.particle_final.copy()
        mps.particles.index = [55, 22, 44, 33]
        psets = mps.to_particle_sets(ignore_keep=True)
        desired_local = desired.copy()
        desired_local.index = [55, 22, 44, 33]
        assert_frame_equal(
            psets.data_df.sort_index(), desired_local.sort_index())
        np_test.assert_array_equal(
            psets.get_index(tomo='alpha', set_name='Class A'), [55, 22, 44])
        np_test.assert_array_equal(
            psets.get_index(tomo='charlie', set_name='Class A'), [33])
       
        # do not ignore keep
        mps = MultiParticleSets()
        mps.tomos = self.tomo_star.copy()
        mps.particles = self.particle_final.copy()
        mps.particles['keep'] = [True, False, False, True]
        psets = mps.to_particle_sets(ignore_keep=False)
        desired_loc = desired.copy()
        desired_loc = desired.loc[[0, 3], :].copy() #.reset_index(drop=True)
        assert_frame_equal(psets.data_df, desired_loc)

        # coord_cols
        #tps = testParticleSets()
        #tps.setUp()
        mps = MultiParticleSets()
        mps.tomos = self.tomo_star.copy()
        mps.particles = self.particle_final.copy()
        #desired['x'] = [136, 162, 174, 276]
        #desired['y'] = [78, 104, 116, 174]
        #desired['z'] = [42, 56, 74, 112]
        desired['x'] = self.particle_regframe[
            self.mps.orig_coord_reg_frame_cols[0]]
        desired['y'] = self.particle_regframe[
            self.mps.orig_coord_reg_frame_cols[1]]
        desired['z'] = self.particle_regframe[
            self.mps.orig_coord_reg_frame_cols[2]]
        psets = mps.to_particle_sets(
            coord_cols=mps.orig_coord_reg_frame_cols, ignore_keep=True)
        assert_frame_equal(psets.data_df, desired)
               
    def test_find_min_distances(self):
        """Tests find_min_distances()
        """

        # df_1 > df_2
        mps = MultiParticleSets()
        df_short = self.particle_final.loc[[1, 3]].reset_index()
        desired = self.particle_final[
            [mps.tomo_id_col] + mps.coord_init_frame_cols].copy()
        desired['index_2'] = [0, 0, 0, 1]
        desired['dist_pix'] = [13., 0, 6, 0] 
        actual = mps.find_min_distances(
            df_1=self.particle_final, df_2=df_short, group_col=mps.tomo_id_col,
            coord_cols_1=mps.coord_init_frame_cols, distance_col='dist_pix')
        assert_frame_equal(actual, desired)
            
        # df_2 > df_1
        mps = MultiParticleSets()
        df_short = self.particle_final.loc[[1, 3]].reset_index()
        df_short[mps.coord_reg_frame_cols[2]] = [75, 42]
        desired = (self.particle_final.loc[
            [1, 3], [mps.tomo_id_col] + mps.coord_reg_frame_cols]
                   .copy().reset_index(drop=True))
        desired[mps.coord_reg_frame_cols[2]] = [75, 42]
        desired['index_long'] = [1, 3]
        desired['distance'] = [1., 2] 
        actual = mps.find_min_distances(
            df_1=df_short, df_2=self.particle_final, group_col=mps.tomo_id_col,
            coord_cols_1=mps.coord_reg_frame_cols, ind_col_2='index_long')
        assert_frame_equal(actual, desired)

        # no groups, df_1 > df_2
        mps = MultiParticleSets()
        df_short = self.particle_final.loc[[1, 3]].reset_index()
        desired = self.particle_final[mps.coord_init_frame_cols].copy()
        desired['index_2'] = [0, 0, 0, 1]
        desired['dist_pix'] = [13., 0, 6, 0] 
        actual = mps.find_min_distances(
            df_1=self.particle_final, df_2=df_short, group_col=None,
            coord_cols_1=mps.coord_init_frame_cols, distance_col='dist_pix')
        assert_frame_equal(actual, desired)
        
        # no groups, df_2 > df_1
        mps = MultiParticleSets()
        df_short = self.particle_final.loc[[1, 3]].reset_index()
        df_short[mps.coord_reg_frame_cols[2]] = [75, 42]
        desired = (self.particle_final.loc[[1, 3], mps.coord_reg_frame_cols]
                   .copy().reset_index(drop=True))
        desired[mps.coord_reg_frame_cols[2]] = [75, 42]
        desired['index_long'] = [1, 3]
        desired['distance'] = [1., 2] 
        actual = mps.find_min_distances(
            df_1=df_short, df_2=self.particle_final,
            coord_cols_1=mps.coord_reg_frame_cols, ind_col_2='index_long')
        assert_frame_equal(actual, desired)

        # df_2 no elements in one group 
        mps = MultiParticleSets()
        df_short = self.particle_final.loc[[1, 2]].reset_index()
        desired = self.particle_final[
            [mps.tomo_id_col] + mps.coord_init_frame_cols].copy()
        desired['index_2'] = [0, 0, 1, -1]
        desired['dist_pix'] = [13., 0, 0, -1] 
        actual = mps.find_min_distances(
            df_1=self.particle_final, df_2=df_short, group_col=mps.tomo_id_col,
            coord_cols_1=mps.coord_init_frame_cols, distance_col='dist_pix')
        assert_frame_equal(actual, desired)

        # df_1 no elements in one group 
        mps = MultiParticleSets()
        df_short = self.particle_final.loc[[1, 2]].reset_index()
        df_short[mps.coord_reg_frame_cols[2]] = [75, 42]
        desired = (self.particle_final.loc[
            [1, 2], [mps.tomo_id_col] + mps.coord_reg_frame_cols]
                   .copy().reset_index(drop=True))
        desired[mps.coord_reg_frame_cols[2]] = [75, 42]
        desired['index_long'] = [1, 2]
        desired['distance'] = [1., 32.] 
        actual = mps.find_min_distances(
            df_1=df_short, df_2=self.particle_final, group_col=mps.tomo_id_col,
            coord_cols_1=mps.coord_reg_frame_cols, ind_col_2='index_long')
        assert_frame_equal(actual, desired)

        # df_2 is None, single group selection
        mps = MultiParticleSets()
        desired = pd.DataFrame(
            {'distance': [0, 1, 3, 0, np.sqrt(17**2 + 13**2),
                          np.sqrt(2), np.sqrt(2), 1, 0, 0],
             'index_2': [45, 42, 43, 42, 44, 22, 21, 14, 15, 14]},
            index=self.part_index) 
        actual = mps.find_min_distances(
            df_1=self.particles, df_2=None, group_col=mps.tomo_id_col,
            coord_cols_1=mps.coord_reg_frame_cols[:2],
            coord_cols_2=mps.coord_reg_frame_cols[:2])
        assert_frame_equal(actual[['distance', 'index_2']], desired)

        # df_2 is None, single group selection but specified in a list
        mps = MultiParticleSets()
        desired = pd.DataFrame(
            {'distance': [0, 1, 3, 0, np.sqrt(17**2 + 13**2),
                          np.sqrt(2), np.sqrt(2), 1, 0, 0],
             'index_2': [45, 42, 43, 42, 44, 22, 21, 14, 15, 14]},
            index=self.part_index) 
        actual = mps.find_min_distances(
            df_1=self.particles, df_2=None, group_col=[mps.tomo_id_col],
            coord_cols_1=mps.coord_reg_frame_cols[:2],
            coord_cols_2=mps.coord_reg_frame_cols[:2])
        assert_frame_equal(actual[['distance', 'index_2']], desired)

        # df_2 is None, multiple group selection
        mps = MultiParticleSets()
        desired = pd.DataFrame(
            {'distance': [1, 1, 3, np.sqrt(2*17**2), np.sqrt(2*17**2),
                          np.sqrt(2), np.sqrt(2), 1, 1, -1],
             'index_2': [43, 42, 43, 46, 45, 22, 21, 14, 13, -1]},
            index=self.part_index) 
        actual = mps.find_min_distances(
            df_1=self.particles, df_2=None,
            group_col=[mps.tomo_id_col, mps.class_name_col],
            coord_cols_1=mps.coord_reg_frame_cols[:2],
            coord_cols_2=mps.coord_reg_frame_cols[:2])
        assert_frame_equal(actual[['distance', 'index_2']], desired)
       
    def test_in_region(self):
        """Tests in_region()
        """

        mps = MultiParticleSets()
        path_prefix = os.path.join(self.dir, 'particles', 'regions')

        # default
        mps.tomos = self.tomo_off_regid.copy()
        mps.particles = self.particle_final.copy()
        actual = mps.in_region(path_prefix=path_prefix)
        np_test.assert_array_equal(
            actual[mps.in_region_col], [True, True, True, True])
        assert_frame_equal(
            actual.drop(mps.in_region_col, axis=1), self.particle_final)

        # arg coord_cols
        mps.tomos = self.tomo_off_regid.copy()
        mps.particles = self.particle_final.copy()
        actual = mps.in_region(
            coord_cols=mps.coord_reg_frame_cols, path_prefix=path_prefix)
        np_test.assert_array_equal(
            actual[mps.in_region_col], [True, True, True, True])

        # none in
        mps.tomos = self.tomo_off_regid.copy()
        mps.particles = self.particle_final.copy()
        actual = mps.in_region(
            coord_cols=mps.orig_coord_cols, path_prefix=path_prefix)
        np_test.assert_array_equal(
            actual[mps.in_region_col], [False, False, False, False])

        # one part not in
        particles = self.particle_final.copy()
        particles.loc[1, mps.coord_reg_frame_cols[2]] = 77
        actual = mps.in_region(particles=particles, path_prefix=path_prefix)
        np_test.assert_array_equal(
            actual[mps.in_region_col], [True, False, True, True])
        
        # none from charlie in
        particles = self.particle_final.copy()
        particles.loc[3, mps.coord_reg_frame_cols[2]] = 77
        actual = mps.in_region(particles=particles, path_prefix=path_prefix)
        np_test.assert_array_equal(
            actual[mps.in_region_col], [True, True, True, False])
        
        # no tomos
        mps.particles = self.particle_final.copy()
        actual = mps.in_region(tomos=['xxx'], path_prefix=path_prefix)
        np_test.assert_equal(mps.in_region_col not in actual.columns, True)
        assert_frame_equal(actual, self.particle_final)

        # one tomo ('charlie')
        mps.particles = self.particle_final.copy()
        actual = mps.in_region(tomos=['charlie'], path_prefix=path_prefix)
        np_test.assert_equal(
            np.array_equal(
                actual[mps.in_region_col].to_numpy(dtype=float),
                np.array([np.nan, np.nan, np.nan, True]),
                equal_nan=True),
            True)

        # both tomos, one after another
        mps.particles = self.particle_final.copy()
        mps.particles.loc[1, mps.coord_reg_frame_cols[2]] = 77
        mps.in_region(tomos=['alpha'], path_prefix=path_prefix)
        np_test.assert_equal(
            np.array_equal(
                mps.particles[mps.in_region_col].to_numpy(dtype=float),
                np.array([True, False, True, np.nan]),
                equal_nan=True),
            True)
        mps.in_region(tomos=['charlie'], path_prefix=path_prefix)
        np_test.assert_array_equal(
            mps.particles[mps.in_region_col], [True, False, True, True])
        assert_frame_equal(
            actual.drop(mps.in_region_col, axis=1), self.particle_final)
        
        # both tomos, one after another
        mps.particles = self.particle_final.copy()
        mps.particles.loc[1, mps.coord_reg_frame_cols[2]] = 77
        mps.particles.loc[3, mps.coord_reg_frame_cols[2]] = 77
        mps.in_region(tomos=['alpha'], path_prefix=path_prefix)
        np_test.assert_equal(
            np.array_equal(
                mps.particles[mps.in_region_col].to_numpy(dtype=float),
                np.array([True, False, True, np.nan]),
                equal_nan=True),
            True)
        mps.in_region(tomos=['charlie'], path_prefix=path_prefix)
        np_test.assert_array_equal(
            mps.particles[mps.in_region_col], [True, False, True, False])
        assert_frame_equal(
            actual.drop(mps.in_region_col, axis=1), self.particle_final)

    def test_select(self):
        """Tests select()
        """

        # update=False
        mps = MultiParticleSets()
        mps.tomos = self.tomos
        mps.particles = self.particle_final
        actual = mps.select(tomo_ids=['bravo'], update=False)
        assert_frame_equal(actual.tomos, mps.tomos.loc[[1]])
        np_test.assert_equal(actual.particles.shape[0], 0) 
        assert_frame_equal(mps.tomos, self.tomos)
        assert_frame_equal(mps.particles, self.particle_final)
        
        # update=True
        mps.select(tomo_ids=['bravo'], update=True)
        assert_frame_equal(mps.tomos, mps.tomos.loc[[1]])
        np_test.assert_equal(mps.particles.shape[0], 0)

        # class criteria
        mps = MultiParticleSets()
        mps.tomos = self.tomos
        mps.particles = self.particle_final
        mps.particles[mps.class_name_col] = ['A', 'B', 'A', 'C']
        actual = mps.select(
            class_names=['A', 'C'], class_numbers=[11], update=False)
        assert_frame_equal(actual.tomos, mps.tomos)
        assert_frame_equal(actual.particles, mps.particles.loc[[0, 3]])

        # use rules arg
        mps = MultiParticleSets()
        mps.tomos = self.tomos
        mps.particles = self.particle_final
        mps.particles[mps.class_name_col] = ['A', 'B', 'A', 'C']
        actual = mps.select(
            rules={mps.class_name_col: ['A', 'C'], mps.class_number_col: [11]},
            update=False)
        assert_frame_equal(actual.tomos, mps.tomos)
        assert_frame_equal(actual.particles, mps.particles.loc[[0, 3]])

        # mix rules and other args
        mps = MultiParticleSets()
        mps.tomos = self.tomos
        mps.particles = self.particle_final
        mps.particles[mps.class_name_col] = ['A', 'B', 'A', 'C']
        actual = mps.select(
            rules={mps.class_name_col: ['A', 'C']}, class_numbers=[11], 
            update=False)
        assert_frame_equal(actual.tomos, mps.tomos)
        assert_frame_equal(actual.particles, mps.particles.loc[[0, 3]])        

    def test_exclude_single(self):
        """Test exclude_single()
        """

        # setup
        mps = MultiParticleSets()
        data = {
            mps.tomo_id_col: [
                'alpha', 'alpha', 'alpha', 'alpha', 'alpha', 
                'bravo', 'bravo', 'charlie', 'charlie', 'charlie'],
            mps.particle_id_col: list(range(100, 110)),
            mps.class_name_col: 'U',
            mps.coord_reg_frame_cols[0]: [5, 5, 5, 10, 10, 22, 22, 33, 33, 33],
            mps.coord_reg_frame_cols[1]: [5, 6, 9, 10, 12, 22, 25, 33, 35, 36]}
        index = [42, 43, 44, 45, 46, 21, 22, 13, 14, 15]
        part = pd.DataFrame(data, index=index)
        part_alpha = part[part[mps.tomo_id_col] == 'alpha'].copy()
        part_charlie = part[part[mps.tomo_id_col] == 'charlie'].copy()
        part_none = part[part[mps.tomo_id_col] == 'aaa'].copy()

        # no previous keep
        actual = mps.exclude_single(
            particles=part_alpha, coord_cols=mps.coord_reg_frame_cols[:2],
            exclusion=2, pixel_nm=0.8)
        desired = pd.Series(
            [True, False, True, True, False], index=[42, 43, 44, 45, 46])
        np_test.assert_array_equal(actual, desired)
        actual = mps.exclude_single(
            particles=part_charlie, coord_cols=mps.coord_reg_frame_cols[:2],
            exclusion=2, pixel_nm=0.8)
        desired = pd.Series([True, False, True], index=[13, 14, 15])
        np_test.assert_array_equal(actual, desired)

        # previous keep, consider_keep=True
        part_alpha[mps.keep_col] = [True, True, False, False, True]
        actual = mps.exclude_single(
            particles=part_alpha, coord_cols=mps.coord_reg_frame_cols[:2],
            exclusion=2, pixel_nm=0.8, consider_keep=True)
        desired = pd.Series(
            [True, False, False, False, True], index=[42, 43, 44, 45, 46])
        pd.testing.assert_series_equal(actual, desired)
        np_test.assert_array_equal(actual, [True, False, False, False, True])
 
        # previous keep, but consider_keep=False
        part_alpha[mps.keep_col] = [True, True, False, False, True]
        actual = mps.exclude_single(
            particles=part_alpha, coord_cols=mps.coord_reg_frame_cols[:2],
            exclusion=2, pixel_nm=0.8, consider_keep=False)
        desired = pd.Series(
            [True, False, True, True, False], index=[42, 43, 44, 45, 46])      
        np_test.assert_array_equal(actual, desired)

        # previous keep, consider_keep=True
        part_charlie[mps.keep_col] = [False, False, False]
        actual = mps.exclude_single(
            particles=part_charlie, coord_cols=mps.coord_reg_frame_cols[:2],
            exclusion=2, pixel_nm=0.8, consider_keep=True)
        np_test.assert_array_equal(actual, [False, False, False])
 
        # no particles
        actual = mps.exclude_single(
            particles=part_none, coord_cols=mps.coord_reg_frame_cols[:2],
            exclusion=2, pixel_nm=0.8)
        np_test.assert_array_equal(actual, pd.Series([], dtype=bool))

        # all tomos and sets together (just in case there's a need for
        # such application
        actual = mps.exclude_single(
            particles=self.particles, coord_cols=mps.coord_reg_frame_cols[:2],
            exclusion=2, pixel_nm=0.8, 
            consider_keep=False)
        np_test.assert_array_equal(actual, self.particles_together_desired)
              
    def test_exclude_one(self):
        """Tests exclude_one().
        """

        # setup
        mps = MultiParticleSets()

        # consider_keep=False
        tomo_id = 'alpha'
        parts = self.particles[self.particles[self.mps.tomo_id_col] == tomo_id]
        actual = mps.exclude_one(
            particles=parts, coord_cols=mps.coord_reg_frame_cols[:2],
            exclusion=2, pixel_nm=0.8, class_name_col=mps.class_name_col,
            consider_keep=False)
        np_test.assert_array_equal(actual, self.particles_desired[:5])
       
        # consider_keep=True
        tomo_id = 'alpha'
        parts = self.particles.copy()
        parts['keep'] = self.part_keep
        parts = parts[self.particles[self.mps.tomo_id_col] == tomo_id]
        actual = mps.exclude_one(
            particles=parts, coord_cols=mps.coord_reg_frame_cols[:2],
            exclusion=2, pixel_nm=0.8, class_name_col=mps.class_name_col,
            consider_keep=True)
        np_test.assert_array_equal(actual, self.particles_keep_desired[:5])
       
    def test_exclude(self):
        """Tests exclude()
        """
        
        # setup
        mps = MultiParticleSets()

        # check that different data sets are not mixed
        # no previous keep col
        actual = mps.exclude(
            particles=self.particles, coord_cols=mps.coord_reg_frame_cols[:2],
            exclusion=2, pixel_nm=0.8, class_name_col=mps.class_name_col,
            consider_keep=False)
        np_test.assert_array_equal(actual, self.particles_desired)
        np_test.assert_array_equal(
            actual.index.to_numpy(), self.particles.index.to_numpy())

        # check that different data sets are not mixed
        # no previous keep col
        actual = mps.exclude(
            particles=self.particles, coord_cols=mps.coord_reg_frame_cols[:2],
            exclusion=2, pixel_nm=0.8, class_name_col=mps.class_name_col,
            consider_keep=False, tomos=['charlie', 'bravo', 'alpha'])
        np_test.assert_array_equal(actual, self.particles_desired)
        np_test.assert_array_equal(
            actual.index.to_numpy(), self.particles.index.to_numpy())

        # ignore previous keep col
        parts = self.particles.copy()
        parts['keep'] = self.part_keep
        actual = mps.exclude(
            particles=parts, coord_cols=mps.coord_reg_frame_cols[:2],
            exclusion=2, pixel_nm=0.8, class_name_col=mps.class_name_col,
            consider_keep=False)
        np_test.assert_array_equal(actual, self.particles_desired)

        # consider_keep
        parts = self.particles.copy()
        parts['keep'] = self.part_keep
        actual = mps.exclude(
            particles=parts, coord_cols=mps.coord_reg_frame_cols[:2],
            exclusion=2, pixel_nm=0.8, class_name_col=mps.class_name_col,
            consider_keep=True)
        np_test.assert_array_equal(actual, self.particles_keep_desired)

        # exlusion None, no previous keep col
        actual = mps.exclude(
            particles=self.particles, coord_cols=mps.coord_reg_frame_cols[:2],
            exclusion=None, pixel_nm=0.8, class_name_col=mps.class_name_col,
            consider_keep=False, tomos=['charlie', 'bravo', 'alpha'])
        desired = pd.Series(True, index=self.particles.index)
        assert_series_equal(actual, desired)

        # exlusion None, ignore previous keep col
        parts = self.particles.copy()
        parts['keep'] = self.part_keep
        actual = mps.exclude(
            particles=parts, coord_cols=mps.coord_reg_frame_cols[:2],
            exclusion=None, pixel_nm=0.8, class_name_col=mps.class_name_col,
            consider_keep=False)
        desired = pd.Series(True, index=self.particles.index)
        assert_series_equal(actual, desired)

        # exclusion None, consider_keep
        parts = self.particles.copy()
        parts['keep'] = self.part_keep
        actual = mps.exclude(
            particles=parts, coord_cols=mps.coord_reg_frame_cols[:2],
            exclusion=None, pixel_nm=0.8, class_name_col=mps.class_name_col,
            consider_keep=True)
        desired = parts['keep'].rename()
        assert_series_equal(actual, desired)
         
    def test_group_n(self):
        """Tests group_n()
        """

        # no total
        mps = MultiParticleSets()
        mps.particles = self.particles
        actual = mps.group_n(
            group_by=[mps.tomo_id_col, mps.class_name_col], total=False)
        desired = pd.DataFrame({
            mps.tomo_id_col: ['alpha', 'bravo', 'charlie'],
            'U': [3, 2, 2], 'V': [2, 0, 1]}).set_index(mps.tomo_id_col)
        np_test.assert_array_equal(actual, desired)

        # with total
        actual = mps.group_n(
            group_by=[mps.tomo_id_col, mps.class_name_col], total=True)
        desired.loc['Total'] = [7, 3]
        np_test.assert_array_equal(actual, desired)


    def test_get_region_shapes(self):
        """Tests get_region_shapes()
        """
        
        # abs paths
        mps = MultiParticleSets()
        tomo_star_orig = self.tomo_star.copy()
        actual = mps.get_region_shapes(tomos=self.tomo_star, update=False)
        assert_frame_equal(actual, self.reg_shapes)
        assert_frame_equal(self.tomo_star, tomo_star_orig)

        # relative paths
        self.tomo_star[mps.region_col] = [
            'particles/regions/syn_alpha_bin2_crop_seg.mrc',
            'particles/regions/syn_bravo_bin2_crop_seg.mrc',
            'particles/regions/syn_charlie_bin2_crop_seg.mrc']
        tomo_star_orig = self.tomo_star.copy()
        tomo_star = self.tomo_star.copy()
        actual = mps.get_region_shapes(
            tomos=tomo_star, curr_dir=self.dir, update=False)
        assert_frame_equal(actual, self.reg_shapes)
        assert_frame_equal(tomo_star, tomo_star_orig)
      
        # relative paths, update
        tomo_star = self.tomo_star.copy()
        tomo_star_orig = self.tomo_star.copy()
        tomo_star = self.tomo_star.copy()
        actual = mps.get_region_shapes(
             tomos=tomo_star, curr_dir=self.dir, update=True)
        assert_frame_equal(actual, tomo_star)
        cols = [self.mps.tomo_id_col] + self.mps.region_shape_cols
        assert_frame_equal(actual[cols], self.reg_shapes)
        cols = tomo_star_orig.columns
        assert_frame_equal(actual[cols], tomo_star_orig)
        np_test.assert_array_equal(
            actual.columns.to_list(),
            cols.to_list() + self.mps.region_shape_cols)

    def test_find_inside_one(self):
        """Tests find_inside_one()
        """

        mps = MultiParticleSets()

        # ignore tomo ids
        #tomo_id = 'alpha'
        parts = self.particle_regframe.copy()
        parts['tmp'] = [9, 8, 7, 6]
        parts.set_index('tmp', inplace=True)
        parts.index.name = None
        actual = mps.find_inside_one(
            particles=parts, coord_cols=self.mps.orig_coord_reg_frame_cols,
            shape=(170, 120, 120))
        desired = pd.Series([True, True, False, False], index=[9, 8, 7, 6])
        assert_series_equal(actual, desired)

        # one tomo
        tomo_id = 'alpha'
        parts = self.particle_regframe[
            self.particle_regframe[self.mps.tomo_id_col] == tomo_id].copy()
        parts['tmp'] = [6, 7, 8]
        parts.set_index('tmp', inplace=True)
        parts.index.name = None
        actual = mps.find_inside_one(
            particles=parts, coord_cols=self.mps.orig_coord_reg_frame_cols,
            shape=(180, 110, 110))
        desired = pd.Series([True, True, False], index=[6, 7, 8])
        assert_series_equal(actual, desired)

    def test_find_inside(self):
        """Test find_inside()
        """
        
        mps = MultiParticleSets()

        # explitit tom and particles
        parts = self.particle_final.copy()
        parts['tmp'] = [9, 8, 7, 6]
        parts.set_index('tmp', inplace=True)
        parts.index.name = None
        tomos = self.tomo_star.copy()
        tomos[self.mps.region_shape_cols] = np.array([
            [180, 100, 80],
            [0, 0, 0],
            [90, 90, 90]])            
        actual = mps.find_inside(
            tomos=tomos, particles=parts,
            coord_cols=self.mps.coord_reg_frame_cols,
            shape_cols=self.mps.region_shape_cols)
        desired = pd.Series([True, False, False, True], index=[9, 8, 7, 6])
        assert_series_equal(actual, desired)

        # arguments as attributes
        mps.tomos = tomos
        mps.particles = parts
        actual = mps.find_inside(
            coord_cols=self.mps.coord_reg_frame_cols,
            shape_cols=self.mps.region_shape_cols)
        desired = pd.Series([True, False, False, True], index=[9, 8, 7, 6])
        assert_series_equal(actual, desired)
        

        
        
        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMultiParticleSets)
    unittest.TextTestRunner(verbosity=2).run(suite)
