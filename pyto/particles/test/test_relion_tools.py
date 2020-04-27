"""

Tests relion_tools.py

# Author: Vladan Lucic (Max Planck Institute of Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from builtins import range
#from past.utils import old_div

__version__ = "$Revision$"


import os
import unittest

import numpy as np
import numpy.testing as np_test 

import pyto.particles.relion_tools as relion_tools

class TestRelionTools(np_test.TestCase):
    """
    """

    def setUp(self):
        curr_dir, base = os.path.split(__file__)
        self.test_dir =  os.path.join(curr_dir, 'test_data')
        self.test_base =  os.path.join(curr_dir, 'test_data/')


    def test_get_n_particle_class_change(self):
        """
        Tests get_n_particle_class_change()
        """

        # class_ None
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=None, mode='change',
            fraction=False, tablename='data_', out='list')
        np_test.assert_equal(res, [20])
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=None, mode='change',
            fraction=True, tablename='data_', out='dict')
        np_test.assert_equal(res, {5 : 0.4})

        # individual classes
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=[1,2,3,4,5], 
            mode='change', fraction=False, tablename='data_', out='dict')
        np_test.assert_equal(res, {5 : [3, 5, 4, 6, 2]})
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=[1,2,3,4,5], 
            mode='change', fraction=True, tablename='data_', out='dict')
        np_test.assert_equal(res, {5 : [3/10., 5/10., 4/10., 6/13., 2/7.]})
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=[1,2,3,4,5], 
            mode='change', fraction=True, tablename='data_', out='list')
        np_test.assert_equal(res, [[3/10., 5/10., 4/10., 6/13., 2/7.]])
       
       # grouped classes
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=[1,(2,3,4),5], 
            mode='change', fraction=False, tablename='data_', out='dict')
        np_test.assert_equal(res, {5 : [3, 7, 2]})
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=[1,(2,4),(3,5)], 
            mode='change', fraction=True, tablename='data_', out='dict')
        np_test.assert_equal(res, {5 : [3/10., 8/23., 5/17.]})
       
        # individual classes to_from
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=[1,2,3,4,5], 
            mode='to_from', fraction=False, tablename='data_', out='dict')
        np_test.assert_equal(
            res, {5 : np.array([[7, 1, 2, 0, 0],
                                [0, 5, 2, 3, 0],
                                [0, 0, 6, 3, 1],
                                [2, 0, 0, 7, 1],
                                [1, 4, 0, 0, 5]])})

        # grouped classes to_from
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=[1,(2,4),(3,5)], 
            mode='to_from', fraction=False, tablename='data_', out='dict')
        np_test.assert_equal(
            res, {5 : np.array([[7, 1, 2],
                                [2, 15, 3],
                                [1, 7, 12]])})
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=[1,(2,3,4),5],
            mode='to_from', fraction=True, tablename='data_', out='dict')
        np_test.assert_equal(
            res, {5 : np.array([[7/10., 3/33., 0],
                                [2/10., 26/33., 2/7.],
                                [1/10., 4/33., 5/7.]])})
        res = relion_tools.get_n_particle_class_change(
            basename=self.test_base, iters=[2,5], class_=[1,(2,3,4),5],
            mode='to_from', fraction=True, tablename='data_', out='list')
        np_test.assert_equal(
            res, [np.array([[7/10., 3/33., 0],
                            [2/10., 26/33., 2/7.],
                            [1/10., 4/33., 5/7.]])])

    def test_find_file(self):
        """
        Tests find_file()
        """

        # no continuation
        res = relion_tools.find_file(
            basename=self.test_base, suffix='_data.star', iter_=2)
        np_test.assert_equal(res, [self.test_base + '_it002_data.star'])
        res = relion_tools.find_file(
            basename=self.test_base, suffix='_data.star', iter_=3)
        np_test.assert_equal(res is None, True) 
        res = relion_tools.find_file(
            basename=self.test_base, suffix='_data.star', iter_=2, half=2)
        np_test.assert_equal(res is None, True) 
        res = relion_tools.find_file(
            basename=self.test_base, suffix='_data.star', iter_=None)
        np_test.assert_equal(res is None, True) 

        # with continuation
        res = relion_tools.find_file(
            basename=self.test_dir+'/a', suffix='_model.star', iter_=4, half=2)
        np_test.assert_equal(
            res, [self.test_dir + '/a_it004_half2_model.star'])
        res = relion_tools.find_file(
            basename=self.test_dir+'/a', suffix='_model.star', iter_=5, half=2)
        np_test.assert_equal(
            set(res), 
            set([self.test_dir + '/a_ct4_it005_half2_model.star',
                 self.test_dir + '/a_ct5_it005_half2_model.star']))

    def test_get_array_data(self):
        """
        Tests get_array_data() and implicitly array_data_generator().
        """

        labels = [
            'rlnMicrographName', 'rlnCoordinateY', 'rlnLogLikeliContribution']
        res = relion_tools.get_array_data(
            starfile=os.path.join(self.test_dir, '_it002_data.star'),
            tablename='data', labels=labels,
            types=[str, float, float])
        np_test.assert_equal(list(res.keys()), labels)
        mic_names = res['rlnMicrographName']
        coord_y = res['rlnCoordinateY']
        llcontrib = res['rlnLogLikeliContribution']
        np_test.assert_equal(len(mic_names), 50)
        np_test.assert_equal(len(coord_y), 50)
        np_test.assert_equal(len(llcontrib), 50)
        np_test.assert_equal(
            mic_names[0],
            ('../../tomo-reconstruct-01_stimul_2.21/alignment_ck01-03_'
             + 'relion/tomos/ck01-03/ck01-03_bin-0.mrc'))
        np_test.assert_equal(
            mic_names[46],
            ('../../tomo-reconstruct-01_stimul_2.21/alignment_ck01-03_'
             + 'relion/tomos/ck01-03/ck01-03_bin-0.mrc'))
        np_test.assert_equal(
            coord_y[[3, 29, 30, 37, 42]], [2056., 200., 440, 1216., 1304]) 
        np_test.assert_equal(
            llcontrib[[1, 3, 10, 27]],
            [1.871939e+07, 1.871939e+07, 1.867890e+07, 1.869831e+07])
 
    def test_write_table(self):
        """
        Tests write_table()
        """

        labels = [
            'rlnMicrographName',
            'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ', 
            'rlnImageName', 'rlnCtfImage', 'rlnGroupNumber', 
            'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi', 
            'rlnAngleTiltPrior', 'rlnAngleRotPrior', 'rlnAnglePsiPrior', 
            'rlnOriginX', 'rlnOriginY',
            'rlnClassNumber', 'rlnNormCorrection', 'rlnRandomSubset',
            'rlnOriginZ', 'rlnLogLikeliContribution',
            'rlnMaxValueProbDistribution', 'rlnNrOfSignificantSamples']
        types = [
            str, float, float, float, str, str, int,
            float, float, float, float, float, float, float, float,
            int, float, int, float, float, float, int]

        # read file
        original = relion_tools.get_array_data(
            starfile=os.path.join(self.test_dir, '_it005_data.star'),
            tablename='data', labels=labels, types=types)
        np_test.assert_equal(set(original.keys()), set(labels))

        # write file and read again
        tmp_file_name = os.path.join(self.test_dir, 'tmp.star')
        relion_tools.write_table(
            starfile = tmp_file_name, labels=labels, data=original,
            format_='auto', tablename='data_')
        second = relion_tools.get_array_data(
            starfile=tmp_file_name, tablename='data',
            labels=labels, types=types)

        # test
        np_test.assert_equal(set(second.keys()), set(labels))
        for lab in labels:
            np_test.assert_equal(second[lab], original[lab])       

        # rm test file
        try:
            os.remove(tmp_file_name)
        except OSError:
            print("Tests fine but could not remove {}".format(tmp_file_name))
      
    def test_two_way_class(self):
        """
        Tests two_way_class
        """

        res = relion_tools.two_way_class(
            basename=self.test_base, iters=[5], mode=('class', 'find'),
            pattern=(list(range(1,6)),
                     ['tomo-reconstruct-01','tomo-reconstruct-02']),
            label=('rlnClassNumber', 'rlnMicrographName'), tablename='data_',
            suffix='_data.star', type_=(int, str), iter_format='_it%03d',
            method='contingency')
        np_test.assert_equal(
            res[5].contingency, [[10, 0], [5, 5], [2, 8], [7, 6], [6, 1]]) 
        np_test.assert_almost_equal(
            res[5].fraction, 
            [[1, 0], [0.5, 0.5], [0.2, 0.8], [7/13., 6/13.], [6/7., 1/7.]]) 
        np_test.assert_almost_equal(
            res[5].total_fract[0], [30/50., 20/50.])
        np_test.assert_almost_equal(
            res[5].total_fract[1], [0.2, 0.2, 0.2, 13/50., 7/50.])

        res = relion_tools.two_way_class(
            basename=self.test_base, iters=5, mode=('class', 'find'),
            pattern=([(1,2), 3, (4,5)], [
                'tomo-reconstruct-01','tomo-reconstruct-02']),
            label=('rlnClassNumber', 'rlnMicrographName'), tablename='data_',
            suffix='_data.star', type_=(int, str), iter_format='_it%03d',
            method='contingency')
        np_test.assert_equal(
            res.contingency, [[15, 5], [2, 8], [13, 7]]) 
        np_test.assert_almost_equal(
            res.fraction, [[0.75, 0.25], [0.2, 0.8], [13/20., 7/20.]]) 
        np_test.assert_almost_equal(
            res.total_fract[0], [30/50., 20/50.])
        np_test.assert_almost_equal(
            res.total_fract[1], [0.4, 0.2, 20/50.])

        res = relion_tools.class_group_interact(
            method='contingency', basename=self.test_base, iters=5, 
            classes=list(range(1,6)), group_mode='find',
            group_pattern=['tomo-reconstruct-01','tomo-reconstruct-02'],
            group_label='rlnMicrographName', group_type=str,
            tablename='data_')
        np_test.assert_equal(
            res.contingency, [[10, 0], [5, 5], [2, 8], [7, 6], [6, 1]]) 
        np_test.assert_almost_equal(
            res.fraction, 
            [[1, 0], [0.5, 0.5], [0.2, 0.8], [7/13., 6/13.], [6/7., 1/7.]]) 

    def test_symmetrize_structure(self):
        """
        Tests symmetrize_structure()
        """

        # array: up in +z, end in +y 
        gg = np.zeros((10,10,10))
        gg[4,2:7,2:7] = np.array(
            [[0,0,0,0,0],
             [0,0,0,0,0],
             [1,2,3,4,5],
             [0,0,0,0,6],
             [0,0,0,0,0]])

        # mask
        ma = np.zeros((10,10,10))
        ma[4,2:7,2:7] = np.array(
            [[0,0,0,0,0],
             [0,0,0,0,0],
             [0,0,0,1,1],
             [0,0,0,0,1],
             [0,0,0,0,0]])

        # c2 mask
        res = relion_tools.symmetrize_structure(
            structure=gg, symmetry='C2', origin=[4,4,4], mask=ma)
        desired_4xx = np.array(
            [[0,0,0,0,0],
             [0,0,0,0,3],
             [0,0,0,4,5],
             [0,0,0,0,3],
             [0,0,0,0,0]])
        desired_x4x = np.array(
            [[0,0,0,0,0],
             [0,0,0,0,0],
             [0,0,0,4,5],
             [0,0,0,0,0],
             [0,0,0,0,0]])
        np_test.assert_almost_equal(res[4,2:7,2:7], desired_4xx)
        np_test.assert_almost_equal(res[2:7,4,2:7], desired_x4x)

        # c4
        res = relion_tools.symmetrize_structure(
            structure=gg, symmetry='C4', origin=[4,4,4])
        desired_4xx = np.array(
            [[0,0,0,0,0],
             [0,0,0,0,1.5],
             [1,2,3,4,5],
             [0,0,0,0,1.5],
             [0,0,0,0,0]])
        desired_x4x = np.array(
            [[0,0,0,0,0],
             [0,0,0,0,1.5],
             [1,2,3,4,5],
             [0,0,0,0,1.5],
             [0,0,0,0,0]])
        np_test.assert_almost_equal(res[4,2:7,2:7], desired_4xx)
        np_test.assert_almost_equal(res[2:7,4,2:7], desired_x4x)

        # d2
        res = relion_tools.symmetrize_structure(
            structure=gg, symmetry='D2', origin=[4,4,4])
        desired_4xx = np.array(
            [[0,0,0,0,0],
             [1.5,0,0,0,1.5],
             [3,3,3,3,3],
             [1.5,0,0,0,1.5],
             [0,0,0,0,0]])
        desired_x4x = np.array(
            [[0,0,0,0,0],
             [0,0,0,0,0],
             [3,3,3,3,3],
             [0,0,0,0,0],
             [0,0,0,0,0]])
        np_test.assert_almost_equal(res[4,2:7,2:7], desired_4xx)
        np_test.assert_almost_equal(res[2:7,4,2:7], desired_x4x)

        # d4
        res = relion_tools.symmetrize_structure(
            structure=gg, symmetry='D4', origin=[4,4,4])
        desired_4 = np.array(
            [[0,0,0,0,0],
             [0.75,0,0,0,0.75],
             [3,3,3,3,3],
             [0.75,0,0,0,0.75],
             [0,0,0,0,0]])
        np_test.assert_almost_equal(res[4,2:7,2:7], desired_4)
        np_test.assert_almost_equal(res[2:7,4,2:7], desired_4)


        

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRelionTools)
    unittest.TextTestRunner(verbosity=2).run(suite)


