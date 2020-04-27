"""
Contains common stuff for testing particles

Important note: Whenever data in this file are changed:

  1) Files set_data/segmentation/exp_*/connectors/*.pkl and
  set_data/analysis/work/tethers.pkl need to be removed. They will
  be generated next time test_set.py or test_label_set.py are 
  executed.

  2) File set_data/analysis/work.py may need to be modified

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id:$
"""
from __future__ import unicode_literals
from __future__ import division

__version__ = "$Revision$"

import importlib
import pickle
import os
import itertools

import numpy as np

import pyto
from pyto.grey.image import Image
from pyto.segmentation.morphology import Morphology
from pyto.segmentation.segment import Segment
from pyto.scene.segmentation_analysis import SegmentationAnalysis
from pyto.analysis.observations import Observations
from pyto.analysis.connections import Connections
from pyto.analysis.vesicles import Vesicles

# initialize variables set here
work_path_abs = None
catalogs_var = None
work = None

def setup():
    """
    Sets paths and imports analysis work module.
    """

    global tethers_pickle_path, tethers, sv_pickle_path, sv
    global work_path_abs, work, catalogs_var
    global particle_dir_rel, particle_dir_abs
    
    # structure specific pickles
    tethers_pickle_path = os.path.normpath(os.path.join(
        os.path.dirname(__file__), 
        'set_data/analysis/work/tether.pkl'))
        #'set_data/analysis/work/teth_sv_10.pkl'))
    sv_pickle_path = os.path.normpath(os.path.join(
        os.path.dirname(__file__),
        'set_data/analysis/work/sv_10.pkl'))
    make_struct_specific(
        tethers_path=tethers_pickle_path, sv_path=sv_pickle_path)
    tethers = pickle.load(
        open(tethers_pickle_path, 'rb'), encoding='latin1')
    sv = pickle.load(
        open(sv_pickle_path, 'rb'), encoding='latin1')

    # analysis work
    work_path_rel_here = 'set_data/analysis/work/work.py'
    work_path_rel_exec = os.path.join(
        os.path.dirname(__file__), work_path_rel_here)
    work_path_abs = os.path.abspath(
        os.path.normpath(work_path_rel_exec))
    #print("work_path_abs: {}".format(work_path_abs))
    work_dir, work_name = os.path.split(work_path_abs)
    work_base, work_ext = os.path.splitext(work_name)
    spec = importlib.util.spec_from_file_location(
        work_base, work_path_abs)
    work = spec.loader.load_module(spec.name)

    # catalog variable
    catalogs_var = 'tethers_file'

    # make labels and tomos
    for exp_i, group_i in zip([1, 2, 3], ['x', 'x', 'y']):
        
        # make paths
        pickle_path = os.path.join(
            os.path.dirname(work_path_abs),
            work.catalog_directory,
            work.catalog.tethers_file[
                'group_{}'.format(group_i)]['exp_{}'.format(exp_i)])
        pickle_path = os.path.normpath(pickle_path)
        tomo_path = os.path.join(
            os.path.dirname(pickle_path), '../3d/tomo-{}.mrc'.format(exp_i))
        tomo_path = os.path.normpath(tomo_path)

        # make datasets if not exist
        sa, label_particles, tomo, tomo_particles, coords = make_test_datasets(
            pickle_path=pickle_path, tomo_path=tomo_path,  mode=exp_i)

        mode_str = str(exp_i)
        globals()['pickle_path_'+mode_str] = pickle_path
        globals()['sa_'+mode_str] = sa
        globals()['label_particles_'+mode_str] = label_particles
        globals()['coords_'+mode_str] = coords
        globals()['tomo_path_'+mode_str] = tomo_path
        globals()['tomo_'+mode_str] = tomo
        globals()['tomo_particles_'+mode_str] = tomo_particles

    # particles
    particle_dir_rel = 'set_data/particles/set_x'
    particle_dir_abs =  os.path.normpath(os.path.join(
        os.path.dirname(__file__), particle_dir_rel))
    
def make_struct_specific(tethers_path, sv_path=None):
    """
    Makes tethers and sv specific pickles if needed
    """

    if not os.path.exists(tethers_path):

        # make object
        tethers = Connections(mode='connectors')
        group_x = Observations()
        group_x.setValue(
            name='ids', identifier='exp_1',
            value=np.array([2,4,5]), indexed=True)
        group_x.setValue(
            name='ids', identifier='exp_2',
            value=np.array([]), indexed=True)
        tethers['group_x'] = group_x
        group_y = Observations()
        group_y.setValue(
            name='ids', identifier='exp_3',
            value=np.array([6,7]), indexed=True)
        tethers['group_y'] = group_y

        # write
        pickle.dump(tethers, open(tethers_path, 'wb'), -1)
            
    if sv_path is not None:
        if not os.path.exists(sv_path):
            pass
        
def make_test_datasets(pickle_path, tomo_path, mode):
    """
    Makes an individual dataset pickle and tomo to be used for testing
    """

    # initialize object and coords
    labels = Segment()
    class Dummy(object): pass
    coords = Dummy()

    if mode == 1:
    
        # labels
        labels_data = np.zeros((10, 10, 10), dtype='int')
        labels_data[2,0,3] = 2
        labels_data[5:8,3,8] = 4
        labels_data[7:9, 8:9, 9] = 5
        labels.inset = [slice(10,20), slice(0,10), slice(30,40)]

        # tomo
        data = np.ones((40,40,40), 'float32')
        data[10:20, 0:10, 30:40] = np.where(labels_data==0, 1, -labels_data)

        # coordinates
        coords.centers_labels_init_rel = np.array(
            [[2, 0, 3], [6, 3, 8], [8, 8, 9]])
        coords.centers_init_abs = np.array(
            [[12, 0, 33], [16, 3, 38], [18, 8, 39]])
        coords.left_corners_6_adj_abs = np.array(
            [[9, 0, 30], [13, 0, 34], [15, 5, 34]])
        coords.centers_6_adj_abs = np.array(
            [[12, 3, 33], [16, 3, 37], [18, 8, 37]])
        coords.left_corners_labels_6_adj_rel = np.array(
            [[-1, 0, 0], [3, 0, 4], [5, 5, 4]])
        coords.centers_labels_6_adj_rel = np.array(
            [[2, 3, 3], [6, 3, 7], [8, 8, 7]])

    elif mode == 2:
        
        # labels
        labels_data = np.zeros((10, 10, 10), dtype='int')
        labels.inset = [slice(10,20), slice(0,10), slice(30,40)]

        # tomo
        data = np.ones((40,40,40), 'float32')
        data[10:20, 0:10, 30:40] = np.where(labels_data==0, 1, -labels_data)

        # coordinates
        coords.centers_labels_init_rel = np.array([])
        coords.centers_init_abs = np.array([])
        coords.left_corners_6_adj_abs = np.array([])
        coords.centers_6_adj_abs = np.array([])
        coords.left_corners_labels_6_adj_rel = np.array([])
        coords.centers_labels_6_adj_rel = np.array([])
        
    elif mode == 3:
    
        # labels
        labels_data = np.zeros((10, 10, 10), dtype='int')
        labels_data[0, 0, 0] = 6
        labels_data[9, 9, 9] = 7
        labels.inset = [slice(0,10), slice(10,20), slice(0,10)]

        # tomo
        data = np.ones((10,20,10), 'float32')
        data[0:10, 10:20, 0:10] = np.where(labels_data==0, 1, -labels_data)

        # coordinates
        coords.centers_labels_init_rel = np.array(
            [[0, 0, 0], [9, 9, 9]])
        coords.centers_init_abs = np.array(
            [[0, 10, 0], [9, 19, 9]])
        coords.left_corners_6_adj_abs = np.array(
            [[0, 7, 0], [4, 14, 4]])
        coords.centers_6_adj_abs = np.array(
            [[3, 10, 3], [7, 17, 7]])
        coords.left_corners_labels_6_adj_rel = np.array(
            [[0, -3, 0], [4, 4, 4]])
        coords.centers_labels_6_adj_rel = np.array(
            [[3, 0, 3], [7, 7, 7]])

    else:
        raise ValueError(
            "The value of argument mode: {} is not correct.".format(mode))
        
    labels.setData(data=labels_data, copy=False)
    morphology = Morphology(segments=labels)
    sa = SegmentationAnalysis()
    sa.segments = labels
    sa.morphology = morphology

    # pickle the object
    if not os.path.exists(pickle_path):
        pickle.dump(sa, open(pickle_path, 'wb'), -1)

    # label particle images
    label_particles = [
        labels.newFromInset(
            inset=[slice(begin, begin+6) for begin in corner],
            mode='relative', expand=True)
        for corner in coords.left_corners_labels_6_adj_rel]  
    
    # make tomo
    tomo = Image(data=data)
    if not os.path.exists(tomo_path): tomo.write(file=tomo_path)

    # tomo particle images
    tomo_particles = [
        tomo.newFromInset(
            inset=[slice(begin, begin+6) for begin in corner],
            mode='absolute', expand=False)
        for corner in coords.left_corners_6_adj_abs]  

    return sa, label_particles, tomo, tomo_particles, coords
    
