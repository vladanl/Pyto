"""
Contains class TileSet for manipulation of tomogram tiles

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
#from __future__ import unicode_literals
#from __future__ import division
#from builtins import zip
#from past.utils import old_div

__version__ = "$Revision$"

import os
#import sys
#import subprocess
#import importlib
import pickle
import itertools
#import warnings

import numpy as np
#import scipy as sp
try:
    import pandas as pd
except ImportError:
    pass # Python 2
import pyto

from .set_path import SetPath
from .set import Set


class TileSet(Set):
    """
    """

    def __init__(
            self, catalog_var, tile_shape, tomo_tiles_dir, label_tiles_dir,
            catalog_dir, old_dir=None, new_dir=None, pickle_name='tiles.pkl',
            dir_mode=0o775, test=False):
        """
        Sets variables
        """

        # from argument
        self.catalog_var = catalog_var
        self.tile_shape = tile_shape
        self.tomo_tiles_dir = tomo_tiles_dir
        self.label_tiles_dir = label_tiles_dir
        self.pickle_name = pickle_name
        self.catalog_dir = catalog_dir
        self.old_dir = old_dir
        self.new_dir = new_dir
        self.dir_mode = dir_mode
        self.test = test

        # tiles name and directory formats
        #self.tomo_tiles_dir_format = '{}_/{}_tomo'
        #self.label_tiles_dir_format = '{}_/{}_labels'
        #self.tomo_tile_name_format = '{}_/{}_tomo_id-{}.mrc'
        #self.label_tile_name_format = '{}_/{}_labels_id-{}.mrc'
        self.tomo_tiles_dir_format = '{}_tomo'
        self.label_tiles_dir_format = '{}_labels'
        self.tomo_tile_name_format = '{}_tomo_id-{}.mrc'
        self.label_tile_name_format = '{}_labels_id-{}.mrc'

    def extract_tiles(
            self, struct, group_names, identifiers, tomo_proc=None,
            label_proc=None):
        """
        """

        self.tomo_proc = tomo_proc
        self.label_proc = label_proc

        # pick specified tomos
        for ident in identifiers:
            for g_name in group_names:
                if ident in struct[g_name].identifiers:
                    break
            else:
                raise ValueError(
                    f"Group name for identifier {ident} was not found.")

            # get absolute tile slices
            tile_slices_abs = self.get_tiles_abs(
                struct=struct, group_name=g_name, identifier=ident,
                catalog_dir=self.catalog_dir,
                old_dir=self.old_dir, new_dir=self.new_dir)
            self._tile_slices_abs = tile_slices_abs
            
            # prepare output
            self.make_paths(
                tile_slices_abs=tile_slices_abs, group_name=g_name,
                identifier=ident)

            # read tomo and write tomo tiles
            self._tomo = pyto.grey.Image.read(
                file=self._tomo_path, header=True, memmap=True)
            pixel_size = struct[g_name].getValue(
                identifier=ident, name='pixel_size')
            self.write_tiles(
                image=self._tomo, tile_slices_abs=tile_slices_abs,
                tile_paths=self._tomo_tile_paths, mode='tomo',
                processor=self.tomo_proc, pixel_size=pixel_size)

            # write boundary tiles
            self.write_tiles(
                image=self._seg.boundary, tile_slices_abs=tile_slices_abs,
                tile_paths=self._label_tile_paths, mode='labels',
                processor=self.label_proc)

            # add data
            self.add_data(group_name=g_name, identifier=ident)

        # pickle
        self.pickle()

    def get_tiles_abs(
            self, struct, group_name, identifier, catalog_dir,
            old_dir=None, new_dir=None):
        """
        Get absolute slices

        Sets:
          - self._tomo_path
          - self._seg

        Returns list of absolute slices
        """

        # get paths
        self._sp = pyto.particles.SetPath(catalog_var=self.catalog_var)
        self._individ_pickle_path = self._sp.get_pickle_path(
            group_name=group_name, identifier=identifier, struct=struct,
            catalog_dir=catalog_dir, old_dir=old_dir, new_dir=new_dir)
        self._tomo_path = self._sp.get_tomo_path(
            pickle_path=self._individ_pickle_path)

        # load individual pickle
        with open(self._individ_pickle_path, 'rb') as pickle_fd:
            seg = pickle.load(pickle_fd, encoding='latin1')
        self._seg = seg

        # get lower left corners
        # Note: Elements of self._left_corners_abs correspond to axes
        # (x, y ...); each element is a non-repetitive list of corner
        # coords for that axis. 3D coords for all corners is obtained by
        # cartesian product of the axis lists (elements of
        # self._left_corners_abs)
        self._left_corners_abs = []
        for image_dim, sl, tile_dim in zip(
                seg.boundary.data.shape, seg.boundary.inset, self.tile_shape):
            self._left_corners_abs.append(
                (sl.start + np.array(range(image_dim // tile_dim)) * tile_dim))

        # get absolute tile insets (order reversed to fit F array order
        self._left_corners_abs.reverse()
        tiles_abs = [
            [slice(x_coord, x_coord + self.tile_shape[0]),
             slice(y_coord, y_coord + self.tile_shape[1]),
             slice(z_coord, z_coord + self.tile_shape[2])]
            for z_coord, y_coord, x_coord
            in itertools.product(*self._left_corners_abs)]

        return tiles_abs

    def make_paths(self, tile_slices_abs, group_name, identifier):
        """
        Make paths for tomo and label tiles

        Sets:
          - self._tomo_tile_paths
          - self._label_tile_paths

        """

        # make tiles dirs
        self._tomo_tiles_dir = os.path.join(
            self.tomo_tiles_dir,
            self.tomo_tiles_dir_format.format(identifier))
        os.makedirs(self._tomo_tiles_dir, mode=self.dir_mode, exist_ok=True)
        self._label_tiles_dir = os.path.join(
            self.label_tiles_dir,
            self.label_tiles_dir_format.format(identifier))
        os.makedirs(self._label_tiles_dir, mode=self.dir_mode, exist_ok=True)

        # make tomo and label tile paths
        n_digits = np.log10(len(tile_slices_abs)).astype(int) + 1
        id_format = '{:0' + str(n_digits) + 'd}'
        self._ids = list(range(len(tile_slices_abs)))
        self._tomo_tile_paths = [
            os.path.normpath(os.path.join(
                self._tomo_tiles_dir,
                self.tomo_tile_name_format.format(
                    identifier, id_format.format(id_))))
            for id_ in self._ids]
        self._label_tile_paths = [
            os.path.normpath(os.path.join(
                self._label_tiles_dir,
                self.label_tile_name_format.format(
                    identifier, id_format.format(id_))))          
            for id_ in self._ids]

    def write_tiles(
            self, image, tile_slices_abs, tile_paths, mode, processor=None,
            pixel_size=None):
        """

        Sets:
          - self._label_ids
        """

        self._label_ids = []
        for id_, (sl, path) in enumerate(zip(tile_slices_abs, tile_paths)):

            # extract tiles
            tile_data = image.useInset(
                inset=sl,  mode=u'absolute', expand=False, update=False)
            tile = image.__class__(data=tile_data)
            
            # process tile
            if processor is not None:
                if mode == 'tomo':
                    tile = processor(tile)
                elif mode == 'labels':
                    tile = processor(tile, self._sp.tomo_info)
                    self._label_ids.append(tile.ids)
                else:
                    raise ValueError(
                        f"Argument mode: {mode} not understood. It has to be"
                        + " 'tomo' or 'labels'.")
                   
            # write tile
            if self.test:
                print(path)
            else:
                if pixel_size is None:  
                    tile.write(file=path, header=self._tomo.header)
                else:
                    tile.write(file=path, pixel=pixel_size)

    def add_data(self, group_name, identifier):
        """
        """

        # make current metadate table
        curr_metadata = pd.DataFrame({
            'identifier' : identifier, 'group_name' : group_name,
            'tomo_dir': self._tomo_tiles_dir,
            'label_dir': self._label_tiles_dir,
            'orig_tomo_path': self._tomo_path,
            'orig_pickle_path': self._individ_pickle_path},
            index = [0]
            )

        # add current metadata
        try:
            self.metadata = self.metadata.append(
                curr_metadata, ignore_index=True)
        except AttributeError:
            self.metadata = curr_metadata
        self.metadata.drop_duplicates().reset_index(drop=True)
        
        # make current data table
        tomo_names = [os.path.split(pat)[1] for pat in self._tomo_tile_paths]
        label_names = [os.path.split(pat)[1] for pat in self._label_tile_paths]
        left_cor_abs = np.array(
            [[sl.start for sl in corner]
             for corner in self._tile_slices_abs])
        curr_data = pd.DataFrame({
            'identifier' : identifier, 'group_name' : group_name,
            'image_id':self._ids,
            'tomo_name' : tomo_names,
            'label_name' : label_names,
            'left_corner_x' : left_cor_abs[:, 0],
            'left_corner_y' : left_cor_abs[:, 1],
            'left_corner_z' : left_cor_abs[:, 2],
            'label_ids': self._label_ids}
        )

        # add current data
        try:
            self.data = self.data.append(curr_data, ignore_index=True)
        except AttributeError:
            self.data = curr_data

    def pickle(self):
        """
        Removes unpickleable and large stuff of this instance and 
        pickles the rest.
        """

        if self.pickle_name is not None:

            # remove unpickable (module)
            self._sp.tomo_info = None

            # remove large
            self._tomo = None
            self._seg = None

            # pickle
            tiles_pickle_path = os.path.join(
                self.tomo_tiles_dir, self.pickle_name)
            with open(tiles_pickle_path, 'wb') as pickle_fd:
                pickle.dump(self, pickle_fd)
            print(f"Saved pickle {tiles_pickle_path}") 

    def normalize(
            self, image, dtype=None, mean=None, std=None,
            min_limit=None, max_limit=None):
        """
        Normalizes image data and changes dtype.
        
        """

        image.normalize(
            mean=mean, std=std, min_limit=min_limit, max_limit=max_limit)
        if dtype is not None:
            #image.data = np.array(image.data)
            image.data = image.data.astype(dtype, order='K')

        return image

    def adjust_labels(
            self, image, tomo_info, membrane_id=0, vesicle_id=0, presyn_id=0,
            dtype=None):
        """
        """

        # initialize
        remove_old_ids = []
        reassign = {}
        keep_new_ids = []

        # membrane id
        old_mem_id = np.setdiff1d(tomo_info.boundary_ids, tomo_info.vesicle_ids)
        if membrane_id == 0:
            remove_old_ids += old_mem_id
        else:
            reassign.update(dict((old, membrane_id) for old in old_mem_id))
            keep_new_ids.append(membrane_id)

        # vesicle ids
        old_ves_ids = tomo_info.vesicle_ids
        if vesicle_id == 0:
            remove_old_ids += old_ves_ids
        else:
            reassign.update(dict((old, vesicle_id) for old in old_ves_ids))
            keep_new_ids.append(vesicle_id)

        # presynaptic cytoplasm id
        old_presyn_id = tomo_info.segmentation_region
        if presyn_id == 0:
            remove_old_ids.append(old_presyn_id)
        else:
            reassign.update({old_presyn_id : presyn_id})
            keep_new_ids.append(presyn_id)

        # change labels
        # Note: remove_old_ids and keep_new_ids are currently not used
        # because clean=True; kept in the code just in case need to change
        image.reorder(order=reassign, clean=True)
         
        # change dtype
        if dtype is not None:
            image.data = image.data.astype(dtype, order='K')
        #print(f"final ids {image.ids}")
            
        return image
