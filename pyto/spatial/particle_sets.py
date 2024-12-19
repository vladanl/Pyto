"""
Particle sets for colocalization
 
# Author: Vladan Lucic 
# $Id$
"""

__version__ = "$Revision$"

import os
import importlib

import numpy as np
import pandas as pd

import pyto
from . import coloc_functions as col_func
from pyto.segmentation.labels import Labels

# only for from_pyseg()
# For some reason importing the same module the second time works, even
# though importing the first time raises exception (because of dependencies)
pyseg_import_failed = False
try:
#    from pyorg import sub
    from pyorg.sub.star import Star
except ModuleNotFoundError:
    try:
       from pyorg.sub.star import Star 
    except (ModuleNotFoundError, NameError):
        pyseg_import_failed = True
try:  
    from pyorg.globals import unpickle_obj
except ModuleNotFoundError:
    try:
        from pyorg.globals import unpickle_obj
        unpickle_obj
    except (ModuleNotFoundError, NameError):
        pyseg_import_failed = True
if pyseg_import_failed:
    print(
        "Warning: Pyseg could not be loaded. Therefore, calling "
        + " pyto.spatial.ParticleSetsfrom_pyseg() will fail, but everything "
        + "else should be fine.")
    

class ParticleSets:
    """Particle sets for colocalization

    """

    def __init__(self, name_mode='_', index=True):
        """Sets attributes from arguments and initializes data structures 

        Arguments:
          - name_mode: specifies how are particle set names obtained from 
          coloclization name
          - index, flag indication if this istance should contain index
          variable, which is particularly important for attribute data_df
        """
        
        self.name_mode = name_mode

        # indicates if index should be read to and written from data_df 
        self.index = index

        # data structures
        self._coords = {}
        self._index = None  # set to {} only if needed
        self._regions = {}
        self._region_paths = {}
        self._pixel_nm = {}

        # dataframe related
        self.tomo_col = 'tomo_id'
        self.set_name_col = 'set'
        self.region_path_col = 'region_path'
        self.pixel_col = 'pixel_nm'
        self.coord_cols = ['x', 'y', 'z']
        self.coord_col_prefix = 'coord_'
        self.in_region_col = 'in_region'
        #self.index_col = 'index'
        
        
    ##########################################################
    #
    # Methods that access data directly
    #
        
    def set_coords(self, tomo, set_name, value):
        """Sets coordinates 

        """

        value = np.asarray(value)
        try:
            self._coords[tomo][set_name] = value
        except KeyError:
            self._coords[tomo] = {}
            self._coords[tomo][set_name] = value
        
    def get_coords(self, tomo=None, set_name=None, catch=False):
        """Gets particle coordinates for the specified tomo and set_name. 

        If both (args) tomo and set_name already exist (that is they are in
        self.tomos and self.set_names), but the particle coordinates
        for this particular combination were not set, returns None.

        If (arg) tomo or set_name do not exist (that is at least one of
        them is not in self.tomos and self.set_names, respectively),
        raises ValeError.

        If arg tomo (set_name) is None, returns a dictionary where keys
        are all tomos (set_names) and the values are coordinates for the
        specified set_name (tomo). At least one of (tomo, set_names) has
        to be different from None.

        Arguments:
          - tomo: tomo name 
          - set_name: particle set name
          - catch: flag indicating whether and exception is rased if the
          specified tomo or set_name does not exist

        Returns point coordinates:
          - ndarray (n_points x n_dim) if neither tomo not set_names is None
          - dict where keys are tomos (set_names) and values coordinates 
          as stated above if tomos (set_names) are None
        """

        # multiple
        if set_name is None:
            res = dict(
                [(nam, self.get_coords(tomo=tomo, set_name=nam, catch=catch))
                 for nam in self.set_names])
            return res

        if isinstance(set_name, (list, tuple)):
            res = dict(
                [(nam, self.get_coords(tomo=tomo, set_name=nam, catch=catch))
                 for nam in set_name])
            return res

        if tomo is None:
            res = dict(
                [(to, self.get_coords(tomo=to, set_name=set_name, catch=catch))
                 for to in self.tomos])
            return res

        if isinstance(tomo, (list, tuple)):
            res = dict(
                [(to, self.get_coords(tomo=to, set_name=set_name, catch=catch))
                 for to in tomo])
            return res

        # single
        try:
            res = self._coords[tomo].get(set_name, None)
            if (res is None) and (set_name not in self.set_names):
                if catch:
                    res = None
                else:
                    raise ValueError(f"Set {set_name} does not exist")
        except KeyError:
            if catch:
                res = None
            else:
                raise ValueError(f"Tomo {tomo} does not exist")

        return res

    def set_index(self, tomo, set_name, value):
        """Sets index

        """

        value = np.asarray(value)
        if self._index is None:
            self._index = {}
        try:
            self._index[tomo][set_name] = value
        except KeyError:
            self._index[tomo] = {}
            self._index[tomo][set_name] = value
        
    def get_index(self, tomo=None, set_name=None, catch=False):
        """Gets particle index for the specified tomo and set_name. 

        If both (args) tomo and set_name already exist (that is they are in
        self.tomos and self.set_names), but the index
        for this particular combination were not set, returns None.

        If (arg) tomo or set_name do not exist (that is at least one of
        them is not in self.tomos and self.set_names, respectively),
        raises ValeError.

        If arg tomo (set_name) is None, returns a dictionary where keys
        are all tomos (set_names) and the values are indices for the
        specified set_name (tomo). At least one of (tomo, set_names) has
        to be different from None.

        If index is not set (self._index is None), returns None. 

        Arguments:
          - tomo: tomo name 
          - set_name: particle set name
          - catch: flag indicating whether and exception is rased if the
          specified tomo or set_name does not exist

        Returns index:
          - ndarray (n_points x n_dim) if neither tomo not set_names is None
          - dict where keys are tomos (set_names) and values coordinates 
          as stated above if tomos (set_names) are None
        """

        if self._index is None:
            return None
        
        # multiple
        if set_name is None:
            res = dict(
                [(nam, self.get_index(tomo=tomo, set_name=nam, catch=catch))
                 for nam in self.set_names])
            return res

        if isinstance(set_name, (list, tuple)):
            res = dict(
                [(nam, self.get_index(tomo=tomo, set_name=nam, catch=catch))
                 for nam in set_name])
            return res

        if tomo is None:
            res = dict(
                [(to, self.get_index(tomo=to, set_name=set_name, catch=catch))
                 for to in self.tomos])
            return res

        if isinstance(tomo, (list, tuple)):
            res = dict(
                [(to, self.get_index(tomo=to, set_name=set_name, catch=catch))
                 for to in tomo])
            return res

        # single
        try:
            res = self._index[tomo].get(set_name, None)
            if (res is None) and (set_name not in self.set_names):
                if catch:
                    res = None
                else:
                    raise ValueError(
                        f"Set {set_name} in tomo {tomo} does not exist")
        except KeyError:
            if tomo not in self.tomos:
                if catch:
                    res = None
                else:
                    raise ValueError(
                        f"Set {set_name} in tomo {tomo} does not exist")
            else:
                res = None

        return res

    def set_region_path(self, tomo, value, set_name=None, remove_region=True):
        """Sets region file name.

        If arg remove_region is True (default), it also sets the 
        corresponding region value to None in order to avoit that region 
        path and region are inconsistent.
        """

        # multiple
        if set_name is None:
            for nam in self.set_names:
                self.set_region_path(
                    tomo=tomo, value=value, set_name=nam)

        # single
        try:
            self._region_paths[tomo][set_name] = value
        except KeyError:
            self._region_paths[tomo] = {}
            self._region_paths[tomo][set_name] = value

        # remove current self._regions entry
        if remove_region:
            self.set_region(tomo=tomo, set_name=set_name, value=None)
        
    def get_region_path(self, tomo, set_name):
        """Gets region file name

        If both (args) tomo and set_name already exist (that is, they are in
        self.tomos and self.set_names), but the region_path
        for this particular combination is not set, returns None.

        If (arg) tomo or set_name do not exist (that is at least one of
        them is not in self.tomos and self.set_names, respectively),
        raises ValeError.


        """

        try:
            res = self._region_paths[tomo].get(set_name, None)
            if (res is None) and (set_name not in self.set_names):
                raise ValueError(f"Set {set_name} does not exist")
        except KeyError:
            if tomo not in self.tomos:
                raise ValueError(f"Tomo {tomo} does not exist")
            else:
                res = None

        return res

    def set_region(self, tomo, value, set_name=None):
        """Sets region array.

        """

        # multiple
        if set_name is None:
            for nam in self.set_names:
                self.set_region(
                    tomo=tomo, value=value, set_name=nam)

        # single
        try:
            self._regions[tomo][set_name] = value
        except KeyError:
            self._regions[tomo] = {}
            self._regions[tomo][set_name] = value
        
    def get_region(self, tomo, set_name, save_from_file=False):
        """Gets region

        """

        # try region array
        try:
            res = self._regions[tomo].get(set_name, None)
            if (res is None) and (set_name not in self.set_names):
                raise ValueError(f"Set {set_name} does not exist")
        except KeyError:
            if tomo not in self.tomos:
                raise ValueError(f"Tomo {tomo} does not exist")
            else:
                res = None

        # try region file name
        if res is None:
            try:
                file_ = self.get_region_path(tomo=tomo, set_name=set_name)
            except ValueError:
                pass
            else:
                if file_ is not None:
                    
                    # read image file and save image data and pixel
                    image = Labels.read(file=file_, memmap=True)
                    res = image.data
                    if save_from_file:
                        self.set_pixel_nm(tomo=tomo, value=image.pixelsize)
                        self.set_region(
                            tomo=tomo, set_name=set_name, value=image.data)
                        
        return res

    def set_pixel_nm(self, value, tomo=None):
        """Sets pixel size in nm

        If arg tomo is None, sets pixels size for all tomos to the 
        specified value.
        """
        if tomo is None:
            for tomo in self.tomos:
                self.set_pixel_nm(tomo=tomo, value=value)
        else:
            self._pixel_nm[tomo] = value

    def get_pixel_nm(self, tomo):
        """Gets pixel size in nm

        Raises ValeError if the specified tomo is not in self.tomos.
        """

        try:
            res = self._pixel_nm[tomo]
        except KeyError:
            if tomo not in self.tomos:
                raise ValueError(f"Tomo {tomo} does not exist")
            else:
                res = None
        return res
            
    @property
    def tomos(self):
        """Tomo names
        """
        res = [
            tomo_nam for tomo_nam in self._coords.keys()
            if self._coords.get(tomo_nam, None) is not None] 
        return res

    @property
    def set_names(self):
        """Particle set names """

        res = set(
            set_nam for tomo in self._coords for set_nam in self._coords[tomo]
            if self._coords[tomo].get(set_nam, None) is not None)
        res = list(res)
        return res

    ##########################################################
    #
    # Higher level methods that do not access data directly
    #
        
    def set_coords_index(self, tomo, set_name, coords, index):
        """Sets particle coords and index for the specified tomo and set_name. 

        """
        self.set_coords(tomo=tomo, set_name=set_name, value=coords)
        if index is not None:
            self.set_index(tomo=tomo, set_name=set_name, value=index)
        
    def get_coords_index(self, tomo=None, set_name=None, catch=False):
        """Gets particle coords and index for the specified tomo and set_name. 

        """
        coords = self.get_coords(tomo=tomo, set_name=set_name, catch=catch)
        index = self.get_index(tomo=tomo, set_name=set_name, catch=catch)
        return coords, index
        
    def set_coords_regpath(self, tomo, set_name, coords, region_path):
        """Sets particle coordinates and regions file name.

        First sets particle coordinates to arg coords and then region
        file name to arg region_path.

        """
        self.set_coords(tomo=tomo, set_name=set_name, value=coords)
        self.set_region_path(tomo=tomo, set_name=set_name, value=region_path)

    def setup_regions(self):
        """Sets all regions and pixels.

        Reads all region files, which sets regions and pixle sizes for
        all existing tomos.
        """
        for tomo in self.tomos:
            for set_nam in self.set_names:
                self.get_region(
                    tomo=tomo, set_name=set_nam, save_from_file=True)
        
    def get_n_points(self, tomo=None, set_name=None):
        """Returns number of points.

        If arg tomo (set_name) is None, returns number of points for  all 
        tomos (set_names). 

        Non-existing tomos in arg tomo and non-existing set names in arg 
        set_name are counted to have 0 points.

        Arguments:
          - tomo: tomo name 
          - set_name: particle set name
        """

        # both tomo and set_name multiple
        if (((tomo is None) or isinstance(tomo, (list, tuple)))
            and ((set_name is None) or isinstance(set_name, (list, tuple)))):
            if tomo is None:
                select_tomos = self.tomos
            else:
                select_tomos = tomo
            n_point = np.sum(
                [self.get_n_points(tomo=to, set_name=None)
                 for to in select_tomos])
            return n_point

        coords = self.get_coords(tomo=tomo, set_name=set_name, catch=True)
        if ((tomo is None) or isinstance(tomo, (list, tuple))
            or (set_name is None) or isinstance(set_name, (list, tuple))):

            # one multiple the other single
            n_point = np.sum(
                [co.shape[0] for key, co in coords.items()
                 if (co is not None)])

        else:

            # both tomo and set_name single 
            if coords is not None:
                n_point = coords.shape[0]
            else:
                n_point = 0

        return n_point
            
    @property
    def pixel_nm(self):
        """Pixel size in nm

        Returns dict of (tomo, pixel_size) values.

        Returns None for pixel size of tomos where pixel size was not set,
        but are in self.tomos.
        """
        res = dict([(to, self.get_pixel_nm(tomo=to)) for to in self.tomos])
        return res

    ##########################################################
    #
    # Working with other particle set formats
    #

    @classmethod
    def import_module(cls, path, curr_dir=None):
        """Imports module from a given path.

        If the specified path is relative, arg curr_dir is prepended to the
        arg path. If curr_dir is None, os.getpwd() is used.

        Arguments:
          - path: module path, absolute or relative
          - curr_dir: current directory

        Returns imported module
        """
        
        if curr_dir is None:
            curr_dir = os.getcwd()

        if os.path.isabs(path):
            abs_path = os.path.normpath(path)
        else:
            abs_path = os.path.normpath(os.path.join(curr_dir, path))
        spec = importlib.util.spec_from_file_location(
            os.path.basename(abs_path), abs_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        return mod
    
    def extract_pset_access(
            self, psets_module, set_names=None, mode='pyseg'):
        """Extracts particle set star files and ids.

        Looks in a module that contain information about particle 
        sets generated by pyseg (arg psets_module) in order to find
        variables whose values are paths to star files and ids that define
        pyseg-generated particles sets. 

        These star files and ids can be used to access particle set 
        coordinates. Namely, each star file has a column that contains 
        particle set pickle file paths (star file label _psPickleFile) and
        each id specifies the row where the pickle of interest is located 
        (indexed from 0).

        The star file paths and ids are found for particle sets contained 
        in arg set_names.

        Currently, there are two modes that define how variable names are 
        constructed in the particle set modules:

          mode 'pyseg' (used from 2023 for Munc13-SNAP25 colocalization):
            - star file path variables: 'star_' + pset_name
            - particle set ids: 'id_' + pset_id
          mode 'pyseg_2021' (used for the Sci advances 2021 colocalization): 
            - star file path variables: 'star_prefix' + pset_name
            - particle set ids: 'l_id_' + pset_id
          where pset is particle set name ('pre', 'post', ...) and pset_id
          are ints.

        Arguments:
          - set_names: (list) particle set names (such as 
          ["setX", "setY", "setZ"]). If None all sets are used
          - psets_module: module containing info about particle location
          - mode: 'pseg' (default) or 'pseg_2021'

        Returns pset_stars, pset_ids:
          - pset_stars: (dict) particle set star file paths where keys are
          particle set names and values are the corresponding pyseg tpl 
          particle set pickle paths
          - pset_ids: (dist) particle set ids where keys are particle 
          set ids and values are the corresponfing ids
        """

        if mode == 'pyseg':
            star_prefix = 'star_'
            id_prefix = 'id_'
        elif mode == 'pyseg_2021':
            star_prefix = 'in_star_'
            id_prefix = 'l_id_'

        if set_names is None:
            set_names = self.set_names
            
        if (mode == 'pyseg') or (mode == 'pyseg_2021'):    
            psets_dir = os.path.dirname(psets_module.__file__)
            pset_stars = dict([
                (set_nam,
                 os.path.normpath(os.path.join(
                      psets_dir, getattr(psets_module, star_prefix + set_nam))))
                for set_nam in set_names])
            pset_ids = dict([
                (set_nam, getattr(psets_module, id_prefix + set_nam))
                for set_nam in set_names])

        else:
            raise ValueError(
                "Currently, only modes 'pyseg' and 'pyseg_2021' are "
                + "implemented.") 

        return pset_stars, pset_ids
    
    def from_pyseg(
            self, coloc_name=None, set_names=None, psets_module=None,
            star_files=None, pset_ids=None,
            mode='pyseg', region_ids=None, tomo_id_mode=None, region_func=None,
            tomos=None, asint=False):
        """Imports particle coords from pyseg created particle sets

        Particle coordinates and the corresponding regions (where they are 
        located) can be specified in two ways:
          1) By args star_files and pset_ids (psets_module has to be None)
          2) Specifying module (arg psets_module) that contains variables 
          corresponding to star_files and pset_ids (args star_files and
          pset_ids) are ignored. A module or a path to the module can be 
          specified 
        In both cases, values of star_files and pset_ids are used to determine
        the path to the pickle file conataining the particle set of intrest.

        Arg mode specifies how variables are named in module psets_module.
        Specifically:
          - mode = 'pyseg_2021': names of particle star file variables have
          to start with 'in_star_', and of ids with 'l_id_'
          - mode = 'pyseg': names of particle star file variables have
          to start with 'star_', and of ids with 'id_'

        Region (tomo) ids can be specified in the following ways:
          - Directly from (arg) region_ids, where keys are paths to region 
          images (as in PySeg particle set pickles) and values are the ids.
          In this case region image paths have to be known before calling
          this method.
          - Specifying (arg) tomo_id_mode, in which case this method first 
          reads a region path (from PySeg particle set pixels) and then calls
          coloc_functions.get_tomo_id(mode=tomo_id_mode) to get the
          corresponding region id. See coloc_functions.get_tomo_id() doc
          for the available tomo_id_modes. In this case, the tomo_id_mode 
          corresponding to the intended use of this method has to be already 
          implemented in coloc_functions.get_tomo_id() (as argument mode).
          - Specifying custom made function that takes a region path and
          returns the corresponding region id. This is the most general way.
        One of args region_ids, tomo_id_mode and region_func has to be 
        specified, the first found (in this order) is used.
        
        Arguments:
          - coloc_name: colocalization name (such as "setX_setY_setZ")
          - set_names: (list) particle set names (such as 
          ["setX", "setY", "setZ"]). If None all sets are used
          - psets_module: module, or a path to the module containing info 
          about particle location
          - star_files: (dict) keys are particle set names and 
          values are objects that contain coordinates and region file paths 
          in pyseg created particle sets, or a single object used for all 
          particle sets
          - pset_ids: (dict) where keys are particle set names and values
          are ints that are needed to specify particle sets in pyseg
          created particle sets
          - mode: specifies how variables are named in psets_module
          - tomos: (list) ids of tomos specified in pyseg particle sets
          are used
          should be read, if None, all tomos for the specified particles   
          - region_ids: (dict) keys are paths to regions (tomos) and values
          their tomo ids
          - tomo_id_mode: mode used to determine tomo ids for region file
          paths using coloc_functions.get_tomo_id(mode=tomo_id_mode)
          - region_func: function that takes region file path as an
          arguments and returns tomo id
          - asint: flag indicating if point coordinates are converted 
          to ints
        """

        # find set_names if not specified
        if coloc_name is not None:
            set_names = col_func.get_names(name=coloc_name, mode=self.name_mode)
        elif set_names is None:
            set_names = self.set_names
            
        # get star files and ids from particle sets module, if specified
        if psets_module is not None:
            if isinstance(psets_module, str):
                psets_module = self.import_module(path=psets_module)
            psets_dir = os.path.dirname(psets_module.__file__)
            star_files, pset_ids = self.extract_pset_access(
                set_names=set_names, psets_module=psets_module,
                mode=mode)
        elif (star_files is None) or (pset_ids is None):
            raise ValueError(
                "Either argument psets_module, or both star_files and "
                + "pset_ids have to be specified.")

        # loop over sets
        for ind, set_nam in enumerate(set_names):

            # 
            if isinstance(star_files, dict):
                star_fi = star_files[set_nam]
            #elif isinstance(star_files, list):
            #    star_fi = star_files[ind]
            else:
                star_fi = star_files
            set_id = pset_ids[set_nam]
            
            # read pickle where data for this particle set is stored
            try:
                #tar = sub.Star()
                star = Star()
            except NameError:
                print(
                    "PySeg needs to be available on this system in order "
                    + "to run from_pyseg()")
                raise 
            star.load(star_fi)
            ltomos_pkl = star.get_element('_psPickleFile', set_id)
            ltp = unpickle_obj(ltomos_pkl)

            # read coords and tomo file names for all tomos
            for region_path, set_one in ltp.get_tomos().items():

                # figure out tomo id 
                if region_ids is not None:
                    to_id = region_ids[region_path]
                elif tomo_id_mode is not None:
                    to_id = col_func.get_tomo_id(
                        path=region_path, mode=tomo_id_mode)
                elif region_func is not None:
                    to_id = region_func(region_path)
                else:
                    raise ValueError(
                        "At least one of arguments region_ids, tomo_id_mode "
                        + "and region_func has to be specified.")

                # skip tomos that are not specified
                if tomos is not None:
                    if to_id not in tomos:
                        continue

                # get coordinates
                coords = set_one.get_particle_coords()
                if asint:
                    coords = coords.round().astype(int)
                self.set_coords_regpath(
                    tomo=to_id, set_name=set_nam, coords=coords,
                    region_path=region_path)
    
    # To implement: methods that impose convex hull on regions (and save?)

    def from_pyto(
            self, resolve_fn, coloc_name=None, set_names=None,
            psets_module=None, tomos=None, subclass_col='subclass_name',
            ignore_keep=False):
        """Imports particle coordinates from pyto generated particle sets.

        Arguments:
          - resolve_fn: function that resolves colocalization-like
          particle set names

        """

        # find set_names if not specified
        if coloc_name is not None:
            set_names = col_func.get_names(name=coloc_name, mode=self.name_mode)
        elif set_names is None:
            set_names = self.set_names

        # load particle sets module if needed
        if isinstance(psets_module, str):
            psets_module = self.import_module(path=psets_module)

        # loop over sets
        for ind, set_nam in enumerate(set_names):

            # read current set in MultiParticleSets form
            mps, mps_class_name, mps_class_num = resolve_fn(
                set_name=set_nam, psets_module=psets_module)
            mps.select(
                tomo_ids=tomos, class_names=[mps_class_name],
                class_numbers=mps_class_num, update=True)
            mps.subclass_name_col = subclass_col
            mps.particles[mps.subclass_name_col] = set_nam
            
            # add mps particles to this instance
            inst_local = mps.to_particle_sets(
                set_name_col=mps.subclass_name_col, ignore_keep=ignore_keep)
            self.add_data_df(inst_local.data_df)

            
    ######################################################
    #
    # Basic DatFrame operations
    #
            
    @property
    def data_df(self):
        """Data represented as pandas DataFrame.

        Puts tomo names, particle set names, coordinates, region paths
        and pixel size, but not regions in the resulting dataframe.
        """
        return self.get_data_df()

    def get_data_df(self, tomos=None, set_names=None):        
        """Data represented as pandas DataFrame.

        Puts tomo names, particle set names, coordinates, region paths
        and pixel size, but not regions in the resulting dataframe.

        Index has to be set for all or none of the existing particles.

        Arguments:
          - tomos: (list) tomo names, if None all tomos are used
          - set_names: (list) particle set names, if None all sets are used

        Returns (pandas.DataFrame) particles 
        """

        # arguments
        if tomos is None:
            tomos = self.tomos
        if set_names is None:
            set_names = self.set_names
        
        n_dim = None
        df = None
        df_created = False
        first_time = True
        for tomo in tomos:
            pixel = self.get_pixel_nm(tomo=tomo)
            for set_nam in set_names:

                # get data
                coords = self.get_coords(tomo=tomo, set_name=set_nam)
                if coords is None:
                    continue
                region_path = self.get_region_path(tomo=tomo, set_name=set_nam)
                if self.index:
                    index = self.get_index(tomo=tomo, set_name=set_nam)
                    if first_time:
                        if index is not None:
                            index_found = True
                        else:
                            index_found = False
                        first_time = False
                    if (index is None) and index_found:
                        raise ValueError(
                            f"Index is not set for set {set_nam} and tomo "
                            + f"{tomo} but it is set for some other set / tomo "
                            + "combinations. Index has to be set for all or none "
                            + "of the particles.")
                    if (index is not None) and not index_found:
                        raise ValueError(
                            f"Index is set for set {set_nam} and tomo {tomo} "
                            + "but it was not set for some other set / tomo "
                            + "combinations. Index has to be set for all or "
                            + "none of the particles.")
                else:
                    index = None

                # put in dataframe
                if n_dim is None:
                    n_dim, coord_cols = self.get_coord_cols(coords=coords)
                if n_dim is not None:
                    n_dim_local, _ = self.get_coord_cols(coords=coords)
                    if (n_dim_local is None):  # covers strange cases
                        continue
                    coord_dict = dict([
                        (coord_cols[dim_ind], coords[:, dim_ind])
                        for dim_ind in range(n_dim)])
                    df_dict = {
                        self.tomo_col: tomo, self.set_name_col: set_nam,
                        **coord_dict,
                        self.region_path_col: region_path,
                        self.pixel_col: pixel}
                    if index is not None:
                        local_df = pd.DataFrame(df_dict, index=index)
                        df = pd.concat([df, local_df], ignore_index=False)
                    else:
                        local_df = pd.DataFrame(df_dict)
                        df = pd.concat([df, local_df], ignore_index=True)
                    df_created = True

        # set table index if variable index row is present 
        #if index is not None:
        #    df.set_index(self.index_col)
                        
        return df

    @data_df.setter
    def data_df(self, df):
        """Sets values in this instance from the specified pandas dataframe.

        The order of coordinates, as seen in self.get_coords(tomo, set_name)
        is determined as follows:
          - If coordinate df column names are in self.coord_cols, the order is
          determined by the same as in self.coord_cols.
          - If coordinate df column names start with self.coord_col_prefix,
          the order is determined by sorting the column name parts that come
          after the prefix. In this case, parts that come after the prefix
          have to be integers.
        """

        # figure out columns for coordinates, make sure properly ordered
        coord_cols = [ax for ax in self.coord_cols if ax in df.columns]
        if len(coord_cols) == 0:
            coord_cols_int = [
                int(ax.removeprefix(self.coord_col_prefix))
                for ax in df.columns if ax.startswith(self.coord_col_prefix)]
            coord_cols_int.sort()
            coord_cols = [
                f"{self.coord_col_prefix}{ind}" for ind in coord_cols_int]
        n_dim = len(coord_cols)

        #
        grouped = df.groupby([self.tomo_col, self.set_name_col])
        for to_set, df_one in grouped:

            # set coords
            to, set_nam = to_set
            coords = df_one[coord_cols].values
            self.set_coords(tomo=to, set_name=set_nam, value=coords)

            # set index
            #if self.index_col in df_one:
            if self.index:
                #index = df_one[self.index_col].to_numpy()
                index = df_one.index.to_numpy()
                self.set_index(tomo=to, set_name=set_nam, value=index)

            # set region path
            if self.region_path_col in df_one.columns:
                reg = df_one[self.region_path_col].unique()
                if len(reg) == 1:
                    self.set_region_path(
                         tomo=to, set_name=set_nam, value=reg[0])
                else:
                    raise ValueError(
                        f"Dataframe could not be converted to ParticleSets "
                        + f"because multiple region paths ({reg}) are given "
                        + f"for tomo {to} and particle set name {set_nam}.")
        
            # set pixel
            if self.pixel_col in df_one.columns:
                pix = df_one[self.pixel_col].unique()
                if len(pix) == 1:
                    self.set_pixel_nm(tomo=to, value=pix[0])
                else:
                    raise ValueError(
                        f"Dataframe could not be converted to ParticleSets "
                        + f"because multiple pixel sizes ({pix}) are given "
                        + f"for tomo {to} and particle set name {set_nam}.")

    def add_data_df(self, df):
        """Adds the specified table (pandas.DataFrame) to the existing data.

        The row order and indexing) of the resuting self.data_df is not 
        guaranteed to be the same very time.

        If self.index is True, index from arg df is preserved.

        Argument:
          - df: (pandas.DataFrame) table to be added

        Updates the current instance.
        """

        curr_df = self.data_df
        if (curr_df is None) or (curr_df.shape[0] == 0):
            self.data_df = df
        else:
            if self.index:
                ignore_index = False
            else:
                ignore_index = True
            self.data_df = pd.concat([curr_df, df], ignore_index=ignore_index)
        
    ###################################################
    #
    # Extracting info from particle sets
    #
                
    def get_coord_cols(self, coords):
        """Return list of dataframe column names corresponding to coordinates.

        If arg coords contains points, it determined number of dimensions 
        as coords.shape[1] and then makes column names according to 
        attributes self.coord_cols and self.coord_col_prefix.

        Even if arg coords is an empty 2d ndarray 
        (np.array([]).reshape(0, ndim)), the above procedure is followed.

        If arg coords is np.array([]), or None, returns (None, None).
        """

        if (coords is not None):
            if (coords.size > 0) or len(coords.shape) == 2:
                n_dim = coords.shape[1]
            else:
                n_dim = None
        else:
            n_dim = None

        if n_dim is not None:
            if n_dim <= 3:
                cols = self.coord_cols[:n_dim]
            else:
                cols = [
                    f"{self.coord_col_prefix}{ind}" for ind in range(n_dim)] 
        else:
            cols = None

        return n_dim, cols
                
    def get_n_particles(self, set_names=None, normalize=None):
        """Gets number of particles in each set for each tomo.

        Returns dataframe containing number of particles in the specified sets

        Arguments:
          - set_names: (list) particle set names
          - normalize: name of the column used to normalize all other values 

        Returns (pandas.DataFrame) number of particles for each tomo (rows) 
        and particle set (columns)
        """

        # get dataframe and extract from there
        psets_df = self.data_df
        psets_grouped = psets_df.groupby(
            [self.tomo_col, self.set_name_col], as_index=False, 
            dropna=False)[['x']].count()
        n_particles = pd.pivot_table(
            psets_grouped, values='x',
            index=[self.tomo_col], 
            columns=[self.set_name_col])

        # convert nans to 0, ints, remove index name 
        n_particles.fillna(0, inplace=True)
        n_particles = n_particles.astype(int)
        n_particles = n_particles.reset_index().rename_axis(None, axis=1)

        # keep only specified sets
        if set_names is not None:
            n_particles = n_particles[[self.tomo_col] + set_names].copy()
            
        if normalize:
            set_names_clean = set_names.copy()
            set_names_clean.remove(normalize)
            n_particles[set_names_clean] = n_particles[set_names_clean].div(
                n_particles[normalize], axis=0)
            
        return n_particles       

    def in_region(self, tomos=None, set_names=None):
        """Determines whether particles belong to the corresponding regions.

        Important: This method is sutable only for region images where only
        one region is defined. This is because a region is defined as 
        all pixels >0 in the image specified by self.get_region_path() 
        (the same as image returned by self.get_region()). For region
        images that contain multiple regions, use 
        MultiParticleSets.in_region() instead.
        """

        # get coord columns
        data_df = self.data_df
        coord_cols = [ax for ax in self.coord_cols if ax in data_df.columns]

        grouped = data_df.groupby([self.tomo_col, self.set_name_col])
        for to_set, df_one in grouped:
            to, set_nam = to_set

            # skip non-specified tomos and sets
            if (tomos is not None) and (to not in tomos):
                continue
            if (set_names is not None) and (set_nam not in set_names):
                continue

            # make region coords table
            reg = self.get_region(tomo=to, set_name=set_nam)
            reg_coords = np.stack(np.nonzero(reg>0), axis=1)
            reg_coords_df = pd.DataFrame(reg_coords, columns=coord_cols)
            reg_coords_df[self.in_region_col] = True

            # determine whether in region
            coord_reg = (
                df_one
                .reset_index()
                .merge(reg_coords_df, left_on=coord_cols, right_on=coord_cols,
                       how='left', sort=False)
                .set_index('index'))
            coord_reg[self.in_region_col] = \
                coord_reg[self.in_region_col].replace({np.nan: False})

            # add in region to the table
            data_df.loc[df_one.index, self.in_region_col] = \
                coord_reg[self.in_region_col]

        return data_df
    
