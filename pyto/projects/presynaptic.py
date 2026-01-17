"""
Contains class Presynaptic and functions for retrieving and further 
analysis of presynaptic project results obtained by 
scripts/presynaptic_stats.py.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""

__version__ = "$Revision$"


import os
import sys
import pickle
from copy import copy, deepcopy
import inspect

import numpy as np
import pandas as pd

import pyto
from ..core.image import Image
from ..io.module_io import ModuleIO
from ..io.pandas_io import PandasIO
from ..segmentation.neighborhood import Neighborhood
from ..particles.set_path import SetPath
from ..spatial.line_projection import LineProjection
from ..spatial.boundary import BoundaryNormal
from ..spatial.multi_particle_sets import MultiParticleSets


class Presynaptic(object):
    """Manipulation of the presynaptic analysis results.

    Includes further analysis of existing properties as well as calculating
    new properties from segmented structures.

    These results are expected to be obtained by scripts.presynaptic_stats.py 
    script.
    """

    def __init__(self, df_io=None, calling_dir=None, **kwargs):
        """Sets attributes from arguments.

        If arg df_io is not given and calling_dir is given, sets df_io 
        attribute to a new PandasIO object that uses calling_dir.

        """

        # save all args
        self.save_args(kwargs=kwargs)

        # make df_io if needed
        if (df_io is None) and (calling_dir is not None):
            self.df_io = PandasIO(calling_dir=calling_dir)

    def save_args(self, ignore=[], kwargs=None):
        """Saves calling function arguments as attributes (meant for __init__).

        Argument:
          - ignore: (list) argument names that are not set as attributes
        """
         
        frame = inspect.currentframe().f_back
        args, _, keywords, local_vars = inspect.getargvalues(frame)
        [setattr(self, name, local_vars[name]) for name in args
         if name not in ['self'] + ignore]

        for name, value in kwargs.items():
            setattr(self, name, value)
            
    def write_table(
            self, table, base, file_formats=['pkl', 'json', 'hd5'],
            hdf5_name=None, out_desc='', overwrite=True, verbose=True):
        """Writes pandas.DataFrame in the pickle and other formats.

        The same as PandasIO.write_table().
        """
        
        df_io = self.df_io
        if df_io is None:
            df_io = PandasIO(calling_dir=self.calling_dir, verbose=verbose)
        else:
            df_io.verbose = verbose

        df_io.write_table(
            table=table, base=base, file_formats=file_formats,
            hdf5_name=hdf5_name, out_desc=out_desc, overwrite=overwrite)
            
    def read_table(
             self, base, file_formats=['pkl', 'json', 'hd5'],
             hdf5_name=None, out_desc='', overwrite=True, verbose=True):
        """Reads pandas.DataFrame tables from files of different format.

        The same as PandasIO.read_table().
        """
        
        df_io = self.df_io
        if df_io is None:
            df_io = PandasIO(calling_dir=self.calling_dir, verbose=verbose)
        else:
            df_io.verbose = verbose

        table = df_io.read_table(
            base=base, file_formats=file_formats,
            hdf5_name=hdf5_name, out_desc=out_desc, overwrite=overwrite)

        return table
            
    def load(self, path, preprocess=True):
        """Loads presynaptic project results module.

        Arguments:
          - path: path to the presynaptic project script (like 
          pyto.scripts.presynaptic_analysis.py) that was previously 
          executed to obtain multi-tomo presynaptic analysis results 
          - preprocess: (default True) flag indicating whether to
          preprocesses the data (like the argument of the main() in 
          the presynaptic script)

        Returns presynaptic module
        """

        mod_io = ModuleIO(calling_dir=self.calling_dir)
        work = mod_io.load(path=path, preprocess=preprocess)
        return work

    def format(self, scalar, indexed):
        """
        Modifies tethers / connectors presynaptic analysis tether result tables.

        Currently only renames boundaryDistance and boundaryDistance_nm columns
        to boundary_distance and boundary_distance_nm.

        Arguments:
          - scalar: (DataFrame) scalar properties table
          - indexed: (DataFrame) tethers / connectors indexed properties table
        """

        # change labels
        indexed = indexed.rename(columns={
            'boundaryDistance': 'boundary_distance', 
            'boundaryDistance_nm': 'boundary_distance_nm'})
       
        return scalar, indexed

    #
    # Functions that calculate additional features
    #
    
    def rank_thresholds(
            self, indexed, scalar, indexed_other=None,
            rank_name='tether_rank', identifiers=None,
            out_full=True, out_path=None):
        """

        If a dataframe containing only the calculated ranks is returned, 
        indices are the same as in (arg) indexed dataframe.
        """

        if indexed_other is None:
            indexed_other = indexed
        if identifiers is None:
            identifiers = indexed['identifiers'].unique()

        # loop over identifiers
        for ident in identifiers:

            # get min and max threshold at which tethers and structures
            # were detected
            indexed_oth_cond = (indexed_other['identifiers'] == ident)
            other_thresholds = indexed_other.loc[
                indexed_oth_cond, 'thresh'].unique()
            indexed_cond = (indexed['identifiers'] == ident)
            thresholds = indexed.loc[indexed_cond, 'thresh'].unique()
            detect_thresholds = np.union1d(other_thresholds, thresholds)
            thresh_min, thresh_max = (
                detect_thresholds.min(), detect_thresholds.max())

            # find all thresholds between the detected min and max
            scalar_cond = (scalar['identifiers'] == ident)
            all_thresh = scalar.loc[scalar_cond, 'thresholds'].to_numpy()[0]
            min_max_thresh = all_thresh[
                (all_thresh >= thresh_min) & (all_thresh <= thresh_max)]

            # rank thresholds between min and max
            rank = np.linspace(0, 1, len(min_max_thresh))

            # rank actual thresholds
            thresh_rank_df_loc = indexed.loc[indexed_cond, ['thresh']].apply(
                lambda x: rank[np.searchsorted(min_max_thresh, x)])

            # concatenate current ranks with all ranks
            try:
                thresh_rank_df = pd.concat(
                    [thresh_rank_df, thresh_rank_df_loc])
            except (NameError, UnboundLocalError):
                thresh_rank_df = thresh_rank_df_loc

        # add rank to indexed 
        result = thresh_rank_df.rename(columns={'thresh': rank_name})

        if out_full:
            result = indexed.join(result)

        # save 
        if out_path is not None:
            if out_full:
                out_str = "indexed table with threshold ranks added"
            else:
                out_str = "threshold rank table"
            self.write_table(table=result, base=out_path, out_desc=out_str)

        return result

    def get_length(
            self, indexed, scalar, pkl_column, mode, identifiers=None,
            groups=None, distance='b2c', line='straight', length_column=None,
            pixel_column='pixel_size', out_full=True, out_path=None):
        """
        """

        # loop over tomos
        for _, row in scalar.iterrows():

            # read data from the current row
            gr = row['group']
            if (groups is not None) and (gr not in groups):
                continue
            ident = row['identifiers']
            if (identifiers is not None) and (ident not in identifiers):
                continue
            pkl_path = row[pkl_column]

            # get pixel size
            if pixel_column is not None:
                pixel_size = row[pixel_column]
            else:
                pixel_size = None

            # calculate length
            length_df_loc = self.get_length_one(
                pkl_path=pkl_path, group_name=gr, identifier=ident,
                mode=mode, distance=distance, line=line, column=length_column,
                pixel_nm=pixel_size)

            # concatenate current lengths with all lengths
            try:
                length_df = pd.concat(
                    [length_df, length_df_loc], ignore_index=True)
            except (NameError, UnboundLocalError):
                length_df = length_df_loc

        # add to indexed if needed
        if out_full:
            length_df = pd.merge(
                indexed, length_df, on=['group', 'identifiers', 'ids'])

        # save 
        if out_path is not None:
            if out_full:
                out_str = "indexed table with lengths added"
            else:
                out_str = "lengths table"
            self.write_table(table=length_df, base=out_path, out_desc=out_str)

        return length_df

    def get_length_one(
            self, pkl_path, mode, identifier, column=None, group_name=None, 
            distance='b2c', line='straight', pixel_nm=None):
        """
        """

        # load structure
        struct = pickle.load(open(pkl_path, 'rb'), encoding='latin1')

        # shortcuts
        labels = struct.labels
        boundary = struct.boundary
        morphology = struct.morphology 

        # align labels and boundaries 
        enc_inset = struct.boundary.findEnclosingInset(
            inset=struct.labels.inset)
        labels.useInset(
            inset=enc_inset, mode='abs', expand=True, value=0, update=True)
        boundary.useInset(
            inset=enc_inset, mode='abs', expand=True, value=0, update=True);

        # calculate length
        labels.contacts.expand()
        morphology.getLength(
            segments=labels, boundaries=boundary,
            contacts=labels.contacts, distance=distance,
            line=line, mode=mode)
        labels.contacts.compactify()

        # convert to dataframe
        df_dict = {}
        if group_name is not None:
            df_dict['group'] = group_name
        df_dict['identifiers'] = identifier
        df = morphology.to_dataframe(begin=df_dict, names=['length'])

        # convert to nm
        if pixel_nm is not None:
            df['length_nm'] = pixel_nm * df['length']

        # rename columns
        if column is not None:
            columns_rename = {'length': column}
            if pixel_nm is not None:
                columns_rename['length_nm'] = f'{column}_nm'
        df = df.rename(columns=columns_rename)

        return df

    #
    # Functions that combine features from different tables
    #

    def get_length_distance(self, sv, tether, identifier=None):
        """
        Finds the lenght of the shortest tether for each proximal
        vesicle and shows it together with the vesicle distance
        to the active zone membrane.

        Assumes that all vesicle ids are higher than the active 
        zone (presynaptic) membrane id.

        Arguments:
          - sv: vesicles indexed table
          - tether: tethers indexed table
          - identifier:
        """

        # get vesicle id for each tether
        # assumes all vesicle ids are higher than the active zone membrane id
        tether['sv_ids'] = tether['boundaries'].map(lambda x: np.max(x))

        # pick relevant columns 
        sv_df = sv.rename(columns={'ids': 'sv_ids'})
        if identifier is not None:
            sv_cond = (sv_df['identifiers'] == identifier)
            sv_df = sv_df.loc[sv_cond, :]
        sv_df = sv_df[['group', 'identifiers', 'sv_ids', 'minDistance_nm']]
        tet_df = tether.rename(columns={'ids': 'tether_ids'})
        if identifier is not None:
            tet_cond = (tet_df['identifiers'] == identifier)
            tet_df = tet_df.loc[tet_cond, :]
        tet_df = tet_df[[
            'group', 'identifiers', 'tether_ids', 'sv_ids', 'length_nm']]

        # add tether info to svs
        sv_teth = pd.merge(
            sv_df, tet_df, on=['group', 'identifiers', 'sv_ids'], how='outer')

        # check for missing sv ids
        if sv_teth['sv_ids'].isnull().any():
            raise ValueError(
                "Found a tether with non-existing vesicle id. A possible reason "
                + "is that a sv id is lower than the presynaptic membrane id.")

        # convert tether ids back to ints
        sv_teth['tether_ids'] = sv_teth['tether_ids'].astype('Int64')

        # find min tether length for all vesicles and add to teh vesicles
        length_df = sv_teth.groupby(
            ['group', 'identifiers', 'sv_ids'])[['length_nm']].min()
        sv_length = pd.merge(
            sv_df, length_df, left_on=['group', 'identifiers', 'sv_ids'], 
            right_index=True)

        return sv_length

    #
    # Methods that extract info from boundaries and and manipulate images
    #
    
    def get_coords(self, indexed, scalar):
    #def __call__(self, indexed, scalar):
        """Returns converted coordinates from multiple tomos.

        Given a function that returns some coordinates of one tomo 
        (self.func), this method gets those 
        coordinates for multiple tomos, converts the coordinates to 
        another reference (image) frame and puts them all together.

        For example, if self.func is get_contacts_one(), which returns 
        tether contact point coordinates of one tomo 
        (args region_id and coord_cols args are fixed), this
        method returns tether contact coords converted to another 
        reference frame (such as a cropped tomo) from all specified tomos.

        Specifically, this function loops over multiple tomos (using
        tomo_generator(), and for each tomo executes the specified 
        one tomo function (self.func) and converts the coordinates  
        to another reference frame (such as a cropped tomo) using 
        convert_coordinates(). This conversion is performed according 
        to the specified bin factor (self.bin_factor) and tomo offset 
        (self.offsets).

        The results obtained for all tomos are cobmined in one DataFrame
        and returned.

        The function that is executed has to be specified previously as
        self.func. It is applied to one tomo and it has to have the 
        following arguments:
          - identifier: tommo identifier
          - scalar: (pandas.DataFrame) scalar properties of one tomo
          - indexed: (pandas.DataFrame) indexed properties of one tomo
          - object such as pyto.scene.SegmentationAnalysis that contains 
          attributes:
            - scene.labels: segments
            - scene.boundary: regions
            - scene.labels.ids: segment ids
          The function has to return coordinates (pandas.DataFrame)
        
        For example, this function can be obtained from get_contacts_one()
        by fixing arguments region and coord_cols (using functools.partial).

        Function convert_coordinates() converts coordinates obtained

        The following attributes need to be set to execute 
        convert_coordinates():
          - self.coord_cols_pre: input coordinate column names
          - self.coord_cols_column: converted coordinate column names
          - self.in_col: name of a column that shows wheter the converted
          coordinates are inside the target tomos
          - self.bin_factor: bin factor from input coordinates to the target
          tomo (for example, if the original coordinates are from bin4 tomo
          and the target tomo is bin2, this factor should be set to 
          4/2 = 2)
          - self.offsets: (pandas.DataFrame) X/Y/Z offsets of the target 
          tomos 
          - self.offset_cols: names of columns containing offsets 
          - self.shape_cols: names of columns containing shapes of target 
          tomos
          - self.clean: flag indicating whether the coords that are outside 
          the target volume should be removed

        Arguments, both used to generate obects that are passed to self.func:
          - scalar: (pandas.DataFrame) scalar properties of all tomos
          - indexed: (pandas.DataFrame) indexed properties of all tomos
        
        Returns (pandas.DataFrame) converted coordinates for all tomos.
        """
        
        coords = None
        for ident, scalar_one, indexed_one, scene in tomo_generator(
                scalar=scalar, indexed=indexed, groups=self.groups,
                identifiers=self.identifiers, pickle_var=self.pickle_var,
                convert_path_common=self.convert_path_common,
                convert_path_helper=self.convert_path_helper):

            # get coordinates
            coords_one = self.func(
                identifier=ident, scalar=scalar_one, indexed=indexed_one,
                scene=scene)

            # convert to columns project
            if self.offsets is not None:
                coords_one = convert_coordinates(
                    identifier=ident, coords=coords_one,
                    coord_cols_in=self.coord_cols_pre,
                    coord_cols=self.coord_cols_column,
                    in_col=self.in_col, bin_factor=self.bin_factor,
                    offsets=self.offsets, offset_cols=self.offset_cols,
                    shape_cols=self.shape_cols, clean=self.clean)

            # combine current with previous coordinates
            if coords is not None:
                coords = pd.concat(
                    [coords, coords_one], ignore_index=True, sort=False)
            else:
                coords = coords_one
        
        return coords

    def shift_resize_image(
            self, indexed, scalar, out_dir, tomo_id_col, scene_attr,
            csv_offset_cols = ['x_offset', 'y_offset', 'z_offset'],
            csv_shape_cols = ['x_shape', 'y_shape', 'z_shape'],
            image_name_prefix=None, image_name_suffix=None, out_csv_name=None):
        """Shifts and resizes (segmented) image to the specified parameters.

        Shifts and resizes images (tomos) that belong to the projects specified
        by args indexed and scalar. The type of possibly multiple images
        that belong to the project that is processed here is specified by
        arg scene_attr. Images are adjusted based on target offsets and
        shapes specified in csv file self.offsets.

        More precisely, images to be transformed are saved as insets 
        (subtomograms relative to the full tomo size). Therefore, these 
        inset parameters and the target offsets and shapes are used to
        determine the transformed images.

        For example, segmentation images containing membranes and regions 
        that belong to a presynaptic project can be transformed to target
        offset and size obtained from another projects. In this case
        scene_attr='boundary'.

        Writes transformed images together with a csv file containing
        the final tomo offsets and shapes.

        """

        # setup output
        out_offsets = pd.DataFrame(
            columns= [tomo_id_col] + csv_offset_cols + csv_shape_cols)
        if not os.path.exists(out_dir):
           os.makedirs(out_dir)
           
        for ident, scalar_one, indexed_one, scene in tomo_generator(
                scalar=scalar, indexed=indexed, groups=self.groups,
                identifiers=self.identifiers, pickle_var=self.pickle_var,
                convert_path_common=self.convert_path_common,
                convert_path_helper=self.convert_path_helper):

            # get target inset for the current bin
            target_offset_row = self.offsets[self.offsets[tomo_id_col] == ident]
            if target_offset_row.shape[0] != 1:
                raise ValueError(
                    f"There should be exactly one row in offset table for "
                    + f"tomo {ident}, but here there are "
                    + f"{target_offset_row.shape[0]} ")
            target_off_row = target_offset_row.iloc[0]
            target_offset = (
                target_off_row[csv_offset_cols].to_numpy() // self.bin_factor)
            target_shape = (
                target_off_row[csv_shape_cols].to_numpy() // self.bin_factor)
            target_inset = [
                slice(off, off+shape) for off, shape
                in zip(target_offset, target_shape)]

            # adjust image to target at the current bin and write image
            if (image_name_prefix is not None):
                image = getattr(scene, scene_attr)
                image.useInset(
                    inset=target_inset, mode='abs', expand=True, update=True)
                out_path = os.path.join(
                    out_dir, image_name_prefix + ident + image_name_suffix)
                image.write(file=out_path)

            # save offsets
            df_local = pd.DataFrame({tomo_id_col: ident}, index=[0])
            df_local[csv_offset_cols] = target_offset
            df_local[csv_shape_cols] = target_shape
            out_offsets = pd.concat([out_offsets, df_local], ignore_index=True)

        # write offsets
        if out_csv_name is not None:
            out_offsets.to_csv(
                os.path.join(out_dir, out_csv_name), sep=" ", index=False)

    def shift_resize_boundary(
            self, indexed, scalar, out_dir, tomo_id_col,
            scene_attr_target, scene_attr_source='boundary', keep_ids=None,
            csv_offset_cols = ['x_offset', 'y_offset', 'z_offset'],
            csv_shape_cols = ['x_shape', 'y_shape', 'z_shape'],
            image_name_prefix=None, image_name_suffix=None, out_csv_name=None):
        """Shift and resize boundary images according to another image.

        Boundary images are expected to be segmented membranes and regions
        from a presynaptic project.

        Shifts and resizes boundary images (tomos) that belong to the projects 
        specified by args indexed and scalar so that they align with the 
        offset and size of the corresponding target images. 

        Target images are specified by arg scene_attr_target as follows:
          - 'boundary': Shift and resize are chosen to yield the smallest 
          image that contain entire segmented membranes and regions
          - 'labels': Agree with images containing molecular segments, 
          that is tethers if self.pickle_var='tethers_file'.

        Writes transformed images together with a csv file containing
        the final tomo offsets and shapes.

        Arguments:
          - keep_ids: If not None, keep only specified boundary ids
        """
        
        # setup output
        out_offsets = pd.DataFrame(
            columns= [tomo_id_col] + csv_offset_cols + csv_shape_cols)
        if not os.path.exists(out_dir):
           os.makedirs(out_dir)
           
        for ident, scalar_one, indexed_one, scene in tomo_generator(
                scalar=scalar, indexed=indexed, groups=self.groups,
                identifiers=self.identifiers, pickle_var=self.pickle_var,
                convert_path_common=self.convert_path_common,
                convert_path_helper=self.convert_path_helper):

            # adjust source image to target 
            source_image = getattr(scene, scene_attr_source)
            target_image = getattr(scene, scene_attr_target)               
            source_image.useInset(
                inset=target_image.inset, mode='abs', expand=True, update=True)

            # clean if needed
            if keep_ids is not None:
                source_image.keep(ids=keep_ids)
            
            # write image
            if (image_name_prefix is not None):                
                out_path = os.path.join(
                    out_dir, image_name_prefix + ident + image_name_suffix)
                try:
                    source_image.write(file=out_path)
                except ValueError:
                    if (target_image.data.shape == np.array([0, 0, 0])).any():
                        print(
                            f"WARNING: Tomo {ident} could not be written "
                            + "because it has 0 size.")
                    else:
                        print(f"WARNING: Tomo {ident} could not be written.")
                    pass
                    
            # save offsets
            df_local = pd.DataFrame({tomo_id_col: ident}, index=[0])
            target_offset = [x.start for x in target_image.inset]
            target_shape = [x.stop - x.start for x in target_image.inset]
            df_local[csv_offset_cols] = target_offset
            df_local[csv_shape_cols] = target_shape
            out_offsets = pd.concat([out_offsets, df_local], ignore_index=True)

        # write offsets
        if out_csv_name is not None:
            out_offsets.to_csv(
                os.path.join(out_dir, out_csv_name), sep=" ", index=False)

    @classmethod
    def restrict_align_one(
            cls, data, identifier, pickle_var, mode='tether', restrict=True,
            convert_path_common=None, convert_path_helper=None):
        """Align selected boundaries and segments for one tomo.

        Does the following:
          - Reads segments data (tethers or connectors) from multiple 
          tomo pickles (like work.tether) for the specified tomo 
          - Restricts vesicles to those that are bound to the segments 
          (so tethered or connected) and cuts the boundaries tomo 
          accordingly (optional)
          - Aligns segments tomo to the boundaries tomo
          - Returns object of this class containing the above generated

        Individual tomo objects comprising (arg) data have to have 
        attribute 'boundaries' (in indexed_data). If the mode is 'tether'
        the first element of all boundaries elements has to be the same
        and to be the id of the active zone membrane.

        Arguments:
          - data: (pyto.analysis.Groups) multi-tomo segments data,
          e.g. work.tether, where work is a presynaptic module (see load())
          - indetifier: tomo identifier
          - pickle var: name of the variable that contains path to 
          pickled single-tomo segments data (like 'tethers_file')
          - mode: (default 'tether') 'tether' if arg data is tether data, 
            or 'segment' if segment data
          - restrict: flag indicationg whether vesicles are restricted 
          to those that contact the segments
        - convert_path_common, convert_path_helper: (defaults None) paths
        used to change a beginning part of pickle file paths (see 
        tomo_generator doc)

        Return instance of this class, attributes:
          - segments: (pyto.segmentation.Segment) segments data
          - boundaries: (pyto.segmentation.Segment) restricted 
          boundaries data
          - scalar: (pandas.DataFrame) table containing segment scalar data
          - indexed: (pandas.DataFrame) table containing segment indexed data
          - az_id: id of the active zone membrane
          - sv_ids: vesicle ids
        """

        # get tether object and info, and boundary object
        ident, scalar, indexed, scene = [
            (ident, scalar_one, indexed_one, scene) 
            for ident, scalar_one, indexed_one, scene in tomo_generator(
                indexed=data.indexed_data, scalar=data.scalar_data,
                identifiers=[identifier], pickle_var=pickle_var,
                convert_path_common=convert_path_common,
                convert_path_helper=convert_path_helper)
            ][0]
        segments = scene.hierarchy
        boundaries = scene.boundary

        # plasma membrane and connected sv ids
        if mode == 'tether':
            pm_id = np.unique(
                np.vstack(indexed['boundaries'].to_numpy())[:, 0])[0]
            sv_ids = np.unique(
                np.vstack(indexed['boundaries'].to_numpy())[:, 1])
            bound_ids = np.hstack(([pm_id], sv_ids))
        elif mode == 'segment':
            sv_ids = np.unique(np.vstack(indexed['boundaries'].to_numpy()))
            bound_ids = sv_ids
        else:
            raise ValueError(
                f"Argument mode: {mode} was not understood, it has to "
                + "be 'tether' or 'segments'")
            
        # keep only connected svs, crop image
        if restrict:
            boundaries.keep(ids=bound_ids)
            boundaries.makeInset(ids=bound_ids)

        # align tethers with boundaries
        segments.useInset(inset=boundaries.inset, mode='abs', expand=True)

        # make object and set attributes
        inst = cls()
        inst.segments = segments
        inst.boundaries = boundaries
        inst.scalar = scalar
        inst.indexed = indexed
        if mode == 'tether':
            inst.az_id = pm_id
        inst.sv_ids = sv_ids

        return inst

    def find_synapse_direction(
            self, struct, pre_cyto_id, az_id=None, smooth_size=20,
            spherical_names=['synapse_phi_deg', 'synapse_theta_deg'],
            wedge_angle_names=['synapse_phi_y', 'synapse_theta_z']):
        """Determines synapse direction (angles).

        The direction is defined by spherical angles of the vector
        pointing from post to pre terminal.

        The angles are determined by calculating the normal vector
        to the presynaptic membrane oriented towards the presynaptic
        cytoplasm.

        Reguires the following attributes to be set:
          - self.groups
          - self.identifiers
          - self.pickle_var
          - self.convert_path_common
          - self.convert_path_helper
        
        Arguments:
          - struct (pyto.analysis.Connections) oject that contains
        the presynaptic analysis data about connectors or tetehers
        for all tomos, or path to the pickeled object
          - pre_cyto_id: id of presynaptic cytoplasm
          - az_id: id of the active zone membrane, if None (default) it
          is determined as the smallest boundary if
          - smooth_size: radius of a region on the active zone membrane
          used to obtain a weighted average of membrane normal vectors
        """

        # read structure pickle if needed
        if isinstance(struct, str):
            struct = pickle.load(open(struct, 'rb'), encoding='latin1')
    
        for generated in pyto.projects.presynaptic.tomo_generator(
                indexed=struct.indexed_data, scalar=struct.scalar_data,
                groups=self.groups, identifiers=self.identifiers,
                pickle_var=self.pickle_var,
                convert_path_common=self.convert_path_common,
                convert_path_helper=self.convert_path_helper):
            ident, scalar_one, indexed_one, data = generated
            
            if az_id is None:
                az_id = np.min(data.boundaryIds)

            # get normal and angles
            bound = BoundaryNormal(
                image=data.boundary, segment_id=az_id, external_id=pre_cyto_id,
                dist_max_segment=smooth_size)
            bound.find_normal_global()

            # convert angles
            phi = bound.spherical_phi_deg_global
            phi_y = np.abs(np.mod(phi, 180) - 90)
            theta = bound.spherical_theta_deg_global
            theta_z = 90 - np.abs(np.mod(theta, 180) - 90)

            # add converted angles to struct
            group_name = struct.get_identifier_group(identifier=ident)
            if len(group_name) > 1:
                raise ValueError(
                    f"Identifier {ident} exists in multiple groups "
                    + f"({group_name})")
            else:
                group_name = group_name[0]
            if spherical_names is not None:
                struct[group_name].setValue(
                    name=spherical_names[0], identifier=ident, value=phi,
                    indexed=False)
                struct[group_name].setValue(
                    name=spherical_names[1], identifier=ident, value=theta,
                    indexed=False)
            if wedge_angle_names is not None:
                struct[group_name].setValue(
                    name=wedge_angle_names[0], identifier=ident, value=phi_y,
                    indexed=False)
                struct[group_name].setValue(
                    name=wedge_angle_names[1], identifier=ident, value=theta_z,
                    indexed=False)
                
    def make_boundary_angles(
            self, indexed, scalar, bound_id, cyto_id, dist_max_segment,
            spherical=False, euler=True, relion=True,
            euler_mode='zyz_ex_active', degree=False, source_path=None):
        """Calculate normal vector angles for all boundary points.

        Reguires the following attributes to be set:
          - identifiers: tomo ids
          - pickle_var: name of the variable that points to the
          tether or connector structure pickle
          - conver_path_common,  convert_path_helper: paths
          used to change the beginning part of pickle file paths

        Core calculations are done by
        pyto.spatial.BoundaryNormal.find_normals().
        
        Arguments:
          - scalar: (pandas.DataFrame) scalar properties of all tomos
          - indexed: (pandas.DataFrame) indexed properties of all tomos
          - bound_id: id of region from which boundary is extracted
          - cyto_id: id of region that contacts bound_id region, which
          determines the side where boundary is extracted
          - dist_max_segment: length scale at which boundary normal
          vectors are smoothed
          - spherical: flag indicating if spherical angles are written
          in the resulting dataframe (default False)
          - euler: flag indicating if euler angles are written
          in the resulting dataframe (default True)
          - relion: flag indicating if euler angles are calculated
          in the relion way (euler convention active intrinsic zyz in degrees)
          (default True)
          - euler_mode, degree: specifies euler convention and angle units
          in case relion is False (default 'zyz_ex_active' and False,
          meaning radians)
          - source_path: path where the save the resulting MPS object
          (default None, meaning the object is not saved)

        Returns: (MultiParticleSets) object where attribute particles
        contains the calculated angles, as the following columns:
          - tomo_id: tomo identifier
          - coord-x/y/z: boundary point coordinates
          - spherical_phi, spherical_theta: spherical coordinates
          - rlnAngleRot, rlnAngleTilt, rlnAnglePsi: euler angles if in
          the relion convention
          - euler_phi, euler_theta, euler_psi: euler angles if not in
          the relion convention
        """

        # initialize source dataframe
        source = MultiParticleSets()
        source_cols = [source.tomo_id_col, *source.coord_cols]
        if spherical:
            source_cols += ['spherical_phi', 'spherical_theta']
        if euler:
            if relion:
                source_cols += ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
            else:
                source_cols += ['euler_phi', 'euler_theta', 'euler_psi']
        source_parts = []
        lp = pyto.spatial.LineProjection(
            relion=relion, euler_mode=euler_mode, degree=degree,
            euler_range='0_2pi')

        for ident, scalar_one, indexed_one, data \
                in tomo_generator(
                    indexed=indexed, scalar=scalar, 
                    identifiers=self.identifiers, pickle_var=self.pickle_var,
                    convert_path_common=self.convert_path_common, 
                    convert_path_helper=self.convert_path_helper):

            # heurisctic to find bound_id
            if bound_id is None:
                bound_id = np.min(data.boundaryIds)

            # find boundary normals and spherical
            bound = pyto.spatial.BoundaryNormal(
                image=data.boundary, segment_id=bound_id, external_id=cyto_id,
                dist_max_segment=dist_max_segment)
            bound.find_normals()

            # spherical to euler
            if euler:
                euler_angs = np.array([
                    lp.spherical_to_euler([ph, th]) for ph, th 
                    in zip(bound.spherical_phi_deg,
                           bound.spherical_theta_deg)])
            
            # put in a dataframe
            source_data = {
                source.tomo_id_col: ident,
                source.coord_cols[0]: bound.points[:, 0],
                source.coord_cols[1]: bound.points[:, 1],
                source.coord_cols[2]: bound.points[:, 2]}
            if spherical:
                source_data.update({
                    'spherical_phi': bound.spherical_phi,
                    'spherical_theta': bound.spherical_theta})
            if euler:
                if relion:
                    source_data.update({
                        'rlnAngleRot': euler_angs[:, 0],
                        'rlnAngleTilt': euler_angs[:, 1], 
                        'rlnAnglePsi': euler_angs[:, 2]})
                else:
                     source_data.update({
                         'euler_phi': euler_angs[:, 0],
                         'euler_theta': euler_angs[:, 1],
                         'euler_psi': euler_angs[:, 2]})
            source_parts.append(pd.DataFrame(source_data.copy()))

        # make source dataframe
        source.particles = pd.concat(source_parts, ignore_index=True)

        # return or save
        if source_path is not None:
            source.tomos = pd.DataFrame()
            source.write(
                path=source_path, out_desc='boundary angles source',
                verbose=True)

        return source.particles

    def make_tomos_df(
            self, indexed, scalar, coord_bin, region_bin,
            tomo_bin=None, tomo_path_var='image_file_name',
            labels_path_var = 'labels_file_name',
            pixel_size_var='pixel_size', tomo_col='tomo', 
            convert_path_common=None, convert_path_helper=None):
        """Makes tomos dataframe from presynaptic analysis data.

        Arguments:
          - scalar: (pandas.DataFrame) scalar properties of all tomos
          - indexed: (pandas.DataFrame) indexed properties of all tomos
          - coord_bin: bin factor of tomos from which tether coords are
          extracted
          - tomo_bin: bin factor of tomos used for averaging (default None,
          in which case coord_bin is used)
          - region_bin: regions bin factor (usually the same as coord_bin)
          - tomo_path_var: name of the variable that contains tomo path
          in tomo_info file (default 'image_file_name')
          - labels_path_var: name of the variable that contains regions path
          in tomo_info file (default 'labels_file_name')
          - pixel_size_var: pixel size column name (default 'pixel_size')
          - tomo_col: tomo path column name in the output DataFrame
          (default 'tomo')

        Returns (pandas.DataFrame) table where each row contains info
        about one tomo (infor is extracted from arguments).       
        """

        # initialize
        mps = MultiParticleSets()
        set_path_tomo = SetPath(
            common=convert_path_common, helper_path=convert_path_helper,
            tomo_path_var=tomo_path_var)
        set_path_regions = SetPath(
            common=convert_path_common, helper_path=convert_path_helper,
            tomo_path_var=labels_path_var)
        if tomo_bin is None:
            tomo_bin = coord_bin
        tomos = []
        tomo_path = []
        region_path = []
        region_shape = []
        pixel_nm = []

        # loop over tomos
        for ident, scalar_one, indexed_one, data in tomo_generator(
                indexed=indexed, scalar=scalar, 
                identifiers=self.identifiers, pickle_var=self.pickle_var,
                convert_path_common=convert_path_common, 
                convert_path_helper=convert_path_helper):
   
            # read tomo_info
            tomos.append(ident)
            pickle_path = scalar_one[self.pickle_var]
            tomo_path.append(
                set_path_tomo.get_tomo_path(pickle_path=pickle_path))
            reg_path = set_path_regions.get_tomo_path(pickle_path=pickle_path)
            region_path.append(reg_path)
            region_shape.append(
                Image.read(file=reg_path, memmap=True).data.shape)
            pixel_nm.append(scalar_one[pixel_size_var])

        # make dataframe
        tomos_data = {
            mps.tomo_id_col: tomos, tomo_col: tomo_path, 
            mps.region_col: region_path}
        tomos_data.update(dict((off, 0) for off in mps.region_offset_cols))
        region_shape = np.array(region_shape).transpose()
        tomos_data.update(
            dict((sh_col, reg_shape) for sh_col, reg_shape
                 in zip(mps.region_shape_cols, region_shape)))
        tomos_data.update(
            {mps.pixel_nm_col: pixel_nm, mps.coord_bin_col: coord_bin})
        try:
           tomos_data.update({mps.tomo_bin_col: tomo_bin})
        except AttributeError:
            tomos_data.update({'tomo_bin_col': tomo_bin})
        tomos_data.update(
             {mps.region_id_col: -1, mps.region_bin_col: region_bin})
        tomos = pd.DataFrame(tomos_data)

        return tomos
    
        
##########################################################
#
# Functions
#

def tomo_generator(
        indexed, scalar, groups=None, identifiers=None, pickle_var=None,
        convert_path_common=None, convert_path_helper=None):
    """Generator that yields scalar and indexed data for each tomogram.

    If arg pickle_var is specified, it is expected to be a name of a scalar
    variable (equivalently a column name of arg scalar) whose value is
    a path to a pickle file (for example 'tethers_file' or 'sv_file'). 
    
    If the systems where the above pickle was generated and where this
    method is executed have different file system organization, such
    that the path specified by arg pickle_var have different beginning,
    the path can be adjusted by args convert_path_common and 
    convert_path_helper. For example, if the old path (value of pickle_var) is:
        old_path/common/the_same_path
    arg convert_path_common:
        common
    and arg convert_path_helper:
        new_path/common/whatever_else
    the adjust path is (often resulting from os.getcwd()):
        new_path/common/the_same_path
    see also pyto.particles.SetPath doc.

    Arguments:
      - scalar: (pandas.DataFrame) presynaptic analysis scalar data
      - indexed: (pandas.DataFrame) presynaptic analysis indexed data
      - groups: selected group names, or None for all groups
      - identifierss: selected tomo identifiers, or None for all tomos
      - pickle_var: name of a scalar variable (column name of scalar) 
      whose value is a path to a pickle file
      - convert_path_common, convert_path_helper: (defaults None) paths
      used to change a beginning part of pickle file paths (see above)

    Yields:
      - identifier: tomo identifier
      - scalar_one: (pandas.Series) scalar data
      - indexed_one: (pandas.DataFrame) indexed data
      - scene (only if arg pickle_var is not None): unpickled object referenced
      by arg pickle_var
    """

    # select by groups and identifiers
    if groups is not None:
        cond_s = scalar.apply(lambda x: x['group'] in groups, axis=1)
        cond_i = indexed.apply(lambda x: x['group'] in groups, axis=1)
        scalar = scalar[cond_s]
        indexed = indexed[cond_i]
    if identifiers is not None:
        cond_s = scalar.apply(
            lambda x: x['identifiers'] in identifiers, axis=1)
        cond_i = indexed.apply(
            lambda x: x['identifiers'] in identifiers, axis=1)
        scalar = scalar[cond_s]
        indexed = indexed[cond_i]

    # loop
    for ind, scalar_one in scalar.iterrows():

        # standard
        ident = scalar_one['identifiers']
        indexed_one = indexed[indexed['identifiers'] == ident].copy()
        to_yield = (ident, scalar_one, indexed_one)

        # find and load a pickle
        if pickle_var is not None:
            pkl_path = scalar_one[pickle_var]
            if convert_path_common is not None:
                sp = SetPath(
                    common=convert_path_common, helper_path=convert_path_helper)
                pkl_path = sp.convert_path(path=pkl_path)
            data = pickle.load(open(pkl_path, 'rb'), encoding='latin1')
            to_yield = (ident, scalar_one, indexed_one, data)

        yield to_yield

def convert_coordinates(
        identifier, coords, coord_cols_in, coord_cols,
        offsets, offset_cols=None, shape_cols=None, bin_factor=1,
        in_col='in', clean=True):
    """Converts coordinates of points to the target system for one tomo.

    Intended for cases where the input coordinates are specified in the 
    original size, binned tomo, while the target tomos are subtomograms
    of the original tomo at another bin.

    To convert the input coords (arg coords), they are first multiplied 
    by arg bin_factor and then the offsets (arg offsets) are subtracted.
    For example, if (arg) coords are in bin-4 and target syatem bin-2, 
    arg bin_factor should be set to 4/2 = 2

    It is also determined whether the converted coordinates fall within 
    the target tomo.

    Arguments:
      - identifier: tomo id
      - coords: (ndarray n_points x n_dim) input coordinates, to be converted
      - coord_col_in: column names of input coordinates
      - coord_cols: column names ofoutput coordinates
      - offsets: (pandas.DataFrame) offsets and shapes of the target 
      subtomo
      - offset_cols: name of columns in the offset file that contain offsets 
      - shape_cols: name of columns in the offset file that contain shape
      of target tomos
      - bin_factor: bin factor, that is scaling factor that converts from
      (arg) coordinates to the target system
      - in_col: name of the column in the resulting table that shows wheteher 
      points are in target tomos
      - clean: flag indicationg if points that are outside target tomos are 
      removed

    Returns (pandas.DataFame) converted coordinates
    """

    # calculate converted coords for bin and offset
    coords[coord_cols] = bin_factor * coords[coord_cols_in]
    offset_row = offsets[offsets['identifiers'] == identifier]
    if offset_cols is not None:
        coords[coord_cols] = \
            coords[coord_cols] - offset_row[offset_cols].to_numpy()

    # check whether coordinates fall outside of new the image
    offset_cond = (coords[coord_cols] >= 0).apply(
        np.logical_and.reduce, axis=1)
    shapes = offset_row[shape_cols].to_numpy()
    shape_cond = (coords[coord_cols] < shapes).apply(
        np.logical_and.reduce, axis=1)           
    coords[in_col] = offset_cond & shape_cond

    # clean if needed
    if clean:
        coords = coords[coords[in_col]]

    return coords

def get_contacts_one(
        identifier, region_id, scene=None, segments=None, regions=None,
        ids=None, scalar=None, indexed=None,
        coord_cols=['x_contact', 'y_contact', 'z_contact']):
    """Returns cm coordinates of segment contacts to a region for one tomo.

    If arg scene is given, args segments, regions and ids are ignored and 
    the corresponding values are determined from scene.labels, 
    scene.boundary and scene.labels.ids, respectively.

    Arguments scalar and indexed are ignored. They need to be in the signature
    for this function to be used in __call__().
      
    Arguments:
      - identifier: tomo identifier
      - region_id: id of the region to which segment contact are made
      - scene: for example, instance of pyto.scene.SegmentationAnalysis
      - segments: image containing segments
      - regions: image containing the region
      - ids: ids of segment whose contacts are to be determined
      - scalar: (pandas.DataFrame) scalar properties
      - indexed: (pandas.DataFrame) indexed properties
      - coord_cols: names of the resulting coordinate coulmns

    Returns: dataframe:
      - columns: 'identifiers' (tomo identifiers),
        'ids' (segment ids) and coordinates
      - rows correspond to segments
    """

    # extract info from scene and cehck args
    if scene is not None:
        segments = scene.labels
        ids = scene.labels.ids
        regions = scene.boundary
    if ((scene is None)
        and ((segments is None) or (ids is None) or (regions is None))):
        raise ValueError(
            "Arg scene, or args segments, ids and regions need to "
            + "be specified.")
    
    # get absolute coords in old bin
    hood = Neighborhood(segments=segments, ids=ids)
    cm, _ = hood.find_contact_cm(
        regions=regions, region_id=region_id, frame='abs') 

    # make table
    coords = pd.DataFrame(
        {'identifiers': identifier, 'ids': ids, coord_cols[0]: cm[:, 0],
         coord_cols[1]: cm[:, 1], coord_cols[2]: cm[:, 2]})

    return coords

def get_vesicle_centers_one(
        sv_ids, scalar, sv_var='sv_file', convert_path_common=None,
        convert_path_helper=None):
    """Get vesicle center coordinates for one tomogram.

    Finds a path to a single tomo vesicles object (scalar[sv_var]) reads 
    this object and extracts attribute mor.center. This attribute should 
    contain vesicle centers (vesicle ids indexed array).

    A beginning part of the path is converted using args 
    convert_path_common and convert_path_helper 
    (see pyto.particles.SetPath() and pyto.particles.set_path.convert().

    Arguments:
      - scalar: (pandas.DataFrame) Scalar data object that contains data 
      for one tomogram only and has attribute sv_var (such as 
      work.near_sv.scalar_data or work.tether.scalar_data)
      - sv_ids: vesicle ids
      - convert_path_common, convert_path_helper: (defaults None) paths
      used to change a beginning part of pickle file paths (see 
      tomo_generator doc)
 
   Returns (pandas.DataFrame) Table containing vesicle ids and center 
    coordinates, with column names ['sv_id', 'center_x', 'center_y', 
    'center_z'].
    """

    # get centers
    sv_pkl_path = scalar[sv_var]
    sp = SetPath(common=convert_path_common, helper_path=convert_path_helper)
    sv_pkl_path = sp.convert_path(path=sv_pkl_path)
    sv_one = pickle.load(open(sv_pkl_path, 'rb'), encoding='latin1')
    centers = sv_one.mor.center[sv_ids]

    # put in a dataframe
    id_center = np.concatenate([sv_ids[:, np.newaxis], centers], axis=1)
    df = pd.DataFrame(
        id_center, columns=['sv_id', 'center_x', 'center_y', 'center_z'])

    return df

