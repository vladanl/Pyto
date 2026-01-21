"""
Extract particles from pyto.spatial.MultiParticleSets particle sets.

# Author: Vladan Lucic 
# $Id$
"""

__version__ = "$Revision$"

import os
import pickle
from copy import copy, deepcopy
import re
import itertools

import numpy as np
from numpy.random import default_rng
import scipy as sp
from scipy.spatial.distance import cdist
import pandas as pd
import skimage
 
import pyto
from ..geometry.rigid_3d import Rigid3D
from .set_path import SetPath
from pyto.particles.relion_tools import get_array_data, write_table
from pyto.projects.presynaptic import Presynaptic, tomo_generator
from .extract_mps_filter import ExtractMPSFilter
from ..spatial.multi_particle_sets import MultiParticleSets


class ExtractMPS(ExtractMPSFilter):
    """Extract particles from pyto.spatial.MultiParticleSets particle sets.

    Filtering methods are implemented in ExtractMPSFilter
    """

    def __init__(
            self, 
            distance_col='distance', normal_source_index_col='source_index', 
            normal_source_suffix='_source', normal_source_coord_cols=None,
            normal_angle_cols=['normal_theta', 'normal_phi'],
            degree=True, use_priors=False, centers_dtype=int,
            tomo_col='tomo', path_label='rlnMicrographName',
            path_label_morse='psSegImage',
            tomo_id_mode='munc13', tomo_id_func=None, tomo_id_kwargs={},
            id_source_label=None, ctf_label='rlnCtfImage', check_ctf=False,
            tomo_ids=None, box_size=None,
            label_format=None,            
            region_bin=None, region_bin_factor=None, remove_region_initial=True,
            init_coord_cols=None,
            center_init_frame_cols=None, center_reg_frame_cols=None, 
            tomo_l_corner_cols=None, tomo_r_corner_cols=None,
            reg_l_corner_cols=None, reg_r_corner_cols=None,
            tomo_inside_col='tomo_inside', reg_inside_col='region_inside',   
            tomo_particle_col='tomo_particle',
            region_particle_col='reg_particle',
            in_tomo_particle_col='in_tomo_particle',
            root_template='particles_size-{size}', tables_dir='tables', 
            class_names=[], class_code={},
            rng=None, seed=None):
        """Sets attributes from arguments.
        """

        self.distance_col = distance_col
        self.normal_source_index_col = normal_source_index_col
        self.normal_source_suffix = normal_source_suffix
        self.normal_source_coord_cols = normal_source_coord_cols
        self.normal_angle_cols = normal_angle_cols

        self.degree = degree
        self.use_priors = use_priors
        self.centers_dtype = centers_dtype

        self.tomo_col = tomo_col

        self.tomo_id_mode = tomo_id_mode
        self.tomo_id_func = tomo_id_func
        self.tomo_id_kwargs = tomo_id_kwargs
        self.path_label = path_label
        self.path_label_morse = path_label_morse
        self.id_source_label = id_source_label
        self.init_coord_cols = init_coord_cols
        self.ctf_label = ctf_label
        self.check_ctf = check_ctf

        self.tomo_ids = tomo_ids

        self.box_size = box_size

        self.tomo_l_corner_cols = tomo_l_corner_cols
        self.tomo_r_corner_cols = tomo_r_corner_cols
        self.reg_l_corner_cols = reg_l_corner_cols
        self.reg_r_corner_cols = reg_r_corner_cols
        self.tomo_inside_col = tomo_inside_col
        self.reg_inside_col = reg_inside_col
        
        self.remove_region_initial = remove_region_initial
        self.region_bin = region_bin
        self.region_bin_factor = region_bin_factor
        self.center_init_frame_cols = center_init_frame_cols
        self.center_reg_frame_cols = center_reg_frame_cols
 
        
        if label_format is None:
            self.label_format ={
                'rlnMicrographName': '%s', 'rlnCtfImage': '%s', 
                'rlnImageName': '%s', 'rlnCoordinateX': '%d', 
                'rlnCoordinateY': '%d', 
                'rlnCoordinateZ': '%d', 
                'rlnAngleTilt': '%8.3f', 'rlnAngleTiltPrior': '%8.3f', 
                'rlnAnglePsi': '%8.3f', 'rlnAnglePsiPrior': '%8.3f', 
                'rlnAngleRot': '%8.3f'}
        else:
            self.label_format = label_format

        self.tomo_particle_col = tomo_particle_col
        self.region_particle_col = region_particle_col
        self.in_tomo_particle_col = in_tomo_particle_col
        
        self.root_template = root_template
        self.tables_dir = tables_dir
        
        self.class_names = class_names
        self.class_code = class_code

        self.rng = rng
        self.seed = seed

    def extract_particles_task(
            self,  name, particle_to_center, reverse,
            input_mode='tethers_2024', mps_path=None, use_priors=None, 
            particles_path=None, source_path=None,
            tomo_star_path=None, tomo_bin=None, ctf_star_path=None,
            regions_star_path=None, clean_initial=True,
            randomize_rot=True, expand_particle=False, expand_region=True,
            mean=0, std=1, invert_contrast=True,
            name_prefix='particle_', name_suffix='',
            convert_path_common=None, convert_path_helper=None,     
            write_particles=True, write_regions=False, morse_regions=False,
            verbose=True, star_comment='Particles'
            ):
        """Extracts particles from tomos.

        Input data, comprising info about tomograms and particles, can be 
        specified in different ways depending on arg input_mode:
          - 'mps': from a MultiParticleSets object specified by arg
          mps_path
          - 'tethers_2024': (experts only) used for tether averaging 2024
          project because input data was already present for a related 
          pyseg-morse project. In this case, args particles_path, source_path, 
          regions_star_path, ctf_star_path and clean_initial
          have to be specified.

        For input_mode='mps', the MPS object has to contain tomos
        and particles attributes (pandas.DataFrame) as follows:
          - tomos (DataFrame) columns: 
            - 'tomo_id': tomogram identifier, has to be unique 
            - 'tomo': path to (greyscale) tomogram, defines tomo frame 
              (bin factor given in region_bin column) 
            - 'region': path to boundaries (regions), defines region frame
              (bin_factor given in region_bin column)
            - 'rlnCtfImage': path to ctf corresponding to tomo
            - region_bin: bin factor of tomo and region images
            - 'region_offset_x/y/z': position of regions with respect 
              to tomo (at region_bin)
            - 'pixel_nm': pixel size [nm] 
            - 'coord_bin': bin factor for x/y/z_orig coordinates in
              particles table (needed if particles are specified in tomo
              frame, that is if columns 'x/y/z_orig' are used)
            - 'region_id': needed?
          - particles (DataFrames) columns:
            - 'group': experimental group 
            - 'tomo_id': tomogram identifier, same as in tomos
            - 'particle_id': particle id, unique within a tomogram
            - 'class_number', 'class_name': particle class number and name
            (optional)
            - 'x/y/z_orig' (self.orig_coord_cols): particle coordinates 
            in tomo frame at region_bin (needed if 'x/y/z_orig_reg_frame' 
            columns are not present, otherwise ignorred)
            - 'x/y/z_orig_reg_frame' (self.orig_coord_reg_frame_cols): 
            particle coordinates in regions frame at region_bin 
            - 'rlnAngleTilt', 'rlnAngleTiltPrior', 'rlnAnglePsi', 
            'rlnAnglePsiPrior', 'rlnAngleRot': particle angles as given
            by relion (active, intrinsic, zyz, Euler angles)

        The following coordinate systems (frames) are used:
          - coordinate frame: full size tomos, bin factor in coord_bin
            column of mps.tomos dataframe
          - tomo frame: full size tomos, bin factor in tomo_bin column 
            of mps.tomos dataframe, added by this method
          - regions frame: cropped from full size so that origin is at 
            region_offset_x/y/z columns of mps.tomos, bin factor in 
            region_bin column of mps.tomos dataframe

        Particle coordinates (picks) are converted from tomo or regions 
        frames to particle corners in tomo frame, from which particle 
        images are extracted, by the following transformations:
          - (mps mode only): Input particle coords are specified in 
          regions frame (columns orig_coord_reg_frame_cols) or in
          tomo frame (columns orig_coord_cols). In the latter case, 
          coordinates are converted to regions frame.
          - (tethers_2024 mode only): Input particle coordinates are given 
          in regions frame (before projection on thin region). These are 
          specified in the MPS objects read from (arg) particle_path 
          (called mps_init). The coordinates are read from 
          mps_init.particles (Pandas) table, columns specified by 
          mps_init.orig_coord_reg_frame_cols
          - Determines particle center coordinates in regions frame 
          as a displacement from particle position by (arg) particle_to_center
          along the membrane normal in the cytoplasmic direction. The 
          membrane normal direction is determined from the angles
          specified in the MPS object read from (arg) source_path.
          - Particle image centers in the regions frame are converted to 
          the tomo frame
          - Particle image corners are determined from the particle 
          image centers. These are saved in the resulting object 
          (called mps), table mps.particles, columns mps.tomo_l_corner_cols 
          (left corner) and mps.tomo_r_corner_cols (right corner).

        Generates relion particle star file that contains:
          - rlnMicrographName, rlnCtfImage, rlnImageName
          - rlnCoordinateX/Y/Z: particle coordinates
          - rlnAngleTilt, rlnAnglePsi: particle angles from membrane normals
          - rlnAngleTiltPrior, rlnAnglePsiPrior: the same as rlnAngleTilt, 
            rlnAnglePsi
          - rlnAngleRot: randomized particle angle

        Output:
          - particle images
          - final (MPS) object
          - particle star file in Relion format

        In the current workflow (2.2024), arguments write_regions and 
        morse_regions should be both False because region images are
        generated by self.extract_regions_task().

        """

        # should be passed in consructor, left for back compatibility
        if use_priors is not None:
            self.use_priors = use_priors
        
        if input_mode == 'tethers_2024':
            # make mps from tethers averaging 2024 project files
            mps = self.input_from_tethers_2024(
                name=name, particles_path=particles_path,
                source_path=source_path, regions_star_path=regions_star_path,
                ctf_star_path=ctf_star_path, reverse=reverse,
                clean_initial=clean_initial,
                morse_regions=morse_regions, verbose=verbose)

        elif input_mode == 'mps':
            # read data from previously prepared mps and set angles
            mps = MultiParticleSets.read(mps_path, verbose=verbose)
            mps.tomo_col = self.tomo_col
            if (len(np.intersect1d(
                    mps.particles.columns.to_numpy(),
                    mps.orig_coord_reg_frame_cols)) == 0):
                mps.convert_frame(
                    init_coord_cols=mps.orig_coord_cols,
                    final_coord_cols=mps.orig_coord_reg_frame_cols,
                    shift_final_cols=mps.region_offset_cols,
                    init_bin_col=mps.coord_bin_col,
                    final_bin_col=mps.region_bin_col, overwrite=False)

            # set normals
            if source_path is not None:
                if isinstance(source_path, str):
                    source = MultiParticleSets.read(
                        source_path, verbose=verbose)
                else:
                    source = source_path
            else:
                source = None
            mps.particles = self.set_normals(
                mps=mps, mps_coord_cols=mps.orig_coord_reg_frame_cols, 
                source=source, source_coord_cols=self.normal_source_coord_cols,
                reverse=reverse, use_priors=self.use_priors)

            # set tomo paths (removes previos paths if they exist)
            if tomo_star_path is not None:
                mps.tomo_col = self.tomo_col
                self.add_paths(
                    mps=mps, star_path=tomo_star_path, mode='tomos',
                    path_col=mps.tomo_col, tomo_bin=tomo_bin, update=True)

            # set ctf paths
            if ctf_star_path is not None:
                self.add_ctf(
                    mps=mps, star_path=ctf_star_path, update=True,
                    check=self.check_ctf)
                
        else:
            raise ValueError(
                f"Argument input mode ({input_mode}) was not recognized."
                + "Currently defined values are 'mps' and 'tethers_2024'.") 
            
        # randomize rot angle
        if randomize_rot:
            mps.particles['rlnAngleRot'] = 360 * np.random.rand(
                mps.particles.shape[0])

        # determine centers in reg frame
        mps.center_reg_frame_cols = self.center_reg_frame_cols
        self.project_along_normals(
            mps=mps, coord_cols=mps.orig_coord_reg_frame_cols, 
            center_coord_cols=mps.center_reg_frame_cols,
            distance=particle_to_center, update=True)

        # convert centers back to init frame
        # ToDo convert for different bins (use convert_frame()
        mps.center_init_frame_cols = self.center_init_frame_cols
        if mps.tomo_bin_col not in mps.tomos.columns:
            # for backward compatibility 
            self.convert_back(
                mps=mps, init_cols=mps.center_reg_frame_cols, 
                final_cols=mps.center_init_frame_cols, update=True)
        else:
            mps.convert_frame_inverse(
                init_coord_cols=mps.center_reg_frame_cols,
                final_coord_cols=mps.center_init_frame_cols,
                shift_cols=mps.region_offset_cols,
                init_bin_col=mps.region_bin_col, final_bin_col=mps.tomo_bin_col)

        # convert paths
        self.convert_paths(
            mps=mps, 
            common=convert_path_common, helper_path=convert_path_helper, 
            path_cols=[self.ctf_label], tomo_path_col=mps.tomo_col, 
            region_path_col=None, update=True)
        
        # find corners and label particles that fit inside tomo and
        # segmentation images
        mps.tomo_l_corner_cols = self.tomo_l_corner_cols
        mps.tomo_r_corner_cols = self.tomo_r_corner_cols
        mps.tomo_inside_col = self.tomo_inside_col
        mps.reg_inside_col = self.reg_inside_col
        mps.reg_l_corner_cols = self.reg_l_corner_cols
        mps.reg_r_corner_cols = self.reg_r_corner_cols
        self.find_corners(
            mps=mps, image_path_col=mps.tomo_col, box_size=self.box_size, 
            coord_cols=self.center_init_frame_cols,
            l_corner_cols=self.tomo_l_corner_cols, 
            r_corner_cols=self.tomo_r_corner_cols,
            column=self.tomo_inside_col, update=True)
        if morse_regions:
            self.find_corners(
                mps=mps, image_path_col=mps.region_col, box_size=self.box_size, 
                coord_cols=self.center_reg_frame_cols,
                l_corner_cols=self.reg_l_corner_cols, 
                r_corner_cols=self.reg_r_corner_cols,
                column=mps.reg_inside_col, update=True)

        # setup paths
        paths = Paths(
            name=name, root_template=self.root_template, size=self.box_size,
            tables=self.tables_dir)

        # extract particles
        mps.tomo_particle_col = self.tomo_particle_col
        self.write_particles(
            mps=mps, l_corner_cols=self.tomo_l_corner_cols,
            r_corner_cols=self.tomo_r_corner_cols, 
            image_path_col=mps.tomo_col, dir_=paths.particles_dir,
            expand=expand_particle, select_col=mps.tomo_inside_col,
            mean=mean, std=std, invert_contrast=invert_contrast,
            name_prefix=name_prefix, name_suffix=name_suffix,
            particle_path_col=mps.tomo_particle_col,
            convert_path_common=convert_path_common, 
            convert_path_helper=convert_path_helper,
            update=True, write=write_particles)
        if verbose:
            print(f"\nAll particles:")
            if write_particles:
                print(f"Wrote particles to {paths.particles_dir}")
            else:
                print(f"Particles were not written to {paths.particles_dir}")
                
        # extract segments
        if morse_regions:
            mps.region_particle_col = self.region_particle_col
            self.write_particles(
                mps=mps, l_corner_cols=self.reg_l_corner_cols,
                r_corner_cols=self.reg_r_corner_cols, 
                image_path_col=mps.region_col, dir_=paths.regions_dir,
                expand=expand_region, select_col=mps.tomo_inside_col,
                mean=None, std=None,
                name_prefix='seg_', name_suffix='',
                particle_path_col=mps.region_particle_col,
                convert_path_common=convert_path_common, 
                convert_path_helper=convert_path_helper,
                update=True, write=write_regions)

        # save particle mps
        mps.write(
            path=paths.mps_path_tmp, verbose=verbose, out_desc="preliminary")

        # write star files and the corresponding table
        # extract labels from actual tables 
        labels = self.get_labels(mps=mps, use_priors=self.use_priors)
        self.make_star(
            mps=mps, labels=labels, star_path=paths.star_path,
            verbose=verbose, comment=star_comment, out_desc="all particles")

        # make star file for each particle subclass
        if ((len(self.class_code) > 0)
            and (self.class_names is None or (len(self.class_names) > 0))
            and (mps.class_number_col in mps.particles.columns)
            and (mps.class_name_col in mps.particles.columns)):
            if verbose:
                print("\nParticle classes:")
            self.split_star(
                mps=mps,
                class_names=self.class_names, class_code=self.class_code,
                labels=self.get_labels(mps, use_priors=self.use_priors), 
                star_path=paths.star_path, star_comment=star_comment,
                verbose=verbose, out_desc="particles")
        
    def extract_regions_task(
            self, mps, input_mode='presynaptic', scalar=None, indexed=None,
            struct_path_col=None, region_path_mode=None, 
            convert_path_common=None, convert_path_helper=None,
            path_col=None, offset_cols=None, shape_cols=None, bin_col=None,
            expand=True, normalize_kwargs={}, smooth_kwargs={}, dilate=None,
            out_dtype=None, fun=None, fun_kwargs={}, write_regions=True,
            regions_name='regions', name_prefix='seg_', name_suffix='',
            mps_path=None, star_comment='Regions', verbose=True):
        """Extracts boundary (regions) or segment particle images (subtomos).

        Boundary and segment particles can be extracted from boundary and
        segment tomos, respectively, of from structure pickles obtained 
        by presynaptic analysis, depending on arg input_mode.

        Boundary and segments input data is specified depending on 
        arg input_mode:
          - 'presynaptic': by pyto presynaptic analysis, where 
          structure pickles are obtained by pyto hiererchical
          connectivity and saved as SegmentationAnalysis objects.
          - 'mps': (experts only) from a MultiParticleSets object passed
          directly or specified by a path (arg mps in both cases). In this 
          case, regions tomos (column 'regions' of mps.tomos have to 
          contain all boundaries, and their ids need to be consistent 
          with value of arg normalize_kwargs. 

        Region coordinates are calculated as:
          - Start with center_init_frames coords calculated by 
          extract_particles_task() (meant to be full size tomo bin)
          - Convert to centers in the boundary region of structure pickles 
          frame:
            bin_factor * x - boundary_offset(inset)
          - Particle box corners are determined from the above centers
          - Particle images are extracted from the boundary regions of 
          structure pickles, the size depends on the bin_factor
          - Particle images are magnified by 1/bin_factor

        Saves the modified MultiParticleSets at the (standard) location
        specified by self.root_template and arg regions_name. In addition, 
        if arg mps_path is specified, the same file is also saved at the
        specified location. The later is meant to write the pickle in the
        original particles tables dir.

        Particle images can be modified using args fun,
        fun_kwargs and normalize_kwargs. Specifically, if arg fun is 
        not specified, the following transformations are applied using
        self.prepare_func() (see this function for more info):
          - zoom if self.region_bin_factor != 1
          - boundary ids are normalized using arg normalize_kwargs
          - morphologically smooth boundaries
          - dilation if arg dilate is specified and != 0
          - image dtype is changed if arg dtype is given 
        In this case arg fun_kwargs is ignored.

        Alternatively, if arg fun is specified, it is applied on all
        individual particle images. If fun is a single function, it
        is applied as:
            fun(particle_data, **fun_kwargs)
        where particle data is ndarray of image pixel values, and it has
        to return (a modified) image ndarray. 

        If fun is a list (or tuple) of functions, each of them is 
        applied in the order specified. In this case, the outpur of the
        preceeding function is passed as the first argument. Arg 
        fun_kwargs has to be a list where each element is a dict 
        containing the remaining arguments of the respective function. 
        All funcrions have to have the signature described above. 
        In this case, args normalize_kwargs, dilate and dtype ae ignored.

        For example, to apply a normalization, magnification and another
        function, set args:
          - fun=(normalize_fun, mag_fun, other_fun)
          - fun_kwargs=(normalize_fun_kwargs, mag_fun_kwargs, other_fun_kwargs)

        In 'pkl_segment' mode (arg region_path_mode), all other segments
        that may be present in a particle image are removed before
        applying functions specified by arg fun.

        Arguments:
          - mps: (MultiParticleSets) particles object, or (str) path
          to saved particle object

          - write_regions: flag indicating if region images are written
          - regions_name: name given to the extracted regions (boundaries), 
          used as the directory name where region images are saved, 
          if None 'regions' is used
          - name_prefix: part of the region image name before the 
          particle id (default 'seg_')
          - name_suffix: part of the region image name after the 
          particle id and before extension (default '')

          - mps_path: if specified, the resulting MultiParticleSets pickle
          is saved at this path (in addition to saving it at the  
          standard path

        """

        # read mps if parh specified
        if isinstance(mps, str):
             mps = MultiParticleSets.read(mps, verbose=verbose)
        
        # setup paths
        paths = Paths(
            name=regions_name, root_template=self.root_template,
            regions=regions_name, size=self.box_size, tables=self.tables_dir)
            
        # figure out region_bin and region_bin_factor
        self.find_bins(mps=mps)
        
        # get input data from other sources if needed
        if input_mode == 'presynaptic':
            mps = self.input_from_presynaptic(
                mps=mps, scalar=scalar, indexed=indexed,
                struct_path_col=struct_path_col,
                region_path_mode=region_path_mode,
                convert_path_common=convert_path_common,
                convert_path_helper=convert_path_helper,
                path_col=path_col, offset_cols=offset_cols,
                shape_cols=shape_cols, bin_col=bin_col)
            shape_cols = mps.region_shape_cols
            
        elif input_mode == 'mps':
            # convert paths
            self.convert_paths(
                mps=mps, 
                common=convert_path_common, helper_path=convert_path_helper, 
                path_cols=[self.ctf_label], tomo_path_col=None, 
                region_path_col=path_col, update=True)
        else:
            raise ValueError(
                f"Argument input mode ({input_mode}) was not recognized."
                + "Currently defined values are 'mps' and 'presynaptic'.") 
         
        # set attributes to mps
        mps.tomo_l_corner_cols = self.tomo_l_corner_cols
        mps.tomo_r_corner_cols = self.tomo_r_corner_cols
        mps.reg_l_corner_cols = self.reg_l_corner_cols
        mps.reg_r_corner_cols = self.reg_r_corner_cols
        mps.tomo_inside_col = self.tomo_inside_col
        mps.reg_inside_col = self.reg_inside_col
        mps.tomo_particle_col = self.tomo_particle_col
        mps.region_particle_col = self.region_particle_col 

        # find corner coords in regions frame
        particle_size_loc = self.box_size // self.region_bin_factor
        self.find_corners(
            mps=mps, image_path_col=mps.region_col, box_size=particle_size_loc,
            coord_cols=self.center_reg_frame_cols,
            l_corner_cols=self.reg_l_corner_cols,
            r_corner_cols=self.reg_r_corner_cols, 
            shape_cols=shape_cols, column=self.reg_inside_col,
            update=True)

        # prepare image processing functions (magnify, normalize, dilate)
        if fun is None:
            fun, fun_kwargs = self.prepare_func(
                zoom_factor=self.region_bin_factor,
                smooth_kwargs=smooth_kwargs,
                normalize_kwargs=normalize_kwargs, dilate=dilate,
                dtype=out_dtype)            
            #fun = (normalize_bound_fun, mag_fun)
            #fun_kwargs = (normalize_bound_fun_kwargs, mag_fun_kwargs)
        else:
            if ((len(normalize_kwargs) > 0) or (len(smooth_kwargs) > 0)
                or (dilate is not None) or (out_dtype is not None)):
                print("Warning: Because argument fun is specified, arguments "
                      + f"normalize_kwargs ({normalize_kwargs}), "
                      + f"dilate ({dilate}) "
                      + f"and dtype ({out_dtype}) are ignored. ")

        # write images
        self.write_particles(
            mps=mps, l_corner_cols=self.reg_l_corner_cols,
            r_corner_cols=self.reg_r_corner_cols, 
            image_path_col=mps.region_col, image_path_mode=region_path_mode,
            dir_=paths.regions_dir, 
            expand=expand, select_col=mps.tomo_inside_col,
            mean=None, std=None, fun=fun, fun_kwargs=fun_kwargs,
            name_prefix=name_prefix, name_suffix=name_suffix,
            particle_path_col=mps.region_particle_col,
            convert_path_common=convert_path_common, 
            convert_path_helper=convert_path_helper,
            write=write_regions, update=True)
        if verbose:
            print(f"All {regions_name}:")
            print(f"Wrote {regions_name} ({region_path_mode}) "
                  + f"to {paths.regions_dir}")
        
        # save mps locally 
        mps.write(path=paths.mps_path, verbose=verbose, out_desc="preliminary")

        # save the same mps also in another place if mps_path is given 
        if mps_path is not None:
            mps.write(path=mps_path, verbose=verbose, out_desc="preliminary")

        #  make star file
        labels = self.get_labels(mps=mps, use_priors=self.use_priors)
        self.make_star(
            mps=mps, labels=labels, star_path=paths.star_path,
            comment=star_comment, verbose=verbose,
            out_desc=f"all {regions_name}")
        
        # make star file for each particle subclass
        if ((len(self.class_code) > 0)
            and (self.class_names is None or (len(self.class_names) > 0))
            and (mps.class_number_col in mps.particles.columns)
            and (mps.class_name_col in mps.particles.columns)):
            if verbose:
                print(f"\nIndividual classes of {regions_name}:")
            self.split_star(
                mps=mps,
                class_names=self.class_names, class_code=self.class_code,
                labels=self.get_labels(mps, use_priors=self.use_priors), 
                star_path=paths.star_path, star_comment=star_comment,
                verbose=verbose, out_desc=f"{regions_name}")

    def clean_particles_task(
            self, processing_cases, preliminary_tables_dir,
            expected_regions, expected_segments=None, found_col='found_ids_col',
            star_comment='', verbose=True):
        """Removes particles based on regions and segments.

        Reads 

        Arguments:
          - processing_cases: (list) processing cases to be cleaned
          - preliminary_tables_dir: directory 
        """

        # read regions mps
        paths = Paths(
            name='regions', root_template=self.root_template,
            size=self.box_size, tables=preliminary_tables_dir)
        mps_regions = MultiParticleSets.read(
            path=paths.mps_path, verbose=verbose, out_desc="regions")
        mps_regions.check_ids(
            expected=expected_regions, update=True, found_col=found_col,
            verbose=False)
        keep = mps_regions.particles[found_col]

        # read segments mps
        if expected_segments is not None:
            paths = Paths(
                name='segments', root_template=self.root_template,
                size=self.box_size, tables=preliminary_tables_dir)
            try:
                mps_segments = MultiParticleSets.read(
                    path=paths.mps_path, verbose=verbose, out_desc="segments")
                mps_segments.check_ids(
                    expected=expected_segments, update=True,
                    found_col=found_col, verbose=False)
                keep = (keep & mps_segments.particles[found_col])
            except FileNotFoundError:
                print(f"MPS pickle at {paths.mps_path} was not found")
                
        # loop over processing cases
        for nam in processing_cases:

            # read mps
            in_paths = Paths(
                name=nam, root_template=self.root_template,
                size=self.box_size, tables=preliminary_tables_dir)
            try:
                mps = MultiParticleSets.read(
                    path=in_paths.mps_path, verbose=verbose,
                    out_desc="particles")
                if verbose:
                    print(f"\nProcessing {nam}: ")
            except FileNotFoundError:
                if verbose:
                    print(f"Processing case {nam} does not exist.")
                continue
                
            # add found column and make clean mps
            if found_col in mps.particles.columns:
                mps.particles[found_col] = mps.particles[found_col] & keep
            else:
                mps.particles[found_col] = keep
            mps_clean = deepcopy(mps)
            mps_clean.tomos = mps.tomos
            mps_clean.particles = mps.particles[keep]

            # save particle mps (all particles but with a flag)
            out_paths = Paths(
                name=nam, root_template=self.root_template,
                size=self.box_size, 
                tables=self.tables_dir)
            mps.write(path=out_paths.mps_path, verbose=verbose)
    
            # write star files and the corresponding table
            labels = self.get_labels(mps=mps_clean, use_priors=self.use_priors)
            combined = self.make_star(
                mps=mps_clean, labels=labels, star_path=out_paths.star_path,
                verbose=verbose, comment=star_comment)

            # make star file for each particle subclass
            self.split_star(
                mps=mps_clean, labels=labels, 
                class_names=self.class_names, class_code=self.class_code,
                star_path=out_paths.star_path, star_comment=star_comment,
                verbose=verbose)
        
    def input_from_tethers_2024(
            self, name, particles_path, source_path,
            regions_star_path, ctf_star_path, reverse, 
            clean_initial=True, morse_regions=False, verbose=True):
        """Generates input MPS data from tether averaging 2024 project.

        Used in self.extract_particles_task().
        """

        # Read MPS particles and source where particles are already
        #  converted to the region image frames and clean them
        mps_init = MultiParticleSets.read(particles_path, verbose=verbose)
        if clean_initial:
            mps_init.particles = mps_init.particles[
                mps_init.particles[mps_init.keep_col]]
        source = MultiParticleSets.read(source_path, verbose=verbose)

        # set normals
        mps_part = self.set_normals(
            mps=mps_init, source=source,
            mps_coord_cols=mps_init.orig_coord_reg_frame_cols, 
            source_coord_cols=source.orig_coord_reg_frame_cols,
            reverse=reverse, use_priors=self.use_priors)
        
        # setup particle mps
        mps = deepcopy(mps_init)
        mps.tomos = mps_init.tomos
        mps.particles = mps_part
        mps.name = name
        # assumes thin region and region images have the same positioning

        # find tomo and segmentation image paths
        mps.tomo_col = self.tomo_col
        self.add_paths(
            mps=mps, star_path=regions_star_path, mode='tomos',
            path_col=mps.tomo_col, update=True)
        if morse_regions:
            self.add_paths(
                mps=mps, star_path=regions_star_path, mode='tomos',
                path_label=self.path_label_morse,
                path_col=mps.tomo_col, update=True)

        # add ctf paths
        self.add_ctf(
            mps=mps, star_path=ctf_star_path, update=True,
            check=self.check_ctf)

        return mps

    def input_from_presynaptic(
            self, mps, scalar, indexed, struct_path_col,
            region_path_mode,
            convert_path_common=None, convert_path_helper=None,
            path_col=None, offset_cols=None, shape_cols=None, bin_col=None):
        """Extracts input data from pyto presynaptic analysis.

        Used in self.extract_regions_task().
        """

        # remove region related columns    
        if self.remove_region_initial:
            self.remove_region_cols(mps=mps) 

         # convert coords to regions
        mps = self.convert_to_struct_region(  
            mps=mps, scalar=scalar, indexed=indexed,
            struct_path_col=struct_path_col, image_path_mode=region_path_mode,
            init_coord_cols=self.init_coord_cols,
            region_coord_cols=self.center_reg_frame_cols, 
            convert_path_common=convert_path_common,
            convert_path_helper=convert_path_helper,
            region_bin=self.region_bin, path_col=path_col, 
            offset_cols=offset_cols, shape_cols=shape_cols, bin_col=bin_col)

        return mps
    
    def set_normals(
            self, mps, mps_coord_cols, source=None, source_coord_cols=None,
            reverse=False, use_priors=True):
        """Find membrane normals for each particle.

        If arg source is specified, it is used to determine the membrane
        normals of mps.particles in the following way. The closest particle
        from (arg) source particle set is determined for each particle
        specified in the (arg) mps particle set, and the source particle
        angles are assigned to their corresponding mps particles.

        Values of the following columns are copied from the closest elements of
        source.particles: 
          - source.particle_rotation_labels, reversed if arg reverse=True 
          (see below)
          - source.particle_id_col, 
          - source.class_name_col, 
          - source.class_number_col: 
        The column names stay the same. In case source.particles colum names 
        overlap with those of the mps.particles, suffix 
        self.normal_source_suffix is added. 

        In addition the following columns to mps.particles are introduced:
          - distance_col: distance to the closest element of source.particles
          - source_index_col: index of the closest element of source.particles
          - normal_angle_cols: values of normal angles theta and phi

        If arg source is None, particle Euler angles have to be given in 
        mps.particles.

        The following applies in both cases.

        If arg reverse is True, Euler angles (columns 
        mps.particle_rotation_labels) are changed so that they define 
        the opposite direction:
            - phi, theta, psi -> phi + pi, pi - theta, psi + pi
        In this case, normal vector angles are determined from the reversed 
        Eulers 

        Table mps.particles or source.particles have to contain at least 
        one set of Euler angles, that is 
          ('rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi'), or 
          ('rlnAngleRot', 'rlnAngleTiltPrior', 'rlnAnglePsiPrior')

        Arguments:
          - mps: (MultiParticleSets) Tomos and particles dataframes
          - mps_coord_cols: columns of mps.particles that contain particle
          coordinates
          - source: (MuliParticleSets) Another particle set, has to contain
          coordinates and angles
          - source_coord_cols: columns of source.particles wrom which
          angles of mps are determined, has to contain particle 
          coordinates of the other set, or None if mps contains angles
          - reverse: (bool) if True, inverse angles
          - use_priors: (bool) if True, use prior angles

        Returns mps.particles with the added columns
        """

        if source is not None:
        
            # find closest pre to all tethers
            min_dist = mps.find_min_distances(
                df_1=mps.particles, df_2=source.particles,
                group_col=mps.tomo_id_col,
                coord_cols_1=mps_coord_cols, coord_cols_2=source_coord_cols, 
                distance_col=self.distance_col,
                ind_col_2=self.normal_source_index_col)

            # add the closest pre to tethers table
            part_1 = mps.particles.join(min_dist[
                [self.normal_source_index_col, self.distance_col]])

            # add the corresponding angles from source to mps (particles)
            source_cols_possible = [
                source.particle_id_col, source.class_name_col,
                source.class_number_col]
            #source_cols = (
            #    source.particle_rotation_labels 
            #    + [col_nam for col_nam in source_cols_possible
            #       if col_nam in source.particles.columns])
            source_cols = [
                col_nam for col_nam
                in source.particle_rotation_labels + source_cols_possible
                if col_nam in source.particles.columns]
            part_2 = pd.merge(  # index from df_2 particles
                part_1, source.particles[source_cols], how='left', 
                left_on=self.normal_source_index_col, right_index=True,
                sort=False, suffixes=['', self.normal_source_suffix])

        else:
            part_2 = mps.particles.copy()
            
        # reverse Eulers if needed
        if reverse:
            angle_cols = [
                'rlnAngleRot', 'rlnAngleTiltPrior', 'rlnAnglePsiPrior']
            try:                    
                priors = part_2[angle_cols].apply(
                    lambda x: Rigid3D.reverse_euler(
                        angles=x, degree=True),
                    axis=1, raw=True)    
                    
                prior_failed = False
            except KeyError:
                prior_failed = True
            else:
                priors_norm = priors[angle_cols].apply(
                    lambda x: Rigid3D.normalize_euler(
                        angles=x, range='0_2pi', degree=True),
                    axis=1, raw=True)
                part_2.update(priors_norm)

            angle_cols = ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
            try:
                posteriors = part_2[angle_cols].apply(
                    lambda x: Rigid3D.reverse_euler(
                        angles=x, degree=True),
                    axis=1, raw=True)
            except KeyError:
                pass
            else:
                posteriors_norm = posteriors[angle_cols].apply(
                    lambda x: Rigid3D.normalize_euler(
                        angles=x, range='0_2pi', degree=True),
                    axis=1, raw=True)

                if prior_failed:
                    part_2.update(posteriors_norm)
                else:
                    part_2.update(
                        posteriors_norm[['rlnAngleTilt', 'rlnAnglePsi']])

        # select angles to make normals
        if use_priors:
            tilt_name = 'rlnAngleTiltPrior'
            psi_name = 'rlnAnglePsiPrior'
        else:
            tilt_name = 'rlnAngleTilt'
            psi_name = 'rlnAnglePsi'

        # get normal angles and put in table
        normals = part_2[[tilt_name, psi_name]].apply(
            lambda x: self.find_spherical(
                angles=x, relion=True, reverse=False),
             axis=1, raw=True)
        normals = normals.rename(
            columns={tilt_name: self.normal_angle_cols[0],
                     psi_name: self.normal_angle_cols[1]})
        part_2 = part_2.join(normals)

        return part_2

    @classmethod
    def find_spherical(
            cls, angles, relion=False, euler_mode='zxz_ex_active',
            euler_range='0_2pi', degree=False, reverse=False):
        """Wrapper for pyto.spatial.LineProjection.find_spherical.

        Initializes LineProjection, executes find_sperical() method
        and passes the returns. All arguments have the same names.
        """
        line_proj = pyto.spatial.LineProjection(
            relion=relion, euler_mode=euler_mode, degree=degree,
            euler_range=euler_range, reverse=reverse)
        return line_proj.find_spherical(angles=angles)

    @classmethod
    def project_along_line(cls, theta, phi, distance=1, degree=False):
        """Wrapper for pyto.spatial.LineProjection.project_along_line().

        Initializes LineProjection, executes find_sperical() method
        and passes the returns. All arguments have the same names.
         """
        line_proj = pyto.spatial.LineProjection(degree=degree)
        res = line_proj.project_along_line(
            theta=theta, phi=phi, distance=distance)
        return res

    def project_along_normals(
            self, mps, coord_cols, center_coord_cols, distance, 
            update=False):
        """Project particles along membrane normals for multiple tomos. 

        Membrane normal angles theta and phi (spherical coordinates) 
        are read from mps.particles columns self.normal_angle_cols 
        (default 'normal_theta' and 'normal_phi').

        Particles are projected (along membrane normals) from coordinates
        specified by mps.particles, columns (arg) coord_cols
        at distance given by (arg) distance.

        Meant to determine particle image centers coordinates as a 
        fixed displacement from particle coords along membrane normals, 
        when membrane normals are given in the relion format 
        ('rlnAngleTiltPrior' and 'rlnAnglePsiPrior').

        Arguments:
          - mps: (MultiParticleSets) Particles
          - coord_cols: initial coordinates columns of mps.particles
          - center_coord_cols: projected coordinates columns of mps.particles
          - distance: projection distance (pixels)
          - update: flag indicating whether mps is updated

        Sets projected particle coords in table mps.particles, columns 
        (arg) center_coord_cols if arg update is True.

        Returns projected particle coords if arg update is False.
        """

        if (distance is not None) and (distance != 0): 

            # find centers
            centers = mps.particles.apply(
                lambda x: (
                    self.project_along_line(
                        theta=x[self.normal_angle_cols[0]],
                        phi=x[self.normal_angle_cols[1]], 
                        distance=distance, degree=self.degree)
                    + x[coord_cols]), 
                axis=1)
            if self.centers_dtype is not None:
                centers = centers.round().astype(self.centers_dtype)

        else:
            centers = mps.particles[coord_cols]

        # rename center columns
        columns_rename = dict(
            [(old, new) for old, new in zip(coord_cols, center_coord_cols)])
        centers.rename(columns=columns_rename, inplace=True)

        # update or return
        if update:
            mps.particles = mps.particles.join(centers)
        else:
            return centers
        
    def convert_back(self, mps, init_cols, final_cols, update=False):
        """Converts coordinates from region to initial frame.

        Conversion amounts to a simple addition of region frame offsets
        (values of mps.region_offset_cols).

        Therefore, the region and init frames have to differ only by (tomo
        dependent) shifts that are specified by columns mps.region_offset_cols
        of table mps.tomos.
 
        Importantly, the init and final frames have to have the same binning.

        Depreciated: Use MUltiParticleSets.convert_frame_inverse()
        instead because init and final frame bins can be different. 
        
        Arguments:
          - mps: (MultiParticleSets) object containing coordinates and
          offsets
          - init_cols: Names of initial coordinate columns in mps.particles
          - init_cols: Names of final (converted) coordinate columns in 
          mps.particles
          - update: flag indicating if mps.particle table is modified 
          or returned
        """

        column_rename = dict(
            [(old, new) for old, new in zip(init_cols, final_cols)])
        part_by_tomos = mps.particles.groupby(mps.tomo_id_col)
        converted_list = []

        # convert for each tomo separately
        for t_id, ind in part_by_tomos.groups.items():
            tomo_row = mps.tomos[mps.tomos[mps.tomo_id_col] == t_id]
            offsets = tomo_row[mps.region_offset_cols].to_numpy()[0]
            conv = mps.particles.loc[ind, init_cols] + offsets
            converted_list.append(conv)

        # add converted to original data
        converted = pd.concat(converted_list, axis=0)
        converted.rename(columns=column_rename, inplace=True)
        result = mps.particles.join(converted, how='left')

        if update:
            mps.particles = result
        else:
            return result

    def find_bins(self, mps):
        """Sets region_bin and region_bin_factor attributes.

        The values of these attributes apply to all tomos.
        
        If these attributes are set already, their values are kept.
        Otherwise, they are determined from columns mps.region_bin_col
        and mps.tomo_bin_col of table mps.tomos. If both fail, raises
        ValueError.

        Sets attrubutes:
          - self.region_bin: region bin
          - self.region_bin_factor: region bin / tomo bin
        
        Argument:
          - mps: (MultiParticleSets) tomos and particles dataframes object
        """

        if self.region_bin is None:
            reg_bins = mps.tomos[mps.region_bin_col].unique()
            if len(reg_bins) != 1:
                raise ValueError(
                    f"Region bin has to be defined by attribute "
                    + "self.region_bin, or it has to be specified in "
                    + f"column {mps.region_bin_col} of mps.tables and "
                    + "it has to have the same value for all tomos." )
            self.region_bin = reg_bins[0]
        if self.region_bin_factor is None:
            err_str = (
                f"Attribute self.region_bin_factor has to be defined, or "
                + f"regiona and tomo bins have to be specified in columns "
                + f"{mps.region_bin_col} and  {mps.tomo_bin_col} of "
                + f"mps.tables and they have to be the same for all tomos.")
            try:
                bin_factors = (mps.tomos[mps.region_bin_col]
                               // mps.tomos[mps.tomo_bin_col]).unique()
                if len(bin_factors) != 1:
                    raise ValueError(err_str)
            except AttributeError:
                raise ValueError(err_str)
            self.region_bin_factor = bin_factors[0]

    def add_paths(
            self, mps, star_path, path_col, mode, path_label=None,
            tomo_bin=None, tomo_bin_col=None, update=False):
        """Adds image paths from star file to MPS particle object table.

        Image paths are read from column (arg) path_label of the star
        file (arg star_file). The paths are saved in the column
        (arg) path_col of mps.tomos or mps.particles table, depending
        on arg mode.

        If the mps table where paths are added already contains
        column of the same name, these values are removed.

        Before adding tomo or region paths, determines tomo ids from
        column mps.micrograph_label from the star file. To do this,
        MultiParticleSets.add_tomo_ids() is used with arguments  
        self.tomo_id_mode, self.tomo_id_func and self.tomo_id_kwargs
        (see add_tomo_id() doc for more info).

        Arguments:
          - mps: (MultiParticleSets)
          - star_path: path to star file containing image paths 
          - mode: 'tomo' to add to mps.tomos, or 'particles' to add
          to mps.particles table
          - path_label: star file image path label, if None (default),
          self.path_label is used
          - path_col: column of mps.tomos or mps.particles table where
          the image paths are entered
          - update: if True mps is updated, if False the modified table
          is returned
        """

        if path_label is None:
            path_label = self.path_label
        if self.id_source_label is None:
            id_source_label = mps.micrograph_label
        else:
            id_source_label = self.id_source_label
            
        # read star that contains segmentation path
        star = pd.DataFrame(get_array_data(
            starfile=star_path, tablename='data', types=str))
        mps.add_tomo_ids(
            table=star, path_col=mps.micrograph_label,
            tomo_id_mode=self.tomo_id_mode,
            tomo_id_func=self.tomo_id_func, tomo_id_kwargs=self.tomo_id_kwargs)
        #star[mps.tomo_id_col] = star[id_source_label].map(
        #    lambda x: pyto.spatial.coloc_functions.get_tomo_id(
        #        path=x, mode=self.tomo_id_mode))
        star = star[[mps.tomo_id_col, path_label]].rename(
            columns={path_label: path_col}).copy()

        # add to table
        if mode == 'tomos':
            result = (mps.tomos  # keep mps.tomos index
                .reset_index()
                .merge(star, on=mps.tomo_id_col, how='left', sort=False,
                       suffixes=('_old', ''))
                .set_index('index'))
            if tomo_bin is not None:
                if tomo_bin_col is None:
                    tomo_bin_col = mps.tomo_bin_col
                result[tomo_bin_col] = tomo_bin
        elif mode == 'particles':
            result = (mps.particles  # keep mps.particles index
                .reset_index()
                .merge(star, on=mps.tomo_id_col, how='left', sort=False,
                       suffixes=('_old', ''))
                .set_index('index'))
        else:
            raise ValueError(
                f"Arg mode ({mode}) can be 'tomos' or 'particles'.")
        result.drop(columns=f"{path_col}_old", inplace=True, errors='ignore')
        result[path_col] = result[path_col].astype('string')

        if update:
            if mode == 'tomos':
                mps.tomos = result
            elif mode == 'particles':
                mps.particles = result
        else:
            return result

    def add_ctf(
            self, mps, star_path, update=False, check=False):
        """Adds ctf path to tomos table.

        The following columns are extracted from the star file that is 
        read from the specified path (arg star_path);
          - mps.micrograph_label, which is used to determine tomo id using
          pyto.spatial.coloc_functions.get_tomo_id(mode=self.tomo_id_mode)
          - self.ctf_label, shows ctf file path

        Column mps.micrograph_label is used only to determine tomo id. 
        This is done using MultiParticleSets.add_tomo_ids() with arguments  
        self.tomo_id_mode, self.tomo_id_func and self.tomo_id_kwargs
        (see add_tomo_ids() doc for more info). Ultimately, it calls
         pyto.spatial.coloc_functions.get_tomo_id(mode=self.tomo_id_mode)
        function, with arguments self.tomo_id_kwargs.
        
        The ctf path is added to mps.tomos table, column self.ctf_label.

        Arguments:
          - mps:
          - star_path: path to the star file containing ctf path
          - update: flag indication if mps.tomos is updated to contain
          ctf path column
          - check: probably not needed

        Returns: modified tomos table if update is False, otherwise None 
        """

        # convert ctf star to dataframe
        template = pd.DataFrame(
            get_array_data(
                starfile=star_path, tablename='data', types=str))
        labels = template.columns.copy()

        # add tomo ids to template
        mps.add_tomo_ids(
            table=template, path_col=mps.micrograph_label,
            tomo_id_mode=self.tomo_id_mode,
            tomo_id_func=self.tomo_id_func, tomo_id_kwargs=self.tomo_id_kwargs)
        #template[mps.tomo_id_col] = template[mps.micrograph_label].map(
        #    lambda x: pyto.spatial.coloc_functions.get_tomo_id(
        #        path=x, mode=self.tomo_id_mode))

        # keep only tomo and ctf paths
        paths_tab = template[
            [mps.tomo_id_col, mps.micrograph_label, self.ctf_label]].copy()
        paths_tab.drop_duplicates(inplace=True, ignore_index=True)
        paths_tab[self.ctf_label] = \
            paths_tab[self.ctf_label].astype('string')

        # add ctf info to tomos table
        result = (mps.tomos
            .reset_index()
            .merge(paths_tab, on=mps.tomo_id_col, how='left',
                   suffixes=('', '_test'))
            .set_index('index'))

        # just to check if tables and star have the same tomo paths (remove?)
        if check and not result[mps.tomo_col].eq(
                result[mps.micrograph_label]).all():
            raise ValueError(
                f"Tomo paths in columns {mps.tomo_col} and 'rlnMicrographName' "
                + f"are not the same")
        result.drop(columns=[mps.micrograph_label], inplace=True)

        if update:
            mps.tomos = result
        else:
            return result

    def convert_paths(
            self, mps, common, helper_path, path_cols=None,
            tomo_path_col=None, region_path_col=None, update=False):
        """Converts paths to another root in mps tables.

        Converts paths in columns (args) tomo_path_col, region_path_col and 
        those listed in path_cols in both mps.tomos and mps.particles
        tables.

        The conversion replaces everything before (arg) common in a path 
        by everything before arg helper_path (see SetPath.convert_paths() 
        for more info).

        If self.tomo_ids is not None, converts paths only for the 
        specified tomos.

        Arguments:
          - mps: (MultiParticleSets) object containing tomo and particle info
          - common: string common to path and arg helper_path
          - helper_path: replacement string, the part before arg common
          is used and the part after is ignored
          - tomo_path_col, region_path_col: columns containing paths 
          to be converted (typically tomo and segmentation paths)
          - path_cols: additional columns containing paths to be converted
          - update: if True, mps is updates, if False a modified MPS
          object is returned

        Returns (MultipleParticleSets) copied and modified mps object, 
        only if arg update is False
        """

        # select tomos
        if self.tomo_ids is not None:
            if update:
                mps.select(tomo_ids=self.tomo_ids, update=update)
            else:
                result = mps.select(tomo_ids=self.tomo_ids, update=update)
        if (self.tomo_ids is None) or update:
            result = mps

        # convert
        set_path = SetPath(common=common, helper_path=helper_path)
        convert_cols = [
            col for col in [tomo_path_col, region_path_col] if col is not None]
        if path_cols is not None:
            convert_cols = convert_cols + list(path_cols)
        for table in [result.tomos, result.particles]:
            for col in convert_cols:
                try:
                    table[col] = table[col].map(
                        lambda x: set_path.convert_path(x))
                    table[col] = \
                        table[col].astype('string')
                except KeyError:
                    if col in table.columns:
                        raise
                    else:
                        pass
            
        #if tomo_path_col is not None:
        #    result.tomos[tomo_path_col] = result.tomos[tomo_path_col].map(
        #        lambda x: set_path.convert_path(x))
        #    result.tomos[tomo_path_col] = \
        #        result.tomos[tomo_path_col].astype('string')
        #if region_path_col is not None:
        #    result.tomos[region_path_col] = result.tomos[region_path_col].map(
        #        lambda x: set_path.convert_path(x))        
        #    result.tomos[region_path_col] = \
        #        result.tomos[region_path_col].astype('string')
        #if path_cols is not None:
        #    for pa_col in path_cols:
        #        try:
        #            result.tomos[pa_col] = result.tomos[pa_col].map(
        #                lambda x: set_path.convert_path(x))
        #            result.tomos[pa_col] = \
        #                result.tomos[pa_col].astype('string')
        #        except KeyError:
        #            if pa_col in result.tomos.columns:
        #                raise
        #            else:
        #                pass
                    
        if not update:
            return result
        
    def find_corners(
            self, mps, image_path_col, box_size, coord_cols,
            l_corner_cols, r_corner_cols, 
            shape_cols=None, column='inside', update=False):
        """Find particle box corners and label those that fit inside.

        Applicable to (greyscale) tomograms, regions and segmentations.

        For real particles that are directly extracted from tomo, arg
        box_size is the intended particle size. However, for regions
        (segmentations) where regions are binned with respect to the
        corresponding particle tomo, arg box_size should be the size
        of the box extracted from regions. For example, if region tomos
        are binned 2x with respect to particle tomos, arg box size 
        should be half of the particle box size.

        Arguments:
          - mps: (MultiParticleSets) particles
          - image_path_col: column of mps.tomos that contain image paths
          - coord_cols: columns of mps.particles that contain particle
          center coords
          - l_corner_cols, r_corner_cols: columns of mps.particles where
          the calculated minimum (lower left) and maximum (upper right)
          box coordinates, respectively, are saved
          - box_size: particle size in the frame of the image from which 
          particles are extracted (in pixels)
          - shape_col: names of columns that contain image shape, if None 
          (default) shape is determined from the header of the image
          - column: column of mps.particles where the flag showing whether
          particle boxes fit in the tomo is saved
          - update: flag that determines whether mps.particles is updated,
          or it is returned (default False)
        """

        # center - l corner and center - r corner distances
        center_coord = box_size // 2
        center_plus = box_size - center_coord    

        # convert particle image centers for each tomo separately
        res_list = []
        part_by_tomos = mps.particles.groupby(mps.tomo_id_col)
        for t_id, ind in part_by_tomos.groups.items():

            # get corner coords
            coords = mps.particles.loc[ind, coord_cols]
            l_corner = coords - center_coord
            l_corner.columns = l_corner_cols
            r_corner = coords + center_plus
            r_corner.columns = r_corner_cols

            # get tomo shape
            tomo_row = mps.tomos[mps.tomos[mps.tomo_id_col] == t_id]
            if shape_cols is not None:
                shape = tomo_row[shape_cols].to_numpy()[0]
            else:
                image_path = tomo_row[image_path_col].to_numpy()[0]
                try:
                    image = pyto.io.ImageIO()
                    image.readHeader(file=image_path)
                    shape = np.asarray(image.shape)
                except FileNotFoundError:
                    shape = None
                    
            # find inside / outside
            if shape is not None:
                inside = ((l_corner >= 0).all(axis=1)
                          & (r_corner < shape).all(axis=1))
            else:
                inside = (l_corner >= 0).all(axis=1)
            inside.rename(column, inplace=True)

            # put corners and inside together
            res_list.append(l_corner.join([r_corner, inside]))

        # add converted to original data
        res_tab = pd.concat(res_list, axis=0)

        if update:
            result = mps.particles.join(res_tab, how='left')
            mps.particles = result
        else:
            return res_tab

    @classmethod
    def write_particles(
            cls, mps, l_corner_cols, r_corner_cols, image_path_col, dir_,
            expand, select_col=None, 
            mean=None, std=None, invert_contrast=False, fun=None, fun_kwargs={},
            image_path_mode='image', name_prefix='particle_', name_suffix='', 
            particle_path_col='particle', convert_path_common=None,
            convert_path_helper=None, update=False, write=True):
        """Writes particle or boundary subtomos.

        Particle or boundary images (subtomos) can be extracted from images
        specified in column arg image_path_col of mps.tomos table.
        In this case, arg image_path_mode has to be 'image'.
        The column name is usually 'tomo' for particles and 'region'
        for boundaries, or specified by mps.tomo_col.

        Boundaries can be and segments have to be extracted from
        presynaptic result (pickle) files. To do that, arg image_path_mode
        has to be 'pkl_boundary' or 'pkl_segment'.
        
        In 'pkl_segment' mode (arg image_path_mode), all other segments
        that may be present in a particle image are removed before
        applying functions specified by arg fun.
        """

        # remove outside particles
        parts_tab = mps.particles
        if select_col is not None:
            parts_tab = parts_tab[parts_tab[select_col]]

        # set flag to make particle images containing only one particle 
        keep_id_only = False
        if image_path_mode == 'pkl_segment':
            keep_id_only = True
            
        #
        p_indices = []
        path_list = []

        # loop over tomos
        part_by_tomos = parts_tab.groupby(mps.tomo_id_col)
        for tomo_id, ind in part_by_tomos.groups.items():

            # get tomo data
            tomo_row = mps.tomos[mps.tomos[mps.tomo_id_col] == tomo_id]        
            tomo_path = tomo_row[image_path_col].to_numpy()[0]

            if write:
            
                if image_path_mode == 'image':
                    image = pyto.core.Image.read(
                        file=tomo_path, header=True, memmap=True)
                    pixelsize = image.pixelsize
                    header = image.header

                elif ((image_path_mode == 'pkl_boundary')
                      or (image_path_mode == 'pkl_segment')):
                    scene = pickle.load(
                        open(tomo_path, 'rb'), encoding='latin1')
                    if image_path_mode == 'pkl_boundary':
                        image = scene.boundary
                    else:
                        image = scene.labels
                        pixelsize = tomo_row[mps.pixel_nm_col].to_numpy()[0]
                        header = None
                        #image.write(file=f"bound_{tomo_id}.mrc")

                else:
                    raise ValueError(
                        f"Argument image_path_mode {image_path_mode} was not "
                        + "understood.")
            else:
                image = None
                
            # loop over particles
            for p_ind, row in parts_tab.loc[ind].iterrows():

                # make slice objects
                l_corner = row[l_corner_cols].to_numpy()
                r_corner = row[r_corner_cols].to_numpy()
                slices = [
                    slice(left, right) for left, right
                    in zip(l_corner, r_corner)]

                if image is not None:
                
                    # get particle data
                    particle_data = image.useInset(
                        inset=slices, mode=u'relative', expand=expand,
                        update=False, returnCopy=True)

                    # process particle
                    if std is not None:
                        particle_data = (
                            std * particle_data / particle_data.std())
                    if mean is not None:
                        particle_data = (
                            particle_data - particle_data.mean() + mean)
                    if invert_contrast:
                        particle_data = -particle_data
                    if keep_id_only:
                        particle_id = row[mps.particle_id_col]
                        particle_data[particle_data != particle_id] = 0
                    if fun is not None:
                        if isinstance(fun, (tuple, list)):
                            for fun_one, fun_kwargs_one in zip(fun, fun_kwargs):
                                particle_data = fun_one(
                                    particle_data, **fun_kwargs_one)    
                        else:
                            particle_data = fun(particle_data, **fun_kwargs)
                        
                # write particle
                particle_id = row[mps.particle_id_col]
                particle_path = os.path.abspath(os.path.join(
                    dir_, tomo_id,
                    f"{name_prefix}{particle_id}{name_suffix}.mrc"))
                if write:
                    particle = pyto.core.Image(data=particle_data)    
                    try:
                        particle.write(
                            file=particle_path, header=header, pixel=pixelsize)
                    except IOError:
                        os.makedirs(os.path.dirname(particle_path))
                        particle.write(
                            file=particle_path, header=header, pixel=pixelsize)

                # add particle path to row
                p_indices.append(p_ind)
                path_list.append(particle_path)

        # add all particle paths to table
        particle_path = pd.DataFrame(
            {particle_path_col: path_list}, index=p_indices)
        parts_tab = parts_tab.join(particle_path, how='left')
        set_path = SetPath(
            common=convert_path_common, helper_path=convert_path_helper)
        try:
            parts_tab[particle_path_col] = parts_tab[particle_path_col].map(
                lambda x: set_path.convert_path(x))
        except ValueError:
            pass
        parts_tab[particle_path_col] = \
            parts_tab[particle_path_col].astype('string')

        if update:
            mps.particles = parts_tab
        else:
            return parts_tab

    def make_star(
            self, mps, labels, star_path=None, comment="From MPS",
            verbose=False, out_desc=""):
        """Writes relion particle star file from particles and tomos tables.

        Takes data from mps.tomos and mps.particles to make a DataFrame that 
        contains all required data for a particle star file, and saves this
        as file. Then, writes a star file from this data.

        The output star file contains columns defined in self.get_labels().

        Arguments:
          - mps: (MultiParticleSets) particle sets object that contain tomo
          and particle data in attributes tomos and particles, respectively
          - labels: (dict) keys are star file labels, and values the 
          corresponding column names, for example self.get_labels(mps)
          - star_path: path to the out star file, or None for not writing
          - comment: comment addet to star file
          - verbose: flag indicating whether writting info about writting
          the generated dataframe, passed to pyto.io.PandasIO.write()
          - out_desc: description of the file that is written, used
          only if verbose is True 

        Returns DataFrame corresponding to the generated star file. 
        """

        # find labels that are in tomos but not in particles
        tomo_labels = dict([
            (lab, col) for lab, col in labels.items()
            if not col in mps.particles.columns])
        tomo_clean = mps.tomos[[mps.tomo_id_col] + list(tomo_labels.values())]

        # put tomo info to particles
        combined = (mps.particles
            .reset_index()
            .merge(tomo_clean, on=mps.tomo_id_col, how='left',
                   suffixes=('_bad', ''), sort=False)
            .set_index('index'))

        # convert data to dict
        data = {}
        for lab, col in labels.items():
            data[lab] = combined[col].to_numpy()

        # write 
        if star_path is not None:

            # write star file
            try:
                write_table(
                    starfile=star_path, labels=list(labels.keys()), data=data, 
                    format_=self.label_format, tablename='data_',
                    delimiter=' ', comment=f"# {comment}")
                if verbose:
                    print(f"Wrote {out_desc} star file {star_path}")
            except FileNotFoundError:
                os.makedirs(os.path.dirname(star_path))
                write_table(
                    starfile=star_path, labels=list(labels.keys()), data=data, 
                    format_=self.label_format, tablename='data', delimiter=' ', 
                    comment=f"# {comment}")                 
                if verbose:
                    print(f"Wrote {out_desc} star file {star_path}")

            # DataFrame corresponding to the star file
            star_sp = star_path.rsplit('.', 1)
            tab_path = f"{star_sp[0]}_{star_sp[1]}.pkl"
            pyto.io.PandasIO.write(
                table=combined, base=tab_path, file_formats=['json'],
                verbose=verbose,
                out_desc=f"DataFrame version of {out_desc} star file")

        return combined

    def split_star(
            self, mps, class_code, labels, star_path,
            class_names=None, star_comment=None, verbose=True, out_desc=""):
        """Makes star files for each subclass separately

        Splits particles of (arg) mps based on values of column 
        mps.class_number_col (of mps.particles) according to subclasses 
        defined by (arg) class_code. Only particles that have one of 
        the classes defined by (arg) class_names in the column 
        mps.class_name_col (of mps.particles) are selected for 
        subclasses.

        Writes a star file for each class, where path are obtained by
        replacing the trailing '_all.star' or arg star path by 
        '<subclass>.star', where subclass is a value of arg class_code.

        Arguments:
          - mps: (MultiParticleSets) particles
          - class_code: (dict) keys are class numbers as given in column 
          mps.class_number_col of mps.particles and values are
          class names
          - labels: star file labels, such as self.get_labels(mps)
          - star_path: path to star file that contains all particles, has 
          to end with ('_all.star')
          - class_names: one or more class names (elements of column
          mps.class_name_col of mps.particles table)
          - star_comment: comment written at the beginning of star files 
        """

        # write star file for each subclass separately
        for number, subclass in class_code.items():
            mps_curr = mps.select(
                class_names=class_names, class_numbers=[number], update=False)
            star_path_curr = star_path.replace('_all.star', f'_{subclass}.star')
            star_comment_curr = star_comment.replace(
                'All', subclass.capitalize())
            self.make_star(
                mps=mps_curr, labels=labels, 
                star_path=star_path_curr, comment=star_comment_curr,
                verbose=verbose, out_desc=f"class {subclass} {out_desc}")

    def get_labels(self, mps, use_priors=True):
        """Associates relion star file labels to mps columns.

        Returns dictionary where keys are relion particle star file labels
        and values are the corresponding column names of mps.tomos
        and mps.particles.
        """

        # use only priors from mps if requested, otherwise priors if exist  
        if use_priors:
            tilt_label = 'rlnAngleTiltPrior'
            psi_label = 'rlnAnglePsiPrior'
            tilt_label_prior = 'rlnAngleTiltPrior'
            psi_label_prior = 'rlnAnglePsiPrior'
        else:
            tilt_label = 'rlnAngleTilt'
            psi_label = 'rlnAnglePsi'
            if 'rlnAngleTiltPrior' in mps.particles.columns:
                tilt_label_prior = 'rlnAngleTiltPrior'
            else:
                tilt_label_prior = 'rlnAngleTilt'
            if 'rlnAnglePsiPrior' in mps.particles.columns:
                psi_label_prior = 'rlnAnglePsiPrior'
            else:
                psi_label_prior = 'rlnAnglePsi'
                
        labels_loc = {
            'rlnMicrographName': mps.tomo_col, 'rlnCtfImage': self.ctf_label, 
            'rlnImageName': self.tomo_particle_col,
            'rlnCoordinateX': mps.center_init_frame_cols[0], 
            'rlnCoordinateY': mps.center_init_frame_cols[1], 
            'rlnCoordinateZ': mps.center_init_frame_cols[2], 
            'rlnAngleTilt': tilt_label,
            'rlnAngleTiltPrior': tilt_label_prior, 
            'rlnAnglePsi': psi_label,
            'rlnAnglePsiPrior': psi_label_prior, 
            'rlnAngleRot': 'rlnAngleRot'}
        
        return labels_loc

    @classmethod
    def remove_region_cols(cls, mps):
        """Removes columns that contain region info

        """
        drop_cols = ([mps.region_col, mps.region_id_col] 
                     + mps.region_offset_cols + mps.region_shape_cols)
        for col in drop_cols:
            try:
                mps.tomos.drop(columns=[col], inplace=True)
            except KeyError:
                pass
        drop_cols = (
            mps.orig_coord_reg_frame_cols + mps.coord_reg_frame_cols 
            + mps.center_reg_frame_cols
            + mps.reg_l_corner_cols  + mps.reg_r_corner_cols
            + [mps.reg_inside_col])
        for col in drop_cols:
            try:
                mps.particles.drop(columns=[col], inplace=True)
            except KeyError:
                pass

    def convert_to_struct_region(
            self, mps, scalar, indexed, struct_path_col, image_path_mode,
            init_coord_cols, region_coord_cols, 
            convert_path_common=None, convert_path_helper=None,
            region_bin=1, path_col=None, offset_cols=None,
            shape_cols=None, bin_col=None):
        """Converts coordinates to region contained in a structure pickle.

        Meant to transform (full size) tomo frame coordinates to coordinates
        of regions (boundaries) saved in structural pickles.

        To acheieve this, coordinates are converted like:
          bin_factor * x - offset
        where:
          bin_factor = 1 / self.bin_factor
          offset: offset (inset) of the boundary region structure pickles

        Arguments:
          - struct_segment: attribute of structure object that holds
          the image, can be 'boundary' for boundaries (regions) or 
          'labels' (same as 'hierarchy') for segmented particles
          - offset_cols, shape_cols: names of columns containing offsets 
          and shape of regions images
          - init_coord_cols: column names of initial coordinates
          - region_coord_cols: column names where the transformed 
          coords are stored
        """

        # default columns
        if path_col is None:
            path_col = mps.region_col
        if offset_cols is None:
            offset_cols = mps.region_offset_cols    
        if shape_cols is None:
            shape_cols = mps.region_shape_cols
        if bin_col is None:
            bin_col = mps.region_bin_col

        tomos = mps.tomos.copy()
        part_list = []
        for to_id, scalar_one, indexed_one, scene in tomo_generator(
            scalar=scalar, indexed=indexed, identifiers=self.tomo_ids, 
            pickle_var=struct_path_col, convert_path_common=convert_path_common,
            convert_path_helper=convert_path_helper):

            # check if tomo exists in mps.tomos
            try:
                tomo_ind = \
                    mps.tomos[mps.tomos[mps.tomo_id_col] == to_id].index[0]
            except IndexError:
                continue

            # get boundary object and extract data
            scene_pkl_path = scalar_one[struct_path_col]
            if image_path_mode == 'pkl_boundary':
                bound = scene.boundary
            elif image_path_mode == 'pkl_segment':
                bound = scene.labels
            offsets = [sl.start for sl in bound.inset]
            shape = bound.data.shape

            # add boundary pickle path, offset and shape to tomos
            tomos.loc[tomo_ind, path_col] = scene_pkl_path
            tomos.loc[tomo_ind, offset_cols] = offsets
            tomos.loc[tomo_ind, shape_cols] = shape
            tomos.loc[tomo_ind, bin_col] = region_bin
            # removed because coords in different bins exist
            # together in particles table
            #bin_fact = tomos.loc[tomo_ind, mps.coord_bin_col] / region_bin
            bin_fact = 1 / self.region_bin_factor

            # extract particles for the current tomo
            part_one = \
                mps.particles[mps.particles[mps.tomo_id_col]==to_id].copy()

            # convert coords
            coords_orig = part_one[init_coord_cols].to_numpy()
            coords_final = (
                bin_fact * coords_orig - np.asarray(offsets).reshape(1, -1))
            #coords_final = np.rint(coords_final).astype(int)
            part_one[region_coord_cols] = coords_final
            part_list.append(part_one)

        # make full particles table
        converted_part = pd.concat(part_list, axis=0)
        converted_part[region_coord_cols] = \
            converted_part[region_coord_cols].round().astype(int)

        result = deepcopy(mps)
        result.tomos = tomos
        result.particles = converted_part

        return result

    def prepare_func(
            self, zoom_factor=1, zoom_order=0, normalize_kwargs={},
            smooth_kwargs={}, dilate=None, dtype=None):
        """Returns function(s) and argument(s) that modify images.

        Returns a list of functions and list of (dict) kwargs that 
        correspond to the functions, in the following order:
          - function scipy.ndimage.zoom, kwargs {'zoom': zoom_factor, 
          'order': zoom_order}, if arg zoom_factor != 1
          - function self.normalize_bound_ids, kwargs normalize_kwargs,
          if len(normalize_kwargs) > 0
          - function self.smooth_bounds, kwargs boundary_ids, external_ids,
          and zoom (default arg zoom_factor, can be owerridden by arg
          smooth_kwargs)
          - function scipy.ndimage.grey_dilation, kwargs 
          {'footprint': skimage.morphology.ball(dilate)}, if arg dilate 
          is specified and dilate != 0
          - function numpy.asarray, kwargs{'dtype': dtype}, if arg 
          dtype is specified
        See individual function (method) docs for more info.

        Returns:
          - fun: (list) functions in the order they should be applied
          - fun_kwargs: (list of dicts) kwargs for the abve functions
          in the same order
        """

        fun = []
        fun_kwargs = []
        if zoom_factor != 1:
            fun.append(sp.ndimage.zoom)
            fun_kwargs.append({'zoom': zoom_factor, 'order': zoom_order})
        if len(normalize_kwargs) > 0:
            fun.append(self.normalize_bound_ids)
            fun_kwargs.append(normalize_kwargs)
        if (smooth_kwargs is not None) and len(smooth_kwargs) > 0:
            fun.append(self.smooth_bounds)
            smooth_kwargs_loc = {'zoom': zoom_factor}
            smooth_kwargs_loc.update(smooth_kwargs)
            fun_kwargs.append(smooth_kwargs_loc)
        if (dilate is not None) and (dilate != 0):
            fun.append(sp.ndimage.grey_dilation)
            structure = skimage.morphology.ball(dilate)
            fun_kwargs.append({'footprint': structure})
        if dtype is not None:
            fun.append(np.asarray)
            fun_kwargs.append({'dtype': dtype})
            
        return fun, fun_kwargs
            
    @classmethod
    def normalize_bound_ids(
            cls, data, min_id_old=None, id_new=None, id_conversion={},
            dtype=np.int16):
        """Sets boundary ids to the specified (normalized) values.

        Used for segmented images such as those showing boundaries, regions
        or other segments. Makes a new image where pixel values of the 
        initial image (arg data) are replaced as follows: 
          - all values that are >= (arg) min_id_old are replaced by 
          (arg) id_new
          - pixels having values equal to keys of (arg) id_conversion 
          are replaced by their corresponding values
          - all other pixels are set to 0

        The resulting boundary image contains boundaries only for
        the ids specified by id_new and values of id_conversion. 

        For example if initially vesicles have labels [10, 11, 12, ...],
        plasma membrane 2 and cytoplasmic region 3, and the intended
        value for all vesicles is 8, plasma membrane 4 and cytosol 1, use:
          normalize_bound_ids(
              data, min_id_old=5, id_new=10, id_conversion={2: 4, 3: 1})

        If the resulting image is meant to be saved, (arg) dtype has to be 
        one of the allowed data types for the intened image format.

        Arguments:
          - data: (ndarray) initial image
          - min_id_old: min value of all ids that are replaced by id_new 
          - id_new: replacement calue for >= min_id_old
          - id_conversion: (dict) 1-1 mapping old_value: new_value
          - dtype: dtype of the final image (default np.int16)

        Returns (ndarray) modified image (arg data is modified)
        """

        if min_id_old is not None:
            new_data = np.where(data>=min_id_old, id_new, 0)
        else:
            new_data = np.zeros_like(data)
        for old, new in id_conversion.items():
            new_data += np.where(data==old, new, 0).astype(data.dtype)
        if dtype is not None:
            new_data = new_data.astype(dtype)

        return new_data

    @classmethod
    def smooth_bounds(
            cls, data, bound_ids, external_ids, operations=None, zoom=2):
        """Morphologial smoothing of boundaries.

        """

        if zoom == 1:
            return data
        
        from ..spatial.boundary_smooth import BoundarySmooth
        bs = BoundarySmooth(
            image=data, segment_id=bound_ids, external_id=external_ids)
        if operations is None:
            operations = ''.join([op*(zoom//2) for op in 'deed'])
        image = bs.morphology_pipe(operations=operations)
        return image
    
                
class Paths:
    """Contains attributes specifying particle and table paths
    """
    
    def __init__(
            self, name, root_template='particles_size-{size}',
            regions='regions', size=64, tables='tables'):
        self.name = name
        self.root_template = root_template
        self.root = root_template.format(size=size)
        self.regions = regions
        self.size = size
        self.tables = tables
        
    @property
    def particles_root(self):
        #return f'../particles_bin-2_size-{self.size}'
        return self.root
    
    @property
    def particles_dir(self):
        return os.path.join(self.particles_root, self.name)

    @property
    def regions_dir(self):
        return os.path.join(self.particles_root, self.regions)
    
    @property
    def mps_path(self):
        return os.path.join(
            self.particles_dir, f'{self.tables}/{self.name}.pkl')
    
    @property
    def mps_path_tmp(self):
        return os.path.join(
            self.particles_dir, f'{self.tables}/{self.name}_tmp.pkl')
    
    @property
    def star_path(self):
        return os.path.join(
            self.particles_dir, f'{self.tables}/{self.name}_all.star')

    @property
    def mps_star_path(self):
        return os.path.join(
            self.particles_dir, f'{self.tables}/{self.name}_all_star.pkl')
