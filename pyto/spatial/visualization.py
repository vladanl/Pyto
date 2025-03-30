"""
Contains class Visualization used to map and visualize particles in tomograms.

Usage is shown in the particle mapping example (examples/map_particles/).

# Author: Vladan Lucic 
# $Id$
"""

__version__ = "$Revision$"


import os

import numpy as np
import pandas as pd

import pyto
from pyto.io.pandas_io import PandasIO
from pyto.geometry.rigid_3d import Rigid3D
import pyto.particles.relion_tools as relion_tools
from pyto.spatial.multi_particle_sets import MultiParticleSets


class Visualization:
    """Particle visualization in tomograms.
    """

    def __init__(
            self, tomo_id_mode=None, tomo_id_func=None, tomo_id_kwargs={},
            micrograph_star='rlnMicrographName', image_star='rlnImageName',
            particle_id_star_col = 'particle_id', degree=True,
            star_euler_mode='zyz_in_passive', euler_mode='zyz_ex_active',
            coord_cols=['x_orig', 'y_orig', 'z_orig'],
            coords_map=['x_map','y_map','z_map'],
            coords_map_a=['x_map_a', 'y_map_a', 'z_map_a'], 
            angle_cols=['phi', 'theta', 'psi'],
            spherical_cols=['sph_theta', 'sph_phi'],
            vector_cols=['v_x', 'v_y', 'v_z'], csv_cols='default',
            box_center_cols = ['box_center_x', 'box_center_y', 'box_center_z'],
            box_min_cols = ['box_min_x', 'box_min_y', 'box_min_z'],
            box_max_cols = ['box_max_x', 'box_max_y', 'box_max_z']):
        """Sets attributes from arguments.

        Args tomo_id_mode, tomo_id_func and tomo_id_kwargs are used
        to determine tomogram ids from the (arg) micrograph_star of
        from the star file (star_path arg of make_map_params()), as
        explained in the docs of make_map_params()).
        
        Arguments:
          - tomo_id_mode, tomo_id_func, tomo_id_kwargs: used to
          determine tomo id from star files 
          - micrograph_star: star file column name from which tomo id
          is determined  (if None, 'rlnMicrographName' is used)
          - image_star: star file column from which particle id is determined
          (if None, 'rlnImageName' is used)
          - degree: flag indicating if angles are specified in degrees
          both in star file and the resulting particle table (default True)
          - star_euler_mode: Euler rotation coonvention in star file
          (default 'zyz_in_passive')
          - euler_mode: Euler rotation coonvention in the final
          particle table (default 'zyz_ex_active')
          - coord_cols: particle center coordinate columns is star file
          (default ['x_orig', 'y_orig', 'z_orig'])
          - coords_map, coords_map_a: particle center coordinate columns
          in the map tomogram system (default ['x_map','y_map','z_map']
          and ['x_map_a', 'y_map_a', 'z_map_a'], respectively) 
          - angle_cols: columns containing Euler angles in (arg)
          euler_mode convention (default ['phi', 'theta', 'psi'])
          - spherical_cols: columns containing spherical aggles 
          (default ['sph_theta', 'sph_phi'])
          - vector_cols: columns containing unit vector in the direction
          of the spherical angles (default ['v_x', 'v_y', 'v_z'])
          - csv_cols: table columns that are saved in the csv file, None
          for all columns, 'default' for the necessary columns
          (default 'default')
          - box_center_cols: columns contining coordinates of particle
          boxes where they are positioned in a map (default
          ['box_center_x', 'box_center_y', 'box_center_z']
          - box_min_cols, bax_max_cols: columns containing min and max
          corner coordinates  of particle boxes where they are
          positioned in a map (default ['box_min_x', 'box_min_y',
          'box_min_z'] and ['box_max_x', 'box_max_y', 'box_max_z'],
          respectively)
        """

        self.tomo_id_mode = tomo_id_mode
        self.tomo_id_func = tomo_id_func
        self.tomo_id_kwargs = tomo_id_kwargs
        self.micrograph_star = micrograph_star
        self.image_star = image_star
        self.particle_id_star_col = particle_id_star_col
        self.degree = degree
        self.star_euler_mode = star_euler_mode
        self.euler_mode = euler_mode
        
        self.coord_cols = coord_cols
        self.coords_map = coords_map
        self.coords_map_a = coords_map_a
        self.angle_cols = angle_cols
        self.spherical_cols = spherical_cols
        self.vector_cols = vector_cols
        self.csv_cols = csv_cols
        if self.csv_cols == 'default':
            self.csv_cols = (
                ['tomo_id', self.particle_id_star_col]
                + self.coords_map + self.coords_map_a + self.vector_cols)
        self.box_center_cols = box_center_cols
        self.box_min_cols = box_min_cols
        self.box_max_cols = box_max_cols
        
    def make_map_params(
            self, star_path, tomo_id, 
            coord_bin=1, map_pixel_nm=1, map_bin=1, map_offset=0,
            table_name='data', priors=False, 
            csv_path=None, float_format='%8.3f', class_number=-1):
        """Extracts parameters used for mapping from a particle star file.

        Extracts mapping parameters (location and angular orientation)
        from a star file (arg star_path) for one tomogram. The tomogram
        is specified by tomogram id (arg tomo_id).

        Tomogram id is determined from the self.mirograph_star column of
        the star file, in the following order (self.tomo_id_mode,
        self.tomo_id_func and self.tomo_id_kwargs are set in __init__()
        from arguments of the same names):
          - if self.tomo_id_mode is not None, the function returned by 
          coloc_functions.get_tomo_id(tomo_id_mode) is executed to
          obtain tomo id
          - if self.tomo_id_func is not None, tomo id is the result of 
          self.tomo_id_func(star_value, **self.tomo_id_kwargs)
          - if both self.tomo_id_mode and self.tomo_id_func are None,
          (the unchanged) values of the self.mirograph_star column
          are used
        For example, if self.tomo_id_func is set to:
            (lambda x: os.path.splitext(os.path.basename(x))[0])
        tomo ids are determined by removing the directories and file
        name extensions of the paths specified in the
        self.mirograph_star column. 
        
        Original particle positions are determined from star file columns
        'rlnCoordinateXYZ', 'rlnOriginXYZ' and 'rlnOriginXYZAngst', and
        angles are determined from 'rlnAngleTiltPsiRot'. See docs for
        multi_particle_sets.read_star() for particle mode for more info
        about reading star file.

        The map particle positions (coordinates) are obtained from the
        original particle positions and binning factors as:

          map_coords = original_coords * coord_bin / map_bin - map_offset

        Particle angles are determined by converting star file angles
        (Euler convention arg star_euler_mode, default zyz intrinsic
        passive) to Euler convention self.euler_mode (default zyz
        extrinsic active). Attribute self.degree (default True)
        determines if angles are in degrees or radians.

        Csv file containing selective data (self.csv_cols) is saved
        at (arg) csv_path. Overwrites a previously existing file.
        Creates directories if needed.
        
        In addition to the star file columns, the following columns are
        added to the returned table:
          - 'tomo_id': tomo id, extracted from self.micrograph_star
          star file column 
          - self.particle_id (default 'particle_id'): particle id,
          extracted from (arg) image_star (if None, 'rlnImageName' is used)
          star file column
          - self.coords_map (default 'x_orig', 'y_orig', 'z_orig'):
          calculated particle coordinates in map tomo in pixels
          - self.coords_map_a (default 'x_orig', 'y_orig', 'z_orig'):
          calculated particle coordinates in map tomo in A (needed
          for paraview visualization)
          - self.spherical_cols (default sph_theta, sph_phi): particle
          orientation in spherical coordinates (ignores one angle)
          - self.angle_cols (default phi, theta, psi): particle
          orientation Euler angles in convention specified by
          self.euler_mode (default zyz extrinsic active)
          - self.vector_cols (default v_x, v_y, v_z: coordinates of a
          unit vector defined by the particle spherical coordinates
          (needed for paraview visualization)
        
        Arguments:
          - star_path: path to the star file
          - tomo_id: id of tomogram
          - coord_bin: bin factor of the star file coordinates (default 1)
          - map_pixel_nm: pixel size of the map [nm] (default 1)
          - map_bin: bin factor of the map coordinates (default 1)
          - map_offset: (list of 3 elements or 0) position (offset) of
          map coordinates with respect to the star coordinates at the
          map bin (default 0)
          - table_name: star file table name (default 'data')
          - priors: Flaf indicating whether to use prior angles (default False)
          - csv_path: output csv file path, if None csv is not written
          (default None)
          - float_format: float format in the csv file (default '%8.3f')
          - class_number: indicates which particle class from the star
          file is used, -1 to use all particles (default -1)

        Returns (pandas.DataFrame): star file data and calculated parameters
        """

        # chose angles from star file
        if priors:
            star_angles = [
                'rlnAngleRot', 'rlnAngleTiltPrior', 'rlnAnglePsiPrior']
        else:
            star_angles = ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
    
        # read star
        mps = MultiParticleSets()
        mps.micrograph_label = self.micrograph_star
        mps.image_label = self.image_star
        particles = mps.read_star(
            path=star_path, mode='particle', tomo_id_mode=self.tomo_id_mode,
            tomo_id_func=self.tomo_id_func, tomo_id_kwargs=self.tomo_id_kwargs,
            tablename=table_name, do_origin=True, class_number=class_number)

        # extract coords from star and convert them
        if not isinstance(map_offset, (tuple, list, np.ndarray)):
            if map_offset == 0:
                map_offset = np.zeros(len(self.coords_map))
        parts = particles.query("tomo_id == @tomo_id").copy()
        parts[self.coords_map] = (
            parts[self.coord_cols] * coord_bin / map_bin
            - np.asarray(map_offset))
        parts[self.coords_map_a] = parts[self.coords_map] * map_pixel_nm * 10

        # determine spherical angles from star file angles
        line_proj = pyto.spatial.LineProjection(relion=True)
        parts[self.spherical_cols] = parts.apply(
            lambda x: pd.Series(
                line_proj.find_spherical([x[ang] for ang in star_angles]),
                index=["theta", "phi"]),
            axis=1)

        # convert star angles to 'zyz_ex_active'
        deg_to_rad = 1
        if self.degree:
            deg_to_rad = np.pi / 180
        if self.angle_cols is not None:
            parts[self.angle_cols] = parts.apply(
                lambda x: pd.Series(
                    np.asarray(Rigid3D.convert_euler(
                        angles=np.array(
                            [x[ang] for ang in star_angles]) * deg_to_rad, 
                        init=self.star_euler_mode,
                        final=self.euler_mode)) / deg_to_rad,
                    index=self.vector_cols),
                axis=1)
            
        # make unit vectors from spherical angles
        if self.vector_cols is not None:
            parts[self.vector_cols] = parts.apply(
                lambda x: pd.Series(
                    Rigid3D.euler_to_vector(
                        angles=np.array([x[ang] for ang in star_angles]), 
                        mode=self.star_euler_mode, degree=True),
                    index=self.vector_cols),
                axis=1)

        # save to csv
        if csv_path is not None:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            if self.csv_cols is None:
                parts.to_csv(
                    csv_path, index_label='index', float_format=float_format)
            else:
                parts[self.csv_cols].to_csv(
                    csv_path, index_label='index', float_format=float_format)
            print(f"Wrote paraview input csv file at {csv_path}")

        return parts

    def map_glyphs(
            self, glyph, params, map_shape, map_path, 
            glyph_bin=1, map_pixel_nm=1, map_bin=1, spline_order=2, 
            threshold=None, dtype='float32', overlap_ok=True,
            glyph_out_prefix=None, debug=False):
        """Maps glyphs to particle positions and angular orientations.

        Generates a tomogram where glyphs (such as particle averages,
        arg glyph) are positioned and rotated according to the particle
        parameters (arg params).

        For each particle, glyph is transformed as follows:
          - initial glyph is rotated according to angles given in
          params table (columns self.angle_cols) in Euler convention
          self.euler_mode
          - scalled by factor map_bin/glyph bin
          - thresholded at ath threshold
          - converted to type (arg) dtype
          - center of the glyph is placed at position given by columns
          self.coords_map of the params table (corners are saved in
          self.min_cols and self.max_cols of the returned table)
          - transformed glyphs are saved as individual mrc images
        
        The size of the glyph is adjusted to the map bin factor using
        args map_bin and glyph_bin.

        Saves a map and individual transformed glyphs unless the
        respective paths are None. Overwrites previously existing files
        and creates directories if needed.
        
        Arguments:
          - glyph: path to glyph image (str) or image data (np.ndarray) 
          - params: (pd.DataFrame) table that contains particle coordinates
          and angular orientations
          - map_shape: shape of the map tomo
          - map_path: path of the map tomo
          - glyph_bin: glyph bin factor (default 1)
          - map_pixel_nm: map pixel size [nm] (default 1)
          - map_bin: glyph bin factor (default 1)
          - spline_order: spline interpolation order, passes as argument
          order to  scipy.ndimage.map_coordinates() (default 2)
          - threshold: transformed glyph threshold level (default None)
          - dtype: transformed glyph dtype (default 'float32')
          - overlap_ok: flag indicating whether glyphs can are allowed to
          overlap (default True)
          - glyph_out_prefix: directory and file name suffix, if None
          individual transformed glyphs are not saved (default None)
          - debug: debug flag (default False)

        Returns (pandas.DataFrame) particle table with additional columns
        (self.box_center_cols, self.box_min_cols, self.box_max_cols)
        """

        # initialize box table
        box = pd.DataFrame(
            columns=(self.box_center_cols + self.box_min_cols
                     + self.box_max_cols))
    
        # read glyph data and convert it to float so it transforms better 
        if isinstance(glyph, str):
            glyph_image = pyto.core.Image.read(glyph, header=True)
            glyph = glyph_image.data
        glyph = glyph.astype(float)
     
        # initial and final glyph center and shape
        glyph_shape_init = np.asarray(glyph.data.shape)
        center_init = glyph_shape_init // 2
        scale = glyph_bin / map_bin
        glyph_shape = glyph_shape_init * scale
        glyph_shape = np.round(glyph_shape).astype(int)
        center = glyph_shape // 2
        glyph_slice = [
            slice(start, stop) for start, stop 
            in zip(center_init - glyph_shape // 2,
                   center_init + glyph_shape // 2)]
        glyph_slice_origin = [slice(0, stop) for stop in glyph_shape]
    
        deg_to_rad = 1
        if self.degree:
            deg_to_rad = np.pi / 180

        # initialize map image
        data = np.zeros(map_shape, dtype=dtype)
        image = pyto.core.Image(data=data)

        # place glyphs in map image
        for index, row in params.iterrows():
            particle_id = row[self.particle_id_star_col]
        
            # rotate and scale glyph
            angles = (
                np.array([row[ang] for ang in self.angle_cols]) * deg_to_rad)
            r = Rigid3D.make_r_euler(angles=angles, mode=self.euler_mode)
            r3d_rot = Rigid3D(q=r)
            rot_glyph = r3d_rot.transformArray(
                array=glyph, center=center_init, order=spline_order)
            r3d_scale = Rigid3D(q=np.identity(3), scale=scale)
            transf_glyph = r3d_scale.transformArray(
                array=rot_glyph, order=spline_order)
            transf_glyph = transf_glyph[*glyph_slice_origin]
            transf_glyph_image = pyto.core.Image(
                data=transf_glyph.astype(dtype))
        
            # threshold glyph and convert type
            if threshold is not None:
                transf_glyph = transf_glyph > threshold
            if dtype is not None:
                transf_glyph = transf_glyph.astype(dtype)

            # get particle coords and calculate glyph (box) corner coords
            coords = np.array([row[x] for x in self.coords_map]) 
            coords = np.round(coords).astype(int)
            l_corner_coords = coords - center
            r_corner_coords = l_corner_coords + glyph_shape
            box.loc[index, self.box_center_cols] = coords
            box.loc[index, self.box_min_cols] = l_corner_coords
            box.loc[index, self.box_max_cols] = r_corner_coords

            # save transformed glyph
            if glyph_out_prefix is not None:
                os.makedirs(os.path.dirname(glyph_out_prefix), exist_ok=True)
                path = glyph_out_prefix + f"particle-{particle_id}.mrc"
                transf_glyph_image.write(file=path, pixel=map_pixel_nm)
                print(f"Wrote transformed glyph (particle id {particle_id}) "
                      + f"at {map_path}")

            # make transformed glyph image object and adjust to map size
            transf_glyph_image.inset = [
                slice(start, stop) for start, stop
                in zip(l_corner_coords, r_corner_coords)]
            if not image.isInside(inset=transf_glyph_image.inset):
                intersect = transf_glyph_image.findIntersectingInset(
                    inset=image.inset)
                transf_glyph_image.useInset(inset=intersect, mode='absolute')
                if debug:
                    print(
                        f"Glyph particle_id {particle_id} (table index {index})"
                        + f" cut to fit "
                        + f"inside map and placed at position {coords}.")

            # check if glyphs overlap
            if data[*transf_glyph_image.inset].max() > 0:
                if overlap_ok:
                    if debug:
                        print(
                            f"Glyph particle_id {particle_id} (table index "
                            + f"{index}) placed at position {coords}, "
                            + f"although it overlaps with another glyph.")
                else:
                    if debug:
                        print(
                            f"Glyph particle_id {particle_id} (table index "
                            + f"{index}) was not placed at position {coords} "
                            + f"because it overlaps with another glyph.")
                    continue

            # place glyph in map
            data[*transf_glyph_image.inset] += transf_glyph_image.data

        # save map image
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        image.write(file=map_path, pixel=map_pixel_nm)
        print(f"Wrote mapped glyphs tomo at {map_path}")

        # add corners to table
        table = pd.concat([params, box], axis=1)

        return table
