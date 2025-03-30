"""
Methods that filter, smooth and randomize pixels in particles.

# Author: Vladan Lucic
# $Id$
"""

__version__ = "$Revision$"

import abc
import os
import re

import numpy as np
from numpy.random import default_rng
import scipy as sp

import pyto
from pyto.util.pandas_plus import merge_left_keep_index
from ..geometry.rigid_3d import Rigid3D
from .set_path import SetPath


class ExtractMPSFilter(abc.ABC):
    """Abstract class that provides particle filtering  methods.

    Meant to be inherited by ExtractMPS.
    """

    @abc.abstractmethod
    def __init__(self):
        pass            
    
    def filter_particles_task(
            self, mps, sigma_nm, name_init, name_filtered,
            fun=sp.ndimage.gaussian_filter, fun_kwargs={}, 
            verbose=True, star_comment='Gauss low-pass filtered'):
        """Filters particles, saves them and write star files

        Arguments:

          - name_init: name of the initial particles name, specifies 
          directory whare the particles are located
          - name_filtered:  name of the filtered particles name, specifies 
          directory whare the filtered particles are written
        """

        paths = pyto.particles.extract_mps.Paths(
            name=name_filtered, root_template=self.root_template,
            size=self.box_size)

        # filter particles
        mps_filt = self.filter_particles(
            mps=mps, fun=fun, fun_kwargs=fun_kwargs,
            sigma_nm=sigma_nm,
            particle_col=mps.tomo_particle_col, 
            in_particle_col=self.in_tomo_particle_col, 
            pattern=name_init, replace=name_filtered)
        if verbose:
            print("All particles:")
            print(f"Wrote particles to {paths.root}/{name_filtered}")
        mps_filt.name = paths.name
        mps_filt.in_tomo_particle_col = self.in_tomo_particle_col

        # save filtered mps
        mps_filt.write(path=paths.mps_path, verbose=verbose)

        # make star file
        self.make_star(
            mps=mps_filt, labels=self.get_labels(mps_filt),
            star_path=paths.star_path, comment=star_comment, 
            verbose=verbose)

        # make star file for each tether subclass
        if len(self.class_code) > 0:
            if verbose:
                print(f"\nIndividual classes of {name_filtered}:")
            self.split_star(
                mps=mps_filt,
                class_names=self.class_names, class_code=self.class_code,
                labels=self.get_labels(mps), 
                star_path=paths.star_path, star_comment=star_comment,
                verbose=verbose)

    def smooth_regions_task(
            self, region_other, region_ids, name_init, name_smooth=None,
            name_regions=None, prefix='', star_comment='', verbose=True):
        """Selectively filters regions of particle images.
        
        Takes initial particle images (arg name_init) and replaces 
        previously defined regions by the corresponding parts of 
        other particle set(s). The regions are particle-dependent and 
        have to be previously defined (and saved as another particle set). 

        For example, if regions show membranes and the other
        particle sets are obtined by filtering (e.g. smoothing) the
        initial particle set, the result is the initial particle set
        where membranes are selectively filtered.

        If arg name_regions is not None, it us used to determine the 
        region image paths. Otherwise, it is read from column 
        self.region_particle_col of particles table. 

        Args regio_other and region_ids determined which part of 
        the initial particle images is replaced by which "other"
        particle sets. For example, given:
          region_other = {
              'membrane': slightly_filtered_set_name,
              'vesicle': strongly_filtered_set_name}
          region_ids = {'membrane': 2, 'vesicle': 5}
        part of the initial particle image that correspond to 
        region_image=2 will be replaced by the corresponding part of 
        the corresponding image of slightly_filtered_set_name, and 
        region_image=5 by vesicle_set_name.

        The name of the resulting particle image set can be specified by
        arg name_smooth. Alternatively, it name_smooth is None,
        it is composed of the name of arg prefix and the
        name derived from arg region_other (see make_smooth_name() doc).

        Copies the current particle paths (column 
        mps_orig.tomo_particle_col of particles table) to 
        self.in_tomo_particle_col). Before that, removes the previous 
        values of column self.in_tomo_particle_col. The resulting
        particle paths are then saved in column self.tomo_particle_col.

        The MultiParticleSets object containing the resulting (randomized) 
        particle sets is saved. In addition writes star files for all 
        randomized particles as well as for the classes of this particle 
        set (defined by self.class_names and self.class_code). Star
        file labels are written using self.get_labels().

        Arguments:
          - name_init: initial particle image set name
          - name_smooth: name of the final particle image set
          - name_regions: name of the regions image set (default None)
          - region_other: (dict) region name (str): other image set name
          - region_ids: (dict) region name (str): region id (int) where 
          - prefix: used to compose the name of the final set if 
          name_smooth is None (default '')
          - star_comment: comment written at the beginning of star files
        """

        # make paths for smooth images
        if name_smooth is None:
            name_smooth = self.make_smooth_name(
                region_other=region_other, prefix=prefix)
        paths_smooth = pyto.particles.extract_mps.Paths(
            name=name_smooth, root_template=self.root_template,
            size=self.box_size)

        # read input particles
        paths_orig = pyto.particles.extract_mps.Paths(
            name=name_init, root_template=self.root_template,
            size=self.box_size)
        mps_orig = pyto.spatial.MultiParticleSets.read(path=paths_orig.mps_path)
        mps_orig.name = name_init

        # smooth
        mps_smooth = self.make_region_smooth(
            mps=mps_orig, fun=self.combine_images, region_other=region_other,
            region_ids=region_ids, paths=paths_smooth,
            name_regions=name_regions, particle_col=mps_orig.tomo_particle_col, 
            in_particle_col=self.in_tomo_particle_col, prefix=prefix)
        if verbose:
            print("\nAll particles:")
            print(f"Wrote particles to {paths_smooth.particles_dir}")
 
        # save smooth mps
        mps_smooth.write(path=paths_smooth.mps_path, verbose=verbose)

        # make star file
        self.make_star(
            mps=mps_smooth, labels=self.get_labels(mps_smooth),
            star_path=paths_smooth.star_path, comment=star_comment, 
            verbose=verbose)

        # make star file for each tether subclass
        if len(self.class_code) > 0:
            if verbose:
                print(f"\nIndividual classes of {name_smooth}:")
            self.split_star(
                mps=mps_smooth, labels=self.get_labels(mps_orig), 
                class_names=self.class_names, class_code=self.class_code,
                star_path=paths_smooth.star_path, star_comment=star_comment, 
                verbose=verbose)

    def randomize_task(
            self, name_init, name_random,  mask, mask_mode='image',
            name_segment=None, star_comment='Randomized',
            verbose=True, test=False):
        """Randomize particle images outside a mask.

        Reads particles specified by arg name_init, imposes the specified
        mask(s) to randomize particle pixels outside the mask and
        writes the randomized particles in a location specified by arg 
        name_random.

        Arg mask_mode specified the form of arg mask, it can be:
          - 'image': arg mask is a single image that is used as a mask for
          all particles
          - 'path_col': each particle has the corresponding mask (or 
          segment), the mask particle set is specified by arg name_segment
          and arg mask is the name of particles table column
          where paths to the mask set are written 
        See randomize() docs for more info about both modes.

        The MultiParticleSets object containing the resulting (randomized) 
        particle sets is saved. In addition writes star files for all 
        randomized particles as well as for the classes of this particle 
        set (defined by self.class_names and self.class_code). Star
        file labels are written using self.get_labels().

        Arguments:
          - name_init: name of the particle set to be randomozed
          - name_random: name of the randomized particle set
          - mask: mask image (ndarray) or column name depending on arg 
          mask_mode
          - mask_mode: specified the form of arg mask, 'image' (default)
          or 'path_col'
          - name_segment: name of the particle set comprising 
          particle-specific masks (segment), used only for 'path_col' mode
          - star_comment: comment written at the beginning of star files
          - test: if True, instead of randomizing sets pixels to 0
          (default False)
        """

        # read input particles
        paths_init = pyto.particles.extract_mps.Paths(
            name=name_init, root_template=self.root_template,
            size=self.box_size)
        mps_init = pyto.spatial.MultiParticleSets.read(path=paths_init.mps_path)
        mps_init.name = name_init

        # paths for randomized 
        paths_random = pyto.particles.extract_mps.Paths(
            name=name_random, root_template=self.root_template,
            size=self.box_size)

        # add segment col from segment to particles
        if mask_mode == 'path_col':
            paths_segment = pyto.particles.extract_mps.Paths(
                name=name_segment, root_template=self.root_template,
                size=self.box_size)
            mps_segment = pyto.spatial.MultiParticleSets.read(
                path=paths_segment.mps_path, verbose=verbose)
            mps_init.particles.drop(
                columns=mask, errors='ignore', inplace=True)
            common_cols = [mps_init.tomo_id_col, mps_init.particle_id_col]
            seg =  mps_segment.particles[common_cols + [mask]]
            mps_init.particles = merge_left_keep_index(
                mps_init.particles, seg, on=common_cols, validate='one_to_one')
            
        if self.rng is None:
            self.rng = default_rng(seed=self.seed)

        mps_rand = self.randomize(
            mps=mps_init, mask=mask, mask_mode=mask_mode,
            particle_col=mps_init.tomo_particle_col,
            in_particle_col=self.in_tomo_particle_col, 
            pattern=name_init, replace=name_random, test=test)
        if verbose:
            print("\nAll particles:")
            print(f"Wrote particles to {paths_random.particles_dir}")
        mps_rand.name=paths_random.name
        mps_rand.in_tomo_particle_col = self.in_tomo_particle_col

        # save filtered mps
        mps_rand.write(path=paths_random.mps_path, verbose=verbose)

        # make star file
        self.make_star(
            mps=mps_rand, labels=self.get_labels(mps_rand),
            star_path=paths_random.star_path, comment=star_comment, 
            verbose=verbose)

        # make star file for each tether subclass
        if len(self.class_code) > 0:
            if verbose:
                print(f"\nIndividual classes of {name_random}:")
            self.split_star(
                mps=mps_rand, labels=self.get_labels(mps_rand), 
                class_names=self.class_names, class_code=self.class_code,
                star_path=paths_random.star_path, star_comment=star_comment, 
                verbose=verbose)
         
    @classmethod
    def filter_particles(
            cls, mps, fun, fun_kwargs, sigma_nm,
            particle_col, in_particle_col, pattern, replace):
        """Filters particles

        """

        # copy mps and add column for the new particle paths
        mps_2 = mps.copy()
        mps_2.particles.rename(
            columns={particle_col: in_particle_col}, inplace=True)
        mps_2.particles[particle_col] = mps_2.particles.apply(
            lambda x: re.sub(pattern, replace, x[in_particle_col]), 
            axis=1)

        # loop over particles
        for p_ind, row in mps_2.particles.iterrows():

            # convert sigma to pixels
            pixel_nm = row[mps_2.pixel_nm_col]
            if sigma_nm is not None:
                sigma_pix = sigma_nm / pixel_nm
                fun_kwargs.update({'sigma': sigma_pix})

            # read, filter, write
            in_path = row[in_particle_col]
            out_path = row[particle_col]
            try:
                pyto.grey.Image.modify(
                    old=in_path, new=out_path, fun=fun,
                    fun_kwargs=fun_kwargs, pass_data=True)
            except FileNotFoundError:
                out_dir = os.path.dirname(out_path) 
                os.makedirs(out_dir)
                pyto.grey.Image.modify(
                    old=in_path, new=out_path, fun=fun,
                    fun_kwargs=fun_kwargs, pass_data=True)

        return mps_2

    @classmethod
    def make_smooth_name(cls, region_other,  prefix='particles_'):
        """Make out name for particles with smoothed membranes

        Example:
          prefix = ''
          region_other = {'membrane: 'set1', 'vesicle': 'set2'}
        results in: 'membrane-set1_vesicle-set2'

        Example with prefix:
          prefix = 'particles'
          region_other = {
              'membrane: 'particles_set1', 'vesicle': 'particles_set2'}
        results in: 'particles_membrane-set1_vesicle-set2'

        """
        name = prefix + '_'.join(
            [f"{reg}-{filt_name.removeprefix(prefix)}" 
             for reg, filt_name in region_other.items()])
        return name

    @classmethod
    def combine_images(cls, image, region_other, region_ids, region_image):
        """Replaces parts of an image by other images at specified regions.
        
        Starting from an initial image (arg image) replaces parts defined
        by one or more regions (args region_image and  region_other.keys()) 
        by other images (the corresponding region.values()).

        Region names (keys of region_other and region_ids) can be any 
        strings. However, keys of args region_other have to be a subset 
        of region_ids keys.

        For example, if:
          region_other = {
              'membrane': slightly_filtered_image,
              'vesicle': strongly_filtered_image}
          region_ids = {'membrane': 2, 'vesicle': 5}
        parts of image that correspond to region_image=2 will be replaced
        by the corresponding part of slightly_filtered_image and 
        region_image=5 by vesicle_image.

        Initial, replacement and region images have to have the same shape.

        Arguments:
          - image: (ndarray) initial image
          - region_other: (dict) region name (str): other image (ndarray)
          - region_ids: (dict) region name (str): region id (int) 
          - region_image: (ndarray) region image

        Returns: modified image (ndarray)
        """

        data = image.data
        for reg, filt_image in region_other.items():
            data = np.where(
                region_image.data==region_ids[reg], filt_image.data, data)
        return data

    def make_region_smooth(
            self, mps, region_other, fun, region_ids, paths, 
            particle_col, in_particle_col, name_regions=None,
            pattern=None, replace=None, prefix='particles_'):
        """Combines two sets of images based on another (regions) set.

        Developed to process the initial particle set images so that 
        parts corresponding to selected regions (such as plasma and 
        vesicle membranes) are replaced by the corresponding parts 
        of one or more filtered particle set images. This is achieved
        by specifying fun=self.combine_images (see combine_images()
        for more details.

        More generally, given three (or more) sets of particle images
        that correspond to each other:
          - initial particle set images
          - regions (segmented) particle set images
          - one or more other particle set images
        and the set of rules that specify the corresponence between 
        labels on regions set and the other sets (args):
          - region_other
          - region_id
        applies the specified function (arg fun) that takes the above as 
        arguments and returns a modified version of the initial particle
        images.

        The processing function has to have the following signature:
          fun(initial_image, *, region_other, region_ids, region_image)
        and returns a modified image.

        The region particle set paths are determined from regions set name 
        (arg name_regions) if specified, with the help of arg paths. 
        Alternatively, it is read from column self.region_particle_col 
        of particles table.

        The paths to (each of) the other (e.g. filtered) particle set(s) are 
        determined from the general information provided by arg paths
        (specifically paths.root_template and paths.size) and the name 
        of the particle set derived from arg region_other() (see 
        make_smooth_name() docs).

        Copies the current particle paths (column specified by arg 
        particle_col of particles table) to (arg) in_particle_col. 
        Before that, removes the previous values of column (arg) 
        in_particle_col. The resulting particle paths are then 
        saved in column particle_col.

        Arguments:
          - mps: (MultiParticleSets) particle set object
          - region_other: (dict) region name (str): other image set name
          - region_ids: (dict) region name (str): region id (int) where 
          all keys of region_other are present in region_ids 
          - fun: function that processes images
          - name_regions: name of the regions image set (default None)
          - paths: (extract_mps.Paths) obect that contains some info
          needed to determine the paths to the regions and other particle 
          image sets (pickled MultiParticleSets) 
          - particle_col: column of particles table that initially 
          contains paths to particle images that should be processed,
          and at the end contains the (final) processed particle image
          paths
          - in_particle_col: column of particles table where the initial 
          vaules of the column particle_col are saved
          - pattern: part of the initial particle paths that is replaced
          to make the randomized particle paths
          - replace: the replacement part of the randomized partcle paths
          - prefix: (default 'particles_')
        """

        # make new name
        name = self.make_smooth_name(region_other=region_other,  prefix=prefix)
        if pattern is None:
            pattern = mps.name
        if replace is None:
            replace = name

        # copy mps, remove old in_particle_tomo and setup new particle paths
        mps_smooth = mps.copy()
        mps_smooth.particles.drop(
            columns=in_particle_col, errors='ignore', inplace=True)
        mps_smooth.particles.rename(
            columns={particle_col: in_particle_col}, inplace=True)
        mps_smooth.particles[particle_col] = mps_smooth.particles.apply(
            lambda x: re.sub(pattern, replace, x[in_particle_col]), 
            axis=1)

        # get all filtered MPSs
        region_mps = {}
        for reg, filt_name in region_other.items():
            paths_filt = pyto.particles.extract_mps.Paths(
                name=filt_name, root_template=paths.root_template,
                size=paths.size)
            mps_filt = pyto.spatial.MultiParticleSets.read(
                path=paths_filt.mps_path)
            region_mps[reg] = mps_filt

        # fugure out regions paths and save them in regions column 
        if name_regions is not None:
            paths_regions = pyto.particles.extract_mps.Paths(
                name=name_regions, root_template=paths.root_template,
                size=paths.size)
            mps_regions = pyto.spatial.MultiParticleSets.read(
                path=paths_regions.mps_path)
            mps_smooth.particles[mps.region_particle_col] = \
                mps_regions.particles[mps_regions.region_particle_col]
            
        # loop over particles
        for p_ind, row in mps_smooth.particles.iterrows():

            # read
            in_path = row[in_particle_col]
            out_path = row[particle_col]
            region_path = row[mps.region_particle_col]
            region_image = pyto.segmentation.Labels.read(
                region_path, memmap=True)

            # get filtered images
            region_filtered = {}
            for reg, mps_local in region_mps.items():
                filt_path = mps_local.particles.loc[p_ind, particle_col]
                filt_image = pyto.core.Image.read(filt_path)
                region_filtered[reg] = filt_image

            # make function arguments
            fun_kwargs = {
                'region_image': region_image,
                'region_other': region_filtered, 
                'region_ids': region_ids}

            # modify and write image
            try:
                pyto.grey.Image.modify(
                    old=in_path, new=out_path, fun=fun, fun_kwargs=fun_kwargs)
            except FileNotFoundError:
                out_dir = os.path.dirname(out_path) 
                os.makedirs(out_dir)
                pyto.grey.Image.modify(
                    old=in_path, new=out_path, fun=fun, fun_kwargs=fun_kwargs)

        return mps_smooth

    def randomize(
            self, mps, mask, particle_col, in_particle_col, pattern, replace,
            mask_mode='image', test=False):
        """Randomize particles outside a particle dependent mask.

        Meant for the following scenarios that are chosen by arg mask_mode:
          - 'image': intended for membrane bound particles, uses a single 
          mask, which is rotated according to each particle separately
          - 'path_col': each particle has its specific mask and the
          masks are already aligned with their respective particles

        In 'image' mask mode, the mask (image) is defined by arg mask. 
        The randomization is executed as follows:
          - For each particle, spherical angles theta and phi angles 
          that define its orientation are read from particles table 
          columns self.normal_angle_cols. 
          - The mask is rotated from the initial mask orientation 
          (phi=theta=0) to the particle orientation around the particle 
          image center (at particle_size // 2)
          - The particle image pixels outside the mask are randomized.
        Consequently, the arg mask should be an image and the mask should 
        be rotationally symmetrical around axis phi=theta=0. 

        Importantly, in the 'image' mask mode, all particles should be
        aligned when roatated back to the initial orientation 
        (phi=theta=0).

        In the 'path_col' mode, arg mask is the name of particles
        column that contain paths to the (particle dependent) mask
        images. The randomization is simply:
          - The particle image pixels outside the particle dependent
          mask are randomized.

        Reads particle images and possibly particle specific masks 
        (in 'path_col' mask mode) and writes the resulting 
        (randomized) particle images. The initial and randomized 
        particles have the same base names but different directory
        path. These are obtained by replacing arg pattern of the initial
        particle paths with arg replace. 

        Copies the current particle paths (column specified by arg 
        particle_col of particles table) to (arg) in_particle_col. 
        Before that, removes the previous values of column (arg) 
        in_particle_col. The new (randomized) particle paths are then 
        saved in column particle_col.

        Arguments:
          - mps: (MultiParticleSets) particle set object
          - mask: single image (ndarray) or column name to particle
          dependent mask images, depending on mask_mode 
          - mask_mode: mask mode, 'image' or 'path_col'
          - particle_col: column of particles table that initially 
          contains paths to particle images that should be randomized,
          and at the end contains the (final) randomized particle image
          paths
          - in_particle_col: column of particles table where the initial 
          vaules of the column particle_col are saved
          - pattern: part of the initial particle paths that is replaced
          to make the randomized particle paths
          - replace: the replacement part of the randomized partcle paths
          - test: if True, instead of randomizing sets pixels to 0
        """

        # copy mps and add column for the new particle paths
        mps_2 = mps.copy()
        mps_2.particles.drop(
            columns=in_particle_col, errors='ignore', inplace=True)
        mps_2.particles.rename(
            columns={particle_col: in_particle_col}, inplace=True)
        mps_2.particles[particle_col] = mps_2.particles.apply(
            lambda x: re.sub(pattern, replace, x[in_particle_col]), 
            axis=1)

        # rotation center
        center_coord = self.box_size // 2 
        rot_center = [center_coord, center_coord, center_coord]
        
        # loop over particles
        for p_ind, row in mps_2.particles.iterrows():

            # get particle specific mask
            if mask_mode == 'image':
             
                # rotate mask
                theta = row[self.normal_angle_cols[0]] * np.pi / 180
                phi = row[self.normal_angle_cols[1]] * np.pi / 180
                r = Rigid3D.make_r_euler(
                    angles=(0, theta, phi), mode='zyz_ex_active')
                r3d = Rigid3D(q=r)
                mask_data = r3d.transformArray(
                    array=mask, center=rot_center, order=0)

            elif mask_mode == 'path_col':

                # get particle corresponding mask
                mask_path = row[mask]
                mask_image = pyto.segmentation.Labels.read(mask_path)
                mask_data = mask_image.data > 0

            # read, filter, write
            in_path = row[in_particle_col]
            out_path = row[particle_col]
            out_dir = os.path.dirname(out_path)
            os.makedirs(out_dir, exist_ok=True)
            fun = self.randomize_single
            fun_kwargs = {'mask': mask_data, 'rng': self.rng, 'test': test}
            pyto.grey.Image.modify(
                old=in_path, new=out_path, fun=fun,
                fun_kwargs=fun_kwargs, pass_data=True)

        return mps_2    
    
    @classmethod
    def randomize_single(cls, image, mask, rng=None, seed=None, test=False):
        """Randomizes image pixels outside the mask.

        Does not modify (arg) image

        Arguments:
          - image (ndarray, or pyto.core.Image where attribute data is
          ndarray): image to be randomised (can be greyscale or segmented)
          - mask: (binary ndarray, or pyto.core.Image where attribute data
          is binary ndarray) mask image
          - rng, seed: np.random ring and seed used to randomize pixels,
          if not specified (default) a new ring is made
          - test: if True, instead of randomizing sets pixels to 0
        """

        image_pyto = False
        if isinstance (image, pyto.core.Image):
            image_data = image.data.copy()
            image_pyto = True
        else:
            image_data = image.copy()
        if isinstance (mask, pyto.core.Image):
            mask = mask.data

        # get out of the mask coords and the respective values
        out_inds = np.nonzero(~mask)
        initial = image_data[out_inds]

        # permute values
        if rng is None:
            rng = default_rng(seed=seed)
        permuted = rng.permutation(initial)
        if not test:
            image_data[out_inds] = permuted
        else:
            image_data[out_inds] = 0

        if image_pyto:
            image.data = image_data
            return image
        else:
            return image_data

