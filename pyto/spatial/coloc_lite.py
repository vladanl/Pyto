"""
Streamlined version of calculating colocalization and analyzing results

# Author: Vladan Lucic
# $Id$
"""

__version__ = "$Revision$"

import os

import numpy as np
from numpy.random import default_rng
import scipy as sp
import pandas as pd 

import pyto
from pyto.io.pandas_io import PandasIO
from . import coloc_functions as col_func
from . import point_pattern
from .particle_sets import ParticleSets 
from .coloc_one import ColocOne
from .coloc_analysis import ColocAnalysis
from .multi_particle_sets import MultiParticleSets


class ColocLite(ColocAnalysis):
    """Streamlined version of calculating colocalization and analyzing results.

    Arguments:
      - pixel_nm: currently not used
    """

    def __init__(
            self, full_coloc=True, keep_dist=False, coloc_mode='less',
            pixel_nm=1, metric='euclidean', dist_mode='fine', p_func=np.greater,
            columns=True, column_factor=2, n_columns_mode='dist',
            seed=None, rng=None, max_iter_rand=100, n_factor=2,
            all_random=False,
            dir_=None, pick=None, dir_prefix='tables_', save_formats=['json'],
            dir_tables='tables', dir_coords='coordinates', dir_suffix=None,
            dir_input_particles='input_particles',
            dir_kept_particles='after_exclusion_particles',
            name_mode='_', join_suffix='data', individual_suffix='data_syn',
            mode=None
            ):
        """Sets attributed from args
        
        The meaning of arg dir is different from the super (ColocAnalysis):
          - here dir/dir_tables is the same as dir in ColocAnalysis
          - if None, output is not written 
        
        Arguments:
          - full_coloc: if True (default), when 3-colocalization is
          calculated by self.colocalize(), the 2-colocalizations between
          the first pattern and each of the other patterns is also calculated  
          - all_random: If False, the standard two simulation types are 
          calculated ('normal' when the first pattern is fixed and 'other'
          where the other patterns are fixed)), if True all sets are
          randomized simultaneously (default False)
          - mode: colocalization tables format (mode), project dependent,
          for example 'munc13_lite'
          - dir_: output directory root, if None output is not written
          - pick: not used
          - dir_prefix: not used
        """

        # colocalization
        self.full_coloc = full_coloc
        self.keep_dist = keep_dist
        self.coloc_mode = coloc_mode
        self.pixel_nm = pixel_nm
        self.metric = metric
        self.dist_mode = dist_mode
        self.p_func = p_func

        # columns
        self.columns = columns
        self.column_factor = column_factor
        self.n_columns_mode = n_columns_mode

        # random
        if rng is None:
            self.rng = default_rng(seed=seed)
        self.seed = seed
        self.n_factor = n_factor
        self.max_iter_rand = max_iter_rand

        # resolve output directory
        self.dir = dir_
        self.dir_suffix = dir_suffix
        self.set_individual_tables(
            dir_tables=dir_tables, dir_coords=dir_coords,
            dir_input_particles=dir_input_particles,
            dir_kept_particles=dir_kept_particles)
            
        # other output related
        self.name_mode = name_mode
        self.mode = mode
        self.join_suffix = join_suffix
        self.individual_suffix = individual_suffix
        self.save_formats = save_formats
        
        # default values of simulation related table column names
        self.set_simulation_suffixes(all_random=all_random)

        #self.data_names = set()
        self.data_names = []

    def set_individual_tables(
            self, dir_tables, dir_coords, dir_input_particles,
            dir_kept_particles):
        """Sets tables, coords and particle dirs.
        """
        
        if (self.dir is not None):
            if self.dir_suffix is not None:
                self.dir = self.dir + '_' + self.dir_suffix
            self.tables_dir = os.path.join(self.dir, dir_tables)
            self.coords_dir = os.path.join(self.dir, dir_coords)
            self.input_particles_dir = os.path.join(
                self.dir, dir_input_particles)
            self.kept_particles_dir = os.path.join(self.dir, dir_kept_particles)
        else:
            self.dir = None
            self.tables_dir = None
            self.coords_dir = None
            self.input_particles_dir = None
            self.kept_particles_dir = None
        
    def set_simulation_suffixes(self, all_random=False):
        """Set attributes defining random simulation variable suffixes.

        Sets the following attributes if all_random is False (default):
          - random_suffixes: ['random', 'random_alt', 'random_combined']
          - simul_suffixes: ['simul_normal', 'simul_other']
          - p_suffixes: ['normal', 'other', 'combined']

        if all_random is True:
          - random_suffixes: ['random']
          - simul_suffixes: ['simul']
          - p_suffixes: ['solo']

        and in both cases:
          - simul_index_col: 'simul_id'

        Argument:
          - all_random: False (default) for the standard two simulation types 
          (called 'normal' and 'other'), and True for simulations where
          all sets are randomized simultaneously
        """
            
        self.all_random = all_random

        # should be the same as in ColocOne
        if not all_random:

            # default, two simulation types
            self.random_suffixes = ['random', 'random_alt', 'random_combined']
            self.simul_suffixes = ['simul_normal', 'simul_other']
            self.p_suffixes = ['normal', 'other', 'combined']
            self.simul_index_col = 'simul_id'

        else:

            # one simulation type, all random
            self.random_suffixes = ['random']
            self.simul_suffixes = ['simul']
            self.p_suffixes = ['solo']
            self.simul_index_col = 'simul_id'
        
    @property
    def _names(self):
        """Data table names (better use self.data_names).

        Just for backcompatibility, self._names is the same as 
        self.data_names except that the former is a list and the latter 
        a set.
        """
        return self.data_names
    
    @_names.setter
    def _names(self, value):
        self.data_names = value

    def copy_setup(self):
        """
        Returns an instance of this class that has the same attributes
        as this instance, except that it does not have any data attribute
        nor self.data_names.
        """

        new = self.__class__()

        # colocalization
        new.full_coloc = self.full_coloc
        new.keep_dist = self.keep_dist
        new.coloc_mode = self.coloc_mode
        new.pixel_nm = self.pixel_nm
        new.metric = self.metric
        new.dist_mode = self.dist_mode
        new.p_func = self.p_func

        # random (likely not needed)
        new.rng = self.rng
        new.seed = self.seed
        new.n_factor = self.n_factor
        new.max_iter_rand = self.max_iter_rand

        # columns
        new.columns = self.columns
        new.column_factor = self.column_factor
        new.n_columns_mode = self.n_columns_mode

        # output directories
        new.dir = self.dir
        new.dir_suffix = self.dir_suffix
        new.tables_dir = self.tables_dir
        new.coords_dir = self.coords_dir

        # other output related
        new.name_mode = self.name_mode
        new.mode = self.mode
        new.join_suffix = self.join_suffix
        new.individual_suffix = self.individual_suffix
        new.save_formats = self.save_formats
         
        # default values of simulation related table column names
        new.random_suffixes = self.random_suffixes
        new.simul_suffixes = self.simul_suffixes
        new.p_suffixes = self.p_suffixes
        new.simul_index_col = self.simul_index_col

        return new

    @classmethod
    def colocalize_run(cls):
        """

        """
        pass

    def colocalize(
            self, particles, distance, n_simul,
            coloc_name=None, coloc_case=None, 
            exclusion=None, hull=False, hull_expand=0, set_names=None,
            log=None):
        """Calculates real and simulated colocalizations for multiple tomos.

        Used for the case when particle sets are available as a pyto object
        pyto.spatial.MultiParticleSets (better) or pyto.spatial.ParticleSets 
        (depreciated, some functionality may not be present) and the 
        colocalization is calculated purely using Pyto. See 
        ColocAnalysis.colocalize_run() for running colocalization using PySeg.

        Colocalization is calculated for all tomos present in the 
        specified (arg) particles. 

        Particles used for colocalization are specified by args 
        particles and coloc_name (which comprises names of particle sets 
        that are used). These particle set names have to be present in
        the column particles.subclass_col of table particles.particles if
        (arg) particles is MultiParticleSets object, or particles.set_name_col
        if (arg) particles is ParticleSets object. That is, particles object 
        can contain particles that belong to particle sets other than those 
        specified by arg coloc_name. 

        This also ensures that index values that may be present in (arg)
        particles, can be directly associated with particles that are found
        (by this method) to form or belong to the calculated colocalizations. 

        For coloc_name = 'foo_xx_ha' and the default constructor values 
        the colocalization results are saved as the following attributes:

        1) 3-colocalization tables (pandas.DataFrames):
          - attributes foo_xx_ha_data for all tomos together and 
          foo_xx_ha_data_syn for results for each tomo separately, both for
          all distances (the suffixes data and data_syn are determined by
          self.join_suffix and self.individual_suffix) 
          - written in self.out_dir/self.tables_dir/foo_xx_ha, file names 
          contain coloc_names, such as foo_xx_ha_data_json.pkl and  
          foo_xx_ha_data_syn_json.pkl, respectively

        2) 2-colocalizationn tables: the same as above, except that these 
        are calculated for 2-colocalizaton between the first and each of the
        other particle sets (so foo_xx and foo_ha)

        In addition, particle sets are saved at various points during 
        the execution, as follows:

        1) Input particles:
          - A single instance of MultiParticleSets that contains 
          all particles that are used for colocalization, as obtained by
          from_mps()
          - Typically, only particles where self.keep_col (in the 
          particles table) are present (from_mps(ignore_keep=False)) 
          - Index in particles table is unique
          - written in self.out_dir/self.input_particles_dir/foo_xx_ha/ in
          the form that can be read by MultiparticleSets.read()

        2) Particles after imposing distance exclusion:
          - Pickled single instance of MultiParticleSets
          - Obtained from the input particles by setting False in 
          self.keep_col particle table column for particles that should be 
          excluded
          - Index in the particles table is unique and corresponds to that 
          of the input particles
          - written in self.out_dir/self.input_particles_dir/foo_xx_ha/ in
          the form that can be read by MultiparticleSets.read()

       3) Coordinates of the particles that define 3-colocs and 2-colocs
          (the first set in arg coloc_name, so foo)
          - 2-colocs are calculated between the first and each of the 
          other particle sets specified by arg coloc_name.
          - Index correspond to the input and particles after exclusion 
          - Index is not unique, because the same particle might be present
          at different colocalization distances
          - written as pandas dataframes in self.out_dir/self.dir_coords/:
            - 3-colocs: foo_xx_ha/coloc3_json.pkl
            - 2-colocs: foo_xx/coloc2_json.pkl and foo_ha/coloc2_json.pkl

        4) Coordinates of particles included in 3- and 2-colocs
          - 2-colocs are calculated between the first and each of the 
          other particle sets specified by arg coloc_name.
          - Index correspond to the input and particles after exclusion 
          - Index is not unique, because the same particle might be present
          at different colocalization distances
         - written as pandas dataframes in self.out_dir/self.dir_coords/:
            - 3-colocs: foo_xx_ha/particles3_json.pkl
            - 2-colocs: foo_xx/particles2_json.pkl and 
            foo_ha/particles2_json.pkl
       
        Exclusion distance is first imposed on real particle sets, provided
        that arg exclusion is not None. The same exclusion distance is then
        applied on all simulated particle sets. It is important to ensure
        that if an exclusion distance was used to generate the particle sets 
        passed to this method, the same or a larger exclusion is used for
        this method.

        Arguments:
          - coloc_name, coloc_case: (the same, one should be specified, 
          coloc_name is preferred, coloc_case is left for backcompatibility)
          colocalization name, contains individual particle set
          names (like 'pre0_tethershort_post2')
          - particles: (ParticleSets, or MultiParticleSets) particle coordinates
          - distance: (list) colocalization distances in nm
          - n_simul: N simulations
          - set_names: (list) particle set names, used only for printing 
          results, if None (default) the names are derived from arg coloc_name
          - hull, hull_expand: not implemented
          - log: path to a log file or an open file
        """

        # check args:
        if (coloc_name is None) and (coloc_case is None):
            raise ValueError(
                "Arg coloc_name (preferable), or coloc_case (depreciated) "
                + "has to be specified.")
        elif coloc_name is None:
            coloc_name = coloc_case
        
        # open log file
        if (log is not None) and isinstance(log, str):
            try:
                self.log = open(log, 'w')
            except FileNotFoundError:
                os.makedirs(os.path.dirname(log), exist_ok=True)
                self.log = open(log, 'w')
        else:
            self.log = log    
           
        # extract particle set names
        if set_names is None:
            set_names = col_func.get_names(name=coloc_name, mode=self.name_mode)
        if self.full_coloc:
            all_cases = [coloc_name] + [
                col_func.make_name(
                    names=([set_names[0]] + [other]), suffix=None)
                for other in set_names[1:]]
        
        # print params
        print("Colocalization parameters:", file=self.log)
        if self.full_coloc and (len(set_names) > 2):
            print(f"    Colocalization cases: {', '.join(all_cases)}",
                  file=self.log)
        else:
            print(f"    Colocalization name: {coloc_name}", file=self.log)
        print("    Colocalization distance [nm]: {} ".format(
            ', '.join(map(str, distance))), file=self.log)
        if self.columns:
            print(f"    Calculate columns: {self.columns}, columns distance "
                  + f"factor {self.column_factor}", file=self.log)
        else:
            print(f"    Calculate columns: {self.columns}", file=self.log)
        print(f"    Exclusion distance [nm]: {exclusion}", file=self.log)
        if hull:
            print(f"    Make convex hull: {hull}, expansion: {hull_expand}",
                  file=self.log)
        else:
            print(f"    Make convex hull: {hull}", file=self.log)
        if self.log is not None:
            print(f"Log file: ", file=self.log)

        # fugure out particles
        if isinstance(particles, ParticleSets):
            mps = MultiParticleSets()
            columns = {particles.set_name_col: mps.subclass_col}
            mps.from_particle_sets(
                particle_sets=particles, columns=columns,
                pixel_to_particles=True)
        elif isinstance(particles, MultiParticleSets):
            mps = particles
            #particles = mps.to_particle_sets(set_name_col=mps.subclass_col)
            
        # print tomo pixel
        print("\nTomo pixel size [nm]", file=self.log)
        #for to, pix in particles.pixel_nm.items():
        #    print(f"    {to}: {pix}", file=self.log)
        for row in mps.tomos[[mps.tomo_id_col, mps.pixel_nm_col]].iterrows():
            print(f"    {row[1][mps.tomo_id_col]}: {row[1][mps.pixel_nm_col]}",
                  file=self.log)
            
        # write input particles
        if self.input_particles_dir is not None:
            mps.write(path=os.path.join(
                self.input_particles_dir, coloc_name, f"{coloc_name}.pkl"),
                      info_fd=self.log)
            
        # initial particle count
        n_initial = mps.group_set_n(
            group_col=mps.tomo_id_col, subclass_col=mps.subclass_col,
            total=True)
        print("N particles initial (before exclusion):", file=self.log)
        print(
            '    ' + n_initial.to_string().replace('\n', '\n    '),
            file=self.log)

        # do particle exclusion and write
        keep = mps.exclude(
            particles=mps.particles, coord_cols=mps.coord_reg_frame_cols,
            #exclusion=exclusion, tomos=particles.tomos,
            exclusion=exclusion, tomos=None,
            set_names=set_names, class_name_col=mps.subclass_col)
        mps.particles[mps.keep_col] = keep
        mps.particles = mps.particles[mps.particles[mps.keep_col]]
        if self.kept_particles_dir is not None:
            mps.write(path=os.path.join(
                self.kept_particles_dir, coloc_name, f"{coloc_name}.pkl"),
                      info_fd=self.log)

        # convert to ParticleSets
        particles = mps.to_particle_sets(
            set_name_col=mps.subclass_col, coord_cols=mps.coord_reg_frame_cols)

        # after exclusion particle count
        n_after = mps.group_set_n(
            group_col=mps.tomo_id_col, subclass_col=mps.subclass_col,
            total=True)
        print("N particles after exclusion:", file=self.log)
        print(
            '    ' + n_after.to_string().replace('\n', '\n    '), file=self.log)
        
        # colocalization for multiple tomos
        self.coloc3_df = None
        self.particles3_df = None
        n_sets = len(set_names)
        self.coloc2_df = (n_sets - 1) * [None]
        self.particles2_df = (n_sets - 1) * [None]
        print("\n", file=self.log)
        for tomo in particles.tomos:
            print(f"Processing tomo {tomo} ...", file=self.log)

            # setup
            pixel_nm = particles.get_pixel_nm(tomo=tomo)
            co = ColocOne(
                full_coloc=self.full_coloc, keep_dist=self.keep_dist,
                mode=self.coloc_mode, pixel_nm=pixel_nm,
                metric=self.metric, columns=self.columns,
                column_factor=self.column_factor,
                n_columns_mode=self.n_columns_mode,
                p_func=self.p_func, dist_mode=self.dist_mode,
                rng=self.rng, max_iter=self.max_iter_rand,
                n_factor=self.n_factor, all_random=self.all_random)

            # run colocalization
            patterns = [
                particles.get_coords(tomo=tomo, set_name=nam, catch=True)
                for nam in set_names]
            regions = [
                particles.get_region(tomo=tomo, set_name=nam)
                for nam in set_names]
            co.make_one(
                patterns=patterns, regions=regions, distance=distance,
                tomo_id=tomo, names=set_names, n_simul=n_simul, exclusion=None,
                exclusion_simul=exclusion)

            # add (all tables of) current tomo data to all tomos data 
            data_names = [
                nam for nam in co.data_names if nam.endswith(co.suffix)]
            for data_nam in data_names:
                local_data = getattr(co, data_nam)
                local_data['pixel_nm'] = pixel_nm
                current_data_name = (
                    data_nam.removesuffix(co.suffix) + self.individual_suffix)
                try:
                    current_data = getattr(self, current_data_name)
                except (AttributeError, NameError):
                    current_data = local_data
                    if current_data_name not in self.data_names:
                        self.data_names.append(current_data_name)
                else:
                    current_data = pd.concat(
                        [current_data, local_data], ignore_index=True)
                setattr(self, current_data_name, current_data)

            # make dataframe of colocalization coords and particles 
            coloc_coords = self.make_coord_tabs_multid(
                bare_multid=co.bare_multid, particles=particles,
                set_names=set_names, tomo=tomo)

            # add to existing
            if len(set_names) > 2:
                if coloc_coords['coloc3'] is not None:
                    self.coloc3_df = pd.concat(
                        [self.coloc3_df, coloc_coords['coloc3']],
                        ignore_index=False)
                    self.particles3_df = pd.concat(
                        [self.particles3_df, coloc_coords['particles3']],
                        ignore_index=False)
            for other_ind in range(n_sets - 1):
                if coloc_coords['coloc2'][other_ind] is not None:
                    self.coloc2_df[other_ind] = pd.concat(
                        [self.coloc2_df[other_ind],
                         coloc_coords['coloc2'][other_ind]],
                        ignore_index=False)
                    self.particles2_df[other_ind] = pd.concat(
                        [self.particles2_df[other_ind],
                         coloc_coords['particles2'][other_ind]],
                        ignore_index=False)

        # make joint tomo data
        print("Making joint tomo tables", file=self.log)
        data_names_cp = self.data_names.copy()
        for data_nam in data_names_cp:
            if not data_nam.endswith(self.individual_suffix):
                continue
            data_syn =  getattr(self, data_nam)
            add_columns, array_columns = col_func.get_aggregate_columns(
                columns=data_syn.columns)
            data = col_func.aggregate(
                data=data_syn, distance=distance, add_columns=add_columns,
                array_columns=array_columns, p_values=True, p_func=self.p_func,
                random_stats=True, random_suff=self.random_suffixes,
                p_suff=self.p_suffixes)
            join_data_nam = (
                data_nam.removesuffix(self.individual_suffix)
                + self.join_suffix)
            setattr(self, join_data_nam, data)
            if join_data_nam not in self.data_names:
                self.data_names.append(join_data_nam)

        # save tables
        if self.tables_dir is not None:
            print(f"\nSaving tables in: {self.tables_dir}", file=self.log)
            for data_nam in self.data_names:
                tables_base = os.path.join(
                    self.tables_dir, coloc_name, data_nam)
                PandasIO.write(
                    table=getattr(self, data_nam), base=tables_base,
                    file_formats=self.save_formats, verbose=False,
                    info_fd=self.log)
                
        # write colocalization and particles coords as dataframes
        if self.coords_dir is not None:
            print(f"\nSaving coordinates under: {self.coords_dir}",
                   file=self.log)
            self.write_coloc_coords(coloc_name=coloc_name)

        # write colocalization and particles coords as star files?
            
        if self.log is not None:
            self.log.close()
                       
    @classmethod
    def extract_multi(cls):
        """Should not be used in this class
        """

        raise ValueError("This method should not be used in this class")

    def make_coord_tabs_multid(
            self, bare_multid, particles, set_names, tomo):
        """Puts colocalization coordinates in dataframes for multiple distances.

        Makes dataframes that contain coordinates of particles that define
        2- and 3-colocalizations and coordinates of particles that belong
        to these colocalizations. Each dataframe conatains data for 
        colocalizationz obrained for multiple distances.

        Values of arg bare_multid, and args particles and set_names have to
        be consistent, please see make_coord_tabs() doc for more info. 

        Arguments:
          - bare_multid: (dict) colocalization results for multiple distances,
          where keys are distances (in nm) and values are the corresponding
          colocalizations (BareColoc instances)
          - particles: (ParticlesSets) particle coordinates
          - set_names: (list) particle set names 
          - tomo: tomo name

        Returns (dict) containing particle coordinates that:
          - 'coloc3': dataframe that defines 3-colocalizations
          - 'particles3': dataframe containing particles that belong to 
          3-colocalizations
          - 'coloc2: list of dataframes that define 2-colocalizations
          - 'particles2: list of dataframes containing particles that belong
          to 2-colocalizations
        Note: if arg bare contains only 2-colocalizations, only the last 
        two of the above key, value pairs are set and these are list of 
        length 1.

        """

        coloc3_df = None
        particles3_df = None
        n_sets = len(set_names)
        coloc2_df = (n_sets - 1) * [None]
        particles2_df = (n_sets - 1) * [None]
        for di, bare in bare_multid.items():

            # get data for each distance separately
            coords_1d = self.make_coord_tabs(
                 bare=bare, particles=particles, set_names=set_names,
                 tomo=tomo, distance=di)

            # 3-colocs
            if len(set_names) > 2:
                try:
                    coloc3_df = pd.concat(
                        [coloc3_df, coords_1d['coloc3']], ignore_index=False)
                except ValueError:
                    coloc3_df = None
                try:
                    particles3_df = pd.concat(
                        [particles3_df, coords_1d['particles3']],
                        ignore_index=False)
                except ValueError:
                    particles3_df = None
            else:
                coloc3_df = None
                particles3_df = None
                
            # 2-colocs
            for other_ind in range(n_sets - 1):
                try:
                    coloc2_df[other_ind] = pd.concat(
                        [coloc2_df[other_ind], coords_1d['coloc2'][other_ind]],
                        ignore_index=False)
                except ValueError:
                    coloc2_df[other_ind] = None
                try:
                    particles2_df[other_ind] = pd.concat(
                        [particles2_df[other_ind],
                         coords_1d['particles2'][other_ind]],
                        ignore_index=False)
                except ValueError:
                    particles2_df[other_ind] = None
                
        result = {'coloc3': coloc3_df, 'particles3': particles3_df,
                  'coloc2': coloc2_df, 'particles2': particles2_df}
        return result
        
    def make_coord_tabs(self, bare, particles, set_names, tomo, distance):
        """Puts colocalization coordinates for one distance in dataframes.

        Makes dataframes that contain coordinates of particles that define
        2- and 3-colocalizations and coordinates of particles that belong
        to these colocalizations.

        Colocalization results contained in arg bare have to be obtained
        from particle sets specified by arg set_names, using particle sets
        given in arg particles. That is, particle sets obtained by:

          particle_coords = [
              particles.get_coords(tomo=tomo, set_name=nam)
              for nam in set_names]
        
        have to be (previously) used to calculate colocalizations as:

          bare = BareColoc()
          bare.calculate_distances(patterns=particle_coords)
        
        In this way, the number of particles (in arg particles) for  any 
        of the specified particle sets (arg set_names) and tomo (arg tomo)
        has to be the same as in the number of the corresponding particles
        in the calculated colocalization (arg bare). This makes it possible 
        to pass index from particles to the datafames generated here.

        However, arg particles can contain particle sets and tomograms other
        than those specified here (args set_names and tomo) and used to 
        make colocalization (arg bare).

        Arguments:
          - bare: (BareColoc) colocalization results
          - particles: (ParticlesSets) particle coordinates, has to
          contain index (see ParticleSets .get/set_index, ._index)
          - set_names: (list) particle set names 
          - tomo: tomo name
          - distance: colocalization distance [nm]

        Returns (dict) containing particle coordinates that:
          - 'coloc3': dataframe that defines 3-colocalizations
          - 'particles3': dataframe containing particles that belong to 
          3-colocalizations
          - 'coloc2: list of dataframes that define 2-colocalizations
          - 'particles2: list of dataframes containing particles that belong
          to 2-colocalizations
        Note: if arg bare contains only 2-colocalizations, only the last 
        two of the above key, value pairs are set and these are list of 
        length 1.
        """

        if len(set_names) > 2:
        
            # 3-colocalization
            coloc3_coords_all = particles.get_coords(
                tomo=tomo, set_name=set_names[0])
            coloc3_index_all = particles.get_index(
                tomo=tomo, set_name=set_names[0])
            try:
                coloc3_coords = coloc3_coords_all[bare.coloc3]
                coloc3_index = coloc3_index_all[bare.coloc3]
            except (TypeError, IndexError):
                coloc3_df = None
            else:
                coloc3_ps = ParticleSets()
                coloc3_ps.set_coords(
                    tomo=tomo, set_name='coloc3', value=coloc3_coords)
                coloc3_ps.set_index(
                    tomo=tomo, set_name='coloc3', value=coloc3_index)
                coloc3_ps.set_pixel_nm(
                    tomo=tomo, value=particles.get_pixel_nm(tomo=tomo))
                coloc3_df = coloc3_ps.data_df
            if coloc3_df is not None:
                cols3 = [
                    co for co in ['distance'] + coloc3_df.columns.tolist()
                    if co not in [coloc3_ps.region_path_col]]
                coloc3_df['distance'] = distance
                coloc3_df = coloc3_df[cols3].copy()

            # particles in 3d colocalizations
            particles3_ps = ParticleSets()
            for set_ind in range(len(set_names)):
                particles3_coords_all = particles.get_coords(
                    tomo=tomo, set_name=set_names[set_ind])
                particles3_index_all = particles.get_index(
                    tomo=tomo, set_name=set_names[set_ind])
                try:
                    particles3_coords = particles3_coords_all[
                        bare.particles3[set_ind]]
                    particles3_index = particles3_index_all[
                        bare.particles3[set_ind]]
                except (TypeError, IndexError):
                    pass
                else:
                    particles3_ps.set_coords(
                        tomo=tomo, set_name=set_names[set_ind],
                        value=particles3_coords)
                    particles3_ps.set_index(
                        tomo=tomo, set_name=set_names[set_ind],
                        value=particles3_index)
                    particles3_ps.set_pixel_nm(
                        tomo=tomo, value=particles.get_pixel_nm(tomo=tomo))
            particles3_df = particles3_ps.data_df
            if particles3_df is not None:
                cols3 = [
                    co for co in (
                        ['distance'] + particles3_df.columns.tolist())
                    if co not in [particles3_ps.region_path_col]]
                particles3_df['distance'] = distance
                particles3_df = particles3_df[cols3].copy()

        else:
            coloc3_df = None
            particles3_df = None
            
        # 2-colocalizations and particles
        coloc2_df = []
        particles2_df = []
        for other_ind in range(1, len(set_names)):

            # 2-colocalizations
            try:
                coloc2_coords = particles.get_coords(
                    tomo=tomo, set_name=set_names[0])[bare.coloc2[other_ind-1]]
                coloc2_index = particles.get_index(
                    tomo=tomo, set_name=set_names[0])[bare.coloc2[other_ind-1]]
            except (TypeError, IndexError):
                coloc2_df_local = None
            else:
                coloc2_ps = ParticleSets()
                coloc2_ps.set_coords(
                    tomo=tomo, set_name='coloc2', value=coloc2_coords)
                coloc2_ps.set_index(
                    tomo=tomo, set_name='coloc2', value=coloc2_index)
                coloc2_ps.set_pixel_nm(
                    tomo=tomo, value=particles.get_pixel_nm(tomo=tomo))
                coloc2_df_local = coloc2_ps.data_df
            if coloc2_df_local is not None:
                cols2 = [
                    co for co in ['distance'] + coloc2_df_local.columns.tolist()
                    if co not in [coloc2_ps.region_path_col]]
                coloc2_df_local['distance'] = distance
                coloc2_df_local = coloc2_df_local[cols2].copy()
            coloc2_df.append(coloc2_df_local)

            # particles in 2-colocs
            particles2_ps = ParticleSets()
            try:
                particles2_coords_0 = (
                    particles.get_coords(tomo=tomo, set_name=set_names[0])
                    [bare.particles2[other_ind-1][0]])
                particles2_index_0 = (
                    particles.get_index(tomo=tomo, set_name=set_names[0])
                    [bare.particles2[other_ind-1][0]])
            except (TypeError, IndexError):
                pass
            else:
                particles2_ps.set_coords(
                    tomo=tomo, set_name=set_names[0],
                    value=particles2_coords_0)
                particles2_ps.set_index(
                    tomo=tomo, set_name=set_names[0],
                    value=particles2_index_0)
            try:
                particles2_coords_1 = (
                    particles.get_coords(
                        tomo=tomo, set_name=set_names[other_ind])
                    [bare.particles2[other_ind-1][1]])
                particles2_index_1 = (
                    particles.get_index(
                        tomo=tomo, set_name=set_names[other_ind])
                    [bare.particles2[other_ind-1][1]])
            except (TypeError, IndexError):
                pass
            else:
                particles2_ps.set_coords(
                    tomo=tomo, set_name=set_names[other_ind],
                    value=particles2_coords_1)
                particles2_ps.set_index(
                    tomo=tomo, set_name=set_names[other_ind],
                    value=particles2_index_1)
            particles2_ps.set_pixel_nm(
                tomo=tomo, value=particles.get_pixel_nm(tomo=tomo))
            particles2_df_local = particles2_ps.data_df
            if particles2_df_local is not None:
                cols2 = [
                    co for co in (
                        ['distance'] + particles2_df_local.columns.tolist())
                    if co not in [particles2_ps.region_path_col]]
                particles2_df_local['distance'] = distance
                particles2_df_local = particles2_df_local[cols2].copy()
            particles2_df.append(particles2_df_local)

        result = {'coloc3': coloc3_df, 'particles3': particles3_df,
                  'coloc2': coloc2_df, 'particles2': particles2_df}
        return result

    def write_coloc_coords(self, coloc_name):
        """Writes subcolumn and particle coordinates in dataframes.

        Reguires attributes:
          - self.coloc3_df, self.particles3_df: coordinates of 3-colocalizations
          and of particles in the 3-colocalizations, respectively (needed only 
          for 3-colocalizations)
          - self.coloc2_df, self.particles2_df: coordinates of 2-colocalizations
          and of particles in the 2-colocalizations, respectively (needed both 
          for 3- and 2-colocalizations)
        """

        df_io = PandasIO(file_formats=self.save_formats, info_fd=self.log)

        set_names = col_func.get_names(name=coloc_name, mode=self.name_mode)
        coloc_names = col_func.make_full_coloc_names(
            names=set_names, suffix=None, mode=self.name_mode)
        if len(set_names) > 2:
            base = os.path.join(self.coords_dir, coloc_name, 'coloc3')
            df_io.write_table(
                table=self.coloc3_df, base=base, out_desc='3-colocalizations')
            base = os.path.join(self.coords_dir, coloc_name, 'particles3')
            df_io.write_table(
                table=self.particles3_df, base=base,
                out_desc='particles in 3-colocalizations')
            for other_ind in range(1, len(coloc_names)):
                base = os.path.join(
                    self.coords_dir, coloc_names[other_ind], 'coloc2')
                df_io.write_table(
                    table=self.coloc2_df[other_ind-1], base=base,
                    out_desc='2-colocalizations')
                base = os.path.join(
                    self.coords_dir, coloc_names[other_ind], 'particles2')
                df_io.write_table(
                    table=self.particles2_df[other_ind-1], base=base,
                    out_desc='particles in 2-colocalizations')

        else:
            base = os.path.join(self.coords_dir, coloc_name, 'coloc2')
            df_io.write_table(
                table=self.coloc2_df[0], base=base,
                out_desc='2-colocalizations')
            base = os.path.join(self.coords_dir, coloc_name, 'particles2')
            df_io.write_table(
                table=self.particles2_df[0], base=base,
                out_desc='particles in 2-colocalizations')

    def write_coloc_coords_star(self, coloc_name):
        """Writes subcolumn and particle coordinates in star files.

        ToDo
        """

        # like write_coloc_coords

        # break by distances (also sets?), use groupby()

        # use to_csv(sep='\t') to write data

        pass
    
