"""
Reading and preprocessing of pyto colocalization data.

Pyseg specific colocalization methods are in coloc_pyseg.py.
 
# Author: Vladan Lucic
# $Id$
"""

__version__ = "$Revision$"

import abc
import os

import numpy as np
import pandas as pd

import pyto
from pyto.io.pandas_io import PandasIO
import pyto.util.pandas_plus as pandas_plus


class ColocTableRead(abc.ABC):
    """Abstract class that provides colocalization table read methods.

    Meant to be inherited by ColocAnalysis.
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    @classmethod
    def read_table_multi(
            cls, dir_=None, dir_tables='tables', pick=None,
            names=None, columns=None,
            verbose=True, suffix='_data.pkl', save_formats=['pkl', 'json'],
            join_suffix='data', individual_suffix='data_syn', 
            other=None, out_desc='coloc_results', mode=None):
        """
        Reads preprocessed colocalization results from pickled tables.

        Each table is assigned to a variable having the same name as the 
        table file base name without extension. In this way, the variable
        and table file names are the same as in extract_multi().

        Two variables are set for each name (element of arg names):
          - name + '_' + join_suffix : data for all tomograms together
          - name + '_' + individual_suffix: data for individual tomograms
        Variables are set as attributes of object specified by arg module,
        where arg module can also be a module. 

        If arg columns is not None, the variable containing data from all 
        tomograms together will show only the specified columns. 

        Arguments:
          - names: list of colocalization names corresponding to the data 
          to be read, if None all data in tables directory are read  
          - module: module for the created variables
          - dir_: directory where tables are located
          - pick: name of this colocalization project (used if dir_ is None)
          - columns: list of columns
          - suffix: (not used anymore, perhap needed for backcompatibility) 
          pickled tables name suffix 
          - verbose: If True, prints names of tables that are not found 
          - save_formats: (list) formats in which tables are saved 
          (see pyto.io.PandasIO)
          - join_suffix: suffix given to colocalization data tables that 
          contain data for all tomos together
          - individual_suffix: suffix given to colocalization data tables that 
          contain data for individual tomos
          - mode: defines how to form paths to the raw colocalization 
          data and defines how to extract tomo id (see extract_data() doc)
         """

        if (dir_ is None) and (pick is None):
            raise ValueError("Either pick or dir_ argument has to be given")
        try:
            # when cls instance of ColocLite
            obj = cls(
                dir_=dir_, dir_tables=dir_tables,
                pick=pick, mode=mode, save_formats=save_formats,
                join_suffix=join_suffix, individual_suffix=individual_suffix)
        except TypeError:
            # when cls instance of ColocAnalysis
            obj = cls(
                dir_=dir_, 
                pick=pick, mode=mode, save_formats=save_formats,
                join_suffix=join_suffix, individual_suffix=individual_suffix)
            
        # get colocalization names
        # also warns if tables are found directly in tables/ dir
        names_direct = False
        if names is None:
            names = [
                file_ for file_ in os.listdir(obj.tables_dir)
                if os.path.isdir(os.path.join(obj.tables_dir, file_))]
            names_direct = [
                file_.split('_' + obj.individual_suffix)[0]
                for file_ in os.listdir(obj.tables_dir)
                if obj.individual_suffix in file_]
            names_direct = np.unique(np.asarray(names_direct)).tolist()
            if len(names) == 0:
                names = names_direct
                names_direct = True
                print(
                    f"Warning: Colocalization result tables should be saved in "
                    + f"directories like "
                    + f"{os.path.join(obj.tables_dir, 'setX_setY_setZ')}, "
                    + f"and not directly in {obj.tables_dir} directory")
            elif len(names_direct) > 0:
                print(
                    f"Warning: Some colocalizatio results are found directly "
                    + f"in {obj.tables_dir} directory and not in its "
                    + "subdirectories. These resuts will be ignored.")
        if len(names) == 0:
            print("Warning: no colocalization results found.")

        for nam in names:

            # figure out if tables in subdirs
            tables_in_subdir = False
            if os.path.isdir(os.path.join(obj.tables_dir, nam)):
                tables_in_subdir = True
                subdir = os.path.join(obj.tables_dir, nam)
                names_in_dir = [
                    file_.split('_' + obj.individual_suffix)[0]
                    for file_ in os.listdir(subdir)
                    if obj.individual_suffix in file_]
            else:
                names_in_dir = [nam]
                print(
                    f"Warning: Colocalization result tables should be "
                    + f"saved in {os.path.join(obj.tables_dir, nam)}, "
                    + f"as opposed to {obj.tables_dir} directory")
            try:
                obj._names.extend(names_in_dir)
            except (NameError, AttributeError):
                obj._names = names_in_dir
            pdio = PandasIO(
                file_formats=save_formats, verbose=verbose)

            try:
                for nam_local in names_in_dir:
                    
                    # joined data
                    obj_name = nam_local + '_' + obj.join_suffix
                    if not names_direct:
                        path = os.path.join(obj.tables_dir, nam, obj_name)
                    else:
                        path = os.path.join(obj.tables_dir, obj_name)
                    data = pdio.read_table(base=path, out_desc=out_desc)
                    if columns is not None:
                        columns_local = [
                            col for col in columns if col in data.columns]
                        data = data[columns_local]
                    setattr(obj, obj_name, data)

                    # individual tomo data
                    obj_name_syn = nam_local + '_' + obj.individual_suffix
                    if not names_direct:
                        path = os.path.join(obj.tables_dir, nam, obj_name_syn)
                    else:
                        path = os.path.join(obj.tables_dir, obj_name_syn)
                    data_syn = pdio.read_table(base=path)
                    setattr(obj, obj_name_syn, data_syn)

                    if other is not None:
                        for other_suffix in other:
                            obj_name_other = nam_local + '_' + other_suffix
                            path = os.path.join(
                                obj.tables_dir, nam, obj_name_other)
                            data_other = pdio.read_table(base=path)
                            setattr(obj, obj_name_other, data_other)
                                                
            except FileNotFoundError:
                if verbose:
                    print('Colocalization {} not found'.format(nam))

        return obj
    
    @classmethod
    def select_and_group(
            cls, id_group, dir_, dir_tables_proc,
            ids=None, remove_ids=None, names=None, group_suffix='data_group',
            join_suffix='data', individual_suffix='data_syn', 
            id_col='id', id_label='identifiers', group_label='group',
            mode=None, p_values=True, random_stats=True,
            overwrite=False, #add_group=False,
            save_formats=['json'], verbose=True):
        """Selects tomograms and make group tables.

        Meant to be used as pre-processing of the raw colocalization
        data, that is the data directly obtained by executing
        colocalization scripts.

        Loops over directories containing colocalization results. These 
        directories are located in (arg) dir_ directory. Each one contains 
        results of one colocalization ran (which can produce more that one
        colocalization, such as one 3-colocalization and two related
        2-colocalizations. If arg names is not None, only the specified 
        colocalization run directories are selected. 

        In each selected directory, finds colocalizations that have
        pickled individual (single tomo) colocalization results. 
        If arg overwrite is False, skips colocalizations for which
        group table pickle already exists. 
       
        For each colocalization found as diescribed above, does 
        the following:
          - reads individual tomo colocalization results
          - determines group for each tomo from another table (arg
          id_group) by reading its columns (args) id_label and group_label
          - calculates group colocalizations from the individual results 
          and writes them in the same directories with suffix (arg)
          group_suffix
          - if arg add_group is True and the individual results do not 
          contain group info, adds group info column and writes (pickles) 
          them back at the same location, thus overwriting the individual
          tomo results
          - also reads joined colocalization data and returns instance
          of this class that contains joined, group and individual data

        Arguments:
          - dir_: directory that contains all colocalization run 
          directories

          - ids, remove_ids
          - names: names of the colocalization runs that should be
          processed, None (default) to process all
          - id_col: name of the tomo id column in coloc tables (default 
          'id), not to be confused with id_label
          - id_group: (pandas.Dataframe) table that contains group names
          - id_label: name of the tomo id column in id_group table (default
          'identifiers'), not to be confused with id_col
          - group_label: name of the group column in id_group table (default
          'group')
          - individual_suffix, group_suffix, join_suffix: suffices for
          individual, group and joined data, respectively
          - random_stats, p_values: flags indicating if random results and
          p-values are calculated (default True for both)
          - overwrite: flag indicating if group data are owerwriten
          (default False)
          - add_group: flag indicating if group name is added to individual 
          colocalization results, which allows overwriting individual
          colocalization results (default False)
          - save_formats: formats in which pickled tables are saved (see
          pyto.io.PandasIO.write_table() arg file formats for more info)
          - verbose: flag indicating is some processing state info 
          are printed

        Returns (instance of this class) all processed colocalization 
        results. Depreciated, better read the pickled results.
        """

        # most likely not needed anymore 
        pick = None 
        out_desc = 'coloc_results'
        #if (dir_ is None) and (pick is None):
        #    raise ValueError("Either pick or dir_ argument has to be given")

        # tomo ids
        if (ids is not None) and (remove_ids is not None):
            raise ValueError(
                "At most one of the arguments 'ids' and 'remove_ids' "
                + "can be specified.")
        if ids is not None:
            keep_ids = ids
        
        obj = cls(
            dir_=dir_, mode=mode, save_formats=save_formats,
            join_suffix=join_suffix, individual_suffix=individual_suffix)
        #if dir_tables_proc is None:
        #    dir_tables_proc = obj.dir_tables
        obj_proc = cls(
            dir_=dir_, dir_tables=dir_tables_proc, mode=mode,
            save_formats=save_formats,
            join_suffix=join_suffix, individual_suffix=individual_suffix)
        
        # loop over coloc case directories
        for nam in os.listdir(obj.tables_dir):

            # check if this name not in names
            if (names is not None) and (nam not in names):
                continue

            # list directory
            curr_dir = os.path.join(obj.tables_dir, nam)
            try:
                table_pkls = os.listdir(curr_dir)
            except NotADirectoryError:
                continue

            # make and list dir for preprocessed
            curr_dir_proc = os.path.join(obj_proc.tables_dir, nam)
            os.makedirs(curr_dir_proc, exist_ok=True)
            table_pkls_proc = os.listdir(curr_dir_proc)                
                
            # loop over colocalizations (individual coloc tables)
            for tab_pkl in table_pkls:
                if obj.individual_suffix not in tab_pkl:
                    continue

                # both 2- and 3-colocs in this dir
                nam_23 = tab_pkl.split('_' + obj.individual_suffix)[0]
                
                # check if groups exist in proc dir
                group_base = f"{nam_23}_{group_suffix}"
                group_path = os.path.join(curr_dir, group_base)
                group_path_proc = os.path.join(curr_dir_proc, group_base)
                group_exists = np.array(
                    [group_base in tab_pkl_2 for tab_pkl_2
                     in table_pkls_proc]).any()

                # check if need to make individual and joined
                individ_base = f"{nam_23}_{individual_suffix}"
                individ_path = os.path.join(curr_dir, individ_base)
                individ_path_proc = os.path.join(curr_dir_proc, individ_base)
                individ_exists = np.array(
                    [individ_base in tab_pkl_2 for tab_pkl_2
                     in table_pkls_proc]).any()

                # check if joined exist in proc dir
                join_base = f"{nam_23}_{join_suffix}"
                join_path = os.path.join(curr_dir, join_base)
                join_path_proc = os.path.join(curr_dir_proc, join_base)
                join_exists = np.array(
                    [join_base in tab_pkl_2 for tab_pkl_2
                     in table_pkls_proc]).any()

                # decide if further processing needed for this coloc
                if not overwrite and group_exists:
                    continue
                if verbose:
                    print(f"Processing {nam_23} ...")
                
                # read individual coloc data
                individ_base = f"{nam_23}_{obj.individual_suffix}"
                individ_path = os.path.join(curr_dir, individ_base)
                data_syn = PandasIO.read(
                    base=individ_path, verbose=False, out_desc=out_desc)
                #setattr(obj, individ_base, data_syn)

                # read join coloc data
                join_base = f"{nam_23}_{obj.join_suffix}"
                join_path = os.path.join(curr_dir, join_base)
                data = PandasIO.read(
                    base=join_path, verbose=False, out_desc=out_desc)
                #setattr(obj, join_base, data)
                obj.add_data(name=nam_23, data=data, data_syn=data_syn)

                # tomo ids
                if remove_ids is not None:
                    keep_ids = [
                        id_ for id_ in data_syn[id_col].unique()
                        if id_ not in remove_ids]
                    
                # select tomos, remake and write joined 
                data_proc, data_syn_proc = obj.select(
                    name=nam_23, ids=keep_ids,
                    random_stats=random_stats, p_values=p_values)
                #setattr(obj_proc, individ_base, data_syn_proc)
                #setattr(obj_proc, join_base, data_proc)
                obj_proc.add_data(
                    name=nam_23, data=data_proc, data_syn=data_syn_proc)
                if overwrite or not join_exists:
                    PandasIO.write(
                        table=data_proc, base=join_path_proc, verbose=False,
                        file_formats=save_formats, out_desc=out_desc)
                   
                # separate into groups
                col_split = obj_proc.split_by_groups(
                    name=nam_23, id_group=id_group,
                    group_label=group_label, id_label=id_label,
                    p_values=p_values, random_stats=random_stats)[0]

                # make group, group column group after distance, write
                for gr, tab in col_split.items():
                    if tab is None:
                        continue
                    tab.insert(
                        tab.columns.to_list().index('distance') + 1,
                        column=group_label, value=gr)
                data_group = pd.concat(col_split.values(), ignore_index=True)
                setattr(obj_proc, group_base, data_group)
                PandasIO.write(
                    table=data_group, base=group_path_proc, verbose=False,
                    file_formats=save_formats, out_desc=out_desc)

                # add group column to individual data after id
                if group_label not in data_syn_proc:
                    id_group_loc = id_group[[id_label, group_label]]
                    data_syn_proc = pandas_plus.merge_left_keep_index(
                        data_syn_proc, id_group_loc, left_on=id_col,
                        right_on=id_label, sort=False)
                    cols = data_syn_proc.columns.to_list()
                    cols.insert(cols.index(id_col) + 1, group_label)
                    data_syn_proc = data_syn_proc[cols[:-1]]
                    setattr(obj_proc, individ_base, data_syn_proc)

                # write individual
                if overwrite or not individ_exists:
                    PandasIO.write(
                        table=data_syn_proc, base=individ_path_proc,
                        verbose=False,
                        file_formats=save_formats, out_desc=out_desc)

        return obj_proc
                    
    def separate_by_group(
            self, names=None, group_label='group', group_suffix='data_group',
            groups=None):
        """Separates colocalization results into groups.

        Provides the same result as split_by_groups(). The difference is 
        that this method reads previously generated group data from saved 
        pickled tables, while split_by_groups() calculates group 
        colocalizations from individual tomo colocalization results. 
        Consequently, this method is much faster.

        Sets attributes that contain colocalization group data. 

        Arguments:
          - name: colocalization name, if None all colocalizations 
          (self._names) are used
          - group_label: group column name (default 'group')
          - group_suffix: suffix given to attributes of this instance
          that contain group data
          - groups: names of groups to be processed, if None (default) 
          all groups are used
        """

        result = {}

        # loop over coloc names
        for nam in self._names:
            if (names is not None) and (nam not in names):
                continue

            # get data and names for the current coloc name
            data_syn = self.get_data(nam)[1]
            name_syn = self.get_individual_name(nam)
            name_join = self.get_join_name(nam)
            data_group = self.get_group_data(name=nam, group_suffix=group_suffix)
            name_group = self.get_group_name(name=nam, group_suffix=group_suffix)

            # loop over groups
            groups_uni = data_syn[group_label].unique()
            for gr in groups_uni:
                if (groups is not None) and (gr not in groups):
                    continue
                
                data_syn_loc = data_syn[data_syn[group_label] == gr].copy()
                data_group_loc = data_group[
                    data_group[group_label] == gr].copy()

                try:
                    coloc_loc = result[gr]
                except KeyError:
                    coloc_loc = self.copy_setup()
                coloc_loc.add_data(
                    name=nam, data=data_group_loc, data_syn=data_syn_loc)
                result[gr] = coloc_loc
                
        return result
