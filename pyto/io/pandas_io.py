"""
Class PandasIO contains methods for reading and writting Pandas.DataFrame
tables.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""

__version__ = "$Revision$"

import os
import sys
import importlib
import pickle

import numpy as np
import pandas as pd


class PandasIO(object):
    """
    Methods to read and write pandas.DataFrame objects. 

    Meant for cases when a dataframe needs to be written and read in 
    environments that may have incompatible pandas versions. The
    approach here is to write the same table in one of more formats:

      - directly pickled dataframe
      - dataframe serialized by json and then pickled
      - hdf5 format

    Because these formats have different excensions, the specified 
    file path should not contain an extension.

    The default approach is to write all versions (if possible) and to 
    read one by one until it is sucessful.    

    Usage from an external program like a python shell or a notebook. On
    one system:

      >>> df_io = PandasIO(calling_dir=os.getcwd())
      >>> df_io.write(
              table=dataframe,
              base=relative_path_from_external_wo_extension)

    On the other system:

      >>> df_io = PandasIO(calling_dir=os.getcwd())
      >>> df_io.read(base=relative_path_from_external_wo_extension)
    """

    def __init__(
        self, calling_dir='', file_formats=['pkl', 'hdf5', 'json'],
        overwrite=True, verbose=True):
        """
        Sets attributes from arguments
        """
        
        self.calling_dir = calling_dir
        self.file_formats = file_formats
        self.overwrite = overwrite
        self.verbose = verbose

    def write_table(
            self, table, base, file_formats=None, hdf5_name=None,
            out_desc='', overwrite=None):
        """
        """

        # remove extension from base?
        
        # figure out overwrite
        if overwrite is None:
            overwrite = self.overwrite
        
        # initialization 
        if file_formats is None:
            file_formats = self.file_formats
        verbose = self.verbose
        path, _  = self.get_pickle_path(base)

        # make out directory if it doesn't exist
        dir_ = os.path.split(path)[0]
        if not os.path.exists(dir_):
            os.makedirs(dir_, exist_ok=True)

        if ('pkl' in file_formats) and (base is not None):
            path, path_short = self.get_pickle_path(base)
            if overwrite or not (os.path.exists(path)):
                table.to_pickle(path)
                if verbose:
                    print(f"Pickled {out_desc} to {path_short}")
            else:
                if verbose:
                    print(
                        f"Did not pickle {out_desc} to {path_short} because "
                        + "the file exists and overwrite flag is True")
                    
        if (('hdf5' in file_formats) and (hdf5_name is not None)
            and (base is not None)):
            path, path_short, key = self.get_hdf5_path_key(
                base=base, hdf5_name=hdf5_name)
            try:
                if overwrite or not (os.path.exists(path)):
                    table.to_hdf(path, key, mode='a')
                    if verbose:
                        print(
                            f'Wrote {out_desc} to {path_short} with key {key}')
                else:
                    if verbose:
                        print(
                            f"Did not write {out_desc} to {path_short} because"
                            + " the file exists and overwrite flag is True")
            except ModuleNotFoundError:
                pass

        if ('json' in file_formats) and (base is not None):
            path, path_short = self.get_pickle_path(base, json=True)
            if overwrite or not (os.path.exists(path)):
                json_str = table.to_json()
                with open(path, 'wb') as fd:
                    pickle.dump(json_str, fd)
                    if verbose:
                        print(
                            f"Pickled json converted {out_desc} to "
                            + f"{path_short}")
            else:
                if verbose:
                    print(
                        f"Did not pickle {out_desc} to {path_short} because "
                        + "the file exists and overwrite flag is True")

    def read_table(self, base, file_formats=None, hdf5_name=None, out_desc=''):
        """
        """

        verbose = self.verbose

        if file_formats is None:
            file_formats = self.file_formats

        if 'pkl' in file_formats:
            path, path_short = self.get_pickle_path(base)
            try:
                table = pd.read_pickle(path)
                if verbose:
                    print(f"Read {out_desc}  pickle {path_short}")
                return table
            except AttributeError:
                print(
                    f"Info: Could not read {out_desc} pickle {path_short}, "
                    + "likely because pandas versions do not match.")
            except OSError:
                pass

        if ('hdf5' in file_formats) and (hdf5_name is not None):
            path, path_short, key = self.get_hdf5_path_key(
                base=base, hdf5_name=hdf5_name)
            try:
                table = pd.read_hdf(path, key)
                if verbose:
                    print(
                        f"Read {out_desc} hdf5 file {path_short} "
                        + f"with key {key}") 
                return table
            except OSError:
                pass

        if 'json' in file_formats:
            path, path_short = self.get_pickle_path(base, json=True)
            try:
                with open(path, 'rb') as fd:
                    json_str = pickle.load(fd, encoding='latin1')
                table = pd.read_json(json_str)
                if verbose:
                    print(
                        f"Read {out_desc} pickled json string "
                        + f"from {path_short}")
                return table
            except OSError:
                pass

        raise ValueError(f"Could not read {out_desc} having base {base}")
        
    def get_pickle_path(self, base, json=False):
        """
        """
        if json:
            path_short = f'{base}_json.pkl'
        else:
            path_short = f'{base}.pkl'
        if not os.path.isabs(path_short):
            path = os.path.join(self.calling_dir, path_short)
        return path, path_short

    def get_hdf5_path_key(self, base, hdf5_name):
        """
        """
        dir_, key = os.path.split(base)
        path_short = os.path.join(dir_, f'{hdf5_name}.h5')
        if not os.path.isabs(path_short):
            path = os.path.join(self.calling_dir, path_short)
        return path, path_short, key


