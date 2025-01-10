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
from io import StringIO

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

    The default approach is to write all versions (if possible) and to 
    read one by one until it is sucessful.    

    Usage from an external program like a python shell or a notebook. On
    one system:

      >>> PandasIO.write(
              table=dataframe, base=relative_path_from_external_wo_extension)

    or

      >>> df_io = PandasIO()
      >>> df_io.write_table(
              table=dataframe, base=relative_path_from_external_wo_extension)

    On the other system:

      >>> PandasIO.read(base=relative_path_from_external_wo_extension)

    or

      >>> df_io = PandasIO()
      >>> df_io.read_table(base=relative_path_from_external_wo_extension)

    Alternatively, if the file path is specified in respect to another dir, use:

      >>> PandasIO.write(
              table=dataframe, calling_dir=another_dir, 
              base=relative_path_from_another_dir)

    and 

      >>> PandasIO.read(
              calling_dir=another_dir, base=relative_path_from_another_dir)


    """

    def __init__(
        self, calling_dir=None, file_formats=['pkl', 'hdf5', 'json'],
            overwrite=True, verbose=True, info_fd=None):
        """Sets attributes from arguments.

        Arguments:
          - calling_dir: directory from which this method is called
          - file formats: default ['pkl', 'hdf5', 'json'] or a subset of
          - overwrite: flag indicating whether files are owerwritten
          - verbose: flag indicating if a statement is printed for everyhing
          file that is read or written
          - info_fd: file descriptor for the above read/write messages
        """

        # from arguments
        if calling_dir is not None:
            self.calling_dir = calling_dir
        else:
            self.calling_dir = os.getcwd()
        self.file_formats = file_formats
        self.overwrite = overwrite
        self.info_fd = info_fd
        self.verbose = verbose

        # temporaty column to store (a non-unique) index for write / read
        self.index_col = '_index'
        
    @classmethod
    def write(
            cls, table, base, calling_dir=None,
            file_formats=['pkl', 'json', 'hd5'], overwrite=True,
            hdf5_name=None, verbose=True, out_desc='', info_fd=None):
        """Writes a specified pandas.DataFrame in multiple formats.

        The written dataframe is expected to be read by read() or read_table().

        The implemented formats are:
          - 'pkl': pickled dataframe, extension 'pkl'
          - 'json': pickled json serialized dataframe, extension '_json.pkl'
          - 'hdf5': hdf5 format, extension _hdf5_name'.h5' (experimental) 

        Writes the dataframe in all formats specified by arg file_formats.
        If arg file_formats is None, writes in all three formats.

        The specified file path (arg base) should 
        not contain an extension, or it can end with .pkl in which case
        .pkl is replaced by '_json.pkl' or _hdf5_name'.h5'.

        Makes new directories if those of arg base do not exist.

        Arguments:
          - table: (pandas.DataFrame) table to we written
          - base: file path, can be absolute or relative to self.calling_dir
          - calling_dir: directory from which this method is called
          - file formats: default ['pkl', 'hdf5', 'json'] or a subset of
          - hdf5_name: used to make hdf5 file name
          - overwrite: flag indicating whether files are owerwritten
          - verbose: flag indicating if a statement is printed for every
          file that is  written
          - out_desc: description of the file that is read or written, used
          only for the out messages if verbose is True 
          - info_fd: file descriptor for the above read/write messages
        """
        pdio = cls(
            calling_dir=calling_dir, file_formats=file_formats,
            overwrite=overwrite, verbose=verbose, info_fd=info_fd)
        pdio.write_table(
            table=table, base=base, hdf5_name=hdf5_name, out_desc=out_desc)

    @classmethod
    def read(
            cls, base, calling_dir=None, file_formats=['pkl', 'json', 'hd5'],
            hdf5_name=None, verbose=True, out_desc='', info_fd=None):
        """Reads pandas DataFrame written by write() or write_table().

        Currently supported formats are:
          - 'pkl': pickled dataframe, extension 'pkl'
          - 'json': pickled json serialized dataframe, extension '_json.pkl'
          - 'hdf5': hdf5 format, extension _hdf5_name'.h5' (experimental) 

        Files of formats specified in arg file_formats, (or self.file_formats, 
        if this arg is None) are attempted to be read in the order given by
        file_formats. As soon as one file is sucessfully read, the 
        corresponding table is returned.

        Arguments:
          - base: file path, can be absolute or relative to self.calling_dir
          - calling_dir: directory from which this method is called
          - file formats: default ['pkl', 'hdf5', 'json'] or a subset of
          - hdf5_name: used to make hdf5 file name
          - verbose: flag indicating if a statement is printed for every
          file that is read
          - out_desc: description of the file that is read or written, used
          only for the out messages if self.verbose is True 
          - info_fd: file descriptor for the above read/write messages
        """
        pdio = cls(
            calling_dir=calling_dir, file_formats=file_formats, verbose=verbose,
            info_fd=info_fd)
        data = pdio.read_table(
            base=base, hdf5_name=hdf5_name, out_desc=out_desc)
        return data
               
    def write_table(
            self, table, base, file_formats=None, hdf5_name=None,
            out_desc='', overwrite=None):
        """Writes a specified pandas.DataFrame in multiple formats.

        The written dataframe is expected to be read by read() or read_table().

        Currently supported formats are:
          - 'pkl': pickled dataframe, extension 'pkl'
          - 'json': pickled json serialized dataframe, extension '_json.pkl'
          - 'hdf5': hdf5 format, extension _hdf5_name'.h5' (experimental) 

        Writes the dataframe in all formats specified by arg file_formats.
        If arg file_formats is None, writes in all three formats.

        For json, if index values are not unique, saves then in col 
        self.index_col and resets index. Reading the saved file using 
        self.read_table(), recovers the original index. 

        The specified file path (arg base) should 
        not contain an extension, or it can end with .pkl in which case
        .pkl is replaced by '_json.pkl' or _hdf5_name'.h5'.

        Makes new directories if those of arg base do not exist.

        Arguments:
          - table: (pandas.DataFrame) table to we written
          - base: file path, can be absolute or relative to self.calling_dir
          - file formats: default ['pkl', 'hdf5', 'json'] or a subset of
          - hdf5_name: used to make hdf5 file name
          - out_desc: description of the file that is read or written, used
          only for the out messages if self.verbose is True 
          - overwrite: flag indicating whether files are owerwritten
        """

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
        if (dir_ is not None) and (dir_ != '') and not os.path.exists(dir_):
            os.makedirs(dir_, exist_ok=True)

        # check whether to owerwrite
        if ('pkl' in file_formats) and (base is not None):
            path, path_short = self.get_pickle_path(base)
            if overwrite or not (os.path.exists(path)):
                table.to_pickle(path)
                if verbose:
                    print(f"Pickled {out_desc} to {path_short}",
                          file=self.info_fd)
            else:
                raise ValueError(
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
                            f'Wrote {out_desc} to {path_short} with key {key}',
                            file=self.info_fd)
                else:
                    raise ValueError(
                        f"Did not write {out_desc} to {path_short} because"
                        + " the file exists and overwrite flag is True")
            except ModuleNotFoundError:
                pass

        if ('json' in file_formats) and (base is not None):
            path, path_short = self.get_pickle_path(base, json=True)
            if overwrite or not (os.path.exists(path)):
                table = self.save_index(table=table)
                json_str = table.to_json()
                with open(path, 'wb') as fd:
                    pickle.dump(json_str, fd)
                    if verbose:
                        print(
                            f"Pickled json converted {out_desc} to "
                            + f"{path_short}", file=self.info_fd)
            else:
                raise ValueError(
                    f"Did not pickle {out_desc} to {path_short} because "
                    + "the file exists and overwrite flag is True")

    def read_table(self, base, file_formats=None, hdf5_name=None, out_desc=''):
        """Reads pandas.DataFrame tables written by write() or write_table().

        Currently supported formats are:
          - 'pkl': pickled dataframe, extension 'pkl'
          - 'json': pickled json serialized dataframe, extension '_json.pkl'
          - 'hdf5': hdf5 format, extension _hdf5_name'.h5' (experimental) 

        Files of formats specified in arg file_formats, (or self.file_formats, 
        if this arg is None) are attempted to be read in the order given by
        file_formats. As soon as one file is sucessfully read, the 
        corresponding table is returned.

        For json, if col self.index_col exists in the saved file, replaces
        the ndex by this column. It is thus comparible with write_table()

        Arguments:
          - base: file path, can be absolute or relative to self.calling_dir
          - file formats: default ['pkl', 'hdf5', 'json'] or a subset of
          - hdf5_name: used to make hdf5 file name
          - out_desc: description of the file that is read or written, used
          only for the out messages if self.verbose is True 

        Returns (pandas.DataFrame) data table
        """

        verbose = self.verbose

        if file_formats is None:
            file_formats = self.file_formats

        for ff in file_formats:
            
            if ff == 'pkl':
                path, path_short = self.get_pickle_path(base)
                try:
                    table = pd.read_pickle(path)
                    if verbose:
                        print(f"Read {out_desc} pickle {path_short}")
                    return table
                except AttributeError:
                    if verbose:
                        print(
                            f"Info: Could not read {out_desc} pickled "
                            + f"pandas.DataFrame {path_short},"
                            + " likely because pandas versions do not match.")
                except OSError:
                    pass
                except pickle.PickleError:
                    pass

            if (ff == 'hdf5') and (hdf5_name is not None):
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

            if ff == 'json':
                path, path_short = self.get_pickle_path(base, json=True)
                try:
                    with open(path, 'rb') as fd:
                        json_str = pickle.load(fd, encoding='latin1')
                    #table = pd.read_json(json_str)
                    table = pd.read_json(StringIO(json_str))
                    table = self.recover_index(table=table)
                    if verbose:
                        print(
                            f"Read {out_desc} pickled json string "
                            + f"from {path_short}")
                    return table
                except OSError:
                    pass

        raise ValueError(f"Could not read {out_desc} having path {base}")
        
    def get_pickle_path(self, base, json=False):
        """Makes path for pkl and json formats. 

        First removes'_json.pkl' or '.pkl' from arg base, if possible. Then
        adds '_json.pkl' to arg base if arg json is True and '.pkl' if 
        it is False.

        Arguments:
          - base: base path
          - json: Flag indication if json format (False means pkl format)

        Returns (path, path_short): Paths that include and do not incude the
        calling directory (self.calling_dir), respectfully
        """

        # remove .pkl extension from base
        if base.endswith('_json.pkl'):
            base = base.removesuffix('_json.pkl')
        elif base.endswith('.pkl'):
            base = base.removesuffix('.pkl')
        
        if json:
            path_short = f'{base}_json.pkl'
        else:
            path_short = f'{base}.pkl'
        if not os.path.isabs(path_short):
            path = os.path.join(self.calling_dir, path_short)
        else:
            path = path_short
            
        return path, path_short

    def get_hdf5_path_key(self, base, hdf5_name):
        """Makes path for hdf5 format.

        Arguments:
          - base: base path
          - hdf5_name: name needed for hdf5 file

        Returns (path, path_short, key): 
          - path, path_short: paths that include and do not incude the
          calling directory (self.calling_dir), respectfully
          - key: key for hdf5 file
        """

        # remove .pkl extension from base
        long_path, ext = os.path.splitext(base)
        if ext == '.pkl':
            base = long_path
        
        dir_, key = os.path.split(base)
        path_short = os.path.join(dir_, f'{hdf5_name}.h5')
        if not os.path.isabs(path_short):
            path = os.path.join(self.calling_dir, path_short)
        else:
            path = path_short

        return path, path_short, key

    def save_index(self, table):
        """Move index values to another column if index is non-unique.

        """

        if table.index.has_duplicates:
            table = table.copy()
            table[self.index_col] = table.index
            table = table.reset_index(drop=True)

        return table

    def recover_index(self, table):
        """Revert index values from anther column, opposite of save_index().

        """

        if self.index_col in table.columns:
            table = table.set_index(self.index_col)
            table.index.name = None
        return table
