"""
Contains class Presynaptic for a further analysis of a presynaptic
project results.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""

__version__ = "$Revision$"


import os
import sys
import pickle

import numpy as np
import pandas as pd

import pyto
from ..io.module_io import ModuleIO
from ..io.pandas_io import PandasIO


class Presynaptic(object):
    """
    """

    def __init__(self, df_io=None, calling_dir=''):
        """
        """

        if df_io is not None:
            self.df_io = PandasIO(calling_dir=calling_dir)
            self.write_table = self.df_io.write_table
            self.read_table = self.df_io.read_table
        else:
            self.df_io = None
            
        if calling_dir is not None and (calling_dir != ''):
            self.calling_dir = calling_dir
        elif df_io is None:
            self.calling_dir = calling_dir
        else:
            self.calling_dir = df_io.calling_dir
        
    def load(self, path, preprocess=True):
        """
        Loads a module that contains the presynaptic project results
        """

        mod_io = ModuleIO(calling_dir=self.calling_dir)
        work = mod_io.load(path=path, preprocess=preprocess)
        return work

    def format(self, scalar, indexed):
        """
        Formats presynaptic analysis tether tables 
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
        vesicles and shows it together with the vesicle distance
        to the active zone membrane.

        Assumes that all vesicle ids are higher than the active 
        zone (presynaptic) membrane id.

        Arguments:
          - near_sv: proximal vesicles indexed table
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

