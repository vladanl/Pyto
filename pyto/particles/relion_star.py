"""
Class for handing a single relion star file.

Work in progress. Intended to take some functionality of relion_tools.

# Author: Vladan Lucic
# $Id:$
"""

__version__ = "$Revision$"

import os

import numpy as np
import pandas as pd

import pyto


class RelionStar:
    """
    Class for handling a single relion star file.
    """

    def __init__(self, label_prefix='_', float_fmt='%12.6f'):
        """Sets arguments
        """
        self.label_prefix = label_prefix
        self.float_fmt = float_fmt

    def parse(
            self, starfile, tablename, convert=True, check_labels=True,
            verbose=False):
        """Reads table from relion-format star file.

        Extracts data from the table specified by arg tablename 

        Sets following attributes:
          - self.labels: labels of the data table
          - self.data: (pandas.DataFrame) data table
          - self.top: lines from top of the star file until the data
          table block
          - self.block_head: lines from the block header
          - self.label_lines: label lines 
          - self.data_lines: data table lines
          - self.bottom: lines that come after the data table
        
        Arguments:
          - starfile: path to star file
          - tablename: table name
          - convert: flag indicating if the star file table is converted
          to pandas.DataFrame
          - check_labels: checks whether lable indices increase uniformly
          by 1 and raises Value Error if they do not (default True)
          - verbose: flag indicating if file read info is printed
        """

        # initialization
        at_top = True
        block_found = False
        labels_found = False
        at_bottom = False
        self.top = []
        self.block_head = []
        self.label_lines = []
        self.data_lines = []
        self.bottom = []

        # check if starfile exists
        if not os.path.exists(starfile):
            raise ValueError("Starfile: " + str(starfile) + " doesn't exist.")

        # read file
        with open(starfile, "r") as file:

            for line in file:
                line = line.rstrip("\n")

                if at_top:
                    if not line.startswith(tablename):
                        self.top.append(line)
                    else:
                        at_top = False
                        block_found = True
                        self.block_head.append(line)

                elif block_found and not labels_found:
                    if line.startswith('loop_'):
                        labels_found = True
                    self.block_head.append(line)

                elif block_found and labels_found:
                    if line.startswith('_'):
                        self.label_lines.append(line)

                    elif (len(line) == 0) or line.isspace():
                        at_top = False
                        block_found = False
                        labels_found = False
                        at_bottom = True
                        self.bottom.append(line)

                    else:
                        self.data_lines.append(line)

                elif at_bottom:
                    self.bottom.append(line)

            if verbose:
                print(f"Read star file: {starfile}")

        # make labels
        self.labels = self.parse_label_lines(check=check_labels)

        # dataframe
        self.data = self.data_lines_to_df(convert=convert)

    def write(
            self, starfile, data=None, labels=None, verbose=False):
        """Write star file.

        """

        # put all lines together
        if data is None:
            data = self.data
        data_lines = self.df_to_data_lines(data)
        if labels is None:
            label_lines = self.label_lines
        else:
            label_lines = [
                f"_{lab} #{ind+1}" for ind, lab in enumerate(labels)] 

        # write
        with open(starfile, "w") as file:
            for line in (
                    self.top + self.block_head + label_lines + data_lines
                    + self.bottom):
                file.write(line + '\n')

            if verbose:
                print(f"Wrote star file: {starfile}")
        
    def parse_label_lines(self, check=True):
        """Extracts labels from self.label_lines.
        """

        # parse label liness
        labels, label_indices = np.transpose(
            [line.split() for line in self.label_lines])

        # get labels
        labels = [lab.lstrip(self.label_prefix) for lab in labels]

        # check label index order
        if check:
            indices = [int(lab_ind.lstrip('#')) for lab_ind in label_indices]
            if not (np.arange(1, len(indices)+1) == indices).all():
                raise ValueError(
                    f"Label indices ({indices}) are not increasing uniformely "
                    + "with step one")

        return labels

    def data_lines_to_df(self, convert=True):
        """Convert data lines to pandas.DataFrame
        """

        # make dataframe
        data_array = [line.split() for line in self.data_lines]
        df = pd.DataFrame(data_array, columns=self.labels)

        # convert dtypes
        df = df.convert_dtypes()
        if convert:
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    pass

        return df

    def df_to_data_lines(self, data=None, labels=None):
        """Convert data from pandas.DataFrame to lines.

        """

        if data is None:
            data = self.data
        if labels is None:
            labels = self.labels

        col_space = int(self.float_fmt.split('%')[1].split('.')[0])    
        data_str = data.to_string(
            header=False, index=False,
            float_format=(lambda x: self.float_fmt % x), col_space=col_space)
        data_lines_raw = data_str.split('\n')

        # for some reason sometimes lines get leading spaces, so remove 
        data_lines = [lin.lstrip() for lin in data_lines_raw]

        return data_lines
    
