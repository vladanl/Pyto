"""
Pytorch Dataset class

Can be used for any features-based classification, although originally
used for tether classification.

# Author: Vladan Lucic 
# $Id$
"""

__version__ = "$Revision$"


import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class FeaturesDatasetNN(Dataset):
    """
    Features dataset for neural nets.
    """
    
    def __init__(
            self, data=None, features=None, target=None,
            X=None, y=None, device=None):
        """
        Sets device and arguments as attributes, and figures out functions.

        Data can be specified in the following ways:
          - A single table containing all data (pandas.DataFrame)
          - Separate tables (pd.DataFrame) containing training and test data
          - Separate arrays (np.ndarray) containing training and test data

        Arguments:
          - data (pandas.DataFrame): data table
          - features: names of data columns that contain feature variables
          - target: name(s) of one or more data columns that contains 
          target variable  
        """

        # set device
        self.device = device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        # set attributes
        if data is not None:
            self.data = data
            self.features = features
            self.target = target
            self._len = self._len_one_df
            self._get_item = self._get_item_one_df

        elif (X is not None):
            self.X = X
            self.y = y
            if isinstance(X, pd.DataFrame):
                self._len = self._len_x_y_df            
                self._get_item = self._get_item_x_y_df
            elif isinstance(X, np.ndarray):
                self._len = self._len_x_y_np            
                self._get_item = self._get_item_x_y_np
                
    def __len__(self):
        return self._len()

    def _len_one_df(self):
        return len(self.data)

    def _len_x_y_df(self):
        return len(self.X)

    def _len_x_y_np(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):

        return self._get_item(idx)

    def _get_item_one_df(self, idx):
        
        row = self.data.iloc[idx]
        feat = row[self.features].to_numpy(dtype=float)
        feat = torch.from_numpy(feat).float().to(self.device)
        if isinstance(self.target, (list, tuple)):
            targ = row[self.target].to_numpy(dtype=int)
            targ = torch.from_numpy(targ)  #.int()
        else:
            targ = row[self.target]
            targ = torch.tensor(targ)
        targ = targ.to(self.device)
        
        return feat, targ

    def _get_item_x_y_df(self, idx):

        row_x = self.X.iloc[idx]
        feat = row_x.to_numpy(dtype=float)
        feat = torch.from_numpy(feat).float().to(self.device)
        if self.y is not None:
            row_y = self.y.iloc[idx]
            if len(self.y.shape) == 2:
                targ = row_y.to_numpy(dtype=int)
                targ = torch.from_numpy(targ)  #.int()
            else:
                targ = row_y
                targ = torch.tensor(targ)
            targ = targ.to(self.device)
        else:
            targ = None

        return feat, targ

    def _get_item_x_y_np(self, idx):

        row_x = self.X[idx, :]
        feat = torch.from_numpy(row_x).float().to(self.device)
        if len(self.y.shape) == 2:
            row_y = self.y[idx, :]
            targ = torch.from_numpy(row_y)  #.int()
        else:
            targ = torch.tensor(self.y[idx])
        targ = targ.to(self.device)

        return feat, targ
        
