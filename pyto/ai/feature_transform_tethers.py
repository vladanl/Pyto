"""
Functions that combine and scale tether features.

Orginally developed for the classification of tethers from the
presynaptic Munc13-SNAP25 project 

# Author: Vladan Lucic 
# $Id$
"""

__version__ = "$Revision$"


import os
import sys
import subprocess
import importlib
import pickle
import logging
import functools

import numpy as np
import scipy as sp
import pandas as pd
import sklearn as sk
from sklearn import preprocessing as sk_preproc
from sklearn.compose import ColumnTransformer

import pyto


def combine_features(data):
    """Makes combination of features.

    Arguments:
      - data: (pandas.DataFrame) features table (not modified)

    Returns (data, features):
      - data: (pandas.DataFrame) table containing initial and new features
      - features: (list) names of new features
    """
    
    data = data.copy()

    # no camel case
    data = data.rename(columns={'nLoops': 'n_loops'})

    data['length_curve'] = data['length_nm'] - data['length_min_nm']
    data['length_curve_ratio'] = data['length_nm'] / data['length_min_nm']
    data['length_max_min'] = data['length_max_nm'] - data['length_min_nm']
    data['length_max_min_ratio'] = (
        data['length_max_nm'] / data['length_min_nm'])
    data['length_max_min_relative'] = (
        (data['length_max_nm'] - data['length_min_nm'])
        / data['length_min_nm'])
    data['length_bound_ratio'] = (
        data['boundary_distance_nm'] / data['length_min_nm'])
        
    data['v-s'] = data['volume'] - data['surface']
    data['svr'] = data['surface'] / data['volume']
    data['len_v_ratio'] = data['length_max_nm'] / data['volume_nm']
    data['euler_v_ratio'] = data['euler'] / data['volume']
    data['loops_v_ratio'] = data['n_loops'] / data['volume']
    data['n_holes'] = data['euler'] - 1 + data['n_loops']

    features = [
        'length_curve', 'length_curve_ratio', 'length_max_min', 
        'length_max_min_ratio', 'length_max_min_relative', 'length_bound_ratio',
        'v-s', 'svr', 'len_v_ratio', 'euler_v_ratio', 'loops_v_ratio',
        'n_loops', 'n_holes']

    return data, features

def get_transformer_robust(features=None):
    """Robust feature scaling.

    All features are scaled using sklearn.preprocessing.RobustScaler()
    except svr scaled using sklearn.preprocessing.StandardScaler().

    Argument:
      - features: (list of features, default None) if specified, keeps only 
      the specified features

    Returns:
      - transformer (sk.compose.ColumnTransformer)
      - list of features
    
    """

    robust_multi = sk.pipeline.make_pipeline(
        sk_preproc.RobustScaler(),
        sk_preproc.FunctionTransformer(lambda x: 100*x))

    trans_list = [
        ("length", sk_preproc.RobustScaler(), 
         ['length_nm', 'length_min_nm', 'length_median_nm', 'length_max_nm',
          'boundary_distance_nm', 'distance_nm']),
        ('length_diff', sk_preproc.RobustScaler(), 
         ['length_curve', 'length_max_min']),
        ('length_ratio', sk_preproc.RobustScaler(), 
         ['length_curve_ratio', 'length_max_min_ratio', 
          'length_max_min_relative', 'length_bound_ratio']),
        ('morphology', sk_preproc.RobustScaler(), 
         ['thresh_rank', 'volume_nm', 'surface_nm']),
        ('morphology_diff', sk_preproc.RobustScaler(), ['v-s']), 
        ('morphology_svr', sk_preproc.StandardScaler(), ['svr']), 
        ('morphology_ratio', sk_preproc.RobustScaler(), 
         ['len_v_ratio', 'euler', 'n_loops', 'loops_v_ratio']),
        #('drop', 'drop', ['ids', 'volume', 'surface']) 
        ]

    # remove features that are not in the specified features
    if features is not None:
        feat_set = set(features)
        trans_list = [
            (tr_one[0], tr_one[1], list(set(tr_one[2]) & feat_set))
            for tr_one in trans_list]
        trans_list = [tr_one for tr_one in trans_list if len(tr_one) > 0]
    
    ct = ColumnTransformer(trans_list, remainder='drop')

    columns = get_columns(trans_list=trans_list)
    
    return ct, columns

def get_transformer_2(features=None):
    """
    All standard

    Probably not used
    """

    trans_list = [
        ("length", sk_preproc.StandardScaler(), 
         ['length_nm', 'length_min_nm', 'length_median_nm', 'length_max_nm',
          'boundary_distance_nm', 'distance_nm']),
        ('length_diff', sk_preproc.StandardScaler(), 
         ['length_curve', 'length_max_min']),
        ('length_ratio', sk_preproc.StandardScaler(), 
         ['length_curve_ratio', 'length_max_min_ratio', 
          'length_max_min_relative', 'length_bound_ratio']),
        ('morphology', sk_preproc.StandardScaler(), 
         ['thresh_rank', 'volume_nm', 'surface_nm']),
        ('morphology_diff', sk_preproc.StandardScaler(), ['v-s']), 
        ('morphology_svr', sk_preproc.StandardScaler(), ['svr']), 
        ('morphology_ratio', sk_preproc.StandardScaler(), 
         ['len_v_ratio', 'euler', 'n_loops', 'loops_v_ratio']),
        #('drop', 'drop', ['ids', 'volume', 'surface']) 
        ]

    # remove features that are not in the specified features
    if features is not None:
        feat_set = set(features)
        trans_list = [
            (tr_one[0], tr_one[1], list(set(tr_one[2]) & feat_set))
            for tr_one in trans_list]
        trans_list = [tr_one for tr_one in trans_list if len(tr_one) > 0]
    
    ct = ColumnTransformer(trans_list, remainder='drop')

    columns = get_columns(trans_list=trans_list)
    
    return ct, columns

def get_transformer_logstd(features=None, remainder='drop'):
    """Transformer using log and standard scaler.

    Makes transformer that does the following;
      - Features having long tails are log-transformed
      - All features are transformed by the standard scaler 
      (sklearn.preprocessing.StandardScaler())

    Argument:
      - features: (list of features, default None) if specified, keeps only 
      the specified features

    Returns:
      - transformer (sk.compose.ColumnTransformer)
      - list of features
    """

    # individual transformers
    ln_std = sk.pipeline.make_pipeline(
        sk_preproc.FunctionTransformer(np.log), sk_preproc.StandardScaler())
    ln1_std = sk.pipeline.make_pipeline(
        sk_preproc.FunctionTransformer(lambda x: np.log(x+1)), 
        sk_preproc.StandardScaler())
    ln1m_std = sk.pipeline.make_pipeline(
        sk_preproc.FunctionTransformer(lambda x: np.log(-x+1.1)), 
        sk_preproc.StandardScaler())

    # features
    trans_list = [
        ("length_std", sk_preproc.StandardScaler(),
          ['length_nm', 'length_min_nm']),
        ("length", ln_std,
          ['length_median_nm', 'length_max_nm', 'boundary_distance_nm',
           'distance_nm']),
        ('length_diff', ln1_std, 
          ['length_curve', 'length_max_min', 'length_max_min_relative']),
        ('length_ratio', ln_std, 
          ['length_curve_ratio', 'length_max_min_ratio']),
        ('pass1', 'passthrough', ['length_bound_ratio']),
        ('morphology_std', sk_preproc.StandardScaler(), 
          ['thresh_rank']),
        ('morphology', ln1_std, 
          ['volume_nm', 'surface_nm']),
        ('morphology_diff', ln1_std, ['v-s']), 
        #('morphology_svr', sk_preproc.StandardScaler(), ['svr']), 
        ('morphology_svr', 'passthrough', ['svr']), 
        ('morphology_ratio', ln_std, ['len_v_ratio']),
        ('topology', ln1_std, ['n_loops', 'loops_v_ratio']),
         #('passthrough', 'passthrough', 
         # ['ids', 'volume', 'surface']) 
        ]
        
    # remove features that are not in the specified features
    if features is not None:
        feat_set = set(features)
        trans_list = [
            (tr_one[0], tr_one[1], list(set(tr_one[2]) & feat_set))
            for tr_one in trans_list]
        trans_list = [tr_one for tr_one in trans_list if len(tr_one) > 0]

    ct = sk.compose.ColumnTransformer(trans_list, remainder=remainder)
    columns = get_columns(trans_list=trans_list)

    return ct, columns

def get_transformer_logrobust(features=None, remainder='drop'):
    """Transformer using log and robust scaler.

    Defines transformation rules for several features commonly used
    in the presynaptic workflow. If the data that is transformed do not 
    contain some of the features for which the rules are defined in 
    this function, the features of the data that are to be transformed 
    have to be specified by arg features.

    Arguments:
      - features: (default None) list of features that are transformed, 
      or None for all features
      - remainder: 'drop' (default) or 'passthrough' sets rules for 
      the features not defined here (like in sk.compose.ColumnTransformer)

    Makes transformer that does the following;
      - Features having long tails are log-transformed
      - All features are transformed by the standard scaler 
      (sklearn.preprocessing.RobustScaler())

    Argument:
      - features: (list of features, default None) if specified, sets
      transformation rules only for the specified features

    Returns:
      - transformer (sk.compose.ColumnTransformer)
      - list of features
    """

    # individual transformers
    ln_robust = sk.pipeline.make_pipeline(
        sk_preproc.FunctionTransformer(np.log), sk_preproc.RobustScaler())
    ln1_robust = sk.pipeline.make_pipeline(
        sk_preproc.FunctionTransformer(lambda x: np.log(x+1)), 
        sk_preproc.RobustScaler())
    ln1m_robust = sk.pipeline.make_pipeline(
        sk_preproc.FunctionTransformer(lambda x: np.log(-x+1.1)), 
        sk_preproc.RobustScaler())

    # features
    trans_list = [
        ("length_robust", sk_preproc.RobustScaler(),
          ['length_nm', 'length_min_nm']),
        ("length", ln_robust,
          ['length_median_nm', 'length_max_nm', 'boundary_distance_nm',
           'distance_nm']),
        ('length_diff', ln1_robust, 
          ['length_curve', 'length_max_min', 'length_max_min_relative']),
        ('length_ratio', ln_robust, 
          ['length_curve_ratio', 'length_max_min_ratio']),
        ('pass1', 'passthrough', ['length_bound_ratio']),
        ('morphology_robust', sk_preproc.RobustScaler(), 
          ['thresh_rank']),
        ('morphology', ln1_robust, 
          ['volume_nm', 'surface_nm']),
        ('morphology_diff', ln1_robust, ['v-s']), 
        #('morphology_svr', sk_preproc.RobustScaler(), ['svr']), 
        ('morphology_svr', 'passthrough', ['svr']), 
        ('morphology_ratio', ln_robust, ['len_v_ratio']),
        ('topology', ln1_robust, ['n_loops', 'loops_v_ratio']),
         #('passthrough', 'passthrough', 
         # ['ids', 'volume', 'surface']) 
        ]
        
    # remove features that are not in the specified features
    if features is not None:
        feat_set = set(features)
        trans_list = [
            (tr_one[0], tr_one[1], list(set(tr_one[2]) & feat_set))
            for tr_one in trans_list]
        trans_list = [tr_one for tr_one in trans_list if len(tr_one) > 0]

    ct = sk.compose.ColumnTransformer(trans_list, remainder=remainder)
    columns = get_columns(trans_list=trans_list)

    return ct, columns

def transform_df(data, transformer):
    """Transforms data using the specified transformer.

    If arg transformer is a ColumnTransformer instance, it is used 
    to transform the specified data.

    Alternatively, if atg transformer is an integer (denoted as N), function 
    get_transformer_N() of this module is used as is used to transform 
    the data. In this case, get_transformer_N() has to be a transformer.

    In any case, instantiated transformer (transformer()) has to have
    fit_transform() method.

    Columns that are not transfomed by the transformer are passed as they
    are to the returned table.

    Arguments:
      - data: (pandas.DataFrame) data to be transformed
      - transformer: either a function that takes no arguments and returns
      a ColumnTransformer (or a similar object) and a list of transformed
      columns, or an integer

    Returns (pandas.DataFrame) transformed data.
    """

    data = data.copy()

    if isinstance(transformer, int):
        this_module = sys.modules[__name__]
        transformer = getattr(this_module, f'get_transformer_{transformer}')
    ct, ct_cols = transformer()
    data[ct_cols] = pd.DataFrame(
        ct.fit_transform(data), columns=ct_cols, index=data.index)

    return data

def get_columns(trans_list):
    """Returns all features specified in trans_list.
    """

    cols = functools.reduce(
        lambda x, y: x + y, 
        [item[2] for item in trans_list])
    return cols



