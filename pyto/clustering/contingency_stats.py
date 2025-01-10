"""
Contains class ContingencyStats for statistical analysis of frequency
distributions specified as a contingency table.

Rows and columns of contingency classes used here are completely related.
This makes this module very different from contingency.py, which concernes
tow clustering or classification of the same elements. 


# Author: Vladan Lucic
# $Id$
"""

__version__ = "$Revision$"


import collections
import itertools

import numpy as np
import scipy as sp
import pandas as pd 
import statsmodels.api as sm


class ContingencyStats:
    """Statistical analysis of frequencies

    """

    def __init__(self, statistic=None, stat_name=None, dof=None, p=None):
        """Sets attributed from arguments
        """

        self.statistic = statistic
        self.stat_name = stat_name
        self.dof = dof
        self.p = p

    @classmethod
    def expected_contingency(
            cls, data, row_weights='margin', column_weights='margin'):
        """Makes expected values for the specified contingency table.

        Expected values are calculated using the expected values for the 
        rows and for the columns, which are determined based on args
        row_weights and column_weights, as follows:
          - 'margin': for rows data is summed along axis 0 and for columns
          along axis 1
          - list of values (row_margin length data.shape[1] and column_margin
          length data.shape[0]), do not need to be normalized
          - 1: row_weights = 1 means that all columns have the same probability
          and colum_weights = that all rows have the same probability

        Arguments:
          - data: (2d ndarray) data, different measurements (samples) are 
          typically arranged on axis 0 (rows) and different features (outcomes) 
          along axis 1 (columns)
          - row_weights, columns_weights: weights for features and measurements,
          respectively

        Returns table of expected values (2d ndarray) where elements correspond
        to the elements of arg data.
        """

        # margins
        row_margin = data.sum(axis=0)
        column_margin = data.sum(axis=1)
        total = data.sum()

        # row probabilities
        if isinstance(row_weights, str):
            if row_weights == 'margin':
                r_weights = row_margin
        elif isinstance(row_weights, (list, tuple, np.ndarray)):
            r_weights = np.asarray(row_weights)
        elif row_weights == 1:
            r_weights = np.ones(data.shape[1])
        else:
            raise ValueError(
                f"Argument row_weights {row_weights} was not understood")
        row_p = r_weights / r_weights.sum()   

        # column probabilities
        if isinstance(column_weights, str) and (column_weights == 'margin'):
            c_weights = column_margin
        elif isinstance(column_weights, (list, tuple, np.ndarray)):
            c_weights = np.asarray(column_weights)
        elif column_weights == 1:
            c_weights = np.ones(data.shape[0])
        else:
            raise ValueError(
                f"Argument column_weights {column_weights} was not understood")
        col_p = c_weights / c_weights.sum()   

        expected = np.outer(col_p, row_p) * total

        return expected                   

    @classmethod
    def get_statistic(
            cls, data, stat_name, expected=None, row_weights='margin',
            column_weights='margin', mode='total', dof=None):
        """Get contingency table statistic.

        Statistics specified by arg stat_name is calculated using the 
        actual (arg data) and expected values. The expected values can be 
        specified explicitly (arg expected) or based on the data (args 
        row_weights and colum_weights, see expected_contingency()).

        Arguments:
          - data: (2d ndarray) data, different measurements are typically 
          arranged on axis 0 (rows) and different features (outcomes) along 
          axis 1 (columns)
          - stat_name: statistics name, currently implemented "G" (same as 
          "log-likelihood ratio"), "chi-square" ("chi2") and "pearson"
          ("Pearson")
          - row_weights, columns_weights: weights for features and measurements,
          respectively (see expected_contingency() doc)
          - mode: determines how the statistic is calculated:
            - "total": all data together
            - "rows": for each row separately
            - "columns": for each column separately (aka pooled)
          - dof: degrees of freedom, if None calculated from the specified data 
          shape and the weights 

        Returns instance of this class with following attributes:
          - statistic: statistic value(s)
          - stat_name: statistic name
          - dof: degees of freedom
          - p: p-value(s)
          - resid: residuals
          - expected: expected values
        Attributes statistic and p have one value for mode='total', 
        data.shape[0] for mode='rows' and data.shape[1] for mode='columns'.
        """

        # set arguments for 1d data
        data_ndim = data.ndim
        if data.ndim == 1:
            data = data.reshape(1, -1)
            column_weights = 1
            mode = 'rows'

        # get expected
        if expected is None:
            expected = cls.expected_contingency(
                data=data, row_weights=row_weights,
                column_weights=column_weights)

        # calculate statistic
        if (stat_name == "G") or (stat_name == "log-likelihood ratio"):
            resid = 2 * data * np.log(data / expected)
        elif ((stat_name == "chi-square") or (stat_name == "chi2")
              or (stat_name == "Chi2")):
            resid = (data - expected)**2 / expected
        elif (stat_name == "pearson") or (stat_name == "Pearson"):
            resid = (data - expected)**2 / np.sqrt(expected)    
        else:
            raise ValueError(
                f"Argument statistic {statistic} is not understood")

        # dof
        if dof is None:
            if data.ndim == 1:
                dof = data.shape[1] - 1
            elif mode == 'total':
                dof = data.shape[0] * data.shape[1] - 1
                if not (isinstance(row_weights, str)
                        and (row_weights == 'margin')):
                    dof -= data.shape[0] - 1
                if not (isinstance(column_weights, str)
                        and (column_weights == 'margin')):
                    dof -= data.shape[1] - 1
            elif mode == 'rows':
                dof = data.shape[1] - 1
            elif mode == 'columns':
                dof = data.shape[0] - 1

        # inference
        if mode == 'total':
            stat = resid.sum()        
        elif mode == 'rows':
            stat = resid.sum(axis=1)        
        elif mode == 'columns':
            stat = resid.sum(axis=0)
        p_val = 1 - sp.stats.chi2.cdf(stat, df=dof) 

        if data_ndim == 1:
            expected = expected[0]
            resid = resid[0]
            stat = stat[0]
            p_val = p_val[0]

        res = cls(statistic=stat, stat_name=stat_name, dof=dof, p=p_val)
        res.resid = resid
        res.expected = expected

        return res

    def summary_table(self, data, columns, p=None):
        """Puts results of get_statistic() in a table.

        Arguments:
          - data: (2d ndarray) data, different measurements are typically 
          arranged on axis 0 (rows) and different features (outcomes) along 
          axis 1 (columns); has to be the same as passes to get_statistic()

        Returns (pandas.DataFrame) table summarizing results
        """

        increase = data[0] / self.expected[0]
        table = pd.DataFrame(
            np.vstack((increase, self.statistic, self.p)),
            columns=columns,
            index=['increase', 'statistic', 'p'])
        if p is not None:
            table = table.loc[:, table.loc['p'] < 0.05]

        return table
            
    @classmethod
    def get_association(
            cls, data, stat_name, row_weights, column_weights, dof=None):
        """Get association (heterogeneity) statistic.

        Statistic is calculated from total - pooled data and expected values, 
        where pooled is calculated from sum of the data values over axis 0 
        (so across samples).

        Arg row_weights is used to calculate both total and pooled statistic,
        while arg column_weights only for total statistic.

        Arguments:
          - data: (2d ndarray) data, different measurements are
          arranged on axis 0 (rows) and different features (outcomes) along 
          axis 1 (columns)
          - stat_name: statistics name, currently implemented "G" (same as 
          "log-likelihood ratio"), "chi-square" ("chi2") and "pearson"
          ("Pearson")
          - row_weights: weights for features (see expected_contingency() doc)
          - dof: degrees of freedom, if None calculated from the specified data 
          shape and the weights 

        Returns object with following attributes:
          - statistic: statistic value
          - stat_name: statistic name
          - dof: degees of freedom
          - p: p-value
        """

        # total
        total_all = cls.get_statistic(
            data, stat_name=stat_name, row_weights=row_weights,
            column_weights=column_weights, mode='total')
        total = total_all.statistic.sum()

        # pooled
        pooled = 0
        if not (isinstance(row_weights, str) and (row_weights == 'margin')):
            pooled_rows = cls.get_statistic(
                data.sum(axis=0), stat_name=stat_name, row_weights=row_weights)
            pooled += pooled_rows.statistic.sum()
        if not (isinstance(column_weights, str)
                and (column_weights == 'margin')):
            pooled_cols = cls.get_statistic(
                data.sum(axis=1), stat_name=stat_name,
                row_weights=column_weights)
            pooled += pooled_cols.statistic.sum()

        # assoc
        assoc = total - pooled

        # inference
        if dof is None:
            dof = (data.shape[0] - 1) * (data.shape[1] - 1)
        p_val = 1 - sp.stats.chi2.cdf(assoc, df=dof)

        res = cls(statistic=assoc, stat_name=stat_name, dof=dof, p=p_val)

        return res

    @staticmethod
    def get_p_unplanned(chi2, n_compare, dof):
        """P-value for unplanned chi2 comparisons.
        
        Calculates Sidak-adjusted p-value for n_compare unplanned comparisons
        given the chi2 statistic and dof.

        Biometry Box 17.5 (pg 722)
        
        Arguments:
        - chi2: chi2 statistic
        - n_compare: n unplanned comparisons

        Returns: Sidak-adjusted p-value
        """
        
        alpha_nonadj = 1 - sp.stats.chi2.cdf(chi2, df=dof) 
        alpha = 1 - (1 - alpha_nonadj)**n_compare
        return alpha

    @staticmethod
    def get_chi2_unplanned(p_value, n_compare, dof):
        """Critical chi2 for specified p value and n comparisons.

        Calculates chi2 value that corresponds to the specified p-value,
        when n_compare unplanned comparisons are made.

        Biometry Box 17.5 (pg 722)
        
        Arguments:
          - p_value: required p-value
          - n_compare: n unplanned comparisons
          - dof: degrees of freedom

        Returns: critical chi2 value
        """
        alpha_prime = 1 - (1 - p_value)**(1/n_compare)
        chi2 = sp.stats.chi2.ppf(1-alpha_prime, df=dof)
        return chi2

    @staticmethod
    def get_p_all_sets(chi2, shape):
        """P-value for all possible sets of unplanned comparisons.

        Calculates adjusted p-value for testing all possible sets
        given the chi2 statistic.

        Biometry Box 17.5 (pg 722)

        Arguments:
          - chi2: chi2 statistic
          - shape: data shape

        Returns adjusted p-value
        """

        dof = (shape[0] - 1) * (shape[1] - 1)
        alpha = 1 - sp.stats.chi2.cdf(chi2, df=dof)
        return alpha

    @classmethod
    def pairwise(
            cls, data, pair_axis, stat_name='G', groups=None,
            row_weights='margin', column_weights='margin', dof=1,
            verbose=False):
        """Pairwise heterogeneity

        Biometry, Box 17.5 second part

        """

        if groups is not None:
            data = data.loc[groups]
        if pair_axis == 0:
            elements = data.index
            data = data.loc[:, groups]
        else:
            elements = data.columns
            data = data.loc[groups, :]

        n_pairs = len(elements) * (len(elements) - 1) / 2
        assoc = []
        pairs = list(itertools.combinations(elements, 2))
        for pair in pairs:
            if pair_axis == 0:
                data_loop = data.loc[pair, :]
            else:
                data_loop = data.loc[:, pair]
            interact = cls.get_association(
                data=data_loop.values, stat_name=stat_name,
                row_weights=row_weights, column_weights=column_weights)
            interact.corrected_p = cls.get_p_unplanned(
                chi2=interact.statistic, n_compare=n_pairs, dof=1)
            assoc.append(interact)
            if verbose:
                print(
                    f"Pair {pair} {interact.stat_name} statistic "
                    + f"{interact.statistic:.3f}, dof {interact.dof}, "
                    + f"corrected p-value {interact.corrected_p:.4f}")

        result_struct = collections.namedtuple(
            'Struct', 'pairs associations')
        result = result_struct._make((pairs, assoc))
        return result

    @classmethod
    def find_two_distinct_sets(
            cls, data, p_value, set_axis, order_group, row_weights,
            column_weights, stat_name='G', dof=None, verbose=False):
        """Finds sets that differ the most

        Biometry, box 17.5.

        Arguments:
          - data
          - p_value: defines homogeneous sets
          - set_axis: data axis that contain set elements (0 rows form sets, 
            1 if columns)
          - order group: label used to order probabilities (has to be on the 
            axis other than set_axis)

        Returns (namedtuple):
          - high
          - low
        """

        if dof is None:
            dof = (data.shape[0] - 1) * (data.shape[1] - 1)
        if set_axis == 0:
            group_axis = 1
        else:
            group_axis = 0

        # order 
        p_group = data.div(data.sum(axis=group_axis), axis=set_axis)
        p_group_order = p_group.sort_values(by=order_group, axis=set_axis)
        if set_axis == 0:
            set_elements_order = p_group_order.index
        else:
            set_elements_order = p_group_order.columns
        descend_order = set_elements_order[::-1]

        # find homogeneousfrom the highest p 
        order = descend_order
        high_homo, high_sets, high_assoc =  cls.find_max_homogeneous(
            data=data, order=order, p_value=p_value, set_axis=set_axis,
            row_weights=row_weights, column_weights=column_weights,
            stat_name=stat_name, dof=dof, verbose=verbose)

        # find homogeneous from the lowest p
        order = set_elements_order
        low_homo, low_sets, low_assoc =  cls.find_max_homogeneous(
            data=data, order=order, p_value=p_value, set_axis=set_axis,
            row_weights=row_weights, column_weights=column_weights,
            stat_name=stat_name, dof=dof, verbose=verbose)
        
        high_set = descend_order[:-len(low_homo)]
        low_set = descend_order[len(high_homo):]
        sets = high_sets + low_sets
        assoc = high_assoc + low_assoc

        res_struct = collections.namedtuple(
            'Max_homo', 'high low sets associations')
        res = res_struct._make((high_set.values, low_set.values, sets, assoc))
        
        return res

    @classmethod
    def find_max_homogeneous(
            cls, data, order, p_value, set_axis, row_weights,
            column_weights, stat_name, dof, verbose=False):
        """Find maximal homogeneous set from the beginning

        Used in find_two_distinct_sets()
        """

        sets = []
        assoc = []
        for ind in range(2, len(order)+1):
            curr_set = order[:ind]
            if set_axis == 0:
                curr_data = data.loc[curr_set, :]
            else:
                curr_data = data.loc[:, curr_set]
            hetero = cls.get_association(
                curr_data.values, stat_name=stat_name, row_weights=row_weights, 
                column_weights=column_weights, dof=dof)
            sets.append(curr_set)
            assoc.append(hetero)
            if verbose:
                print(f"Set {list(curr_set)}, statistic {hetero.statistic:.3f}, "
                      + f"p-value {hetero.p:.4f}, dof {hetero.dof}")
            if hetero.p < p_value:
                homo_set = curr_set[:-1]
                break
        else:
            homo_set = curr_set

        return homo_set, sets, assoc
