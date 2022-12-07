"""
Scipy related utility functions.

# Author: Vladan Lucic, Max Planck Institute for Biochemistry
# $Id$
"""
from __future__ import unicode_literals
from __future__ import division
from builtins import zip
#from past.utils import old_div

__version__ = "$Revision$"


import numpy
import scipy as sp
import scipy.stats as stats
import pandas as pd


def chisquare_2(f_obs_1, f_obs_2, yates=False):
    """
    Calculates chi-square between two arrays of observation frequencies and
    returns the result. 

    Differs from scipy.stats.chisquare() in that this function calculates
    significance between two distributions and that the two distributions 
    can have different number of data points.
    
    If both observation frequency arrays have 0 at the same position, those
    values are removed and the degrees of freedom are adjusted using
    the new array length. This is equivalent to removing an empty bin.

    Arguments:
      - f_obs_1, f_obs_2: frequencies of observations 1 and 2

    Returns: chi-square, associated p-value 
    """
    
    # prepare variables
    f_obs_1 = numpy.asarray(f_obs_1)
    f_obs_2 = numpy.asarray(f_obs_2)

    # remove elements where both arrays have 0's
    both_nonzero = (f_obs_1 > 0) & (f_obs_2 > 0)
    if not both_nonzero.all():
        f_obs_1 = f_obs_1[both_nonzero]
        f_obs_2 = f_obs_2[both_nonzero]
    
    # chisquare
    if not yates:

        # no yates
        sum_1 = float(f_obs_1.sum())
        sum_2 = float(f_obs_2.sum())

        # calculate chi-square value
        chisq = [(el_1 * sum_2 - el_2 * sum_1) ** 2 / float(el_1 + el_2) 
                 for el_1, el_2 in zip(f_obs_1, f_obs_2)]
        chisq = numpy.array(chisq, dtype='float').sum() / (sum_1 * sum_2)

    else:

        # with yates (easier to see what happens)
        data = numpy.vstack([f_obs_1, f_obs_2])

        # expectations
        sum_freq = data.sum(axis=1)
        sum_obs = data.sum(axis=0)
        expect = numpy.outer(sum_freq, sum_obs) / float(data.sum())

        # chisquare
        chisq = ((numpy.abs(data - expect) - 0.5)**2 / expect).sum()

    # probability (same as stats.chi2.sf())
    #p = stats.chisqprob(chisq, len(f_obs_1)-1)  depreciated
    p = stats.distributions.chi2.sf(chisq, len(f_obs_1)-1)

    return chisq, p


def ttest_ind_nodata(mean_1, std_1, n_1, mean_2, std_2, n_2):
    """
    Student's t-test between two independent samples. Unlike in ttest_ind(), 
    the samples (data) are not given, instead the basic statstical quantities 
    (mean, standard deviation and number of measurements) are used to do the 
    test. Returns t-value and a two-tailed confidence level.

    Arguments:
      - mean_1, mean_2: means of samples 1 and 2
      - std_1, std_2: standard deviations of samples 1 and 2 calculated using
      n-1 degrees of freedom where n is the number of measurements in a sample
      - n_1, n_2: number of measurements 
      Arguments can be ndarrays instead of single numbers. 

    Returns: (t_value, two_tail_confidence)
    """

    # sums of squares
    sum_squares_1 = std_1 ** 2 * (n_1 - 1)
    sum_squares_2 = std_2 ** 2 * (n_2 - 1)

    # std of the defference between means
    pooled_var = (sum_squares_1 + sum_squares_2) / (n_1 + n_2 - 2.)
    std_means = numpy.sqrt(pooled_var * (1. / n_1 + 1. / n_2))

    # t-value and confidence
    t = (mean_1 - mean_2) / std_means
    confidence = 2 * stats.t.sf(numpy.abs(t), n_1 + n_2 - 2)

    return t, confidence


def anova_two_level(
        data, group_label, subgroup_label, value_label, output='struct'):
    """
    Two-level nested ANOVA with samples of unequal size.

    The two levels are called groups and subgroups. The statistics (sum 
    of squares, mean squares and degrees of freedom) are calculated for
    the following:
      - within subgroups
      - between subgroups (within groups)
      - between groups

    The inference is calculated using one-tailed F-test as follows:
      - between subgroups: F = mean_sq_between_sub / mean_sq_within_sub
      - between groups: F = mean_sq_between_groups / mean_sq_between_sub

    Arguments:
      - data: (Pandas.DataFrame) all data
      - group_label: column name specifying groups
      - subgroup_label: column name specifying subgroups
      - value_label: column name containing data
      - output: output format: 'struct' or 'dataframe'

    Return in case arg output is 'struct' is a results object (actually 
    a structure) with attributes: 
      - results.within_subgroups
      - results.between_subgroups
      - results.between_groups
    Each of these attributes is an object (actually a structure) having
    attributes:
      - sum_squares: sum of squares (value - mean)**2
      - deg_freedom: degrees of freedom
      - mean_squares: sum_squares / deg_freedom
    Attributes between_groups and between_subgroups also have:
      - f: F-value 
      - p: p-value from one-tailed F with the corresponding degrees of freedom

    Return in case arg output is 'dataframe': pandas.DataFrame that contains
    all the above data.
    """
    
    # prepare data
    data.reset_index(inplace=True)
    data.set_index(
        keys=[group_label, subgroup_label], inplace=True, append=False)
    data = data[[value_label]].copy()

    # initialize results
    class Dummy(object): pass
    results = Dummy()
    
    #
    # Within subgroups
    #
    
    # number of data points
    n_1 = data.groupby([group_label, subgroup_label]).count()
    n_1.rename(columns={value_label: 'here'}, inplace=True)
    n_1['total'] = n_1['here']
    
    # deg freedom
    dfree_1 = n_1['here'].sum() - n_1['here'].count()
    
    # means
    mean_1 = data.groupby([group_label, subgroup_label]).mean()
    mean_1.rename(columns={value_label: 'mean'}, inplace=True)
    
    # sums of squares and means of squares
    sum_sq_1_all = ((data[value_label] - mean_1['mean'])**2)
    sum_sq_1_all = sum_sq_1_all.groupby([group_label, subgroup_label]).sum()
    sum_sq_1 = sum_sq_1_all.sum()
    #print(f"sum_sq_1: {sum_sq_1}")
    ms_1 = sum_sq_1 / dfree_1
    #print(f"ms_1: {ms_1}")
    
    # add to results
    results.within_subgroups = Dummy()
    results.within_subgroups.sum_squares = sum_sq_1
    results.within_subgroups.mean_squares = ms_1
    results.within_subgroups.deg_freedom = dfree_1
    
    #
    # Within groups, between subgroups
    #
    
    # number of data 
    n_2 = n_1.groupby(group_label).count()[['here']]
    n_2['total'] = n_1.groupby(group_label).sum()['total']
    
    # deg freedom
    dfree_2 = n_2['here'].sum() - n_2['here'].count()
    
    # means
    mean_2 = data.groupby(group_label).mean()
    mean_2.rename(columns={value_label: 'mean'}, inplace=True)
    
    # sums of squares and means of squares
    sum_sq_2_all = ((mean_1['mean'] - mean_2['mean'])**2 * n_1['total'])
    sum_sq_2_all = sum_sq_2_all.groupby([group_label]).sum()
    sum_sq_2 = sum_sq_2_all.sum()
    #print(f"sum_sq_2: {sum_sq_2}")
    ms_2 = sum_sq_2.sum() / dfree_2
    #print(f"ms_2: {ms_2}")
    
    # add to results
    results.between_subgroups = Dummy()
    results.between_subgroups.sum_squares = sum_sq_2
    results.between_subgroups.mean_squares = ms_2
    results.between_subgroups.deg_freedom = dfree_2
    
    #
    # Between groups
    #
    
    # number of data 
    n_3 = pd.DataFrame(
        {"here": n_2['total'].count(), "total": n_2['total'].sum()}, index=[0])
    
    # deg freedom
    dfree_3 = n_3['here'].sum() - n_3['here'].count()
    
    # means
    mean_3 = pd.DataFrame({'mean': data[value_label].mean()}, index=[0])
    
    # sums of squares and means of squares
    sum_sq_3_all = (
        (mean_2['mean'] - mean_3.at[0, 'mean'])**2 * n_2['total']).sum()
    sum_sq_3 = sum_sq_3_all.sum()
    #print(f"sum_sq_3: {sum_sq_3}")
    ms_3 = sum_sq_3.sum() / dfree_3
    #print(f"ms_3: {ms_3}")
    
    # add to results
    results.between_groups = Dummy()
    results.between_groups.sum_squares = sum_sq_3
    results.between_groups.mean_squares = ms_3
    results.between_groups.deg_freedom = dfree_3
    
    #
    # F tests
    #
    
    # between subgroups
    f = ms_2 / ms_1
    p = sp.stats.f.sf(f, dfree_2, dfree_1)
    #print(f"Between subgroups within groups: F={f}, p={p}")
    results.between_subgroups.f = f 
    results.between_subgroups.p = p
    
    # between groups
    f = ms_3 / ms_2
    p = sp.stats.f.sf(f, dfree_3, dfree_2)
    #print(f"Between groups: F={f}, p={p}")
    results.between_groups.f = f 
    results.between_groups.p = p

    if output == 'struct':
        return results

    elif output == 'dataframe':
        for level, level_name in zip(
                [results.between_groups, results.between_subgroups,
                 results.within_subgroups],
                ['Between groups', 'Between subgroups', 'Within subgroups']):
            res_dict = {
                'Level': level_name, 'deg freedom': level.deg_freedom,
                'sum squares': level.sum_squares,
                'mean squares': level.mean_squares}
            try:
                res_dict.update({'F': level.f, 'p': level.p})
            except AttributeError:
                pass
            try:
                res_df = res_df.append(res_dict, ignore_index=True, sort=False)
            except NameError:
                res_df = pd.DataFrame(res_dict, index=[0])
        return res_df

    elif output == 'dataframe_line':
        res_dict = {
            'Between subgroups F': results.between_subgroups.f,
            'Between subgroups p': results.between_subgroups.p,
            'Between groups F': results.between_groups.f,
            'Between groups p': results.between_groups.p}
        res_df = pd.DataFrame(res_dict, index=[0])
        return res_df
    
    else:
        raise ValueError("Arg output has to be 'struct', or 'dataframe'.")
    
    return results
