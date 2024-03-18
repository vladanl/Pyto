"""
Useful pandas-like functions
 
# Author: Vladan Lucic
# $Id$
"""

__version__ = "$Revision$"

import pandas as pd


def merge_left_keep_index(
        left, right, on=None, left_on=None, right_on=None,
        left_index=False, right_index=False, sort=False,
        suffixes=("_x", "_y"), copy=True, indicator=False,
        validate='many_to_one'):
    """Left merge with keeping the index of the left.

    Identical to pandas.merge(how='left') except that index of left is 
    kept in the resulting table.

    Arguments are the same as for pandas.merge, except:
      - on: always 'left'
      - sort: default False
      - vaidate: default 'many_to_one'

    Returns table resulting from merge
    """

    left_index_name = left.index.name
    if left_index_name is None:
        tmp_left_index_name = 'index'
    else:
        tmp_left_index_name = left_index_name
    result = (
        left
        .reset_index()
        .merge(
            right, on=on, how='left', left_on=left_on, right_on=right_on,
            left_index=left_index, right_index=right_index, sort=sort,
            suffixes=suffixes, copy=copy, indicator=indicator,
            validate=validate)
        .set_index(tmp_left_index_name))
    result.index.name = left_index_name
    
    return result

