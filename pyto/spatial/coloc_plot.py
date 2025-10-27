"""
Contains functions that plot colocalization analysis

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""

__version__ = "$Revision$"


import os
import re

import numpy as np
import scipy as sp
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt

import pyto
import pyto.spatial.coloc_functions as col_func
from .coloc_functions import get_layers, get_2_names, make_name
    

####################################################################
#
# Plotting and related functions
#

def make_nice_label(label, sets):
    """Makes nice looking labels for colocalization names. 

    Arg label is split in pieces separated by '_' and each piece is 
    substituted by the corresponding value of arg sets.

    Arguments:
      - label: (str) typically a colocalization name (e.g. 'pre_tether_post')
      - sets: (dict) substitution rules in the form of {'old_piece:
      'nice_piece'}

    Returns nice looking label
    """

    # check label only one name 
    if isinstance(label, (list, tuple)):
        if len(label) == 1:
            label = label[0]
        else:
            raise ValueError(
                f"Argument label {label} should contain only one name")

    # if mossible, replace the whole label, otherwise peaces
    nice = sets.get(label, None)
    if nice is not None:
        new_label = nice
    else:
        psets = get_layers(label)
        nice_sets = [sets.get(lay, lay) for lay in psets]
        new_label = ' - '.join(nice_sets)
    
    return new_label

def table_generator(coloc=None, name=None, groups=None, single=False):
    """Generator that makes an iterator over colocalization results.

    Each element returned by the iterator contains a label and a 
    coloclization table (pandas.DataFrame) where rows correspond to 
    colocalization distances.

    The following cases are implemented:

    1) One or more colocalization names, all tomos together

    Arg coloc specified, name contains one or more colocalization 
    names, single=False: The returned iterator contains a table 
    for each coloclization specified (arg name), for all tomos together.

    2) One colocalization name, each tomogram separately

    Arg coloc specified, name is one colocalization name, single=True:
    The returned iterator contains a table for each tomo (contained in
    colocalization data) separately.

    3) One colocalization name, each group separately, DataFrame version

    Arg coloc is None, arg groups is dict of labels (keys) and 
    colocalization data as DataFrames (values). Args name and single 
    are ignored: The returned iterator contains the specified 
    colocalization tables (values of arg groups).

    4) One colocalization name, each group separately, ColocAnalysis

    Arg coloc is None, arg groups is dict of group names (keys) and 
    colocalization data as ColocAnalysis objects (values). Arg name is 
    a colocalization name, while arg single is ignored: The returned 
    iterator contains the colocalization tables for the specified 
    colocalization names, for each of the colocalizations given in arg 
    groups. The specifeid colocalization name has to be present in all 
    colocalization objects. 

    Aguments:
      - coloc: (ColocAnalysis) colocalization object
      - name: one or more colocalization names
      - groups: dictionary where keys are group names and values are the
      corresponding colocalization tables
      - single: Flag indication if individual tomo data is returned, used 
      only if arg coloc is specified

    Returns iterator that in each iteration returns a label and the 
    corresponding colocalization data.
    """

    # sanity check
    if (coloc is None or name is None) and groups is None:
        raise ValueError(
            "Argument groups, or both coloc and name need to be specified.")
        
    # figire out data arguments
    if coloc is not None and name is not None and not single:
        
        # one coloc name, tomos together
        if isinstance(name, str):
            name = [name]
        data_gen = (
            (nam, getattr(coloc, nam + '_' + coloc.join_suffix))
            for nam in name)
                    
    elif coloc is not None and name is not None and single:

        # one coloc name, individual tomos
        tomos_tab = getattr(coloc, name + '_' + coloc.individual_suffix)
        data_gen = (
            (tid, tomos_tab[tomos_tab.id == tid].sort_values(by='distance')) 
            for tid in tomos_tab['id'].unique())
                                                              
    elif groups is not None:

        # multiple coloc objects / tables
        from .coloc_analysis import ColocAnalysis
        if np.all(
            [isinstance(value, pd.DataFrame) 
             for value in groups.values() if value is not None]):
            
            # multiple coloc tables
            data_gen = groups.items()
            
        elif np.all(
            [isinstance(value, ColocAnalysis)  
             for value in groups.values() if value is not None]):
              
            # multiple coloc objects
            if name is not None and isinstance(name, str):
                data_gen = (
                    (gr, getattr(col, name + '_' + col.join_suffix)) 
                    for gr, col in groups.items() if col is not None)
            else:
                raise ValueError(
                    "Arg name has to be specified and it has to be a str when "
                    + "arg groups contains colocalization objects "
                    + "(ColocAnalysis).")
                
        else:
            raise ValueError("Problem with coloc, names and groups arguments.")
    else:
        raise ValueError("Problem with coloc, names and groups arguments.")
             
    return data_gen
            
def plot_p(
        coloc=None, name=None, groups=None, single=False,
        y_var='p_subcol_combined', tomos=None, sets={}, pp=None, ax=None,
        **fig_kw):
    """Plots p-values for one colocalization. 

    """
    return plot_data(
        coloc=coloc, name=name, groups=groups, single=single,
        y_var=y_var, tomos=tomos, simulated={}, normalize=None,
        sets=sets, pp=pp, ax=ax, **fig_kw)
        
def plot_data(
        coloc=None, name=None, groups=None, single=False,
        y_var='n_subcol', tomos=None, simulated={}, normalize=None,
        sets={}, pp=None, ax=None, **fig_kw):
    """Plots data for one colocalization. 

    Args coloc, name, groups and single are used to select colocalization
    data, as explained in table_generator() doc. 

    If arg y_vars contain multiple values, they should be either p-values
    related ('p_subcol_normal', 'p_subcol_other' and 'p_subcol_combined'), 
    or other variables.

    Produces nice looking plots for default values of colocalization 
    parameters related to simulation suffixes, both for the standard 
    simulations ('normal' and 'other') and for all random simulations
    (see ColocLite() arg all_random and method set_simulation_suffixes().
    """

    # figure out what is plotted
    join_coloc = False
    if (coloc is not None) and not single:
        join_coloc = True        
    one_coloc = False
    if (name is not None) and (isinstance(name, str) or len(name) == 1):
        one_coloc = True
        n_y_vars = 1
    if isinstance(y_var, str):
        y_var_list = [y_var]
    elif len(y_var) == 1:
        y_var_list = y_var
        y_var = y_var_list[0]
    else:
        n_y_vars = len(y_var)
        y_var_list = y_var

    # figure out colors for plotting p (just to keep simulated consistent)
    p_plot = False
    p_vars = ['p_subcol_normal', 'p_subcol_other', 'p_subcol_combined']
    if not set(p_vars).isdisjoint(set(y_var_list)):
        p_plot = True
        colors = {'p_subcol_normal': 'C0', 'p_subcol_other': 'C1'}
        if not set(p_vars).issuperset(y_var_list):
            raise ValueError(
                "Arg y_vars cannot contain both p-value and other variables.")

    p_var_all_random = 'p_subcol_solo'
    if p_var_all_random in y_var_list:
        p_plot = True
        colors = {'p_subcol': 'C0'}
        if len(y_var_list) > 1:
            raise ValueError(
                "Arg y_vars cannot contain both p-value and other variables.")
       
    # plotting one coloc, with simulations
    data_simul_plot = False
    if one_coloc and (n_y_vars == 1) and (len(simulated) > 0):
        colors = {}
        if 'normal' in simulated:
            colors['normal'] = 'C0'
        if 'alt' in simulated:
            colors['alt'] = 'C1'
        if 'other' in simulated:
            colors['other'] = 'C1'
        data_simul_plot = True
            
    # start plot
    if ax is None:
        fig, ax = plt.subplots()
    x_var = 'distance'

    # get tables according to arguments
    data_gen = table_generator(
        coloc=coloc, name=name, groups=groups, single=single)

    # plot, loop over all 
    for label, table in data_gen:

        # skip if no data or tomo not in the list
        if table is None:
            continue
        if single and (tomos is not None) and (label not in tomos):
            continue

        # normalization
        if (normalize is None) or (not normalize):
            area = 1
        else:
            area = np.pi * table.distance**2 
        
        label_nice = make_nice_label(label, sets=sets)
        if n_y_vars == 1:

            #try:
                #color = pp.color.get(y_var)
                #color = colors.get(y_var)
            #except (AttributeError, KeyError):
                #color = None

            # plot one coloc feature
            y_var_nice = make_nice_label(y_var, sets=sets)
            if join_coloc and one_coloc:
                plot_label = y_var_nice
            else:
                plot_label = label_nice
            if p_plot:

                # plot p-values
                #if not single or (groups is not None):
                ax.plot(
                    x_var, y_var, 'x', data=table, linestyle='-',
                    label=plot_label, **fig_kw)

            elif data_simul_plot:

                # plot data other than p-values with simulations
                ax.plot(
                    table[x_var], (table[y_var]/area), 'o', linestyle='',
                    color="C2", label=y_var_nice, **fig_kw)
                for simul_name, y_simul in simulated.items():
                    y_simul_var = y_simul + '_mean'
                    y_simul_err = y_simul + '_std'
                    ax.errorbar(
                        table.distance, table[y_simul_var]/area,
                        yerr=table[y_simul_err]/area, 
                        fmt='x', color=colors.get(simul_name),
                        label=f'Simulations {simul_name}', **fig_kw)

            else:

                # plot data other than p-values without simulations
                ax.plot(
                    table[x_var], table[y_var]/area, 'o', linestyle='',
                    label=plot_label, **fig_kw)
                
        else:

            # plot multiple coloc features
            for y_var_one in y_var:
                y_var_nice = make_nice_label(y_var_one, sets=sets)
                if one_coloc:
                    plot_label = y_var_nice
                else:
                    plot_lab = label_nice + " " + y_var_nice
                try:
                    #color = pp.color.get(y_var_one)
                    color = colors.get(y_var_one)
                except (AttributeError, KeyError):
                    color = None
                ax.plot(
                    x_var, y_var_one, 'x', data=table, linestyle='-',
                    color=color, label=plot_label, **fig_kw)
                
        distances = table[x_var].unique()

    # make title
    if one_coloc:
        title_main = make_nice_label(name, sets=sets)
    elif join_coloc and (n_y_vars == 1):
        title_main = make_nice_label(y_var, sets=sets)
    else:
        title_main = "Multiple colocalizations"

    # p-plot limits
    if p_plot:
        ax.plot(
            [distances[0], distances[-1]], [0.95, 0.95], 'k', linestyle='--',
            **fig_kw)
        ax.set_ylim(-0.05, 1.05)

    # finish plot
    ax.legend(loc='best')
    ax.set_xlabel('Distance [nm]')
    if p_plot:
        ax.set_ylabel('1 - p value')
    elif normalize is None:
        ax.set_ylabel("Number")
    else:
        ax.set_label('Surface density [$1/nm^2$]')
    ax.set_title(f'{title_main}')        
       
    return ax

def plot_32_p(
        name, coloc=None, groups=None, single=False,
        y_var='p_subcol_combined', tomos=None, sets={},
        ax=None, figsize=(15, 3)):
    """
    """
    return plot_32_data(
        name=name, coloc=coloc, groups=groups, single=single,
        y_var=y_var, tomos=tomos, simulated={}, normalize=None,
        sets=sets, ax=ax, figsize=figsize)

def plot_32_data(
        name, coloc=None, groups=None, single=False,
        y_var='n_subcol', tomos=None, simulated={}, normalize=None,
        sets={}, ax=None, figsize=(15, 3)):
    """
    """

    # start plot
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=figsize)

    # plot 3-col
    plot_data(
        coloc=coloc, name=name, groups=groups, single=single,
        y_var=y_var, tomos=tomos, simulated=simulated, normalize=normalize,
        sets=sets, ax=ax[0])

    # figure out 2-colocalizations
    name_0_1, name_0_2 = get_2_names(
        name=name, order=((0, 1), (0, 2)), by_order=True)
        
    # plot 2-colocalizations
    try:
        plot_data(
            coloc=coloc, name=name_0_1, groups=groups, single=single,
            y_var=y_var, tomos=tomos, simulated=simulated, normalize=normalize,
            sets=sets, ax=ax[1])
    except AttributeError:
        pass
    try:
        plot_data(
            coloc=coloc, name=name_0_2, groups=groups, single=single,
            y_var=y_var, tomos=tomos, simulated=simulated, normalize=normalize,
            sets=sets, ax=ax[2])
    except AttributeError:
        pass

    return ax
    
def plot_data_old(
        coloc, name, y_var='n_subcol', simulated={}, normalize=None, 
        suffix='data', mode='_', sets={}, ax=None):
    """Plots colocalization data.

    Usage:

      1) One or more colocalization names from a colocalization object:
        - coloc: colocalization object
        - name: (str or list of strings): one or more colocalization names

      2) One colocalization name from a table:
        - coloc: (pandas.DafaFrame) data for one coloc
        - name: (str) name for the coloc
    """
    
    # plot data
    if ax is None:
        fig, ax = plt.subplots()
    multi_names = True
    if isinstance(name, str):
        name = [name]
        multi_names = False

    # figure out colors (just to keep simulated consistent)
    if ((len(simulated) == 2) and ('normal' in simulated)
        and (('alt' in simulated) or ('other' in simulated))):
        colors = {'normal': 'C0', 'alt': 'C1', 'other': 'C1'}
        n_simul = 2
    else:
        n_simul = 0
        
    for ind, nam in enumerate(name):
        
        # get data and area
        if isinstance(coloc, pd.DataFrame):
            data = coloc
            if multi_names:
                raise ValueError(
                    "Because arg coloc is pandas.DataFrame, arg name "
                    + "can only be a single colocalization name.")
        else:
            data = getattr(
                coloc, make_name(names=[nam], suffix=suffix, mode=mode))
        if normalize is None:
            area = 1
            y_label = "Number"
        elif normalize == 'circle_area':
            area = np.pi * data.distance**2 
            y_label = 'Surface density [$1/nm^2$]'
            
        # plot data
        if multi_names:
            label = f"Data {make_nice_label(nam, sets)}"
        else:
            label = "Data"
        ax.plot(
            data.distance, data[y_var]/area, 'o', linestyle='', 
            color=f"C{ind+n_simul}", label=label)   
    
    # plot simulations
    for lab, y_simul in simulated.items():
        y_simul_var = y_simul + '_mean'
        y_simul_err = y_simul + '_std'
        ax.errorbar(
            data.distance, data[y_simul_var]/area, yerr=data[y_simul_err]/area, 
            fmt='x', color=colors.get(lab),
            label=f'Simulations {make_nice_label(lab, sets)}')
            
    # finish plot
    ax.legend(loc='best')
    ax.set_xlabel('Distance [nm]')
    ax.set_ylabel(y_label)
    ax.set_ylim(-0.02, ax.set_ylim()[1])
    if multi_names:
        title = f"{sets.get(y_var, y_var)}"
    else:
        title = f"{sets.get(y_var, y_var)} in {make_nice_label(name[0], sets)}"
    ax.set_title(title)
    
    return ax

def plot_32_data_old(
        name, coloc=None, y_var='n_subcol', simulated={}, normalize=None,
        suffix='data', mode='_', sets={}, ax=None, figsize=(15, 3)):
    """
    """

    # start plot
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=figsize)

    # plot 3-col
    plot_data_old(
        coloc=coloc, name=name, simulated=simulated, normalize=normalize, 
        y_var=y_var, suffix=suffix, mode=mode, sets=sets, ax=ax[0])

    # figure out 2-colocalizations
    name_0_1, name_0_2 = get_2_names(
        name=name, order=((0, 1), (0, 2)), by_order=True)
        
    # plot 2-colocalizations
    try:
        plot_data_old(
            coloc=coloc, name=name_0_1, simulated=simulated,
            normalize=normalize, 
            y_var=y_var, suffix=suffix, mode=mode, sets=sets, ax=ax[1])
    except AttributeError:
        pass
    try:
        plot_data_old(
            coloc=coloc, name=name_0_2, simulated=simulated,
            normalize=normalize, 
            y_var=y_var, suffix=suffix, mode=mode, sets=sets, ax=ax[2])
    except AttributeError:
        pass

    return ax
    
def plot_multiple(
        coloc, names, tomos=None, single=True, y_var='n_subcol',
        normalize=False, y_var_p=['p_subcol_normal', 'p_subcol_other'],
        simulated={'normal': 'n_subcol_random',
                   'alt': 'n_subcol_random_alt' },
        names_inner=True, sets={},
        n_columns=4, fig_width=12, fig_height_one=2.5,
        layout='constrained', subplot_kw=None, **fig_kw): #**subplot_kw):
    """Plots data and p graphs for multiple colocalization, tomo combinations.  

    Useful to see colocalization data for many different colocalization
    cases and tomos together.
    
    The corresponding data and p-value graphs are always placed one above
    the other.
    
    To show individual tomo data (on separate graphs), arg single has to be
    True. The tomos to show can be specified by arg tomos, or to show all
    tomos contained in arg coloc, tomos should be None.

    To show joined tomo data arg single has to be False and tomos None. 
    
    Arguments:
      - coloc: (ColocalizationAnalysis) colocalization results
      - names: (list) colocalization names
      - tomos: list of tomos or None
      - single: flag indicating whether data from individual tomos are
      ploted on separate graphs
      - y_var: variable plotted on (y-axis of) data graphs (default 'n_subcol')
      - simulated: (dict) simulation variables also plotted on data graphs
      (default {'normal': 'n_subcol_random', 'alt': 'n_subcol_random_alt' }
      - normalize : flag indication if the above data graph variables should
      be radially normalized (sometimes convenient to show data better,
      default False)
      - y_var_p: (list) variables plotted on (y-axis of) p-value
      graphs (default =['p_subcol_normal', 'p_subcol_other'])
      - names_inner: flag indicating if data for the same colocalization
      case but multiple tomos are placed together, otherwise data for the
      same tomo but different colocalization cases are presented together
      (default True)
      - sets: (dict) nicer looking labes for variables
      - n_columns: number of graphs in a row
      - fig_width: figure width passed to figsize argument of plt.subplots()
      (default 12)
      - fig_height_one: height of one graph (default 2.5)
      - layout: layout passed to matplotlib.pyplot.subplots (default
      'constrained', other possibilities 'compressed', 'tight', 'none')
      - subplot_kw: (dict) additional keyword arguments passed to
       matplotlib.pyplot.subplots (should not contain layout because
      specified above)
      - **fig_kw: Work in progress additional keyword arguments passed to
      matplotlib.pyplot.subplots
    """

    # organize tomos and colocalization names
    if isinstance(names, str):
        names = [names]
    if tomos is None:
        if single:
            tomos = coloc.get_data(name=names[0])[1]['id'].unique()
        else:
            tomos = [None]
    elif isinstance(tomos, str):
        tomos = [tomos]
    tomos_single = True if len(tomos) == 1 else False
    if names_inner:
        cases = [(nam, tom) for tom in tomos for nam in names]
    else:
        cases = [(nam, tom) for nam in names for tom in tomos]
    names_single = True if len(names) == 1 else False

    # graph organization
    n_cases = len(cases)
    n_rows_half = (n_cases - 1) // n_columns + 1
    n_rows = 2 * n_rows_half
    if subplot_kw is None:
        subplot_kw = {}
    subplot_kw['layout'] = layout
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(fig_width, fig_height_one * n_rows),
        #constrained_layout=True, sharex=True, sharey=False)
        sharex=True, sharey=False, **subplot_kw)

    first_time = True
    for case_ind, (nam, tom) in enumerate(cases):

        # find ax
        case_row_ind = case_ind // n_columns
        column_ind = np.remainder(case_ind, n_columns)
        ax_above = axes[2*case_row_ind, column_ind]
        ax_below = axes[2*case_row_ind + 1, column_ind]

        # actual plots
        fig_kw = {}  # tmp workaround
        plot_data(
            coloc=coloc, name=nam, simulated=simulated, tomos=tom, 
            single=single, normalize=normalize, sets=sets, ax=ax_above,
            **fig_kw)
        plot_p(
            coloc=coloc, name=nam, y_var=y_var_p, tomos=tom, 
            single=single, sets=sets, ax=ax_below, **fig_kw)
        if not tomos_single and not names_single:
            ax_above.set_title(
                f"{tom}\n{nam}" if names_inner else f"{nam}\n{tom}")
        elif tomos_single:
            ax_above.set_title(f"{nam}" if names_inner else f"{tom}")
        elif names_single:
            ax_above.set_title(f"{tom}" if names_inner else f"{nam}")
        ax_above.set_xlabel("")
        ax_below.set_title("")

        # only one legend
        if not first_time:
            ax_above.get_legend().remove()
            ax_below.get_legend().remove()
        first_time = False

    # remove unused graphs and add overall title
    for no_case_ind in range(column_ind+1, n_columns):
        axes[n_rows-2, no_case_ind].set_visible(False)
        axes[n_rows-1, no_case_ind].set_visible(False)
    if (tomos_single and (tomos[0] is not None)) or names_single:
        big_title = tomos[0] if tomos_single else names[0]
        fig.text(0.5, 1.01, big_title, ha='center')
    
    return fig, axes
