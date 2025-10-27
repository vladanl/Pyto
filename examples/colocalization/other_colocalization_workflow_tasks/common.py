"""
Functions needed for notebooks in this directory.

Author: Vladan Lucic
"""

import numpy as np
import scipy as sp
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt

import pyto


def show_2d_slices(
    image, n_cols=None, vmin=0, vmax=1, figwidth=None, figheight=None, 
    fig=None, axes=None):
    """Plots z-slices from 3D images.

    Arguments:
      - image: (numpy.ndarray) input image
      - n_cols: (int) number of columns in the figure
      - vmin, vmax: min and max greyscale values for plotting images
        passed directly to matplotlib.pyplot.imshow (default 0, 1, 
        respectively)
        - figwidth, figheight: figure width and hight, passed directly
        to matplotlib.pyplot.subplots()
      - fig, axes: Figure and Axes from another plot, if None new 
        figure is made her (default None, None)

    Returns: fig, axes for this plot
    """

    # figure out size
    x_len, y_len, z_len = image.shape
    if n_cols is not None:
        n_rows = (z_len - 1) // n_cols + 1    
    else:
        n_rows = 1
        n_cols = z_len
    if (figheight is None) and (figwidth is None):
        figsize = None
    else:
        if figheight is None:
            figheight = n_rows * figwidth / n_cols
        elif figwidth is None:
            figwidth = n_cols * figheight / n_rows
        figsize = (figwidth, figheight)

    if (fig is None) or (axes is None):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    try:
        axes_1d = axes.reshape(-1)
    except AttributeError:
        axes_1d = [axes]
        
    for z_ind in range(z_len):
        ax = axes_1d[z_ind]
        ax.imshow(
            image[:, :, z_ind].transpose(), cmap='Greys', origin='lower',
            vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_box_aspect(1)

    return fig, axes    
