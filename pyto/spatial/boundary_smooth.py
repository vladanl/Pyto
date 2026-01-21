"""
Contains class BoundarySmooth.

# Author: Vladan Lucic
# $Id: boundary.py 2268 2026-01-20 17:29:20Z vladan $
"""

__version__ = "$Revision: 2268 $"

import numpy as np
import scipy as sp
import pandas as pd

import pyto
from pyto.geometry.vector import Vector


class BoundarySmooth:
    """BoundarySmooth class.

    Containes method morphology_pipe() that can be used to smooth
    a segment by binary morphological operations.
    """

    def __init__(self, image, segment_id, external_id=None, bkg_id=0):
        """Sets attributes from args.

        Arguments:
          - image: (ndarray or pyto.segmentation.Labels) image containg
          the segment that should be smoothed
          - segment_id: (int) id of the segment that should be smoothed
          - external_id: (int, lust, tuple) id(s) of one or more segmentes
          that contacts the smoothed segment (default None)
          - bkg_id: background id (default 0)
        """

        # attributes
        if isinstance(image, pyto.core.Image):
            self.image = image.data
        else:
            self.image = image
        self.n_dim = self.image.ndim
        self.segment_id = segment_id
        self.external_id = external_id
        self.bkg_id = bkg_id

        self.operation_dict = {'e': 1, 'd': -1}

    def morphology_pipe(self, operations, erosion_border=1, multiply=None):
        """Applies a series of binary morphological operations.

        Intended to smooth one segment (defined by self.segment_id).
        Currently implemented for binary erosion in dilation only. Uses
        scipy.ndimage.binary_erosion() and scipy.ndimage.binary_dilation().
        
        Only segment defined by self.segment_id is smoothed. Generally,
        smoothing assignes some pixels that in the input image belonged
        to the background or other segment pixels, to the smoothed segment.

        Pixels that are removed from the segment (that is to be smoothed) by
        smoothing are first assigned to background (self.bkg_id). In
        cases when in the input image the smoothed and another segment
        contact each other, smoothing may result in background pixels
        placed between the smoothed and the other segment. To remedy this,
        an external segment (defined by self.external_id) is dilated over
        the background pixels that belong to the segment to be smoothed
        The extent of this dilation is calculated by the maximal cumulative
        extent of erosions during smoothing. For examplem it is:
          - 1 if operations = 'ed'
          - 0 if operations = 'de'
          - 2 if operations = 'eeddddee'
          - 2 if operations = 'ddeeeedd'       

        For example, if an image was expanded by an integer factor (inverse
        of binning) it is recommended to use the following arg operations:
          - expansion factor 2: operations = 'deed'
          - expansion factor 4: operations = 'ddeeeedd'
        
        Requred attributes:
          - image: (np.ndimage or pyto.core.Image) input image
          - segment_id
          - bkg_id
        
        Arguments:
          - operations: (str or list of chars) series of morphological
          operators, currently implemented 'e' for erosion and
          'd' for dilation, in the order they should be applied
          - erosion_border: border valus used for erosion (passed directly
          to scipy.ndimage.binary_erosion(), arg border_value, default 1)
          - mutiply: if not None, the returned image is obtained by
          multiplying the processed image by this factor and adding
          the input image (default None)

        Returns:
          - If arg multiply is None: processed image
          - If arg multiply is not None: 
            multiply * processed_image + image
        """

        # deal with multiple segment ids
        multi_segment_id = False
        if isinstance(self.segment_id, (list, tuple)):
            multi_segment_id = True
            image = self.image.copy()
            for seg_id in self.segment_id:
                local = self.__class__(
                    image=image, segment_id=seg_id,
                    external_id=self.external_id, bkg_id=self.bkg_id)
                image = local.morphology_pipe(
                    operations=operations, erosion_border=erosion_border,
                    multiply=None)
            if multiply is not None:
                image = multiply * image + self.image
            return image
       
        # smooth binary
        image_loc = (self.image == self.segment_id)
        for op in operations:
            if op == 'e':
                image_loc = sp.ndimage.binary_erosion(
                    image_loc, border_value=erosion_border)
            elif op == 'd':
                image_loc = sp.ndimage.binary_dilation(
                    image_loc, border_value=0)
        image = self.image.copy()
        image[image==self.segment_id] = self.bkg_id
        image[image_loc] = self.segment_id

        # adjust external id
        if (self.external_id is not None) and (self.external_id != self.bkg_id):

            # prepare
            n_dilations = np.add.accumulate(
                [self.operation_dict[op] for op in operations]).max()
            if isinstance(self.external_id, int):
                external_ids = [self.external_id]
            else:
                external_ids = self.external_id

            # put external id where boundary retracted close to external
            for _ in range(n_dilations):
                for ext_id in external_ids:
                    dilated_ext = sp.ndimage.binary_dilation(
                        self.image == ext_id, border_value=0)
                    new_ext = (dilated_ext & (image == self.bkg_id)
                               & (self.image == self.segment_id))
                    image[new_ext] = ext_id
            
        if (multiply is not None) and not multi_segment_id:
            image = multiply * image + self.image

        return image
