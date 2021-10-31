"""
Class Parallelogram provides methods for creating parallelograms in N-dim.
(N >=2). Both the space and the parallelogram are N-dimensional.


# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""

__version__ = "$Revision$"

import itertools
import numpy as np

from .plane import Plane


class Parallelogram(object):
    """
    """

    def __init__(
            self, origin=None, basis=None, surface_label=0, outside_label=1,
            inside_label=-1):
        """
        Sets attributes
        """

        if origin is not None:
            self.origin = origin
        if basis is not None:
            self.basis = basis
        self.surface_label = surface_label
        self.outside_label = outside_label
        self.inside_label = inside_label

    @classmethod
    def make(
            cls, shape, corners=None, origin=None, basis=None,
            thick=1, eps=1e-9, surface_label=0, outside_label=1,
            inside_label=-1, dtype='int'):
        """
        Makes a N-dimensional parallelogram in the N-dim space (N>1) and
        returns the corresponding instance of this class. The image 
        containing this parallelogram is saved as attribute data 
        (N-dimensional ndarray).

        The parallelogram is defined by origin and basis vectors where:
          - origin (1D array of length N) contains coordinates of
          one corner
          - basis vectors (2D array NxN, axis 0 represents points and
          axis 1 dimensions) are coordinates of the N corners that are
          connected to the origin by one edge of the rectangle.

        Origin and basis vectors are determined by the first
        criterion from the following that matches:
          - If arg corners is specified, it has to be N+1 x N array,
          where corners[0, :] are coordinates of the origin and
          corners[1:,:] are coordinates of the basis vectors
          - If args corners and origin are specified, these two are used

        In the image containing the parallelogram, the surface of the
        parallelogram is labeled by 0, the interior by -1 and exterior
        by 1. The surface thickness is specified by arg thick.

        Specifically, the parallelogram is constructed using N-1 dim
        planes defined by the origin and all except one of the basis
        vectors (so N planes), and then a parallel plane that contains
        the omitted basis is constructed for each of the (initial) planes
        (see also Plane.make()).

        Arguments:
          - shape (1D array of len N): shape of the resulting image
          - corners (N+1 x N array) that specifies coordinates of the
          origin (corners[0,:] and N basis vectors (corners[1:,:])
          - origin (1D array of len N): coordinates of the origin corner
          - basis (NxN array, axis 0 denotes points): coordinates of
          basis corners (vectors)
          - thick: thickness of N-1 dim planes forming the surface of
          the parallelogram (+/- thick/2 from the exact plane)
          - dtype: image data type
          - eps: Used to correct for numerical errors in labeling the

        Returns: Instance of this class with attributes:
          - data (N-dim array): image containing the parallelogram
        """

        # parse args
        if corners is None:
            if (origin is not None) and (basis is not None):
                corners = np.vstack((origin, basis))
            else:
                raise ValueError(
                    "Argument corners, or arguments origin and basis have to"
                    + "be specified")
        else:
            corners = np.asarray(corners)
            origin = corners[0]
            basis = corners[1:]

        # initialize and make image
        inst = cls(
            origin=origin, basis=basis, surface_label=surface_label,
            outside_label=outside_label, inside_label=inside_label)
        inst.make_image(shape=shape, thick=thick, eps=eps, dtype=dtype)

        return inst

    def make_image(self, shape, thick=1, eps=1e-9, dtype='int'):
        """
        Makes a N-dimensional parallelogram in the N-dim space (N>1) and
        returns the corresponding instance of this class. The image 
        containing this parallelogram is saved as attribute data 
        (N-dimensional ndarray).

        Makes an image of the current N-dimensional parallelogram in the 
        N-dim space (N>1) and saves it as attribute data (N-dimensional 
        ndarray).

        This is a instance method version of the class method make(), see
        make() docs for further info.

        Attributes origin and basis need to be specified before calling
        this method.

        Sets:
          - self.data

        Arguments:
          - shape (1D array of len N): shape of the resulting image
          - thick: thickness of N-1 dim planes forming the surface of
          the parallelogram (+/- thick/2 from the exact plane)
          - eps: Used to correct for numerical errors in labeling the
          - dtype: image data type
        """

        # loop over basis corners
        self.data = np.ones(shape=shape, dtype=dtype) * self.inside_label
        corners = np.vstack((self.origin, self.basis))
        for index in range(1, corners.shape[0]):

            # make two planes for current points
            npoints = np.delete(corners, index, axis=0)
            point = corners[index]
            plane_orig = Plane.make(
                shape=shape, npoints=npoints, thick=thick, eps=eps)
            plane_parallel = Plane.make(
                shape=shape, normal=plane_orig.normal, point=point,
                thick=thick, eps=eps)

            # reorient
            side = plane_orig.get_side(external=point)
            if side > 0:
                plane_orig.data = -plane_orig.data
            elif side < 0:
                plane_parallel.data = -plane_parallel.data
            elif side == 0:
                raise ValueError(
                    "Could not orient plane because the specified external "
                    + "point is actually on the plane.")

            # combine planes
            planes_0_cond = (
                (plane_orig.data * plane_parallel.data == 0)
                | (self.data == self.surface_label))
            planes_in_cond = (
                (plane_orig.data == -1) & (plane_parallel.data == -1)
                & (self.data == self.inside_label))
            planes_out_cond = (
                (plane_orig.data == 1) | (plane_parallel.data == 1)
                | (self.data == self.outside_label))
            self.data[planes_0_cond] = self.surface_label
            self.data[planes_in_cond] = self.inside_label
            self.data[planes_out_cond] = self.outside_label

    def get_all_corners(self, origin=None, basis=None):
        """
        Returns all corners of the parallelogram defined by (arg) origin
        and basis corners (arg basis).

        If args origin and basis are None, attributes self.origin and
        self.basis are used.

        Arguments:
          - origin (1D array of len N): coordinates of the origin corner
          - basis (NxN array, axis 0 denotes points): coordinates of
          basis corners (vectors)

        Returns coordinates of all 2^N corners as (2^N x N) array
        (includes corners specifies in origin and basis).
        """

        # use arguments or attributes
        if origin is None:
            origin = self.origin
        if basis is None:
            basis = self.basis

        # initialize
        origin = np.asarray(origin)
        basis = np.asarray(basis)
        n_bases, ndim = basis.shape

        # corners: all possible additions between basis-origin vectors
        select_bases = list(itertools.product([0, 1], repeat=n_bases))
        n_terms = np.sum(select_bases, axis=1)
        all_corners = np.dot(select_bases, basis) - np.outer(n_terms-1, origin)

        return all_corners

    def is_equivalent(self, other):
        """
        Check if this object is equivalend to the specified Parallelogram
        (arg paral).

        By definition, parallelograms are equivalent if they have the same
        corners.

        Argument:
          - other: other Parallelogram

        Returns True if equivalent, False is not
        """

        self_corners = self.get_all_corners()
        other_corners = other.get_all_corners()
        self_corners.sort(axis=0)
        other_corners.sort(axis=0)
        
        same = (self_corners == other_corners).all()

        return same

    def get_bounding_box(
            self, corners=None, origin=None, basis=None, form='min-max',
            extend=None, shape=None):
        """
        Returns bounding box (rectangle aligned with the coordinate system
        axes) for the parallelogram defined by arguments.

        The parallelogram is defined from args corners, origin and basis
        as explained in make() docstring. In addition, if none of these
        is specified, attributes self.origin and self.basis are used.

        It arg extend is None, the tight bounding box is returned. If
        (arg) extend is specified, the tight box is extended by the
        specified value(s) in exch direction.

        The bounding box coordinates are returned in the format
        determined by arg form, as follows:
          - 'min-max_exact': Nx2 array containing the minimim and maximum
          coordinate of the bounding box for each of the N dimensions,
          where these are calculated exactly (can be floats)
          - 'min-max': The same as for 'min-max_exact', except that the
          int coordinates are returned, obtained by rounding down (floor
          of) the mins and  rounding up (ceiling of) the maxima.
          - 'min-len': the same as 'min-max' except that minimum and the
          length in each dimension are cpecified
          - 'slice' (list of N slice objects) having the same info as
          'min-max'

       Arguments:
           - corners (N+1 x N array) that specifies coordinates of the
          origin (corners[0,:] and N basis vectors (corners[1:,:])
          - origin (1D array of len N): coordinates of the origin corner
          - basis (NxN array, axis 0 denotes points): coordinates of
          basis corners (vectors)
          - form: format in which the bounding box is returned
          - extend (single number of 1D array of length N): the amount of
          bounding box extension in respect to the tight box (same in
          all directions or specified separately)
          - shape: image shape, used to limit the bounding box
          coordinates determined

         Return bounding box coordinates
        """

        # get corners
        if corners is None:

            # origin and basis from  arguments or attributes
            if origin is None:
                origin = self.origin
            if basis is None:
                basis = self.basis

            # get corners
            corners = self.get_all_corners(origin=origin, basis=basis)

        # calculate box
        min_coords = corners.min(axis=0)
        max_coords = corners.max(axis=0)
        if extend is not None:
            extend = np.asarray(extend)
            min_coords -= extend
            max_coords += extend
        if form != 'min-max_exact':
            min_coords = np.floor(min_coords).astype(int)
            max_coords = np.ceil(max_coords).astype(int)
        if shape is not None:
            shape_coords = np.asarray(shape) - 1
            min_coords = np.where(min_coords >= 0, min_coords, 0)
            max_coords = np.where(
                max_coords <= shape_coords, max_coords, shape_coords)

        # return in required format
        if (form == 'min-max') or (form == 'min-max_exact'):
            res = [[min_one, max_one] for min_one, max_one
                   in zip(min_coords, max_coords)]
        elif form == 'min-len':
            res = [[min_one, max_one + 1 - min_one]
                   for min_one, max_one in zip(min_coords, max_coords)]
        elif form == 'slice':
            res = [slice(min_one, max_one + 1)
                   for min_one, max_one in zip(min_coords, max_coords)]
        else:
            raise ValueError(
                "Argument format ({}) was not recognized".format(form))
        res = np.asarray(res)
        return res

    @classmethod
    def from_bounding_box(cls, box, offset=0):
        """
        Finds origin and bases for a given bounding box.

        Arguments:
          - box: bounding box in the form [[x_min, x_max], [y_min, y_max], ...]
          - offset: adjusts for the origin of the coordinate system in which
          arg box is give, for example, if the system starts at 1 offset
          should be 1, while for system starting at 0 it should be 0

        Retruns instance of this class containing attributes origin and basis.
        """

        # offset
        box = np.asarray(box).transpose() - offset

        # get origin
        origin = box[0]

        # get basis
        for ax_index in range(box.shape[1]):
            local_basis = box[0].copy()
            local_basis[ax_index] = box[1, ax_index]
            try:
                basis = np.vstack((basis, local_basis))
            except (NameError, ValueError):
                basis = local_basis[np.newaxis, :]

        inst = cls(origin=origin, basis=basis)
        return inst

    @classmethod
    def get_previous_bounding_box(
            cls, box, tf, center, offset=0, shape=None, extend=None,
            form='min-max'):
        """
        Given a (final) bounding box (arg box) that is obtained by
        transforming an inial rectangle, this method returns the initial
        bounding box that contains all data in the bounding box.

        The tranformation is specified by arg tf and the transformation
        center by arg center.

        Intended for the following scenario. If an image is rotated
        (or more generally affine transformed) so that the feature of
        interest is confined to a rectangular box, this method returns
        the box for the original (non-transformed) image containing the
        entire feature of interest.

        Arguments:
          - box: bounding box in the form [[x_min, x_max], [y_min, y_max], ...]
          - tf: transformation that yeilded the final box (arg box)
          - center: coordinates of the transformation center
          - offset: adjusts for the origin of the coordinate system in which
          arg box is give, for example, if the system starts at 1 offset
          should be 1, while for system starting at 0 it should be 0
          - extend (single number of 1D array of length N): the amount of
          bounding box extension in respect to the tight box (same in
          all directions or specified separately)
          - shape: image shape, used to limit the bounding box
          coordinates determined
          - form: format in which the bounding box is returned (see doc for
          get_bounding_box()

        Returns the initial bounding box in the form 
          [[x_min, x_max], [y_min, y_max], ...]
        """

        # final box
        rot_crop = Parallelogram.from_bounding_box(
            box=box, offset=offset)

        # adjust transformation for the fact that rot center was not at origin
        tf_0 = tf.resetCenter(center=center)

        # find inverse transform (center at origin)
        tf_0_inv = tf_0.inverse()

        # transform the final box back into a parallelogram
        orig_origin = tf_0_inv.transform(rot_crop.origin, xy_axes='point_dim')
        orig_basis = tf_0_inv.transform(rot_crop.basis, xy_axes='point_dim')
        paral = cls(origin=orig_origin, basis=orig_basis)

        # get bounding box for the parallelogram
        orig_box = paral.get_bounding_box(
            shape=shape, extend=extend, form=form)

        return orig_box
