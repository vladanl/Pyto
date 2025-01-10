"""
Class Plane provides methods for creating n-1 dimensional planes
in n dimensional space.

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""

__version__ = "$Revision$"

import numpy as np


class Plane(object):
    """
    """

    def __init__(self, plane_label=0, positive_label=1, negative_label=-1):
        """
        Sets labeling values.

        Arguments:
          - plane_label, positive_label, negative_label: labels used to
          designate the plane, space outside the plane in the positive
          direction and in the negative direction, respectively
        """

        self.plane_label = plane_label
        self.positive_label = positive_label
        self.negative_label = negative_label


    #################################################################
    #
    # Methods
    #

    @classmethod
    def make(
            cls, shape, normal=None, point=None, npoints=None, distance=False,
            thick=1, eps=1e-9, plane_label=0, positive_label=1,
            negative_label=-1):
        """
        Makes a n-1 dimensional plane in n-dimensional space.

        Returns a ndarray where the plane is labeled by 0, the half-space
        on the positive side of the plane (in the direction of the normal
        vector) by 1 and the negative half-space by -1. If the plane is
        specified by arg npoints, the direction of the normal vector is
        is arbitrary.

        If arg distance is True, an ndarray containing distances to the
        plane is returned (negative distances on the negative half-space).

        A plane can be generated in two ways, in this order:
          - from a normal vector (arg normal) and a point on the plane
          (arg point); both can be floats
          - from n points on the plane (arg npoints), can be floats.
          Obviously, these points should not lay on a lower than n-1
          dimensional plane.

        Consequently, if args normal and point are not specified, arg npoints
        are used (in which case npoints has to be specified).

        Arguments:
          - shape: shape of the output ndarray
          - normal, point: normal vector and one point that define a plane
          - points: n points that define n-1 plane, specified as 2D array
          where axis 0 denotes points and axis 1 dimensions
          - distance: flag indicating whether distances are returned
          - thick: plane thickness (+/- thick/2 from the exact plane)
          - eps: Used to correct for numerical errors in labeling the
          plane when thick=0 (no effect when thick != 0)


        Returns: ndarray containing the specified plane
        """

        # check arguments
        if (normal is not None) and (point is not None):
            input = 'normal_point'
        elif npoints is not None:
            input = 'n_points'
        else:
            raise ValueError(
                "Arguments normal and point, or npoints need to be specified")

        # instantiate
        inst = cls(
            plane_label=plane_label, positive_label=positive_label,
            negative_label=negative_label)
        npoints = np.asarray(npoints)

        # get normals from points, if needed
        if input == 'n_points':
            normal = inst.get_normal(npoints=npoints)
            point = npoints[0, :]

        # normalize vector
        normal = normal / np.sqrt(np.vdot(normal, normal))

        # save plane params
        inst.normal = normal
        inst.point = point

        # make plane array indices
        indices = np.indices(shape)

        # distance to the plane defined by normal and point
        dist = (
            np.tensordot(indices, normal, axes=(0, 0))
            - np.vdot(point, normal))
        if distance:
            inst.distance = dist

        # label plane and half-spaces
        plane = np.where(
            dist > thick / 2, inst.positive_label, inst.plane_label)
        plane = np.where(dist > -thick / 2, plane, inst.negative_label)
        if thick == 0:
            plane = np.where(np.abs(dist) <= eps, inst.plane_label, plane)
        inst.data = plane

        return inst

    def get_normal(self, npoints):
        """
        Calculates normal vector for a n-1 dimensional plane from points
        on that plane (in n-dimensional space).

        The number of points has to equal the number of dimensions.

        Argument:
          - npoints: point on the plane, specified as 2D array where
          axis 0 denotes points and axis 1 dimensions

        Returns: normal vector (normalized)
        """

        # check argument
        npoints = np.asarray(npoints).copy()
        shape = npoints.shape
        ndim = shape[1]
        if shape[0] != shape[1]:
            raise ValueError(
                ("Number of points: {}  has to be the same as the number "
                 + "of dimensions: {}.").format(shape[0], shape[1]))

        # shift coordinates of all points if npoints is singular
        for ind in range(2*ndim):
            if np.linalg.det(npoints) == 0:
                npoints += np.ones_like(npoints)
            else:
                break
        else:
            raise ValueError(
                "Specified argument npoints do not define a plane (det = 0)")

        # calculate
        normal = np.matmul(np.linalg.inv(npoints), np.ones(ndim))
        normal = normal / np.sqrt(np.vdot(normal, normal))

        return normal

    def get_side(self, external):
        """
        The sign of the returned value specifies whether the arg external
        is a point on the positive side of the current plane (+1),
        negative (-1) or on the plane (0).

        Argument:
          - external: a point
        """

        side = np.inner((external - self.point), self.normal)
        return np.sign(side)
