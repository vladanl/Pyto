"""
N-dim convex hull manipulations

# Author: Vladan Lucic 
# $Id$
"""

__version__ = "$Revision$"


import numpy as np
import scipy as sp
from scipy.spatial import ConvexHull

from ..core.image import Image
from ..segmentation.labels import Labels


class ConvexHullUtil(ConvexHull):

    def __init__(self, points, incremental=False, qhull_options=None):
        """
        """
        super().__init__(
            points=points, incremental=incremental,
            qhull_options=qhull_options)

    @classmethod
    def inside_hull(cls, hull_points, target_points, expand=0, eps=1e-9):
        """Make hull from hull_points and determine whether target_points 
        are inside.

        Uses scipy.spatial.ConvexHull to make the hull.

        Equations that define hull simplices are used to determine whether
        target points are inside.

        The hull determined by hull_points can be effectively expanded (or 
        schrunk if expand<0) by specifying arg expand. Note that each convex 
        hull simplex is moved by the specified value, so that the actual 
        expansion depends on the simplices and can be expected to fall 
        between euclidean and checkerboard expansions. 

        Also note that the expansion does not affects the hull, but only the
        determination of the target points location (inside or outside of
        the "extended" hull).

        Arguments:
          - hull_points: (n hull points x n dim) coordinates of points used 
          to make the convex hull
          - target_points: (n target points x n dim) coordinates of points 
          for which it is determined whether they are inside the convex hull
          - expand: the complex hull is extended by this distance (in the
          units that are the same as coordinates)
          - eps: small numerical factor used to make sure that all hull
          points are considered inside

        Returns (inside, hull):
          - inside: (bool, size n target points) shows whether target points
          are inside the hull, in the same order as the target points
          - hull: (instance of this class) convex hull based on hull points
          where arg expand is not taken into account
        """

        # make hull
        hull = cls(points=hull_points)

        # distance of all target points to all hull simplices
        # <0 means inside the hull
        dist_to_hull = (
            np.tensordot(
                target_points,
                hull.equations[:, :-1], axes=([-1], [-1]))
            + hull.equations[:, -1])
        dist_to_hull -= expand

        # find target points that are inside all hull simplices
        inside_individual = (dist_to_hull < 0) | (np.abs(dist_to_hull) < eps)
        inside = np.logical_and.reduce(inside_individual, axis=1)

        return inside, hull

    @classmethod
    def intersect_segment(
            cls, hull_points, segment, segment_id=None, out_path=None,
            expand=0, expand_nm=None, eps=1e-9):
        """Find intersection of a convex hull and a segment.

        Points that determine convex hull have to be specified in pixels.

        If the segment is read from a mrc file, the pixel is determined 
        from the header. Otherwise, pixel size is taken to be 1 nm.

        The hull determined by hull_points can be effectively expanded (or 
        schrunk if expand<0) by specifying arg expand_nm. Note that each 
        convex hull simplex is moved by the specified value, so that 
        the actual expansion depends on the simplices and can be expected 
        to fall between euclidean and checkerboard expansions. 

        Also note that the expansion does not affects the hull, but only the
        determination of the target points location (inside or outside of
        the "extended" hull).

        Arguments:
          - hull_points: (n hull points x n dim) coordinates of points used to 
          make the convex hull (in pixel coordinates)
          - segment: segment that is intersected with the hull, can be:
            - ndarray representing segmented image
            - pyto image object (pyto.core.Image) containing a segmented image 
            - file path to the image containing a segmented image 
          - segment_id: id of the segment that is to be intersected with 
          the hull, if None all pixels >0 are considered the segment
          - out path: file path for the resulting intersection image (used
            only if segment is a file path also
          - expand: outwards extension of the complex hull (in pixel 
          coordinates). Not considered if arg expand_nm is specified and
          segment is read from the specified file path.
          - expand_nm: outwards extension of the complex hull in nm. 
          Considered only if segment is read from the specified file path.
          - eps: small numerical factor used to make sure that all hull
          points are considered inside

        Returns: restricted segment, hull:
          - restricted segment type matches the arg segment type:
            - ndarray
            - pyto image object (pyto.segmentation.Labels)
            - None if segment is the segmented image file path
          - hull: (instance of this class) convex hull based on hull points
          where arg expand is not taken into account
        """

        # read and interpret segment
        if isinstance(segment, np.ndarray):
            data = segment
            segment_type = 'ndarray'
        elif isinstance(segment, Image):
            data = segment.data
            segment_type = 'pyto image'
        elif isinstance(segment, str):
            seg = Labels.read(file=segment, header=True, memmap=True)
            data = seg.data
            if expand_nm is not None:
                expand = expand_nm / seg.pixelsize
            segment_type = 'image file'

        # get segment coordinates (n points x n dim)
        if segment_id is None:
            segment_coords = np.stack(data.nonzero(), axis=1)
        else:
            select_seg = (data == segment_id)
            segment_coords = np.stack(select_seg.nonzero(), axis=1)

        # find segment points that are inside hull
        inside, hull = cls.inside_hull(
            hull_points=hull_points, target_points=segment_coords,
            expand=expand, eps=eps)
        intersect_points = segment_coords[inside]

        # make array containing segment points that are inside hull
        good_indices = tuple(
            intersect_points[:, dim]
            for dim in range(intersect_points.shape[1]))   
        intersect_data = np.zeros_like(data)
        intersect_data[good_indices] = 1

        # make output the same form as input
        if segment_type == 'ndarray':
            return intersect_data, hull 
        elif segment_type == 'pyto image':
            return Labels(intersect_data), hull
        elif segment_type == 'image file':
            seg.data = intersect_data
            if out_path is None:
                return seg, hull
            else:
                seg.write(file=out_path, header=True, pixel=seg.pixelsize)
                return None, hull
