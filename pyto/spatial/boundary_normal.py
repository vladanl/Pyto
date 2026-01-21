"""
Contains class BoundaryNormal.

# Author: Vladan Lucic
# $Id: boundary.py 2268 2026-01-20 17:29:20Z vladan $
"""

__version__ = "$Revision: 2268 $"

import numpy as np
import scipy as sp
import pandas as pd

import pyto
from pyto.geometry.vector import Vector


class BoundaryNormal:
    """BoundaryNormal class.

    Contains methods for manipulation of boundaries of regions
    and segments, or similar surfaces.

    Important methods:
      - find_normals(): Find normal vectors on the boundary
      - find_normal_global(): Find normal vector that approximates the
      entire boundary
    """

    def __init__(
            self, image, segment_id, external_id, get_boundary=True,
            bkg_id=0, no_distance_label=-1,
            dist_max_segment=2, dist_max_external=None, bound_thickness=1,
            alpha=2, beta=1, gamma=1, normalize=True):
        """Sets aattributes from arguments.

        Arguments:
          - image: (pyto.segmentation.Labels or ndarray) region image
          - segment_id: id of the segment from the region image on
          whose boundary normal vectors are to be calculated
          - external_id: id of the segment that contacts the segment
          (normals are to be calculated on the interface of the segment
          to the external)
          - get_boundary: flag indicating if the segment is restrited to
          the interface with the external (default True)
          - bkg_id: background id (default 0)
          - no_distance_label: labels points where normals are not
          calculated, should be <0 (default -1)
          - dist_max_segment: smoothing extent, that is the maximal
          distance between two boundary points at wich the normal at
          one point contributes to smoothing at the other point (default 2)
          - dist_max_external: maximal distance from a boundary point
          to external pixels that are used to calculate the normal at
          this point (default n_dim - 0.1, in which case all external
          points that share at least a vertex are included)
          - bound_thickness: (>=1) boundary thickness in pixels (default 1)
          - alpha, beta, gamma: parameters used to determine rae and
          smoothed normals, see find_normals() and distance_weighted_sum()
          docs (defaults 2, 1, 1, respectively)
          - normalize: flag indicating whether normal vectors are
          normalized to 1 (default True) 
        """

        # attributes from args
        if isinstance(image, pyto.core.Image):
            self.boundary = image.data
        else:
            self.boundary = image
        self.n_dim = self.boundary.ndim
        self.segment_id = segment_id
        self.external_id = external_id
        self.get_boundary = get_boundary
        self.bkg_id = bkg_id
        if no_distance_label >= 0:
            raise ValueError(
                f"Argument no_distance label has to be <0, but the specified"
                + f" value is {no_distance_label}")
        self.no_distance_label = no_distance_label
        self.dist_max_segment = dist_max_segment
        self.dist_max_external = dist_max_external
        if self.dist_max_external is None:
            self.dist_max_external = self.n_dim - 0.1
        self.bound_thickness = bound_thickness
        if self.bound_thickness is None:
            self.bound_thickness = 1
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.normalize = True

        # initialize other attributes
        #self.boundary = None

    def find_normal_global(self):
        """Finds normal at the center of the boundary.

        Boundary center is defined as the boundary point that is the
        closest to the boundary center of mess.

        Results are set as attributes:
          - all attributes set by find_normals() (listed in its doc),
          not that in this case the shapes/lengths are the same but
          n_points=1
          - point_center: (1d ndarray, size n_dim) boundary center coords
          - spherical coords with suffix '_global' ('spherical_phi_global',
          ... ) that have the same values, but are single numbers (instead
          of 1d ndarrays of size 1) 
        """

        # make boundary if needed
        if self.get_boundary:
            self.boundary = self.extract_boundary(image=self.boundary)

        # find point closest to cm
        self.point_center = self.find_closest_to_cm(segment_id=self.segment_id)

        # get smoothed normals
        points = self.point_center.reshape(1, -1)
        self.find_normals(points=points)

        # get spherical
        self.normals_to_spherical(
            normals=self.normals, global_=True, suffix='_global')
        
    def find_normals(self, points=None, raw_all_points=True):
        """Find normal vector to the boundary at all specified points.

        First, finds raw normals, that is normals at all boundary pixels
        that are calculated using neighbors (from self.external_id) of
        the boundary point. All points are used if raw_all_points is
        True, otherwise the specified poiints (arg points).

        The neighborhood size is specified by self.dist_max_external()
        and defined using self.generate_distance_kernels().
        The raw normals are calculated using self.find_normals_raw().
        Uses parameters slef.alpha, beta=1, gamma=0 to give relative
        weights to the neighborhood points (see distance_weighted_sum()).

        Second, the (vector field of) raw normals is smoothed by
        weighted local averaging over a neighborhood containing
        boundary points (for which raw normals were calculated).
        The neighborhood size is specified by self.dist_max_segment
        and defined using self.generate_distance_kernels(). Uses
        paramters alpha=1, self.beta and self.gamma
        to give relative weights to the neighborhood points (see
        distance_weighted_sum()).
        
        Sets the following attributes:
          - self.boundary: boundary image, calculated if self.get_boundary
          is True
          - points: (ndarray n_points x n_dim) coordinates of boundary pixels
          - dist_abs: (list indexed by self.points) each element (1d
          ndarray of size n_neighbors) contains absolute distance to
          all neighbors that are used to calculate the normal vector
          at the corresponding point (neighbors belong to self.external_id)
          - dist_vector: (lists indexed by self.points, each element
          is ndarray n_neighbors x n_dim) showing vector distances
          to neighbors (analoque to self.dist_abs)
          - points_good: (boolean ndarray) shows points used
          - normals: (ndarray n_points x n_dim) normal vector at each point
          - spherical_phi, spherical_phi_deg, spherical_theta,
          spherical_theta_deg: (ndarray, length n_points) spherical
          coordinates phi and theta in radians and degrees
          - additional attributes that have the same names but with
          suffix '_raw' (points_raw, dist_abs_raw, ...) which contain
          results originally obtained by find_normal_raw() and then
          renamed

        Also sets attributes having the same names as the above, but
        with suffix '_raw' to the raw normal field values (generated
        by self.find_normals_raw()).
        
        Arguments:
          - points: (ndarray n_points x n_dims) points where normals
          and angles are calulated, if None (default) all boundary
          points are used
          - raw_all_points: if true (default) calculate raw normals at
          all boundary points
        """

        # use all points for raw normals
        if points is not None:
            points = np.asarray(points)
        if raw_all_points:
            points_raw = None
        else:
            points_raw = points
        
        # find normals and distances based on individual segment points
        self.find_normals_raw(points=points_raw)
        self.rename_attributes(
            names=['dist_abs', 'dist_vector', 'points', 'points_good',
                   'normals', 'spherical_phi', 'spherical_phi_deg',
                   'spherical_theta', 'spherical_theta_deg'],
            suffix='_raw', remove=True)
        
        # make distance structure
        dist_kernel, _ = self.generate_distance_kernels(
            dist_max=self.dist_max_segment, n_dim=self.boundary.ndim)

        # convert normal vectors from coordinates to images (vector field)
        shape = self.boundary.shape
        normal_abs = self.no_distance_label + np.zeros(shape)
        for po in self.points_raw:
            normal_abs[*po] = 1
        normal_vector = []
        for ax_ind in range(self.boundary.ndim):
            norm_vec = np.zeros(shape)
            for po, norm in zip(self.points_raw, self.normals_raw):
                norm_vec[*po] = norm[ax_ind]
            normal_vector.append(norm_vec)

        # get contributions to normal vectors
        if points is None:
            points = self.points_raw
        vector_filter_res = self.setup_vector_filter(
            vector_abs=normal_abs, vectors=normal_vector,
            points=points, dist_kernel=dist_kernel)
        self.dist_abs, self.dist_vector, self.points, self.points_good = \
            vector_filter_res

        # smooth normal vectors
        if len(self.points) > 0:
            self.normals = self.distance_weighted_sum(
                vectors=self.dist_vector, dist=self.dist_abs, alpha=1,
                beta=self.beta, gamma=self.gamma, normalize=self.normalize)
            self.normals = np.asarray(self.normals)
        else:
            self.normals = np.zeros((0, self.n_dim))
           
        # convert normals to spherical angles
        self.normals_to_spherical()

    def find_normals_raw(self, points=None):
        """Find mormal vector at each point based on that point only.

        Sets attributes with the same names as find_normal() (see its
        doc). The only difference is that dist_abs and dist_vector
        contain distances from each boundary point to its
        self.external_id neighbors (as opposed to boundary neighbors).
        
        Arguments:
          - points: (ndarray n_points x n_dims) points where normals
          are calulated, if None (default) all boundary points are used
        """

        # make boundary if needed
        if self.get_boundary:
            self.boundary = self.extract_boundary(image=self.boundary)

        # get points if needed
        if points is None:
            self.points_all = np.array(
                np.nonzero(self.boundary==self.segment_id)).transpose()
        else:
            self.points_all = np.asarray(points)

        # make distance structure
        dist_kernel, coord_kernel = self.generate_distance_kernels(
            dist_max=self.dist_max_external, n_dim=self.boundary.ndim)

        # find all individual distances and vectors that contribute to normals
        selected_res = self.find_distance_vectors(
            image=self.boundary, points=self.points_all,
            dist_kernel=dist_kernel, coord_kernel=coord_kernel,
            segment_id=self.external_id)
        self.dist_abs, self.dist_vector, self.points, self.points_good \
            = selected_res

        # sum up individual and normalize
        if len(self.points) > 0:
            self.normals = self.distance_weighted_sum(
                vectors=self.dist_vector, dist=self.dist_abs, alpha=self.alpha,
                beta=1, gamma=0, normalize=self.normalize)
            self.normals = np.asarray(self.normals)
        else:
            self.normals = np.zeros((0, self.n_dim))
            
        # convert normals to spherical angles
        self.normals_to_spherical()
        
    def find_distance_vectors(
            self, points, dist_kernel, coord_kernel, segment_id, image=None):
        """Calculates distances to a segment.

        For each point (arg points) calculates vectors and distances
        to all points specified by arg segment_id that are within
        the distance kernel (arg dist_kernel). 

        Core calculations for find_normals_raw().

        Arguments:
          - image: (ndarray) image (default None, in which case
          self.image is used)
          - points: (ndarray n_points x n_dims) points where distances
          are calulated
          - distance_kernel: (square ndarray in n_dim dimensions, greyscale),
          distance kernel, as generated by generate_distance_kernels()
          - coord_kernel: (ndarray in n_dim+1 dimensions, axis 0 size
          n_dim, other axes square shape, integer) coordinate kernel,
          as generated by generate_distance_kernels()

        Returns:
          - dist_abs: (list of length n points) elements correspond to
          points, each element is a 1d ndarray of length n_neighbors
          containing absolute distances between the point and all
          neighboring segment_id points 
          - dist_vector: (list of length n_points) elements correspond to
          points, each element is 2d ndarray (axis 1 length 2) containing
          vectors from the point to the neighboring segment_id points
          - points: (ndarray n_points x n_dims) point coordinates
          - points_good: (boolean ndarray of length n_points: elements
          correspond to points, shows whether at least 1 neighboring
          segment_id point was found
        """

        if image is None:
            image = self.image
        
        # initialize loop
        radius = (dist_kernel.shape[0] - 1) // 2
        image_full = [slice(0, x) for x in image.shape]
        im_empty = pyto.grey.Image()

        dist_abs = []
        dist_vector = []
        points_res = []
        points_good = []

        for po in points:

            # make initial image inset centered on point, size of kernel
            low = po - radius
            high = po + radius + 1
            image_inset = [slice(begin, end) for begin, end in zip(low, high)]

            # find common image and kernel insets  
            image_inset_adj = im_empty.findIntersectingInset(
                inset=image_full, inset2=image_inset)
            image_adj_orig = [x.start for x in image_inset]
            dist_kernel_inset_adj = [
                slice(im_slice.start - orig, im_slice.stop - orig) 
                for im_slice, orig in zip(image_inset_adj, image_adj_orig)]
            
            # extract overlaping parts of image and kernels
            bound_one = image[*image_inset_adj]
            dist_kernel_adj = dist_kernel[*dist_kernel_inset_adj]
            coord_kernel_adj = np.asarray(
                [coord_ke[*dist_kernel_inset_adj] for coord_ke in coord_kernel])
            
            # select distances and coord vectors by distance and segment
            dist_mask = (
                (bound_one == segment_id) & (dist_kernel_adj >= 0))
            dist_abs_one = dist_kernel_adj[dist_mask]
            if len(dist_abs_one) == 0:
                points_good.append(False)
                continue
            dist_vector_one = np.asarray(
                [coord_ke[dist_mask]
                 for coord_ke in coord_kernel_adj]).transpose()

            # save
            dist_abs.append(dist_abs_one)
            dist_vector.append(dist_vector_one)
            points_res.append(po)
            points_good.append(True)

        return (dist_abs, dist_vector,
                np.asarray(points_res), points_good)

    def setup_vector_filter(
            self, vector_abs, vectors, points, dist_kernel):
        """Selects neighborhood vectors for each point. 

        For each pont (arg points) finds vectors that are located
        in the neighborhood of the point defined by (arg) dist_kernel.
        The selected vectors and their corresponding values are then
        converted from the image representation (args vectors and
        vector_abs) to the point representation.

        Forms core calculations for find_normals().
        
        Arguments:
          - vector_abs: (ndarray of shape self.boundary.shape) image
          that contains abs value of vectors at each point
          where they are calculated (other points should have value
          self.no_distance_label)
          - vectors: (list of length n_dim, where each element
          is ndarray  of shape self.boundary.shape) cartesian coords
          of vectors at each point where they are calculated
          (other points should have value 0)
          - points: (ndarray n_points x n_dim) coordinates of points
          - dist_kernel: (ndarray) distance kernel that defines the
          neighborhood (here only the shape and ndim of the kernel is
          used)

        Returns:
          - dist_abs: (list of length n points) elements correspond to
          points, each element is a 1d ndarray of length n_neighbors
          containing absolute values of vectors that are located in the
          neighborhood of the point
          - dist_vector: (list of length n_points) elements correspond to
          points, each element is 2d ndarray (axis 1 length 2) containing
          vectors that are located in the neighborhood of the point
          - points_res: (ndarray n_points x n_dim) coordinates of points
          for which neighborhood vectors were found
          - points_good: (boolean ndarray) directly corresponds to arg
          points, shows for which points neighborhood vectors were found 
        """
        
        # initialize loop
        radius = (dist_kernel.shape[0] - 1) // 2
        image_full = [slice(0, x) for x in vector_abs.shape]
        im_empty = pyto.grey.Image()

        dist_abs = []
        dist_vector = []
        points_res = []
        points_good = []

        for po in points:

            # make initial image inset centered on point, size of kernel
            low = po - radius
            high = po + radius + 1
            image_inset = [slice(begin, end) for begin, end in zip(low, high)]

            # find common distance and kernel insets  
            image_inset_adj = im_empty.findIntersectingInset(
                inset=image_full, inset2=image_inset)
            image_adj_orig = [x.start for x in image_inset]
            kernel_inset_adj = [
                slice(im_slice.start - orig, im_slice.stop - orig) 
                for im_slice, orig in zip(image_inset_adj, image_adj_orig)]

            # extract overlaping parts of image and kernels
            distance_absolute_adj = vector_abs[*image_inset_adj]
            dist_vector_adj =  np.asarray(
                [dist_vec[*image_inset_adj] for dist_vec in vectors])
            kernel_adj = dist_kernel[*kernel_inset_adj]

            # select abs and vector distances by existing distances
            mask = (distance_absolute_adj >= 0) & (kernel_adj >= 0)
            dist_abs_one = distance_absolute_adj[mask]
            if len(dist_abs_one) == 0:
                points_good.append(False)
                continue
            dist_vector_one = np.asarray(
                [dist_comp[mask] for dist_comp in dist_vector_adj]).transpose()
            
            # save
            dist_abs.append(dist_abs_one)
            dist_vector.append(dist_vector_one)
            points_res.append(po)
            points_good.append(True)

        return (dist_abs, dist_vector,
                np.asarray(points_res), np.asarray(points_good))
            
    def extract_boundary(self, image):
        """Extracts boundary of the segment that faces external region.

        Arguments:
          - image: (numpy.ndarray)
        """

        structure, _ = self.generate_distance_kernels(
            dist_max=self.bound_thickness, n_dim=image.ndim)
        structure = structure >= 0
        dilated = sp.ndimage.binary_dilation(
            image==self.external_id, structure=structure, 
            border_value=self.bkg_id)
        bound = np.where(
            np.logical_not(dilated) & (image==self.segment_id),
            self.bkg_id, image)

        return bound

    def find_closest_to_cm(self, segment_id):
        """Finds point on a segment that is the closest to its center of mass.

        """
        
        cm = sp.ndimage.center_of_mass(self.boundary, labels=segment_id)
        bound_points_arrays = np.nonzero(self.boundary==segment_id)
        bound_points = np.vstack(bound_points_arrays).transpose()
        distance_vec = bound_points - np.asarray(cm).reshape(1, -1)
        min_position = np.argmin((distance_vec**2).sum(axis=1))
        point_global = bound_points[min_position, :]

        return point_global
    
    def generate_distance_kernels(self, dist_max, n_dim):
        """Makes greyscale distance and coordinate kernels.

        Generates two kernels:
          - distance kernel: Euclidean distances to the center, up to
          the max distance (arg dist_max), elements located at
          larger distance are set to self.no_distance_label
          - coordinate kernel: Distances to the center for each
          coordinate separately (for all kernel elements). 

        The kernel size is the smallest possible that contains all
        distances <= dist_max.
        
        Uses self.no_distance_label.

        Arguments:
          - dist_max: maximal distance for distance kernel
          - n_dim: number of dimensions

        Returns:
          - distance_kernel: (square ndarray in n_dim dimensions, greyscale)
          - coord_kernel: (ndarray in n_dim+1 dimensions, axis 0 size
          n_dim, other axes square shape, integer)
        """

        # make distance structure where all elements show distance
        max_int = np.floor(dist_max).astype(int)
        size = 2 * max_int + 1
        base = np.ones(shape=n_dim*[size], dtype=int)
        center = n_dim * [max_int]
        base[*center] = 0

        # generate kernels
        dist_kernel = sp.ndimage.distance_transform_edt(base)
        coord_kernel = np.indices(base.shape) - max_int
         
        # label elements further than max external
        dist_kernel_mask = dist_kernel > dist_max
        dist_kernel[dist_kernel_mask] = self.no_distance_label

        return dist_kernel, coord_kernel

    def distance_weighted_sum(
            self, vectors, dist, alpha=0, beta=1, gamma=0, normalize=True):
        """Weighted sum of normal vector components.

        Takes multiple vectors (arg vectors) and averages them according
        to their associated weights (specified as distances, which are
        inverse weights, arg dist). The vectors and distances are
        organized in lists of the same length where elements are
        independent from each other (they correspond to "points" of
        an image). The averaging is done for vectors of
        a single point, for each point separately, resulting in one
        average vector per point.
        
        If v are vectors (arg vectors) and d are distances (arg dist) at
        a single point, the weighted sum for that point is calculated as
        (and repeated for each point): 

           normal = sum over neighborhood (v / (beta * d^alpha + gamma))

        Coefficients alpha, beta and gamma can be chosen to represent
        different weighting. For example:
          - alpha=0, beta=1, gamma=0: simple sum, no weighting
          - alpha=2, beta=1, gamma=0: weighting by the inverse square
          dependence on distance
          - gamma > 0: regularization factor, used to give the highest
          weight to vector having associated distances of 0

        Note: This method does not use any attribute nor method of
        this instance (todo: make class or static method?) 
        
        Arguments:
          - vectors: (list of length n_points): coordinates of vectors
          from the point to the neighborhood points, where each element
          is ndarray of shape n_dim x n_neighborhood_points
          - bist: (list of length n_points): distance to neighborhood
          points, where each elemet is ndarray of shape
          n_neighborhood_points
          - alpha, beta, gamma: parameters
          - normalization: if True, the calculated normals are normalized
          to 1

        Returns normals: (ndarray of shape n_points x n_dims) resulting
        normal vectors for each point
        """
        
        normal = [
            np.sum(
                u_co / (
                    beta * u_di.reshape(u_di.shape[0], -1)**alpha + gamma),
                axis=0)
            for u_co, u_di in zip(vectors, dist)]
        if normalize:
            normal_abs = np.linalg.norm(normal, axis=1)
            normal = normal / normal_abs.reshape(normal_abs.shape[0], -1)

        return normal        

    def rename_attributes(self, names, suffix, remove=False):
        """Renames attributes.

        """

        for nam in names:
            try:
                setattr(self, nam + suffix, getattr(self, nam))
                if remove:
                    delattr(self, nam)
            except AttributeError:
                pass

    def normals_to_spherical(self, normals=None, global_=False, suffix=''):
        """Converts normals from cartesian coords to spherical angles.

        If arg global is True, only element 0 of arg normals is
        converted to spherical coords and the spherical coordinates
        are saved as single numbers (instead of ndarrays), while
        other elements are ignored. This is meant for the "global"
        case where only one normal is calculated for the entire boundary.
        
        Sets attributes:
          - spherical_phi{suffix}, spherical_phi_deg{suffix}
          - spherical_theta{suffix}, spherical_theta_deg{suffix}
        If arg global_ is False, all attributes are 1d ndarrays of
        length n_ponts. Otherwise they are single numbers.

        Attributes:
          - normals: (ndarray n_points x n_dim) normal vectors (such
          as self.normal generated by find_normals())
          - global_: flag indicating whether only the element 0 of normals
          is converted to spherical (default False)
          - suffix: suffix added to spherical coordinate variable names
          (default '', so no suffix)
        """

        if normals is None:
            normals = self.normals
            
        normals_vector = Vector(normals)
        if global_:
            phi = normals_vector.phi[0]
            phi_deg = normals_vector.phiDeg[0]
        else:
            phi = normals_vector.phi
            phi_deg = normals_vector.phiDeg
        self.__setattr__(f"spherical_phi{suffix}", phi)
        self.__setattr__(f"spherical_phi_deg{suffix}", phi_deg)
        
        if self.n_dim == 3:
            if global_:
                theta = normals_vector.theta[0]
                theta_deg = normals_vector.thetaDeg[0]
            else:
                theta = normals_vector.theta
                theta_deg = normals_vector.thetaDeg
            self.__setattr__(f"spherical_theta{suffix}", theta)
            self.__setattr__(f"spherical_theta_deg{suffix}", theta_deg)

