"""
Contain classes BoundaryNormal and BoundarySmooth.

# Author: Vladan Lucic
# $Id$
"""

__version__ = "$Revision$"

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
          - self.boundary: boundary image, calculated if self.get_boundary
          is True
          - self.point_center: 
          - self.normals: (ndarray 1 x n_dim) normal vector

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
        self.normals_to_spherical(normals=self.normals)
        
    def find_normals(self, points=None, raw_all_points=True):
        """Find normal vector at each point based on neighborhood.

        First finds raw normals, that is normals at all boundary pixels
        if raw_all_points is True, or if not at the specified poiints
        (arg points), Then the the (vector field of) normals
        is smoothed by weighted local averaging.

        The raw normals are calculated using self.find_normals_raw(). They
        are calcuated using parameters slef.alpha, beta=1, gamma=0 (see
        distance_weighted_sum()).

        The raw normals field is smoothed using paramters alpha=1,
        self.beta and self.gamma (see distance_weighted_sum()).
        
        Sets the following attributes:
          - self.boundary: boundary image, calculated if self.get_boundary
          is True
          - points: (ndarray n_points x n_dim) coordinates of boundary pixels
          - dist_abs: (list indexed by self.points) each element contains
          absolute distance to all boundary points that are used to
          calculate the normal vector at the corresponding point 
          - dist_vector: (lists of three lists, each indexed by self.points)
          vector distances like self.dist_abs
          - points_good: (boolean ndarray) shows points used
          - normals: (ndarray x_points x n_dim) normal vector at each point
          - spherical_phi, spherical_phi_deg, spherical_theta,
          spherical_theta_deg: (ndarray, length n_points) spherical
          coordinates phi and theta in radians and degrees
        
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
        vector_filter_res = self.vector_filter(
            distance_abs=normal_abs, distance_vector=normal_vector,
            #points=self.points_raw, dist_kernel=dist_kernel)
            points=points, dist_kernel=dist_kernel)
        self.dist_abs, self.dist_vector, self.points, self.points_good = \
            vector_filter_res

        # smooth normal vectors
        if len(self.points) > 0:
            self.normals = self.distance_weighted_sum(
                coord=self.dist_vector, dist=self.dist_abs, alpha=1,
                beta=self.beta, gamma=self.gamma, normalize=self.normalize)
            self.normals = np.asarray(self.normals)
        else:
            self.normals = np.zeros((0, self.n_dim))
           
        # convert normals to spherical angles
        self.normals_to_spherical()

    def find_normals_raw(self, points=None):
        """Find mormal vector at each point based on that point only.

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
        selected_res = self.select_distance_vectors(
            image=self.boundary, points=self.points_all,
            dist_kernel=dist_kernel, coord_kernel=coord_kernel,
            segment_id=self.external_id)
        self.dist_abs, self.dist_vector, self.points, self.points_good \
            = selected_res

        # sum up individual and normalize
        if len(self.points) > 0:
            self.normals = self.distance_weighted_sum(
                coord=self.dist_vector, dist=self.dist_abs, alpha=self.alpha,
                beta=1, gamma=0, normalize=self.normalize)
            self.normals = np.asarray(self.normals)
        else:
            self.normals = np.zeros((0, self.n_dim))
            
        # convert normals to spherical angles
        self.normals_to_spherical()
        
    def select_distance_vectors(
            self, image, points, dist_kernel, coord_kernel, segment_id):
        """Calculates distances to a segment.

        Core calculations for find_normals_raw().

        Returns:
          - dist_abs: (list of length n points) elements correspond to
          points, each element is a list containing absolute distances
          between the point and all neighboring segment_id points 
          - dist_vector: (list of length n_points) elements correspond to
          points, each element is 2d nparray (axis 1 length 2) containing
          vectors from the point to the neighboring segment_id points
          - points: (ndarray n_points x n_dims) point coordinates
          - points_good: (boolean ndarray of length n_points: elements
          correspond to points, shows whether at least 1 neighboring
          segment_id point was found
        """
        
        # initialize loop
        radius = (dist_kernel.shape[0] - 1) // 2
        dist_kernel_full = dist_kernel.ndim * [slice(0, dist_kernel.shape[0])] 
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

    def vector_filter(
            self, distance_abs, distance_vector, points, dist_kernel):
        """Filters vector field.

        """
        
        # initialize loop
        radius = (dist_kernel.shape[0] - 1) // 2
        dist_kernel_full = dist_kernel.ndim * [slice(0, dist_kernel.shape[0])] 
        image_full = [slice(0, x) for x in distance_abs.shape]
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
            distance_absolute_adj = distance_abs[*image_inset_adj]
            dist_vector_adj =  np.asarray(
                [dist_vec[*image_inset_adj] for dist_vec in distance_vector])
            kernel_adj = dist_kernel[*kernel_inset_adj]

            # select abs and vector distances by existing distances
            mask = (distance_absolute_adj >= 0) & (kernel_adj >= 0)
            dist_abs_one = distance_absolute_adj[mask]
            if len(dist_abs_one) == 0:
                points_good.append(False)
                continue
            dist_vector_one = np.asarray(
                [dist_comp[mask]for dist_comp in dist_vector_adj]).transpose()

            # save
            dist_abs.append(dist_abs_one)
            dist_vector.append(dist_vector_one)
            points_res.append(po)
            points_good.append(True)

        return (dist_abs, dist_vector,
                np.asarray(points_res), points_good)
            
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

        """

        # make distance structure where all elements show distance
        max_int = np.floor(dist_max).astype(int)
        size = 2 * max_int + 1
        base = np.ones(shape=n_dim*[size], dtype=int)
        center = n_dim * [max_int]
        base[*center] = self.bkg_id

        # generate kernels
        dist_kernel = sp.ndimage.distance_transform_edt(base)
        coord_kernel = np.indices(base.shape) - max_int
         
        # label elements further than max external
        dist_kernel_mask = dist_kernel > dist_max
        dist_kernel[dist_kernel_mask] = self.no_distance_label

        return dist_kernel, coord_kernel

    def distance_weighted_sum(
            self, coord, dist, alpha=0, beta=1, gamma=0, normalize=True):
        """Weighted sum of normal vector components.

        If v is vecor at each point the sum is calculated as: 

           sum over neighborgood (v / (beta * |v|^alpha + gamma))
        """
        
        normal = [
            np.sum(
                u_co / (
                    beta * u_di.reshape(u_di.shape[0], -1)**alpha + gamma),
                axis=0)
            for u_co, u_di in zip(coord, dist)]
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

    def normals_to_spherical(self, normals=None, suffix=''):
        """Converts normal (cartesian) to spherical angles.

        """

        if normals is None:
            normals = self.normals
            
        normals_vector = Vector(normals)
        self.__setattr__(f"spherical_phi{suffix}", normals_vector.phi)
        self.__setattr__(f"spherical_phi_deg{suffix}", normals_vector.phiDeg)
        
        if self.n_dim == 3:
            self.__setattr__(f"spherical_theta{suffix}", normals_vector.theta)
            self.__setattr__(
                f"spherical_theta_deg{suffix}", normals_vector.thetaDeg)


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
