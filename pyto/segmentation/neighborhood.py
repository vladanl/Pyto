"""
Contains class Neighborhood that is intended to be use to generate and 
characterize parts of segments that are localized in a neighborhood 
of other segments (here called regions).

# Author: Vladan Lucic
# $Id$
"""

import logging

import numpy as np
import scipy as sp
import scipy.ndimage as ndimage

from .segment import Segment
from .features import Features
from .morphology import Morphology


class Neighborhood(Features):
    """Generates and characterizes neighborhoods of specified segments.

    Important attributes:
      - contact_cm: center of mass coordinates of contacts

    Important methods:
      - make_contacts()
      - find_contacts_cm()

    Note: unlike most of other classes that inherit from Features, but like
    DistanceTo class, the data of this class is internally storred in a 
    compact form. 

    """

    def __init__(self, segments=None, ids=None):
        """
        Initializes attributes.

        Arguments:
          - segments: (Labels) segments
          - ids: segment ids (if None segments.ids used)
        """

        # set segments and ids
        #super(DistanceTo, self).__init__(segments=segments, ids=ids)
        self.segments = segments
        self._ids = ids
        if self._ids is None:
            if self.segments is not None:
                self._ids = segments.ids
        else:
            self._ids = np.asarray(ids)

        # local data attributes
        self.dataNames = ['contact_cm']
        self.compact = True

    def make_contacts(
            self, regions, region_id, segments=None, ids=None, extend=0,
            dilate_structure=None):
        """Makes contacts between segments and regions

        Contacts are defined as parts of segments that are neighbors of 
        the specified region (arg regions and reg_id) according to the
        specified structure (arg, dilate_structure).

        Respects inset of both segments and regions.

        Arguments:
          - regions: (Labels) regions image
          - region_id: id of the region whose neighborhood should be determined
          - segments: (Labels) segments (if None, self.segments is used)
          - ids: segment ids (if None, self.ids is used)
          - extend: if different from 0, the returned inset is extended by
          this amount (in pixels)
          - dilate_structure: (ndarray) structuring element used for dilation.
          If None, a new one is generated using square connectivity 1 (like
          scipy default), so nieghbors in 3D have a common face 

        Returns image showing the contacts, which are labeled the same as 
        segments. The inset of the returned image is the smallest one_
        that contains the specified region and segments, extended by arg
        extend.          
        """

        # arguments
        if segments is None:
            segments = self.segments
        if ids is None:
            ids = self.ids
        #if region_ids is None:
        #    region_ids = regions.ids
        #region_ids = np.asarray(region_ids)

        # make corresponding insets on segments and regions
        # Note: not copying arrays, se need to make sure these are not modified
        inset = segments.findInset(
            ids=ids, mode='abs', additional=regions, additionalIds=region_id,
            extend=extend, expand=True)
        seg_data = segments.useInset(
            inset=inset, mode='abs', expand=True, value=0,
            update=False, returnCopy=False)
        reg_data = regions.useInset(
            inset=inset, mode='abs', expand=True, value=0,
            update=False, returnCopy=False)

        # dilate and make contacts
        reg_dilated = ndimage.binary_dilation(
            reg_data==region_id, structure=dilate_structure)
        contacts_data = np.where(reg_dilated, seg_data, 0)
        contacts = Segment(data=contacts_data)
        contacts.inset = inset

        return contacts

    def find_contact_cm(
            self, regions, region_id, segments=None, ids=None, extend=0,
            dilate_structure=None, frame='abs'):
        """Finds center of mass coordinates for contact regions of segments.

        Contacts are found using make_contacts(), see the doc string for the 
        contact definition.

        The coordinates are given in reference to the specified frame, as follows:
          - 'abs': absolute, so original image coordinates
          - 'rel': relative, so in the coordinates of the image inset returned
          by make_contacts()
          - 'segments': relative to segments.inset
          - 'regions': relative to regions.inset

        Sets attributes:
          - contact_cm: (ndarray, n_ids x n_dims) center of mass coordinates
          - segments (if arg segments specified)
          - ids (if arg ids specified)

        Arguments:
          - regions: (Labels) regions image
          - region_id: id of the region whose neighborhood should be determined
          - segments: (Labels) segments (if None, self.segments is used)
          - ids: segment ids (if None, self.ids is used)
          - extend: if different from 0, the returned inset is extended by
          this amount (in pixels)
          - dilate_structure: (ndarray) structuring element used for dilation.
          If None, a new one is generated using square connectivity 1 (like
          scipy default), so nieghbors in 3D have a common face 

        Returns (contact_cm, contact_image)
          - contact_cm: (ndarray, n_ids x n_dims) center of mass coordinates,
          same as self.contact_cm
          - contact_image: (Labels; the same as returned from make_contacts()) 
          image showing the contacts, which are labeled the same as 
          segments. The inset of the returned image is the smallest one_
          that contains the specified region and segments, extended by arg
          extend.     
        """

        # find contacts
        contacts = self.make_contacts(
            regions=regions, region_id=region_id, segments=segments, ids=ids,
            extend=extend, dilate_structure=dilate_structure)

        # get contact cms (in the compact form)
        mor = Morphology(segments=contacts, ids=ids)
        cm = mor.getCenter(real=False, inset=False)[contacts.ids]

        # convert coords to the specified frame
        if frame == 'rel':
            pass
        elif frame == 'abs':
            cm = cm + np.array([sl.start for sl in contacts.inset])
        elif frame == 'segments':
            if segments is None:
                segments = self.segments
            cm = (cm + np.array([sl.start for sl in contacts.inset])
                  - np.array([sl.start for sl in segments.inset]))
        elif frame == 'regions':
            cm = (cm + np.array([sl.start for sl in contacts.inset])
                  - np.array([sl.start for sl in regions.inset]))
        self.contact_cm = cm
            
        return cm, contacts
    
    def generate_neighborhoods(
        self, regions, ids=None, region_ids=None, size=None, 
        max_distance=None, distance_mode='min', remove_overlap=False):
        """
        Generates neighborhoods of each specified region on each of the 
        segments.

        Regions are specified by args region and region_ids, and segments by ids.
        A neighborhood of a given region on a segment is defined as a 
        subset of the segment that contains elements that are at most (arg) 
        size/2 away from the closest segment element to the region, as long as 
        the distance to the region is not above (arg) max_distance.

        If multiple elements of segment are at the minimal distance to the
        region, only one of them is selected as the closest element
        (according to the output of scipy.ndimage.extrema()).

        The distance between a region and segments is calculated according to 
        the arg distance_mode. First the (min) distance between segments and 
        each point of regions is calculated. Then the min/max/mean/median of 
        the calculated distances, or the (min) distance between the region 
        center and the segments is used.

        If remove_overlap is True, parts of segments that overlap with regions 
        are removed from the calculated neighborhoods.

        Respect inset in both self and regions. This means that the original
        positioning (before using insets) has to be the same in both self.data
        and regions.data arrays.

        Arguments:
          - ids: segment ids
          - regions: (Segment) regions
          - region_ids: region ids
          - size: size of a neighborhood in the direction perpendicular to
          the direction towards the corresponding region (diameter-like)
          - max_distance: max distance between segments and a given region
          - distance_mode: how a distance between layers and a region is
          calculated (min/max/mean/median)
          - remove_overlap: if True a neighbor can not contain a part of a region

        Yields:
          - region id: id of current neighborhood
          - neighborhoods: (Segment) neighborhood corresponing to region id
          - all neighborhoods: (Segment) all neighborhoods together
        """

        # parse arguments
        if ids is None:
            ids = self.ids
        ids = np.asarray(ids)
        if region_ids is None:
            region_ids = regions.ids
        region_ids = np.asarray(region_ids)

        # save initial insets and data of segments
        self_inset = self.segments.inset
        self_data = self.segments.data
        reg_inset = regions.inset
        reg_data = regions.data.copy()

        # make a working copy of an inset of this instance and clean it
        self.segments.makeInset(ids=ids, additional=regions, additionalIds=region_ids)
        seg = Segment(data=self.segments.data, ids=ids, copy=True, clean=True)
        seg.inset = self.segments.inset

        # Revert self.segments to initial state (won't be used further down)
        self.segments.inset = self_inset
        self.segments.data = self_data

        # save current seg
        # not needed anymore because seg.data is not changed
        #seg_inset = seg.inset
        #seg_data = seg.data

        # find regions that are not further than max_distance to segments
        if max_distance is not None:

            # find min distances from each region to segments 
            regions.useInset(inset=seg.inset, mode='abs')
            dist = regions.distanceToRegion(
                ids=region_ids, region=1*(seg.data>0),
                regionId=1, mode=distance_mode)
            regions.inset = reg_inset
            regions.data = reg_data

            # get ids of close regions 
            region_ids = ((dist <= max_distance) & (dist >= 0)).nonzero()[0]
            region_ids = region_ids.compress(region_ids>0)

        # make Segment to hold all neighborhoods
        all_hoods = Segment(data=seg.data, ids=ids, copy=False, clean=False)
        all_hoods.inset = seg.inset
        all_hoods.makeInset(ids=ids)
        all_hoods.data = np.zeros_like(all_hoods.data)

        # make neighborhoods for each region id
        for reg_id in region_ids:

            # make insets that contain segments and the current region id
            inset_curr = seg.findInset(
                ids=ids, mode='abs', additional=regions, additionalIds=[reg_id])
            seg_curr = seg.makeInset(
                ids=ids, additional=regions, additionalIds=[reg_id],
                update=False, returnCopy=True)
            regions.useInset(inset=inset_curr, mode='abs')

            # distances to the current region
            reg_dist_in = np.where(regions.data==reg_id, 0, 1)
            if (reg_dist_in > 0).all():  # workaround for scipy bug 1089
                raise ValueError("Can't calculate distance_function ",
                                 "(no background)")
            else:
                reg_dist = ndimage.distance_transform_edt(reg_dist_in)

            if max_distance is not None:
                seg_curr[reg_dist > max_distance] = 0
                
            # if a region overlaps with a segment remove the overlap and warn
            if remove_overlap and \
                   ((seg_curr > 0) & (regions.data == reg_id)).any():
                seg_curr = np.where(regions.data==reg_id, 0, seg_curr)
                logging.warning(
                    "Density.calculateNeighbourhood: region " 
                    + str(reg_id) + " overlap with segments." 
                    + "Removed the overlap from segments." )

            # make a neighbourhood of current region on each segment
            if size is not None:

                hood_data = np.zeros_like(seg_curr)
                for seg_id in ids:

                    # find the closest point on the current segment
                    min_, _, min_pos, max_pos = \
                        ndimage.extrema(input=reg_dist, labels=seg_curr,
                                        index=seg_id)

                    # find inset that contains only the current segment 
                    fine_inset = ndimage.find_objects(seg_curr==seg_id)
                    try:
                        fine_inset = fine_inset[0]
                    except IndexError:
                        continue

                    # prepare input distance array for hood 
                    hood_in = np.ones(shape=seg_curr.shape)
                    hood_in[min_pos] = 0

                    # make hood for this region and segment
                    if (hood_in[fine_inset] > 0).all():  # workaround for
                                                         # scipy bug 1089
                        raise ValueError("Can't calculate distance_function ",
                                         "(no background)")
                    else:
                        hood_dist = ndimage.distance_transform_edt(
                            hood_in[fine_inset])
                    fine_hood_data = hood_data[fine_inset]
                    fine_seg = seg_curr[fine_inset]
                    fine_hood_data[
                        (fine_seg == seg_id) & (hood_dist <= size/2.)] = seg_id

            else:
                hood_data = seg.data

            # make hood instance
            hood = Segment(data=hood_data, ids=ids, copy=True)
            hood.inset = seg.inset

            # add the current hood to all hoods
            hood.useInset(inset=all_hoods.inset, mode='abs')
            all_hoods.data = np.where(hood.data>0, 
                                         hood.data, all_hoods.data)
                
            # revert to initial insets and data
            #seg.inset = seg_inset
            #seg.data = seg_data
            regions.inset = reg_inset
            regions.data = reg_data

            # yield
            yield reg_id, hood, all_hoods

