# Satellite eclipse calculations modeule

import numpy as np
import numbers
import datetime
import rocs.checkutils as checkutils


class Eclipse:

    """
    Class for eclipsing satellite caclulations

    """

    def __init__(self,r_sat,r_sun,eclipsing_body,radius_eclbody,
                                                        r_eclbody=None):

        """
        Initialize Eclipse class

        Keyword arguments:
            r_sat [list or numpy array] : 3-column array/list where the columns
                                         represent the x,y,z coordinates of
                                         the satellite in an ECEF frame
            r_sun [list or numpy array] : 3-column array/list where the columns
                                         represent the x,y,z coordinates of
                                         the Sun in an ECEF frame
            eclipsing_body [str]       : name of the eclipsing body
                                         (Earth, Moon, etc.)
            radius_eclbody [number] : radius of the eclipsing body in meters
            r_eclbody [list or numpy array] : 3-column array/list where the
                                         columns represent the x,y,z
                                         coordinates of the eclipsing body in
                                         an ECEF frame. If the eclipsing body
                                         is Earth, not required as it will be
                                         zeros

        Updates:
            self.r_sat [numpy array]
            self.r_sun [numpy array]
            self.eclipsing_body [str]
            self.r_eclbody [numpy array]
            self.l [numpy array] : fraction of solar disk seen by satellite
                                   l = 1.0       : no eclipse
                                   l = 0.0       : full eclipse
                                   0.0 < l < 1.0 : partial eclipse
        """

        # Check the given arguments and set the attributes
        checkutils.check_coords(r_sat,ncol=3,minrows=1)
        r_sat = np.array(r_sat)
        self.r_sat = r_sat

        if np.shape(np.array(r_sun)) != np.shape(np.array(r_sat)):
            raise ValueError("r_sat and r_sun must be the same shape!")
        r_sun = np.array(r_sun)
        self.r_sun = r_sun

        if not isinstance(eclipsing_body,str):
            raise TypeError("The given eclipsing_body must be of string type!")
        if eclipsing_body == 'Earth' or eclipsing_body == 'earth':
            r_eclbody = np.zeros_like(r_sun)
        else:
            if r_eclbody is None:
                raise ValueError("For an eclipsing body other than the Earth, "
                                 "r_eclbody must be given!")
            elif np.shape(np.array(r_eclbody)) != np.shape(np.array(r_sat)):
                raise ValueError("r_eclbody must be the same shape as r_sat "
                                     "and r_sun!")
            r_eclbody = np.array(r_eclbody)
        self.r_eclbody = r_eclbody

        if not isinstance(radius_eclbody,numbers.Number):
            raise TypeError("The given radius_eclbody must be a number")
        self.radius_eclbody = radius_eclbody

        # radii of the Sun
        radius_sun = 696340000

        # Perform preliminary calculations

        # vector of eclipsing body -> Sun
        r_eclbody_sun = r_sun - r_eclbody

        # vector of Sun -> satellite
        r_sun_sat = r_sat - r_sun

        # distance between the eclipsing body and the Sun
        d_eclbody_sun = np.linalg.norm(r_eclbody_sun,axis=1)

        # distance between the satellite and the Sun
        d_sun_sat = np.linalg.norm(r_sun_sat,axis=1)

        # vector of eclsiping body -> satellite
        r_eclbody_sat = r_sat - r_eclbody

        # unit vector of Sun -> satellite
        u_sun_sat = r_sun_sat/d_sun_sat[:,None]

        # projection of eclipsing_body --> satellite vector onto the
        # Sun --> satellite vector (dot product of the two vectors)
        proj = np.sum(r_eclbody_sat*u_sun_sat,axis=1)

        # cross product of the eclipsing_body --> satellite vector and the
        # Sun --> satellite unit vector
        cr = np.cross(r_eclbody_sat,u_sun_sat)

        # apparent seperation of the center of the Sun and the eclipsing body
        sep = np.linalg.norm(cr,axis=1)/proj

        # apparent radii of the Sun and the eclipsing body as seen from
        # the satellite
        r_sun_apparent = radius_sun/d_sun_sat
        r_eclbody_apparent = radius_eclbody/proj

        # Calculate lambda, the fraction of the Sun disk visible from the
        # satellite

        # Set lambda to 1.0 initially (i.e. no eclipse)
        l = np.ones(len(r_sat))

        # Go through where eclipse is possible to check if there is an eclipse
        # impossible eclipse; lambda = 0
        no_eclipse = (d_sun_sat <= d_eclbody_sun)

        # possible eclipse
        possible_eclipse = (r_sun_apparent + r_eclbody_apparent) > sep
        possible_eclipse = np.logical_and(possible_eclipse,~no_eclipse)

        # full eclipse; lambda remains 1
        full_eclipse = (r_eclbody_apparent + r_sun_apparent) >= sep
        full_eclipse = np.logical_and(full_eclipse,possible_eclipse)
        ind = np.where(full_eclipse)
        l[ind] = 0.0

        #partial eclipse; lambda needs to be calculated
        partial_eclipse = np.logical_and(possible_eclipse,~full_eclipse)

        # eclipsing body lies in the Sun's disk
        ecl_lies_in_sun = (r_sun_apparent - r_eclbody_apparent) >= sep
        ecl_lies_in_sun = np.logical_and(ecl_lies_in_sun,partial_eclipse)
        ind = np.where(ecl_lies_in_sun)
        l[ind] = ((r_sun_apparent[ind]**2-r_eclbody_apparent[ind]**2)
                                                    /r_sun_apparent[ind]**2)

        # Otherwise, eclipsing body and the Sun make an intersection
        intersecting_disks = np.logical_and(partial_eclipse,~ecl_lies_in_sun)
        ind_insc = np.where(intersecting_disks)

        # r of smaller disk and larger disk
        r_small_disk = np.array([min(l1, l2) for l1, l2 in
                                zip(r_sun_apparent,r_eclbody_apparent)])
        r_large_disk = np.array([max(l1, l2) for l1, l2 in
                                zip(r_sun_apparent,r_eclbody_apparent)])

        # one disk much larger than the other one
        much_larger_disk = r_large_disk/r_small_disk > 5.0
        much_larger_disk = np.logical_and(much_larger_disk,intersecting_disks)
        close_disk_size = np.logical_and(~much_larger_disk,intersecting_disks)

        # half of the angle subtended in the smaller disk by arc of
        # intersection
        phi = np.full_like(r_small_disk,np.nan)
        phi[ind_insc] = np.arccos(
            (r_small_disk[ind_insc]**2 + sep[ind_insc]**2
                - r_large_disk[ind_insc]**2)
                /(2.0*r_small_disk[ind_insc]*sep[ind_insc]))

        area1 = (np.pi-phi)*r_small_disk**2
        area2 = np.full_like(area1,np.nan)
        area3 = np.full_like(area1,np.nan)
        hgt = np.full_like(area1,np.nan)
        theta = np.full_like(area1,np.nan)

        ind = np.where(much_larger_disk)
        hgt[ind] = np.sqrt(
                r_small_disk[ind]**2-(sep[ind]-r_large_disk[ind])**2)
        area2[ind] = hgt[ind]*(sep[ind]-r_large_disk[ind])
        area3[ind] = 0.0

        ind = np.where(close_disk_size)
        hgt[ind] = r_small_disk[ind]*np.sin(phi[ind])
        theta[ind] = np.arcsin(hgt[ind]/r_large_disk[ind])
        area2[ind] = sep[ind]*hgt[ind]
        area3[ind] = theta[ind]*r_large_disk[ind]**2

        # area of non-overlapped portion of the small disc
        area = area1 + area2 + area3

        # redfine area1 and area2 based on which disk is the smaller one
        area1 = np.pi*r_sun_apparent**2

        # eclipsing body is the smaller disk
        ecl_smaller = r_sun_apparent > r_eclbody_apparent
        ecl_smaller = np.logical_and(ecl_smaller,intersecting_disks)
        ind = np.where(ecl_smaller)
        area2[ind] = np.pi*r_eclbody_apparent[ind]**2
        l[ind] = (area1[ind]+area[ind]-area2[ind])/area1[ind]

        # Sun is the smaller disk
        sun_smaller = r_sun_apparent <= r_eclbody_apparent
        sun_smaller = np.logical_and(sun_smaller,intersecting_disks)
        ind = np.where(sun_smaller)
        l[ind] = area[ind]/area1[ind]

        self.l = l


    def get_ecl_times(self,time):

        """
        Get the eclipse times in  afrom-to format

        Keyword arguments:
            time [list or numpy array] : array/list of datetime objects
                                         corresponding to the positions given
                                         in initialization

        Updates:
            self.eclipsing [str] : flag for showing if the given satellite
                                   experiences any eclipsing by the given body
                                   'full'    : experiences full eclipse
                                   'partial' : experiences only partial elipse
                                               (but not full eclipse)
                                   'none'    : does not experience any eclipse
            self.ecl_times [numpy array] : 3-column array containing eclipse
                                           times where columns represent:
                                           [time_from,time_to,eclipse_type]
        """

        # Check the given time
        if not isinstance(time,(list,np.ndarray,datetime.datetime)):
            raise TypeError("The given time needs to be either a datetime "
                            "object or a list/array of datetime objects")
        if not all(isinstance(item,datetime.datetime)
                        for item in np.atleast_1d(time)):
            raise TypeError("There are non-datetime items in time")
        if np.shape(time) != (len(self.r_sat),):
            raise ValueError("The given time must be either a datetime"
                                 " object or a 1d array/list with the same "
                                 "length as the given r_sat.")
        self.time = np.atleast_1d(time)

        # time resolution
        differences = ([abs(t2 - t1).seconds for t1,t2 in
                                    zip(self.time[:-1], self.time[1:])])
        tres = max(differences)

        ecl_times = []
        eclipsing = 'none'

        # Look for full eclipses
        ind = np.where(self.l==0)
        time_eclipse = self.time[ind]
        if time_eclipse.size:
            eclipsing = 'full'
            time_from = time_eclipse[0]
            for c,t in enumerate(time_eclipse):
                if (c<len(time_eclipse)-1):
                    next_time = time_eclipse[c+1]
                    if ((next_time-t).seconds > tres):
                        time_to = t
                        ecl_times.append([time_from,time_to,'full'])
                        time_from = next_time
            time_to = t
            ecl_times.append([time_from,time_to,'full'])

        # Look for partial eclipses
        ind = np.where((self.l>0) & (self.l<1))
        time_eclipse = self.time[ind]
        if time_eclipse.size:
            if eclipsing == 'none':
                eclipsing = 'partial'
            time_from = time_eclipse[0]
            for c,t in enumerate(time_eclipse):
                if (c<len(time_eclipse)-1):
                    next_time = time_eclipse[c+1]
                    if ((next_time-t).seconds > tres):
                        time_to = t
                        ecl_times.append([time_from,time_to,'partial'])
                        time_from = next_time
            time_to = t
            ecl_times.append([time_from,time_to,'partial'])

        self.ecl_times = ecl_times
        self.eclipsing = eclipsing




