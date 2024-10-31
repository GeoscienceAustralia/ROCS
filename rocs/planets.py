# Planetary position calculations module

import numpy as np
import datetime
import numbers
from rocs.gpscal import gpsCal
import rocs.coordinates as coordinates


class AnalyticalPosition:

    """
    Class for calculating the position of a planet based on analytical solution
    Valid between 1950 to 2050

    """

    def __init__(self,planet,ref_frame,time_utc,ut1_utc,xp=None,yp=None):

        """
        Initialize PlanetPositionAnalytical class

        Keyword arguments:
            planet [str]    : planet for which the position is to be calculated
            ref_frame [str] : the reference frame for the position of the
                              planet
            time_utc [datetime or list/array of datetimes] : UTC time(s) for
                                                            calculating the
                                                            planetary positions
            ut1_utc [scalar or list/array] : UT1-UTC in seconds corresponding
                                             to the time_utc attribute of the
                                             class object
            xp [scalar or list/array]   : polar x motion in radians
                                          corresponding to the time_utc
                                          attribute of the class object
            yp [scalar or list/array]   : polar y motion in radians
                                          corresponding to the time_utc
                                          attribute of the class object

        Updates:
            self.planet [str]
            self.ref_frame [str]
            self.time_utc [array]
            self.ut1_utc [array]
            self.time_ut1 [array]
            self.xp [array]
            self.yp [array]
            self.r [numpy array] : 3-column array where the columns represent
                                   the x,y,z of the planet position in the
                                   specified reference frame, and the rows
                                   correspond to the given time_utc epochs
        References:
            - U.S. Nautical Almanac Office and U.S. Naval Observatory (2007)
              The astronomical almanac for the year 2007.
            - Ferrao (2013) Positioning with combined GPS and GLONASS
              observations, MSc thesis, Tecnico Lisbona.
            - Michalsky (1988) The Astronomical almanac's algorithm for
              approximate solar position (1950-2050). Solar Energy.
            - Seidelman (2007) Explanatory supplement to the astronomical
              almanac, US Naval Observatory.

        """

        # Check the given arguments and set the attributes
        if not isinstance(planet,str):
            raise TypeError("The input planet needs to be a string")
        allowed_planets = ['sun','moon']
        if planet not in allowed_planets:
            raise TypeError(f"The input planet {planet} not recognized!\n"
                            f"Allowed planets: {allowed_planets}")
        self.planet = planet

        if not isinstance(ref_frame,str):
            raise TypeError("The input ref_frame needs to be a string")
        allowed_ref_frames = ['ECI','ECEF']
        if ref_frame not in allowed_ref_frames:
            raise TypeError(f"The input reference frame {ref_frame} not "
                            f"recognized!\nAllowed reference frames: "
                            f"{allowed_ref_frames}")
        self.ref_frame =  ref_frame

        if not isinstance(time_utc,(list,np.ndarray,datetime.datetime)):
            raise TypeError("The input time_utc needs to be either a datetime "
                            "object or a list/array of datetime objects")
        if not all(isinstance(item,datetime.datetime)
                        for item in np.atleast_1d(time_utc)):
            raise TypeError("There are non-datetime items in time_utc")
        if (any(item < datetime.datetime(1950,1,1,0,0)
                for item in np.atleast_1d(time_utc))
            or any(item >= datetime.datetime(2050,1,1,0,0)
                for item in np.atleast_1d(time_utc))):
                raise ValueError("The analytical equations for calculating "
                                 "the approximate alamanc positions are only "
                                 "valid between 1950-2050!")
        self.time_utc = np.atleast_1d(time_utc)

        if not isinstance(ut1_utc,(list,np.ndarray,numbers.Number)):
            raise TypeError("The given ut1_utc needs to be either a number "
                            "or a list/array of numbers")
        if not all(isinstance(item,numbers.Number)
                        for item in np.atleast_1d(ut1_utc)):
            raise TypeError("There are non-number items in ut1_utc")
        if np.shape(np.atleast_1d(ut1_utc)) != np.shape(self.time_utc):
            raise ValueError("Shape mismatch between self.time_utc "
                            f"{np.shape(self.time_utc)} and ut1_utc "
                            f"{np.shape(np.atleast_1d(ut1_utc))}")
        self.ut1_utc = np.atleast_1d(ut1_utc)

        # For conversion to ECEF, we need xp and yp
        if (ref_frame == 'ECEF'):
            if (xp is None or yp is None):
                raise ValueError("xp and yp must be given for "
                                 "ECI to ECEF conversion")

            if not isinstance(xp,(list,np.ndarray,numbers.Number)):
                raise TypeError("The given xp needs to be either a "
                                "number or a list/array of numbers")
            if not all(isinstance(item,numbers.Number)
                            for item in np.atleast_1d(xp)):
                raise TypeError("There are non-number items in xp")
            if np.shape(np.atleast_1d(xp)) != np.shape(self.time_utc):
                raise ValueError("Shape mismatch between self.time_utc "
                                f"{np.shape(self.time_utc)} and xp "
                                f"{np.shape(np.atleast_1d(xp))}")
            self.xp = np.atleast_1d(xp)

            if not isinstance(yp,(list,np.ndarray,numbers.Number)):
                raise TypeError("The given yp needs to be either a "
                                "number or a list/array of numbers")
            if not all(isinstance(item,numbers.Number)
                            for item in np.atleast_1d(yp)):
                raise TypeError("There are non-number items in yp")
            if np.shape(np.atleast_1d(yp)) != np.shape(self.time_utc):
                raise ValueError("Shape mismatch between self.time_utc "
                                f"{np.shape(self.time_utc)} and yp "
                                f"{np.shape(np.atleast_1d(yp))}")
            self.yp = np.atleast_1d(yp)

        # convert the UTC time to UT1
        time_ut1 = []
        for c,utc in enumerate(self.time_utc):
            gc = gpsCal()
            gc.set_yyyy_MM_dd_hh_mm_ss(utc.year,utc.month,utc.day,utc.hour,
                                                        utc.minute,utc.second)
            ut1 = gc.ut1(self.ut1_utc[c])
            time_ut1.append(ut1)
        self.time_ut1 = np.atleast_1d(time_ut1)

        # Calculate the number of Julian centuries since JD2000
        # (2000/01/01 at 12:00)
        jd2000 = datetime.datetime(2000,1,1,12,0,0)
        t = np.array([(time-jd2000).total_seconds()/86400.0/36525.0
                            for time in self.time_ut1])

        # Calculate the positions in ECI, then convert to ECEF if required
        if self.planet == 'sun':

            # obliquity of the ecliptic [degrees]
            eps = 23.439291 - 0.0130042*t

            # mean anomaly [degress]
            m = 357.5277233 + 35999.05034*t
            m = np.array([item%360.0 for item in m])

            # ecliptic longitude [degrees]
            l = (280.460 + 36000.770*t + 1.914666471*np.sin(np.deg2rad(m))
                                       + 0.000139589*np.cos(np.deg2rad(2*m)))
            l = np.array([item%360.0 for item in l])

            # astronomical unit (average distance between Earth and Sun
            #                    [meters])
            au = 149597870691.0

            # distance between the Earth and the Sun [meters]
            r_magnitude = au*(1.000140612 - 0.016708617*np.cos(np.deg2rad(m))
                                - 0.000139589*np.cos(np.deg2rad(2*m)))

            # vector of the Sun position in ECI [meters]
            r0 = r_magnitude*np.cos(np.deg2rad(l))
            r1 = r_magnitude*np.cos(np.deg2rad(eps))*np.sin(np.deg2rad(l))
            r2 = r_magnitude*np.sin(np.deg2rad(eps))*np.sin(np.deg2rad(l))
            r = np.transpose(np.vstack((r0,r1,r2)))

        elif self.planet == 'moon':

            # perturbing factors from the Earth's nutation [radians]
            f0 = (3600.0*134.96340251 + 1717915923.2178*t + 31.8792*t**2
                    + 0.051635*t**3 - 0.00024470*t**4)/3600.0
            f0 = np.array([item - int(item/360.0)*360.0 for item in f0])
            f0 = np.array([item%360.0 for item in f0])
            f0 = np.deg2rad(f0)

            f1 = (3600.0*357.52910918 + 129596581.0481*t - 0.5532*t**2
                    + 0.000136*t**3 - 0.00001149*t**4)/3600.0
            f1 = np.array([item%360.0 for item in f1])
            f1 = np.deg2rad(f1)

            f2 = (3600.0*93.27209062 + 1739527262.8478*t - 12.7512*t**2
                    - 0.001037*t**3 + 0.00000417*t**4)/3600.0
            f2 = np.array([item%360.0 for item in f2])
            f2 = np.deg2rad(f2)

            f3 = (3600.0*297.85019547 + 1602961601.2090*t - 6.3706*t**2
                    + 0.006593*t**3 - 0.00003169*t**4)/3600.0
            f3 = np.array([item%360.0 for item in f3])
            f3 = np.deg2rad(f3)

            # obliquity of the ecliptic [degrees]
            eps = 23.439291 - 0.0130042*t

            # ecliptic longitude [degrees]
            l = (218.32 + 481267.883*t + 6.29*np.sin(f0)
                    - 1.27*np.sin(f0-2*f3) + 0.66*np.sin(2*f3)
                    + 0.21*np.sin(2*f0) - 0.19*np.sin(f1)
                    - 0.11*np.sin(2*f2))
            l = np.array([item%360.0 for item in l])

            # perturbation factor [degrees]
            beta = (5.13*np.sin(f2) + 0.28*np.sin(f0+f2)
                    - 0.28*np.sin(f2-f0) - 0.17*np.sin(f2-2*f3))
            #beta = np.array([item - int(item/360.0)*360.0 for item in beta])
            beta = np.array([item%360.0 for item in beta])

            # scaling factor? [degrees]
            pi = (0.9508 + 0.0518*np.cos(f0) + 0.0095*np.cos(f0-2*f3)
                    + 0.0078*np.cos(2*f3) + 0.0028*np.cos(2*f0))
            pi = np.array([item%360.0 for item in pi])

            # Earth's equatorial radius (IERS2003) [meters]
            a = 6378136.6

            # distance between the Moon and Earth [meters]
            r_magnitude = a/np.sin(np.deg2rad(pi))

            # vector of the moon position in ECI [meters]
            r0 = r_magnitude*np.cos(np.deg2rad(beta))*np.cos(np.deg2rad(l))
            r1 = r_magnitude*(np.cos(np.deg2rad(eps))*np.cos(np.deg2rad(beta))
                    *np.sin(np.deg2rad(l))
                    - np.sin(np.deg2rad(eps))*np.sin(np.deg2rad(beta)) )
            r2 = r_magnitude*(np.sin(np.deg2rad(eps))*np.cos(np.deg2rad(beta))
                    *np.sin(np.deg2rad(l))
                    + np.cos(np.deg2rad(eps))*np.sin(np.deg2rad(beta)) )
            r = np.transpose(np.vstack((r0,r1,r2)))

        else:
            raise TypeError(f"The input planet {self.planet} not recognized!")

        if self.ref_frame == 'ECI':

            pass

        elif self.ref_frame == 'ECEF':

            # convert to ECEF
            coords = coordinates.Rectangular(r,'ECI',self.time_utc)

            coords.ToECEF(iau_model='IAU76/80/82/94',
                        transformation_method='equinox',
                        evaluation_method='classical_angles',
                        ut1_utc=self.ut1_utc,xp=self.xp,yp=self.yp)
            r = coords.coords

        self.r = r

