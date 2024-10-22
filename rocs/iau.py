# International Astronomical Union models module

import numpy as np
import datetime
import numbers
from rocs.gpscal import gpsCal
from rocs.rotation import Rotation


class IAU:

    """
    Class if IAU models
    Based on IERS conventions and SOFA package
    References:
        International Astronomival Union  (2020): SOFA tools for Earth Attitude
        Petit and Luzum eds. (2010): IERS Conventions (2010) (IERS Technical
        Note; No. 36), International Earth Rotation and Reference Systems
        Service (IERS) Central Bureau

    """
    def __init__(self,time_utc):

        """
        Initialize iau model class

        Keyword arguments:
            time_utc [datetime or list/array of datetimes] : UTC time(s)

        Updates:
            self.time_utc [array of datetimes]

        """

        # Check the given arguments and set the attributes
        if not isinstance(time_utc,(list,np.ndarray,datetime.datetime)):
            raise TypeError("The given time_utc needs to be either a datetime "
                            "object or a list/array of datetime objects")
        if not all(isinstance(item,datetime.datetime)
                        for item in np.atleast_1d(time_utc)):
            raise TypeError("There are non-datetime items in time_utc")
        self.time_utc = np.atleast_1d(time_utc)

        # time calculations
        # Calculate t, which is time from 2000/01/01 12:00 UTC in centuries
        # measured in international atomic time (no leap seconds)
        tt = []
        for utc in self.time_utc:
            gc = gpsCal()
            gc.set_yyyy_MM_dd_hh_mm_ss(utc.year,utc.month,utc.day,utc.hour,
                                                        utc.minute,utc.second)
            tai = gc.tai()
            tt.append(tai + datetime.timedelta(0,32.184))
        jd2000 = datetime.datetime(2000,1,1,12,0,0)
        t = np.array([(time-jd2000).total_seconds()/86400.0/36525.0
                                                        for time in tt])
        self.t = t


    def fundamental_fk5(self):
        """
        Calculate fundamental arguments in the FK5 reference frame

        Updates:
            self.fundamental_args [array] : array of fundamental arguments
                                            where the columns correspond to
                                            time epochs, and the rows are as
                                            below:
                row 0: el : mean longitude of the Moon minus mean longitude of
                            the Moon's perigee [radians]
                row 1: elp : mean longitude of the Sun minus mean longitude of
                             the Sun's perigee [radians]
                row 2: f : mean longitude of the Moon minus mean longitude of
                           the Moon's node [radians]
                row 3: d : mean elongation of the Moon from the Sun [radians]
                row 4: om : longitude of the mean ascending node of the lunar
                            orbit on the ecliptic, measured from the mean
                            equinox of date [radians]
        """
        t = self.t

        # mean longitude of the Moon minus mean longitude of the Moon's
        # perigee [radians]
        el  = (((485866.733 + 715922.633*t + 31.310*t**2 + 0.064*t**3)/3600.0)
                                                                *np.pi/180.0
                + ((1325.0*t)%1.0)*2*np.pi)
        el = el%(2*np.pi)

        # mean longitude of the Sun minus mean longitude of the Sun's perigee
        # [radians]
        elp = (((1287099.804 + 1292581.224*t - 0.577*t**2 - 0.012*t**3)/3600.0)
                                                                *np.pi/180.0
                + ((99.0*t)%1.0)*2*np.pi)
        elp = elp%(2*np.pi)

        # mean longitude of the Moon minus mean longitude of the Moon's node
        # [radians]
        f  = (((335778.877 + 295263.137*t - 13.257*t**2 + 0.011*t**3)/3600.0)
                                                                *np.pi/180.0
                + ((1342.0*t)%1.0)*2*np.pi)
        f = f%(2*np.pi)

        # mean elongation of the Moon from the Sun [radians]
        d  = (((1072261.307 + 1105601.328*t - 6.891*t**2 + 0.019*t**3)/3600.0)
                                                                *np.pi/180.0
                + ((1236.0*t)%1.0)*2*np.pi)
        d = d%(2*np.pi)

        # longitude of the mean ascending node of the lunar orbit on the
        # ecliptic, measured from the mean equinox of date [radians]
        om  = (((450160.280 - 482890.539*t + 7.455*t**2 + 0.008*t**3)/3600.0)
                                                                *np.pi/180.0
                + ((-5.0*t)%1.0)*2*np.pi)
        om = om%(2*np.pi)

        # Create an array of fundamental arguments
        self.fundamental_args = np.column_stack((el,elp,f,d,om)).T


    def precession_iau1976(self):

        """
        Create the precession matrices based on the IAU1976 model

        Updates:
            self.precession [dict] : precession matrix for each UTC time given
            self.precession_model [str] : precession model

        """

        t = self.t

        # Euler angles [arc degrees]
        zeta = (2306.2181*t + 0.30188*t**2 + 0.017998*t**3)/3600.0
        zeta = np.array([item - int(item/360.0)*360.0
                                            for item in zeta])
        z = (2306.2181*t + 1.09468*t**2 + 0.018203*t**3)/3600.0
        z = np.array([item - int(item/360.0)*360.0 for item in z])
        theta = (2004.3109*t - 0.42665*t**2 - 0.041833*t**3)/3600.0
        theta = np.array(
                [item - int(item/360.0)*360.0 for item in theta])

        # Go epochwise (may need to optimize in future)
        precession = {}
        for c,epoch in enumerate(t):

            # Precession rotation matrix precession = r3.r2.r1
            r1 = Rotation(-np.deg2rad(zeta[c]),3).rot
            r2 = Rotation(np.deg2rad(theta[c]),2).rot
            r3 = Rotation(-np.deg2rad(z[c]),3).rot
            precession[epoch] = np.matmul(r3,np.matmul(r2,r1))

        self.precession = precession
        self.precession_model = "iau1976"

    def nutation_iau1980(self):

        """
        Create the nutation matrices based on the IAU1980 model

        Updates:
            self.fundamental_args [array] : see fundamental_fk5
            self.nutation [dict] : nutation matrix for each UTC time given
            self.nutation_model [str] : nutation model
            self.dpsi [array] : nutation in longitude for each UTC time
            self.deps [array] : nutation in obliquity for each UTC time
            self.eps0 [array] : mean obliquity of the ecliptic for each UTC
                                time

        """

        # Table of multiples of arguments and coefficients for nutation
        # The columns are:
        #       Multiple of                 Longitude           Obliquity
        #  L    L'    F     D    Omega    coeff. of sin      coeff. of cos
        #                                   1         t        1       t
        # units are 0.1 mas fir coefficients and mas per Julian millennium for
        # rates of change
        nut = np.array((
        [ 0.0,  0.0,  0.0,  0.0,  1.0, -171996.0, -1742.0,  92025.0,  89.0],
        [ 0.0,  0.0,  0.0,  0.0,  2.0,    2062.0,     2.0,   -895.0,   5.0],
        [-2.0,  0.0,  2.0,  0.0,  1.0,      46.0,     0.0,    -24.0,   0.0],
        [ 2.0,  0.0, -2.0,  0.0,  0.0,      11.0,     0.0,      0.0,   0.0],
        [-2.0,  0.0,  2.0,  0.0,  2.0,      -3.0,     0.0,      1.0,   0.0],
        [ 1.0, -1.0,  0.0, -1.0,  0.0,      -3.0,     0.0,      0.0,   0.0],
        [ 0.0, -2.0,  2.0, -2.0,  1.0,      -2.0,     0.0,      1.0,   0.0],
        [ 2.0,  0.0, -2.0,  0.0,  1.0,       1.0,     0.0,      0.0,   0.0],
        [ 0.0,  0.0,  2.0, -2.0,  2.0,  -13187.0,   -16.0,   5736.0, -31.0],
        [ 0.0,  1.0,  0.0,  0.0,  0.0,    1426.0,   -34.0,     54.0,  -1.0],
        [ 0.0,  1.0,  2.0, -2.0,  2.0,    -517.0,    12.0,    224.0,  -6.0],
        [ 0.0, -1.0,  2.0, -2.0,  2.0,     217.0,    -5.0,    -95.0,   3.0],
        [ 0.0,  0.0,  2.0, -2.0,  1.0,     129.0,     1.0,    -70.0,   0.0],
        [ 2.0,  0.0,  0.0, -2.0,  0.0,      48.0,     0.0,      1.0,   0.0],
        [ 0.0,  0.0,  2.0, -2.0,  0.0,     -22.0,     0.0,      0.0,   0.0],
        [ 0.0,  2.0,  0.0,  0.0,  0.0,      17.0,    -1.0,      0.0,   0.0],
        [ 0.0,  1.0,  0.0,  0.0,  1.0,     -15.0,     0.0,      9.0,   0.0],
        [ 0.0,  2.0,  2.0, -2.0,  2.0,     -16.0,     1.0,      7.0,   0.0],
        [ 0.0, -1.0,  0.0,  0.0,  1.0,     -12.0,     0.0,      6.0,   0.0],
        [-2.0,  0.0,  0.0,  2.0,  1.0,      -6.0,     0.0,      3.0,   0.0],
        [ 0.0, -1.0,  2.0, -2.0,  1.0,      -5.0,     0.0,      3.0,   0.0],
        [ 2.0,  0.0,  0.0, -2.0,  1.0,       4.0,     0.0,     -2.0,   0.0],
        [ 0.0,  1.0,  2.0, -2.0,  1.0,       4.0,     0.0,     -2.0,   0.0],
        [ 1.0,  0.0,  0.0, -1.0,  0.0,      -4.0,     0.0,      0.0,   0.0],
        [ 2.0,  1.0,  0.0, -2.0,  0.0,       1.0,     0.0,      0.0,   0.0],
        [ 0.0,  0.0, -2.0,  2.0,  1.0,       1.0,     0.0,      0.0,   0.0],
        [ 0.0,  1.0, -2.0,  2.0,  0.0,      -1.0,     0.0,      0.0,   0.0],
        [ 0.0,  1.0,  0.0,  0.0,  2.0,       1.0,     0.0,      0.0,   0.0],
        [-1.0,  0.0,  0.0,  1.0,  1.0,       1.0,     0.0,      0.0,   0.0],
        [ 0.0,  1.0,  2.0, -2.0,  0.0,      -1.0,     0.0,      0.0,   0.0],
        [ 0.0,  0.0,  2.0,  0.0,  2.0,   -2274.0,    -2.0,    977.0,  -5.0],
        [ 1.0,  0.0,  0.0,  0.0,  0.0,     712.0,     1.0,     -7.0,   0.0],
        [ 0.0,  0.0,  2.0,  0.0,  1.0,    -386.0,    -4.0,    200.0,   0.0],
        [ 1.0,  0.0,  2.0,  0.0,  2.0,    -301.0,     0.0,    129.0,  -1.0],
        [ 1.0,  0.0,  0.0, -2.0,  0.0,    -158.0,     0.0,     -1.0,   0.0],
        [-1.0,  0.0,  2.0,  0.0,  2.0,     123.0,     0.0,    -53.0,   0.0],
        [ 0.0,  0.0,  0.0,  2.0,  0.0,      63.0,     0.0,     -2.0,   0.0],
        [ 1.0,  0.0,  0.0,  0.0,  1.0,      63.0,     1.0,    -33.0,   0.0],
        [-1.0,  0.0,  0.0,  0.0,  1.0,     -58.0,    -1.0,     32.0,   0.0],
        [-1.0,  0.0,  2.0,  2.0,  2.0,     -59.0,     0.0,     26.0,   0.0],
        [ 1.0,  0.0,  2.0,  0.0,  1.0,     -51.0,     0.0,     27.0,   0.0],
        [ 0.0,  0.0,  2.0,  2.0,  2.0,     -38.0,     0.0,     16.0,   0.0],
        [ 2.0,  0.0,  0.0,  0.0,  0.0,      29.0,     0.0,     -1.0,   0.0],
        [ 1.0,  0.0,  2.0, -2.0,  2.0,      29.0,     0.0,    -12.0,   0.0],
        [ 2.0,  0.0,  2.0,  0.0,  2.0,     -31.0,     0.0,     13.0,   0.0],
        [ 0.0,  0.0,  2.0,  0.0,  0.0,      26.0,     0.0,     -1.0,   0.0],
        [-1.0,  0.0,  2.0,  0.0,  1.0,      21.0,     0.0,    -10.0,   0.0],
        [-1.0,  0.0,  0.0,  2.0,  1.0,      16.0,     0.0,     -8.0,   0.0],
        [ 1.0,  0.0,  0.0, -2.0,  1.0,     -13.0,     0.0,      7.0,   0.0],
        [-1.0,  0.0,  2.0,  2.0,  1.0,     -10.0,     0.0,      5.0,   0.0],
        [ 1.0,  1.0,  0.0, -2.0,  0.0,      -7.0,     0.0,      0.0,   0.0],
        [ 0.0,  1.0,  2.0,  0.0,  2.0,       7.0,     0.0,     -3.0,   0.0],
        [ 0.0, -1.0,  2.0,  0.0,  2.0,      -7.0,     0.0,      3.0,   0.0],
        [ 1.0,  0.0,  2.0,  2.0,  2.0,      -8.0,     0.0,      3.0,   0.0],
        [ 1.0,  0.0,  0.0,  2.0,  0.0,       6.0,     0.0,      0.0,   0.0],
        [ 2.0,  0.0,  2.0, -2.0,  2.0,       6.0,     0.0,     -3.0,   0.0],
        [ 0.0,  0.0,  0.0,  2.0,  1.0,      -6.0,     0.0,      3.0,   0.0],
        [ 0.0,  0.0,  2.0,  2.0,  1.0,      -7.0,     0.0,      3.0,   0.0],
        [ 1.0,  0.0,  2.0, -2.0,  1.0,       6.0,     0.0,     -3.0,   0.0],
        [ 0.0,  0.0,  0.0, -2.0,  1.0,      -5.0,     0.0,      3.0,   0.0],
        [ 1.0, -1.0,  0.0,  0.0,  0.0,       5.0,     0.0,      0.0,   0.0],
        [ 2.0,  0.0,  2.0,  0.0,  1.0,      -5.0,     0.0,      3.0,   0.0],
        [ 0.0,  1.0,  0.0, -2.0,  0.0,      -4.0,     0.0,      0.0,   0.0],
        [ 1.0,  0.0, -2.0,  0.0,  0.0,       4.0,     0.0,      0.0,   0.0],
        [ 0.0,  0.0,  0.0,  1.0,  0.0,      -4.0,     0.0,      0.0,   0.0],
        [ 1.0,  1.0,  0.0,  0.0,  0.0,      -3.0,     0.0,      0.0,   0.0],
        [ 1.0,  0.0,  2.0,  0.0,  0.0,       3.0,     0.0,      0.0,   0.0],
        [ 1.0, -1.0,  2.0,  0.0,  2.0,      -3.0,     0.0,      1.0,   0.0],
        [-1.0, -1.0,  2.0,  2.0,  2.0,      -3.0,     0.0,      1.0,   0.0],
        [-2.0,  0.0,  0.0,  0.0,  1.0,      -2.0,     0.0,      1.0,   0.0],
        [ 3.0,  0.0,  2.0,  0.0,  2.0,      -3.0,     0.0,      1.0,   0.0],
        [ 0.0, -1.0,  2.0,  2.0,  2.0,      -3.0,     0.0,      1.0,   0.0],
        [ 1.0,  1.0,  2.0,  0.0,  2.0,       2.0,     0.0,     -1.0,   0.0],
        [-1.0,  0.0,  2.0, -2.0,  1.0,      -2.0,     0.0,      1.0,   0.0],
        [ 2.0,  0.0,  0.0,  0.0,  1.0,       2.0,     0.0,     -1.0,   0.0],
        [ 1.0,  0.0,  0.0,  0.0,  2.0,      -2.0,     0.0,      1.0,   0.0],
        [ 3.0,  0.0,  0.0,  0.0,  0.0,       2.0,     0.0,      0.0,   0.0],
        [ 0.0,  0.0,  2.0,  1.0,  2.0,       2.0,     0.0,     -1.0,   0.0],
        [-1.0,  0.0,  0.0,  0.0,  2.0,       1.0,     0.0,     -1.0,   0.0],
        [ 1.0,  0.0,  0.0, -4.0,  0.0,      -1.0,     0.0,      0.0,   0.0],
        [-2.0,  0.0,  2.0,  2.0,  2.0,       1.0,     0.0,     -1.0,   0.0],
        [-1.0,  0.0,  2.0,  4.0,  2.0,      -2.0,     0.0,      1.0,   0.0],
        [ 2.0,  0.0,  0.0, -4.0,  0.0,      -1.0,     0.0,      0.0,   0.0],
        [ 1.0,  1.0,  2.0, -2.0,  2.0,       1.0,     0.0,     -1.0,   0.0],
        [ 1.0,  0.0,  2.0,  2.0,  1.0,      -1.0,     0.0,      1.0,   0.0],
        [-2.0,  0.0,  2.0,  4.0,  2.0,      -1.0,     0.0,      1.0,   0.0],
        [-1.0,  0.0,  4.0,  0.0,  2.0,       1.0,     0.0,      0.0,   0.0],
        [ 1.0, -1.0,  0.0, -2.0,  0.0,       1.0,     0.0,      0.0,   0.0],
        [ 2.0,  0.0,  2.0, -2.0,  1.0,       1.0,     0.0,     -1.0,   0.0],
        [ 2.0,  0.0,  2.0,  2.0,  2.0,      -1.0,     0.0,      0.0,   0.0],
        [ 1.0,  0.0,  0.0,  2.0,  1.0,      -1.0,     0.0,      0.0,   0.0],
        [ 0.0,  0.0,  4.0, -2.0,  2.0,       1.0,     0.0,      0.0,   0.0],
        [ 3.0,  0.0,  2.0, -2.0,  2.0,       1.0,     0.0,      0.0,   0.0],
        [ 1.0,  0.0,  2.0, -2.0,  0.0,      -1.0,     0.0,      0.0,   0.0],
        [ 0.0,  1.0,  2.0,  0.0,  1.0,       1.0,     0.0,      0.0,   0.0],
        [-1.0, -1.0,  0.0,  2.0,  1.0,       1.0,     0.0,      0.0,   0.0],
        [ 0.0,  0.0, -2.0,  0.0,  1.0,      -1.0,     0.0,      0.0,   0.0],
        [ 0.0,  0.0,  2.0, -1.0,  2.0,      -1.0,     0.0,      0.0,   0.0],
        [ 0.0,  1.0,  0.0,  2.0,  0.0,      -1.0,     0.0,      0.0,   0.0],
        [ 1.0,  0.0, -2.0, -2.0,  0.0,      -1.0,     0.0,      0.0,   0.0],
        [ 0.0, -1.0,  2.0,  0.0,  1.0,      -1.0,     0.0,      0.0,   0.0],
        [ 1.0,  1.0,  0.0, -2.0,  1.0,      -1.0,     0.0,      0.0,   0.0],
        [ 1.0,  0.0, -2.0,  2.0,  0.0,      -1.0,     0.0,      0.0,   0.0],
        [ 2.0,  0.0,  0.0,  2.0,  0.0,       1.0,     0.0,      0.0,   0.0],
        [ 0.0,  0.0,  2.0,  4.0,  2.0,      -1.0,     0.0,      0.0,   0.0],
        [ 0.0,  1.0,  0.0,  1.0,  0.0,       1.0,     0.0,      0.0,   0.0]
            ))

        t = self.t

        # Calculate fundamental arguments in the FK5 reference frame
        self.fundamental_fk5()

        # create the array of arguments for input into sin/cos
        arg = np.matmul(nut[:,0:5],self.fundamental_args)

        # create the array of sine and cosine coefficients (t should be in
        # millenia instead of centuries)
        v = np.column_stack((np.ones(len(t)),t/10.0)).T
        #v = np.array([1,t/10.0])
        s = np.matmul(nut[:,5:7],v)
        c = np.matmul(nut[:,7:9],v)

        # create the array of sine and cosine terms
        dp = s*np.sin(arg)
        de = c*np.cos(arg)

        # Sum over all the sines and cosines, and convert from 0.1 mas to
        # radians
        dpsi = (dp.sum(axis=0)*1e-4/3600.0)*np.pi/180.0 # nutation in longitude
        deps = (de.sum(axis=0)*1e-4/3600.0)*np.pi/180.0 # nutation in obliquity

        # Add corrections of frame bias, precession-rates and geophysical with
        # respect to IAU1976/1980 [mas --> radians]
        ddp80 = (-55.0655*1e-3/3600.0)*np.pi/180.0
        dde80 = (-6.358*1e-3/3600.0)*np.pi/180.0

        dpsi = dpsi + ddp80
        deps = deps + dde80

        # mean obliquity of the ecliptic based on IAU1980 model [radians]
        eps0 = (84381.448 - 46.8150*t - 0.00059*t**2 + 0.001813*t**3)/3600.0
        eps0 = eps0*np.pi/180.0

        # Go epochwise (may need to optimize in future)
        nutation = {}
        for c,epoch in enumerate(t):

            # Nutation rotation matrix nutation = r3.r2.r1
            r1 = Rotation(eps0[c],1).rot
            r2 = Rotation(-dpsi[c],3).rot
            r3 = Rotation(-(eps0[c]+deps[c]),1).rot
            nutation[epoch] = np.matmul(r3,np.matmul(r2,r1))

        self.nutation = nutation
        self.nutation_model = "iau1980"
        self.dpsi = dpsi
        self.deps = deps
        self.eps0 = eps0


    def gmst_iau1982(self,ut1_utc):

        """
        Calculate Greenwich mean sidereal time (GMST) based on IAU 1982 model

        Keyword arguments:
            ut1_utc [scalar or list/array] : UT1-UTC in seconds corresponding
                                        to the t attribute of the class object
        Updates:
            self.ut1_utc [array]
            self.time_ut1 [array]
            self.gmst [array]
            self.gmst_model [array]

        """
        # Check the given arguments and set the attributes
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

        # Calculate UT1 times
        time_ut1 = []
        time_ut1_0 = []
        seconds = []
        for c,utc in enumerate(self.time_utc):
            gc = gpsCal()
            gc.set_yyyy_MM_dd_hh_mm_ss(utc.year,utc.month,utc.day,utc.hour,
                                                        utc.minute,utc.second)
            # UT1 time
            ut1 = gc.ut1(self.ut1_utc[c])

            # UT1 at 00:00
            ut1_0 = ut1.replace(hour=0,minute=0,second=0,microsecond=0)

            # Number of seconds within the day
            sec = (ut1 - ut1_0).total_seconds()

            time_ut1.append(ut1)
            time_ut1_0.append(ut1_0)
            seconds.append(sec)

        self.time_ut1 = np.atleast_1d(time_ut1)
        time_ut1_0 = np.atleast_1d(time_ut1_0)
        seconds = np.atleast_1d(seconds)

        # JD2000
        jd2000 = datetime.datetime(2000,1,1,12,0,0)

        # Number of Julian centuries since JD2000 (2000/01/01 12:00) to the
        # calculation day at hour zero
        t = np.array([(time-jd2000).total_seconds()/86400.0/36525.0
                            for time in time_ut1_0])

        # GMST at 00:00
        gmst0 = 24110.54841 + 8640184.812866*t + 0.093104*t**2 - 6.2e-6*t**3

        # add the remainder of the day to get GMST in seconds
        gmst = gmst0 + 1.002737909350795*seconds

        # Normalize to the part of the day in seconds
        gmst = np.array([item%86400.0 for item in gmst])

        # convert to radians
        gmst = gmst*np.pi/43200.0

        self.gmst = gmst
        self.gmst_model = "iau1982"


    def gast_iau1994(self,ut1_utc):

        """
        Calculate Greenwich apparent sidereal time (GAST), and its rotation
        matrix based on IAU 1982 model for GMST and IAU1994 model for equation
        of equinoxes

        Keyword arguments:
            ut1_utc [scalar or list/array] : UT1-UTC in seconds corresponding
                                        to the t attribute of the class object
        Updates:
            self.ut1_utc [array]
            self.time_ut1 [array]
            self.fundamental_args [array] : see fundamental_fk5
            self.nutation [dict] : nutation matrix for each UTC time given
            self.nutation_model [str] : nutation model
            self.dpsi [array] : nutation in longitude for each UTC time
            self.deps [array] : nutation in obliquity for each UTC time
            self.eps0 [array] : mean obliquity of the ecliptic for each UTC
                                time
            self.gmst [array]
            self.gmst_model [array]
            self.gast [array]
            self.gast_matrix [dict]
            self.gast_model [array]

        """
        # Check the given arguments and set the attributes
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

        # For the correction to GMST, we need to calculate the equation of
        # equinoxes from the following:
        # om : longitude of the mean ascending node of the lunar orbit on the
        #      ecliptic, measured from the mean equinox of date
        # dpsi : nutation in longitude
        # eps0 : mean obliquity of the ecliptic
        # we get om from fundamental_fk5, and dpsi and eps0 from
        # nutation_iau1980

        self.fundamental_fk5()
        om = np.array(self.fundamental_args[4,:])
        self.nutation_iau1980()
        dpsi = self.dpsi
        eps0 = self.eps0

        # equation of equinoxes based on IAU1994 model
        eqeq94 = (dpsi*np.cos(eps0) + (0.00264*np.sin(om)
                                + 0.000063*np.sin(2.0*om))*np.pi/180.0/3600.0)

        # GMST from 1982 model
        self.gmst_iau1982(self.ut1_utc)
        gmst = self.gmst

        # Add the correction to get GAST
        gast = gmst + eqeq94

        # Create the rotation matrix
        # Go epochwise (may need to optimize in future)
        t = self.t
        gast_matrix = {}
        for c,epoch in enumerate(t):
            r = Rotation(gast[c],3).rot
            gast_matrix[epoch] = r

        self.gast = gast
        self.gast_matrix = gast_matrix
        self.gast_model = "iau1994"


    def polar_motion(self,xp,yp):

        """
        Calculate polar motion matrices

        Keyword arguments:
            xp [scalar or list/array] : polar x motion in radians corresponding
                                        to the t attribute of the class object
            yp [scalar or list/array] : polar y motion in radians corresponding
                                        to the t attribute of the class object

        Updates:
            self.xp [array] : polar x motion for each UTC time
            self.yp [array] : polar y motion for each UTC time
            self.polar [dict] : polar motion matrix for each UTC time given

        """
        # Check the given arguments and set the attributes
        if not isinstance(xp,(list,np.ndarray,numbers.Number)):
            raise TypeError("The given xp needs to be either a number "
                            "or a list/array of numbers")
        if not all(isinstance(item,numbers.Number)
                        for item in np.atleast_1d(xp)):
            raise TypeError("There are non-number items in xp")
        if np.shape(np.atleast_1d(xp)) != np.shape(self.time_utc):
            raise ValueError("Shape mismatch between self.time_utc "
                            f"{np.shape(self.time_utc)} and xp "
                            f"{np.shape(np.atleast_1d(xp))}")
        self.xp = np.atleast_1d(xp)

        if not isinstance(yp,(list,np.ndarray,numbers.Number)):
            raise TypeError("The given yp needs to be either a number "
                            "or a list/array of numbers")
        if not all(isinstance(item,numbers.Number)
                        for item in np.atleast_1d(yp)):
            raise TypeError("There are non-number items in yp")
        if np.shape(np.atleast_1d(yp)) != np.shape(self.time_utc):
            raise ValueError("Shape mismatch between self.time_utc "
                            f"{np.shape(self.time_utc)} and yp "
                            f"{np.shape(np.atleast_1d(yp))}")
        self.yp = np.atleast_1d(yp)

        t = self.t

        # Go epochwise (may need to optimize in future)
        polar = {}
        for c,epoch in enumerate(t):

            # polar motion rotation matrix polar = r2.r1
            r1 = Rotation(-yp[c],1).rot
            r2 = Rotation(-xp[c],2).rot
            polar[epoch] = np.matmul(r2,r1)

        self.polar = polar


    def celestial_to_terrestrial(self,precession_model=None,
                                 nutation_model=None,gast_model=None,
                                 polar_motion=None,ut1_utc=None,
                                 xp=None,yp=None):

        """
        Calculate polar motion matrices

        Keyword arguments:
            precession_model [str] : precession model used
            nutation_model [str] : nutation model used
            gast_model [str] : GAST model used
            polar_motion [bool] : Apply polar motion?
            ut1_utc [scalar or list/array] : UT1-UTC in seconds corresponding
                                        to the t attribute of the class object
            xp [scalar or list/array] : polar x motion in radians corresponding
                                        to the t attribute of the class object
            yp [scalar or list/array] : polar y motion in radians corresponding
                                        to the t attribute of the class object

        Updates:
            self.nutation_precession [dict] : nutation-precession rotation
                                              matrix for each UTC time
            self.c2t_nopolar [dict] : celestial to terrestrial rotation matrix
                                      with no polar motion applied for each
                                      UTC time
            self.c2t [dict] : full celestial to terretrial rotation matrix for
                              each UTC time
            plus all the attributes updated over the calling of models

        """

        # Check the given arguments and set the attributes
        allowed_precession = ['iau1976']
        allowed_nutation = ['iau1980']
        allowed_gast = ['iau1994']

        if (precession_model is not None
                and precession_model not in allowed_precession):
            raise ValueError(f"The given precession model {precession_model} "
                             f"not recognized! allowed precession models: "
                             f"{allowed_precession}")

        if (nutation_model is not None
                and nutation_model not in allowed_nutation):
            raise ValueError(f"The given nutationmodel {nutation_model} not "
                             f"recognized! allowed nutation models: "
                             f"{allowed_nutation}")

        if (gast_model is not None and gast_model not in allowed_gast):
            raise ValueError(f"The given gast model {gast_model} not "
                             f"recognized! allowed gast models: "
                             f"{allowed_gast}")

        if polar_motion is not None:
            if not isinstance(polar_motion,bool):
                raise TypeError("The given polar_motion needs to be of "
                                "boolean type")
        else:
            polar_motion = False

        if (gast_model is not None and ut1_utc is None):
            raise ValueError("ut1_utc must be given to calculate gast")

        if (polar_motion is True and (xp is None or yp is None) ):
            raise ValueError("xp and yp must be given to calculate polar "
                             "motion rotation matrix")

        if ut1_utc is not None:
            if not isinstance(ut1_utc,(list,np.ndarray,numbers.Number)):
                raise TypeError("The given ut1_utc needs to be either a "
                                "number or a list/array of numbers")
            if not all(isinstance(item,numbers.Number)
                            for item in np.atleast_1d(ut1_utc)):
                raise TypeError("There are non-number items in ut1_utc")
            if np.shape(np.atleast_1d(ut1_utc)) != np.shape(self.time_utc):
                raise ValueError("Shape mismatch between self.time_utc "
                                f"{np.shape(self.time_utc)} and ut1_utc "
                                f"{np.shape(np.atleast_1d(ut1_utc))}")
            self.ut1_utc = np.atleast_1d(ut1_utc)

        if xp is not None:
            if not isinstance(xp,(list,np.ndarray,numbers.Number)):
                raise TypeError("The given xp needs to be either a number "
                                "or a list/array of numbers")
            if not all(isinstance(item,numbers.Number)
                            for item in np.atleast_1d(xp)):
                raise TypeError("There are non-number items in xp")
            if np.shape(np.atleast_1d(xp)) != np.shape(self.time_utc):
                raise ValueError("Shape mismatch between self.time_utc "
                                f"{np.shape(self.time_utc)} and xp "
                                f"{np.shape(np.atleast_1d(xp))}")
            self.xp = np.atleast_1d(xp)

        if yp is not None:
            if not isinstance(yp,(list,np.ndarray,numbers.Number)):
                raise TypeError("The given yp needs to be either a number "
                                "or a list/array of numbers")
            if not all(isinstance(item,numbers.Number)
                            for item in np.atleast_1d(yp)):
                raise TypeError("There are non-number items in yp")
            if np.shape(np.atleast_1d(yp)) != np.shape(self.time_utc):
                raise ValueError("Shape mismatch between self.time_utc "
                                f"{np.shape(self.time_utc)} and yp "
                                f"{np.shape(np.atleast_1d(yp))}")
            self.yp = np.atleast_1d(yp)

        # call each model and calculate the rotation matrix
        t = self.t
        if precession_model == 'iau1976':
            self.precession_iau1976()
        elif precession_model is None:
            precession = {}
            for c,epoch in enumerate(t):
                precession[epoch] = np.identity(3)
            self.precession = precession
            self.precession_model = None
        else:
            raise ValueError(f"precession model {precession_model} not "
                              "recognized!")

        if nutation_model == 'iau1980':
            self.nutation_iau1980()
        elif nutation_model is None:
            nutation = {}
            for c,epoch in enumerate(t):
                nutation[epoch] = np.identity(3)
            self.nutation = nutation
            self.nutation_model = None
        else:
            raise ValueError(f"nutation model {nutation_model} not "
                              "recognized!")

        if gast_model == 'iau1994':
            self.gast_iau1994(self.ut1_utc)
        elif gast_model is None:
            gast_matrix = {}
            for c,epoch in enumerate(t):
                gast_matrix[epoch] = np.identity(3)
            self.gast_matrix = gast_matrix
            self.gast_model = None
        else:
            raise ValueError(f"GAST model {gast_model} not recognized!")

        if polar_motion is True:
            self.polar_motion(self.xp,self.yp)
        else:
            polar = {}
            for c,epoch in enumerate(t):
                polar[epoch] = np.identity(3)
            self.polar = polar

        nutation_precession = {}
        c2t_nopolar = {}
        c2t = {}
        for c,epoch in enumerate(t):
            np_matrix = np.matmul(self.nutation[epoch],
                                                self.precession[epoch])
            c2t_nopol = np.matmul(self.gast_matrix[epoch],
                                            np_matrix)
            c2t_matrix = np.matmul(self.polar[epoch],c2t_nopol)

            nutation_precession[epoch] = np_matrix
            c2t_nopolar[epoch] = c2t_nopol
            c2t[epoch] = c2t_matrix

        self.nutation_precession = nutation_precession
        self.c2t_nopolar = c2t_nopolar
        self.c2t = c2t




