# Coordinate transformations module

import numpy as np
import datetime
import numbers
import rocs.checkutils as checkutils
from rocs.rotation import Rotation
from rocs.iau import IAU


def diag_block_mat_boolindex(L):
    """
    create a block-diagonal matrix from any number of same-size matrices
    """
    shp = L[0].shape
    mask = np.kron(np.eye(len(L)), np.ones(shp))==1
    out = np.zeros(np.asarray(shp)*len(L),dtype=float)
    out[mask] = np.concatenate(L).ravel()
    return out



class Rectangular:

    """
    Class for a rectangular (Cartesian) reference frame

    """

    def __init__(self,coords,ref_frame,time_utc):

        """
        Initialize Rectangular class

        Keyword arguments:
            coords [list or numpy array] : 3-column array/list where the
                                        columns represent the x,y,z coordinates
                                        in the desired cartesian reference
                                        frame, and the rows are for different
                                        points in the reference frame
            ref_frame [str] : the reference frame for the given coordinates
            time_utc [datetime or list/array of datetimes] : UTC time(s)
                                                             corresponding to
                                                             the coordinates

        Updates:
            self.coords [numpy array]
            self.ref_frame [str]
            self.time_utc [array of datetimes]

        """

        # Check the given arguments and set the attributes
        checkutils.check_coords(coords,ncol=3,minrows=1)
        c_arr = np.array(coords)
        checkutils.check_array(c_arr,np.shape(c_arr)[0],np.shape(c_arr)[1])
        self.coords = c_arr

        if not isinstance(ref_frame,str):
            raise TypeError("The input ref_frame needs to be a string")
        allowed_ref_frames = ['ECI','ECEF']
        if ref_frame not in allowed_ref_frames:
            raise TypeError(f"The given reference frame {ref_frame} not "
                            f"recognized!\nAllowed reference frames: "
                            f"{allowed_ref_frames}")
        self.ref_frame =  ref_frame

        if not isinstance(time_utc,(list,np.ndarray,datetime.datetime)):
            raise TypeError("The given time_utc needs to be either a datetime "
                            "object or a list/array of datetime objects")
        if not all(isinstance(item,datetime.datetime)
                        for item in np.atleast_1d(time_utc)):
            raise TypeError("There are non-datetime items in time_utc")
        if np.shape(np.atleast_1d(time_utc)) == (1,):
            self.time_utc = np.full(
                    len(self.coords),np.atleast_1d(time_utc)[0])
        else:
            if np.shape(time_utc) != (len(self.coords),):
                raise ValueError("The given time_utc must be either a datetime"
                                 " object or a 1d array/list with the same "
                                 "length as the given coords.")
            self.time_utc = np.atleast_1d(time_utc)


    def ToECEF(self,iau_model='IAU2006/2000A',transformation_method='CIO',
                evaluation_method='series',ut1_utc=None,xp=None,yp=None):

        """
        convert the coordinates to Earth-Centered Earth-Fixed (ECEF) system

        Keyword arguments:
            iau_model [str]             : the IAU model for transformation:
                                            IAU76/80/82/94
                                            IAU2000A
                                            IAU2006/2000A
            transformation_method [str] : the transformation method:
                                            CIO
                                            equinox
            evaluation_method [str]     : the evaluation method:
                                            classical_angles
                                            series
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
            self.iau_model [str]
            self.transformation_method [str]
            self.evaluation_method [str]
            self.coords [numpy array]
            self.ref_frame [str]

        """

        # Check the given arguments and set the attributes
        allowed_iau_models = (['IAU76/80/82/94','IAU2000A','IAU2006/2000A'])
        if iau_model not in allowed_iau_models:
            raise ValueError(f"The given IAU model {iau_model} not "
                             f"recognized!\nAllowed IAU models: "
                             f"{allowed_iau_models}")

        # Get the original reference frame
        ref_frame_orig = self.ref_frame

        if self.ref_frame == 'ECI':

            # Initialize an IAU class for calculations
            iau = IAU(self.time_utc)

            if iau_model == 'IAU76/80/82/94':
                transformation_method == 'equinox'
                evaluation_method == 'classical_angles'

                # we need xp,yp and ut1_utc
                if (ut1_utc is None or xp is None or yp is None):
                    raise ValueError("ut1_utc, xp and yp must be given for "
                                     "ECI to ECEF conversion")

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

                # call celestial_to_terrestrial to get the rotation matrix
                iau.celestial_to_terrestrial(precession_model='iau1976',
                        nutation_model='iau1980',gast_model='iau1994',
                        ut1_utc=self.ut1_utc,xp=self.xp,yp=self.yp)

                # Convert the coordinates from ECI to ECEF
                c2t = diag_block_mat_boolindex(tuple(iau.c2t.values()))
                coords_c = np.reshape(self.coords,(3*len(self.coords),))
                coords_t = np.matmul(c2t,coords_c)
                self.coords = np.reshape(coords_t,(len(self.coords),3))
                self.ref_frame = 'ECEF'

        else:
            raise ValueError(f"The conversion from {ref_frame_orig} to ECEF "
                    "is not implemented!")
