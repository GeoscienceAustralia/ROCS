# Module for checking utilities

import numpy as np
import numbers
import logging


logger = logging.getLogger(__name__)

def check_coords(coords,ncol=3,minrows=3):

    # Check the input coordinate system
    #
    # Input:
    # coords: supposed to be an m by 3 array-like; where m is the number of
    # points,
    #    and the three columns are for x,y and z.
    # ncol: number of column expected (default 3 for x,y,z)
    # minrows: minimum number of rows expected (default 3)

    if (np.ndim(coords) != 2):
        logger.error("The input array must be m by 3 where m is the "
                         "number of points, and the columns relate to "
                         "x,y and z",stack_info=True)
        raise ValueError("The input array must be m by 3 where m is the "
                         "number of points, and the columns relate to "
                         "x,y and z")

    if (np.shape(coords)[1] != ncol):
        logger.error(f"The input array must have exactly {ncol} columns",
                        stack_info=True)
        raise ValueError(f"The input array must have exactly {ncol} columns")

    if (np.shape(coords)[0] < minrows):
        logger.error(f"The input array must contain at least {minrows} rows",
                        stack_info=True)
        raise ValueError(f"The input array must contain at least {minrows} "
                          "rows")


def check_scalar(c):

    # Check the input scalar

    # Input:
    # c: supposed to be a real scalar

    if np.ndim(c) != 0:
        logger.error("The input attribute must be a scalar",stack_info=True)
        raise ValueError("The input attribute must be a scalar")
    if not isinstance(c,numbers.Number):
        logger.error("The input attribute must be a real number",
                        stack_info=True)
        raise TypeError("The input attribute must be a real number")
    if isinstance(c,complex):
        logger.error("The input attribute cannot be a complex number",
                        stack_info=True)
        raise TypeError("The input attribute cannot be a complex number")


def check_array(coords,m,n=None):

    # check the input 1- or 2-d array
    #
    # Input:
    # coords: supposed to be a 1- or 2-d array of real numbers
    # if n is specifed:
    #       m,n: dimensions of coords
    # if n is not specified:
    #       m: length of coords

    # Check the size
    if n is None:
        if np.shape(coords) != (m,):
            logger.error("The input attribute must be an array of length "
                              + str(m),stack_info=True)
            raise ValueError("The input attribute must be an array of length "
                              + str(m))

        # Check the type
        for item in coords:
            if not isinstance(item,numbers.Number):
                logger.error("The input attribute can only contain numbers",
                                stack_info=True)
                raise TypeError("The input attribute can only contain numbers")
            if isinstance(item,complex):
                logger.error("The input attribute cannot contain complex "
                             "numbers",stack_info=True)
                raise TypeError("The input attribute cannot contain complex "
                                "numbers")
    else:
        if np.shape(coords) != (m,n):
            logger.error("The input attribute must be a "
                         + str(m) + " by " + str(n) + " array",stack_info=True)
            raise ValueError("The input attribute must be a "
                              + str(m) + " by " + str(n) + " array")

        # Check the type
        for row in coords:
            for item in row:
                if not isinstance(item,numbers.Number):
                    logger.error("The input attribute can only contain "
                                 "numbers",stack_info=True)
                    raise TypeError("The input attribute can only contain "
                                    "numbers")
                if isinstance(item,complex):
                    logger.error("The input attribute cannot contain complex "
                                 "numbers",stack_info=True)
                    raise TypeError("The input attribute cannot contain "
                                    "complex numbers")


