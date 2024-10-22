# Rotation matrix of angle theta around the given axis and derivative of it

import numpy as np
import logging
import rocs.checkutils as checkutils


logger = logging.getLogger(__name__)


class Rotation:

    def __init__(self,theta,axis):

        # theta: rotation angle in radians
        # axis: 1, 2 or 3

        # Check the given attributes
        checkutils.check_scalar(theta)

        if axis not in ([1,2,3]):
            logger.error("The rotation axis rotation.axis can only "
                         "be 1, 2 or 3",stack_info=True)
            raise ValueError("The rotation axis rotation.axis can only "
                             "be 1, 2 or 3")

        self.theta = theta
        self.axis  = axis


        # Create rotation matrix along axis
        #
        # R: a 3-by-3 rotation matrix

        #      |1 0            0          |
        # R1 = |0 cos(theta1)  sin(theta1)|
        #      |0 -sin(theta1) cos(theta1)|
        #
        #      |cos(theta2) 0 -sin(theta2)|
        # R2 = |0           1 0           |
        #      |sin(theta2) 0 cos(theta2) |
        #
        #      |cos(theta3)  sin(theta3) 0|
        # R3 = |-sin(theta3) cos(theta3) 0|
        #      |0            0           1|

        # generic matrix
        R = np.array([[np.cos(theta),np.sin(theta),-np.sin(theta)],
                      [-np.sin(theta),np.cos(theta),np.sin(theta)],
                      [np.sin(theta),-np.sin(theta),np.cos(theta)]])

        # Based on the axis, replace spceific rows and columns
        if axis == 1:
            R[:,0] = 0.0
            R[0,:] = 0.0
            R[0,0] = 1.0
        elif axis == 2:
            R[:,1] = 0.0
            R[1,:] = 0.0
            R[1,1] = 1.0
        elif axis == 3:
            R[:,2] = 0.0
            R[2,:] = 0.0
            R[2,2] = 1.0
        else:
            logger.error("axis can only be 1, 2 or 3",stack_info=True)
            raise ValueError("axis can only be 1, 2 or 3")

        self.rot = R


        # Create the derivative of rotation matrix along axis
        #
        # Output:
        # dR: a 3-by-3 matrix

        #              |0  0            0          |
        # dR1dtheta1 = |0 -sin(theta1)  cos(theta1)|
        #              |0 -cos(theta1) -sin(theta1)|
        #
        #              |-sin(theta2) 0 -cos(theta2)|
        # dR2dtheta2 = |0            0  0          |
        #              | cos(theta2) 0 -sin(theta2)|
        #
        #              |-sin(theta3)  cos(theta3) 0|
        # dR3dtheta3 = |-cos(theta3) -sin(theta3) 0|
        #              | 0            0           0|

        theta = self.theta
        axis = self.axis

        # generic matrix
        dR = np.array([[-np.sin(theta),np.cos(theta),-np.cos(theta)],
                       [-np.cos(theta),-np.sin(theta),np.cos(theta)],
                       [np.cos(theta),-np.cos(theta),-np.sin(theta)]])

        # Based on the axis, replace spceific rows and columns
        if axis == 1:
            dR[:,0] = 0.0
            dR[0,:] = 0.0
            dR[0,0] = 0.0
        elif axis == 2:
            dR[:,1] = 0.0
            dR[1,:] = 0.0
            dR[1,1] = 0.0
        elif axis == 3:
            dR[:,2] = 0.0
            dR[2,:] = 0.0
            dR[2,2] = 0.0
        else:
            logger.error("axis can only be 1, 2 or 3",stack_info=True)
            raise ValueError("axis can only be 1, 2 or 3")

        self.drot =dR

