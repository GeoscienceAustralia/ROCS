# 7-parameter similarity Helmert transformation

import numpy as np
from scipy import optimize
import warnings
import logging
import time
import rocs.checkutils as checkutils
from rocs.rotation import Rotation

logger = logging.getLogger(__name__)

# Toggle between a version compatible with old software and the new version
old_version = True

class Helmert:


    def __init__(self,helmert=None,sighelmert=None,coords0=None,coords1=None,
                      sigmas0=None,sigmas1=None,satinfo=None,orbflags=None,
                      weighted_center=True,acname=None):

        # Helmert transformation vector:
        #self.helmert = [Tx,Ty,Tz,theta1,theta2,theta3,s]

        if helmert is None:

            # Default helmert = [0,0,0,0,0,0,1]
            self.helmert = np.zeros(7)
            self.helmert[6] = 1.0

        else:

            # Check the given helmert
            checkutils.check_array(helmert,7)

            # After the above check, set Helmert parameters
            self.helmert = np.array(helmert)

        if sighelmert is None:

            # Default sighelmert = [1,1,1,1,1,1,1]
            self.sighelmert = np.ones_like(self.helmert)

        else:

            # Check the given sighelmert
            checkutils.check_array(sighelmert,7)

            # After the above check, set sigma of Helmert parameters
            self.sighelmert = np.array(sighelmert)

        # coords0 and coords1 are the two coordinate systems
        if coords0 is None and coords1 is None: # both coords0 and coords1 None

            # Default both to 3*3 zeros
            self.coords0 = np.zeros((3,3))
            self.coords1 = np.zeros((3,3))

        elif coords1 is None: # coords0 defined but coords1 None

            # Default coords1 to zeros of the same size as coords0
            checkutils.check_coords(coords0)
            self.coords0 = np.array(coords0)
            self.coords1 = np.zeros_like(coords0)

        elif coords0 is None: # coords0 None but coords1 defined

            # Default coords0 to zeros of the same size as coords1
            checkutils.check_coords(coords1)
            self.coords1 = np.array(coords1)
            self.coords0 = np.zeros_like(coords1)

        else: # both coords0 and coords1 defined

            checkutils.check_coords(coords0)
            checkutils.check_coords(coords1)

            # Also, check if coords0 and coords1 have the same dimensions
            if (np.shape(coords0) != np.shape(coords1)):
                logger.error("The two input coordinates must of the same "
                             "size",stack_info=True)
                raise ValueError("The two input coordinates must of the same "
                                 "size")

            self.coords0 = np.array(coords0)
            self.coords1 = np.array(coords1)

        # sigmas0 is an array containing the standard deviations of the coords0
        if sigmas0 is None:

            self.sigmas0 = np.ones_like(self.coords0)

        else:

            # check the size
            checkutils.check_coords(sigmas0)
            if (np.shape(sigmas0) != np.shape(coords0)):
                logger.error("The input coordinate sigmas must be the same"
                             "size as the input coordinates",stack_info=True)
                raise ValueError("The input coordinate sigmas must be the same"
                                 "size as the input coordinates")
            self.sigmas0 = np.array(sigmas0)

        # sigmas1 is an array containing the standard deviations of the coords1
        if sigmas1 is None:

            self.sigmas1 = np.ones_like(self.coords1)

        else:

            # check the size
            checkutils.check_coords(sigmas1)
            if (np.shape(sigmas1) != np.shape(coords1)):
                logger.error("The input coordinate sigmas must be the same"
                             "size as the input coordinates",stack_info=True)
                raise ValueError("The input coordinate sigmas must be the same"
                                 "size as the input coordinates")
            self.sigmas1 = np.array(sigmas1)

        if satinfo is not None:

            # Check the given satinfo
            if np.shape(satinfo) != (np.shape(self.coords0)[0],4):
                logger.error(f"\nThe given satinfo must be a "
                             f"{np.shape(self.coords0)[0]} by 4 array\n"
                             f"Shape of the given satinfo: "
                             f"{np.shape(satinfo)}\n",
                             stack_info=True)
                raise ValueError(f"The input satinfo must be a "
                                 f"{np.shape(self.coords0)[0]} by 4 array")

            for row in satinfo:

                if not isinstance(row[0],str):
                    logger.error(f"\nThe first column of satinfo "
                                 f"(constellation ID) must be strings\n",
                                 stack_info=True)
                    raise TypeError(f"The first column of satinfo "
                                     f"(constellation ID) must be strings")

                if not isinstance(row[1],int):
                    logger.error(f"\nThe second column of satinfo (PRN "
                                 f"number) must be integers\n",
                                 stack_info=True)
                    raise TypeError(f"The second column of satinfo (PRN "
                                     f"number) must be integers")

                if not isinstance(row[2],int) and not np.isnan(row[2]):
                    logger.error(f"\nThe third column of satinfo (SVN number) "
                                 f"must be integers or nans\n",
                                 stack_info=True)
                    raise TypeError(f"The third column of satinfo (SVN number "
                                     f") must be integers or nans")

                if not isinstance(row[3],str):
                    logger.error(f"\nThe fourth column of satinfo "
                                 f"(satellite block) must be strings\n",
                                 stack_info=True)
                    raise TypeError(f"The fourth column of satinfo "
                                     f"(satellite block) must be strings")

            if any(np.isnan(svn) for svn in satinfo[:,2]):
                logger.warning("\nThere are unknown SVN numbers in satinfo.\n")


            # After the above checks, set satinfo
            self.satinfo = np.array(satinfo)

        if orbflags is not None:

            # check the given orbflags
            if np.shape(orbflags) != np.shape(self.coords0):
                logger.error(f"\nThe given orbflags must be the same shape as "
                             f"the given coordinates", stack_info=True)
                raise ValueError(f"The input orbflags must be a "
                                 f"{np.shape(self.coords0)[0]} by 3 array")
            allowed_flags = (['okay','missing_val_other','excluded_sat_other',
                              'missing_sys_other','missing_blk_other','missing_sat_other',
                              'missing_val','excluded_sat','missing_sys','missing_blk',
                              'missing_sat','excluded_sat_all','unweighted_sys','unweighted_sat'])
            for row in orbflags:
                for flag in row:
                    if flag not in allowed_flags:
                        logger.error("\nFlags in the orbflags can only be "
                                     f"one of the following:\n "
                                     f"{allowed_flags} \n", stack_info=True)
                        raise ValueError(f"Flag {flag} not recognized!")
        else:

            # If not specified, set a default orbflags of 'okay' for all data
            orbflags = np.full_like(coords0,'okay',dtype=object)

        # After the above checks, set orbflags attribute
        self.orbflags = orbflags

        # Check the given weighted_center flag
        if not isinstance(weighted_center,bool):
            logger.error("\nThe given weighted_center must be a True/False "
                         "boolean flag\n", stack_info=True)
            raise TypeError("weighted_center is not of type bool!")

        # After the above check, set weighted_center attribute
        self.weighted_center = weighted_center

        # filtered versions for the calculations
        self.coords0_flt = self.coords0
        self.coords1_flt = self.coords1
        self.sigmas0_flt = self.sigmas0
        self.sigmas1_flt = self.sigmas1
        if hasattr(self,'satinfo'):
            self.satinfo_flt = self.satinfo
        self.acname = acname


    def printInfo(self):

        # print out the Helmert parameters

        logger.info("")
        logger.info("Helmert transformation parameters: ")
        logger.info(f"Tx    : {self.helmert[0]} +- {self.sighelmert[0]}")
        logger.info(f"Ty    : {self.helmert[1]} +- {self.sighelmert[1]}")
        logger.info(f"Tz    : {self.helmert[2]} +- {self.sighelmert[2]}")
        logger.info(f"theta1: {self.helmert[3]} +- {self.sighelmert[3]}")
        logger.info(f"theta2: {self.helmert[4]} +- {self.sighelmert[4]}")
        logger.info(f"theta3: {self.helmert[5]} +- {self.sighelmert[5]}")
        logger.info(f"scale : {self.helmert[6]} +- {self.sighelmert[6]}")
        logger.info("")
        logger.info("First coordinate system coords0: ")
        logger.info(self.coords0)
        logger.info(np.shape(self.coords0))
        logger.info("")
        logger.info("Second coordinate system coords1: ")
        logger.info(self.coords1)
        logger.info(np.shape(self.coords1))
        logger.info("")
        logger.info("sigmas0: ")
        logger.info(self.sigmas0)
        logger.info("")
        logger.info("sigmas1: ")
        logger.info(self.sigmas1)
        logger.info("")


    def transform(self):

        # Forward transform the coordinate system coords0 to coords1

        # Parameters of the transformation
        Tx     = self.helmert[0]
        Ty     = self.helmert[1]
        Tz     = self.helmert[2]
        theta1 = self.helmert[3]
        theta2 = self.helmert[4]
        theta3 = self.helmert[5]
        scale  = self.helmert[6]

        # Rotation matrix R = R3.R2.R1
        R1 = Rotation(theta1,1).rot
        R2 = Rotation(theta2,2).rot
        R3 = Rotation(theta3,3).rot
        R = np.matmul(R3,np.matmul(R2,R1))

        # Translation vector
        T = np.array([Tx,Ty,Tz])

        # Transform the coordinates: X1 = s.R.(X0+T)
        self.coords1 = np.transpose(
                scale*np.matmul(R,np.transpose(self.coords0+T)))

        # Error propagation to determine the sigmas on coords1
        # sigmas1^2 = s^2.R^2.sigmas0^2
        ST = np.sqrt(np.transpose(
            scale**2*np.matmul(np.power(R,2),np.transpose(self.sigmas0))))


    def Jacobian(self):

        # Create the Jacobian matrix containing derivatives of
        # X1 = scale*R*(X0+T) with respect to the Helmert parameters
        # Another implementation would be X1 = scale*R*X0 + T
        # However, the first implementation, which is chosen here,
        # avoids having ones in Jacobian matrix which further avoids
        # getting zeros in the double-exponential function derivative (which
        # has sign function in it; sum of a set of signs could yield zero).
        # Also, create the weight matrix

        # Parameters of the transformation
        Tx     = self.helmert[0]
        Ty     = self.helmert[1]
        Tz     = self.helmert[2]
        theta1 = self.helmert[3]
        theta2 = self.helmert[4]
        theta3 = self.helmert[5]
        scale  = self.helmert[6]

        T = np.array([Tx,Ty,Tz])

        # Rotation matrix R = R3.R2.R1
        R1 = Rotation(theta1,1).rot
        R2 = Rotation(theta2,2).rot
        R3 = Rotation(theta3,3).rot
        R = np.matmul(R3,np.matmul(R2,R1))

        # derivative of rotation matrix with respect to the three rotations
        dR1dtheta1 = Rotation(theta1,1).drot
        dR2dtheta2 = Rotation(theta2,2).drot
        dR3dtheta3 = Rotation(theta3,3).drot

        # dRdtheta1 = R3.R2.dR1dtheta1
        # dRdtheta2 = R3.dR2dtheta2.R1
        # dRdtheta3 = R3.R2.dR1dtheta1
        dRdtheta1 = np.matmul(R3,np.matmul(R2,dR1dtheta1))
        dRdtheta2 = np.matmul(R3,np.matmul(dR2dtheta2,R1))
        dRdtheta3 = np.matmul(dR3dtheta3,np.matmul(R2,R1))

        # get the number of points m
        m = np.shape(self.coords0_flt)[0]

        # Initialize the Jacobian matrix A
        A = np.zeros((3*m,7))

        # Initialize the weight matrix W
        # To preserve memory, only store the diagonal elements. Non-diagonal
        # elements are zero, thus not needed to be stored
        W = np.zeros((3*m))

        # Derivative wrt translations
        # dX1dT = scale*R
        dX1dT = scale*R
        dX1dT = np.tile(dX1dT,[m,1])

        # Derivative wrt rotations
        dX1dtheta1 = np.array(np.transpose(scale*np.matmul(
                dRdtheta1,np.transpose(self.coords0_flt+T))).flatten())
        dX1dtheta2 = np.array(np.transpose(scale*np.matmul(
                dRdtheta2,np.transpose(self.coords0_flt+T))).flatten())
        dX1dtheta3 = np.array(np.transpose(scale*np.matmul(
                dRdtheta3,np.transpose(self.coords0_flt+T))).flatten())

        # Derivative wrt scale
        dX1dscale = np.array(np.transpose(np.matmul(
            R,np.transpose(self.coords0_flt+T))).flatten())

        # Jacobian matrix A
        A = np.transpose(np.vstack(
                (dX1dT[:,0],dX1dT[:,1],dX1dT[:,2],dX1dtheta1,
                    dX1dtheta2,dX1dtheta3,dX1dscale)))

        # Weight matrix W
        W = (1.0/(self.sigmas1_flt**2)).flatten()

        self.A = A
        self.W = W


    def l2norm(self,dx_threshold=1e-8,maxiter=1):

        # Use L2 norm (least-squares) to estimate Helmert parameters
        # between the two coordinate systems coords0 and coords1. The current
        # self.helmert parameters will be used as initial values,
        # iterations of least-squares will be performed, and the
        # estimated corrections will be applied; so, self.helmert will
        # be updated. Also in the end of this function, self.minim_funcs()
        # is called so the rms, as well as abdev and the robust functions
        # for l1 norm minimization are calculated/updated.

        # Check the given attributes
        checkutils.check_scalar(dx_threshold)
        checkutils.check_scalar(maxiter)
        if not isinstance(maxiter,int):
            logger.error("The given value for maxiter must be an integer",
                            stack_info=True)
            raise TypeError("The given value for maxiter must be an integer")

        # Find 'okay' data; I think we can change this to all 'okay',
        # 'missing_val_other' and 'missing_sat_other' for all centers,
        # weighted or unweighted
        okay_rows = np.where(
                        (self.orbflags=='okay').all(axis=1) &
                        (~np.isnan(self.coords0)).all(axis=1) &
                        (~np.isnan(self.coords1)).all(axis=1))[0]
        if self.weighted_center is False:
            okay_rows = np.where(
                        (self.orbflags!='missing_val').all(axis=1) &
                        (self.orbflags!='missing_sys').all(axis=1) &
                        (self.orbflags!='missing_blk').all(axis=1) &
                        (self.orbflags!='missing_sat').all(axis=1) &
                        (self.orbflags!='excluded_sat_all').all(axis=1) &
                        (~np.isnan(self.coords0)).all(axis=1) &
                        (~np.isnan(self.coords1)).all(axis=1))[0]

        # Only perform l2norm if there is at least 1 'okay' data
        if len(okay_rows > 0):

            # Exclude if sat/epoch data is missing from any weighted center
            # (only 'okay' data is used)
            self.coords0_flt = self.coords0[okay_rows,:]
            self.coords1_flt = self.coords1[okay_rows,:]
            self.sigmas0_flt = self.sigmas0[okay_rows,:]
            self.sigmas1_flt = self.sigmas1[okay_rows,:]
            if hasattr(self,'satinfo'):
                self.satinfo_flt = self.satinfo[okay_rows,:]

            iter = 0
            deltaXcap = 9999.0*np.ones(len(self.helmert))

            # Reiterate until the deltaXcap threshold reaches or maximum
            # number of iterations have been performed
            while ( any(item >= dx_threshold for item in deltaXcap) and
                    iter < maxiter):

                # Determine the Jacobian matrix A
                self.Jacobian()

                # get the number of points m
                m = np.shape(self.coords0_flt)[0]

                # Create the observations vector l, which is the flattened
                # coords1_flt array
                l = self.coords1_flt.flatten()

                # Create the computational observations vector (lc = A*params)
                lc = np.matmul(self.A,self.helmert)

                # Calculate the normal matrix N = A'WA
                N = np.matmul(np.multiply(np.transpose(self.A),self.W),self.A)

                # Calculate C = A'W(l-lc)
                C = np.matmul(np.multiply(np.transpose(self.A),self.W),l-lc)

                # Determine the estimated adjustments to the parameters
                # deltaXcap = inv(N).C = inv(A'WA).A'W(l-lc)
                deltaXcap = np.matmul(np.linalg.inv(N),C)

                # Update self.helmert
                self.helmert = self.helmert + deltaXcap

                # Calculate the residuals
                vcap = l-np.matmul(self.A,self.helmert)
                self.residuals = vcap.reshape(self.coords1_flt.shape)

                # Calculate rms = sqrt(vcap'vcap/df) where df = No. of
                # observations minus No. of parameters
                rms = np.sqrt(np.matmul(np.transpose(vcap),vcap)
                                /(len(vcap)-len(self.helmert)))

                # Sigma of the Helmert parameters sigXcap = rms.sqrt(inv(N))
                # We are only interested in the diagonal elements for sigmas of
                # the estimated Helmert parameters.
                sigXcap = np.zeros(len(self.helmert))
                for i in range(len(sigXcap)):
                    sigXcap[i] = rms * np.sqrt(np.linalg.inv(N)[i,i])
                self.sighelmert = sigXcap

                iter += 1
                self.vcap = vcap

            logger.debug(f"Number of iterations for L2 norm solution: {iter}")
            logger.debug(f"L2 norm solution:\nhelmert: {self.helmert}")

        else:
            logger.warning("There are no 'okay' data; skipping l2 norm")


    def minim_funcs(self):

        # Calculate functions for the minimization problem. This includes l2
        # norm minimization (rms) as well as l1 norm (robust) functions.
        #
        # For the l1 norm, define the set of robust maximum likelihood
        # functions (M-estimates) to be solved.
        #
        # The (local) M-estimate to be minimized is in general:
        #
        # N-1                    N-1
        # ---                    ---
        # \       yi - y(xi|a)   \                  yi - y(xi|a)
        #  | rho(-------------) = |  rho(z)  , z = -------------
        # /          sigi        /                     sigi
        # ---                    ---
        # i=0                    i=0
        #
        # The above function needs to be minimized over a (i.e. the set of
        # k elements of vector a that minimize the above function)
        # rho is the negative logarithm of the probability density of
        # the distribution
        #
        # if we define the derivative of rho(z) as psi(z):
        #
        # psi(z) = drho(z)/dz
        #
        # the above minimization can be written as the following set of
        # M equations:
        #
        #     N-1
        #     ---
        #     \   1        yi - y(xi)     dy(xi|a)
        # 0 =  | ----*psi(------------)*(----------) , k = 0,...,M-1
        #     /  sigi        sigi            dak
        #     ---
        #     i=0
        #
        # Therefore, the minimization problem turns into a root finding
        # problem.
        #
        # The psi function acts as a weighting function for the individual
        # data points.
        # Depending on the assumed distribution of errors, we can use
        # different psi functions (normal distribution will be l2 norm:
        # more deviated points get more weights:
        # rho(z) = (1/2)z^2 ; psi(z) = z)
        #
        # Here, we use a double (two-sided) exponential function:
        # rho(z) = |z| ; psi(z) = sign(z)
        # all points get the same weights
        #
        #
        # So our choice of robust functions are:
        #
        #     N-1
        #     ---
        #     \   1         yi - y(xi)     dy(xi|a)
        # F =  | ----*sign(------------)*(----------) , k = 0,...,6
        #     /  sigi         sigi            dak
        #     ---
        #     i=0
        #
        # The minimization algorithm is based on:
        # Press WH, Teukolsky SA, Vetterling WT, Flannery BP (2007) Numerical
        # recipes 3rd edition: the art of scientific computing. Cambridge
        # University Press, Cambridge
        # The same reference contains other psi functions that could be used
        #
        # We also calculate some other minimization functions.
        #
        # The list of functions calculated:
        # Fexp   : robust function to be rooted for maximum likelihood
        #          estimate using a 2-sided exponential distribution of errors
        # abdev  : absolute deviation
        # rms    : root mean square error

        # Find 'okay' rows
        okay_rows = np.where(
                  (self.orbflags != 'excluded_sat_other').all(axis=1) &
                  (self.orbflags != 'missing_sys_other').all(axis=1) &
                  (self.orbflags != 'missing_blk_other').all(axis=1) &
                  (self.orbflags != 'missing_sat_other').all(axis=1) &
                  (self.orbflags != 'unweighted_sat').all(axis=1) &
                  (self.orbflags != 'excluded_sat').all(axis=1) &
                  (~np.isnan(self.coords0)).all(axis=1) &
                  (~np.isnan(self.coords1)).all(axis=1))[0]
        logger.debug(f"orbflags {self.orbflags}")
        logger.debug(f"okay_rows {okay_rows} {len(okay_rows)}")
        logger.debug(f"coords0 {np.shape(self.coords0)}")
        self.coords0_flt = self.coords0[okay_rows,:]
        self.coords1_flt = self.coords1[okay_rows,:]
        self.sigmas0_flt = self.sigmas0[okay_rows,:]
        self.sigmas1_flt = self.sigmas1[okay_rows,:]
        logger.debug(f"coords0 {self.coords0} {np.shape(self.coords0)}")
        logger.debug(f"coords0_flt {self.coords0_flt} "
                     f"{np.shape(self.coords0_flt)}")

        # 'fine' rows are more relaxed than 'okay' rows
        fine_rows = np.where(
                (self.orbflags!='missing_val').all(axis=1) &
                (self.orbflags!='missing_sys').all(axis=1) &
                (self.orbflags!='missing_blk').all(axis=1) &
                (self.orbflags!='missing_sat').all(axis=1) &
                (self.orbflags!='excluded_sat_all').all(axis=1) &
                (~np.isnan(self.coords0)).all(axis=1) &
                (~np.isnan(self.coords1)).all(axis=1))[0]

        coords0_fine = self.coords0[fine_rows,:]
        coords1_fine = self.coords1[fine_rows,:]
        sigmas0_fine = self.sigmas0[fine_rows,:]
        sigmas1_fine = self.sigmas1[fine_rows,:]
        orbflags_fine = self.orbflags[fine_rows,:]
        if hasattr(self,'satinfo'):
            satinfo_fine = self.satinfo[fine_rows,:]
        logger.debug(f"coords1_fine {coords1_fine} "
                     f"{np.shape(coords1_fine)} "
                     f"{type(coords1_fine)}")

        # Determine the modelled data (computed observations vector)
        # We do this by creating another instance of the helmert class
        helmert_modeled = Helmert(helmert=self.helmert,
                                  coords0=coords0_fine,
                                  sigmas0=sigmas0_fine)
        t0 = time.process_time()
        helmert_modeled.transform()
        t1 = time.process_time() - t0
        logger.debug(f"time transform: {t1}")

        # get the number of points m
        m_okay = np.shape(self.coords0_flt)[0]
        m_fine = np.shape(coords0_fine)[0]
        m_all = np.shape(self.coords0)[0]
        logger.debug(f"m_all={m_all}, m_fine={m_fine}, m_okay={m_okay}")
        logger.debug(f"helmert_modeled.coords1: {helmert_modeled.coords1}")

        # Initialize sign vector and 1/sig vector (both 1 by 3*m)
        sign = np.zeros((1,3*m_okay))
        oneover_sig = np.zeros((1,3*m_okay))

        # Initialize some minimization functions
        abdev = 0.0
        rms = 0.0
        c_okay = 0

        # if satinfo exists, also initialize satellite-specific,
        # block-specific and constellation-specific minimization functions
        if hasattr(self,'satinfo'):

            # Satellite specific minimization functions
            sat_abdev = {}
            sat_rms = {}

            # Block specific minimization functions
            blk_abdev = {}
            blk_rms = {}

            # Constellation specific minimization functions
            sys_abdev = {}
            sys_rms = {}

        # Calculate the absolute and square deviations between the modelled
        # and observed data (matrix way)
        # Exclude if sat/epoch data is missing/excluded from any weighted
        # center; i.e. only calculate center abdev and rms if all the
        # weighted centers have data
        D = abs(coords1_fine - helmert_modeled.coords1)
        okay_rows_D = np.where(
                 (orbflags_fine != 'excluded_sat_other').all(axis=1) &
                 (orbflags_fine != 'missing_sys_other').all(axis=1) &
                 (orbflags_fine != 'missing_blk_other').all(axis=1) &
                 (orbflags_fine != 'missing_sat_other').all(axis=1) &
                 (orbflags_fine != 'unweighted_sat').all(axis=1) &
                 (orbflags_fine != 'excluded_sat').all(axis=1))[0]
        D_okay = D[okay_rows_D]
        logger.debug(f"D_okay: {D_okay}")
        if hasattr(self,'satinfo'):
            satinfo_okay = satinfo_fine[okay_rows_D]
        logger.debug(f"len(D_okay): {len(D_okay)}")
        self.abdev = np.nansum(D_okay)/(3*(len(D_okay))-len(self.helmert))
        self.rms = np.sqrt(np.nansum(D_okay**2)/(3*m_okay-len(self.helmert)))
        sign = (np.sign((coords1_fine - helmert_modeled.coords1)/sigmas1_fine))
        sign_okay = sign[okay_rows_D].flatten()
        oneover_sig = 1.0/sigmas1_fine
        oneover_sig_okay = oneover_sig[okay_rows_D].flatten()
        logger.debug(f"oneover_sig_okay: {oneover_sig_okay}")

        wht_rows_D = np.where(
                 (orbflags_fine != 'excluded_sat_other').all(axis=1) &
                 (orbflags_fine != 'missing_sys_other').all(axis=1) &
                 (orbflags_fine != 'missing_blk_other').all(axis=1) &
                 (orbflags_fine != 'missing_sat_other').all(axis=1) &
                 (orbflags_fine != 'unweighted_sat').all(axis=1) &
                 (orbflags_fine != 'unweighted_sys').all(axis=1) &
                 (orbflags_fine != 'excluded_sat').all(axis=1))[0]
        D_wht = D[wht_rows_D]
        self.abdev_wht = np.nansum(D_wht)/(3*(len(D_wht))-len(self.helmert))

        if hasattr(self,'satinfo'):

            sats = np.unique(satinfo_fine.astype("<U15"),axis=0)
            sats = np.array(sats,dtype='object')
            sats[:,[1,2]] = sats[:,[1,2]].astype(int)
            sats = sats[np.lexsort((sats[:, 0], sats[:, 1]))]
            for sat in sats:
                sat_rows = np.where( (satinfo_fine==sat).all(axis=1))[0]
                if len(sat_rows) > 0:
                    Dsat = D[sat_rows]
                    sat_abdev[sat[0],sat[1],sat[2]] = (
                            np.nansum(Dsat)/(3*len(sat_rows)-len(self.helmert)))
                    sat_rms[sat[0],sat[1],sat[2]] = np.sqrt(
                            np.nansum(Dsat**2)/(3*len(sat_rows)-len(self.helmert)))
                    logger.debug(f"coords1_fine: {coords1_fine[sat_rows]}")
                    logger.debug(f"helmert_modeled.coords1: {helmert_modeled.coords1[sat_rows]}")
                    logger.debug(f"Dsat: {Dsat}")
                    logger.debug(f"TSTHEL: {self.acname} {sat[1]} {3*len(sat_rows)} "
                                 f"{np.nansum(Dsat**2)} {sat_rms[sat[0],sat[1],sat[2]]*100.0}")

            blocks = np.unique(sats[:,3])
            blocks.sort()
            for blk in blocks:
                blk_rows = np.where((satinfo_fine[:,3]==blk) &
                        (orbflags_fine != 'excluded_sat_other').all(axis=1) &
                        (orbflags_fine != 'missing_sat_other').all(axis=1) &
                        (orbflags_fine != 'unweighted_sat').all(axis=1) &
                        (orbflags_fine != 'excluded_sat').all(axis=1))[0]
                if len(blk_rows) > 0:
                    Dblk = D[blk_rows]
                    blk_abdev[blk] = (
                            np.nansum(Dblk)/(3*len(blk_rows)-len(self.helmert)))
                    blk_rms[blk] = np.sqrt(
                            np.nansum(Dblk**2)/(3*len(blk_rows)-len(self.helmert)))

            systems = np.unique(sats[:,0])
            systems.sort()
            for sys_id in systems:
                sys_rows = np.where((satinfo_fine[:,0]==sys_id) &
                        (orbflags_fine != 'excluded_sat_other').all(axis=1) &
                        (orbflags_fine != 'missing_blk_other').all(axis=1) &
                        (orbflags_fine != 'missing_sat_other').all(axis=1) &
                        (orbflags_fine != 'unweighted_sat').all(axis=1) &
                        (orbflags_fine != 'excluded_sat').all(axis=1))[0]
                if len(sys_rows) > 0:
                    Dsys = D[sys_rows]
                    sys_abdev[sys_id] = (
                            np.nansum(Dsys)/(3*len(sys_rows)-len(self.helmert)))
                    sys_rms[sys_id] = np.sqrt(
                            np.nansum(Dsys**2)/(3*len(sys_rows)-len(self.helmert)))

            self.sat_abdev = sat_abdev
            self.sat_rms = sat_rms
            self.blk_abdev = blk_abdev
            self.blk_rms = blk_rms
            self.sys_abdev = sys_abdev
            self.sys_rms = sys_rms

        # Consider sign of zero to be 1
        sign_okay[sign_okay == 0] = 1.0

        # Create the Jacobian matrix
        self.Jacobian()

        # Now form the robust function by performing the summation through
        # multiplying the sign and Jacobian matrices:
        # Fexp = (1/sigma).sign * A
        self.Fexp = (np.matmul(oneover_sig_okay*sign_okay,self.A).
                                reshape(len(self.helmert)))

        if any(item ==0.0 for item in self.Fexp):
            logger.debug(f"There are zeros in Fexp\n"
                         f"self.Fexp: {self.Fexp}\n"
                         f"oneover_sig_okay: {oneover_sig_okay}\n"
                         f"sign_okay: {sign_okay}\n"
                         f"self.A column 2:\n{self.A[:,2]}")


    def bracket(self,interval=None,sigscale=3,maxiter=100):

        # Performs the bracketing: starting from the current state of
        # self.helmert as the opening bracket, tries to find two sets of
        # Helmert parameters so the robust functions Fexp have a root between
        # the two sets of Helmert parameters

        # Inputs:
        # interval: the vector defining the interval for moving the brackets
        # sigscale: the scale given to Helmert sigmas to determine interval
        # Note:
        #       One of the interval or sigscale is required
        #       sigscale is ignored if interval is given
        #
        # maxiter: maximum number of iterations for bracketing

        # Check the input arguments

        # interval/sigscale
        if interval is None:
            if hasattr(self,'sighelmert'):
                checkutils.check_scalar(sigscale)
                interval = sigscale*self.sighelmert
            else:
                logger.error("Input argument interval is not given, and there"
                            " is no sighelmert attribute! Try calling "
                            " helmert.l2norm() first or specifying interval!",
                            stack_info=True)
                raise TypeError("bracket() missing argument: 'interval'")
        else:
            checkutils.check_array(interval,7)

        logger.debug(f"interval: {interval}")

        # maxiter
        checkutils.check_scalar(maxiter)
        if not isinstance(maxiter,int):
            logger.error("The given value for maxiter must be an integer",
                            stack_info=True)
            raise TypeError("The given value for maxiter must be an integer")


        # Create two instances of helmert class for the bracketing
        logger.debug(f"coords0 bracket {self.coords0} {np.shape(self.coords0)}")
        logger.debug(f"orbflags bracket {self.orbflags} {np.shape(self.orbflags)}")

        # The first is the same as the current one
        helmert1 = Helmert(helmert=self.helmert,sighelmert=self.sighelmert,
                           coords0=self.coords0,sigmas0=self.sigmas0,
                           coords1=self.coords1,sigmas1=self.sigmas1,
                           orbflags=self.orbflags,
                           weighted_center=self.weighted_center)
        helmert1.minim_funcs()
        helm1 = helmert1.helmert
        sighelm1 = helmert1.sighelmert
        F1 = helmert1.Fexp

        # The second instance of the helmert class as prescribed below
        helm2 = helm1 + interval*np.sign(F1)
        helmert2 = Helmert(helmert=helm2,sighelmert=sighelm1,
                           coords0=self.coords0,sigmas0=self.sigmas0,
                           coords1=self.coords1,sigmas1=self.sigmas1,
                           orbflags=self.orbflags,
                           weighted_center=self.weighted_center)
        helmert2.minim_funcs()
        helm2 = helmert2.helmert
        sighelm2 = helmert2.sighelmert
        F2 = helmert2.Fexp

        logger.debug("Initial brackets from L2 norm:")
        logger.debug(f"helm1 {helm1}")
        logger.debug(f"helm2 {helm2}")
        logger.debug(f"rms1 F1 {helmert1.rms} {F1}")
        logger.debug(f"rms2 F2 {helmert2.rms} {F2}")

        # Find the brackets that have different signs, so we
        # are sure that there is a root between them
        c = 0
        while (any(item > 0 for item in F1*F2) and c < maxiter):
            for i in range(0,len(self.helmert)):
                if (F1[i]*F2[i] > 0):
                    helmnew = 2.0*helm2[i]-helm1[i]
                    helm1[i] = helm2[i]
                    helm2[i] =  helmnew
                    if old_version is True:
                        F1[i] = F2[i] # the old way where correlations
                                      # are not considered. we now recalculate
                                      # both F1 and F2

            helmert1 = Helmert(helmert=helm1,sighelmert=sighelm1,
                               coords0=self.coords0,sigmas0=self.sigmas0,
                               coords1=self.coords1,sigmas1=self.sigmas1,
                               orbflags=self.orbflags,
                               weighted_center=self.weighted_center)

            helmert2 = Helmert(helmert=helm2,sighelmert=sighelm2,
                               coords0=self.coords0,sigmas0=self.sigmas0,
                               coords1=self.coords1,sigmas1=self.sigmas1,
                               orbflags=self.orbflags,
                               weighted_center=self.weighted_center)

            # call minim_funcs
            helmert1.minim_funcs()
            helmert2.minim_funcs()
            if old_version is False:
                F1 = helmert1.Fexp
            F2 = helmert2.Fexp
            logger.debug(f"brackets after iteration {c+1}")
            logger.debug(f"helm1 {helm1}")
            logger.debug(f"helm2 {helm2}")
            logger.debug(f"rms1 F1 {(3*len(self.coords0)-7)*helmert1.rms**2} "
                        f"{F1}")
            logger.debug(f"rms2 F2 {(3*len(self.coords0)-7)*helmert2.rms**2} "
                        f"{F2}")
            c += 1

        logger.debug(f'No. of iterations for bracketing: {c}\n')
        if (c == maxiter):
            logger.warning(f"Number of iterations in bracketing for L1 norm "
                           f"solution reached maxiter ({maxiter}). Use the "
                           f"results with caution!")

        self.helmert1 = helm1
        self.helmert2 = helm2
        self.F1 = F1
        self.F2 = F2


    def bisection(self,helm1=None,helm2=None,precision_level=None,
                    sigscale=0.1,precision_limits=[1e-15,1e-13],maxiter=100):

        # Performs the bisection method using the bracketed Helmert parameters
        #
        # Inputs:
        #
        # helm1 and helm2: the initial brackets for Helmert parameters
        # precision_level: vector difining the target precision levels for
        #                  Helmert parameters
        # sigscale: the scale given to Helmert sigmas to determine precision
        #           level
        # Note:
        #       One of the precision_level or sigscale is required
        #       sigscale is ignored if precision_level is given
        #
        # precision_limits: array-like of lenght 2 giving the allowed \
        #                   minimum and maximum limits of precision levels for
        #                   all the parameters: [min_prec,max_prec]
        # maxiter: maximum number of iterations for bracketing


        # Check the input arguments

        # helm1 and helm2
        if helm1 is None:
            if hasattr(self,'helmert1'):
                helm1 = self.helmert1
            else:
                logger.error("The first bracket is not given, and there is no"
                    " helm1 attribute! Try calling helmert.bracket() first!",
                        stack_info=True)
                raise TypeError("bisection() missing argument: 'helm1'")
        else:
            checkutils.check_array(helm1,7)

        if helm2 is None:
            if hasattr(self,'helmert2'):
                helm2 = self.helmert2
            else:
                logger.error("The second bracket is not given, and there is no"
                    " helm2 attribute! Try calling helmert.bracket() first!",
                        stack_info=True)
                raise TypeError("bisection() missing argument: 'helm2'")
        else:
            checkutils.check_array(helm2,7)

        # precision_level/sigscale
        if precision_level is None:
            if hasattr(self,'sighelmert'):
                checkutils.check_scalar(sigscale)
                precision_level = sigscale*self.sighelmert
            else:
                logger.error("Input argument precision_level is not given, "
                             "and there is no sighelmert attribute! Try "
                             "calling helmert.l2norm() first or specifying "
                             "precision_level!", stack_info=True)
                raise TypeError("bisection() missing argument: "
                                "'precision_level'")
        else:
            checkutils.check_array(precision_level,7)

        logger.debug(f"precision_level before checking limits:\n "
                     f"{precision_level}")

        # precision_limits:
        checkutils.check_array(precision_limits,2)

        # Change precision_level based on precision_limits if required
        for i,item in enumerate(precision_level):
            if i in [0,1,2]: # translations
                if item < 1e-7:
                    logger.warning(f"The precision_level for Helmert parameter"
                                f" {i} ({precision_level[i]}) too"
                                f" small; set to 1e-07 to"
                                f" avoid too many iterations")
                    precision_level[i] = 1e-7
                if item > 1e-5:
                    logger.warning(f"The precision_level for Helmert parameter"
                                   f" {i} ({precision_level[i]}) too"
                                   f" large; set to 1e-5 to"
                                   f" converge to mm level")
                    precision_level[i] = 1e-5
            else: # rotations and scale
                if item < precision_limits[0]:
                    logger.warning(f"The precision_level for Helmert parameter"
                                f" {i} ({precision_level[i]}) too"
                                f" small; set to {precision_limits[0]} to"
                                f" avoid too many iterations")
                    precision_level[i] = precision_limits[0]
                if item > precision_limits[1]:
                    logger.warning(f"The precision_level for Helmert parameter"
                                   f" {i} ({precision_level[i]}) too"
                                   f" large; set to {precision_limits[0]} to"
                                   f" converge to mm level")
                    precision_level[i] = precision_limits[1]

        logger.debug(f"precision levels for Helmert parameters: "
                    f"{precision_level}")

        # maxiter
        checkutils.check_scalar(maxiter)
        if not isinstance(maxiter,int):
            logger.error("The given value for maxiter must be an integer",
                            stack_info=True)
            raise TypeError("The given value for maxiter must be an integer")


        if old_version is True:
            if hasattr(self,'F1'):
                F1 = self.F1
            else:
                logger.error("The old_version flag is True but there is no F1 "
                        "attribute! Try calling helmert.bracket() first!",
                        stack_info=True)
                raise AttributeError("object has no attribute 'F1'")
            if hasattr(self,'F2'):
                F2 = self.F2
            else:
                logger.error("The old_version flag is True but there is no F2 "
                        "attribute! Try calling helmert.bracket() first!",
                        stack_info=True)
                raise AttributeError("object has no attribute 'F2'")

        # Perform the bisection while the difference between two brackets is
        # larger than the precision levels specified for all the Helmert 
        # parameters (or the number of iterations reaches the maximum number
        # specified)

        # Initialize helm_mid (essential for the case of coords0 == coords1)
        helm_mid = (helm1+helm2)/2.0
        helmert_mid = Helmert(helmert=helm_mid,sighelmert=self.sighelmert,
                                coords0=self.coords0,sigmas0=self.sigmas0,
                                coords1=self.coords1,sigmas1=self.sigmas1,
                                orbflags=self.orbflags,
                                weighted_center=self.weighted_center)
        helmert_mid.minim_funcs()
        Fmid = helmert_mid.Fexp

        c = 0
        while (any( item < 0 for item in (precision_level-abs(helm1-helm2)) )
                and c < maxiter):

            # Calculate the mid-point of the brackets
            helm_mid = (helm1+helm2)/2.0

            t0_helmert = time.process_time()
            helmert_mid = Helmert(helmert=helm_mid,sighelmert=self.sighelmert,
                                  coords0=self.coords0,sigmas0=self.sigmas0,
                                  coords1=self.coords1,sigmas1=self.sigmas1,
                                  orbflags=self.orbflags,
                                  weighted_center=self.weighted_center)
            t1_helmert = time.process_time() - t0_helmert
            t0_minim = time.process_time()
            helmert_mid.minim_funcs()
            t1_minim = time.process_time() - t0_minim
            t0_Fexp = time.process_time()
            Fmid = helmert_mid.Fexp
            t1_Fexp = time.process_time() - t0_Fexp
            logger.debug(f"t1_helmert, t1_minim, t1_Fexp: "
                         f"{t1_helmert} {t1_minim} {t1_Fexp}")

            if old_version is False:
                helmert1 = Helmert(helmert=helm1,sighelmert=self.sighelmert,
                                   coords0=self.coords0,sigmas0=self.sigmas0,
                                   coords1=self.coords1,sigmas1=self.sigmas1,
                                   orbflags=self.orbflags,
                                   weighted_center=self.weighted_center)

                helmert2 = Helmert(helmert=helm2,sighelmert=self.sighelmert,
                                   coords0=self.coords0,sigmas0=self.sigmas0,
                                   coords1=self.coords1,sigmas1=self.sigmas1,
                                   orbflags=self.orbflags,
                                   weighted_center=self.weighted_center)

                helmert1.minim_funcs()
                F1 = helmert1.Fexp
                helmert2.minim_funcs()
                F2 = helmert2.Fexp

            logger.debug(f"iteration {c}")
            logger.debug(f"helm1: {helm1}\nhelm2: {helm2}\n"
                         f"helm_mid: {helm_mid}")
            logger.debug(f"F1: {F1}\nF2: {F2}\nFmid: {Fmid}")
            logger.debug(f"|helm1-helm2|:\n {abs(helm1-helm2)}")

            # Replace the mid-points with the bracket of the same sign to
            # shorten the bisection window
            for i in range(len(helm1)):

                # check if F1 and F2 happened to have the same sign
                # This is why bisection is not designed for multi dimensions.
                # Other algorithms like Simplex could be considered
                if F1[i]*F2[i] >= 0.0:

                    # in this case,replace the closest one to Fmid by helm_mid
                    d1 = abs(Fmid[i]-F1[i])
                    d2 = abs(Fmid[i]-F2[i])
                    if d1<d2:
                        helm1[i] = helm_mid[i]
                        if old_version is True:
                            F1[i] = Fmid[i]
                    else:
                        helm2[i] = helm_mid[i]
                        if old_version is True:
                            F2[i] = Fmid[i]

                    # Issue a warning to the user
                    logger.warning(f"at iteration {c+1}: F1 and F2 happened "
                                   f"to have the same signs. The closest one "
                                   f"to Fmid is chosen to be replaced by "
                                   f"the midpoint ")
                else:

                    if Fmid[i]*F1[i] >= 0.0:
                        helm1[i] = helm_mid[i]
                        if old_version is True:
                            F1[i] = Fmid[i]
                    else:
                        helm2[i] = helm_mid[i]
                        if old_version is True:
                            F2[i] = Fmid[i]

            c += 1

        logger.debug(f'No. of iterations for bisection: {c}\n')
        if (c == maxiter):
            logger.warning(f"Number of iterations in bisection for L1 norm "
                           f"solution reached maxiter ({maxiter}). Use the "
                           f"results with caution!")

        logger.debug(f"bisection results after {c} iterations:\n"
                     f"helm: {helm_mid}\n"
                     f"Fexp: {Fmid}\n"
                     f"|helm1-helm2|:\n {abs(helm1-helm2)}")

        logger.debug(f"Bisection solution after {c} iterations:")
        logger.debug(f"helm: {helm_mid}")

        self.helmert = helm_mid
        self.Fexp = Fmid

        # update minimum functions
        self.minim_funcs()


    def CalAbdev(self,helm):

        # Function to compute absolute deviation given a set of Helmert
        # parameters helm; used as input for scipy minimization (l1norm)

        # Create an instance of helmert class using the given helm
        helm_instance = Helmert(helmert=helm,sighelmert=self.sighelmert,
                                coords0=self.coords0,sigmas0=self.sigmas0,
                                coords1=self.coords1,sigmas1=self.sigmas1,
                                orbflags=self.orbflags,
                                weighted_center=self.weighted_center)

        # Run minim_funcs method on the instance to get abdev
        helm_instance.minim_funcs()

        # Return abdev
        return helm_instance.abdev


    def CalRes(self,helm):

        # Function to compute residuals vector given a set of Helmert
        # parameters helm; used as input for scipy least squares (LS)

        # Create an instance of helmert class using the given helm
        helm_instance = Helmert(helmert=helm,sighelmert=self.sighelmert,
                                coords0=self.coords0,sigmas0=self.sigmas0,
                                coords1=self.coords1,sigmas1=self.sigmas1,
                                orbflags=self.orbflags,
                                weighted_center=self.weighted_center)

        # Determine the Jacobian matrix A for the instance
        helm_instance.Jacobian()

        # Create the observations vector l, which is the flattened coords1
        # array
        l = helm_instance.coords1.flatten()

        # Create the computational observations vector (lc = A*params)
        lc = np.matmul(helm_instance.A,helm_instance.helmert)

        # Calculate the residuals
        vcap = l-np.matmul(helm_instance.A,helm_instance.helmert)

        # return the residuals
        return vcap


    def l1norm(self,helm0=None,method='Nelder-Mead',maxiter=100):

        # using scipy optimize module, estimate the Helmert parameters by
        # minimizing the absolute deviation of the mean absolute deviation
        # function (i.e. assuming a double-exponential distribution function,
        # for which the density function is absolute deviation (see 
        # minim_funcs() method)

        # Check the given initial Helmert parameters
        # (default is self.helmert)
        if helm0 is None:
            helm0 = self.helmert
        else:
            checkutils.check_array(helm0,7)

        # Check the given method of minimization
        if method not in ['Nelder-Mead','Powell','CG','BFGS','Newton-CG',
                             'L-BFGS-B','TNC','COBYLA','SLSQP',
                             'trust-constr','dogleg','trust-ncg',
                             'trust-exact','trust-krylov']:
            logger.error(f"The input method {method} is not a recosgnized "
                         f"solver by scipy.optimize.minimize")
            raise ValueError(f"Unknown solver {method}")

        # Check the given maxiter
        checkutils.check_scalar(maxiter)
        if not isinstance(maxiter,int):
            logger.error("The given value for maxiter must be an integer",
                            stack_info=True)
            raise TypeError("The given value for maxiter must be an integer")

        # Run the scipy minimize
        # There are several options that can be used for mininmization,
        # including tolerations for x and f. As many of these options are 
        # method-specific, they are not passed as arguments to the method.
        # Different methods can be assessed to find the best, or the script
        # can be made more generic as to pass the correct option based on the
        # method chosen. For the options refer to scipy documentation.
        # TBD if necessary
        l1norm_solution = optimize.minimize(self.CalAbdev,helm0,method=method,
                                            options={'maxiter':maxiter})

        # update Helmert parameters
        self.helmert = l1norm_solution.x
        self.abdev = l1norm_solution.fun

        logger.debug(f"l1norm_solution: {l1norm_solution}")

        if l1norm_solution.success is False:
            logger.warning("L1 norm minimzation problem has not been "
                           "successful! Use the results with caution!")

        logger.debug(f"L1 norm Minimzation solution after "
                    f"{l1norm_solution.nit} iterations:\n"
                    f"helm: {l1norm_solution.x}\n"
                    f"success: {l1norm_solution.success}\n"
                    f"status: {l1norm_solution.status}\n"
                    f"Number of function evaluations: "
                    f"{l1norm_solution.nfev}\n"
                    f"Solver message: {l1norm_solution.message}\n"
                    f"abdev: {l1norm_solution.fun}\n")

        # update minimum functions (abdev is already updated but will do this
        # anyway for other functions)
        self.minim_funcs()


    def l2norm_scipy(self,helm0=None,method='trf',ftol=1e-8,xtol=1e-8):


        # uses scipy optimize least_squares method to solve for a 
        # least-squares solution of the Helmert parameters

        # Check the given initial Helmert parameters
        # (default is self.helmert)
        if helm0 is None:
            helm0 = self.helmert
        else:
            checkutils.check_array(helm0,7)

        # Check the given method of minimization
        if method not in ['trf','dogbox','lm']:
            logger.error(f"The input method {method} is not a recosgnized "
                         f"solver by scipy.optimize.least_squares")
            raise ValueError(f"Unknown solver {method}")

        # Check the given tolerances
        checkutils.check_scalar(ftol)
        checkutils.check_scalar(xtol)

        # Run the scipy optimize least_squares
        # There are several options that could be used. Refer to the scipy
        # documentation. TBD if necessary
        l2norm_solution = optimize.least_squares(self.CalRes,helm0,ftol=ftol,
                                                 xtol=xtol)

        # update Helmert parameters
        self.helmert = l2norm_solution.x

        logger.debug(f"l2norm_solution: {l2norm_solution}")

        if l2norm_solution.success is False:
            logger.warning("L2 norm minimzation problem has not been "
                           "successful! Use the results with caution!")

        logger.debug(f"L2 norm Minimzation solution after "
                    f"{l2norm_solution.nfev} function evaluations:\n"
                    f"helm: {l2norm_solution.x}\n"
                    f"success: {l2norm_solution.success}\n"
                    f"status: {l2norm_solution.status}\n"
                    f"Solver message: {l2norm_solution.message}\n"
                    f"cost function value: {l2norm_solution.cost}\n"
                    f"1st-order optimality measure: "
                    f"{l2norm_solution.optimality}\n")

        # update minimum functions
        self.minim_funcs()



#----------------------------------------------------------------------

if __name__ == "__main__":

    #helmert1 = Helmert([2,3,4,0,0,0,2],coords0=[[1,1,1],[2.5,2.4,2.2],[3.6,3.7,5],[3,4,5]],sigmas0=[[0.2,0.3,0.25],[0.24,0.32,0.57],[0.78,1.15,0.57],[0.54,2.1,1.2]])
    coords=[[12908.438637, -10025.115840,  20508.373977],
            [15079.968827,  -2630.253926,  22211.246286],
            [23313.808954, -10932.861931,   6118.858775],
            [20721.531368,  -2122.508558, -16544.439968],
            [-21032.861002,   1830.470014, -16349.972246],
            [-6551.363943, -24529.143018,  -7545.001977],
            [6145.944042, -22903.241294, -10980.933964],
            [24787.542561,   3766.698126,   9291.839227],
            [10459.030800, -11836.885227, -21446.749065],
            [-2003.840373,  17518.668479,  19954.907803],
            [-11918.525418, -17026.839834, -16540.090875],
            [-24034.134364,   9383.795236,   5861.320085],
            [-22826.376648, -13657.462694,   1852.531329],
            [6060.582427, -18936.839408,  17545.130199],
            [-25088.447006,  -1194.003253,   8463.170260],
            [17704.951928,   5584.874095, -19488.460051],
            [-3396.674852, -17552.616593,  20026.492178],
            [-5597.088199,  23742.610593, -10264.639359],
            [-12814.443316, -18896.440759,  13673.661409],
            [-12454.294791,  -9241.513176, -21493.998357],
            [15860.682874,   2932.783763,  21730.111035],
            [ -961.884088, -16463.460832,  21312.144959],
            [-13869.071128,  19517.036235,  11310.168972],
            [-15536.066519,   1716.492700,  21075.943816],
            [-19260.987560,  17577.551005,  -4529.192943],
            [10034.802446,  12333.836317, -21413.360007],
            [23056.216185,  13331.450416,  -3198.777892],
            [6046.656384,  25457.765360,  -4592.672112],
            [-12220.685797,   9825.111943, -21498.041633],
            [-877.604021, -26355.777864,  -1405.678417],
            [10397.743912,  20557.783342, -13057.252264],
            [ 8382.452652,  17852.750536,  18077.240947]]
    coords0 = [[x * 1000 for x in row] for row in coords]
    #helm_params = [-0.0002,-0.0005,0.005,2.96e-10,2.38e-10,2.18e-10,1-5e-11]
    helm_params = [-2,-0.0005,0.005,2.96e-10,2.38e-10,2.18e-10,1-5e-11]
    helmert1 = Helmert(helmert=helm_params,coords0=coords0)
    helmert1.transform()
    print("Helmert parameters applied to coords0 are written to coords1:")
    helmert1.printInfo()
    coords1_orig = helmert1.coords1
    helmert1.l2norm()
    helmert1.bracket()
    helmert1.bisection()
    helmert2 = Helmert(helmert=helmert1.helmert,coords0=coords0)
    helmert2.transform()
    print("Helmert parameters estimated between coords0 and coords1\n"
          "and then applied to coords0 to derive new coords1\n"
          "(which should be close to the original parameters):")
    helmert2.printInfo()
    coords1_new = helmert2.coords1
    print(f"residuals:\n{coords1_new - coords1_orig}")
