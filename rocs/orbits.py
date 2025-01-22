# Orbit combination module; includes preprocessing and combination classes

import numpy as np
import logging
import datetime
from scipy.interpolate import lagrange,InterpolatedUnivariateSpline
from rocs.io_data import SatelliteMetadata
import rocs.checkutils as checkutils
from rocs.helmert import Helmert
from rocs.gpscal import gpsCal


logger = logging.getLogger(__name__)

# Toggle between a version compatible with old software and the new version
old_version = True


class Data:

    # To represent a data point corresponding to x and y = f(x)

    def __init__(self, x, y):
        self.x = x
        self.y = y


def interpolate(f: list, xi: int, n: int) -> float:

    # function to interpolate the given data points using Lagrange's formula
    # This function implements Lagrange's interpolation based on
    # the example from GeeksforGeeks.
    # Source: GeeksforGeeks, https://www.geeksforgeeks.org/lagranges-interpolation/
    #
    # xi -> corresponds to the new data point whose value is to be obtained
    # n -> represents the number of known data points

    # Initialize result
    result = 0.0
    for i in range(n):

        # Compute individual terms of above formula
        term = f[i].y
        for j in range(n):
            if j != i:
                term = term * (xi - f[j].x) / (f[i].x - f[j].x)

        # Add current term to result
        result += term

    return result

def extract_windows_vectorized(array, clearing_time_index, max_time, sub_window_size):
    start = clearing_time_index + 1 - sub_window_size + 1

    sub_windows = (
        start +
        # expand_dims are used to convert a 1D array to 2D array.
        np.expand_dims(np.arange(sub_window_size), 0) +
        np.expand_dims(np.arange(max_time + 1), 0).T
    )

    return array[sub_windows]



class OrbitPrep:

    """
    class of individual orbits with preprocessing methods

    """

    def __init__(self,sp3all,ac_contribs=None,sat_metadata=None):

        """
        initialize OrbitPrep class

        Keyword arguments:
            sp3all [dict]                           : dictionary containing
                                                      all individual sp3 orbit
                                                      dictionaries
            ac_contribs [dict], optional            : center contributions to
                                                      the combination
            sat_metadata
                [class 'io_data.SatelliteMetadata'],
                optional                            : an instance of
                                                    input SatelliteMetadata
                                                    class
        Updates:
            self.sp3all [dict]
            self.ac_contribs [dict]
            self.sat_metadata [class 'io_data.SatelliteMetadata']

        """

        # Check the given sp3all
        if not isinstance(sp3all,dict):
            logger.error("\nThe given sp3all must be a dictionary\n",
                         stack_info=True)
            raise TypeError("sp3all is not a dictionary!")

        if not all(isinstance(key,str) for key in sp3all.keys()):
            logger.error("\nKeys of sp3all must all be strings\n",
                         stack_info=True)
            raise TypeError("sp3all keys must be strings!")

        if not all(isinstance(item,dict) for item in sp3all.values()):
            logger.error("\nValues of sp3all must all be dictionaries\n",
                         stack_info=True)
            raise TypeError("sp3all values must be dictionaries!")

        # After the above checks, assign sp3all to a class attribute
        self.sp3all = sp3all

        # Check the given ac_contribs
        if ac_contribs is not None:
            if not isinstance(ac_contribs,dict):
                logger.error("\nThe given argument ac_contribs must be a "
                             "dictionary\n")
                raise TypeError("ac_contribs must be of type dict")
            if not ac_contribs:
                logger.error("\nThere must be at least one dictionary item in "
                             "ac_contribs\n")
                raise ValueError("ac_contibs is empty!")

            # Assign the attribute
            self.ac_contribs = ac_contribs

        # Check the given sat_metadata
        if sat_metadata is not None:

            if not isinstance(sat_metadata,SatelliteMetadata):
                logger.error("\nsat_metadata must be an instance of "
                             "SatelliteMetadata class\n")
                raise TypeError("sat_metadata must an instance of "
                                "SatelliteMetadata class")

            # Assign the attribute
            self.sat_metadata = sat_metadata


    def filter_contribs(self):

        """
        Filter sp3all attribute based on ac_contribs attribute so each center
        only contains weighted and unweighted data, and not excluded data

        Keyword arguments:

        Updates:
            self.sp3all [dict]             : filtered sp3all dictionary
            self.weighted_centers [list]   : list of weighted centers
            self.unweighted_centers [list] : list of unweighted centers
            self.filtered [bool]           : boolean flag to show if sp3all has
                                             been filtered
        """

        # Check if ac_contribs attribute exists
        if not hasattr(self,'ac_contribs'):
            logger.error(f"\nNo ac_contribs attribute exists!\n")
            raise AttributeError(f"No ac_contribs attribute")

        sp3all_filtered = {}
        weighted_centers = []
        unweighted_centers = []
        weighted_sats = {}
        unweighted_sats = {}
        weighted_cens_by_sys = {}
        unweighted_cens_by_sys = {}

        # Loop over all ACs
        for acname in self.sp3all.keys():

            # Initialize the list of filtered satellites
            sats = self.sp3all[acname]['header']['sats']
            sat_accuracy = self.sp3all[acname]['header']['sat_accuracy']
            epochs = self.sp3all[acname]['data']['epochs']
            sats_weighted = []
            sat_accuracy_weighted = []
            sats_unweighted = []
            sat_accuracy_unweighted = []
            sats_wu = [] # weighted and unweighted sats (not excluded)
            sat_accuracy_wu = []


            # weighted centers
            if 'weighted' in self.ac_contribs:

                weighted = self.ac_contribs['weighted']

                if weighted is not None:

                    # weighted systems
                    if ('systems' in weighted
                            and weighted['systems'] is not None
                            and acname in weighted['systems']
                            and weighted['systems'][acname] is not None):
                        systems = weighted['systems'][acname]

                        # Add sat to sats_weighted, if weighted
                        for c,sat in enumerate(sats):
                            if sat[0] in systems and sat not in sats_weighted:
                                sats_weighted.append(sat)
                                sat_accuracy_weighted.append(sat_accuracy[c])
                            if sat[0] in systems and sat not in sats_wu:
                                sats_wu.append(sat)
                                sat_accuracy_wu.append(sat_accuracy[c])

                    # weighted prns
                    if ('prns' in weighted and weighted['prns'] is not None
                            and acname in weighted['prns']
                            and weighted['prns'][acname] is not None):
                        prns = weighted['prns'][acname]

                        # Add sat to sats_weighted, if weighted
                        for c,sat in enumerate(sats):
                            if sat in prns and sat not in sats_weighted:
                                sats_weighted.append(sat)
                                sat_accuracy_weighted.append(sat_accuracy[c])
                            if sat in prns and sat not in sats_wu:
                                sats_wu.append(sat)
                                sat_accuracy_wu.append(sat_accuracy[c])

                    # weighted svns
                    if ('svns' in weighted and weighted['svns'] is not None
                            and acname in weighted['svns']
                            and weighted['svns'][acname] is not None):
                        svns = weighted['svns'][acname]

                        # Add sat to sats_weighted, if weighted
                        if hasattr(self,'sat_metadata'):
                            for c,sat in enumerate(sats):
                                for epoch in epochs:
                                    svn = self.sat_metadata.get_svn(
                                                    sat[0],int(sat[1:]),epoch)
                                    svn_str = sat[0] + str(svn).zfill(3)
                                    if (svn_str in svns
                                            and sat not in sats_weighted):
                                        sats_weighted.append(sat)
                                        sat_accuracy_weighted.append(
                                                            sat_accuracy[c])
                                    if (svn_str in svns
                                            and sat not in sats_wu):
                                        sats_wu.append(sat)
                                        sat_accuracy_wu.append(
                                                            sat_accuracy[c])
                        else:
                            logger.warning("Weighted svns exist in ac_contribs"
                                           " but there is no satellite "
                                           "metadata information.\n Ignoring "
                                           f"weighted svns: {svns}\n")

            # unweighted centers
            if 'unweighted' in self.ac_contribs:

                unweighted = self.ac_contribs['unweighted']

                if unweighted is not None:

                    # unweighted systems
                    if ('systems' in unweighted
                            and unweighted['systems'] is not None
                            and acname in unweighted['systems']
                            and unweighted['systems'][acname] is not None):
                        systems = unweighted['systems'][acname]

                        # Add sat to sats_unweighted, if unweighted
                        for c,sat in enumerate(sats):
                            if (sat[0] in systems
                                    and sat not in sats_unweighted):
                                sats_unweighted.append(sat)
                                sat_accuracy_unweighted.append(sat_accuracy[c])
                            if (sat[0] in systems
                                    and sat not in sats_wu):
                                sats_wu.append(sat)
                                sat_accuracy_wu.append(sat_accuracy[c])

                        # Remove sat from sats_weighted, if unweighted
                        ind_rm = ([c for c,sat in enumerate(sats_weighted)
                                                        if sat[0] in systems])
                        for i in sorted(ind_rm, reverse=True):
                            del sats_weighted[i]
                            del sat_accuracy_weighted[i]

                    # unweighted prns
                    if ('prns' in unweighted and unweighted['prns'] is not None
                            and acname in unweighted['prns']
                            and unweighted['prns'][acname] is not None):
                        prns = unweighted['prns'][acname]

                        # Add sat to sats_unweighted, if unweighted
                        for c,sat in enumerate(sats):
                            if sat in prns and sat not in sats_unweighted:
                                sats_unweighted.append(sat)
                                sat_accuracy_unweighted.append(sat_accuracy[c])
                            if sat in prns and sat not in sats_wu:
                                sats_wu.append(sat)
                                sat_accuracy_wu.append(sat_accuracy[c])

                        # Remove sat from sats_weighted, if unweighted
                        ind_rm = ([c for c,sat in enumerate(sats_weighted)
                                                            if sat in prns])
                        for i in sorted(ind_rm, reverse=True):
                            del sats_weighted[i]
                            del sat_accuracy_weighted[i]

                    # unweighted svns
                    if ('svns' in unweighted and unweighted['svns'] is not None
                            and acname in unweighted['svns']
                            and unweighted['svns'][acname] is not None):
                        svns = unweighted['svns'][acname]

                        if hasattr(self,'sat_metadata'):

                            # Add sat to sats_unweighted, if unweighted
                            for c,sat in enumerate(sats):
                                for epoch in epochs:
                                    svn = self.sat_metadata.get_svn(
                                                    sat[0],int(sat[1:]),epoch)
                                    svn_str = sat[0] + str(svn).zfill(3)

                                    if (svn_str in svns
                                        and sat not in sats_unweighted):
                                        sats_unweighted.append(sat)
                                        sat_accuracy_unweighted.append(
                                                            sat_accuracy[c])
                                    if (svn_str in svns
                                        and sat not in sats_wu):
                                        sats_wu.append(sat)
                                        sat_accuracy_wu.append(
                                                            sat_accuracy[c])

                            # Remove sat from sats_weighted, if unweighted
                            ind_rm = []
                            for c,sat in enumerate(sats_weighted):
                                remove_sat = True
                                for epoch in epochs:
                                    svn = self.sat_metadata.get_svn(
                                                    sat[0],int(sat[1:]),epoch)
                                    svn_str = sat[0] + str(svn).zfill(3)
                                    if svn_str not in svns:
                                        remove_sat = False
                                if remove_sat:
                                    ind_rm.append(c)
                            if ind_rm:
                                for i in sorted(ind_rm, reverse=True):
                                    del sats_weighted[i]
                                    del sat_accuracy_weighted[i]
                        else:
                            logger.warning("Unweighted svns exist in "
                                           "ac_contribs but there is no "
                                           "satellite metadata information.\n "
                                           "Ignoring unweighted svns: "
                                           f"{svns}\n")

            # remove excluded centers
            if 'excluded' in self.ac_contribs:

                excluded = self.ac_contribs['excluded']

                if excluded is not None:

                    # excluded systems
                    if ('systems' in excluded
                            and excluded['systems'] is not None
                            and acname in excluded['systems']
                            and excluded['systems'][acname] is not None):
                        systems = excluded['systems'][acname]

                        # Remove sat from sats_weighted, if excluded
                        ind_rm = ([c for c,sat in enumerate(sats_weighted)
                                                        if sat[0] in systems])
                        for i in sorted(ind_rm, reverse=True):
                            del sats_weighted[i]
                            del sat_accuracy_weighted[i]

                        # Remove sat from sats_unweighted, if excluded
                        ind_rm = ([c for c,sat in enumerate(sats_unweighted)
                                                        if sat[0] in systems])
                        for i in sorted(ind_rm, reverse=True):
                            del sats_unweighted[i]
                            del sat_accuracy_unweighted[i]

                        # Remove sat from sats_wu, if excluded
                        ind_rm = ([c for c,sat in enumerate(sats_wu)
                                                        if sat[0] in systems])
                        for i in sorted(ind_rm, reverse=True):
                            del sats_wu[i]
                            del sat_accuracy_wu[i]

                    # excluded prns
                    if ('prns' in excluded and excluded['prns'] is not None
                            and acname in excluded['prns']
                            and excluded['prns'][acname] is not None):
                        prns = excluded['prns'][acname]

                        # Remove sat from sats_weighted, if excluded
                        ind_rm = ([c for c,sat in enumerate(sats_weighted)
                                                            if sat in prns])
                        for i in sorted(ind_rm, reverse=True):
                            del sats_weighted[i]
                            del sat_accuracy_weighted[i]

                        # Remove sat from sats_unweighted, if excluded
                        ind_rm = ([c for c,sat in enumerate(sats_unweighted)
                                                            if sat in prns])
                        for i in sorted(ind_rm, reverse=True):
                            del sats_unweighted[i]
                            del sat_accuracy_unweighted[i]

                        # Remove sat from sats_wu, if excluded
                        ind_rm = ([c for c,sat in enumerate(sats_wu)
                                                            if sat in prns])
                        for i in sorted(ind_rm, reverse=True):
                            del sats_wu[i]
                            del sat_accuracy_wu[i]

                    # excluded svns
                    if ('svns' in excluded and excluded['svns'] is not None
                            and acname in excluded['svns']
                            and excluded['svns'][acname] is not None):
                        svns = excluded['svns'][acname]

                        if hasattr(self,'sat_metadata'):

                            # Remove sat from sats_weighted, if excluded
                            ind_rm = []
                            for c,sat in enumerate(sats_weighted):
                                remove_sat = True
                                for epoch in epochs:
                                    svn = self.sat_metadata.get_svn(
                                                    sat[0],int(sat[1:]),epoch)
                                    svn_str = sat[0] + str(svn).zfill(3)
                                    if svn_str not in svns:
                                        remove_sat = False
                                if remove_sat:
                                    ind_rm.append(c)
                            if ind_rm:
                                for i in sorted(ind_rm, reverse=True):
                                    del sats_weighted[i]
                                    del sat_accuracy_weighted[i]

                            # Remove sat from sats_unweighted, if excluded
                            ind_rm = []
                            for c,sat in enumerate(sats_unweighted):
                                remove_sat = True
                                for epoch in epochs:
                                    svn = self.sat_metadata.get_svn(
                                                    sat[0],int(sat[1:]),epoch)
                                    svn_str = sat[0] + str(svn).zfill(3)
                                    if svn_str not in svns:
                                        remove_sat = False
                                if remove_sat:
                                    ind_rm.append(c)
                            if ind_rm:
                                for i in sorted(ind_rm, reverse=True):
                                    del sats_unweighted[i]
                                    del sat_accuracy_unweighted[i]

                            # Remove sat from sats_wu, if excluded
                            ind_rm = []
                            for c,sat in enumerate(sats_wu):
                                remove_sat = True
                                for epoch in epochs:
                                    svn = self.sat_metadata.get_svn(
                                                    sat[0],int(sat[1:]),epoch)
                                    svn_str = sat[0] + str(svn).zfill(3)
                                    if svn_str not in svns:
                                        remove_sat = False
                                if remove_sat:
                                    ind_rm.append(c)
                            if ind_rm:
                                for i in sorted(ind_rm, reverse=True):
                                    del sats_wu[i]
                                    del sat_accuracy_wu[i]

                        else:
                            logger.warning("Excluded svns exist in ac_contribs"
                                           " but there is no satellite "
                                           "metadata information.\n Ignoring "
                                           f"svn exclusions for: {svns}\n")

            # Update the main weighted_sats and unweighted_sats dictionary
            weighted_sats[acname] = sats_weighted
            unweighted_sats[acname] = sats_unweighted

            # if sats_weighted is not empty, we will have a sp3 dict for the
            # weighted data of this center
            if sats_weighted:

                # list of systems for this center
                sys_ids_weighted = []
                for sat in sats_weighted:
                    if sat[0] not in sys_ids_weighted:
                        sys_ids_weighted.append(sat[0])

                # list of variables in the original sp3 dictionary of this center
                varnames = []
                for key in self.sp3all[acname]['data']:
                    if (type(key) is tuple):
                        if (key[2] not in varnames):
                            varnames.append(key[2])

                # The weighted sp3 dictionary for this AC
                sp3_weighted = {}

                # The header is the same as the original header except for
                # numsats, sats, sat_accuracy, and file_type
                sp3_weighted['header'] = {}
                for key in self.sp3all[acname]['header'].keys():
                    sp3_weighted['header'][key] = (self.sp3all[acname]
                                                            ['header'][key])
                    sp3_weighted['header']['numsats'] = len(sats_weighted)
                    sp3_weighted['header']['sats'] = sats_weighted
                    sp3_weighted['header']['sat_accuracy'] \
                                                    = sat_accuracy_weighted
                    if len(sys_ids_weighted) > 1:
                        sp3_weighted['header']['file_type'] = 'M '
                    else:
                        sp3_weighted['header']['file_type'] \
                                                    = sys_ids_weighted[0]

                # Data epochs are the same
                sp3_weighted['data'] = {}
                sp3_weighted['data']['epochs'] = (self.sp3all[acname]
                                                        ['data']['epochs'])

                # Only store data for sats_weighted
                for epoch in sp3_weighted['data']['epochs']:
                    for sat in sats_weighted:
                        for varname in varnames:
                            key = (sat,epoch,varname)
                            if key in self.sp3all[acname]['data']:
                                sp3_weighted['data'][key] = (self.sp3all
                                                        [acname]['data'][key])

                # add ac to weighted_centers
                weighted_centers.append(acname)

                weighted_cens_by_sys[acname] = sys_ids_weighted

            # if sats_unweighted is not empty, we will have a sp3 dict for the
            # unweighted data of this center
            if sats_unweighted:

                # list of systems for this center
                sys_ids_unweighted = []
                for sat in sats_unweighted:
                    if sat[0] not in sys_ids_unweighted:
                        sys_ids_unweighted.append(sat[0])

                # list of variables in the original sp3 dictionary of this center
                varnames = []
                for key in self.sp3all[acname]['data']:
                    if (type(key) is tuple):
                        if (key[2] not in varnames):
                            varnames.append(key[2])

                # The unweighted sp3 dictionary for this AC
                sp3_unweighted = {}

                # The header is the same as the original header except for
                # numsats, sats, sat_accuracy, and file_type
                sp3_unweighted['header'] = {}
                for key in self.sp3all[acname]['header'].keys():
                    sp3_unweighted['header'][key] = (self.sp3all[acname]
                                                            ['header'][key])
                    sp3_unweighted['header']['numsats'] = len(sats_unweighted)
                    sp3_unweighted['header']['sats'] = sats_unweighted
                    sp3_unweighted['header']['sat_accuracy'] \
                                                    = sat_accuracy_unweighted
                    if len(sys_ids_unweighted) > 1:
                        sp3_unweighted['header']['file_type'] = 'M '
                    else:
                        sp3_unweighted['header']['file_type'] \
                                                    = sys_ids_unweighted[0]

                # Data epochs are the same
                sp3_unweighted['data'] = {}
                sp3_unweighted['data']['epochs'] = (self.sp3all[acname]
                                                        ['data']['epochs'])

                # Only store data for sats_unweighted
                for epoch in sp3_unweighted['data']['epochs']:
                    for sat in sats_unweighted:
                        for varname in varnames:
                            key = (sat,epoch,varname)
                            if key in self.sp3all[acname]['data']:
                                sp3_unweighted['data'][key] = (self.sp3all
                                                        [acname]['data'][key])

                # add ac to unweighted_centers
                unweighted_centers.append(acname)
                unweighted_cens_by_sys[acname] = sys_ids_unweighted

            # if sats_wu is not empty, we will have a sp3 dict for the
            # weighted/unweighted data of this center
            if sats_wu:

                # list of systems for this center
                sys_ids_wu = []
                for sat in sats_wu:
                    if sat[0] not in sys_ids_wu:
                        sys_ids_wu.append(sat[0])

                # list of variables in the original sp3 dictionary of this
                # center
                varnames = []
                for key in self.sp3all[acname]['data']:
                    if (type(key) is tuple):
                        if (key[2] not in varnames):
                            varnames.append(key[2])

                # The weighted/unweighted sp3 dictionary for this AC
                sp3_wu = {}

                # The header is the same as the original header except for
                # numsats, sats, sat_accuracy, and file_type
                sp3_wu['header'] = {}
                for key in self.sp3all[acname]['header'].keys():
                    sp3_wu['header'][key] = (self.sp3all[acname]
                                                            ['header'][key])
                    sp3_wu['header']['numsats'] = len(sats_wu)
                    sp3_wu['header']['sats'] = sats_wu
                    sp3_wu['header']['sat_accuracy'] \
                                                    = sat_accuracy_wu
                    if len(sys_ids_wu) > 1:
                        sp3_wu['header']['file_type'] = 'M '
                    else:
                        sp3_wu['header']['file_type'] \
                                                    = sys_ids_wu[0]

                # Data epochs are the same
                sp3_wu['data'] = {}
                sp3_wu['data']['epochs'] = (self.sp3all[acname]
                                                        ['data']['epochs'])

                # Only store data for sats_wu
                for epoch in sp3_wu['data']['epochs']:
                    for sat in sats_wu:
                        for varname in varnames:
                            key = (sat,epoch,varname)
                            if key in self.sp3all[acname]['data']:
                                sp3_wu['data'][key] = (self.sp3all
                                                        [acname]['data'][key])


                sp3all_filtered[acname] = sp3_wu

        # Update sp3all with the filtered version
        self.sp3all = sp3all_filtered

        # Re-generate weighted_centers and unweighted_centers using weighted_sats and unweighted_sats
        # Note this definition should only be used to indicate what centres have at least
        # one satellite contributing
        weighted_centers = []
        unweighted_centers = []
        for acname in weighted_sats:
            if (len(weighted_sats[acname]) > 0 and acname not in weighted_centers):
                weighted_centers.append(acname)
        for acname in unweighted_sats:
            if (len(unweighted_sats[acname]) > 0 and acname not in weighted_centers
                    and acname not in unweighted_centers):
                unweighted_centers.append(acname)
        weighted_centers.sort()
        unweighted_centers.sort()

        logger.debug(f"weighted_sats: {weighted_sats}")
        logger.debug(f"unweighted_sats: {unweighted_sats}")

        logger.debug(f"weighted_centers: {weighted_centers}")
        logger.debug(f"unweighted_centers: {unweighted_centers}")

        logger.debug(f"weighted_cens_by_sys: {weighted_cens_by_sys}")
        logger.debug(f"unweighted_cens_by_sys: {unweighted_cens_by_sys}")

        # Assign to attributes
        self.weighted_centers = weighted_centers
        self.unweighted_centers = unweighted_centers
        self.filtered = True
        self.weighted_sats = weighted_sats
        self.unweighted_sats = unweighted_sats
        self.weighted_cens_by_sys = weighted_cens_by_sys
        self.unweighted_cens_by_sys = unweighted_cens_by_sys


    def resample(self,sample_rate):

        """
        Resample sp3all attribute to the given sampling rate

        Keyword arguments:
        sample_rate [int]: the sampling rate of the resampled sp3all in seconds

        Updates:
            self.sp3all [dict]
        """

        # Check the given sampling rate
        if not isinstance(sample_rate,int):
            logger.error(f"The given sample_rate must be an integer",
                            stack_info=True)
            raise TypeError(f"The given sample_rate {sample_rate} is not of "
                            f"type integer")

        # Initialize the resampled sp3 dictionary
        sp3all_rs = {}

        for acname in self.sp3all:

            # Copy the header items to the resampled sp3 dictionary
            sp3all_rs[acname] = {}
            sp3all_rs[acname]['header'] = {}
            for key in self.sp3all[acname]['header'].keys():
                sp3all_rs[acname]['header'][key] = (self.sp3all[acname]
                                                            ['header'][key])
            sp3all_rs[acname]['data'] = {}

            # read the original sample rate
            sample_rate_orig = self.sp3all[acname]['header']['epoch_int']
            epochs_orig = list(self.sp3all[acname]['data']['epochs'])

            # Check if we need to interpolate or to down-sample
            if sample_rate < sample_rate_orig:

                # if we know that center is unweighted, do not interpolate
                if (hasattr(self,'weighted_centers') and
                        acname not in self.weighted_centers):
                    sp3all_rs[acname]['data'] = self.sp3all[acname]['data']
                    epochs_rs = list(epochs_orig)

                else:

                    # Interpolate
                    epoch_start = epochs_orig[0]
                    epoch_end = epochs_orig[-1]
                    epochs_interp = []
                    for sat in  self.sp3all[acname]['header']['sats']:
                        xcoords = {}
                        ycoords = {}
                        zcoords = {}
                        epochs = list(epochs_orig)
                        xcoords_interp = {}
                        ycoords_interp = {}
                        zcoords_interp = {}

                        for epoch in epochs:
                            if ((sat,epoch,'xcoord') in
                                    self.sp3all[acname]['data']
                                    and self.sp3all[acname]['data']
                                            [(sat,epoch,'xcoord')] != 0):
                                xcoords[epoch] = (self.sp3all[acname]['data']
                                                        [(sat,epoch,'xcoord')])
                            else:
                                xcoords[epoch] = np.nan
                            if ((sat,epoch,'ycoord') in
                                    self.sp3all[acname]['data']
                                    and self.sp3all[acname]['data']
                                            [(sat,epoch,'ycoord')] != 0):
                                ycoords[epoch] = (self.sp3all[acname]['data']
                                                        [(sat,epoch,'ycoord')])
                            else:
                                ycoords[epoch] = np.nan
                            if ((sat,epoch,'zcoord') in
                                    self.sp3all[acname]['data']
                                    and self.sp3all[acname]['data']
                                            [(sat,epoch,'zcoord')] != 0):
                                zcoords[epoch] = (self.sp3all[acname]['data']
                                                        [(sat,epoch,'zcoord')])
                            else:
                                zcoords[epoch] = np.nan

                        xlist = []
                        ylist = []
                        zlist = []
                        epochs.sort()
                        for epoch in epochs:
                            xlist.append(xcoords[epoch])
                            ylist.append(ycoords[epoch])
                            zlist.append(zcoords[epoch])

                        tsec = np.arange(0,11*sample_rate_orig,sample_rate_orig)
                        epochs_window = epochs[5:-5]
                        xwindows = extract_windows_vectorized(
                                    np.array(xlist),9,len(xlist)-11,11)
                        ywindows = extract_windows_vectorized(
                                    np.array(ylist),9,len(ylist)-11,11)
                        zwindows = extract_windows_vectorized(
                                    np.array(zlist),9,len(zlist)-11,11)
                        xpoly = []
                        ypoly = []
                        zpoly = []
                        for i in range(len(xwindows)):
                            xpoly.append([Data(tsec[k],xwindows[i][k])
                                                for k in range(len(tsec))])
                            ypoly.append([Data(tsec[k],ywindows[i][k])
                                                for k in range(len(tsec))])
                            zpoly.append([Data(tsec[k],zwindows[i][k])
                                                for k in range(len(tsec))])
                        epochs_interp = []
                        x_interp = []
                        y_interp = []
                        z_interp = []

                        # Interpolation for the first 4 epochs
                        for k in range(3,5):
                            for t in np.arange(0,sample_rate_orig,sample_rate):
                                epochs_interp.append(
                                    epochs[k]+datetime.timedelta(seconds=t))
                                x_interp.append(interpolate(
                                        xpoly[0],k*sample_rate_orig+t,11))
                                y_interp.append(interpolate(
                                        ypoly[0],k*sample_rate_orig+t,11))
                                z_interp.append(interpolate(
                                        zpoly[0],k*sample_rate_orig+t,11))

                        for i in range(len(xpoly)):
                            for t in np.arange(0,sample_rate_orig,sample_rate):
                                epochs_interp.append(
                                    epochs_window[i]
                                    + datetime.timedelta(seconds=t))
                                x_interp.append(interpolate(
                                            xpoly[i],5*sample_rate_orig+t,11))
                                y_interp.append(interpolate(
                                            ypoly[i],5*sample_rate_orig+t,11))
                                z_interp.append(interpolate(
                                            zpoly[i],5*sample_rate_orig+t,11))

                        # Interpolation for the last 4 epochs (up to when there
                        # is original data; no extrapolation!)
                        for k in range(0,1):
                            for t in np.arange(0,sample_rate_orig,sample_rate):
                                epochs_interp.append(
                                    epochs[len(epochs)-5+k]
                                        + datetime.timedelta(seconds=t))
                                x_interp.append(interpolate(
                                        xpoly[-1],(k+6)*sample_rate_orig+t,11))
                                y_interp.append(interpolate(
                                        ypoly[-1],(k+6)*sample_rate_orig+t,11))
                                z_interp.append(interpolate(
                                        zpoly[-1],(k+6)*sample_rate_orig+t,11))

                        # First and last few epochs - no interpolation
                        epochs_interp.insert(0,epochs[2])
                        x_interp.insert(0,xlist[2])
                        y_interp.insert(0,ylist[2])
                        z_interp.insert(0,zlist[2])
                        epochs_interp.insert(0,epochs[1])
                        x_interp.insert(0,xlist[1])
                        y_interp.insert(0,ylist[1])
                        z_interp.insert(0,zlist[1])
                        epochs_interp.insert(0,epochs[0])
                        x_interp.insert(0,xlist[0])
                        y_interp.insert(0,ylist[0])
                        z_interp.insert(0,zlist[0])
                        epochs_interp.append(epochs[-4])
                        x_interp.append(xlist[-4])
                        y_interp.append(ylist[-4])
                        z_interp.append(zlist[-4])
                        epochs_interp.append(epochs[-3])
                        x_interp.append(xlist[-3])
                        y_interp.append(ylist[-3])
                        z_interp.append(zlist[-3])
                        epochs_interp.append(epochs[-2])
                        x_interp.append(xlist[-2])
                        y_interp.append(ylist[-2])
                        z_interp.append(zlist[-2])
                        epochs_interp.append(epochs[-1])
                        x_interp.append(xlist[-1])
                        y_interp.append(ylist[-1])
                        z_interp.append(zlist[-1])

                        for i,epoch in enumerate(epochs_interp):
                            sp3all_rs[acname]['data'][(sat,epoch,'xcoord')] = (
                                                        x_interp[i])
                            sp3all_rs[acname]['data'][(sat,epoch,'ycoord')] = (
                                                        y_interp[i])
                            sp3all_rs[acname]['data'][(sat,epoch,'zcoord')] = (
                                                        z_interp[i])

                    epochs_interp.sort()

                    epochs_rs = list(epochs_interp)

                    for sat in  self.sp3all[acname]['header']['sats']:
                        for epoch in epochs_interp:
                            if ((sat,epoch,'xcoord') in
                                    sp3all_rs[acname]['data']):
                                sp3all_rs[acname]['data'][(sat,epoch,'Pflag')] = 1
                            else:
                                sp3all_rs[acname]['data'][(sat,epoch,'Pflag')] = 0
                            sp3all_rs[acname]['data'][(sat,epoch,'EPflag')] = 0
                            sp3all_rs[acname]['data'][(sat,epoch,'Vflag')] = 0
                            sp3all_rs[acname]['data'][(sat,epoch,'EVflag')] = 0
            else:

                # Get the downsample rate
                downsample_rate = sample_rate/sample_rate_orig

                # Check that downsampling rate is integer
                if int(downsample_rate) != downsample_rate:
                    logger.error(f"The given sample_rate must be an integer "
                                 f"multiple of all of the original sampling "
                                 f"rates.\nsample_rate is {sample_rate} but "
                                 f"original sample rate for {acname} is "
                                 f"{sample_rate_orig}", stack_info=True)
                    raise ValueError(f"Downsampling is not possible because "
                                     f"the sampling rate is not an integer "
                                     f"multiple of the original sampling rate "
                                     f"for {acname}")


                # downsampled epochs
                epochs_downsampled = epochs_orig[::int(downsample_rate)]
                epochs_rs = list(epochs_downsampled)

                # Copy data only at the sampling rate
                for key in self.sp3all[acname]['data'].keys():
                    if (key != 'epochs' and key[1] in epochs_rs):
                        sp3all_rs[acname]['data'][key] = (self.sp3all[acname]
                                                                ['data'][key])

            # Update the resmapled sp3 dictionary
            sp3all_rs[acname]['data']['epochs'] = epochs_rs

            # Update header information
            sp3all_rs[acname]['header']['start_year'] = (epochs_rs[0].year)
            sp3all_rs[acname]['header']['start_month'] = (epochs_rs[0].month)
            sp3all_rs[acname]['header']['start_day'] = (epochs_rs[0].day)
            sp3all_rs[acname]['header']['start_hour'] = (epochs_rs[0].hour)
            sp3all_rs[acname]['header']['start_min'] = (epochs_rs[0].minute)
            sp3all_rs[acname]['header']['start_sec'] = (epochs_rs[0].second)
            sp3all_rs[acname]['header']['num_epochs'] = len(epochs_rs)
            sp3all_rs[acname]['header']['epoch_int'] = float(sample_rate)
            if (hasattr(self,'weighted_centers') and
                    acname not in self.weighted_centers):
                sp3all_rs[acname]['header']['epoch_int'] = float(
                                                            sample_rate_orig)

        # Update sp3all with the downsampled version
        self.sp3all = sp3all_rs
        self.sample_rate = sample_rate


    def rm_dv(self,dv,no_rm_dv):

        """
        Remove maneuvered satellites from AC solutions

        Keyword arguments:
        dv [array]: array of year,doy,hour,minute,prn for DV events
        no_rm_dv [list]: list of AC's for which  we want to keep maneuvered
                         satellites; this should normally be an AC that are
                        using some sort of modlling for maneuver events
        Updates:
            self.sp3all [dict]
        """

        # convert center names in no_rm_dv in case they are not
        no_rm_dv = [item.upper() for item in no_rm_dv]

        # Initialize the DV-removed sp3 dictionary
        sp3all_nodv = {}

        # Loop over centres
        for acname in self.sp3all:

            if acname not in no_rm_dv:

                sp3all_nodv[acname] = {}
                sp3all_nodv[acname]['data'] = {}
                sp3all_nodv[acname]['header'] = {}

                # copy all epochs from the original dict to the new DV-removed dict
                epochs = list(self.sp3all[acname]['data']['epochs'])
                sp3all_nodv[acname]['data']['epochs'] = epochs

                # If there is a maneuvere for a satellite, remove any epochs
                # after the maneuvere
                sats = list(self.sp3all[acname]['header']['sats'])
                sat_accuracy = list(self.sp3all[acname]['header']['sat_accuracy'])
                numsats = int(self.sp3all[acname]['header']['numsats'])
                remsat = {}
                for row in dv:
                    if (row[0] >= min(epochs) and row[0] <= max(epochs)
                                                        and row[1] in sats):
                        logger.info(f"{row[1]} removed from {acname} solution "
                                    f"from {row[0]} due to maneuver")
                        remsat[row[1]] = []
                        for epoch in epochs:
                            if epoch >= row[0]:
                                remsat[row[1]].append(epoch)

                # if more than 50% of epochs are to be removed for this
                # satellite, remove the satelite for the whole period
                for sat in remsat:
                    if len(remsat[sat])/len(epochs) > 0.5:
                        remsat[sat] = epochs
                        ind = sats.index(sat)
                        del sats[ind]
                        del sat_accuracy[ind]
                        numsats -= 1
                        logger.info(f"{sat} removed fully from {acname} solution "
                                    f"because more than 50% of epochs were removed")

                # copy data except for maneuvering sats
                for key in self.sp3all[acname]['data']:
                    if key != 'epochs':
                        sp3all_nodv[acname]['data'][key] = (self.sp3all[acname]
                                                                    ['data'][key])
                        if (key[0] in remsat and key[1] in remsat[key[0]]):
                            sp3all_nodv[acname]['data'][(key[0],key[1],'xcoord')] = 0.0
                            sp3all_nodv[acname]['data'][(key[0],key[1],'ycoord')] = 0.0
                            sp3all_nodv[acname]['data'][(key[0],key[1],'zcoord')] = 0.0
                            sp3all_nodv[acname]['data'][(key[0],key[1],'clock')] = 999999.999999
                            sp3all_nodv[acname]['data'][(key[0],key[1],'xsdev')] = ' '
                            sp3all_nodv[acname]['data'][(key[0],key[1],'ysdev')] = ' '
                            sp3all_nodv[acname]['data'][(key[0],key[1],'zsdev')] = ' '
                            sp3all_nodv[acname]['data'][(key[0],key[1],'csdev')] = ' '

                # Copy the header items to the DV-removed sp3 dictionary
                for key in self.sp3all[acname]['header']:
                    if key not in ['sats','sat_accuracy','numsats']:
                        sp3all_nodv[acname]['header'][key] = (self.sp3all[acname]
                                                            ['header'][key])
                sp3all_nodv[acname]['header']['sats'] = sats
                sp3all_nodv[acname]['header']['sat_accuracy'] = sat_accuracy
                sp3all_nodv[acname]['header']['numsats'] = numsats

            else:
                sp3all_nodv[acname] = self.sp3all[acname]

        self.sp3all = sp3all_nodv


    def to_arrays(self):

        """
        Convert sp3all and sat_metadata attributes to orbits, epoch and
        satinfo arrays, so they can be used by OrbitComb class

        Updates:
            self.weighted_centers
                         [list] : list of weighted centers
            self.unweighted_centers
                         [list] : list of unweighted centers
            self.orbits [dict]  : orbit arrays (x,y,z) of each AC (shape number
                                  of observations by 3)
            self.epochs [array] : epochs corresponding to rows of each orbits
                                  array (same length as each of the arrays in
                                  orbits)
            self.satinfo [array]: full satellite information corresponding to
                                  each orbits array (same length as epochs but
                                  with four columns for constellation ID, PRN,
                                  SVN and satellite block)
            self.cenflags [dict]: center flags to indicate whether a center is
                                  weighted or unweighted
            self.clocks [dict]:   clocks/clock sdev of each AC from the sp3 file
                                  (shape number of observations by 2)
        """

        # If not previously filtered, and there is an ac_contribs attribute,
        # filter sp3all to only include weighted and unweighted data,
        # and update list of weighted and unweighted centers
        if (not hasattr(self,'filtered') or
                (hasattr(self,'filtered') and not self.filtered) ):

            if hasattr(self,'ac_contribs'):

                # filter sp3all attribute
                self.filter_contribs()

            else:

                # assign weighted_centers to all centers, and unweighted centers
                # to no center
                self.weighted_centers = list(self.sp3all.keys())
                self.unweighted_centers = []

        # Get list of all epochs and satellites across all the sp3 dicts from
        # different weighted centers
        epochs = []
        sats = []
        for acname in self.weighted_centers:
            for epoch in self.sp3all[acname]['data']['epochs']:
                if epoch not in epochs:
                    epochs.append(epoch)
            for sat in self.weighted_sats[acname]:
                if sat not in sats:
                    sats.append(sat)
        epochs.sort()
        sats.sort()

        # Initializations
        orbits = {}
        clocks = {}
        all_epochs = []
        satinfo = np.zeros((len(epochs)*len(sats),4),dtype=object)
        satinfo[:,0] = np.zeros(len(epochs)*len(sats),dtype=str)
        satinfo[:,3] = np.zeros(len(epochs)*len(sats),dtype=str)

        # loop through centers, epochs and sats, and fill in orbits,
        # all_epochs and satinfo
        for ac_counter,acname in enumerate(self.sp3all.keys()):

            # Initialize orbits[acname]; missing values will be nan
            orbits[acname] = np.full((len(epochs)*len(sats),3),np.nan)
            clocks[acname] = np.full((len(epochs)*len(sats),2),np.nan)

            c = 0
            for epoch in epochs:
                for sat in sats:

                    # Check if there is data at this center/sat/epoch
                    if (sat,epoch,'xcoord') in self.sp3all[acname]['data']:

                        xcoord  = (self.sp3all[acname]
                                            ['data'][(sat,epoch,'xcoord')])
                        if xcoord != 0.0 and xcoord < 999999: # missing/bad values
                            orbits[acname][c,0] = 1000.0*xcoord

                    if (sat,epoch,'ycoord') in self.sp3all[acname]['data']:

                        ycoord  = (self.sp3all[acname]
                                            ['data'][(sat,epoch,'ycoord')])
                        if ycoord != 0.0 and ycoord < 999999: # missing/bad values
                            orbits[acname][c,1] = 1000.0*ycoord

                    if (sat,epoch,'zcoord') in self.sp3all[acname]['data']:

                        zcoord  = (self.sp3all[acname]
                                            ['data'][(sat,epoch,'zcoord')])
                        if zcoord != 0.0 and zcoord < 999999: # missing/bad values
                            orbits[acname][c,2] = 1000.0*zcoord

                    if (sat,epoch,'clock') in self.sp3all[acname]['data']:

                        clock  = (self.sp3all[acname]
                                            ['data'][(sat,epoch,'clock')])
                        if clock != 0.0 and clock < 999999: # missing/bad values
                            clocks[acname][c,0] = clock

                    if (sat,epoch,'csdev') in self.sp3all[acname]['data']:

                        csdev  = (self.sp3all[acname]
                                            ['data'][(sat,epoch,'csdev')])
                        try:
                            if float(csdev) != 0.0: # missing/bad values
                                clocks[acname][c,1] =  float(csdev)
                        except:
                            pass

                    # Only fill in self.epochs and satinfo once (as they are
                    # the same for all ACs)
                    if ac_counter == 0:

                        all_epochs.append(epoch)

                        # Constellation ID
                        satinfo[c,0] = sat[0]

                        # PRN
                        satinfo[c,1] = int(sat[1:])

                        if hasattr(self,'sat_metadata'):

                            # svn
                            satinfo[c,2] = (self.sat_metadata.get_svn(sat[0],
                                                        int(sat[1:]),epoch))

                            # satellite block
                            satinfo[c,3] = (self.sat_metadata.
                                    get_sat_identifier(sat[0],satinfo[c,2]).
                                    block)

                        else:

                            # If no metadata, set SVN as NaN, and block as 
                            #'Unknown'
                            satinfo[c,2] = np.nan
                            satinfo[c,3] = 'Unknown'
                    c += 1

        # Create center flags
        cenflags = {}
        for acname in orbits:
            if acname in self.weighted_centers:
                cenflags[acname] = 'weighted'
            elif acname in self.unweighted_centers:
                cenflags[acname] = 'unweighted'
            else:
                raise ValueError(f"center {acname} not in weighted_centers "
                                  "nor in unweighted_centers")

        # Update attributes
        self.orbits = orbits
        self.clocks = clocks
        self.epochs = all_epochs
        self.satinfo = satinfo
        self.cenflags = cenflags



class OrbitComb:

    """
    class of orbit combination

    """

    def __init__(self,orbits,epochs,satinfo,cenflags,weighted_cens_by_sys,unweighted_cens_by_sys,weighted_sats,unweighted_sats,clocks=None,sat_metadata=None):

        """
        initialize OrbitComb class

        Input arguments:
            orbits [dict]  : dictionary containing all individual orbit arrays
                             which must be all the same size (Number of
                             observations by 3 for x,y,z)
            epochs [array] : an array of the same length of each of
                             the arrays in orbits
            satinfo [array]: an array of the same length as epochs but with 4
                             columns for constellation ID, PRN, SVN and
                             satellite block
            cenflags [dict]: center flags to indicate whether a center is
                                  weighted or unweighted
            clocks [dict]  : dictionary containing all individual clocks and
                             their standard deviations dircetly from sp3 files
                             of the individual ACs (all the same size: Number
                             of observations by 2 for clock,csdev)
            sat_metadata
                [class 'io_data.SatelliteMetadata'],
                optional                            : an instance of
                                                    input SatelliteMetadata
                                                    class
        Updates:
            self.orbits [dict]
            self.epochs [array]
            self.satinfo [array]
            self.cenflags [dict]
            self.weighted_centers [list]
            self.unweighted_centers [list]
            self.cen_weights [dict]
            self.sat_sigmas [dict]
            self.sat_weights [dict]
            self.exclude_highrms [list of tuples]
            self.unweighted_max_high_satrms [list]
            self.unweighted_high_tra [list of tuples]
            self.exclude_lowcen [list]
            self.clocks [dict]

        """

        # Check the given orbits and assign to attribute
        if not isinstance(orbits,dict):
            logger.error("\nThe given orbits must be a dictionary\n",
                            stack_info=True)
            raise TypeError("orbits is not of type dictionary!")
        if not orbits:
            logger.error("\nThere must be at least on dictionary item in "
                         "orbits\n", stack_info=True)
            raise ValueError("orbits is empty!")
        if not all(isinstance(key,str) for key in orbits.keys()):
            logger.error("\nKeys of orbits dictionary must be strings\n",
                            stack_info=True)
            raise TypeError("orbits keys must be strings!")
        if not all(item.shape == list(orbits.values())[0].shape
                        for item in orbits.values()):
            logger.error("\nArrays in orbits dictionary must be all of the "
                         "same shape\n", stack_info=True)
            raise ValueError("Arrays in orbits dictionary are not of the same "
                             "shape!")
        for item in orbits.values():
            checkutils.check_coords(item,minrows=1)

        self.orbits = orbits

        # Check the given epochs and assign to attribute
        if len(epochs) != len(list(orbits.values())[0]):
            logger.error(f"\nThe given epochs must be of length "
                         f"{len(list(orbits.values())[0])}\nLength of "
                         f"the given epochs is {len(epochs)}\n",
                         stack_info=True)
            raise ValueError(f"The given epochs must be of length "
                             f"{len(list(orbits.values())[0])}")
        if not all(isinstance(item,datetime.datetime) for item in epochs):
            logger.error("\nThe given epoch can only contain "
                         "datetime.datetime objects\n", stack_info=True)
            raise TypeError("There are non-datetime items in epochs")

        self.epochs = epochs

        # Check the given satinfo and assign to attribute
        if np.shape(satinfo) != (list(orbits.values())[0].shape[0],4):
            logger.error(f"\nThe given satinfo must be a "
                         f"{list(orbits.values())[0].shape[0]} by 4 "
                         f"array\nShape of the given satinfo: "
                         f"{np.shape(satinfo)}\n", stack_info=True)
            raise ValueError(f"The given satinfo must be a "
                             f"{list(orbits.values())[0].shape[0]} by 4 "
                             f"array")
        for row in satinfo:
            if not isinstance(row[0],str):
                logger.error(f"\nThe first column of satinfo "
                             f"(constellation ID) must be strings\n",
                             stack_info=True)
                raise TypeError(f"The first column of satinfo "
                                f"(constellation ID) must be strings")
            if not isinstance(row[1],int):
                logger.error(f"\nThe second column of satinfo (PRN number) "
                             f"must be integers\n",stack_info=True)
                raise TypeError(f"The second column of satinfo (PRN "
                                f"number) must be integers")
            if not isinstance(row[2],int) and not np.isnan(row[2]):
                logger.error(f"\nThe third column of satinfo (SVN number) "
                             f"must be integers or nans\n", stack_info=True)
                raise TypeError(f"The third column of satinfo (SVN number "
                                f") must be integers or nans")
            if not isinstance(row[3],str):
                logger.error(f"\nThe fourth column of satinfo (satellite "
                             f"block) must be strings\n", stack_info=True)
                raise TypeError(f"The fourth column of satinfo (satellite "
                                f"block) must be strings")
        if any(np.isnan(svn) for svn in satinfo[:,2]):
            logger.warning("\nThere are unknown SVN numbers in satinfo. Only "
                           "PRN information are used for identifying "
                           "satellites.\n")
        if 'Unknown' in satinfo[:,3]:
            logger.warning("\nThere are unknown satellite blocks in "
                           "satinfo.\n")
        self.satinfo = satinfo

        # Check the given cenflags
        if not isinstance(cenflags,dict):
            logger.error("\nThe given cenflags must be a dictionary\n",
                            stack_info=True)
            raise TypeError("cenflags is not of type dictionary!")
        for acname in orbits:
            if acname not in cenflags:
                logger.error(f"\ncenter {acname} has not been assigned a "
                              "flag\n", stack_info=True)
                raise ValueError(f"center {acname} present in orbits but "
                                  "not in cenflags")
            if cenflags[acname] not in ['weighted','unweighted']:
                logger.error("\ncenflags items can only be either 'weighted' "
                             "or 'unweighted'\n")
                raise ValueError(f"center flag for {acname}: "
                                 f"{cenflags[acname]} not recognized!")
        self.cenflags = cenflags

        self.weighted_cens_by_sys = weighted_cens_by_sys
        self.unweighted_cens_by_sys = unweighted_cens_by_sys
        self.weighted_sats = weighted_sats
        self.unweighted_sats = unweighted_sats

        # Get a list of weighted and unweighted centers
        weighted_centers = []
        unweighted_centers = []
        for acname in self.orbits:
            if self.cenflags[acname] == 'weighted':
                weighted_centers.append(acname)
            elif self.cenflags[acname] == 'unweighted':
                unweighted_centers.append(acname)
            else:
                raise ValueError(f"center flag {self.cenflags[acname]} for "
                                 f"center {acname} not recognised!")
        self.weighted_centers = weighted_centers
        self.unweighted_centers = unweighted_centers

        # Initialize cen_weights
        cen_weights = {}
        for acname in orbits:
            cen_weights[acname] = 1.0
        self.cen_weights = cen_weights

        # Initialize sat_sigmas and sat_weights
        sat_sigmas = {}
        sat_weights = {}
        for row in satinfo:
            system_id = row[0]
            prn = row[1]
            svn = row[2]
            if (system_id,prn,svn) not in sat_sigmas:
                sat_sigmas[system_id,prn,svn] = 1.0
                sat_weights[system_id,prn,svn] = 1.0
        self.sat_sigmas = sat_sigmas
        self.sat_weights = sat_weights

        # Initialize exclusions, and satflags and orbflags by calling flags
        self.exclude_highrms = []
        self.unweighted_max_high_satrms = []
        self.unweighted_high_tra = []
        self.exclude_lowcen = []
        self.flags()

        self.clocks = clocks

        # Check the given sat_metadata
        if sat_metadata is not None:

            if not isinstance(sat_metadata,SatelliteMetadata):
                logger.error("\nsat_metadata must be an instance of "
                             "SatelliteMetadata class\n")
                raise TypeError("sat_metadata must an instance of "
                                "SatelliteMetadata class")

            # Assign the attribute
            self.sat_metadata = sat_metadata

    def flags(self):

        """
        Create orbflags and satflags

        Updates:
            self.orbflags [dict]: orbit data flags for each AC (same shape as
                                  self.orbits)
            self.satflags [dict]: overall satellite flags for each AC (dict
                                  with keys as (ac,sat)
            self.ngood [dict]: number of good (used) data for each satellite
                                for each ac (ac,sat)
        """

        # Get a list of all satellites
        sats = []
        systems = []
        blocks = []
        sats_full = []
        for row in self.satinfo:
            sat = (row[0],row[1],row[2])
            satfull = (row[0],row[1],row[2],row[3])
            sys = row[0]
            blk = row[3]
            if sat not in sats:
                sats.append(sat)
            if sys not in systems:
                systems.append(sys)
            if blk not in blocks:
                blocks.append(blk)
            if satfull not in sats_full:
                sats_full.append(satfull)
        sats.sort()
        systems.sort()
        blocks.sort()
        sats_full.sort()

        logger.debug("sats %s", sats)

        # Initialize
        orbflags = {}
        satflags = {}
        ngood = {}

        logger.debug(f"acnames :  {self.orbits.keys()}")
        # Loop through centers, and fill in orbflags and satflags
        for acname in self.orbits:

            # Initialize dicts needed for inspection on satflags
            missing_val = {}
            missing_sat = {}
            missing_sys = {}
            missing_blk = {}
            unweighted_sys = {}
            for sat in sats:
                missing_val[acname,sat] = False
                missing_sat[acname,sat] = True
            for sys in systems:
                missing_sys[acname,sys] = True
            for blk in blocks:
                missing_blk[acname,blk] = True

            # Initialize orbflags[acname]; default to missing_val
            orbflags[acname] =  np.full_like(self.orbits[acname],'missing_val',
                                                                dtype=object)


            # Loop through all data rows, and determine the flags
            for r,row in enumerate(self.orbits[acname]):

                sat = (self.satinfo[r,0],self.satinfo[r,1],
                                                    self.satinfo[r,2])
                sys = self.satinfo[r,0]
                blk = self.satinfo[r,3]
                prn = sat[0]+str(sat[1]).zfill(2)
                for c in range(0,3):
                    if not np.isnan(row[c]):
                        orbflags[acname][r,c] = 'okay'
                        missing_sat[acname,sat] = False
                        missing_sys[acname,sys] = False
                        missing_blk[acname,blk] = False
                        if (acname in self.unweighted_sats and prn in self.unweighted_sats[acname]):
                                orbflags[acname][r,c] = 'unweighted_sat'
                        if (acname in self.unweighted_cens_by_sys and sys in self.unweighted_cens_by_sys[acname]):
                            orbflags[acname][r,c] = 'unweighted_sys'
                        if (acname,sat) in self.exclude_highrms:
                            orbflags[acname][r,c] = 'excluded_sat'
                        if sat in self.exclude_lowcen:
                            orbflags[acname][r,c] = 'excluded_sat_all'
                    else:
                        missing_val[acname,sat] = True


            # Fill in satflags from the inspections performed
            for i,sat in enumerate(sats):
                sys = sat[0]
                blk = sats_full[i][3]
                prn = sat[0]+str(sat[1]).zfill(2)
                if sat in self.exclude_lowcen:
                    satflags[acname,sat] = 'excluded_sat_all'
                elif missing_sys[acname,sys]:
                    satflags[acname,sat] = 'missing_sys'
                elif missing_blk[acname,blk]:
                    satflags[acname,sat] = 'missing_blk'
                elif missing_sat[acname,sat]:
                    satflags[acname,sat] = 'missing_sat'
                elif (acname,sat) in self.exclude_highrms:
                    satflags[acname,sat] = 'excluded_sat'
                elif (acname in self.unweighted_cens_by_sys
                        and sys in self.unweighted_cens_by_sys[acname]):
                        satflags[acname,sat] = 'unweighted_sys'
                elif (acname in self.unweighted_sats
                        and prn in self.unweighted_sats[acname]):
                        satflags[acname,sat] = 'unweighted_sat'
                elif missing_val[acname,sat]:
                    satflags[acname,sat] = 'missing_val'
                else:
                    satflags[acname,sat] = 'okay'

            # Revise the orbflags so if there is a missing sat, change
            # missing_val flags to missing_sat flags for that satellite
            for r,row in enumerate(self.orbits[acname]):
                sat = (self.satinfo[r,0],self.satinfo[r,1],self.satinfo[r,2])
                if satflags[acname,sat] == 'missing_sys':
                    for c in range(0,3):
                        orbflags[acname][r,c] = 'missing_sys'
                if satflags[acname,sat] == 'missing_blk':
                    for c in range(0,3):
                        orbflags[acname][r,c] = 'missing_blk'
                if satflags[acname,sat] == 'missing_sat':
                    for c in range(0,3):
                        orbflags[acname][r,c] = 'missing_sat'

        # Revise orbflags and satflags to add missing_val_other,
        # missing_sat_other and excluded_sat_other flags
        for acname in orbflags:
            for r,row in enumerate(orbflags[acname]):
                sat = (self.satinfo[r,0],self.satinfo[r,1],self.satinfo[r,2])
                if (acname,sat) not in ngood:
                    ngood[acname,sat] = 0
                for c in range(0,3):
                    if orbflags[acname][r,c] == ['okay']:
                        ac_others = list(self.weighted_centers)
                        if acname in ac_others:
                            ac_others.remove(acname)
                        other_flags = []
                        for ac_other in ac_others:
                            other_flags.append(orbflags[ac_other][r,c])
                        if 'missing_sat' in other_flags:
                            worst_flag = 'missing_sat'
                        elif 'missing_blk' in other_flags:
                            worst_flag = 'missing_blk'
                        elif 'missing_sys' in other_flags:
                            worst_flag = 'missing_sys'
                        elif 'excluded_sat' in other_flags:
                            worst_flag = 'excluded_sat'
                        elif 'missing_val' in other_flags:
                            worst_flag = 'missing_val'
                        else:
                            worst_flag = 'okay'
                        if worst_flag != 'okay':
                            orbflags[acname][r,c] = worst_flag + '_other'
                    if (orbflags[acname][r,c] not in
                        ['missing_val','excluded_sat','missing_sat',
                            'missing_blk','missing_sys','excluded_sat_all']):
                            ngood[acname,sat] += 1

        for acname in orbflags:
            for sat in sats:
                if satflags[acname,sat] in ['okay','missing_val']:
                    ac_others = list(self.weighted_centers)
                    if acname in ac_others:
                        ac_others.remove(acname)
                    other_flags = []
                    for ac_other in ac_others:
                        other_flags.append(satflags[ac_other,sat])
                    if 'missing_sat' in other_flags:
                        worst_flag = 'missing_sat'
                    elif 'missing_blk' in other_flags:
                        worst_flag = 'missing_blk'
                    elif 'missing_sys' in other_flags:
                        worst_flag = 'missing_sys'
                    elif 'excluded_sat' in other_flags:
                        worst_flag = 'excluded_sat'
                    elif 'missing_val' in other_flags:
                        worst_flag = 'missing_val'
                    else:
                        worst_flag = 'okay'
                    if satflags[acname,sat] == 'okay':
                        if worst_flag != 'okay':
                            satflags[acname,sat] = worst_flag + '_other'
                    elif satflags[acname,sat] == 'missing_val':
                        if worst_flag not in ['okay','missing_val']:
                            satflags[acname,sat] = worst_flag + '_other'

        # Update attributes
        self.orbflags = orbflags
        self.satflags = satflags
        self.ngood = ngood


    def transform(self,transformations):

        """
        Transform the orbits using the given transformation parameters

        Keyword arguments:
            transformations [dict] : transformation parameters for each ac

        Updates:
            self.orbits [dict]

        """

        orbits_transformed = {}
        for acname in self.orbits:

            if acname[0:3] in transformations:

                helm = Helmert(helmert=transformations[acname[0:3]],
                               coords0=self.orbits[acname])
                helm.transform()
                orbits_transformed[acname] = helm.coords1
            else:
                orbits_transformed[acname] =self.orbits[acname]

        self.orbits = orbits_transformed


    def weight(self,cen_wht_method='global',sat_wht_method='RMS_L1',
                l2_dx_threshold=1e-8,l2_maxiter=1,bracket_sigscale=3,
                bracket_interval=None,bracket_maxiter=100,
                bisection_precision_level=None,bisection_sigscale=0.1,
                bisection_precision_limits=[1e-15,1e-13],
                bisection_maxiter=100):

        """
        Perform the orbit weighting

        Input arguments:
            cen_wht_method [str], optional              : method for weighting
                                                          the centres
                                   options:
                                      'global', default : weight each centre
                                                          based on the whole
                                                          constellations being
                                                          combined
                                      'by_constellation': weight each centre
                                                          by constellation
                                      'by_block'        : weight each centre by
                                                          satellite block
                                      'by_sat'          : weight each centre by
                                                          satellite
            sat_wht_method [str], optional              : method for weighting
                                                          the satellites
                                   options:
                                      'RMS_L1', default : satellite weights
                                                         from sat-specific RMS
                                                         of a L1-norm solution
                                                         of helmert
                                                         transformation between
                                                         each orbit and the
                                                         mean orbit, averaged
                                                         over the centres
            Other optional arguments (refer to the documentations of the
                                      relevant Helmert class methods for full
                                      descriptions):
            l2_dx_threshold, l2_maxiter                 : dx_threshold and
                                                          l2_maxiter options
                                                          passed to
                                                          Helmert.l2norm
            bracket_interval,
            bracket_sigscale, bracket_maxiter           : interval, sigscale
                                                          and maxiter options
                                                          passed to
                                                          Helmert.bracket
            bisection_precision_level,
            bisection_sigscale,
            bisection_precision_limits,
            bisection_maxiter                           : precision_level,
                                                          sigscale,
                                                          precision_limits, and
                                                          maxiter options
                                                          passed to
                                                          Helmert.bisection

        Updates:
            self.cen_wht_method [str]                   : center weighting
                                                          method
            self.sat_wht_method [str]                   : satellite weighting
                                                          method
            self.cen_weights [dict]                     : centre weights
            self.sat_sigmas [dict]                      : satellite-specific
                                                          sigmas
            self.sat_weights [dict]                     : satellite weights

        """

        # Check the given cen_wht_method and sat_wht_method
        if cen_wht_method not in (['global','by_constellation','by_block',
                                   'by_sat']):
            logger.error(f"\nCentre weighting method {cen_wht_method} not "
                         f"recognized!\n", stack_info=True)
            raise ValueError(f"Centre weighting method {cen_wht_method} not "
                             f"recognized!")

        if cen_wht_method not in ['global','by_constellation','by_block',
                                  'by_sat']:
                logger.warning(f"Center weighting method {cen_wht_method} not "
                               f"implemented yet! Using the default global "
                               f"method.")
                cen_wht_method = 'global'

        if sat_wht_method not in (['RMS_L1']):
            logger.error(f"\nSatellite weighting method {sat_wht_method} not "
                         f"recognized!\n", stack_info=True)
            raise ValueError(f"Satellite weighting method {sat_wht_method} "
                             f"not recognized!")

        if sat_wht_method != 'RMS_L1':
            logger.warning(f"Satellite weighting method {sat_wht_method} not "
                        f"implemented yet! Using the default L1_norm method.")
            sat_wht_method = 'L1_norm'

        # Take a simple mean of the orbits from weighted centres
        orbits_tuple = ()
        masks_tuple = ()
        logger.debug(f"weighted: {self.weighted_centers}")
        logger.debug(f"unweighted: {self.unweighted_centers}")
        for acname in self.weighted_centers:

            # Exclude if sat/epoch data is missing/excluded for this center
            okay_rows = np.where(
                    (self.orbflags[acname]!='missing_val').all(axis=1) &
                    (self.orbflags[acname]!='unweighted_sys').all(axis=1) &
                    (self.orbflags[acname]!='unweighted_sat').all(axis=1) &
                    (self.orbflags[acname]!='excluded_sat').all(axis=1) &
                    (self.orbflags[acname]!='excluded_sat_all').all(axis=1) &
                    (self.orbflags[acname]!='missing_sys').all(axis=1) &
                    (self.orbflags[acname]!='missing_blk').all(axis=1) &
                    (self.orbflags[acname]!='missing_sat').all(axis=1) )[0]
            mask = np.full_like(self.orbflags[acname],True,dtype=bool)
            mask[okay_rows,:] = False
            orbits_tuple = orbits_tuple + (self.orbits[acname],)
            masks_tuple = masks_tuple + (mask,)
        orbits_masked = np.ma.masked_array(orbits_tuple,masks_tuple)
        orbitmean = np.ma.average(orbits_masked,axis=0)
        orbitmean = orbitmean.filled(np.nan)
        logger.debug(f"orbits_masked")
        logger.debug(f"orbits_masked: {orbits_masked} {np.shape(orbits_masked)}")
        logger.debug(f"orbitmean: {orbitmean} {np.shape(orbitmean)}")
        logger.debug(f"orbitmean first epoch: {orbitmean[0:26,:]}")

        # Special case of only one weighted orbit (used for comparisons)
        if len(self.weighted_centers) == 1:
            self.cen_wht_method = 'global'
            cen_weights = {}
            for acname in self.weighted_centers:
                cen_weights[acname] = 1.0
            self.cen_weights = cen_weights

        else:

            # Loop through centres, estimate helmert parameters, and calculate weights
            cen_weights = {}
            satsig2 = {} # satellite-specific sigma^2s for each AC
            for acname in self.orbits:

                # weighted_center flag
                if acname in self.weighted_centers:
                    weighted_center = True
                elif acname in self.unweighted_centers:
                    weighted_center = False
                else:
                    raise ValueError(f"center {acname} not in weighted_centers"
                                      " nor in unweighted_centers")

                logger.debug(f"orbits {acname} {self.orbits[acname]} "
                            f"{np.shape(self.orbits[acname])}")
                logger.debug(f"orbflags {acname} {self.orbflags[acname]}")

                # Exclude if sat/epoch data is missing for this center
                okay_rows = np.where(
                        (self.orbflags[acname]!='missing_val').all(axis=1) &
                        (self.orbflags[acname]!='missing_sys').all(axis=1) &
                        (self.orbflags[acname]!='missing_blk').all(axis=1) &
                        (self.orbflags[acname]!='missing_sat').all(axis=1) )[0]
                mask = np.full_like(self.orbflags[acname],True,dtype=bool)
                mask[okay_rows,:] = False
                orbit_masked = np.ma.masked_array(self.orbits[acname],mask)

                logger.debug(f"orbit_masked {acname}: {orbit_masked}")

                # Create an instance of Helmert class between the centre orbit
                # and the mean orbit 
                helm = Helmert(coords0=orbit_masked,coords1=orbitmean,
                        satinfo=self.satinfo,orbflags=self.orbflags[acname],
                        weighted_center=weighted_center,acname=acname)

                # Perform L2 norm (to get the a priori helmert parameters)
                helm.l2norm(dx_threshold=l2_dx_threshold,maxiter=l2_maxiter)

                # Perform L1 norm solution
                helm.bracket(interval=bracket_interval,
                            sigscale=bracket_sigscale,maxiter=bracket_maxiter)
                helm.bisection(precision_level=bisection_precision_level,
                           sigscale=bisection_sigscale,
                           precision_limits=bisection_precision_limits,
                           maxiter=bisection_maxiter)

                # Calculate center weight
                if acname in self.weighted_centers:
                    if cen_wht_method == 'global':
                        logger.debug(f"abdev {acname}: {helm.abdev}")
                        cen_weights[acname] = 1/helm.abdev_wht**2
                        logger.debug(f"cen_weight {acname}: "
                                     f"{cen_weights[acname]}")
                    elif cen_wht_method == 'by_constellation':
                        for sys_id in helm.sys_abdev.keys():
                            if sys_id in self.weighted_cens_by_sys[acname]:
                                cen_weights[acname,sys_id] = (
                                        1/helm.sys_abdev[sys_id]**2)
                    elif cen_wht_method == 'by_block':
                        for block in helm.blk_abdev.keys():
                            cen_weights[acname,block] = (1/
                                                    helm.blk_abdev[block]**2)
                    elif cen_wht_method == 'by_sat':
                        for (sys_id,prn,svn) in helm.sat_abdev.keys():
                            sat = (sys_id,prn,svn)
                            if (self.satflags[acname,sat] != 'excluded_sat'
                                    and self.satflags[acname,sat] != 'unweighted_sys'
                                    and self.satflags[acname,sat] != 'unweighted_sat'):
                                cen_weights[acname,sys_id,prn,svn] = (
                                        1/helm.sat_abdev[sys_id,prn,svn]**2)

                # Caclulate satellite-specific sigma^2's
                logger.debug(f"sat_rms {acname}: {helm.sat_rms}")
                for (system_id,prn,svn) in helm.sat_rms:

                    # Only include data if the center is weighted, and the
                    # satellite is not missing/excluded from any weighted center
                    sat = (system_id,prn,svn)
                    logger.debug(f"acname, sat: {acname} {sat}")
                    logger.debug(f"satflags: {self.satflags[acname,sat]}")
                    logger.debug(f"weihgted_centers: {self.weighted_centers}")
                    if (acname in self.weighted_centers
                            and self.satflags[acname,sat] in
                                ['okay','missing_val_other','missing_val']):
                        if sat_wht_method == 'RMS_L1':
                            if (system_id,prn,svn) not in satsig2:
                                satsig2[system_id,prn,svn] = {}
                                satsig2[system_id,prn,svn][acname] = (
                                            helm.sat_rms[system_id,prn,svn]**2)
                            else:
                                satsig2[system_id,prn,svn][acname] = (
                                            helm.sat_rms[system_id,prn,svn]**2)

            # End of centers loop

            logger.debug(f"satsig2: {satsig2}")
            # Take the average of satellite sigmas over all the weighted centres
            sat_sigmas = {}
            sat_weights = {}
            for (system_id,prn,svn) in satsig2.keys():
                sat_sigmas[system_id,prn,svn] = np.sqrt(
                            np.mean(list(satsig2[system_id,prn,svn].values())))

                # the legacy software way of doing this
                if old_version is True:
                    sat_sigmas[system_id,prn,svn] = np.sqrt(
                            np.mean(list(satsig2[system_id,prn,svn].values()))
                            + 1/len(list(satsig2[system_id,prn,svn].values())))

                # Satellite weights (for statistics only)
                sat_weights[system_id,prn,svn] = (
                            1/sat_sigmas[system_id,prn,svn]**2)

            # Update attributes
            self.cen_weights = cen_weights
            self.sat_sigmas = sat_sigmas
            self.sat_weights = sat_weights
            self.cen_wht_method = cen_wht_method

        self.sat_wht_method = sat_wht_method

        logger.debug(f"cen_weights: {self.cen_weights}")
        logger.debug(f"sat_sigmas: {self.sat_sigmas}")
        logger.debug(f"sat_weights: {self.sat_weights}")


    def combine(self,l2_dx_threshold=1e-8,l2_maxiter=1,bracket_sigscale=3,
                bracket_interval=None,bracket_maxiter=100,
                bisection_precision_level=None,bisection_sigscale=0.1,
                bisection_precision_limits=[1e-15,1e-13],
                bisection_maxiter=100):

        """
        Perform the orbit weighting and combination

            Optional arguments (refer to the documentations of the relevant
                                Helmert class methods for full descriptions):
            l2_dx_threshold, l2_maxiter      : dx_threshold and l2_maxiter
                                               options passed to Helmert.l2norm
            bracket_interval,
            bracket_sigscale, bracket_maxiter: interval, sigscale and maxiter
                                               options passed to
                                               Helmert.bracket
            bisection_precision_level,
            bisection_sigscale,
            bisection_precision_limits,
            bisection_maxiter                : precision_level, sigscale,
                                               precision_limits, and maxiter
                                               options passed to
                                               Helmert.bisection

        Updates:
            self.combined_orbit [array]      : combined orbit
            self.cen_rms [dict]              : RMS of centers
            self.cen_abdev [dict]            : absolute deviation of centres
            self.sat_rms [dict]              : overal sat-specific rms's
            self.sat_abdev [dict]            : overal sat-specific abdev's

        """

        # Take a weighted mean of the orbits from weighted centres
        orbits_tuple = ()
        masks_tuple = ()
        for acname in self.weighted_centers:

            # Exclude if sat/epoch data is missing/excluded/unweighted for this center
            okay_rows = np.where(
                    (self.orbflags[acname]!='missing_val').all(axis=1) &
                    (self.orbflags[acname]!='unweighted_sys').all(axis=1) &
                    (self.orbflags[acname]!='unweighted_sat').all(axis=1) &
                    (self.orbflags[acname]!='excluded_sat').all(axis=1) &
                    (self.orbflags[acname]!='excluded_sat_all').all(axis=1) &
                    (self.orbflags[acname]!='missing_sys').all(axis=1) &
                    (self.orbflags[acname]!='missing_blk').all(axis=1) &
                    (self.orbflags[acname]!='missing_sat').all(axis=1) )[0]
            mask = np.full_like(self.orbflags[acname],True,dtype=bool)
            mask[okay_rows,:] = False
            orbits_tuple = orbits_tuple + (self.orbits[acname],)
            masks_tuple = masks_tuple + (mask,)
        orbits_masked = np.ma.masked_array(orbits_tuple,masks_tuple)
        logger.debug(f"orbits_masked: {orbits_masked}")

        if self.cen_wht_method == 'global':

            wcen = []
            for acname in self.weighted_centers:
                wcen.append(self.cen_weights[acname])
            orbitmean = np.ma.average(orbits_masked,axis=0,weights=wcen)

        elif self.cen_wht_method == 'by_constellation':

            m = np.shape(orbits_masked)[1]
            orbitmean = np.ma.array(np.full((m,3),np.nan))
            for c,row in enumerate(self.satinfo):
                wcen = []
                sys_id = row[0]
                for acname in self.weighted_centers:
                    if (acname,sys_id) in self.cen_weights.keys():
                        wcen.append(self.cen_weights[acname,sys_id])
                    else:
                        wcen.append(0)
                orbitmean[c] = np.ma.average(
                                        orbits_masked[:,c],axis=0,weights=wcen)

        elif self.cen_wht_method == 'by_block':

            m = np.shape(orbits_masked)[1]
            orbitmean = np.ma.array(np.full((m,3),np.nan))
            for c,row in enumerate(self.satinfo):
                wcen = []
                block = row[3]
                for acname in self.weighted_centers:
                    if (acname,block) in self.cen_weights.keys():
                        wcen.append(self.cen_weights[acname,block])
                    else:
                        wcen.append(0)
                orbitmean[c] = np.ma.average(
                                        orbits_masked[:,c],axis=0,weights=wcen)

        elif self.cen_wht_method == 'by_sat':

            m = np.shape(orbits_masked)[1]
            orbitmean = np.ma.array(np.full((m,3),np.nan))
            for c,row in enumerate(self.satinfo):
                wcen = []
                sys_id = row[0]
                prn = row[1]
                svn = row[2]
                for acname in self.weighted_centers:
                    if (acname,sys_id,prn,svn) in self.cen_weights.keys():
                        wcen.append(self.cen_weights[acname,sys_id,prn,svn])
                    else:
                        wcen.append(0)
                orbitmean[c] = np.ma.average(
                                        orbits_masked[:,c],axis=0,weights=wcen)

        orbitmean = orbitmean.filled(np.nan)
        logger.debug(f"orbitmean: {orbitmean}")



        # Initialize transformed orbits
        transformed_orbits = {}
        transform_params = {}

        # Initialize center rms and abdev
        cen_rms = {}
        cen_rms_wht = {}
        cen_abdev = {}
        cen_abdev_wht = {}

        # Loop through centres and estimate helmert parameters
        for acname in self.orbits:

            # weighted_center flag
            if acname in self.weighted_centers:
                weighted_center = True
            elif acname in self.unweighted_centers:
                weighted_center = False
            else:
                raise ValueError(f"center {acname} not in weighted_centers nor"
                                  "in unweighted_centers")

            # using satellite sigmas, create sigmas for weighted least-squares
            sigmas = np.ones_like(self.orbits[acname])
            logger.debug(f"sat_sigmas: {self.sat_sigmas}")
            for r in range(len(self.orbits[acname])):
                system_id = self.satinfo[r,0]
                prn = self.satinfo[r,1]
                svn = self.satinfo[r,2]
                if (system_id,prn,svn) in self.sat_sigmas:
                    sigmas[r,0] = self.sat_sigmas[system_id,prn,svn]
                    sigmas[r,1] = self.sat_sigmas[system_id,prn,svn]
                    sigmas[r,2] = self.sat_sigmas[system_id,prn,svn]
                else:
                    logger.debug(f"({system_id},{prn},{svn}) not in "
                                  "sat_sigmas. The satellite is likely "
                                  "missing from at least one center. Setting"
                                   " the sigma as 1.")

            # Exclude if sat/epoch data is missing for this center
            okay_rows = np.where(
                    (self.orbflags[acname]!='missing_val').all(axis=1) &
                    (self.orbflags[acname]!='missing_sys').all(axis=1) &
                    (self.orbflags[acname]!='missing_blk').all(axis=1) &
                    (self.orbflags[acname]!='missing_sat').all(axis=1) &
                    (self.orbflags[acname]!='excluded_sat_all').all(axis=1))[0]
            mask = np.full_like(self.orbflags[acname],True,dtype=bool)
            mask[okay_rows,:] = False
            orbit_masked = np.ma.masked_array(self.orbits[acname],mask)

            # Create an instance of Helmert class between the centre orbit and
            # the mean orbit 
            helm = Helmert(coords0=orbit_masked,coords1=orbitmean,
                           sigmas1=sigmas,satinfo=self.satinfo,
                           orbflags=self.orbflags[acname],
                           weighted_center=weighted_center)

            # Perform L2 norm (to get the a priori helmert parameters)
            helm.l2norm(dx_threshold=l2_dx_threshold,maxiter=l2_maxiter)

            # Reset sigmas (and hence satellite weights) to one
            helm.sigmas1 = np.ones_like(sigmas)
            helm.sigmas1_flt = np.ones_like(helm.sigmas1_flt)

            # Perform L1 norm solution

            # Special case of only one weighted center; avoid unnecessary
            # iterations for the weighted center
            if (len(self.weighted_centers) == 1 and weighted_center):
                helm.bracket(interval=bracket_interval,
                        sigscale=bracket_sigscale,maxiter=1)
                helm.bisection(precision_level=bisection_precision_level,
                            sigscale=bisection_sigscale,
                            precision_limits=bisection_precision_limits,
                            maxiter=1)
            else:

                helm.bracket(interval=bracket_interval,
                        sigscale=bracket_sigscale,maxiter=bracket_maxiter)
                helm.bisection(precision_level=bisection_precision_level,
                               sigscale=bisection_sigscale,
                               precision_limits=bisection_precision_limits,
                               maxiter=bisection_maxiter)

            # Store rms and abdev statistics
            cen_rms[acname] = helm.rms
            cen_abdev[acname] = helm.abdev
            cen_abdev_wht[acname] = helm.abdev_wht
            for key in helm.sys_rms:
                cen_rms[acname,key] = helm.sys_rms[key]
                cen_abdev[acname,key] = helm.sys_abdev[key]
            for key in helm.blk_rms:
                cen_rms[acname,key] = helm.blk_rms[key]
                cen_abdev[acname,key] = helm.blk_abdev[key]
            for key in helm.sat_rms:
                cen_rms[acname,key] = helm.sat_rms[key]
                cen_abdev[acname,key] = helm.sat_abdev[key]

            # Perform forward transform for the center
            helm.transform()
            transformed_orbits[acname] = helm.coords1

            # Get the helmert parameters;
            # convert rotations from radians into arc seconds
            # convert scale to part-per-million deviation from 1
            tr_pars = np.zeros(7)
            for i in  range(0,3):
                tr_pars[i] = helm.helmert[i]
            for i in range(3,6):
                tr_pars[i] = helm.helmert[i]*180.0*3600.0/np.pi
            tr_pars[6] = (helm.helmert[6]-1.0)*1e6
            transform_params[acname] = tr_pars
            logger.debug(f"transformed_orbits[{acname}]: "
                         f"{transformed_orbits[acname]}")
            logger.debug(f"helmert parameters {acname}: {helm.helmert}")

            # End of centers loop

        # Calculate satellite-specific rms over all the centers
        # This will be the weighted mean of the satellite-specific rms over
        # the centers with weights being 1/cen_abdev**2

        # Create a list of all the satellites
        sats = []
        for key in cen_rms:
            if not isinstance(key,str):
                if isinstance(key[1],tuple):
                    if key[1] not in sats:
                        sats.append(key[1])

        # Loop over satellites and calculate sat-specific rms (sat_rms)
        # and satellite accuracy exponent codes (sat_accuracy)
        logger.debug(f"center {acname}")
        logger.debug(f"cen_rms: {cen_rms}")
        logger.debug(f"cen_abdev: {cen_abdev}")
        sat_rms = {}
        sat_accuracy = {}
        rms_all_sats = []
        rms_sats_by_sys = {}
        for sat in sats:
            wcen = []
            rmscen = []
            sys = sat[0]
            svn = sat[2]
            sat_id = self.sat_metadata.get_sat_identifier(sys,svn)
            blk = sat_id.block
            ng = 0
            if sys not in rms_sats_by_sys:
                rms_sats_by_sys[sys] = []
            for acname in self.weighted_centers:
                if ( (acname,sat) in cen_rms
                     and self.satflags[acname,sat] != 'excluded_sat'
                     and self.satflags[acname,sat] != 'unweighted_sys'
                     and self.satflags[acname,sat] != 'unweighted_sat'):
                    if self.cen_wht_method == 'global':
                        wcen.append(1/cen_abdev_wht[acname]**2)
                    elif self.cen_wht_method == 'by_constellation':
                        wcen.append(1/cen_abdev[acname,sys]**2)
                    elif self.cen_wht_method == 'by_block':
                        wcen.append(1/cen_abdev[acname,blk]**2)
                    elif self.cen_wht_method == 'by_sat':
                        wcen.append(1/cen_abdev[acname,sat]**2)
                    rmscen.append(cen_rms[acname,sat]**2)
                    ng += self.ngood[acname,sat]

                # For outlier removal, we now have missing_sys and missing_sys_other
                # so if a constellation is fully missing, the satellites will get a
                # total rms
                if ( (acname,sat) in cen_rms
                     and self.satflags[acname,sat] not in
                        ['excluded_sat','missing_sat_other','missing_blk_other',
                            'unweighted_sys','unweighted_sat']):
                    rms_all_sats.append(cen_rms[acname,sat]**2)
                    rms_sats_by_sys[sys].append(cen_rms[acname,sat]**2)

            # check if the satellite data comes from at least two centers
            # with at least 90 percent overlap of x,y,z data;
            # if not, set the satellite accuracy code 0 as unknown
            if (len(rmscen) <= 1 or ng < 0.9*2*3*len(set(self.epochs))):
                sat_rms[sat] = 0.0
                sat_accuracy[sat] = 0
            else:
                sat_rms[sat] = np.sqrt(
                    np.ma.average(rmscen,weights=wcen)/(len(rmscen)-1))
                sat_accuracy[sat] = round(np.log2(1000.0*sat_rms[sat]))

        sat_rms_mean = np.sqrt(np.mean(rms_all_sats))
        sat_rms_sys = {}
        for sys in rms_sats_by_sys:
            sat_rms_sys[sys] = np.sqrt(np.mean(rms_sats_by_sys[sys]))
        logger.debug(f"rms_sats_by_sys: {rms_sats_by_sys}")
        logger.debug(f"sat_rms: {sat_rms}")
        logger.debug(f"sat_rms_mean: {len(rms_all_sats)} {sat_rms_mean}")
        logger.debug(f"sat_rms_sys: {sat_rms_sys}")

        # Loop over satellites and calculate sat-specific abdev
        sat_abdev = {}
        for sat in sats:
            abdevcen = []
            for acname in self.weighted_centers:
                if ( (acname,sat) in cen_abdev
                    and self.satflags[acname,sat] != 'excluded_sat'
                    and self.satflags[acname,sat] != 'unweighted_sys'
                    and self.satflags[acname,sat] != 'unweighted_sat'):
                    abdevcen.append(cen_abdev[acname,sat])
            if len(abdevcen) <= 1:
                sat_abdev[sat] = 0.0
            else:
                sat_abdev[sat] = np.ma.average(abdevcen)
        logger.debug(f"sat_abdev: {sat_abdev}")

        orbits_tuple = ()
        masks_tuple = ()
        for acname in self.weighted_centers:

            # Exclude if sat/epoch data is missing/excluded/unweighted for this center
            okay_rows = np.where(
                    (self.orbflags[acname]!='missing_val').all(axis=1) &
                    (self.orbflags[acname]!='unweighted_sys').all(axis=1) &
                    (self.orbflags[acname]!='unweighted_sat').all(axis=1) &
                    (self.orbflags[acname]!='excluded_sat').all(axis=1) &
                    (self.orbflags[acname]!='missing_sys').all(axis=1) &
                    (self.orbflags[acname]!='missing_blk').all(axis=1) &
                    (self.orbflags[acname]!='missing_sat').all(axis=1) )[0]
            logger.debug(f"okay_rows {acname} {np.shape(okay_rows)}")
            mask = np.full_like(self.orbflags[acname],True,dtype=bool)
            mask[okay_rows,:] = False
            orbits_tuple = orbits_tuple + (transformed_orbits[acname],)
            masks_tuple = masks_tuple + (mask,)
        orbits_masked = np.ma.masked_array(orbits_tuple,masks_tuple)

        if self.cen_wht_method == 'global':

            wcen = []
            for acname in self.weighted_centers:
                wcen.append(self.cen_weights[acname])
            orbitmean = np.ma.average(orbits_masked,axis=0,weights=wcen)
            m = np.shape(orbits_masked)[1]
            sdev = np.ma.array(np.full((m,3),np.nan))
            for c,row in enumerate(self.satinfo):
                n = (orbits_masked[:,c].count())/3
                if n==1:
                    for j in range(0,3):
                        sdev[c,j] = 0
                else:
                    sdev[c] = np.sqrt(
                            np.ma.average((orbits_masked[:,c]-orbitmean[c])**2,
                            axis=0,weights=wcen)/(n-1))

        elif self.cen_wht_method == 'by_constellation':

            m = np.shape(orbits_masked)[1]
            orbitmean = np.ma.array(np.full((m,3),np.nan))
            sdev = np.ma.array(np.full((m,3),np.nan))
            for c,row in enumerate(self.satinfo):
                wcen = []
                sys_id = row[0]
                n = 0
                for acname in self.weighted_centers:
                    if (acname,sys_id) in self.cen_weights.keys():
                        wcen.append(self.cen_weights[acname,sys_id])
                        n += 1
                    else:
                        wcen.append(0)
                orbitmean[c] = np.ma.average(
                                        orbits_masked[:,c],axis=0,weights=wcen)
                if n==1:
                    for j in range(0,3):
                        sdev[c,j] = 0
                else:
                    sdev[c] = np.sqrt(
                            np.ma.average((orbits_masked[:,c]-orbitmean[c])**2,
                            axis=0,weights=wcen)/(n-1))

        elif self.cen_wht_method == 'by_block':

            m = np.shape(orbits_masked)[1]
            orbitmean = np.ma.array(np.full((m,3),np.nan))
            sdev = np.ma.array(np.full((m,3),np.nan))
            for c,row in enumerate(self.satinfo):
                wcen = []
                block = row[3]
                n = 0
                for acname in self.weighted_centers:
                    if (acname,block) in self.cen_weights.keys():
                        wcen.append(self.cen_weights[acname,block])
                        n += 1
                    else:
                        wcen.append(0)
                orbitmean[c] = np.ma.average(
                                        orbits_masked[:,c],axis=0,weights=wcen)
                if n==1:
                    for j in range(0,3):
                        sdev[c,j] = 0
                else:
                    sdev[c] = np.sqrt(
                            np.ma.average((orbits_masked[:,c]-orbitmean[c])**2,
                            axis=0,weights=wcen)/(n-1))

        elif self.cen_wht_method == 'by_sat':

            m = np.shape(orbits_masked)[1]
            orbitmean = np.ma.array(np.full((m,3),np.nan))
            sdev = np.ma.array(np.full((m,3),np.nan))
            for c,row in enumerate(self.satinfo):
                wcen = []
                sys_id = row[0]
                prn = row[1]
                svn = row[2]
                n =0
                for acname in self.weighted_centers:
                    if (acname,sys_id,prn,svn) in self.cen_weights.keys():
                        wcen.append(self.cen_weights[acname,sys_id,prn,svn])
                        n += 1
                    else:
                        wcen.append(0)
                orbitmean[c] = np.ma.average(
                                        orbits_masked[:,c],axis=0,weights=wcen)
                if n==1:
                    for j in range(0,3):
                        sdev[c,j] = 0
                else:
                    sdev[c] = np.sqrt(
                            np.ma.average((orbits_masked[:,c]-orbitmean[c])**2,
                            axis=0,weights=wcen)/(n-1))

        orbitmean = orbitmean.filled(np.nan)
        sdev = sdev.filled(np.nan)

        # Update attributes
        self.combined_orbit = orbitmean
        self.sdev = sdev
        self.cen_rms = cen_rms
        self.cen_abdev = cen_abdev
        self.cen_abdev_wht = cen_abdev_wht
        self.sat_rms = sat_rms
        self.sat_accuracy = sat_accuracy
        self.sat_abdev = sat_abdev
        self.sat_rms_mean = sat_rms_mean
        self.sat_rms_sys = sat_rms_sys
        self.transform_params = transform_params


    def assess(self,sat_rms_tst=None,sat_rms_tst_unweighted=None,coef_sat=470.0,
                    thresh_sat=None,max_high_satrms=None,
                    trn_tst=None,thresh_trn=[None,None,None],
                    numcen_tst=None,min_numcen=None):
        """
        Assess the quality of combination results against the given thresholds

        Keyword arguments:
            sat_rms_tst [str] : option for testing of satellite rms
                    options: 'auto' : thresholds set by multiplying a
                                      coefficient by the mean rms over each
                                      constellation
                             'manual' : thresholds set by user
                             'strict' : thresholds will be lower of user-
                                        defined and automatic
                             None     : No satellite rms test
            sat_rms_tst_unweighted [str] : similar to sat_rms_tst but for
                                            unweighted centers
            coef_sat [float] : coefficient for sat rms auto approach
            thresh_sat [dict]  : threshold for satellite rms (cm) for each
                                 satellite system for manual approach
            max_high_satrms [int] : maximum number of high-rms satellites for
                                    a center

            trn_tst [str] : option for testing of transformation parameters
                   options: 'auto'      : thresholds automatically set
                            'manual'    : thresholds set by user
                            'strict' : thresholds will be lower of user-
                                          defined and automatic
                            None        : No transformation test
            thresh_trn [list] : transformation thresholds for translation (mm),
                                rotation (mas) and scale (ppb)

            numcen_tst [str] : option for testing of minimum number of centers
                               for each satellite
                      options: 'strict' : minimum number set by user
                               'eased'  : minimum number set by user but may
                                          be eased depending on the number of
                                          centers
                                None    : No test for minimum number of centers
            min_numcen [int] : minimum number of centers for each satellite

        Updates:
            self.rejection [bool] : whether rejections are found or not
                                    (True/False)
            self.exclude_highrms [list of tuples] : list of cen/sat exclusions
                                                    due to high rms
            self.unweighted_max_high_satrms [list] : list of centers unweighted
                                                    due to high number of sats
                                                    excluded because of high rms
            self.unweighted_high_tra [listof tuples] : list of centers unweighted due to
                                                        high transformation parameters
                                                        along with the index and value
                                                        of the outlier transformation
                                                        parameter
            self.exclude_lowcen [list] : list of sat exclusions
                                            due to low number of
                                            centers contributing
        """

        # Check the given arguments
        if sat_rms_tst is not None:
            allowed_sat_rms_tst = ['auto','manual','strict']
            if sat_rms_tst not in allowed_sat_rms_tst:
                logger.error("\nsat_rms_tst must be one of "
                             f"{allowed_sat_rms_tst}\n", stack_info=True)
                raise ValueError(f"The given sat_rms_tst {sat_rms_tst} is not in "
                                 f"{allowed_sat_rms_tst}")

        if sat_rms_tst_unweighted is not None:
            allowed_sat_rms_tst = ['auto','manual','strict']
            if sat_rms_tst_unweighted not in allowed_sat_rms_tst:
                logger.error("\nsat_rms_tst_unweighted must be one of "
                             f"{allowed_sat_rms_tst}\n", stack_info=True)
                raise ValueError(f"The given sat_rms_tst_unweighted {sat_rms_tst_unweighted} is not in "
                                 f"{allowed_sat_rms_tst}")

        checkutils.check_scalar(coef_sat)

        if len(self.weighted_centers) == 1:
            logger.info("One weighted center only; sat_rms_tst set to manual")
            sat_rms_tst = 'manual'

        if sat_rms_tst == 'manual' and thresh_sat is None:
            logger.error("\nthresh_sat must be specified when sat_rms_tst is "
                         "manual\n", stack_info=True)
            raise ValueError("sat_rms_tst is manual but thresh_sat not "
                             "specified!")

        if thresh_sat is not None:
            if not isinstance(thresh_sat,dict):
                logger.error("\nThe given thresh_sat must be a dict\n",
                                stack_info=True)
                raise TypeError("The given thresh_sat is not of type dict")
            for key in thresh_sat:
                if not isinstance(key,str):
                    logger.error("\nThe keys in the given thresh_sat must be "
                                 "of type str\n",stack_info=True)
                    raise TypeError("There are non-str keys in the given "
                                    "thresh_sat")
                checkutils.check_scalar(thresh_sat[key])

        if max_high_satrms is not None:
            if not isinstance(max_high_satrms,int):
                logger.error("\nThe given max_high_satrms must be an "
                             "integer\n", stack_info=True)
                raise TypeError("The given max_high_satrms is not of type int")
        self.max_high_satrms = max_high_satrms

        if trn_tst is not None:
            allowed_trn_tst = ['auto','manual','strict']
            if trn_tst not in allowed_trn_tst:
                logger.error(f"\ntrn_tst must be one of {allowed_trn_tst}\n",
                                stack_info=True)
                raise ValueError("The given trn_tst is not in "
                                 "{allowed_trn_tst}")

        if not isinstance(thresh_trn,list):
            logger.error("\nthresh_trn must be a list of three items for "
                         "translation, rotation and scale thresholds\n",
                         stack_info=True)
            raise TypeError("The given thresh_trn is not of type list")
        if len(thresh_trn) != 3:
            logger.error("\nthresh_trn must be a list of three items for "
                         "translation, rotation and scale thresholds\n",
                         stack_info=True)
            raise TypeError(f"The given thresh_trn is of length "
                            f"{len(thresh_trn)}")
        for item in thresh_trn:
            if item is not None:
                checkutils.check_scalar(item)

        if numcen_tst is not None:
            allowed_numcen_tst = ['strict','eased']
            if numcen_tst not in allowed_numcen_tst:
                logger.error("\nnumcen_tst must be one of "
                             f"{allowed_numcen_tst}\n", stack_info=True)
                raise ValueError("The given numcen_tst is not in "
                                 "{allowed_numcen_tst}")

        if min_numcen is not None:
            if not isinstance(min_numcen,int):
                logger.error("\nThe given min_numcen must be an "
                             "integer\n", stack_info=True)
                raise TypeError("The given min_numcen is not of type int")

        # Set the default rejection to False
        rejection = False

        # Print out transformation values
        logger.debug("Transformation values (mm, mas, ppb)")
        for acname in self.transform_params:
            logger.debug(f"{acname}: "
                f"{[item*1000.0 for item in self.transform_params[acname]]}")

        # Get a list of all centers
        centers_str = []
        for ac in self.weighted_centers:
            centers_str.append(ac + ' ')
        centers_str = centers_str + self.unweighted_centers

        # Create a list of all satellites
        sats = []
        for key in self.cen_rms:
            if not isinstance(key,str):
                if isinstance(key[1],tuple):
                    if key[1] not in sats:
                        sats.append(key[1])

        # Print out satellite rms statistics
        logger.info("Sat-specific rms statistics (cm)")
        logger.info(f" PRN |  SVN |  "
                    f"{' |  '.join([ac for ac in centers_str])} |  IGS")
        logger.info(f"------------{'--------'*(len(centers_str)+1)}")
        for sat in sats:
            rms = []
            for acname in self.weighted_centers + self.unweighted_centers:
                if (acname,sat) in self.cen_rms:
                    rms_str = f"{self.cen_rms[acname,sat]*100:^5.2f}"
                else:
                    rms_str = "     "
                rms.append(rms_str)
            if sat in self.sat_rms:
                rms_str = f"{self.sat_rms[sat]*100:^5.2f}"
            else:
                rms_str = "     "
            rms.append(rms_str)
            prn = sat[0] + str(sat[1]).zfill(2)
            svn = sat[0] + str(sat[2]).zfill(3)
            logger.info(f" {prn} | {svn} | "
                        f"{' | '.join([item for item in rms])}")
        logger.info("\n")

        # Print out satellite flags
        logger.info("Satellite flags")
        logger.info(f" PRN |  SVN | "
                    f"{' | '.join([ac.center(19) for ac in centers_str])}")
        logger.info(f"------------"
                    f"{'----------------------'*(len(centers_str))}")
        for sat in sats:
            prn = sat[0] + str(sat[1]).zfill(2)
            svn = sat[0] + str(sat[2]).zfill(3)
            flags = []
            for acname in self.weighted_centers + self.unweighted_centers:
                if (acname,sat) in self.satflags:
                    flag = self.satflags[acname,sat].center(19)
                else:
                    flag = "                  "
                flags.append(flag)
            logger.info(f" {prn} | {svn} | "
                        f"{' | '.join([item for item in flags])}")
        logger.info("\n")


        logger.debug(f"thresh_sat={thresh_sat}, thresh_trn={thresh_trn}, "
                    f"min_numcen={min_numcen}")

        # Test 1 - sat-specific rms (remove one sat/cen pair at once)

        # Only do the test if requested, and any previous tests have been 
        # successfully passed
        if (sat_rms_tst is not None and not rejection):

            # Reset outlier thresholds if auto or strict
            thresh_sat_used = {}
            if sat_rms_tst == 'auto':
                for sys in self.sat_rms_sys:
                    thresh_sat_used[sys] = coef_sat*self.sat_rms_sys[sys]
            elif sat_rms_tst == 'strict':
                for sys in self.sat_rms_sys:
                    if (sys not in thresh_sat
                            or thresh_sat[sys] is None):
                        thresh_sat_used[sys] = coef_sat*self.sat_rms_sys[sys]
                    else:
                        thresh_sat_used[sys] = min(thresh_sat[sys]/100.0,
                                                coef_sat*self.sat_rms_sys[sys])
            elif sat_rms_tst == 'manual':
                for sys in self.sat_rms_sys:
                    if (sys in thresh_sat
                            and thresh_sat[sys] is not None):
                        thresh_sat_used[sys] = thresh_sat[sys]/100.0
            else:
                raise ValueError(f"sat_rms_tst {sat_rms_tst} is not recognized!")

            thresh_sat_used_print = {k: float(v) for k, v in thresh_sat_used.items()}
            logger.info(f"thresh_sat_used: {thresh_sat_used_print}")

            # Look only into weighted centers used for combination (not excluded)
            sat_rms = {}
            for acname in self.weighted_centers:
                for sat in sats:
                    sys = sat[0]
                    if sys not in sat_rms:
                        sat_rms[sys] = {}
                    if ( (acname,sat) in self.cen_rms
                            and self.satflags[acname,sat] != 'excluded_sat'
                            and self.satflags[acname,sat] != 'unweighted_sys'
                            and self.satflags[acname,sat] != 'unweighted_sat'):
                        sat_rms[sys][acname,sat] = self.cen_rms[acname,sat]

            # Find the largest outlier rms
            outlier_found = False
            maxrms = 0.0
            for sys in sat_rms:
                if ( sys in thresh_sat_used
                    and max(sat_rms[sys].values()) > thresh_sat_used[sys]/100.0 ):

                    logger.debug(f"max sat_rms weighted: {max(sat_rms[sys].values())*100}")

                    outlier_found = True

                    # Find the highest rms
                    if max(sat_rms[sys].values()) > maxrms:
                        maxrms = max(sat_rms[sys].values())
                        sys_out = sys
                        outlier_censat = max(sat_rms[sys],key=sat_rms[sys].get)

            # If outlier found, proceed to the exclusion
            if outlier_found:

                # There is at least one sat rms larger than the threshold
                rejection = True

                # Add the corresponding cen/sat pair to exclude_highrms
                cen_outlier = outlier_censat[0]
                sat_outlier = outlier_censat[1]
                logger.info("Exclusion due to outlier satellite RMS: "
                            f"{cen_outlier} {sat_outlier}")

                # Update exclude_highrms
                if (cen_outlier,sat_outlier) not in self.exclude_highrms:
                    self.exclude_highrms.append((cen_outlier,sat_outlier))

                # If maximum number of outlier satellites reaches for a center,
                # unweight that center
                if max_high_satrms is not None:
                    n_outlier = 0
                    for key in self.exclude_highrms:
                        if key[0] == cen_outlier:
                            n_outlier += 1
                    if n_outlier > max_high_satrms:
                        self.unweighted_max_high_satrms.append(cen_outlier)
                        self.cenflags[cen_outlier] = 'unweighted'
                        self.weighted_centers.remove(cen_outlier)
                        self.unweighted_centers.append(cen_outlier)
                        logger.info(f"Maximum number of satellite outliers "
                                    f"exceeded for center {cen_outlier}: "
                                    f"{n_outlier} > {max_high_satrms}\n"
                                    f"Center {cen_outlier} unweighted.\n")

        # Test 2 - transformation parameters test
        # If any transformation parameter of a center exceeds a threshold,
        # unweight that center. If exceeded for more than one center, only
        # unweight the center with the largest transformation

        # Only do the test if requested, there are more than one weighted
        # centers, and any previous test has been successfully passed
        ncen = len(self.weighted_centers)
        if (trn_tst is not None
            and not all(thresh != None for thresh in thresh_trn)
            and ncen > 1
            and not rejection):

            # User-defined thresholds
            rms_tra_usr = thresh_trn[0]
            rms_rot_usr = thresh_trn[1]
            rms_sca_usr = thresh_trn[2]
            if rms_tra_usr is None:
                rms_tra_usr = 9999.0
            if rms_rot_usr is None:
                rms_rot_usr = 9999.0
            if rms_sca_usr is None:
                rms_sca_usr = 9999.0

            if trn_tst == 'manual':

                # Get the outlier levels
                rms_tra = rms_tra_usr
                rms_rot = rms_rot_usr
                rms_sca = rms_sca_usr

            elif trn_tst == 'auto' or trn_tst == 'strict':

                # Calculate the outlier levels
                tra2 = []
                rot2 = []
                sca2 = []
                for acname in self.weighted_centers:
                    for i in range(0,3):
                        tra2.append((self.transform_params[acname][i])**2)
                    for i in range(3,6):
                        rot2.append((self.transform_params[acname][i])**2)
                    sca2.append((self.transform_params[acname][6])**2)

                fact1 = 950.0*np.sqrt(3*ncen)
                fact2 = 980.0*np.sqrt(ncen)
                if ncen < 5:
                    fact1 = 4700.0
                    fact2 = 4700.0
                rms_tra = fact1*np.sqrt(np.mean(tra2))
                rms_rot = fact1*np.sqrt(np.mean(rot2))
                rms_sca = fact2*np.sqrt(np.mean(sca2))
                logger.debug(f"fact1={fact1} fact2={fact2}")
                logger.debug(f"rms_tra={rms_tra}, rms_rot={rms_rot}, "
                             f"rms_sca={rms_sca}")
                if trn_tst == 'strict':
                    rms_tra = min(rms_tra,rms_tra_usr)
                    rms_rot = min(rms_rot,rms_rot_usr)
                    rms_sca = min(rms_sca,rms_sca_usr)

            else:
                logger.error("Transformatoion test approach must be manual, "
                             "auto or strict!", stack_info=True)
                raise ValueError(f"Transformation test approach {trn_tst} "
                                  "not recognized!")

            if rms_sca < 1.0:
                logger.warning(f"rms_sca {rms_sca} > 1.0. "
                                "Setting rms-sca to 1.0")
                rms_sca = 1.0

            logger.debug(f"rms_tra={rms_tra}, rms_rot={rms_rot}, "
                        f"rms_sca={rms_sca}")

            # vector of transformation thresholds
            trn_thresh = np.zeros(7)
            for i in range(0,3):
                trn_thresh[i] = rms_tra/1000.0
            for i in range(3,6):
                trn_thresh[i] = rms_rot/1000.0
            trn_thresh[6] = rms_sca/1000.0

            # Search for the highest outlier
            trn_out_flg = False
            max_fact = 0.0
            fact = {}
            logger.debug(f"trn_thresh: {trn_thresh}")
            for acname in self.weighted_centers:
                logger.debug(f"tr {acname}: {self.transform_params[acname]}")
                fact[acname] = abs(self.transform_params[acname]/trn_thresh)
                if any(fact[acname] > 1):
                    trn_out_flg = True
                    if any(fact[acname] > max_fact):
                        max_fact = max(fact[acname])
                        ind_max_fact = list(fact[acname]).index(max_fact)
                        cen_outlier = acname

            # any rejection found?
            if trn_out_flg:

                rejection = True

                # Unweight the center with the largest transformation outlier
                self.unweighted_high_tra.append((cen_outlier,ind_max_fact,max_fact))
                self.cenflags[cen_outlier] = 'unweighted'
                self.weighted_centers.remove(cen_outlier)
                self.unweighted_centers.append(cen_outlier)
                logger.info(f"Transformation parameter too large for center "
                            f"{cen_outlier}:\n"
                            f"{self.transform_params[acname]}\n"
                            f"Thresholds: {thresh_trn}\n"
                            f"Center {cen_outlier} unweighted.\n")

        # Test 3 - high center rms : this is the same as test 1 except that
        # it is performed for unweighted centers
        logger.debug(f"rejection: {rejection}")
        if (sat_rms_tst_unweighted and thresh_sat is not None and not rejection):

            # Look only into unweighted centers (and sats not already excluded)
            sat_rms = {}
            for acname in self.weighted_centers + self.unweighted_centers:
                n_outlier = 0
                for key in self.exclude_highrms:
                    if key[0] == acname:
                        n_outlier += 1
                for sat in sats:
                    sys = sat[0]
                    if ( (acname,sat) in self.cen_rms
                            and self.satflags[acname,sat] == 'unweighted_sys'
                            and self.satflags[acname,sat] == 'unweighted_sat'
                            and acname not in self.unweighted_max_high_satrms
                            and n_outlier <= max_high_satrms ):
                        if sys not in sat_rms:
                            sat_rms[sys] = {}
                        sat_rms[sys][acname,sat] = self.cen_rms[acname,sat]

            # Find the largest outlier rms
            outlier_found = False
            maxrms = 0.0
            for sys in sat_rms:
                logger.debug(f"max sat_rms unweighted: {max(sat_rms[sys].values())*100}")
                if ( sys in thresh_sat_used
                        and max(sat_rms[sys].values()) > thresh_sat_used[sys]/100.0 ):

                    outlier_found = True

                    # Find the highest rms
                    if max(sat_rms[sys].values()) > maxrms:
                        maxrms = max(sat_rms[sys].values())
                        sys_out = sys
                        outlier_censat = max(sat_rms[sys],key=sat_rms[sys].get)

            # If outlier found, proceed to the exclusion
            if outlier_found:

                # There is at least one sat rms larger than the threshold
                rejection = True

                # Add the corresponding cen/sat pair to exclude_highrms
                cen_outlier = outlier_censat[0]
                sat_outlier = outlier_censat[1]
                logger.info("Exclusion due to outlier satellite RMS: "
                            f"{cen_outlier} {sat_outlier} {sat_rms[sys_out][cen_outlier,sat_outlier]}")

                # Update exclude_highrms
                if (cen_outlier,sat_outlier) not in self.exclude_highrms:
                    self.exclude_highrms.append((cen_outlier,sat_outlier))

                # If maximum number of outlier satellites reaches for a center,
                # issue a warning
                if max_high_satrms is not None:
                    n_outlier = 0
                    for key in self.exclude_highrms:
                        if key[0] == cen_outlier:
                            n_outlier += 1
                    if n_outlier > max_high_satrms:
                        logger.info(f"Number of satellite outliers higher than"
                                    f" the maximum for the unweighted center "
                                    f"{cen_outlier}: "
                                    f"{n_outlier} > {max_high_satrms}\n")

        # Test 4 - minimum number of centers for each satellite to be included

        # Only perform this test if requested, and any previous test has been
        # successfully passed
        if (numcen_tst is not None and min_numcen is not None
                and not rejection):

            # Get the number of centers for the satellite which has the
            # highest number of contributors
            max_num_ac = 0
            ncen = {}
            for sat in sats:
                ncen[sat] = 0
                for acname in self.weighted_centers:
                    if ((acname,sat) in self.satflags
                            and self.satflags[acname,sat] != 'missing_sat'
                            and self.satflags[acname,sat] != 'missing_blk'
                            and self.satflags[acname,sat] != 'missing_sys'
                            and self.satflags[acname,sat] != 'unweighted_sys'
                            and self.satflags[acname,sat] != 'unweighted_sat'
                            and self.satflags[acname,sat] != 'excluded_sat'):
                        ncen[sat] += 1
                if ncen[sat] > max_num_ac:
                    max_num_ac = ncen[sat]

            # If the max number of acs for a satellite is smaller than the
            # requested number of acs, set that max as the min number of acs
            if min_numcen > max_num_ac:
                min_numcen = max_num_ac

            # If eased, some other special circumstances
            if numcen_tst == 'eased':
                if (max_num_ac <= 4 and min_numcen > 2):
                    min_numcen = 2
                if (max_num_ac <= 5 and min_numcen > 3):
                    min_numcen = 3

            # Now look for any satellite which has lower number of acs than
            # the minimum specified; exclude that satellite from all the
            # centers if not already excluded
            for sat in sats:
                if ncen[sat] < min_numcen:

                    if sat not in self.exclude_lowcen:
                        logger.info(f"There are only {ncen[sat]} centers for "
                                    f"{sat} < {min_numcen}. Excluding the "
                                    f"satellite from combination.\n")
                        rejection = True
                        self.exclude_lowcen.append(sat)

        self.min_numcen = min_numcen

        # Update rejection attribute
        self.rejection = rejection


    def to_sp3dict(self,sample_rate,sp3_header):

        """
        Convert combined_orbit, epochs and satinfo attributes to a sp3
        dictionary, so it can be used by io_data module for writing to a sp3
        file

        Updates:
            self.sp3_combined [dict]: combined sp3 dictionary

        """

        # Create a list of all satellites
        sats = []
        sat_accuracy = []
        sys_list = []
        for key in self.cen_rms:
            if not isinstance(key,str):
                if isinstance(key[1],tuple):
                    sys_id = key[1][0]
                    prn = key[1][1]
                    sat = sys_id + str(prn).zfill(2)
                    if sat not in sats:
                        sats.append(sat)
                        sat_accuracy.append(self.sat_accuracy[
                                        (key[1][0],key[1][1],key[1][2])])
                    if sys_id not in sys_list:
                        sys_list.append(sys_id)
        # sort
        sat_accuracy = [x for _, x in sorted(zip(sats, sat_accuracy))]
        sats.sort()

        # weighted centers
        acnames = []
        acnames_0 = []
        ac_ctr = 0
        ctr = 0
        for ac in self.weighted_centers:
            ac_ctr += 1
            ctr += 1
            acnames_0.append(ac)
            if ctr == 14 or ac_ctr == len(self.weighted_centers):
                acnames.append(' '.join([item for item in acnames_0]))
                ctr = 0
                acnames_0 = []

        # Initialize the sp3 dictionary
        sp3_combined = {}
        sp3_combined['header'] = {}
        sp3_combined['data'] = {}

        # Header lines
        sp3_combined['header']['version'] = '#d'
        sp3_combined['header']['pvflag'] = 'P'
        sp3_combined['header']['data_used'] = 'ORBIT'
        sp3_combined['header']['coord_sys'] = sp3_header['coord_sys']
        sp3_combined['header']['orbit_type'] = 'HLM'
        sp3_combined['header']['agency'] = 'IGS'
        if len(sys_list) > 1:
            sp3_combined['header']['file_type'] = 'M'
        else:
            sp3_combined['header']['file_type'] = sys_list[0]
        sp3_combined['header']['time_system'] = 'GPS'
        sp3_combined['header']['base_pos'] = 1.25
        sp3_combined['header']['base_clk'] = 1.025
        if sp3_header['cmb_type'] == "REPRO3":
            sp3_combined['header']['comments'] = [
                    f"{sp3_header['cmb_type']} ORBIT COMBINATION FROM WEIGHTED "
                    "AVERAGE OF:"]
        else:
            sp3_combined['header']['comments'] = [
                    f"STRICTLY EXPERIMENTAL MULTI-GNSS COMBINATION",
                    f"{sp3_header['cmb_type']} ORBIT COMBINATION FROM WEIGHTED "
                    "AVERAGE OF:"]
        sp3_combined['header']['comments'].extend(acnames)
        sp3_combined['header']['comments'].extend([
                 "REFERENCED TO GPS CLOCK AND TO WEIGHTED MEAN POLE:",
                f"PCV:{sp3_header['antex']} OL/AL:{sp3_header['oload']} "
                f"NONE     Y  ORB:CMB CLK:{sp3_header['clk_src']}"])

        # should include other high-eccentric satellites in the future if any
        if (sample_rate > 300 and 'E' in sys_list):
            sp3_combined['header']['comments'].extend([
                 "WARNING: The highly eccentric satellites could have",
                 "errors of up to ~10 mm due to the 15-minute sampling rates"])

        sp3_combined['header']['start_year'] = self.epochs[0].year
        sp3_combined['header']['start_month'] = self.epochs[0].month
        sp3_combined['header']['start_day'] = self.epochs[0].day
        sp3_combined['header']['start_hour'] = self.epochs[0].hour
        sp3_combined['header']['start_min'] = self.epochs[0].minute
        sp3_combined['header']['start_sec'] = float(self.epochs[0].second)
        gc = gpsCal()
        gc.set_yyyy_MM_dd_hh_mm_ss(self.epochs[0].year,self.epochs[0].month,
                                   self.epochs[0].day,self.epochs[0].hour,
                                   self.epochs[0].minute,self.epochs[0].second)
        sp3_combined['header']['gpsweek'] = gc.wwww()
        sp3_combined['header']['sow'] = gc.sow()
        sp3_combined['header']['epoch_int'] = sample_rate
        sp3_combined['header']['modjul'] = int(gc.mjd())
        sp3_combined['header']['frac'] = gc.mjd() - int(gc.mjd())

        sp3_combined['header']['sats'] = sats
        sp3_combined['header']['numsats'] = len(sats)
        sp3_combined['header']['sat_accuracy'] = sat_accuracy

        # Get a unique list of epochs
        eps = []
        for ep in self.epochs:
            if ep not in eps:
                eps.append(ep)
        sp3_combined['data']['epochs'] = eps
        sp3_combined['header']['num_epochs'] = len(eps)

        pred_start_ultra = self.epochs[0] + datetime.timedelta(days=1)

        # Loop over all orbit rows
        for c,row in enumerate(self.combined_orbit):

            sat = self.satinfo[c,0]+str(self.satinfo[c,1]).zfill(2)
            epoch = self.epochs[c]

            sp3_combined['data'][(sat,epoch,'Pflag')] = 1
            sp3_combined['data'][(sat,epoch,'EPflag')] = 0
            sp3_combined['data'][(sat,epoch,'Vflag')] = 0
            sp3_combined['data'][(sat,epoch,'EVflag')] = 0
            sp3_combined['data'][(sat,epoch,'xcoord')] = row[0]/1000.0 # to km
            sp3_combined['data'][(sat,epoch,'ycoord')] = row[1]/1000.0 # to km
            sp3_combined['data'][(sat,epoch,'zcoord')] = row[2]/1000.0 # to km
            if self.clocks is not None:
                sp3_combined['data'][(sat,epoch,'clock')]  = self.clocks[sp3_header['clk_src']][c,0]
                sp3_combined['data'][(sat,epoch,'csdev')]  = self.clocks[sp3_header['clk_src']][c,1]
            base_pos = sp3_combined['header']['base_pos']
            if (sp3_header['cmb_type'] == "ULTRA RAPID" and epoch >= pred_start_ultra):
                sp3_combined['data'][(sat,epoch,'orbit_pred')] = 'P'
                sp3_combined['data'][(sat,epoch,'clk_pred')]   = 'P'
            if not np.isnan(self.sdev[c,0]):
                if self.sdev[c,0]*1000.0 > base_pos:
                    sp3_combined['data'][(sat,epoch,'xsdev')] = (int(round(
                        np.log(self.sdev[c,0]*1000.0)/
                        np.log(base_pos)))) # to exponents in mm
                    if sp3_combined['data'][(sat,epoch,'xsdev')] > 99:
                        sp3_combined['data'][(sat,epoch,'xsdev')] = 99
                elif self.sdev[c,0]*1000.0 > 0.0:
                    sp3_combined['data'][(sat,epoch,'xsdev')] = 1
            if not np.isnan(self.sdev[c,1]):
                if self.sdev[c,1]*1000.0 > base_pos:
                    sp3_combined['data'][(sat,epoch,'ysdev')] = (int(round(
                        np.log(self.sdev[c,1]*1000.0)/
                        np.log(base_pos)))) # to exponents in mm
                    if sp3_combined['data'][(sat,epoch,'ysdev')] > 99:
                        sp3_combined['data'][(sat,epoch,'ysdev')] = 99
                elif self.sdev[c,1]*1000.0 > 0.0:
                    sp3_combined['data'][(sat,epoch,'ysdev')] = 1
            if not np.isnan(self.sdev[c,2]):
                if self.sdev[c,2]*1000.0 > base_pos:
                    sp3_combined['data'][(sat,epoch,'zsdev')] = (int(round(
                        np.log(self.sdev[c,2]*1000.0)/
                        np.log(base_pos)))) # to exponents in mm
                    if sp3_combined['data'][(sat,epoch,'zsdev')] > 99:
                        sp3_combined['data'][(sat,epoch,'zsdev')] = 99
                elif self.sdev[c,2]*1000.0 > 0.0:
                    sp3_combined['data'][(sat,epoch,'zsdev')] = 1

        # Update attributes
        self.sample_rate = sample_rate
        self.sp3_combined = sp3_combined

