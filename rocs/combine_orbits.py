# Full orbit combination

import glob
import argparse
import logging
import os
import sys
import numpy as np
import datetime
import rocs.orbits as orbits
from rocs.gpscal import gpsCal
import rocs.settings as settings
import rocs.io_data as io_data
import rocs.checkutils as checkutils
from rocs.report import OrbitReport

logger = logging.getLogger(__name__)

def combine_orbits(gpsweek,dow,hr,config):

    ## common campaign specifications

    # author
    author = config['campaign']['author']
    if not isinstance(author,str):
        logger.error("\nAuthor must be a string\n",stack_info=True)
        raise TypeError(f"{author} not a string")

    # contact
    contact = config['campaign']['contact']
    if not isinstance(contact,str):
        logger.error("\nContact msust be a string\n",stack_info=True)
        raise TypeError(f"{contact} not a string")

    # solution type identifier
    sol_id = config['campaign']['sol_id']
    allowed_sol = ['ULT','RAP','FIN']
    if sol_id not in allowed_sol:
        logger.error("\nSolution type identifier must be one of"
                     f"{allowed_sol}\n",stack_info=True)
        raise ValueError(f"{sol_id} not in {allowed_sol}")

    if sol_id == 'ULT':
        solution = 'ultra-rapid'
    elif sol_id == 'RAP':
        solution = 'rapid'
    elif sol_id == 'FIN':
        solution = 'final'

    if solution == 'ultra-rapid':
        len_data = '02D'
    else:
        len_data = '01D'

    # campaign/project specification
    camp_id = config['campaign']['camp_id']
    allowed_camp = ['DEM','MGX','OPS','TST']
    if camp_id not in allowed_camp and cam_id[0:1] != 'R':
        logger.error("\nCampaign specification must be one of"
                f"{allowed_camp} or Rnn for Repro\n")
        raise ValueError(f"{camp_id} not recognized!")

    # combination name
    cmb_name = config['campaign']['cmb_name']
    if not isinstance(cmb_name,str):
        logger.error("\nCombination name abbreviation "
                    "must be a string\n",stack_info=True)
        raise TypeError(f"{cmb_name} is not a string!")
    if len(cmb_name) != 3:
        logger.error("\nCombination name abbreviation must be a "
                     "3-character string\n",stack_info=True)
        raise ValueError(f"{cmb_name} has length {len(cmb_name)}!")

    # version identifier for the combined orbit
    vid = config['campaign']['vid']
    if not isinstance(vid,int):
        logger.error("\nCombination version identifier must be an integer\n",
                        stack_info=True)
        raise TypeError(f"Combination version identifier is of type "
                        f"{type(vid)}!")
    if vid < 0 or vid > 9:
        logger.error("\nCombination version identifier must be in the "
                     "range 0-9\n",stack_info=True)
        raise ValueError(f"Combination version identifier {vid} is not in the "
                         f"range 0-9!")

    # information on whether to cut solutions at the start or end
    cut_start = config['campaign']['cut_start']
    if not isinstance(cut_start,int):
        logger.error("\ncut_start must be an integer\n",
                        stack_info=True)
        raise TypeError(f"{cut_start} is of type {type(cut_start)}!")

    cut_end = config['campaign']['cut_end']
    if not isinstance(cut_end,int):
        logger.error("\ncut_end must be an integer\n",
                        stack_info=True)
        raise TypeError(f"{cut_end} is of type {type(cut_end)}!")

    # submissions root directory
    subm_rootdir = config['campaign']['subm_rootdir']
    if not isinstance(subm_rootdir,str):
        logger.error("\nSubmission root directory must be specified "
                     "as a string\n",stack_info=True)
        raise TypeError(f"Submission root directory {subm_rootdir} is not "
                         "a string!")

    # products root directory
    prod_rootdir = config['campaign']['prod_rootdir']
    if not isinstance(prod_rootdir,str):
        logger.error("\nProducts root directory must be specified as "
                     "a string\n",stack_info=True)
        raise TypeError(f"Products root directory {prod_rootdir} is not "
                         "a string!")

    # satellite metadata file
    sat_metadata_file = config['campaign']['sat_metadata_file']
    if sat_metadata_file is not None and not isinstance(sat_metadata_file,str):
        logger.error("\nPath to satellite metadata file must be specified as "
                     "a string\n",stack_info=True)
        raise TypeError(f"satellite metadata {sat_metadata_file} is not "
                         "a string!")

    # EOP format
    eop_format = config['campaign']['eop_format']
    allowed_eop_format = ['IERS_EOP14_C04','IERS_EOP_rapid','IGS_ERP2']
    if eop_format is not None and eop_format not in allowed_eop_format:
        logger.error(f"\neop_format must be one of {allowed_eop_format}\n",
                stack_info=True)
        raise ValueError(f"eop_fomat {epo_format} not recognized!")

    # EOP file
    eop_file = config['campaign']['eop_file']
    if eop_file is not None:
        if not isinstance(eop_file,str):
            logger.error("\nPath to EOP data filename "
                    "must be a string\n",stack_info=True)
            raise TypeError(f"{eop_file} is not a string! {type(eop_file)}")

    # reference frame combination summary location
    rf_rootdir = config['campaign']['rf_rootdir']
    if rf_rootdir is not None:
        if not isinstance(rf_rootdir,str):
            logger.error("\nPath to reference frame combination summaries "
                    "must be a string\n",stack_info=True)
            raise TypeError(f"{rf_rootdir} is not a string!")

    # reference frame combination summary filename
    rf_name = config['campaign']['rf_name']
    if rf_name is not None:
        if not isinstance(rf_name,str):
            logger.error("\nReference frame combination summary filename "
                    "must be a string\n",stack_info=True)
            raise TypeError(f"{rf_name} is not a string!")

    # NANU summary file
    nanu_sumfile = config['campaign']['nanu_sumfile']
    if nanu_sumfile is not None:
        if not isinstance(nanu_sumfile,str):
            logger.error("\nNANU summary filename "
                    "must be a string\n",stack_info=True)
            raise TypeError(f"{nanu_sumfile} is not a string!")

    # AC acronyms
    ac_acronyms = config['campaign']['ac_acronyms']
    if not isinstance(ac_acronyms,dict):
        logger.error("\nac_acronyms must be a dict\n",
                    stack_info=True)
        raise TypeError("ac_acronyms is not of type dict")
    for key in ac_acronyms:
        if not isinstance(key,str):
            logger.error("\nAC acronyms must be "
                    "of type str\n",stack_info=True)
            raise TypeError(f"AC acronym {key} is not a string!")
        if not isinstance(ac_acronyms[key],str):
            logger.error("\nAC acronym descriptions must be "
                    "of type str\n",stack_info=True)
            raise TypeError(f"AC acronym description {ac_acronyms[key]}"
                    " is not a string!")

    ## orbit combination options

    # contributions to the orbit combination
    ac_contribs_orbs = config['orbits']['ac_contribs']
    if not isinstance(ac_contribs_orbs,dict):
        logger.error("\nac_contribs_orbs must be a dict\n",
                    stack_info=True)
        raise TypeError(f"{ac_contribs_orbs} is not of type dict")
    allowed_ac_contribs_keys = ['weighted','unweighted','excluded']
    for key in ac_contribs_orbs:
        if key not in allowed_ac_contribs_keys:
            logger.error("\nThe keys in ac_contribs_orbs must be "
                        f"one of {allowed_ac_contribs_keys}\n",stack_info=True)
            raise ValueError(f"The key {key} not recognized! ")
        if not isinstance(ac_contribs_orbs[key],dict):
            logger.error("\nThe values in ac_contribs_orbs must be "
                        "of type dict\n",stack_info=True)
            raise TypeError("There are non-dict valuess in "
                            f"{ac_contribs_orbs} ")
        allowed_ac_contribs_keys1 = ['systems','prns','svns']
        for key1 in ac_contribs_orbs[key]:
            if key1 not in allowed_ac_contribs_keys1:
                logger.error(f"\nThe keys in {ac_contribs_orbs[key]} "
                            f"must be one of {allowed_ac_contribs_keys1}\n",
                            stack_info=True)
                raise ValueError(f"The key {key1} not recognized! ")
            if ac_contribs_orbs[key][key1] is not None:
                for key2 in ac_contribs_orbs[key][key1]:
                    if not isinstance(key2,str):
                        logger.error("\nCenter names must be of type str",
                                        stack_info=True)
                        raise TypeError(f"The key {key2} not str!")
                    if len(key2) != 3:
                        logger.error("\n Center names in "
                                    f"{ac_contribs_orbs[key][key1]} must be "
                                     "3-character strings\n",stack_info=True)
                        raise ValueError(f"Center name {key2} not 3 "
                                          "characters!")
                    if not isinstance(ac_contribs_orbs[key][key1][key2],list):
                        logger.error(f"\nCenter name contributions "
                                    "must be of type list\n",stack_info=True)
                        raise TypeError(f"{ac_contribs_orbs[key][key1][key2]}"
                                        " is not a list!")
                    for item in ac_contribs_orbs[key][key1][key2]:
                        if not isinstance(item,str):
                            logger.error("\nConstellation codes must be "
                                        " of type str\n",stack_info=True)
                            raise TypeError(f"{item} in "
                                f"{ac_contribs_orbs[key][key1][key2]}"
                                " is not a string!")
                        if len(item) != 1:
                            logger.error("\nConstellation codes must be "
                                        " 1-character strings\n"
                                        ,stack_info=True)
                            raise ValueError(f"{item} in "
                                        f"{ac_contribs_orbs[key][key1][key2]}"
                                        " is not a 3-character string")

    # orbit sampling
    orbit_sampling = config['orbits']['sampling']
    if orbit_sampling is not None and not isinstance(orbit_sampling,int):
        logger.error("\nOrbit sampling must be an integer\n", stack_info=True)
        raise TypeError(f"Orbit sampling {orbit_sampling} is not an integer!")

    # center weighting method
    # In case of weighting by constellation/block/sat, check if metadata file is
    # specified; if not revert to global weighting and issue a warning
    if (config['orbits']['cen_wht_method'] in
                ['by_constellation','by_block','by_sat']
            and sat_metadata_file is None):
        logger.warning(f"\nCenter weighting {cen_wht_method} is requested but "
                       f"there is no satellite metadata file\nspecified. "
                       f"Satellite metadata is required to identify satellite "
                       f"blocks.\nSetting the centre weighting method to the "
                       f"default global method.\n")
        config['orbits']['cen_wht_method'] = 'global'

    cen_wht_method = config['orbits']['cen_wht_method']
    allowed_cen_wht_method = ['global','by_constellation','by_block','by_sat']
    if cen_wht_method not in allowed_cen_wht_method:
        logger.error("\nCenter weighting method must be one of "
                     f"{allowed_cen_wht_method}\n", stack_info=True)
        raise ValueError(f"Center weighting method {cen_wht_method} is not in"
                         f"{allowed_cen_wht_method}")

    # satellite weighting method
    sat_wht_method = config['orbits']['sat_wht_method']
    if sat_wht_method not in ['RMS_L1']:
        logger.error("\nSatellite weighting method can only be RMS_L1\n",
                        stack_info=True)
        raise ValueError(f"Satellite weighting method {sat_wht_method} is not "
                         f"recognised!")

    # reference frame alignment options
    rf_align = config['orbits']['rf_align']
    if not isinstance(rf_align,list):
        logger.error("\nrf_align must be a list\n",stack_info=True)
        raise TypeError(f"{rf_align} is not a list!")
    for item in rf_align:
        if not isinstance(item,bool):
            logger.error("\nrf_align items must be booleans\n",stack_info=True)
            raise TypeError(f"{item} is not a boolean!")

    # list of centres for which UT1 differences should be applied
    # as corrections to Z rotations
    ut1_rot = config['orbits']['ut1_rot']
    if ut1_rot is not None:
        if not isinstance(ut1_rot,list):
            logger.error("\nut1_rot must be a list\n ",stack_info=True)
            raise TypeError(f"{ut1_rot} is not a list!")
        for item in ut1_rot:
            if not isinstance(item,str):
                logger.error("\nCenter names in ut1_rot must be strings\n",
                        stack_info=True)
                raise TypeError(f"{item} not a string!")
            if len(item) != 3:
                logger.error("\nCenter names in ut1_rot must be 3-character "
                             "strings\n",stack_info=True)
                raise ValueError(f"Center name {item} not a 3-character "
                                  "string!")

    # EOP format for the ut1 correction centers
    ut1_eop_format = config['orbits']['ut1_eop_format']
    allowed_eop_format = ['IERS_EOP14_C04','IERS_EOP_rapid','IGS_ERP2']
    if ut1_eop_format not in allowed_eop_format:
        logger.error(f"\nut1_eop_format must be one of {allowed_eop_format}\n",
                    stack_info=True)
        raise ValueError(f"eop_format {epo_format} not recognized!")

    # maneuvering satellites options
    rm_dv = config['orbits']['rm_dv']
    if not isinstance(rm_dv,bool):
        logger.error("\nrm_dv must be boolean\n",stack_info=True)
        raise TypeError(f"{rm_dv} is not of type boolean!")

    no_rm_dv = config['orbits']['no_rm_dv']
    if no_rm_dv is not None:
        if not isinstance(no_rm_dv,list):
            logger.error("\nno_rm_dv must be a list\n ",stack_info=True)
            raise TypeError(f"{no_rm_dv} is not a list!")
        for item in no_rm_dv:
            if not isinstance(item,str):
                logger.error("\nCenter names in no_rm_dv must be strings\n",
                        stack_info=True)
                raise TypeError(f"{item} not a string!")
            if len(item) != 3:
                logger.error("\nCenter names in no_rm_dv must be 3-character "
                             "strings\n",stack_info=True)
                raise ValueError(f"Center name {item} not a 3-character "
                                  "string!")

    # outlier detection (assess) options
    allowed_sat_rms_tst = ['auto','manual','strict']
    sat_rms_tst = config['orbits']['assess']['sat_rms_tst']
    if (sat_rms_tst is not None
            and sat_rms_tst not in allowed_sat_rms_tst):
        logger.error("\nsat_rms_tst must be one of "
                f"{allowed_sat_rms_tst}\n", stack_info=True)
        raise ValueError(f"The given sat_rms_tst {sat_rms_tst} is not in "
                 f"{allowed_sat_rms_tst}")

    sat_rms_tst_unweighted = (config['orbits']['assess']
                                    ['sat_rms_tst_unweighted'])
    if (sat_rms_tst_unweighted is not None
            and sat_rms_tst_unweighted not in allowed_sat_rms_tst):
        logger.error("\nsat_rms_tst_unweighted must be one of "
                    f"{allowed_sat_rms_tst}\n", stack_info=True)
        raise ValueError("The given sat_rms_tst_unweighted "
                        f"{sat_rms_tst_unweighted} is not in "
                        f"{allowed_sat_rms_tst}")

    coef_sat = config['orbits']['assess']['coef_sat']
    checkutils.check_scalar(coef_sat)

    thresh_sat = config['orbits']['assess']['thresh_sat']
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

    max_high_satrms = config['orbits']['assess']['max_high_satrms']
    if max_high_satrms is not None:
        if not isinstance(max_high_satrms,int):
            logger.error("\nThe given max_high_satrms must be an "
                    "integer\n", stack_info=True)
            raise TypeError("The given max_high_satrms is not of type int")

    trn_tst = config['orbits']['assess']['trn_tst']
    if trn_tst is not None:
        allowed_trn_tst = ['auto','manual','strict']
        if trn_tst not in allowed_trn_tst:
            logger.error(f"\ntrn_tst must be one of {allowed_trn_tst}\n",
                    stack_info=True)
            raise ValueError("The given trn_tst is not in "
                    "{allowed_trn_tst}")

    thresh_trn = config['orbits']['assess']['thresh_trn']
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

    numcen_tst = config['orbits']['assess']['numcen_tst']
    if numcen_tst is not None:
        allowed_numcen_tst = ['strict','eased']
        if numcen_tst not in allowed_numcen_tst:
            logger.error("\nnumcen_tst must be one of "
                    f"{allowed_numcen_tst}\n", stack_info=True)
            raise ValueError("The given numcen_tst is not in "
                    "{allowed_numcen_tst}")

    min_numcen = config['orbits']['assess']['min_numcen']
    if min_numcen is not None:
        if not isinstance(min_numcen,int):
            logger.error("\nThe given min_numcen must be an "
                    "integer\n", stack_info=True)
            raise TypeError("The given min_numcen is not of type int")

    max_iter = config['orbits']['assess']['max_iter']
    if not isinstance(max_iter,int):
        logger.error("\nmax_iter must be an integer\n", stack_info=True)
        raise TypeError(f"max_iter {max_iter} is not an integer!")
    if max_iter < 1:
        logger.error("\nmax_iter must be a positive number\n", stack_info=True)
        raise TypeError(f"max_iter {max_iter} is not a positive number!")

    # SP3 header information
    sp3_header = config['orbits']['sp3_header']
    if not isinstance(sp3_header,dict):
        logger.error("\nSP3 header information must be specified "
                "as a dictionary\n")
        raise TypeError(f"SP3 header {sp3_header} is not of type dict!")
    allowed_sp3_header_keys = ['coord_sys','cmb_type','clk_src','antex','oload']
    for key in sp3_header:
        if key not in allowed_sp3_header_keys:
            logger.error("\nSP3 header must be one of "
                        f"{allowed_sp3_header_keys}\n",stack_info=True)
            raise ValueError(f"SP3 header item {key} not recognized!")
        if not isinstance(sp3_header[key],str):
            loogger.error("\nSP3 header items must be string\n",stack_info=True)
            raise TypeError(f"SP3 hedaer {key} : {sp3_header[key]} not a "
                             "string!")

    # print out the command line
    command = " ".join(sys.argv)
    logger.info("\nStarted the combination program\nCommand line:\n"
                f"{command}\n")

    if solution == 'ultra-rapid':
        dowhr_line = f"day of week {dow}, hour {str(hr).zfill(2)}.\n"
    else:
        dowhr_line = f"day of week {dow}.\n"

    logger.info("\nStarted the orbit combination for GPS week "
                f"{str(gpsweek).zfill(4)} {dowhr_line}\n"
                f"Configuration file\n"
                f"Solution: {solution}\nSampling interval for orbit "
                f"combination (seconds): {orbit_sampling}\nRoot directory "
                f"for submissions: {subm_rootdir}\nRoot directory for "
                f"combination products: {prod_rootdir}\nSatellite metadata "
                f"file: {sat_metadata_file}\nCenter weighting method: "
                f"{cen_wht_method}\nSatellite weighting method: "
                f"{sat_wht_method}\n")

    logger.debug(f"\nconfig {config}\n")

    # Determine some directories
    subm_weekdir = subm_rootdir + '/w' + str(gpsweek).zfill(4)
    prod_weekdir = prod_rootdir + '/w' + str(gpsweek).zfill(4)
    rf_weekdir = rf_rootdir + '/w' + str(gpsweek).zfill(4)

    # Make directories if they do not exist
    if (not os.path.isdir(prod_rootdir)):
            os.mkdir(prod_rootdir)
    if (not os.path.isdir(prod_weekdir)):
            os.mkdir(prod_weekdir)

    # satellite metadata file
    if sat_metadata_file is not None:
        sat_metadata = io_data.SatelliteMetadata(sat_metadata_file)

    # Read the nanu summary file if required
    if rm_dv:
        nanu = io_data.NANU_sum(nanu_sumfile)
        nanu.get_dv(solution)

    # Determine year and day of year
    gc = gpsCal()
    gc.set_wwww_dow(gpsweek,dow)
    year = gc.yyyy()
    doy = gc.ddd()
    month = gc.MM()
    dom = gc.dom()
    start_epoch = (datetime.datetime(year,month,dom,hr,0,0)
                    + datetime.timedelta(seconds=cut_start))
    if solution != 'ultra-rapid':
        end_epoch = (datetime.datetime(year,month,dom,23,59,59)
                    - datetime.timedelta(seconds=cut_end))
    else:
        end_epoch = (start_epoch + datetime.timedelta(days=2)
                                 - datetime.timedelta(seconds=1)
                                 - datetime.timedelta(seconds=cut_end))

    # Look into the submission directory for available submissions
    sp3_subm_all = glob.glob(subm_weekdir+'/???????'+sol_id+'_'+str(year)
                            +str(doy).zfill(3)+str(hr).zfill(2)+'00_'+len_data
                            +'_???_ORB.SP3')

    # Get the list of SP3 files to be read
    # If a configuration file is used, use all the weighted and unweighted
    # centers (for at least one satellite); otherwise, use all available
    # orbit solutions
    if ac_contribs_orbs is not None:

        # set of contributing centers
        contributing_acs = set()

        weighted =  ac_contribs_orbs['weighted']
        for item in weighted:
            if weighted[item] is not None:
                for acname in weighted[item]:
                    contributing_acs.add(acname)
        unweighted =  ac_contribs_orbs['unweighted']
        for item in unweighted:
            if unweighted[item] is not None:
                for acname in unweighted[item]:
                    contributing_acs.add(acname)
        if contributing_acs:
            sp3_subm_list = []
            for sp3_subm in sp3_subm_all:
                acname = sp3_subm[-38:-35]
                if acname in contributing_acs:
                    sp3_subm_list.append(sp3_subm)
        else:
            logger.error(f"There is no center to be used (weighted or "
                         f"unweighted) in the config file ")
            raise ValueError(f"contributing_acs is empty")
    else:
        logger.info(f"There is no configuration file given.\nUsing all "
                    f"available products.\n")
        sp3_subm_list = sp3_subm_all

    sp3_subm_list.sort()

    logger.info(f"Orbit files used for orbit combination:\n")
    for item in sp3_subm_list:
        logger.info(f"{item}")
    logger.info("")

    # Read the orbit sp3 files
    sp3_dict = {}
    for sp3_subm in sp3_subm_list:
        acname = sp3_subm[-38:-35]
        sp3_ac = io_data.sp3(sp3_subm)
        sp3_ac.parse(start_epoch,end_epoch)
        sp3_dict[acname] = sp3_ac.sp3dict

    ## Preprocessing of the orbits

    # Initialize the class instance
    if ac_contribs_orbs is None:
        if sat_metadata_file is None:
            orbs = orbits.OrbitPrep(sp3all=sp3_dict)
        else:
            orbs = orbits.OrbitPrep(sp3all=sp3_dict,sat_metadata=sat_metadata)
    else:
        if sat_metadata_file is None:
            orbs = orbits.OrbitPrep(sp3all=sp3_dict,
                    ac_contribs=ac_contribs_orbs)
        else:
            orbs = orbits.OrbitPrep(sp3all=sp3_dict,
                    ac_contribs=ac_contribs_orbs,sat_metadata=sat_metadata)
        orbs.filter_contribs()

    # Resample if requested
    if orbit_sampling is not None:
        orbs.resample(orbit_sampling)

    # Remove DV maneuvering satellites if needed
    if rm_dv:
        orbs.rm_dv(nanu.dv,no_rm_dv)
        ind = np.where(((nanu.dvfull[:,0]>=start_epoch) &
                        (nanu.dvfull[:,0]<=end_epoch)) |
                        ((nanu.dvfull[:,1]>=start_epoch) &
                        (nanu.dvfull[:,1]<=end_epoch)))
        dvsats = nanu.dvfull[ind]
        dvsats_new = []
        for row in dvsats:
            sys_id = row[2][0:1]
            prn = int(row[2][1:])
            ep = row[0]
            svn_no = sat_metadata.get_svn(sys_id,prn,ep)
            svn = sys_id + str(svn_no).zfill(3)
            dvsats_new.append([row[0],row[1],row[2],svn])
        dvsats = dvsats_new

    # Convert orbit dictionaries to arrays
    orbs.to_arrays()

    # Check if there is any orbit solution to be included
    if not orbs.orbits:
        if ac_contribs_orbs is not None:
            logger.error(f"\nNo orbit solution is included as weighted or "
                         f"unweighted.\nCheck the config file "
                         f"and the orbit directory {subm_weekdir}\n",
                         stack_info=True)
            raise ValueError(f"orbits is empty")
        else:
            logger.error(f"\nNo orbit solution is included as weighted or "
                         f"unweighted.\nCheck the orbit directory "
                         f"{subm_weekdir}\n",stack_info=True)
            raise ValueError(f"orbits is empty")

    logger.debug(f"epochs:\n{orbs.epochs} {np.shape(orbs.epochs)}\n")
    logger.debug(f"orbits:\n{orbs.orbits} {np.shape(orbs.orbits)}\n")
    logger.debug(f"satinfo:\n{orbs.satinfo} {np.shape(orbs.satinfo)}\n")

    ## Orbit combination

    # Initialize the class instance
    # For ultra-rapid, read clocks so an AC clock (clk_src) can be reported
    # along with the combined orbits
    if solution == 'ultra-rapid':
        orbcmb = orbits.OrbitComb(orbits=orbs.orbits,epochs=orbs.epochs,
                            satinfo=orbs.satinfo,cenflags=orbs.cenflags,
                            weighted_cens_by_sys=orbs.weighted_cens_by_sys,
                            unweighted_cens_by_sys=orbs.unweighted_cens_by_sys,
                            weighted_sats=orbs.weighted_sats,
                            unweighted_sats=orbs.unweighted_sats,
                            clocks=orbs.clocks)
    else:
        orbcmb = orbits.OrbitComb(orbits=orbs.orbits,epochs=orbs.epochs,
                            satinfo=orbs.satinfo,cenflags=orbs.cenflags,
                            weighted_cens_by_sys=orbs.weighted_cens_by_sys,
                            unweighted_cens_by_sys=orbs.unweighted_cens_by_sys,
                            weighted_sats=orbs.weighted_sats,
                            unweighted_sats=orbs.unweighted_sats)

    logger.debug(f"orbits original: {orbcmb.orbits}")

    # If reference frame alignment is requested, read the rf combination
    # summary yaml files to get the transformation parameters, and transform
    # the orbits
    transformations = {}
    if any(rf_align):

        # First day of the week
        gc = gpsCal()
        gc.set_wwww_dow(gpsweek,0)
        year_firstdow = gc.yyyy()
        doy_firstdow = gc.ddd()

        # reference frame combination summary file
        rf_summary = (rf_weekdir + '/' + rf_name + '_'
                        + str(year_firstdow).zfill(4)
                        + str(doy_firstdow).zfill(3) + '0000_07D_07D_SUM.YML')
        ref_sum = io_data.Ref_sum(rf_summary)
        ref_sum.transfo(rf_align=rf_align)

        # If the UT1 rotation is requested for any center, read the apriori
        # and the observed ERP files for that center, and apply the Z rotation
        if ut1_rot is not None:

            logger.debug(f"ut1_rot: {ut1_rot}")
            logger.debug("transformations before ut1 correction:\n"
                         f"{ref_sum.transformations}")

            for acname in ut1_rot:

                erp_aprfile = glob.glob(rf_rootdir+'/'+acname.upper()
                                +'????'+'APR_???????????_???_???_ERP.ERP')
                erp_obsfile = glob.glob(rf_rootdir+'/'+acname.upper()
                                +'????'+'OBS_???????????_???_???_ERP.ERP')
                logger.debug(f"erp_aprfile: {erp_aprfile}")
                if len(erp_aprfile) > 1:
                    raise ValueError(f"\nThere must be only one erp "
                                     f"apriori file for a center. The "
                                     f"files found:\n{erp_aprfile}")
                if len(erp_obsfile) > 1:
                    raise ValueError(f"\nThere must be only one erp "
                                     f"observed file for a center. The "
                                     f"files found:\n{erp_obsfile}")

                ref_sum.ut1_rot(acname,erp_aprfile,erp_obsfile,ut1_eop_format)

            logger.debug("transformations after ut1 correction:\n"
                         f"{ref_sum.transformations}")

        transformations = ref_sum.transformations

        orbcmb.transform(transformations[dow])

        logger.debug(f"transformations: {transformations[dow]}")
        logger.debug(f"orbits transformed: {orbcmb.orbits}")

    # Set redo to True so the combination runs the first time
    redo = True

    iter = 0

    # Loop until no outlier is detected or maximum iteration exceeded
    while redo and iter <= max_iter:

        iter += 1

        # Perform the weighting
        orbcmb.weight(cen_wht_method=cen_wht_method,
                            sat_wht_method=sat_wht_method)

        logger.debug(f"centre weights: {orbcmb.cen_weights}")

        # Perform the combination
        orbcmb.combine()

        logger.debug(f"\ncentre RMS's: {orbcmb.cen_rms}\n")
        logger.debug(f"\ncentre abdev's: {orbcmb.cen_abdev}\n")

        logger.debug(f"config {config}\n")
        logger.debug(f"thresh_sat: {thresh_sat}")
        orbcmb.assess(sat_rms_tst=sat_rms_tst,
                sat_rms_tst_unweighted=sat_rms_tst_unweighted,
                coef_sat=coef_sat,thresh_sat=thresh_sat,
                max_high_satrms=max_high_satrms,trn_tst=trn_tst,
                thresh_trn=thresh_trn,numcen_tst=numcen_tst,
                min_numcen=min_numcen)

        orbcmb.flags()

        redo = orbcmb.rejection

        logger.info(f"redo: {redo}")

    # Convert the combined orbit to a sp3 dictionary
    orbcmb.to_sp3dict(sample_rate=orbit_sampling,sp3_header=sp3_header)

    # Write the sp3 dictionary into a sp3 file
    start_epoch = orbcmb.sp3_combined['data']['epochs'][0]
    start_year = start_epoch.year
    start_month = start_epoch.month
    start_day = start_epoch.day
    start_hour = start_epoch.hour
    start_minute = start_epoch.minute
    start_second = start_epoch.second
    start_gc = gpsCal()
    start_gc.set_yyyy_MM_dd_hh_mm_ss(start_year,start_month,start_day,
                                         start_hour,start_minute,start_second)
    start_doy = start_gc.ddd()
    orb_smp = int(orbit_sampling/60)
    smp = str(orb_smp).zfill(2) + 'M'
    cmb_sp3_filename = (cmb_name + str(vid) + camp_id + sol_id + '_'
                    + str(start_year).zfill(4) + str(start_doy).zfill(3)
                    + str(start_hour).zfill(2) + str(start_minute).zfill(2)
                    + '_' + len_data + '_' + smp + '_' + 'ORB' + '.SP3')

    orb = io_data.sp3(sp3file=prod_weekdir+'/'+cmb_sp3_filename,
                      sp3dict=orbcmb.sp3_combined)
    orb.write()


    # Write out the summary file
    orbrep = OrbitReport(orbcmb,sp3_subm_list,cmb_sp3_filename,prod_rootdir,
                        cmb_name,vid,camp_id,sol_id,author,contact,ac_acronyms,
                        rm_dv,dvsats,rf_align,transformations,sat_metadata_file)

    orbrep.eclipse(eop_file,eop_format)
    orbrep.summary()

    if solution == 'ultra-rapid':
        dowhr_line = f"day of week {dow}, hour {hr}.\n"
    else:
        dowhr_line = f"day of week {dow}.\n"

    logger.info(f"\nFinished the combination for GPS week "
                f"{str(gpsweek).zfill(4)} {dowhr_line}\n"
                f"The combined orbit is written into:\n"
                f"{prod_weekdir}/{cmb_sp3_filename}\n")

