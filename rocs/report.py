# Module for reporting and creating summary files

import logging
import numpy as np
import json
from rocs.gpscal import gpsCal
import rocs.io_data as io_data
import rocs.planets as planets
from rocs.eclipse import Eclipse

logger = logging.getLogger(__name__)

class OrbitReport:

    """
    Class of reporting for orbit combinations
    """

    def __init__(self,orbcmb,sp3_subm_list,cmb_sp3_filename,prod_rootdir,
            cmb_name,vid,camp_id,sol_id,author,contact,ac_acronyms,
            rm_dv,dvsats,rf_align,rf_transfo,sat_metadata_file):

        """
        Initialize report_orbits class

        Keyword arguments:
            orbcmb [object of class OrbitComb] : combined orbit
            sp3_subm_list [list] : list of used sp3 files for the combination
            cmb_sp3_filename [str] : filename for the combined sp3 file
            prod_rootdir [str] : root directory for products
            cmb_name [str] : 3-character combination name
            vid [int] : version identifier for the combination
            camp_id [str] : 3-character campaign/project specification
            sol_id [str] : 3-character solution identifier
            author [str] : creator of the combination and the report
            contact [str] : contact information of the author
            ac_acronyms [dict] : dictionary of AC acronyms
            rm_dv [bool] : if DV maneuvering satellites were to be removed
            dvsats [list] : list of DV maneuvering satellites
            rf_align  [list] : reference frame alignment options
            rf_transfo [dict] : dictionary of reference frame alignment
                                transformations
            sat_metadata_file [str] : used satellite metadata file

        """

        self.orbcmb = orbcmb
        self.sp3_subm_list = sp3_subm_list
        self.cmb_sp3_filename = cmb_sp3_filename
        self.cmb_name = cmb_name
        self.author = author
        self.contact = contact
        self.ac_acronyms = ac_acronyms
        self.rm_dv = rm_dv
        self.dvsats = dvsats
        self.rf_align = rf_align
        self.rf_transfo = rf_transfo
        self.sat_metadata_file = sat_metadata_file

        if  sat_metadata_file is not None:
            self.sat_metadata = io_data.SatelliteMetadata(sat_metadata_file)

        # solution specifications
        if sol_id == 'ULT':
            solution = 'ultra-rapid'
        elif sol_id == 'RAP':
            solution = 'rapid'
        elif sol_id == 'FIN':
            solution = 'final'
        elif sol_id == 'MIX':
            solution = 'mix'

        self.solution = solution

        if solution == 'ultra-rapid':
            len_data = '02D'
        else:
            len_data = '01D'

        if camp_id == 'DEM':
            campaign = 'demonstration'
        elif camp_id == 'MGX':
            campaign = 'Multi-GNSS Experiment'
        elif camp_id == 'OPS':
            campaign = 'operational'
        elif camp_id == 'TST':
            campaign = 'test'
        elif camp_id[0:1] == 'R':
            campaign = 'repro' + str(int(camp_id[1:]))

        self.campaign = campaign

        # datetime specifications
        start_epoch = orbcmb.sp3_combined['data']['epochs'][0]
        year = start_epoch.year
        month = start_epoch.month
        day = start_epoch.day
        hr = start_epoch.hour
        minute = start_epoch.minute
        second = start_epoch.second
        gc = gpsCal()
        gc.set_yyyy_MM_dd_hh_mm_ss(year,month,day,hr,minute,second)
        doy = gc.ddd()
        gpsweek = gc.wwww()
        dow = gc.dow()

        self.gpsweek = gpsweek
        self.dow = dow
        self.year = year
        self.doy = doy
        self.hr = hr
        self.minute = minute

        prod_weekdir = prod_rootdir + '/w' + str(gpsweek).zfill(4)

        # Full path and name of the eclipse report file
        self.ecl_fname = (prod_weekdir + '/' + 'eclipse_' + str(year).zfill(4)
                        + str(doy).zfill(3))

        # Full path and name of the SUM summary file
        self.sum_fname = (prod_weekdir + '/' + cmb_name + str(vid) + camp_id
                + sol_id + '_' + str(year).zfill(4) + str(doy).zfill(3)
                + str(hr).zfill(2) + str(minute).zfill(2) + '_' + len_data
                + '_' + len_data + '_' + 'SUM' + '.SUM')

        # Full path and name of the JSON summary file
        self.sumjson_fname = (prod_weekdir+'/'+cmb_name + str(vid) + camp_id
                + sol_id + '_' + str(year).zfill(4) + str(doy).zfill(3)
                + str(hr).zfill(2) + str(minute).zfill(2)
                + '_' + len_data + '_' + len_data + '_' + 'SUM' + '.JSON')

        if solution == 'ultra-rapid':
            end_epoch = orbcmb.sp3_combined['data']['epochs'][-1]
            year_end = end_epoch.year
            month_end = end_epoch.month
            day_end = end_epoch.day
            hr_end = end_epoch.hour
            minute_end = end_epoch.minute
            second_end = end_epoch.second
            gc_end = gpsCal()
            gc_end.set_yyyy_MM_dd_hh_mm_ss(year_end,month_end,day_end,hr_end,
                                            minute_end,second_end)
            wwww_end = gc_end.wwww()
            dow_end = gc_end.dow()
            doy_end = gc_end.ddd()

            self.wwww_end = wwww_end
            self.dow_end = dow_end
            self.year_end = year_end
            self.doy_end = doy_end
            self.hr_end = hr_end
            self.minute_end = minute_end

        # create list of satellite systems, blocks, satellites
        systems = []
        blocks  = []
        sats    = []

        for key in orbcmb.cen_rms:
            if isinstance(key,tuple):
                if isinstance(key[1],str):
                    if len(key[1]) == 1:
                        if key[1] not in systems:
                            systems.append(key[1])
                    else:
                        if key[1] not in blocks:
                            blocks.append(key[1])
                elif isinstance(key[1],tuple):
                    if key[1] not in sats:
                        sats.append(key[1])

        # Sort the systems list
        all_known_sys = ['G','R','E','C','J']
        sys_known = []
        sys_unknown = []
        for sys in systems:
            if sys in all_known_sys:
                sys_known.append(sys)
            else:
                sys_unknown.append(sys)
        sys_known.sort(key=lambda x: all_known_sys.index(x))
        sys_unknown.sort()
        systems = sys_known
        systems.extend(sys_unknown)

        # Sort the blocks list
        all_known_blk = ['GPS','GLO','GAL','BDS','QZS']
        blk_known = []
        blk_unknown = []
        for blk in blocks:
            if blk[0:3] in all_known_blk:
                blk_known.append(blk)
            else:
                blk_unknown.append(blk)
        blk_known.sort(key=lambda x: x[3:])
        blk_known.sort(key=lambda x: all_known_blk.index(x[0:3]))
        blk_unknown.sort()
        blocks = blk_known
        blocks.extend(blk_unknown)

        # Sort the satellites list
        sat_known = []
        sat_unknown = []
        for sat in sats:
            if sat[0] in all_known_sys:
                sat_known.append(sat)
            else:
                sat_unknown.append(sat)
        sat_known.sort(key=lambda x: x[1])
        sat_known.sort(key=lambda x: all_known_sys.index(x[0]))
        sat_unknown.sort(key=lambda x: x[1])
        sat_unknown.sort(key=lambda x: x[0])
        sats = sat_known
        sats.extend(sat_unknown)

        self.systems = systems
        self.blocks = blocks
        self.sats = sats

        # Detailed info on weighted and unweighted centers
        # per constellation/satellite
        weighted_cens = {}
        unweighted_cens = {}

        # Add any extra individual satellites
        satflags = orbcmb.satflags
        for acname in orbcmb.weighted_sats:

            # constellations that are fully weighted for this ac
            for sys_id in systems:
                sats_weighted = []
                sats_unweighted = []
                nsat_total = 0
                nsat_weighted = 0
                nsat_unweighted = 0
                for sat in sats:
                    if sat[0] == sys_id:
                        nsat_total += 1
                        if (satflags[acname,sat] not in ["missing_sat",
                            "missing_blk","missing_sys","excluded_sat",
                            "excluded_sat_all","unweighted_sys",
                            "unweighted_sat"]):
                            sats_weighted.append(sat)
                            nsat_weighted += 1
                        elif (satflags[acname,sat] in
                                ["unweighted_sat","unweighted_sys"]):
                            sats_unweighted.append(sat)
                            nsat_unweighted += 1
                if (nsat_weighted/nsat_total >= 0.5):
                    if acname not in weighted_cens:
                        weighted_cens[acname] = []
                    weighted_cens[acname].append(sys_id)
                    for sat in sats_unweighted:
                        if acname not in unweighted_cens:
                            unweighted_cens[acname] = []
                        unweighted_cens[acname].append(sat)
                elif (nsat_unweighted/nsat_total >= 0.5):
                    if acname not in unweighted_cens:
                        unweighted_cens[acname] = []
                    unweighted_cens[acname].append(sys_id)
                    for sat in sats_weighted:
                        if acname not in weighted_cens:
                            weighted_cens[acname] = []
                        weighted_cens[acname].append(sat)

        logger.debug(f"\nweighted_cens: {weighted_cens}")
        logger.debug(f"\nunweighted_cens: {unweighted_cens}")

        self.weighted_cens = weighted_cens
        self.unweighted_cens = unweighted_cens


    def eclipse(self,eop_file,eop_format):

        """
        create a report of eclipsing satellites

        """

        orbcmb = self.orbcmb

        # constants
        radius_earth = 6371000.0
        radius_moon = 1737100.0

        # Determine the satellites experiencing eclipses (needs an EOP
        # file for converting planetary coordinates from ECI to ECEF)
        ecl_earth = {}
        ecl_moon = {}

        if eop_format is not None:

            # Read EOP data
            time_utc = np.unique(orbcmb.epochs)
            eopdata = io_data.EOPdata(eop_file,eop_format)
            eopdata.get_eop(time_utc)
            eop = eopdata.eop_interp
            xp = eop[:,1]
            yp = eop[:,2]
            ut1_utc = eop[:,3]

            # calculate the Sun positions
            sun = planets.AnalyticalPosition(planet='sun',ref_frame='ECEF',
                                    time_utc=time_utc,ut1_utc=ut1_utc,
                                    xp=xp,yp=yp)
            r_sun = sun.r

            # calculate the moon position
            moon = planets.AnalyticalPosition(planet='moon',ref_frame='ECEF',
                                      time_utc=time_utc,ut1_utc=ut1_utc,
                                      xp=xp,yp=yp)
            r_moon = moon.r

            # eclipse report file
            ecl_file = open(self.ecl_fname,'w')
            header = ("Eclipse events experienced by satellites for week "
                        + str(self.gpsweek).zfill(4) + " day " + str(self.dow))
            ecl_file.write(f"{'-'*len(header)}\n")
            ecl_file.write(f"{header}\n")
            ecl_file.write(f"{'-'*len(header)}\n\n\n")

            # Earth-caused eclipses
            header = ("1) Earth-caused eclipse events")
            ecl_file.write(f" {header}\n")
            ecl_file.write(f" {'-'*len(header)}\n\n")
            header = (" PRN   SVN   DUR.(MIN)       ENTERING              "
                        "EXITING          Type   ")
            ecl_file.write(f" {header}\n")
            ecl_file.write(f" {'-'*len(header)}\n")
            for sat in self.sats:
                ind = np.where((orbcmb.satinfo[:,0]==sat[0])
                            & (orbcmb.satinfo[:,1]==sat[1]))
                r_sat = orbcmb.combined_orbit[ind]
                t_sat = np.array(orbcmb.epochs)[ind]
                ecl = Eclipse(r_sat,r_sun,'earth',radius_earth)
                ecl.get_ecl_times(t_sat)
                ecl_earth[sat] = ecl.eclipsing
                if ecl.eclipsing != 'none':
                    for item in ecl.ecl_times:
                        tfrom = item[0]
                        tto = item[1]
                        duration = ((tto-tfrom).seconds)/60.0
                        durmin = int(duration)
                        if (durmin==0):
                            dursec = int(duration*60.0)
                        else:
                            dursec = int(duration%durmin)
                        line = (f" {sat[0]}{sat[1]:02}   {sat[0]}{sat[2]:03}   "
                                f"{durmin:>3}:{dursec:02}   "
                                f"{tfrom.year:>4}-{tfrom.month:02}-{tfrom.day:02} "
                                f"{tfrom.hour:02}:{tfrom.minute:02}:{tfrom.second:02}"
                                f"   "
                                f"{tto.year:>4}-{tto.month:02}-{tto.day:02} "
                                f"{tto.hour:02}:{tto.minute:02}:{tto.second:02}"
                                f"   {ecl.eclipsing}")
                        ecl_file.write(f" {line}\n")

            # Moon-caused eclipses
            ecl_file.write("\n\n")
            header = ("2) Moon-caused eclipse events")
            ecl_file.write(f" {header}\n")
            ecl_file.write(f" {'-'*len(header)}\n\n")
            header = (" PRN   SVN   DUR.(MIN)       ENTERING              "
                        "EXITING          Type   ")
            ecl_file.write(f" {header}\n")
            ecl_file.write(f" {'-'*len(header)}\n")
            for sat in self.sats:
                ind = np.where((orbcmb.satinfo[:,0]==sat[0])
                            & (orbcmb.satinfo[:,1]==sat[1]))
                r_sat = orbcmb.combined_orbit[ind]
                t_sat = np.array(orbcmb.epochs)[ind]
                ecl = Eclipse(r_sat,r_sun,'moon',radius_moon,r_moon)
                ecl.get_ecl_times(t_sat)
                ecl_moon[sat] = ecl.eclipsing
                if ecl.eclipsing != 'none':
                    for item in ecl.ecl_times:
                        tfrom = item[0]
                        tto = item[1]
                        duration = ((tto-tfrom).seconds)/60.0
                        durmin = int(duration)
                        if (durmin==0):
                            dursec = int(duration*60.0)
                        else:
                            dursec = int(duration%durmin)
                        line = (f" {sat[0]}{sat[1]:02}   {sat[0]}{sat[2]:03}   "
                                f"{durmin:>3}:{dursec:02}   "
                                f"{tfrom.year:>4}-{tfrom.month:02}-{tfrom.day:02} "
                                f"{tfrom.hour:02}:{tfrom.minute:02}:{tfrom.second:02}"
                                f"   "
                                f"{tto.year:>4}-{tto.month:02}-{tto.day:02} "
                                f"{tto.hour:02}:{tto.minute:02}:{tto.second:02}"
                                f"   {ecl.eclipsing}")
                        ecl_file.write(f" {line}\n")

        self.ecl_earth = ecl_earth
        self.ecl_moon = ecl_moon


    def summary(self):

        """
        generate daily reports (or 2-day for ultra-rapid combinations)

        """

        orbcmb = self.orbcmb

        # initiate summary file
        sumfull = open(self.sum_fname,'w')

        # dictionary for creating json file
        sumdict = {}

        header1 = (self.cmb_name + " " + self.campaign + " " + self.solution
                    + " orbit combination for:")
        header2 = ("week "+ str(self.gpsweek).zfill(4) + " day "
                    + str(self.dow) + " (year " + str(self.year)
                    + " doy " + str(self.doy).zfill(3) + ") ")

        if self.solution == 'ultra-rapid':
            header3 = (str(self.hr).zfill(2) + ":" + str(self.minute).zfill(2))
            header4 = (" to week " + str(self.wwww_end).zfill(4) + " day "
                        + str(self.dow_end) + " (year "
                        + str(self.year_end) + " doy "
                        + str(self.doy_end).zfill(3) + ") ")
            header5 = (str(self.hr_end).zfill(2) + ":" + str(self.minute_end).zfill(2))
            header6 = ("The first 24 hours are observed, but the last 24 hours are "
                        "predicted orbits")

        if self.solution == 'ultra-rapid':
            lenmax = max(len(header1),(len(header2)+len(header3)+len(header4)
                                       +len(header5)),len(header6))
            sumfull.write(f"{'-'*lenmax}\n")
            sumfull.write(f"{header1}\n")
            sumfull.write(f"{header2}")
            sumfull.write(f"{header3}")
            sumfull.write(f"{header4}")
            sumfull.write(f"{header5}\n")
            sumfull.write(f"{header6}\n")
            sumfull.write(f"{'-'*lenmax}\n")
        else:
            lenmax = max(len(header1),len(header2))
            sumfull.write(f"{'-'*lenmax}\n")
            sumfull.write(f"{header1}\n")
            sumfull.write(f"{header2}\n")
            sumfull.write(f"{'-'*lenmax}\n")

        if self.campaign == 'demonstration':
            expt_comment = ("* * * THIS COMBINATION IS STRICTLY EXPERIMENTAL "
                            "-- USE WITH CAUTION * * *")
            sumfull.write(f"{expt_comment}\n\n")

        sumfull.write(f" Author: {self.author}\n")
        sumfull.write(f" Contact: {self.contact}\n\n")

        sumdict['header'] = {}
        sumdict['header']['title'] = (self.cmb_name + " " + self.campaign + " "
                                        + self.solution + " orbit combination")
        sumdict['header']['gps week'] = self.gpsweek
        sumdict['header']['day of week'] = self.dow
        sumdict['header']['year'] = self.year
        sumdict['header']['day of year'] = self.doy
        sumdict['header']['author'] = self.author
        sumdict['header']['contact'] = self.contact

        # solution/centers list
        solution_names = {}
        for sp3file in self.sp3_subm_list:
            acname = sp3file[-38:-35]
            solution_names[acname] = sp3file[-38:]
        centers = []
        for acname in solution_names:
            centers.append(acname)
        centers.sort()
        if "IGV" in centers:
            centers.remove("IGV")
            centers.append("IGV")

        sumfull.write(" All AC solutions:\n")
        if solution_names:
            for ac in centers:
                if ac == "IGV":
                    sumfull.write(f"  - {ac}")
                else:
                    sumfull.write(f"  - {ac} = {solution_names[ac]}")
                if ac in self.ac_acronyms:
                    sumfull.write(f" : {self.ac_acronyms[ac]}\n")
                else:
                    sumfull.write("\n")
        else:
            sumfull.write("  - None\n")
        sumfull.write("\n\n")

        sumfull.write(" AC solutions used in the combination:\n\n")
        if self.weighted_cens:
            header = " AC  | Sat. System or PRN/SVN "
            sumfull.write(f" {header}\n")
            lines = []
            len_lines = []
            aclist = [key for key, value in self.weighted_cens.items()
                        if isinstance(value, list) and len(value) > 0]
            aclist.sort()
            lines.append("-")
            len_lines.append(len(header))
            for acname in aclist:
                line = " " + acname + " |"
                c = 0
                for item in self.weighted_cens[acname]:
                    c += 1
                    if isinstance(item,str):
                        line = line + " " + item
                    else:
                        line = (line + " " + item[0] + str(item[1]).zfill(2)
                                + "/" + item[0] + str(item[2]).zfill(3))
                    if c == 7:
                        lines.append(line)
                        len_lines.append(len(line))
                        line = " " + acname + "   "
                        c = 0
                if c > 0:
                    lines.append(line)
                    len_lines.append(len(line))
            lenmax = max(len_lines)
            for line in lines:
                if '-' in line:
                    sumfull.write(f" {'-'*lenmax}\n")
                else:
                    sumfull.write(f" {line}\n")
        else:
            sumfull.write(" No weighted center!\n")
        sumfull.write("\n\n")

        sumfull.write(" AC solutions not used in the combination (for comparison):\n\n")
        if self.unweighted_cens:
            header = " AC  | Sat. System or PRN/SVN "
            sumfull.write(f" {header}\n")
            lines = []
            len_lines = []
            aclist = [key for key, value in self.unweighted_cens.items()
                        if isinstance(value, list) and len(value) > 0]
            aclist.sort()
            if "IGV" in aclist:
                aclist.remove("IGV")
                aclist.append("IGV")
            lines.append("-")
            len_lines.append(len(header))
            for acname in aclist:
                line = " " + acname + " |"
                c = 0
                for item in self.unweighted_cens[acname]:
                    c += 1
                    if isinstance(item,str):
                        line = line + " " + item
                    else:
                        line = (line + " " + item[0] + str(item[1]).zfill(2)
                                + "/" + item[0] + str(item[2]).zfill(3))
                    if c == 7:
                        lines.append(line)
                        len_lines.append(len(line))
                        line = " " + acname + "   "
                        c = 0
                if c > 0:
                    lines.append(line)
                    len_lines.append(len(line))
            lenmax = max(len_lines)
            for line in lines:
                if '-' in line:
                    sumfull.write(f" {'-'*lenmax}\n")
                else:
                    sumfull.write(f" {line}\n")
        else:
            sumfull.write(" None\n")
        sumfull.write("\n\n")

        sumfull.write(" Combined solution:\n")
        sumfull.write(f"  - {self.cmb_sp3_filename[0:3]} = {self.cmb_sp3_filename}\n\n")

        sumfull.write(f" IGS satellite metadata file used:\n")
        sumfull.write(f"  - {self.sat_metadata_file.split('/')[-1]}\n\n")
        sumfull.write(f" AC weighting method:\n")
        sumfull.write(f"  - {orbcmb.cen_wht_method}\n\n")
        sumfull.write(f" Orbit sampling for combination:\n")
        orb_smp = int(orbcmb.sample_rate/60)
        sumfull.write(f"  - {orb_smp} minutes\n\n\n\n")

        sumdict['header']['weighted solutions'] = self.weighted_cens
        sumdict['header']['unweighted solutions for comparison'] = self.unweighted_cens
        sumdict['header']['combined solution'] = {}
        sumdict['header']['combined solution'][self.cmb_sp3_filename[0:3]] = self.cmb_sp3_filename
        sumdict['header']['ac acronyms'] = self.ac_acronyms
        sumdict['header']['satellite metadata'] = self.sat_metadata_file.split('/')[-1]
        sumdict['header']['AC weighting method'] = orbcmb.cen_wht_method
        sumdict['header']['orbit interval'] = orb_smp

        # Outliers, exclusions and pre-processing
        sumdict['preprocess'] = {}
        sumdict['preprocess']['excluded due to high rms sats'] = []
        sumdict['preprocess']['unweighted due to too many high rms sats'] = []
        sumdict['preprocess']['unweighted due to high transformations'] = {}
        sumdict['preprocess']['excluded sats due to low number of centers'] = []
        sumdict['preprocess']['DV maneuvering satellites'] = []

        sumdict['events'] = {}
        sumdict['events']['E'] = []
        sumdict['events']['M'] = []
        sumdict['events']['V'] = []

        header = "1) Outliers, exclusions and pre-processing:"
        sumfull.write(f" {header}\n")
        sumfull.write(f" {'-'*len(header)}\n\n\n")

        # Sat/cen excluded due to high rms
        header = "1.1) Satellites excluded from AC solutions due to high rms:"
        sumfull.write(f" {header}\n")
        sumfull.write(f" {'-'*len(header)}\n\n")
        if orbcmb.exclude_highrms:
            header = "  AC | PRN | SVN "
            sumfull.write(f" {header}\n")
            sumfull.write(f" {'-'*len(header)}\n")
            for item in orbcmb.exclude_highrms:
                line = (" " + item[0][0:3] + " | " + item[1][0]
                            + str(item[1][1]).zfill(2) + " | " + item[1][0]
                            + str(item[1][2]).zfill(3))
                sumfull.write(f" {line}\n")
                sumdict['preprocess']['excluded due to high rms sats'].append((item[0][0:3],
                                        item[1][0]+str(item[1][1]).zfill(2),
                                        item[1][0]+str(item[1][2]).zfill(3)))
        else:
            sumfull.write(" No exclusions!\n")
        sumfull.write("\n\n")

        # Centers unweighted due to too many sat exclusions
        header = ("1.2) AC solutions unweighted due to too many satellite "
                    + f"exclusions (more than {orbcmb.max_high_satrms}):")
        sumfull.write(f" {header}\n")
        sumfull.write(f" {'-'*len(header)}\n\n")
        if orbcmb.unweighted_max_high_satrms:
            for item in orbcmb.unweighted_max_high_satrms:
                line = item[0:3]
                sumfull.write(f" {line}\n")
                sumdict['preprocess']['unweighted due to too many high rms sats'].append(line)
        else:
            sumfull.write(" No AC solution unweighted!\n")
        sumfull.write("\n\n")

       # Centers unweighted due to high transformation parameters
        header1 = ("1.3) AC solutions unweighted due to high Helmert "
                    + f"transformation parameters with")
        header2 = "     respect to the combined orbit:"
        sumfull.write(f" {header1}\n{header2}\n")
        sumfull.write(f" {'-'*len(header1)}\n\n")
        trn_params = ['Tx','TY','TZ','RX','RY','RZ','SC']
        trn_units = ['mm','mm','mm','uas','uas','uas','ppb']
        if orbcmb.unweighted_high_tra:
            for item in orbcmb.unweighted_high_tra:
                acname = item[0]
                trn_param = trn_params[item[1]]
                trn_unit = trn_units[item[1]]
                trn_val = orbcmb.transform_params[acname][item[1]]
                line = (f"{acname[0:3]} : {trn_param} = {trn_val*1e6:6.2f}"
                        f" {trn_unit}")
                sumfull.write(f" {line}\n")
                sumdict['preprocess']['unweighted due to high transformations'][
                        str((acname[0:3],trn_param))
                        ] = trn_val*1000.0
        else:
            sumfull.write(" No AC solution unweighted!\n")
        sumfull.write("\n\n")

        # Sat excluded due to low number of centers
        header = ("1.4) Satellites excluded due to low number of AC solutions "
                    + f"(lower than {orbcmb.min_numcen}):")
        sumfull.write(f" {header}\n")
        sumfull.write(f" {'-'*len(header)}\n\n")
        if orbcmb.exclude_lowcen:
            header = " PRN | SVN "
            sumfull.write(f" {header}\n")
            sumfull.write(f" {'-'*len(header)}\n")
            for item in orbcmb.exclude_lowcen:
                line = (" " + item[0] + str(item[1]).zfill(2) + " | " + item[0]
                            + str(item[2]).zfill(3))
                sumfull.write(f" {line}\n")
                sumdict['preprocess']['excluded sats due to low number of centers'].append(
                        (item[0] + str(item[1]).zfill(2),
                        item[0] + str(item[2]).zfill(3)))
        else:
            sumfull.write(" No exclusions!\n")
        sumfull.write("\n\n")

        # maneuvering satellites
        header = ("1.5) Satellites experiencing Delta-V maneuvers: ")
        sumfull.write(f" {header}\n")
        sumfull.write(f" {'-'*len(header)}\n\n")
        if self.year >= 2004:
            comment = ("Only COD solutions, which model maneuvers, are"
                        + " used after a Delta-V maneuver\n"
                        + " until the end of day, if available.")
        else:
            comment = ("Solutions are removed after a Delta-V maneuver\n"
                        + "until the end of day.")
        sumfull.write(f" {comment}\n\n")
        if self.rm_dv and self.dvsats:
            header = " PRN | SVN  |       from       |        to        "
            sumfull.write(f" {header}\n")
            sumfull.write(f" {'-'*len(header)}\n")
            for item in self.dvsats:
                line = (" " + item[2] + " | " + item[3] + " | "
                            + str(item[0].year) + "_" + str(item[0].month).zfill(2) + "_"
                            + str(item[0].day).zfill(2) + " " + str(item[0].hour).zfill(2)
                            + ":" + str(item[0].minute).zfill(2) + " | "
                            + str(item[1].year) + "_" + str(item[1].month).zfill(2) + "_"
                            + str(item[1].day).zfill(2) + " " + str(item[1].hour).zfill(2)
                            + ":" + str(item[1].minute).zfill(2))
                sumfull.write(f" {line}\n")
                sumdict['preprocess']['DV maneuvering satellites'].append(
                        [item[2],item[3],item[0],item[1]])
        else:
            sumfull.write(" No Delta-V maneuvers!\n")
        sumfull.write("\n\n")

        # pre-alignment of the orbits
        if any(self.rf_align):
            header = ("1.6) Pre-alignment of the orbits: ")
            sumfull.write(f" {header}\n")
            sumfull.write(f" {'-'*len(header)}\n\n")
            comment = ("transformation parameters applied to orbit solutions prior"
                         + " to the combination.\n\n"
                         + " The transformation parameters are estimated during the"
                         + " SINEX combination process,\n"
                         + " performed by the IGS Reference Frame Coordinator.")
            sumfull.write(f" {comment}\n\n")
            if self.year < 2017 and self.campaign == 'repro3':
                comment = ("Additional corrections were added to the Z rotations"
                            + " (RZ) applied to the MIT\n"
                            + " orbit solutions, based on the differences between their"
                            + " a priori and observed\n"
                            + " values of UT1.")
                sumfull.write(f" {comment}\n\n")
            trn_header1 = ['Tx','TY','TZ','RX','RY','RZ','SC']
            trn_header1_aln = []
            sumdict['preprocess']['pre-alignment'] = {}
            for item in trn_header1:
                trn_header1_aln.append(item.center(8))
            trn_header2 = [' [mm]',' [mm]',' [mm]','[uas]','[uas]','[uas]','[ppb]']
            trn_header2_aln = []
            for item in trn_header2:
                trn_header2_aln.append(item.center(8))
            header1 = f"      {'|'.join([item for item in trn_header1_aln])}"
            header2 = f"  AC |{'|'.join([item for item in trn_header2_aln])}"
            sumfull.write(f" {header1}\n")
            sumfull.write(f" {header2}\n")
            sumfull.write(f" {'-'*len(header2)}\n")

            for acname in self.rf_transfo[self.dow]:
                trn = []
                sumdict['preprocess']['pre-alignment'][acname] = {}
                sumdict['preprocess']['pre-alignment'][acname]['T'] = []
                sumdict['preprocess']['pre-alignment'][acname]['R'] = []
                sumdict['preprocess']['pre-alignment'][acname]['S'] = []
                for c,item in enumerate(self.rf_transfo[self.dow][acname]):
                    if c < 3:
                        trn_str = f"{item*1000:6.1f}"
                        if abs(item*1000) > 999.9:
                            trn_str = f"{999:6.0f}"
                        sumdict['preprocess']['pre-alignment'][acname]['T'].append(item*1000)
                    elif c < 6:
                        trn_str = f"{item*1e6*3600*180/np.pi:6.0f}"
                        if abs(item*1e6*3600*180/np.pi) > 99999:
                            trn_str = f"{99999:6.0f}"
                        sumdict['preprocess']['pre-alignment'][acname]['R'].append(
                                                            item*1000*3600*180/np.pi)
                    else:
                        trn_str = f"{(item-1.0)*1e9:6.2f}"
                        if abs((item-1.0)*1e9) > 999.99:
                            trn_str = f"{999:6.0f}"
                        sumdict['preprocess']['pre-alignment'][acname]['S'].append(
                                                                        (item-1.0)*1e9)
                    trn.append(trn_str)
                line = f" {acname} | {' | '.join([item for item in trn])}"
                sumfull.write(f" {line}\n")
            sumfull.write("\n\n")
        sumfull.write("\n")

        # center weights
        weighted_centers = orbcmb.weighted_centers
        unweighted_centers = orbcmb.unweighted_centers
        weighted_sats = orbcmb.weighted_sats
        unweighted_sats = orbcmb.unweighted_sats
        cen_wht_method = orbcmb.cen_wht_method

        centers_str = []
        for ac in weighted_centers:
            centers_str.append(ac+ ' ')
        centers_str = centers_str + unweighted_centers

        sumdict['AC weights'] = {}
        header = f"2) AC weights [%] -- {cen_wht_method} weighting:"
        sumfull.write(f" {header}\n")
        sumfull.write(f" {'-'*len(header)}\n\n")

        if cen_wht_method == 'global':
            logger.info(f"\nCenter weights ({cen_wht_method} weighting):\n")
            logger.info(f"           | {' | '.join([ac for ac in centers_str])}")
            logger.info(f"-----------{'-------'*len(centers_str)}")
            header = f"        |   {'   |   '.join([ac for ac in centers])}"
            sumfull.write(f" {header}\n")
            sumfull.write(f" {'-'*(len(header)+3)}\n")

            wht = []
            whtfull = []
            sumwht = 0
            for acname in centers:
                if acname in weighted_centers:
                    sumwht += orbcmb.cen_weights[acname]
            for acname in centers:
                if acname in weighted_centers:
                    wht_percent = 100.0*orbcmb.cen_weights[acname]/sumwht
                else:
                    wht_percent = 0.0
                sumdict['AC weights'][acname] = wht_percent
                wht_str = f"{wht_percent:^4.0f}"
                wht.append(wht_str)
                whtfull_str = f"{wht_percent:^7.3f}"
                whtfull.append(whtfull_str)
            logger.info(f" Weight | {' | '.join([item for item in wht])}")
            line = f" Weight | {' | '.join([item for item in whtfull])}"
            sumfull.write(f" {line}\n\n\n")

        elif cen_wht_method == 'by_constellation':
            logger.info(f"\nCenter weights (%) ({cen_wht_method} weighting):\n")
            logger.info(f" Sat. System | {' | '.join([ac for ac in centers])}")
            logger.info(f"--------------{'------'*len(centers)}")
            header = f"  Sat."
            sumfull.write(f" {header}\n")
            header = f" System |   {'   |   '.join([ac for ac in centers])}"
            sumfull.write(f" {header}\n")
            sumfull.write(f" {'-'*(len(header)+3)}\n")

            for sys in self.systems:
                wht = []
                whtfull = []
                sumwht = 0
                sumdict['AC weights'][sys] = {}
                for acname in centers:
                    if (acname,sys) in orbcmb.cen_weights:
                        sumwht += orbcmb.cen_weights[acname,sys]
                for acname in centers:
                    if (acname,sys) in orbcmb.cen_weights:
                        wht_percent = 100.0*orbcmb.cen_weights[acname,sys]/sumwht
                    else:
                        wht_percent = 0.0
                    sumdict['AC weights'][sys][acname] = wht_percent
                    wht_str = f"{wht_percent:^4.0f}"
                    whtfull_str = f"{wht_percent:7.3f}"
                    wht.append(wht_str)
                    whtfull.append(whtfull_str)
                logger.info(f"      {sys}      | "
                            f"{' | '.join([item for item in wht])}")
                line = (f"   {sys}    | "
                        + f"{' | '.join([item for item in whtfull])}")
                sumfull.write(f" {line}\n")

        elif cen_wht_method == 'by_block':
            logger.info(f"\nCenter weights (%) ({cen_wht_method} weighting):\n")
            logger.info(f"   Block        | {' | '.join([ac for ac in centers])}")
            logger.info(f"-----------------{'------'*len(centers)}")
            header = f"     Block     |   {'   |   '.join([ac for ac in centers])}"
            sumfull.write(f" {header}\n")
            sumfull.write(f" {'-'*(len(header)+3)}\n")
            for blk in self.blocks:
                wht = []
                whtfull = []
                sumwht = 0
                sumdict['AC weights'][blk] = {}
                for acname in centers:
                    if (acname,blk) in orbcmb.cen_weights:
                        sumwht += orbcmb.cen_weights[acname,blk]
                for acname in centers:
                    if (acname,blk) in orbcmb.cen_weights:
                        wht_percent = 100.0*orbcmb.cen_weights[acname,blk]/sumwht
                    else:
                        wht_percent = 0.0
                    sumdict['AC weights'][blk][acname] = wht_percent
                    wht_str = f"{wht_percent:^4.0f}"
                    whtfull_str = f"{wht_percent:7.3f}"
                    wht.append(wht_str)
                    whtfull.append(whtfull_str)
                logger.info(f"{blk:15} | {' | '.join([item for item in wht])}")
                line = (f"{blk:15}| "
                        + f"{' | '.join([item for item in whtfull])}")
                sumfull.write(f" {line}\n")

        elif cen_wht_method == 'by_sat':
            logger.info("\n\nCenter weight tables separated for each block\n")
            for blk in self.blocks:
                logger.info(f"\n\nCenter weights (%) for block {blk} "
                            f"({cen_wht_method} weighting):\n")
                logger.info(f" PRN |  SVN | {' | '.join([ac for ac in centers])}")
                logger.info(f"-------------{'------'*len(centers)}")
                for sat in self.sats:
                    sat_id = self.sat_metadata.get_sat_identifier(sat[0],sat[2])
                    if sat_id.block == blk:
                        wht = []
                        whtfull = []
                        sumwht = 0
                        for acname in centers:
                            if (acname,sat[0],sat[1],sat[2]) in orbcmb.cen_weights:
                                sumwht += (orbcmb.cen_weights
                                                    [acname,sat[0],sat[1],sat[2]])
                        for acname in centers:
                            if (acname,sat[0],sat[1],sat[2]) in orbcmb.cen_weights:
                                wht_percent = (100.0*orbcmb.cen_weights
                                            [acname,sat[0],sat[1],sat[2]]/sumwht)
                                wht_str = f"{wht_percent:^4.0f}"
                                whtfull_str = f"{wht_percent:7.3f}"
                            else:
                                wht_str = "   "
                                whtfull_str = "      "
                            wht.append(wht_str)
                            whtfull.append(whtfull_str)
                        prn = sat[0] + str(sat[1]).zfill(2)
                        svn = sat[0] + str(sat[2]).zfill(3)
                        logger.info(f" {prn} | {svn} | "
                                    f"{' | '.join([item for item in wht])}")
            logger.info(f"\n\nCenter weights (%) for all satellites "
                        f"({cen_wht_method} weighting):\n")
            logger.info(f" PRN |  SVN | {' | '.join([ac for ac in centers])}")
            logger.info(f"-------------{'------'*len(centers)}")
            header = f" PRN | SVN  |   {'  |   '.join([ac for ac in centers])}"
            sumfull.write(f" {header}\n")
            sumfull.write(f" {'-'*(len(header)+2)}\n")

            for sat in self.sats:
                wht = []
                whtfull = []
                sumwht = 0
                prn = sat[0] + str(sat[1]).zfill(2)
                svn = sat[0] + str(sat[2]).zfill(3)
                sumdict['AC weights'][str((prn,svn))] = {}
                for acname in centers:
                    if (acname,sat[0],sat[1],sat[2]) in orbcmb.cen_weights:
                        sumwht += orbcmb.cen_weights[acname,sat[0],sat[1],sat[2]]
                for acname in centers:
                    if (acname,sat[0],sat[1],sat[2]) in orbcmb.cen_weights:
                        wht_percent = (100.0*orbcmb.cen_weights
                                    [acname,sat[0],sat[1],sat[2]]/sumwht)
                    else:
                        wht_percent = 0.0
                    sumdict['AC weights'][str((prn,svn))][acname] = wht_percent
                    wht_str = f"{wht_percent:^4.0f}"
                    whtfull_str = f"{wht_percent:6.2f}"
                    wht.append(wht_str)
                    whtfull.append(whtfull_str)
                logger.info(f" {prn} | {svn} | "
                            f"{' | '.join([item for item in wht])}")
                line = (f" {prn} | {svn} | "
                        + f"{' | '.join([item for item in whtfull])}")
                sumfull.write(f" {line}\n")
            sumfull.write("\n\n")

        logger.info("\n")
        sumfull.write("\n")

        # RMS statistics
        header = f"3) Orbit combination RMS statistics:"
        sumfull.write(f" {header}\n")
        sumfull.write(f" {'-'*len(header)}\n\n\n")
        sumdict['RMS'] = {}

        logger.info("\nOveral RMS statistics of the centers:\n")
        logger.info(f"         |  {'  |  '.join([ac for ac in centers_str])}")
        logger.info(f"---------{'---------'*len(centers_str)}")
        header = "3.1) Overall RMS [mm] of the ACs:"
        sumfull.write(f" {header}\n")
        sumfull.write(f" {'-'*len(header)}\n\n")
        header = f"     |   {'  |   '.join([ac for ac in centers])}"
        sumfull.write(f" {header}\n")
        sumfull.write(f" {'-'*(len(header)+2)}\n")

        rms = []
        rmsfull = []
        sumdict['RMS'] = {}
        for acname in centers:
            if acname in orbcmb.cen_rms:
                ac = acname
            else:
                ac = acname + "c"
            sumdict['RMS'][acname] = orbcmb.cen_rms[ac]*1000
            rms_str = f"{orbcmb.cen_rms[ac]*1000:^6.0f}"
            rms.append(rms_str)
            rmsfull_str = f"{orbcmb.cen_rms[ac]*1000:6.1f}"
            if orbcmb.cen_rms[ac]*1000 > 9999.9:
                rmsfull_str = f"{9999:6.0f}"
            rmsfull.append(rmsfull_str)
        logger.info(f"RMS(mm)  | {' | '.join([item for item in rms])}")
        line = f" RMS | {' | '.join([item for item in rmsfull])}"
        sumfull.write(f" {line}\n\n\n")

        logger.info("\n\nCenter RMS statistics (mm) by constellation:\n")
        logger.info(f" Sat. System |  {'  |  '.join([ac for ac in centers_str])}")
        logger.info(f"-------------{'---------'*len(centers_str)}")
        header = "3.2) AC RMS [mm] by constellation:"
        sumfull.write(f" {header}\n")
        sumfull.write(f" {'-'*len(header)}\n\n")
        header = f"  Sat."
        sumfull.write(f" {header}\n")
        header = f" System |   {'  |   '.join([ac for ac in centers])}"
        sumfull.write(f" {header}\n")
        sumfull.write(f" {'-'*(len(header)+2)}\n")

        for sys in self.systems:
            rms = []
            rmsfull = []
            sumdict['RMS'][sys] = {}
            for acname in centers:
                if acname in orbcmb.cen_rms:
                    ac = acname
                else:
                    ac = acname + "c"
                if (ac,sys) in orbcmb.cen_rms:
                    rms_str = f"{orbcmb.cen_rms[ac,sys]*1000:^6.0f}"
                    rmsfull_str = f"{orbcmb.cen_rms[ac,sys]*1000:6.1f}"
                    if orbcmb.cen_rms[ac,sys]*1000 > 9999.9:
                        rmsfull_str = f"{9999:6.0f}"
                    sumdict['RMS'][sys][acname] = orbcmb.cen_rms[ac,sys]*1000
                else:
                    rms_str = "      "
                    rmsfull_str = "      "
                rms.append(rms_str)
                rmsfull.append(rmsfull_str)
            logger.info(f"      {sys}      | "
                        f"{' | '.join([item for item in rms])}")
            line = (f"   {sys}    | "
                    + f"{' | '.join([item for item in rmsfull])}")
            sumfull.write(f" {line}\n")
        sumfull.write("\n\n")

        logger.info("\n\nCenter RMS statistics (mm) by block:\n")
        logger.info(
            f"   Block        |  {'  |  '.join([ac for ac in centers_str])}")
        logger.info(f"----------------{'---------'*len(centers_str)}")
        header = "3.3) AC RMS [mm] by block:"
        sumfull.write(f" {header}\n")
        sumfull.write(f" {'-'*len(header)}\n\n")
        header = f"     Block     |   {'  |   '.join([ac for ac in centers])}"
        sumfull.write(f" {header}\n")
        sumfull.write(f" {'-'*(len(header)+2)}\n")

        for blk in self.blocks:
            rms = []
            rmsfull = []
            sumdict['RMS'][blk] = {}
            for acname in centers:
                if acname in orbcmb.cen_rms:
                    ac = acname
                else:
                    ac = acname + "c"
                if (ac,blk) in orbcmb.cen_rms:
                    rms_str = f"{orbcmb.cen_rms[ac,blk]*1000:^6.0f}"
                    rmsfull_str = f"{orbcmb.cen_rms[ac,blk]*1000:6.1f}"
                    if orbcmb.cen_rms[ac,blk]*1000 > 9999.9:
                        rmsfull_str = f"{9999:6.0f}"
                    sumdict['RMS'][blk][acname] = orbcmb.cen_rms[ac,blk]*1000
                else:
                    rms_str = "      "
                    rmsfull_str = "      "
                rms.append(rms_str)
                rmsfull.append(rmsfull_str)
            logger.info(f"{blk:15} | {' | '.join([item for item in rms])}")
            line = (f"{blk:15}| "
                    + f"{' | '.join([item for item in rmsfull])}")
            sumfull.write(f" {line}\n")
        sumfull.write("\n\n")

        logger.info("\n\nCenter RMS statistics (mm) for all satellites:\n")
        logger.info(f" PRN |  SVN |  "
                    f"{'  |  '.join([ac for ac in centers_str])}  |   IGS")
        logger.info(f"------------{'---------'*(len(centers_str)+1)}")
        header = "3.4) AC RMS [mm] by satellite:"
        sumfull.write(f" {header}\n")
        sumfull.write(f" {'-'*len(header)}\n\n")
        comment = ("Event codes:\n"
                + " E: satellite eclipsing caused by Earth\n"
                + " M: satellite eclipsing caused by Moon\n"
                + " V: satellite maneuvering (Delta-V)")
        sumfull.write(f" {comment}\n\n")
        header = (f" PRN | SVN  |   {'  |   '.join([ac for ac in centers])}"
                + f"  | Overall | event ")
        sumfull.write(f" {header}\n")
        sumfull.write(f" {'-'*len(header)}\n")

        for sat in self.sats:
            rms = []
            rmsfull = []
            prn = sat[0] + str(sat[1]).zfill(2)
            svn = sat[0] + str(sat[2]).zfill(3)
            sumdict['RMS'][str((prn,svn))] = {}
            for acname in centers:
                if acname in orbcmb.cen_rms:
                    ac = acname
                else:
                    ac = acname + "c"
                if (ac,sat) in orbcmb.cen_rms:
                    rms_str = f"{orbcmb.cen_rms[ac,sat]*1000:^6.0f}"
                    rmsfull_str = f"{orbcmb.cen_rms[ac,sat]*1000:6.1f}"
                    if orbcmb.cen_rms[ac,sat]*1000 > 9999.9:
                        rmsfull_str = f"{9999:6.0f}"
                    sumdict['RMS'][str((prn,svn))][acname] = orbcmb.cen_rms[ac,sat]*1000
                else:
                    rms_str = "      "
                    rmsfull_str = "      "
                rms.append(rms_str)
                rmsfull.append(rmsfull_str)
            if sat in orbcmb.sat_rms:
                rms_str = f"{orbcmb.sat_rms[sat]*1000:^6.0f}"
                rmsfull_str = f"{orbcmb.sat_rms[sat]*1000:7.1f}"
                if orbcmb.sat_rms[sat]*1000 > 99999.9:
                    rmsfull_str = f"{99999:7.0f}"
                sumdict['RMS'][str((prn,svn))]["Overall"] = orbcmb.sat_rms[sat]*1000
            else:
                rms_str = "      "
                rmsfull_str = "       "
            rms.append(rms_str)
            rmsfull.append(rmsfull_str)

            event =  list("     ")
            if (hasattr(self,'ecl_earth') and sat in self.ecl_earth
                    and self.ecl_earth[sat] in ['full','partial']):
                event[1] = "E"
                sumdict['events']['E'].append(str((prn,svn)))
            if (hasattr(self,'ecl_moon') and sat in self.ecl_moon
                    and self.ecl_moon[sat] in ['full','partial']):
                event[2] = "M"
                sumdict['events']['M'].append(str((prn,svn)))
            if self.rm_dv and self.dvsats:
                for item in self.dvsats:
                    if (item[2] == prn):
                        event[3] = "V"
                        sumdict['events']['V'].append(str((prn,svn)))
            event_str = "".join(event)
            logger.info(f" {prn} | {svn} | {' | '.join([item for item in rms])}")
            line = (f" {prn} | {svn} | "
                    + f"{' | '.join([item for item in rmsfull])}"
                    + f" | {event_str}")
            sumfull.write(f" {line}\n")

        logger.info("\n")

        logger.info("\n")
        sumfull.write("\n")

        logger.info("\nOveral absolute deviation statistics of the centers:\n")
        logger.info(f"         |  {'  |  '.join([ac for ac in centers_str])}")
        logger.info(f"---------{'---------'*len(centers_str)}")
        abdev = []
        abdevfull = []
        for acname in centers:
            if acname in orbcmb.cen_rms:
                ac = acname
            else:
                ac = acname + "c"
            abdev_str = f"{orbcmb.cen_abdev[ac]*1000:^6.0f}"
            abdev.append(abdev_str)
            abdevfull_str = f"{orbcmb.cen_abdev[ac]*1000:^9.3f}"
            abdevfull.append(abdevfull_str)
        logger.info(f"ABDEV(mm)| {' | '.join([item for item in abdev])}")

        logger.info("\n\nCenter absolute deviation statistics (mm) by constellation:\n")
        logger.info(f" Sat. System |  {'  |  '.join([ac for ac in centers_str])}")
        logger.info(f"-------------{'---------'*len(centers_str)}")
        for sys in self.systems:
            abdev = []
            abdevfull = []
            for acname in centers:
                if acname in orbcmb.cen_rms:
                    ac = acname
                else:
                    ac = acname + "c"
                if (ac,sys) in orbcmb.cen_abdev:
                    abdev_str = f"{orbcmb.cen_abdev[ac,sys]*1000:^6.0f}"
                    abdevfull_str = f"{orbcmb.cen_abdev[ac,sys]*1000:^9.3f}"
                else:
                    abdev_str = "      "
                    abdevfull_str = "         "
                abdev.append(abdev_str)
                abdevfull.append(abdevfull_str)
            logger.info(f"      {sys}      | "
                        f"{' | '.join([item for item in abdev])}")

        logger.info("\n\nCenter absolute deviation statistics (mm) by block:\n")
        logger.info(
                f"   Block        |  {'  |  '.join([ac for ac in centers_str])}")
        logger.info(f"----------------{'---------'*len(centers_str)}")
        for blk in self.blocks:
            abdev = []
            abdevfull = []
            for acname in centers:
                if acname in orbcmb.cen_rms:
                    ac = acname
                else:
                    ac = acname + "c"
                if (ac,blk) in orbcmb.cen_abdev:
                    abdev_str = f"{orbcmb.cen_abdev[ac,blk]*1000:^6.0f}"
                    abdevfull_str = f"{orbcmb.cen_abdev[ac,blk]*1000:^9.3f}"
                else:
                    abdev_str = "      "
                    abdevfull_str = "         "
                abdev.append(abdev_str)
                abdevfull.append(abdevfull_str)
            logger.info(f"{blk:15} | {' | '.join([item for item in abdev])}")

        logger.info("\n\nCenter absolute deviation statistics (mm) for all "
                       "satellites:\n")
        logger.info(
            f" PRN |  SVN |  {'  |  '.join([ac for ac in centers_str])}  |   IGS")
        logger.info(f"------------{'---------'*(len(centers_str)+1)}")
        for sat in self.sats:
            abdev = []
            abdevfull = []
            for acname in centers:
                if acname in orbcmb.cen_rms:
                    ac = acname
                else:
                    ac = acname + "c"
                if (ac,sat) in orbcmb.cen_abdev:
                    abdev_str = f"{orbcmb.cen_abdev[ac,sat]*1000:^6.0f}"
                    abdevfull_str = f"{orbcmb.cen_abdev[ac,sat]*1000:^9.3f}"
                else:
                    abdev_str = "      "
                    abdevfull_str = "         "
                abdev.append(abdev_str)
                abdevfull.append(abdevfull_str)
            if sat in orbcmb.sat_abdev:
                abdev_str = f"{orbcmb.sat_abdev[sat]*1000:^6.0f}"
                abdevfull_str = f"{orbcmb.sat_abdev[sat]*1000:^9.3f}"
            else:
                abdev_str = "      "
                abdevfull_str = "         "
            abdev.append(abdev_str)
            abdevfull.append(abdevfull_str)
            prn = sat[0] + str(sat[1]).zfill(2)
            svn = sat[0] + str(sat[2]).zfill(3)
            logger.info(f" {prn} | {svn} | {' | '.join([item for item in abdev])}")
        logger.info("\n")

        sumfull.write("\n\n")

        # Helmert parameter estimates
        header = ("4) 7-parameter transformations of ACs with respect to the "
                    "combined orbit:")
        sumfull.write(f" {header}\n")
        sumfull.write(f" {'-'*len(header)}\n\n")
        trn_header1 = ['Tx','TY','TZ','RX','RY','RZ','SC']
        trn_header1_aln = []
        sumdict['transformation parameters'] = {}
        for item in trn_header1:
            trn_header1_aln.append(item.center(8))
        trn_header2 = [' [mm]',' [mm]',' [mm]','[uas]','[uas]','[uas]','[ppb]']
        trn_header2_aln = []
        for item in trn_header2:
            trn_header2_aln.append(item.center(8))
        header1 = f"      {'|'.join([item for item in trn_header1_aln])}"
        header2 = f"  AC |{'|'.join([item for item in trn_header2_aln])}"
        sumfull.write(f" {header1}\n")
        sumfull.write(f" {header2}\n")
        sumfull.write(f" {'-'*len(header2)}\n")

        for acname in centers:
            if acname in orbcmb.transform_params:
                ac = acname
            else:
                ac = acname + "c"
            trn = []
            sumdict['transformation parameters'][acname] = {}
            sumdict['transformation parameters'][acname]['T'] = []
            sumdict['transformation parameters'][acname]['R'] = []
            sumdict['transformation parameters'][acname]['S'] = []
            c = 0
            for item in orbcmb.transform_params[ac]:
                c += 1
                if c<4:
                    trn_str = f"{item*1000:6.1f}"
                    if abs(item*1000) > 999.9:
                        trn_str = f"{999:6.0f}"
                    trn.append(trn_str)
                    sumdict['transformation parameters'][acname]['T'].append(item*1000)
                elif c<7:
                    trn_str = f"{item*1e6:6.0f}"
                    if abs(item*1e6) > 99999:
                        trn_str = f"{99999:6.0f}"
                    trn.append(trn_str)
                    sumdict['transformation parameters'][acname]['R'].append(item*1000)
                else:
                    trn_str = f"{item*1000:6.2f}"
                    if abs(item*1000) > 999.99:
                        trn_str = f"{999:6.0f}"
                    trn.append(trn_str)
                    sumdict['transformation parameters'][acname]['S'].append(item*1000)

            line = f" {acname} | {' | '.join([item for item in trn])}"
            sumfull.write(f" {line}\n")

        sumfull.write("\n")
        sumfull.close()

        # write out summary in json format
        with open(self.sumjson_fname,'w',encoding='utf-8') as outfile:
            json.dump(sumdict,outfile,ensure_ascii=False, indent=4,default=str)

