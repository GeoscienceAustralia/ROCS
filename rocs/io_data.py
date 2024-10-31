# Module for input and output of data

import logging
import re
import numpy as np
import datetime
from collections import namedtuple
import pathlib
from scipy.interpolate import lagrange
import yaml
from rocs.gpscal import gpsCal


logger = logging.getLogger(__name__)

class sp3:

    """
    Class of sp3 orbit data
    Reads/writes a standard product 3 orbit format (sp3) file
    Capable of reading/writing the extended sp3 (sp3-d) format

    """

    def __init__(self,sp3file,sp3dict=None):

        """
        Initialize sp3 class

        Keyword arguments:
            sp3file [str]             : name of a sp3 file for input or output
            sp3dict [dict], optional : a sp3 dictionary for output

        Updates:
            self.sp3file [str]
            self.sp3dict [dict]

        """

        # Check and assign the given sp3 file
        if not isinstance(sp3file,str):
            logger.error("\nThe keyword argument sp3file must be string",
                            stack_info=True)
            raise TypeError("sp3file is not a string!")

        self.sp3file = sp3file

        # Check and assign the given sp3dict
        if sp3dict is not None:
            if not isinstance(sp3dict,dict):
                logger.error("\nThe keyword argument sp3dict must be a "
                             "dictionary",stack_info=True)
                raise TypeError("sp3dict is not a dictionary!")

            if not sp3dict:
                logger.error("\nThe keyword argument sp3dict is empty")
                raise ValueError("sp3dict is empty!")

            self.sp3dict = sp3dict


    def parse(self,start_epoch=None,end_epoch=None):

        """
        parse self.sp3file into self.sp3dict dictionary

        Keyword arguments:
            start_epoch (datetime.datetime): start epoch to read data
            end_epoch (datetime.datetime)  :  last epoch to read data

        Updates:
            self.sp3dict
        """

        # Check the type of start_epoch and end_epoch
        if (start_epoch is not None and
                not isinstance(start_epoch,datetime.datetime)):
            logger.error("\nThe keyword argument start_epoch must be a "
                         "datetime.datetime object",stack_info=True)
            raise TypeError("start_epoch not datetime.datetime!")

        if (end_epoch is not None and
                not isinstance(end_epoch,datetime.datetime)):
            logger.error("\nThe keyword argument end_epoch must be a "
                         "datetime.datetime object",stack_info=True)
            raise TypeError("end_epoch not datetime.datetime!")
        if start_epoch is None:
            start_epoch = datetime.datetime(1,1,1,0,0)
        if end_epoch is None:
            end_epoch = datetime.datetime(9999, 12, 31, 0, 0)

        # Create some RGX patterns
        epoch_hdr_rgx = re.compile('^\*')
        eof_rgx = re.compile('^EOF')
        sat_rgx = re.compile('^\+ ')
        accu_rgx = re.compile('^\+\+')
        c_rgx = re.compile('^\%c')
        f_rgx = re.compile('^\%f')
        i_rgx = re.compile('^\%i')
        comment_rgx = re.compile('^\/\*')

        flag = 0
        line_num = 0
        sat_counter = 0
        accu_counter = 0
        c_counter = 0
        f_counter = 0
        nout = 0
        sats = []
        sat_accuracy = []
        sp3dict = {}
        sp3dict['data'] = {}
        sp3dict['data']['epochs'] = []
        comments = []

        # Try to open sp3file
        try:
            sp3_fid = open(self.sp3file,'r')
        except IOError:
            logger.error(f"\nThe specified sp3 file {self.sp3file} is not "
                         f"accessible\n",stack_info=True)
            raise IOError(f"File {self.sp3file} not accessible!")
        else:
            with sp3_fid:
                for line in sp3_fid:
                    line_num = line_num + 1

                    if line_num == 1:
                        line = line.ljust(60)
                        sp3dict['header'] = {}
                        sp3dict['header']['version']    = line[0:2]
                        sp3dict['header']['pvflag']     = line[2]
                        sp3dict['header']['start_year'] = int(line[3:8])
                        sp3dict['header']['start_month']= int(line[8:10])
                        sp3dict['header']['start_day']  = int(line[11:13])
                        sp3dict['header']['start_hour'] = int(line[14:16])
                        sp3dict['header']['start_min']  = int(line[17:19])
                        sp3dict['header']['start_sec']  = float(line[20:31])
                        sp3dict['header']['num_epochs'] = int(line[32:39])
                        sp3dict['header']['data_used']  = line[40:45]
                        sp3dict['header']['coord_sys']  = line[46:51]
                        sp3dict['header']['orbit_type'] = line[52:55]
                        sp3dict['header']['agency']     = line[56:60]
                        sp3dict['header']['sats'] = []
                        sp3dict['header']['sat_accuracy'] = []

                    elif line_num == 2:
                        line = line.ljust(60)
                        sp3dict['header']['gpsweek']   = int(line[3:7])
                        sp3dict['header']['sow']       = float(line[8:23])
                        sp3dict['header']['epoch_int'] = float(line[24:38])
                        sp3dict['header']['modjul']    = int(line[39:44])
                        sp3dict['header']['frac']      = float(line[45:60])

                    elif line_num == 3:
                        line = line.ljust(60)
                        sp3dict['header']['numsats'] = int(line[3:6])
                        for i,c in enumerate(range(9,60,3)):
                            sat_counter += 1
                            if sat_counter > sp3dict['header']['numsats']:
                                break
                            sp3dict['header']['sats'].append(line[c:c+3])

                    elif sat_rgx.search(line):
                        line = line.ljust(60)
                        for i,c in enumerate(range(9,60,3)):
                            sat_counter += 1
                            if sat_counter > sp3dict['header']['numsats']:
                                break
                            sp3dict['header']['sats'].append(line[c:c+3])

                    elif accu_rgx.search(line):
                        line = line.ljust(60)
                        for i,c in enumerate(range(9,60,3)):
                            accu_counter += 1
                            if accu_counter > sp3dict['header']['numsats']:
                                break
                            sp3dict['header']['sat_accuracy'].append(
                                                            int(line[c:c+3]))

                    elif c_rgx.search(line):
                        line = line.ljust(60)
                        if c_counter == 0:
                            sp3dict['header']['file_type'] = line[3:5]
                            sp3dict['header']['time_system'] = line[9:12]
                            c_counter += 1
                        else:
                            c_counter = 0

                    elif f_rgx.search(line):
                        line = line.ljust(60)
                        if f_counter == 0:
                            sp3dict['header']['base_pos'] = float(line[3:13])
                            sp3dict['header']['base_clk'] = float(line[14:26])
                            f_counter += 1
                        else:
                            f_counter = 0

                    elif comment_rgx.search(line):
                        line = line.ljust(80)
                        comments.append(line[3:80].rstrip())

                    elif epoch_hdr_rgx.search(line):
                        line = line.ljust(31)
                        flag  = 1
                        year  = int(line[2:7])
                        month = int(line[8:10])
                        dom   = int(line[11:13])
                        hh    = int(line[14:16])
                        mm    = int(line[17:19])
                        sec   = float(line[20:31])
                        gc = gpsCal()
                        gc.set_yyyy_MM_dd_hh_mm_ss(year,month,dom,hh,mm,sec)
                        epoch = gc.dto
                        if epoch < start_epoch or epoch > end_epoch:
                            nout += 1
                            flag = 0
                        else:
                            sp3dict['data']['epochs'].append(epoch)
                            for sat in sp3dict['header']['sats']:
                                sp3dict['data'][(sat,epoch,'Pflag')]  = 0
                                sp3dict['data'][(sat,epoch,'EPflag')] = 0
                                sp3dict['data'][(sat,epoch,'Vflag')]  = 0
                                sp3dict['data'][(sat,epoch,'EVflag')] = 0

                    elif eof_rgx.search(line):
                        pass

                    elif flag == 1:
                        # this should be one of the last elif statements

                        # Symbol P EP V EV
                        if line[0] == 'P': # Position and clock record

                            line = line.ljust(80)

                            # Vehicle ID
                            sat = line[1:4]
                            sp3dict['data'][(sat,epoch,'Pflag')] = 1

                            # x - coordinate(km)
                            sp3dict['data'][(sat,epoch,'xcoord')] = float(
                                                                   line[4:18])
                            # y - coordinate(km)
                            sp3dict['data'][(sat,epoch,'ycoord')] = float(
                                                                   line[18:32])
                            # z - coordinate(km)
                            sp3dict['data'][(sat,epoch,'zcoord')] = float(
                                                                   line[32:46])
                            # clock (microsec)
                            sp3dict['data'][(sat,epoch,'clock')] = float(
                                                                   line[46:60])

                            # Note: We have to read the following as strings
                            # instead of integers, because sometimes we are
                            # missing them in sp3 files

                            # x-sdev (b**n mm)
                            sp3dict['data'][(sat,epoch,'xsdev')] = line[61:63]

                            # y-sdev (b**n mm)
                            sp3dict['data'][(sat,epoch,'ysdev')] = line[64:66]

                            # z-sdev (b**n mm)
                            sp3dict['data'][(sat,epoch,'zsdev')] = line[67:69]

                            # c-sdev (b**n psec)
                            sp3dict['data'][(sat,epoch,'csdev')] = line[70:73]

                            # clock event flag E
                            sp3dict['data'][(sat,epoch,'clk_event')] = (
                                                                    line[74])

                            # clock pred flag P
                            sp3dict['data'][(sat,epoch,'clk_pred')] = line[75]

                            # maneuver flag
                            sp3dict['data'][sat,epoch,'maneuver'] = line[78]

                            # orbit predict flag P
                            sp3dict['data'][(sat,epoch,'orbit_pred')] = (
                                                                    line[79])

                        # Position and clock correlation record
                        elif line[0:2] == 'EP':

                            line = line.ljust(80)
                            sp3dict['data'][(sat,epoch,'EPflag')] = 1

                            # high-resolution x-sdev (mm)
                            sp3dict['data'][(sat,epoch,'xsdev-hres')] = (
                                                                    line[4:8])

                            # high-resolution y-sdev (mm)
                            sp3dict['data'][(sat,epoch,'ysdev-hres')] = (
                                                                    line[9:13])

                            # high-resolution z-sdev (mm)
                            sp3dict['data'][(sat,epoch,'zsdev-hres')] = (
                                                                line[14:18])

                            # high-resolution c-sdev (psec)
                            sp3dict['data'][(sat,epoch,'csdev-hres')] = (
                                                                line[19:26])
                            # xy-correlation
                            sp3dict['data'][(sat,epoch,'xy-corr')] = (
                                                                line[27:35])

                            # xz-correlation
                            sp3dict['data'][(sat,epoch,'xz-corr')] = (
                                                                line[36:44])

                            # xc-correlation
                            sp3dict['data'][(sat,epoch,'xc-corr')] = (
                                                                line[45:53])

                            # yz-correlation
                            sp3dict['data'][(sat,epoch,'yz-corr')] = (
                                                                line[54:62])

                            # yc-correlation
                            sp3dict['data'][(sat,epoch,'yc-corr')] = (
                                                                line[63:71])

                            # zc-correlation
                            sp3dict['data'][(sat,epoch,'zc-corr')] = (
                                                                line[72:80])

                        # Velocity and clock rate-of-change record
                        elif line[0] == 'V':

                            line = line.ljust(80)

                            # Vehicle ID
                            sat = line[1:4]
                            sp3dict['data'][(sat,epoch,'Vflag')] = 1

                            # x - velocity (dm/s)
                            sp3dict['data'][(sat,epoch,'xvel')] = float(
                                                                line[4:18])

                            # y - velocity (dm/s)
                            sp3dict['data'][(sat,epoch,'yvel')] = float(
                                                                line[18:32])

                            # z - velocity(dm/s)
                            sp3dict['data'][(sat,epoch,'zvel')] = float(
                                                                line[32:46])

                            # clock rate-of-change (10**-4 microseconds/second)
                            sp3dict['data'][(sat,epoch,'clkrate')] = float(
                                                                line[46:60])

                            # xvel-sdev (b**n 10**-4 mm/sec)
                            sp3dict['data'][(sat,epoch,'xvel-sdev')] = (
                                                                line[61:63])

                            # yvel-sdev (b**n 10**-4  mm/sec)
                            sp3dict['data'][(sat,epoch,'yvel-sdev')] = (
                                                                line[64:66])

                            # zvel-sdev (b**n 10**-4 mm/sec)
                            sp3dict['data'][(sat,epoch,'zvel-sdev')] = (
                                                                line[67:69])

                            # clkrate-sdev (b**n 10**-4 psec/sec)
                            sp3dict['data'][(sat,epoch,'clkrate-sdev')] = (
                                                                line[70:73])

                        # Velocity and clock rate correlation record
                        elif line[0:2] == 'EV':

                            line = line.ljust(80)
                            sp3dict['data'][(sat,epoch,'EVflag')] = 1

                            # high-resolution xvel-sdev (10**-4 mm/sec)
                            sp3dict['data'][(sat,epoch,'xvelsdev-hres')] = (
                                                                line[4:8])

                            # high-resolution yvel-sdev (10**-4 mm/sec)
                            sp3dict['data'][(sat,epoch,'yvelsdev-hres')] = (
                                                                line[9:13])

                            # high-resolution zvel-sdev (10**-4 mm/sec)
                            sp3dict['data'][(sat,epoch,'zvelsdev-hres')] = (
                                                                line[14:18])

                            # high-resolution clkrate-sdev (10**-4 psec/sec)
                            (sp3dict['data']
                                [(sat,epoch,'clkrate-sdev-hres')]) = (
                                                                line[19:26])

                            # xy-correlation
                            sp3dict['data'][(sat,epoch,'xy-vel-corr')] = (
                                                                line[27:35])

                            # xz-correlation
                            sp3dict['data'][(sat,epoch,'xz-vel-corr')] = (
                                                                line[36:44])

                            # xc-correlation
                            (sp3dict['data']
                                [(sat,epoch,'xvel-clkrate-corr')]) = (
                                                                line[45:53])

                            # yz-correlation
                            sp3dict['data'][(sat,epoch,'yz-vel-corr')] = (
                                                                line[54:62])

                            # yc-correlation
                            (sp3dict['data']
                                [(sat,epoch,'yvel-clkrate-corr')]) = (
                                                                line[63:71])
                            # zc-correlation
                            (sp3dict['data']
                                [(sat,epoch,'zvel-clkrate-corr')]) = (
                                                                line[72:80])

                sp3dict['header']['comments'] = comments
                sp3dict['header']['num_epochs'] -= nout

        self.sp3dict = sp3dict


    def write(self):

        """
        write self.sp3dict to a file named self.sp3file

        """

        sp3dict = self.sp3dict

        # Try to get header information from sp3dict
        try:
            sp3_header = sp3dict['header']
        except KeyError:
            logger.error("Keyword argument sp3dict lacks 'header' key",
                            stack_info=True)
            raise KeyError("'header'")

        try:
            version = sp3_header['version']
            pvflag = sp3_header['pvflag']
            start_year = sp3_header['start_year']
            start_month = sp3_header['start_month']
            start_day = sp3_header['start_day']
            start_hour = sp3_header['start_hour']
            start_min = sp3_header['start_min']
            start_sec = sp3_header['start_sec']
            num_epochs = sp3_header['num_epochs']
            data_used = sp3_header['data_used']
            coord_sys = sp3_header['coord_sys']
            orbit_type = sp3_header['orbit_type']
            agency = sp3_header['agency']
            sats = sp3_header['sats']
            sat_accuracy = sp3_header['sat_accuracy']
            gpsweek = sp3_header['gpsweek']
            sow = sp3_header['sow']
            epoch_int = sp3_header['epoch_int']
            modjul = sp3_header['modjul']
            frac = sp3_header['frac']
            numsats = sp3_header['numsats']
            file_type = sp3_header['file_type']
            time_system = sp3_header['time_system']
            base_pos = sp3_header['base_pos']
            base_clk = sp3_header['base_clk']
            comments = sp3_header['comments']
        except KeyError as e:
            key = e.args[0]
            logger.error(f"Keyword argument sp3dict lacks 'header':'{key}' "
                         f"key",stack_info=True)
            raise KeyError(f"'header':'{key}'")

        # Try to get data epochs from sp3dict
        try:
            sp3_data = sp3dict['data']
        except KeyError:
            logger.error("Keyword argument sp3dict lacks 'data' key",
                            stack_info=True)
            raise KeyError("'data'")

        try:
            epochs = sp3_data['epochs']
        except KeyError:
            logger.error(f"Keyword argument sp3dict lacks 'data':'epochs' "
                         f"key",stack_info=True)
            raise KeyError(f"'data':'{epochs}'")

        # Raise a warning if a file with name self.sp3file already exists
        path = pathlib.Path(self.sp3file)
        if path.is_file():
            logger.warning(f"\nOverwriting the file {self.sp3file}, which "
                           f"already exists.")

        # Try to open sp3file for writing
        try:
            f = open(self.sp3file,'w')
        except IOError:
            logger.error(f"\nThe specified sp3 file {self.sp3file} cannot be "
                         f"opened for writing\n",stack_info=True)
            raise IOError(f"File {self.sp3file} cannot be opened for writing!")
        else:
            with f:

                # Line 1
                f.write('{0:2s}{1:1s}{2:4d} {3:2d} {4:2d} {5:2d} {6:2d} '
                '{7:11.8f} {8:7d} {9:5s} {10:5s} {11:3s} {12:4s}\n'.format(
                version,pvflag,start_year,start_month,start_day,start_hour,
                start_min,start_sec,num_epochs,data_used,coord_sys,
                orbit_type,agency))

                # Line 2
                f.write('{0:2s} {1:4d} {2:15.8f} {3:14.8f} {4:5d} '
                '{5:15.13f}\n'.format('##',gpsweek,sow,epoch_int,modjul,frac))

                # Line 3 first part
                f.write('{0:2s} {1:3d}   '.format('+',numsats))

                # Satellites
                sat_ctr = 0
                line_ctr = 1
                for sat in sats:
                    sat_ctr += 1
                    if (sat_ctr > 17):
                        line_ctr += 1
                        f.write('\n{0:2s}       '.format('+'))
                        sat_ctr = 1
                    f.write('{0:3s}'.format(sat))
                numZeros = 17-sat_ctr
                for i in range(0,numZeros):
                    f.write(' {0:2d}'.format(0))
                numZeroLines = 5 - line_ctr
                for i in range(0,numZeroLines):
                    f.write('\n{0:2s}       '.format('+'))
                    f.write((' {0:2d}'*17).format(0))

                # Satellite accuracies
                f.write('\n{0:2s}       '.format('++'))
                sat_ctr = 0
                line_ctr = 1
                for acc in sat_accuracy:
                    sat_ctr += 1
                    if (sat_ctr > 17):
                        line_ctr += 1
                        f.write('\n{0:2s}       '.format('++'))
                        sat_ctr = 1
                    f.write('{0:3d}'.format(acc))
                numZeros = 17-sat_ctr
                for i in range(0,numZeros):
                    f.write('{0:3d}'.format(0))
                numZeroLines = 5 - line_ctr
                for i in range(0,numZeroLines):
                    f.write('\n{0:2s}       '.format('++'))
                    f.write(('{0:3d}'*17).format(0))

                # c-lines
                f.write('\n{0:2s} {1:2s} {2:2s} {3:3s} {4:3s}'.format(
                        '%c',file_type,'cc',time_system,'ccc'))
                f.write((' {0:4s}'*4).format('cccc'))
                f.write((' {0:5s}'*4).format('ccccc'))
                f.write('\n{0:2s} {1:2s} {2:2s} {3:3s} {4:3s}'.
                        format('%c','cc','cc','ccc','ccc'))
                f.write((' {0:4s}'*4).format('cccc'))
                f.write((' {0:5s}'*4).format('ccccc'))

                # f-lines
                f.write('\n{0:2s} {1:10.7f} {2:12.9f} {3:14.11f} {4:18.15f}'
                        .format('%f',base_pos,base_clk,0,0))
                f.write('\n{0:2s} {1:10.7f} {2:12.9f} {3:14.11f} {4:18.15f}'
                        .format('%f',0,0,0,0))

                # i-lines
                f.write('\n{0:2s}'.format('%i'))
                f.write((' {0:4d}'*4).format(0))
                f.write((' {0:6d}'*4).format(0))
                f.write(' {0:9d}'.format(0))

                f.write('\n{0:2s}'.format('%i'))
                f.write((' {0:4d}'*4).format(0))
                f.write((' {0:6d}'*4).format(0))
                f.write(' {0:9d}'.format(0))

                # comments
                for comment in comments:
                    line = '\n{0:2s} {1:77s}'.format('/*',comment)
                    f.write(line.rstrip())

                # we need at least 4 comment lines
                numFillerComments = 4 - len(comments)
                for i in range(0,numFillerComments):
                    f.write('\n{0:2s} '.format('/*'))

                # data lines
                for i,epoch in enumerate(epochs):
                    f.write('\n{0:2s} {1:4d} {2:2d} {3:2d} {4:2d} {5:2d} '
                    '{6:11.8f}'.format('*',epoch.year,epoch.month,epoch.day,
                        epoch.hour,epoch.minute,epoch.second))
                    for sat in sats:
                        if sp3_data[(sat,epoch,'Pflag')] == 1:
                            if ((sat,epoch,'xcoord') in sp3_data
                                and ~np.isnan(sp3_data[(sat,epoch,'xcoord')])):
                                xcoord = sp3_data[(sat,epoch,'xcoord')]
                            else:
                                xcoord = 0.0
                            if ((sat,epoch,'ycoord') in sp3_data
                                and ~np.isnan(sp3_data[(sat,epoch,'ycoord')])):
                                ycoord = sp3_data[(sat,epoch,'ycoord')]
                            else:
                                ycoord = 0.0
                            if ((sat,epoch,'zcoord') in sp3_data
                                and ~np.isnan(sp3_data[(sat,epoch,'zcoord')])):
                                zcoord = sp3_data[(sat,epoch,'zcoord')]
                            else:
                                zcoord = 0.0
                            if ((sat,epoch,'clock') in sp3_data
                                and ~np.isnan(sp3_data[(sat,epoch,'clock')])):
                                clock = sp3_data[(sat,epoch,'clock')]
                            else:
                                clock = 999999.999999
                            if (sat,epoch,'xsdev') in sp3_data:
                                xsdev = str(sp3_data[(sat,epoch,'xsdev')])
                            else:
                                xsdev = ' '
                            if (sat,epoch,'ysdev') in sp3_data:
                                ysdev = str(sp3_data[(sat,epoch,'ysdev')])
                            else:
                                ysdev = ' '
                            if (sat,epoch,'zsdev') in sp3_data:
                                zsdev = str(sp3_data[(sat,epoch,'zsdev')])
                            else:
                                zsdev = ' '
                            if ((sat,epoch,'csdev') in sp3_data
                                    and ~np.isnan(sp3_data[(sat,epoch,'csdev')])):
                                csdev = str(sp3_data[(sat,epoch,'csdev')])
                            else:
                                csdev = ' '
                            if (sat,epoch,'clk_event') in sp3_data:
                                clk_event = sp3_data[(sat,epoch,'clk_event')]
                            else:
                                clk_event = ' '
                            if (sat,epoch,'clk_pred') in sp3_data:
                                clk_pred = sp3_data[(sat,epoch,'clk_pred')]
                            else:
                                clk_pred = ' '
                            if (sat,epoch,'maneuver') in sp3_data:
                                maneuver = sp3_data[(sat,epoch,'maneuver')]
                            else:
                                maneuver = ' '
                            if (sat,epoch,'orbit_pred') in sp3_data:
                                orbit_pred = sp3_data[(sat,epoch,'orbit_pred')]
                            else:
                                orbit_pred = ' '
                            str1 = '\n{0:1s}{1:3s}'.format('P',sat)
                            str2 = ('{0:14.6f}{1:14.6f}{2:14.6f}{3:14.6f}'
                                    .format(xcoord,ycoord,zcoord,clock))
                            str3 = (' {0:>2s} {1:>2s} {2:>2s} {3:>3s} {4:1s}'
                            '{5:1s}  {6:1s}{7:1s}'.format(xsdev,ysdev,zsdev,
                            csdev,clk_event,clk_pred,maneuver,orbit_pred))
                            line = str1 + str2 + str3
                            f.write(line.rstrip())

                        if sp3_data[(sat,epoch,'EPflag')] == 1:
                            if ((sat,epoch,'xsdev-hres') in sp3_data
                            and ~np.isnan(sp3_data[(sat,epoch,'xsdev-hres')])):
                                xsdev_hres = sp3_data[(sat,epoch,'xsdev-hres')]
                            else:
                                xsdev_hres = ' '
                            if ((sat,epoch,'ysdev-hres') in sp3_data
                            and ~np.isnan(sp3_data[(sat,epoch,'ysdev-hres')])):
                                ysdev_hres = sp3_data[(sat,epoch,'ysdev-hres')]
                            else:
                                ysdev_hres = ' '
                            if ((sat,epoch,'zsdev-hres') in sp3_data
                            and ~np.isnan(sp3_data[(sat,epoch,'zsdev-hres')])):
                                zsdev_hres = sp3_data[(sat,epoch,'zsdev-hres')]
                            else:
                                zsdev_hres = ' '
                            if ((sat,epoch,'csdev-hres') in sp3_data
                            and ~np.isnan(sp3_data[(sat,epoch,'csdev-hres')])):
                                csdev_hres = sp3_data[(sat,epoch,'csdev-hres')]
                            else:
                                csdev_hres = ' '
                            if ((sat,epoch,'xy-corr') in sp3_data
                            and ~np.isnan(sp3_data[(sat,epoch,'xy-corr')])):
                                xy_corr = sp3_data[(sat,epoch,'xy-corr')]
                            else:
                                xy_corr = ' '
                            if ((sat,epoch,'xz-corr') in sp3_data
                            and ~np.isnan(sp3_data[(sat,epoch,'xz-corr')])):
                                xz_corr = sp3_data[(sat,epoch,'xz-corr')]
                            else:
                                xz_corr = ' '
                            if ((sat,epoch,'xc-corr') in sp3_data
                            and ~np.isnan(sp3_data[(sat,epoch,'xc-corr')])):
                                xc_corr = sp3_data[(sat,epoch,'xc-corr')]
                            else:
                                xc_corr = ' '
                            if ((sat,epoch,'yz-corr') in sp3_data
                            and ~np.isnan(sp3_data[(sat,epoch,'yz-corr')])):
                                yz_corr = sp3_data[(sat,epoch,'yz-corr')]
                            else:
                                yz_corr = ' '
                            if ((sat,epoch,'yc-corr') in sp3_data
                            and ~np.isnan(sp3_data[(sat,epoch,'yc-corr')])):
                                yc_corr = sp3_data[(sat,epoch,'yc-corr')]
                            else:
                                yc_corr = ' '
                            if ((sat,epoch,'zc-corr') in spop3_data
                            and ~np.isnan(sp3_data[(sat,epoch,'zc-corr')])):
                                zc_corr = sp3_data[(sat,epoch,'zc-corr')]
                            else:
                                zc_corr = ' '
                            str1 = ('\n{0:2s}  {1:4s} {2:4s} {3:4s} {4:7s}'
                                    .format('EP',xsdev_hres,ysdev_hres,
                                            zsdev_hres,csdev_hres))
                            str2 = (' {0:8s} {1:8s} {2:8s} {3:8s} {4:8s} '
                                    '{5:8s}'.format(xy_corr,xz_corr,xc_corr,
                                    yz_corr,yc_corr,zc_corr))
                            line = str1 + str2
                            f.write(line.rstrip())

                        if sp3_data[(sat,epoch,'Vflag')] == 1:
                            if ((sat,epoch,'xvel') in sp3_data
                                and ~np.isnan(sp3_data[(sat,epoch,'xvel')])):
                                xvel = sp3_data[(sat,epoch,'xvel')]
                            else:
                                xvel = 0.0
                            if ((sat,epoch,'yvel') in sp3_data
                                and ~np.isnan(sp3_data[(sat,epoch,'yvel')])):
                                yvel = sp3_data[(sat,epoch,'yvel')]
                            else:
                                yvel = 0.0
                            if ((sat,epoch,'zvel') in sp3_data
                                and ~np.isnan(sp3_data[(sat,epoch,'zvel')])):
                                zvel = sp3_data[(sat,epoch,'zvel')]
                            else:
                                zvel = 0.0
                            if ((sat,epoch,'clkrate') in sp3_data
                            and ~np.isnan(sp3_data[(sat,epoch,'clkrate')])):
                                clkrate = sp3_data[(sat,epoch,'clkrate')]
                            else:
                                clkrate = 999999.999999
                            if ((sat,epoch,'xvel-sdev') in sp3_data
                            and ~np.isnan(sp3_data[(sat,epoch,'xvel-sdev')])):
                                xvel_sdev = sp3_data[(sat,epoch,'xvel-sdev')]
                            else:
                                xvel_sdev = ' '
                            if ((sat,epoch,'yvel-sdev') in sp3_data
                            and ~np.isnan(sp3_data[(sat,epoch,'yvel-sdev')])):
                                yvel_sdev = sp3_data[(sat,epoch,'yvel-sdev')]
                            else:
                                yvel_sdev = ' '
                            if ((sat,epoch,'zvel-sdev') in sp3_data
                            and ~np.isnan(sp3_data[(sat,epoch,'zvel-sdev')])):
                                zvel_sdev = sp3_data[(sat,epoch,'zvel-sdev')]
                            else:
                                zvel_sdev = ' '
                            if ((sat,epoch,'clkrate-sdev') in sp3_data
                                and ~np.isnan(sp3_data[
                                    (sat,epoch,'clkrate-sdev')])):
                                clkrate_sdev = (sp3_data
                                                [(sat,epoch,'clkrate-sdev')])
                            else:
                                clkrate_sdev = ' '
                            str1 = '\n{0:1s}{1:3s}'.format('V',sat)
                            str2 = ('{0:14.6f}{1:14.6f}{2:14.6f}{3:14.6f}'
                                    .format(xvel,yvel,zvel,clkrate))
                            str3 = (' {0:2s} {1:2s} {2:2s} {3:3s}       '
                                    .format(xvel_sdev,yvel_sdev,zvel_sdev,
                                            clkrate_sdev))
                            line = str1 + str2 + str3
                            f.write(line.rstrip())

                        if sp3_data[(sat,epoch,'EVflag')] == 1:
                            if ((sat,epoch,'xvelsdev-hres') in sp3_data
                                and ~np.isnan(sp3_data[
                                    (sat,epoch,'xvelsdev-hres')])):
                                xvelsdev_hres = (sp3_data
                                                [(sat,epoch,'xvelsdev-hres')])
                            else:
                                xvelsdev_hres = ' '
                            if ((sat,epoch,'yvelsdev-hres') in sp3_data
                                and ~np.isnan(sp3_data
                                    [(sat,epoch,'yvelsdev-hres')])):
                                yvelsdev_hres = (sp3_data
                                                [(sat,epoch,'yvelsdev-hres')])
                            else:
                                yvelsdev_hres = ' '
                            if ((sat,epoch,'zvelsdev-hres') in sp3_data
                                and ~np.isnan(sp3_data
                                    [(sat,epoch,'zvelsdev-hres')])):
                                zvelsdev_hres = (sp3_data
                                                [(sat,epoch,'zvelsdev-hres')])
                            else:
                                zvelsdev_hres = ' '
                            if ((sat,epoch,'clkrate-sdev-hres') in sp3_data
                                and ~np.isnan(sp3_data
                                    [(sat,epoch,'clkrate-sdev-hres')])):
                                clkrate_sdev_hres = (sp3_data
                                            [(sat,epoch,'clkrate-sdev-hres')])
                            else:
                                clkrate_sdev_hres = ' '
                            if ((sat,epoch,'xy-vel-corr') in sp3_data
                                and ~np.isnan(sp3_data
                                    [(sat,epoch,'xy-vel-corr')])):
                                xy_vel_corr = (sp3_data
                                            [(sat,epoch,'xy-vel-corr')])
                            else:
                                xy_vel_corr = ' '
                            if ((sat,epoch,'xz-vel-corr') in sp3_data
                                and ~np.isnan(sp3_data
                                    [(sat,epoch,'xz-vel-corr')])):
                                xz_vel_corr = (sp3_data
                                            [(sat,epoch,'xz-vel-corr')])
                            else:
                                xz_vel_corr = ' '
                            if ((sat,epoch,'yz-vel-corr') in sp3_data
                                and ~np.isnan(sp3_data
                                    [(sat,epoch,'yz-vel-corr')])):
                                yz_vel_corr = (sp3_data
                                            [(sat,epoch,'yz-vel-corr')])
                            else:
                                yz_vel_cor = ' '
                            if ((sat,epoch,'xvel-clkrate-corr') in sp3_data
                                and ~np.isnan(sp3_data
                                    [(sat,epoch,'xvel-clkrate-corr')])):
                                xvel_clkrate_corr = (sp3_data
                                            [(sat,epoch,'xvel-clkrate-corr')])
                            else:
                                xvel_clkrate_corr = ' '
                            if ((sat,epoch,'yvel-clkrate-corr') in sp3_data
                                and ~np.isnan(sp3_data
                                    [(sat,epoch,'yvel-clkrate-corr')])):
                                yvel_clkrate_corr = (sp3_data
                                            [(sat,epoch,'yvel-clkrate-corr')])
                            else:
                                yvel_clkrate_corr = ' '
                            if ((sat,epoch,'zvel-clkrate-corr') in sp3_data
                                and ~np.isnan(sp3_data
                                    [(sat,epoch,'zvel-clkrate-corr')])):
                                zvel_clkrate_corr = (sp3_data
                                            [(sat,epoch,'zvel-clkrate-corr')])
                            else:
                                zvel_clkrate_corr = ' '
                            str1 = ('\n{0:2s}  {1:4s} {2:4s} {3:4s} {4:7s}'
                                    .format('EV',xvelsdev_hres,yvelsdev_hres,
                                            zvelsdev_hres,clkrate_sdev_hres))
                            str2 = (' {0:8s} {1:8s} {2:8s} {3:8s} {4:8s} '
                            '{5:8s}'.format(xy_vel_corr,xz_vel_corr,
                                        xvel_clkrate_corr,yz_vel_corr,
                                        yvel_clkrate_corr,zvel_clkrate_corr))
                            line = str1 + str2
                            f.write(line.rstrip())

                # EOF line
                f.write('\n{0:3s}\n'.format('EOF'))



class SatelliteMetadata:

    """
    Class of satellite metadata
    Reads an IGS-standard satellite sinex metadata file

    """

    def __init__(self,metadata_file):

        """
        Initialize SatelliteMetadata class

        Keyword arguments:
            metadata_file [str] : filename of the metadata sinex file

        Updates:
            self.metadata_file [str]
            self.svn_to_prn [dict]: dictionary containing information to map
                                  from svn to prn
            self.prn_to_svn [dict]: dictionary containing information to map
                                  from prn to svn
            self.sat_identifier [dict]: dictionary containing satellite
                                        identifier information
            self.freq_channel [dict]: dictionary containing frequency channel
                                        information
        """

        # Check the given satellite metadata file
        if not isinstance(metadata_file,str):
            logger.error("The input metadata_file needs to be a string",
                         stack_info=True)
            raise TypeError("The input metadata_file needs to be a string")

        # Set metadata_file attribute
        self.metadata_file = metadata_file

        # dictionaries containing information to map between svn and prn
        svn_to_prn = {}
        prn_to_svn = {}

        # dictionary containing satellite identifier information
        sat_identifier = {}

        # dictionary containing frequency channel information
        freq_channel = {}

        # Regex
        satid_hdr_rgx = re.compile("^\+SATELLITE\/IDENTIFIER")
        svnprn_hdr_rgx = re.compile("^\+SATELLITE\/PRN")
        freqch_hdr_rgx = re.compile("^\+SATELLITE\/FREQUENCY_CHANNEL")
        comment_rgx = re.compile("^\*")
        end_rgx = re.compile("^\-")

        # Try to open the metadata file and fill in the attributes
        try:
            meta_fid = open(self.metadata_file,'r',encoding='ISO-8859-1')
        except IOError:
            logger.error(f"Metadata file {self.metadata_file} is not "
                         f"accessible!", stack_info=True)
            raise IOError(f"File {self.metadata_file} not found!")
        else:
            with meta_fid:
                for line in meta_fid:
                    if satid_hdr_rgx.search(line):
                        for line in meta_fid:
                            if end_rgx.search(line):
                                break
                            if not comment_rgx.search(line):
                                stuff = line.strip().split()
                                system_id = stuff[0][0]
                                svn = int(stuff[0][1:])
                                cospar_id = stuff[1]
                                sat_cat = stuff[2]
                                block = stuff[3]
                                comment = ' '.join(stuff[4:])

                                svn_full = stuff[0]

                                # Fill in sat_identifier dictionary
                                sat_identifier[system_id,svn] = (
                                            [cospar_id,sat_cat,block,comment])

                    if svnprn_hdr_rgx.search(line):
                        for line in meta_fid:
                            if end_rgx.search(line):
                                break
                            if not comment_rgx.search(line):
                                stuff = line.strip().split()
                                system_id = stuff[0][0]
                                svn = int(stuff[0][1:])
                                valid_from = stuff[1].split(":")
                                year_from = int(valid_from[0])
                                doy_from = int(valid_from[1])
                                sod_from = int(valid_from[2])
                                valid_to = stuff[2].split(":")
                                year_to = int(valid_to[0])
                                doy_to = int(valid_to[1])
                                sod_to = int(valid_to[2])
                                if year_to == 0:
                                    year_to = 9999
                                if doy_to == 0:
                                    doy_to = 365
                                gc_from = gpsCal()
                                gc_from.set_yyyy_ddd_sod(year_from,
                                                        doy_from,sod_from)
                                epoch_from = gc_from.dto
                                gc_to = gpsCal()
                                gc_to.set_yyyy_ddd_sod(year_to,doy_to,sod_to)
                                epoch_to = gc_to.dto
                                prn = int(stuff[3][1:])

                                # Fill in svn_to_prn dictionary
                                if ( (system_id,svn) not in
                                        svn_to_prn.keys()):
                                    svn_to_prn[system_id,svn] = (
                                                [[epoch_from,epoch_to,prn]])
                                else:
                                    svn_to_prn[system_id,svn] = np.vstack(
                                                [svn_to_prn[system_id,svn],
                                                [epoch_from,epoch_to,prn]])

                                # Fill in prn_to_svn dictionary
                                if ( (system_id,prn) not in
                                        prn_to_svn.keys()):
                                    prn_to_svn[system_id,prn] = (
                                                [[epoch_from,epoch_to,svn]])
                                else:
                                    prn_to_svn[system_id,prn] = np.vstack(
                                                [prn_to_svn[system_id,prn],
                                                [epoch_from,epoch_to,svn]])


                    if freqch_hdr_rgx.search(line):
                        for line in meta_fid:
                            if end_rgx.search(line):
                                break
                            if not comment_rgx.search(line):
                                stuff = line.strip().split()
                                system_id = stuff[0][0]
                                svn = int(stuff[0][1:])
                                valid_from = stuff[1].split(":")
                                year_from = int(valid_from[0])
                                doy_from = int(valid_from[1])
                                sod_from = int(valid_from[2])
                                valid_to = stuff[2].split(":")
                                year_to = int(valid_to[0])
                                doy_to = int(valid_to[1])
                                sod_to = int(valid_to[2])
                                if year_to == 0:
                                    year_to = 9999
                                if doy_to == 0:
                                    doy_to = 365
                                gc_from = gpsCal()
                                gc_from.set_yyyy_ddd_sod(year_from,
                                                        doy_from,sod_from)
                                epoch_from = gc_from.dto
                                gc_to = gpsCal()
                                gc_to.set_yyyy_ddd_sod(year_to,doy_to,sod_to)
                                epoch_to = gc_to.dto
                                chn = int(stuff[3])
                                comment = stuff[4]

                                # Fill in freq_channel dictionary
                                if ( (system_id,svn) not in
                                        freq_channel.keys()):
                                    freq_channel[system_id,svn] = (
                                        [[epoch_from,epoch_to,chn,comment]])
                                else:
                                    freq_channel[system_id,svn] = np.vstack(
                                            [freq_channel[system_id,svn],
                                            [epoch_from,epoch_to,chn,comment]])


        # Update attributes
        self.svn_to_prn = svn_to_prn
        self.prn_to_svn = prn_to_svn
        self.sat_identifier = sat_identifier
        self.freq_channel = freq_channel


    def get_prn(self,system_id,svn,epoch):
        """
        map svn to prn

        Input arguments:
            system_id [str]:  satellite constellation ID (e.g. 'G','R', etc.)
            svn [int]:        satellite svn number
            epoch [datetime]: epoch in datetime format
        """

        # Check the type of given arguments

        if not isinstance(system_id,str):
            logger.error("\nThe argument system_id must be a string\n")
            raise TypeError("The argument system_id must be a string")

        if not isinstance(svn,int):
            logger.error("\nThe argument svn must be an integer\n")
            raise TypeError("The argument svn must be an integer")

        if not isinstance(epoch,datetime.date):
            logger.error("\nThe argument epoch must be a datetime instance\n")
            raise TypeError("The argument epoch must be a datetime instance")

        found = False
        year = epoch.strftime("%Y").zfill(4)
        doy = epoch.strftime("%j").zfill(3)
        sod = str(epoch.hour*3600 + epoch.minute*60 + epoch.second).zfill(5)

        # Check if svn key exists
        if (system_id,svn) not in self.svn_to_prn.keys():
            logger.error(f"\nSVN {system_id}{str(svn).zfill(3)} not found!\n"
                         f"Check the metadata file {self.metadata_file}\n",
                         stack_info=True)
            raise KeyError(f"('{system_id}', {svn})")


        # Search in svn_to_prn for the svn key
        for row in self.svn_to_prn[system_id,svn]:
            if (epoch >= row[0] and epoch <= row[1]):

                # if found flag already True, it means that we had found 
                # a match previously; issue a warning and use the last match
                if found == True:
                    logger.warning(f"\nPossible duplicate PRN matches in "
                                   f"metadata for SVN and epoch "
                                   f"{system_id}{str(svn).zfill(3)} "
                                   f"{year}:{doy}:{sod}\nCheck the metadata "
                                   f"file {self.metadata_file}\n")

                # prn found; set found flag as True
                prn = row[2]
                found = True

        # if after searching the whole array, found is still False, we haven't
        # found a match; raise error
        if found is False:
            logger.error(f"\nNo PRN match found for SVN and epoch "
                         f"{system_id}{str(svn).zfill(3)} {year}:{doy}:{sod} "
                         f"\nCheck the metadata file {self.metadata_file}\n",
                         stack_info=True)
            raise ValueError(f"No match in self.svn_to_prn for SVN and epoch "
                             f"{system_id}{str(svn).zfill(3)} {epoch}")


        return prn


    def get_svn(self,system_id,prn,epoch):
        """
        map prn to svn

        Input arguments:
            system_id [str] : satellite constellation ID (e.g. 'G','R', etc.)
            prn [int]       : satellite prn number
            epoch [datetime]: epoch in datetime format
        """

        # Check the type of given arguments

        if not isinstance(system_id,str):
            logger.error("\nThe argument system_id must be a string\n")
            raise TypeError("The argument system_id must be a string")

        if not isinstance(prn,int):
            logger.error("\nThe argument prn must be an integer\n")
            raise TypeError("The argument prn must be an integer")

        if not isinstance(epoch,datetime.date):
            logger.error("\nThe argument epoch must be a datetime instance\n")
            raise TypeError("The argument epoch must be a datetime instance")

        found = False
        year = epoch.strftime("%Y").zfill(4)
        doy = epoch.strftime("%j").zfill(3)
        sod = str(epoch.hour*3600 + epoch.minute*60 + epoch.second).zfill(5)

        # Check if prn key exists
        if (system_id,prn) not in self.prn_to_svn.keys():
            logger.error(f"\nPRN {system_id}{str(prn).zfill(2)} not found!\n"
                         f"Check the metadata file {self.metadata_file}\n",
                         stack_info=True)
            raise KeyError(f"('{system_id}', {prn})")


        # Search in prn_to_svn for the prn key
        for row in self.prn_to_svn[system_id,prn]:
            if (epoch >= row[0] and epoch <= row[1]):

                # if found flag already True, it means that we had found 
                # a match previously; issue a warning and use the last match
                if found == True:
                    logger.warning(f"\nPossible duplicate SVN matches in "
                                   f"metadata for epoch and PRN "
                                   f"{year}:{doy}:{sod} "
                                   f"{system_id}{str(prn).zfill(2)}"
                                   f"\nCheck the metadata  file "
                                   f"{self.metadata_file}\n")

                # svn found; set found flag as True
                svn = row[2]
                found = True

        # if after searching the whole array, found is still False, we haven't
        # found a match; raise error
        if found is False:
            logger.error(f"\nNo SVN match found for epoch and PRN "
                         f"{year}:{doy}:{sod} {system_id}{str(prn).zfill(2)}"
                         f"\nCheck the metadata file {self.metadata_file}\n",
                         stack_info=True)
            raise ValueError(f"No match in self.prn_to_svn for PRN and epoch "
                             f"{system_id}{str(prn).zfill(2)} {epoch}")


        return svn


    def get_sat_identifier(self,system_id,svn):
        """
        get satellite identifier information

        Input arguments:
            system_id [str]: satellite constellation ID (e.g. 'G','R', etc.)
            svn [int]:       satellite svn number

        Returns:
            requested_satID [namedtuple]: satellite identifier information
                                          from metadata file, including block,
                                          cospar_id, sat_cat, and comments

        """

        # Check the given argument types
        if not isinstance(system_id,str):
            logger.error("\nThe argument system_id must be a string\n")
            raise TypeError("The argument system_id must be a string")

        if not isinstance(svn,int):
            logger.error("\nThe argument svn must be an integer\n")
            raise TypeError("The argument svn must be an integer")


        satID = namedtuple('satID',
                            'system_id svn block cospar_id sat_cat comment')

        # search in self.sat_identifier for svn
        if (system_id,svn) in self.sat_identifier.keys():

            cospar_id = self.sat_identifier[system_id,svn][0]
            sat_cat = self.sat_identifier[system_id,svn][1]
            block = self.sat_identifier[system_id,svn][2]
            comment = self.sat_identifier[system_id,svn][3]

            requested_satID = satID(system_id,svn,block,cospar_id,sat_cat,
                                    comment)

        else:

            logger.error(f"\nSVN {system_id}{str(svn).zfill(3)} not found!\n"
                         f"Check the metadata file {self.metadata_file}\n",
                         stack_info=True)
            raise KeyError(f"('{system_id}', {svn})")

        return requested_satID


    def get_freq_ch(self,system_id,svn,epoch):
        """
        get frequency channel

        Input arguments:
            system_id [str]:  satellite constellation ID (e.g. 'G','R', etc.)
            svn [int]:        satellite svn number
            epoch [datetime]: epoch in datetime format
        """

        # Check the type of given arguments

        if not isinstance(system_id,str):
            logger.error("\nThe argument system_id must be a string\n")
            raise TypeError("The argument system_id must be a string")

        if not isinstance(svn,int):
            logger.error("\nThe argument svn must be an integer\n")
            raise TypeError("The argument svn must be an integer")

        if not isinstance(epoch,datetime.date):
            logger.error("\nThe argument epoch must be a datetime instance\n")
            raise TypeError("The argument epoch must be a datetime instance")

        found = False
        year = epoch.strftime("%Y").zfill(4)
        doy = epoch.strftime("%j").zfill(3)
        sod = str(epoch.hour*3600 + epoch.minute*60 + epoch.second).zfill(5)

        # Check if svn key exists
        if (system_id,svn) not in self.freq_channel.keys():
            logger.error(f"\nSVN {system_id}{str(svn).zfill(3)} not found!\n"
                         f"Check the metadata file {self.metadata_file}\n",
                         stack_info=True)
            raise KeyError(f"('{system_id}', {svn})")


        # Search in freq_channel for the svn key
        for row in self.freq_channel[system_id,svn]:
            if (epoch >= row[0] and epoch <= row[1]):

                # if found flag already True, it means that we had found 
                # a match previously; issue a warning and use the last match
                if found == True:
                    logger.warning(f"\nPossible duplicate frequency channel "
                                   f"in metadata for SVN and epoch "
                                   f"{system_id}{str(svn).zfill(3)} "
                                   f"{year}:{doy}:{sod}\nCheck the metadata "
                                   f"file {self.metadata_file}\n")

                # channel found; set found flag as True
                chn = row[2]
                found = True

        # if after searching the whole array, found is still False, we haven't
        # found a match; raise error
        if found is False:
            logger.error(f"\nNo frequency channel match found for SVN and "
                         f"epoch {system_id}{str(svn).zfill(3)} "
                         f"{year}:{doy}:{sod}\nCheck the metadata file "
                         f"{self.metadata_file}\n",stack_info=True)
            raise ValueError(f"No match in self.freq_channel for SVN and "
                             f"epoch {system_id}{str(svn).zfill(3)} {epoch}")

        return chn



class EOPdata:

    """
    Class of earth orientation parameters (EOP) data
    Reads EOP parameters from different formats

    """

    def __init__(self,eop_file,eop_format):

        """
        Initialize EOPdata class

        Keyword arguments:
            eop_file [str]   : filename of the EOP data
            eop_format [str] : format of the EOP data

        Updates:
            self.eop_file [str]
            self.eop_format [str]
            self.eop_data [numpy.ndarray]: array containing eop information
                The columns of self.eop_data are:
                [mjd,xp,yp,ut1_utc,lod,xprate,yprate,
                     xp_sig,yp_sig,ut1_utc_sig,lod_sig,xprate_sig,yprate_sig]
                units are:
                [days,rad,rad,sec,sec/day,rad,rad,
                     rad,rad,sec,sec/day,rad,rad]
        """

        # Check the given EOP file
        if not isinstance(eop_file,str):
            logger.error("The input eop_file needs to be a string",
                         stack_info=True)
            raise TypeError("The input eop_file needs to be a string")

        # Set eop_file attribute
        self.eop_file = eop_file

        # Check the given EOP format
        if not isinstance(eop_format,str):
            logger.error("The input eop_format needs to be a string",
                         stack_info=True)
            raise TypeError("The input eop_format needs to be a string")

        allowed_formats = ['IERS_EOP14_C04','IERS_EOP_rapid','IGS_ERP2']
        if eop_format not in allowed_formats:
            logger.error("The input eop_format needs to be in "
                         f"{allowed_formats}",stack_info=True)
            raise TypeError("The input eop_format not recognized!")
        self.eop_format = eop_format

        # Initialize EOP array
        eop_data = np.empty((0,13))

        # Regular expressions
        # header lines of c04 format start with space or #
        c04_header_rgx = re.compile("^(\s|\#)")

        # Try to open the EOP file and fill in the eop_data array
        try:
            eop_fid = open(self.eop_file,'r')
        except IOError:
            logger.error(f"EOP file {self.eop_file} is not "
                         f"accessible!", stack_info=True)
            raise IOError(f"File {self.eop_file} not found!")
        else:
            with eop_fid:
                for line in eop_fid:

                    # Based on the source format, read EOP values
                    if self.eop_format == 'IERS_EOP14_C04':
                        if not c04_header_rgx.search(line):
                            mjd = float(line[12:19])
                            # for xp and yp, convert arc seconds to radians
                            xp = (float(line[19:30])/3600.0)*np.pi/180.0
                            yp = (float(line[30:41])/3600.0)*np.pi/180.0
                            ut1_utc = float(line[41:53])
                            lod = float(line[53:65])
                            xprate = (float(line[65:76])/3600.0)*np.pi/180.0
                            yprate = (float(line[76:87])/3600.0)*np.pi/180.0
                            xp_sig = (float(line[87:98])/3600.0)*np.pi/180.0
                            yp_sig = (float(line[98:109])/3600.0)*np.pi/180.0
                            ut1_utc_sig = float(line[109:120])
                            lod_sig = float(line[120:131])
                            xprate_sig = ((float(line[131:143])/3600.0)
                                            *np.pi/180.0)
                            yprate_sig = ((float(line[143:155])/3600.0)
                                            *np.pi/180.0)
                            eop_data = np.append(eop_data,np.array(
                                [[mjd,xp,yp,ut1_utc,lod,xprate,yprate,
                                  xp_sig,yp_sig,ut1_utc_sig,lod_sig,
                                  xprate_sig,yprate_sig]]),axis=0)

                    elif self.eop_format == 'IERS_EOP_rapid':
                        # Note: Bulletin A values are extracted
                        mjd = float(line[7:15])
                        xp = (float(line[18:27])/3600.0)*np.pi/180.0
                        xp_sig = (float(line[27:36])/3600.0)*np.pi/180.0
                        yp = (float(line[37:46])/3600.0)*np.pi/180.0
                        yp_sig = (float(line[46:55])/3600.0)*np.pi/180.0
                        ut1_utc = float(line[58:68])
                        ut1_utc_sig = float(line[68:78])
                        try:
                            lod = float(line[79:86])*1e-3
                        except:
                            lod = np.nan
                        try:
                            lod_sig = float(line[86:93])*1e-3
                        except:
                            lod_sig = np.nan
                        try:
                            xprate = ((float(line[97:106])*1e-3/3600.0)
                                                                *np.pi/180.0)
                        except:
                            xprate = np.nan
                        try:
                            xprate_sig = ((float(line[106:115])*1e-3/3600.0)
                                                                *np.pi/180.0)
                        except:
                            xprate_sig = np.nan
                        try:
                            yprate = ((float(line[116:125])*1e-3/3600.0)
                                                                *np.pi/180.0)
                        except:
                            yprate = np.nan
                        try:
                            yprate_sig = ((float(line[125:134])*1e-3/3600.0)
                                                                *np.pi/180.0)
                        except:
                            yprate_sig = np.nan
                        eop_data = np.append(eop_data,np.array(
                            [[mjd,xp,yp,ut1_utc,lod,xprate,yprate,
                                xp_sig,yp_sig,ut1_utc_sig,lod_sig,
                                xprate_sig,yprate_sig]]),axis=0)

                    elif self.eop_format == 'IGS_ERP2':
                        stuff = line.strip().split()
                        if (stuff and stuff[0] == 'MJD'):
                            # Column orders
                            clm_order = stuff
                        try:
                            mjd = float(stuff[0])
                        except: # line is header/does not contain values
                            continue
                        xp = (float(stuff[1])*1e-6/3600.0)*np.pi/180.0
                        yp = (float(stuff[2])*1e-6/3600.0)*np.pi/180.0
                        ut1_utc = float(stuff[3])*1e-7
                        lod = float(stuff[4])*1e-7
                        xp_sig = (float(stuff[5])*1e-6/3600.0)*np.pi/180.0
                        yp_sig = (float(stuff[6])*1e-6/3600.0)*np.pi/180.0
                        ut1_utc_sig = float(stuff[7])*1e-7
                        lod_sig = float(stuff[8])*1e-7
                        try:
                            xprate = ((float(stuff[clm_order.index('Xrt')])
                                                    *1e-6/3600.0)*np.pi/180.0)
                        except:
                            xprate = np.nan
                        try:
                            yprate = ((float(stuff[clm_order.index('Yrt')])
                                                    *1e-6/3600.0)*np.pi/180.0)
                        except:
                            yprate = np.nan
                        try:
                            xprate_sig = ((float(stuff[clm_order.
                                    index('Xrtsig')])*1e-6/3600.0)*np.pi/180.0)
                        except:
                            xprate_sig = np.nan
                        try:
                            yprate_sig = ((float(stuff[clm_order.
                                    index('Yrtsig')])*1e-6/3600.0)*np.pi/180.0)
                        except:
                            yprate_sig = np.nan
                        eop_data = np.append(eop_data,np.array(
                            [[mjd,xp,yp,ut1_utc,lod,xprate,yprate,
                                xp_sig,yp_sig,ut1_utc_sig,lod_sig,
                                xprate_sig,yprate_sig]]),axis=0)

        # Update attributes
        self.eop_data = eop_data

    def get_eop(self,time_utc,interp_window=4.0):

        """
        Get Earth Orientation Parameters for a given set of UTC times

        Keyword arguments:
            time_utc [datetime or list/array of datetimes] : UTC time(s)
            interp_window [float] : the window of data used around each UTC
                                    time to be used for interpolation [days]

        Updates:
            self.time_utc [array of datetimes]
            self.eop_interp [numpy.ndarray]: array containing eop information
                                             for the requested UTC times
                The columns of self.eop_interp are:
                [time_utc,xp,yp,ut1_utc,lod,xprate,yprate,
                     xp_sig,yp_sig,ut1_utc_sig,lod_sig,xprate_sig,yprate_sig]
                units are:
                [UTC datetime,rad,rad,sec,sec/day,rad,rad,
                     rad,rad,sec,sec/day,rad,rad]
        """

        # Check the given arguments and set the attributes
        if not isinstance(time_utc,(list,np.ndarray,datetime.datetime)):
            raise TypeError("The given time_utc needs to be either a datetime "
                            "object or a list/array of datetime objects")
        if not all(isinstance(item,datetime.datetime)
                        for item in np.atleast_1d(time_utc)):
            raise TypeError("There are non-datetime items in time_utc")
        self.time_utc = np.atleast_1d(time_utc)

        # Perform a lagrange interpolation to get the parameters at requested
        # epochs
        # The scipy lagrange interpolation is numerically unstable and should
        # not be used for more than 20 points; therefore, we cut only four
        # points around the desired point of interpolation; we have to do
        # this in a loop because the window for each point is different
        eop_interp = np.empty((0,7))
        for t in self.time_utc:
            year = t.year
            month = t.month
            dom = t.day
            hh = t.hour
            mm = t.minute
            sec = t.second
            gc = gpsCal()
            gc.set_yyyy_MM_dd_hh_mm_ss(year,month,dom,hh,mm,sec)
            mjd = gc.mjd()

            # Slice the eop_data to only around the requested time
            ind = np.where(abs(self.eop_data[:,0]-mjd) <= interp_window/2.0)
            if len(ind[0]) > 20:
                logger.warning("\nThe number of data points for lagrange "
                           f"interpolation = {len(ind[0])} > 20.\nThe "
                           "interpolation could be numerically unstable.\n"
                           "Consider setting a smaller interpolation window "
                           "(interp_window).")
            eop_ref = self.eop_data[ind]

            # Perform the lagrange interpolation with normalization
            mjd_mean = np.mean(eop_ref[:,0])
            mjd_scale = np.std(eop_ref[:,0])
            mjd_data = (eop_ref[:,0] - mjd_mean)/mjd_scale
            xp_mean = np.mean(eop_ref[:,1])
            xp_scale = np.std(eop_ref[:,1])
            xp_data = (eop_ref[:,1] - xp_mean)/xp_scale
            yp_mean = np.mean(eop_ref[:,2])
            yp_scale = np.std(eop_ref[:,2])
            yp_data = (eop_ref[:,2] - yp_mean)/yp_scale
            ut1_utc_mean = np.mean(eop_ref[:,3])
            ut1_utc_scale = np.std(eop_ref[:,3])
            ut1_utc_data = (eop_ref[:,3] - ut1_utc_mean)/ut1_utc_scale
            lod_mean = np.mean(eop_ref[:,4])
            lod_scale = np.std(eop_ref[:,4])
            lod_data = (eop_ref[:,4] - lod_mean)/lod_scale
            xprate_mean = np.mean(eop_ref[:,5])
            xprate_scale = np.std(eop_ref[:,5])
            xprate_data = (eop_ref[:,5] - xprate_mean)/xprate_scale
            yprate_mean = np.mean(eop_ref[:,6])
            yprate_scale = np.std(eop_ref[:,6])
            yprate_data = (eop_ref[:,6] - yprate_mean)/yprate_scale

            xp_poly = lagrange(mjd_data,xp_data)
            yp_poly = lagrange(mjd_data,yp_data)
            ut1_utc_poly = lagrange(mjd_data,ut1_utc_data)
            lod_poly = lagrange(mjd_data,lod_data)
            xprate_poly = lagrange(mjd_data,xprate_data)
            yprate_poly = lagrange(mjd_data,yprate_data)

            xp = xp_poly((mjd - mjd_mean)/mjd_scale)*xp_scale + xp_mean
            yp = yp_poly((mjd - mjd_mean)/mjd_scale)*yp_scale + yp_mean
            ut1_utc = (ut1_utc_poly((mjd - mjd_mean)/mjd_scale)*ut1_utc_scale
                                                                + ut1_utc_mean)
            lod = lod_poly((mjd - mjd_mean)/mjd_scale)*lod_scale + lod_mean
            xprate = (xprate_poly((mjd - mjd_mean)/mjd_scale)*xprate_scale
                                                                + xprate_mean)
            yprate = (yprate_poly((mjd - mjd_mean)/mjd_scale)*yprate_scale
                                                                + yprate_mean)

            eop_interp = np.append(eop_interp,np.array(
                            [[mjd,xp,yp,ut1_utc,lod,xprate,yprate]]),axis=0)

        self.eop_interp = eop_interp



class Ref_sum:

    """
    Class of reference frame combination summary file
    Reads the summary file in yaml format

    """

    def __init__(self,rfsum_yaml):

        """
        Read the reference frame summary file

        Keyword arguments:
            rfsum_yaml [str]   : filename of the reference frame summary file

        Updates:
            self.rf_sum [dict] : contents of the rf summary file
        """

        # Check the given summary filename
        if not isinstance(rfsum_yaml,str):
            logger.error("The input rfsum_yaml file needs to be a string",
                         stack_info=True)
            raise TypeError("The input rfsum_yaml needs to be a string")

        # Set eop_file attribute
        self.rfsum_yaml = rfsum_yaml

        # Try to open the summary file
        try:
            with open(self.rfsum_yaml, 'r') as stream:
                rf_sum = yaml.load(stream, Loader=yaml.SafeLoader)
        except IOError:
            logger.error(f"The SINEX combination summary file"
                         f" is not accessible!",stack_info=True)
            raise IOError(f"File {self.rfsum_yaml} not accessible!")
        except yaml.YAMLError as e:
            logger.error(f"Error loading YAML file {self.rfsum_yaml}: {e}", stack_info=True)
            raise

        # Update the attribute
        self.rf_sum = rf_sum


    def transfo(self,rf_align=[True,True,True]):

        """
        Get transformation paramaters into a more friendly dictionary type

        Keyword arguments:
            rf_align [list] : list of booleans showing if we want to keep
                              the tranformation parameters for
                              [Translations,Rotations,Scale]

        Updates:
            self.transformations [dict] : transformation parameters
        """

        transformations = {}

        for ac_item in self.rf_sum['transfo']:

            for item in ac_item:

                acname = item['ac'].upper()
                dow = item['day']
                helmert = np.zeros(7)
                helmert[6] = 1.0
                if rf_align[0]: # mm to meters
                    helmert[0:3] = np.array(item['T'])/1000.0
                # the rotation angle signs should be reversed because the
                # SINEX combinations uses a left-handed coordinate system
                if rf_align[1]: # mas to radians
                    helmert[3:6] = (
                            -np.array(item['R'])/1000.0/3600.0*np.pi/180.0)
                if rf_align[2]: # ppb to scale
                    helmert[6] = 1.0 + item['S']*1e-9

                if dow not in transformations:
                    transformations[dow] = {}
                transformations[dow][acname] = helmert

        self.transformations = transformations


    def ut1_rot(self,acname,eop_aprfile,eop_obsfile,eop_format):

        """
        Correct transformation parameters for UT1 rotation

        Keyword arguments:
            acname [str] : 3-characted name of the AC for which UT1 rotation
                        is to be applied
            eop_aprfile [str] : A-priori EOP file
            eop_obsfile [str] : Observed EOP file
            eop_format [str]  : format of the EOP files (assumes the
                                same format for a-priori and observed)

        Updates:
            self.transformations [dict] : transformation parameters
        """

        for dow in range(0,7):
            if acname in self.transformations[dow]:

                eop_apr = EOPdata(eop_aprfile[0],eop_format).eop_data
                eop_obs = EOPdata(eop_obsfile[0],eop_format).eop_data

                if np.shape(eop_apr) != np.shape(eop_obs):
                    raise ValueError(f"\nSizes of the apriori and observed"
                                    f" EOP files {eop_aprfile[0]} and "
                                    f"{eop_obsfile[0]} must be the same!")

                # correction to Z rotations (apriori Ut1 - observed UT1)
                mjd_cen = eop_apr[:,0]
                zrot = -(eop_apr[:,3] - eop_obs[:,3]) # time seconds
                zrot_dict = {}
                for i,mjd in enumerate(mjd_cen):
                    zrot_dict[mjd] = (zrot[i]*((365.25/366.25)*15/3600)
                            *np.pi/180.0) # time seconds to arc-seconds to radians

                gc = gpsCal()
                gc.set_wwww_dow(gpsweek,dow)
                year = gc.yyyy()
                doy = gc.ddd()
                yr_frac = (doy+0.5)/365.25
                yrfloat = year + yr_frac
                mjd = gc.mjd() + 0.5

                # add ut1 correction to Z rotation
                logger.debug(f"{acname} {mjd} dUT1: {zrot_dict[mjd]}")
                logger.debug(f"trn {acname} before: {self.transformations[dow][acname][5]}")
                self.transformations[dow][acname][5] += zrot_dict[mjd]
                logger.debug(f"trn {acname} after: {self.transformations[dow][acname][5]}")



class NANU_sum:

    """
    Class of nanu summary file
    Reads the nanu summary file and looks for DV events

    """

    def __init__(self,nanusum):

        """
        Read the nanu summary file

        Keyword arguments:
            nanusum [str]   : filename of the nanu summary file

        Updates:
            self.nanu_sum [dict] : dict containing info on DV events
        """

        # Check the given summary filename
        if not isinstance(nanusum,str):
            logger.error("The input nanusum file needs to be a string",
                         stack_info=True)
            raise TypeError("The input nanusum needs to be a string")

        # Set nanu_sum attribute
        self.nanusum_file = nanusum

        # Initialize nanu array
        nanu_sum  = {}

        # Try to open the nanu file and fill in the nanu_sum dict
        try:
            nanu_fid = open(self.nanusum_file,'r')
        except IOError:
            logger.error(f"NANU summary file {self.nanusum_file} is not "
                         f"accessible!", stack_info=True)
            raise IOError(f"File {self.nanusum_file} not found!")
        else:
            with nanu_fid:
                read_flag = False
                for line in nanu_fid:
                    if "+nanu_sum" in line:
                        read_flag = True
                        continue
                    elif "-nanu_sum" in line:
                        read_flag = False
                    if read_flag and line.strip():
                        if ("nanu_sum" not in line and line[0]!="*"):
                            nanu_number = int(line[1:8])
                            nanu_type = line[9:19].strip()
                            if "-" not in line[20:23]:
                                prn = int(line[20:23])
                            else:
                                prn = 0
                            if "-" not in line[25:32]:
                                ref_nanu = int(line[25:32])
                            else:
                                ref_nanu = 0
                            if "-" not in line[34:47]:
                                year_start = int(line[34:38])
                                doy_start  = int(line[39:42])
                                hr_start   = int(line[43:45])
                                min_start  = int(line[45:47])
                            else:
                                year_start = 9999
                                doy_start  = 999
                                hr_start   = 99
                                min_start  = 99
                            if "-" not in line[49:62]:
                                year_end = int(line[49:53])
                                doy_end  = int(line[54:57])
                                hr_end   = int(line[58:60])
                                min_end  = int(line[60:62])
                            else:
                                year_end = 9999
                                doy_end  = 999
                                hr_end   = 99
                                min_end  = 99
                            nanu_sum[nanu_number] = (
                                    [nanu_type,prn,ref_nanu,
                                     year_start,doy_start,hr_start,min_start,
                                     year_end,doy_end,hr_end,min_end])
        self.nanu_sum = nanu_sum


    def get_dv(self,solution):

        """
        Get DV maneuver epochs

        Input:
            solution [str] : solution: ultra-rapid/rapid/final

        Updates:
            self.dv [array] : array of year,doy, start epoch and PRN for the
                              satellites experiencing maneuvers
            self.dvfull [array] : array of year,doy, start and end epoch and PRN
                                  for the maneuvering satellites
        """

        dv = np.empty((0,2))
        dvfull = np.empty((0,3))
        nanu_sum = self.nanu_sum
        for nanu in nanu_sum:
            nanu_type = nanu_sum[nanu][0]
            logger.debug(f"RM_DV solution: {solution}")
            if solution != "ultra-rapid":
                if nanu_type == 'FCSTSUMM':
                    ref_nanu = nanu_sum[nanu][2]
                    if ref_nanu in nanu_sum:
                        nanu_type_ref = nanu_sum[ref_nanu][0]
                        if nanu_type_ref == 'FCSTDV':
                            prn = 'G'+str(nanu_sum[nanu][1]).zfill(2)
                            year_start = nanu_sum[nanu][3]
                            doy_start = nanu_sum[nanu][4]
                            hr_start = nanu_sum[nanu][5]
                            min_start = nanu_sum[nanu][6]
                            year_end = nanu_sum[nanu][7]
                            doy_end = nanu_sum[nanu][8]
                            hr_end = nanu_sum[nanu][9]
                            min_end = nanu_sum[nanu][10]
                            gc = gpsCal()
                            gc.set_yyyy_ddd(year_start,doy_start)
                            month_start = gc.MM()
                            dom_start = gc.dom()
                            dt_start = datetime.datetime(
                                year_start,month_start,dom_start,hr_start,min_start)
                            gc = gpsCal()
                            gc.set_yyyy_ddd(year_end,doy_end)
                            month_end = gc.MM()
                            dom_end = gc.dom()
                            dt_end = datetime.datetime(
                                year_end,month_end,dom_end,hr_end,min_end)
                            dv = np.append(dv,np.array([[dt_start,prn]]),axis=0)
                            dvfull = np.append(dvfull,np.array([[dt_start,dt_end,prn]]),axis=0)
                            delta = datetime.timedelta(days=1)
                            dt = dt_start + delta
                            dt = dt.replace(hour=0,minute=0)
                            while dt <= dt_end:
                                dv = np.append(dv,np.array([[dt,prn]]),axis=0)
                                dt += delta
            else:
                logger.debug(f"RM_DV solution is {solution}")
                if nanu_type == 'FCSTDV':
                    prn = 'G'+str(nanu_sum[nanu][1]).zfill(2)
                    year_start = nanu_sum[nanu][3]
                    doy_start = nanu_sum[nanu][4]
                    hr_start = nanu_sum[nanu][5]
                    min_start = nanu_sum[nanu][6]
                    year_end = nanu_sum[nanu][7]
                    doy_end = nanu_sum[nanu][8]
                    hr_end = nanu_sum[nanu][9]
                    min_end = nanu_sum[nanu][10]
                    gc = gpsCal()
                    gc.set_yyyy_ddd(year_start,doy_start)
                    month_start = gc.MM()
                    dom_start = gc.dom()
                    dt_start = datetime.datetime(
                        year_start,month_start,dom_start,hr_start,min_start)
                    gc = gpsCal()
                    gc.set_yyyy_ddd(year_end,doy_end)
                    month_end = gc.MM()
                    dom_end = gc.dom()
                    dt_end = datetime.datetime(
                        year_end,month_end,dom_end,hr_end,min_end)
                    dv = np.append(dv,np.array([[dt_start,prn]]),axis=0)
                    dvfull = np.append(dvfull,np.array([[dt_start,dt_end,prn]]),axis=0)
                    delta = datetime.timedelta(days=1)
                    dt = dt_start + delta
                    dt = dt.replace(hour=0,minute=0)
                    while dt <= dt_end:
                        dv = np.append(dv,np.array([[dt,prn]]),axis=0)
                        dt += delta
        logger.debug(f"RM_DV dv: {dv}")
        logger.debug(f"RM_DV dvfull: {dvfull}")
        self.dv = dv
        self.dvfull = dvfull

