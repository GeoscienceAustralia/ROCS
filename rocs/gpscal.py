# GPS time calculations

import numpy as np
import datetime as dt
import math

class gpsCal:

    # First GPS epoch 
    # GPS week 0000 => 1980, 01 , 06
    gpsE0 = dt.datetime(int(1980),int(1),int(6))
    oned  = dt.timedelta(days=1)


    def __init__(self):
        self.dto = dt.datetime.utcnow()


    def dto(self):
        return self.dto


    def calendar(self):
        #Sun Aug 20.08.2017 1963.0 57985 232
        format = "%a %b %d %H:%M:%S %Y %j"
        for i in range(0,8):
            s = self.dto.strftime(format)
            s = s + " "+str(self.wwww())+" "+str(self.dow())
            s = s + " "+str(self.mjd())
            print(s)
            self.dto = self.dto + self.oned


    def dec_day(self,ndays=1):
        """
        dec_day(ndays=1)
        decrement the time by ndays (default is 1)
        """
        self.dto = self.dto - (self.oned * ndays)


    def yyyy(self):
        return int(self.dto.strftime("%Y"))


    def yy(self):
        yy = int(str(self.yyyy())[2:4])
        return yy


    def ddd(self):
        return int(self.dto.strftime("%j"))


    def MM(self):
        return int(self.dto.strftime("%m"))


    def dom(self):
        return int(self.dto.strftime("%d"))


    def hh(self):
        """
        Return the hour (int)
        """
        return int(self.dto.strftime("%H"))


    def mm(self):
        """
        Return the minutes (int)
        """
        return int(self.dto.strftime("%M"))


    def ss(self):
        """
        Return the seconds (int)
        """
        return int(self.dto.strftime("%S"))


    def ms(self):
        """
        Return the miliseconds (int)
        """
        return int(self.dto.strftime("%f"))


    def wwww(self):
        """
        Workout the GPS week
        """
        diff  = self.dto - self.gpsE0
        diff_days = diff.days
        week = int(diff_days/7.)
        return week


    def dow(self):
        """
        Workout the GPS day-of-week
        """
        diff  = self.dto - self.gpsE0
        diff_days = diff.days
        week = int(diff_days/7.)
        dow = int(diff_days - (week * 7.))
        return dow


    def sow(self):
        """
        Workout the seconds-of-week
        """
        diff  = self.dto - self.gpsE0
        diff_days = diff.days
        week = int(diff_days/7.)
        dow = int(diff_days - (week * 7.))
        sod = self.hh()*3600 + self.mm()*60 + self.ss() + self.ms()/1000
        sow = (dow*86400)+sod
        return sow


    def jd(self):
        """
        Convert Gregorian Calendar date to Julian Date based on
        the US Navy's Astronomical Equation
        Script adapted from:
        https://stackoverflow.com/a/52431241
        Only valid for 1801 to 2099
        """

        year = self.yyyy()
        month = self.MM()
        day = self.dom()
        hour = self.hh()
        minute = self.mm()
        second = self.ss()

        jd = 367*year - int((7 * (year + int((month + 9) / 12.0))) / 4.0) + int(
                (275 * month) / 9.0) + day + 1721013.5 + (
                 hour + minute / 60.0 + second / math.pow(60,2)) / 24.0 - 0.5 * math.copysign(
                 1, 100 * year + month - 190002.5) + 0.5
        return jd


    def mjd(self):
        return self.jd() - 2400000.5


    def mdt(self):
        """
        mdt Take a date time object and convert it into a matplotlib date
                   All matplotlib date plotting is done by converting date instances into
                   days since the 0001-01-01 UTC

        Usage:  mp_ts = mdt()

        Input:  dto  - a datetime object

        Output: 'mp_ts' (float)
                a matplotlib time stamp which is days from 0001-01-01

        """

        mp_epoch = dt.datetime(1, 1, 1)
        DAY = 86400
        td = self.dto - mp_epoch
        mp_ts = td.days + 1 + (1000000 * td.seconds + td.microseconds) / 1e6 / DAY
        return mp_ts


    def tai(self):
        """
        Calculate TAI (which does not implement leap seconds)
        """
        leap_dates = np.array([
                        dt.datetime(1972,1,1),
                        dt.datetime(1972,7,1),
                        dt.datetime(1973,1,1),
                        dt.datetime(1974,1,1),
                        dt.datetime(1975,1,1),
                        dt.datetime(1976,1,1),
                        dt.datetime(1977,1,1),
                        dt.datetime(1978,1,1),
                        dt.datetime(1979,1,1),
                        dt.datetime(1980,1,1),
                        dt.datetime(1981,7,1),
                        dt.datetime(1982,7,1),
                        dt.datetime(1983,7,1),
                        dt.datetime(1985,7,1),
                        dt.datetime(1988,1,1),
                        dt.datetime(1990,1,1),
                        dt.datetime(1991,1,1),
                        dt.datetime(1992,7,1),
                        dt.datetime(1993,7,1),
                        dt.datetime(1994,7,1),
                        dt.datetime(1996,1,1),
                        dt.datetime(1997,7,1),
                        dt.datetime(1999,1,1),
                        dt.datetime(2006,1,1),
                        dt.datetime(2009,1,1),
                        dt.datetime(2012,7,1),
                        dt.datetime(2015,7,1),
                        dt.datetime(2017,1,1),
                                                ])
        ind = np.where(leap_dates < self.dto)
        leap_seconds = len(ind[0]) + 9
        tai = self.dto + dt.timedelta(0,leap_seconds)
        return tai


    def ut1(self,ut1_utc):
        """
        Calculate UT1
        ut1_utc: UT1-UTC in seconds
        """
        ut1 = self.dto + dt.timedelta(0,ut1_utc)
        return ut1


    def gpstime(self):
        """
        Calculate GPS time
        """
        return self.tai() - dt.timedelta(0,19)


    def set_mdt(self,mdt):
        """
        Set the time using the matplotlib date time stamp
        """
        mp_epoch = dt.datetime(1,1,1)
        stamp = mp_epoch + dt.timedelta(days=(int(mdt)-1))
        self.dto = stamp
        return self


    def set_yyyy_ddd(self,yyyy,ddd):
        """
        Set the time using the format YYYY DDD
        """
        dto = dt.datetime(int(yyyy),1,1,0,0,0,0)
        dto = dto + dt.timedelta(days=(int(ddd) - 1))
        self.dto = dto
        return self


    def set_yyyy_ddd_sod(self,yyyy,ddd,sod):
        """
        Set the time using the format YYYY DDD SOD
        ddd: day of year
        sod: seconds of day
        """
        dto = dt.datetime(int(yyyy),1,1,0,0,0,0)
        dto = dto + dt.timedelta(days=(int(ddd) - 1 + sod/86400.0))
        self.dto = dto
        return self


    def yy2yyyy(self,yy):
        """
        Convert YY to YYYY
        """
        if yy >= 80 and yy < 100:
            yyyy = yy + 1900
        elif yy >= 0 and yy < 80:
            yyyy = yy + 2000
        else:
            print("Error value should be between 0 and 99")
        return yyyy


    def set_yy_ddd(self,yy,ddd):
        """
        Set the time using the format YY DDD
        """
        yyyy = self.yy2yyyy(yy)
        dto = dt.datetime(int(yyyy),1,1,0,0,0,0)
        dto = dto + dt.timedelta(days=(int(ddd) - 1))
        self.dto = dto


    def set_yyyy_MM_dd_hh_mm_ss(self,yyyy,MM,dd,hh,mm,ss):
        """
        Set the time using the format YY DDD
        """
        dto = dt.datetime(int(yyyy),int(MM),int(dd),int(hh),int(mm),int(ss),0)
        self.dto = dto


    def set_wwww(self,wwww):
        """
        set the time to the GPS week
        """
        self.dto = ( (wwww * 7) * self.oned ) + self.gpsE0


    def set_wwww_dow(self,wwww,dow):
        """
        set the time to the GPS week
        """
        self.dto = ( (wwww * 7) * self.oned ) + self.gpsE0 + (dow*self.oned)


    def yyyy_MM_dd_mm_ss_ms(self):
        """
           yyyy_MM_dd_mm_ss_ms(dto)
        Return the values needed to form a valid from string,
        from a date time object

        """
        rtn = []
        rtn.append( self.yyyy() )
        rtn.append( self.MM() )
        rtn.append( self.dom() )
        rtn.append( self.hh() )
        rtn.append( self.mm() )
        rtn.append( self.ss() )
        rtn.append( self.ms() )
        return rtn

#=========================

if __name__ == "__main__":

    gc = gpsCal()
    gc.calendar()
    print("yy;",gc.yy())
    print("ms;",gc.ms())
    gc.set_yyyy_ddd(2017,1)
    print("yy",gc.yy())
    print("ddd",gc.ddd())
    gc.set_yy_ddd(17,151)
    print("yyyy",gc.yyyy())
    print("ddd",gc.ddd())
    print("yyyy_MM_dd_mm_ss_ms:",gc.yyyy_MM_dd_mm_ss_ms())
    gc.set_wwww(1963)
    print("yyyy",gc.yyyy())
    print("ddd",gc.ddd())
    print("yyyy_MM_dd_mm_ss_ms:",gc.yyyy_MM_dd_mm_ss_ms())
    gc.dec_day(3)
    print("yyyy",gc.yyyy())
    print("ddd",gc.ddd())
    print("yyyy_MM_dd_mm_ss_ms:",gc.yyyy_MM_dd_mm_ss_ms())
    print("====================================================================")
    print("yyyy_MM_dd_mm_ss_ms:",gc.yyyy_MM_dd_mm_ss_ms())
    print("mdt",gc.mdt())
    mdt = gc.mdt()
    print("gc.set_mdt(mdt)")
    gc.set_mdt(mdt)
    print("yyyy_MM_dd_mm_ss_ms:",gc.yyyy_MM_dd_mm_ss_ms())
    print("====================================================================")
    print("Using a mdt of 736550.0 ")
    gc.set_mdt(736550.0)
    print("mdt",gc.mdt())
    print("yyyy_MM_dd_mm_ss_ms:",gc.yyyy_MM_dd_mm_ss_ms())
    print("====================================================================")
