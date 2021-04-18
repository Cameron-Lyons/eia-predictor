import ftplib
from ftp_walk import FTPWalk
import io
import pandas as pd
import numpy as np
import sys
import time

VERBOSE = False
STAT_FILE = sys.stdout

def make_uscrn_df(in_file):
    """
    Combines USCRN data into a single CSV for
    ease of use later
    """
    l_read = in_file.read()
    l_read = l_read.decode('utf-8')
    header_list = [
        'WBANNO',
        'LST_DATE',
        'CRX_VN',
        'LONGITUDE',
        'LATITUDE',
        'T_DAILY_MAX',
        'T_DAILY_MIN',
        'T_DAILY_MEAN',
        'T_DAILY_AVG',
        'P_DAILY_CALC',
        'SOLARAD_DAILY',
        'SUR_TEMP_DAILY_TYPE',
        'SUR_TEMP_DAILY_MAX',
        'SUR_TEMP_DAILY_MIN',
        'SUR_TEMP_DAILY_AVG',
        'RH_DAILY_MAX',
        'RH_DAILY_MIN',
        'RH_DAILY_AVG',
        'SOIL_MOISTURE_5_DAILY',
        'SOIL_MOISTURE_10_DAILY',
        'SOIL_MOISTURE_20_DAILY',
        'SOIL_MOISTURE_50_DAILY',
        'SOIL_MOISTURE_100_DAILY',
        'SOIL_TEMP_5_DAILY',
        'SOIL_TEMP_10_DAILY',
        'SOIL_TEMP_20_DAILY',
        'SOIL_TEMP_50_DAILY',
        'SOIL_TEMP_100_DAILY',
    ]

    l_station = pd.read_fwf(io.StringIO(l_read), header=None)
    l_station.columns = header_list
    
    return l_station

def uscrn_ftp(out_csv,years=['2018','2019','2020','2021']):
    header_list = [
        'WBANNO',
        'LST_DATE',
        'CRX_VN',
        'LONGITUDE',
        'LATITUDE',
        'T_DAILY_MAX',
        'T_DAILY_MIN',
        'T_DAILY_MEAN',
        'T_DAILY_AVG',
        'P_DAILY_CALC',
        'SOLARAD_DAILY',
        'SUR_TEMP_DAILY_TYPE',
        'SUR_TEMP_DAILY_MAX',
        'SUR_TEMP_DAILY_MIN',
        'SUR_TEMP_DAILY_AVG',
        'RH_DAILY_MAX',
        'RH_DAILY_MIN',
        'RH_DAILY_AVG',
        'SOIL_MOISTURE_5_DAILY',
        'SOIL_MOISTURE_10_DAILY',
        'SOIL_MOISTURE_20_DAILY',
        'SOIL_MOISTURE_50_DAILY',
        'SOIL_MOISTURE_100_DAILY',
        'SOIL_TEMP_5_DAILY',
        'SOIL_TEMP_10_DAILY',
        'SOIL_TEMP_20_DAILY',
        'SOIL_TEMP_50_DAILY',
        'SOIL_TEMP_100_DAILY',
    ]

    return_df = pd.DataFrame(columns=header_list)

    

    with ftplib.FTP("ftp.ncei.noaa.gov") as ftp:
        ftp.login()
        ftp.cwd('/pub/data/uscrn/products/daily01')
        #print()

        ftpwalk = FTPWalk(ftp)
        
        for i in ftpwalk.walk(path='/pub/data/uscrn/products/daily01'):
            if VERBOSE:
                print('Scanning directory: '+i[0])
            if any(year in i[0] for year in years) and (not 'updates' in i[0]):
                if VERBOSE:
                    print('Reading files...')
                for l_file in i[2]:
                    sio = io.BytesIO()
                    retr_string = "RETR "+i[0]+'/'+l_file
                    #print(retr_string)
                    resp = ftp.retrbinary(retr_string, lambda data: sio.write(data))
                    sio.seek(0) # Go back to the start
                    l_df = make_uscrn_df(sio)
                    return_df = pd.concat([return_df,l_df])
                    
    return_df['LST_DATE'] = pd.to_datetime(return_df['LST_DATE'], format='%Y%m%d')
    return_df = return_df.replace(-9999,np.nan)
    return_df[[
            'SOIL_MOISTURE_5_DAILY',
            'SOIL_MOISTURE_10_DAILY',
            'SOIL_MOISTURE_20_DAILY',
            'SOIL_MOISTURE_50_DAILY',
            'SOIL_MOISTURE_100_DAILY',
        ]] = return_df[[
            'SOIL_MOISTURE_5_DAILY',
            'SOIL_MOISTURE_10_DAILY',
            'SOIL_MOISTURE_20_DAILY',
            'SOIL_MOISTURE_50_DAILY',
            'SOIL_MOISTURE_100_DAILY',
        ]].replace(-99,np.nan)
        
    return_df.to_csv(out_csv, index=False)
    
    return 0


if __name__ == '__main__':
    import argparse

    #'./data/noaa/uscrn_combined.csv''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'output_file', help='USCRN weather data (CSV)')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='display detailed output',
    )
    args = parser.parse_args()

    if args.verbose:
        VERBOSE = True

    if VERBOSE:
            print('Begining uscrn_ftp...', file=STAT_FILE, flush=True)
            start = time.time()
    uscrn_ftp(args.output_file)
    if VERBOSE:
        duration = time.time() - start
        print('uscrn_ftp total time: {:.2f} seconds'.format(duration), file=STAT_FILE, flush=True)
