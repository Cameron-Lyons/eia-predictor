import pandas as pd
import sys
import time

VERBOSE = False
STAT_FILE = sys.stdout

def join_datasets(solar_input,wind_input,inter_input):
    """
    Joins interconnect data with weather data
    by interconnect and outputs to file

    Returns: 0
    """
    if VERBOSE:
        print('Loading dataframes...', file=STAT_FILE, flush=True)
        
    inter_df = pd.read_csv(inter_input, parse_dates=['datetime'])
    solar_df = pd.read_csv(solar_input)
    wind_df = pd.read_csv(wind_input)

    if VERBOSE:
        print('Correcting datetime...', file=STAT_FILE, flush=True)
        
    #inter_df['datetime'] = pd.to_datetime(inter_df['datetime'])
    solar_df['LST_DATE'] = pd.to_datetime(solar_df['LST_DATE'])
    wind_df['DATE'] = pd.to_datetime(wind_df['DATE'])

    solar_df['LST_DATE'] = solar_df['LST_DATE'].apply(lambda x: x.date())
    wind_df['DATE'] = wind_df['DATE'].apply(lambda x: x.date())

    if VERBOSE:
        print('Consolidating interconnect hourly data...', file=STAT_FILE, flush=True)
        
    inter_df['date'] = inter_df['datetime'].apply(lambda x: x.date())
    inter_df = inter_df.groupby([
        'interconnect_long',
        'interconnect_short',
        'data_type',
        'date'
    ]).sum().reset_index()

    inter_df = inter_df[[
        'interconnect_long',
        'interconnect_short',
        'data_type',
        'date',
        'generation'
    ]]

    if VERBOSE:
        print('Merging wind...', file=STAT_FILE, flush=True)
        
    inter_df_wind = inter_df[inter_df['data_type']=='Wind']
    wind_df = wind_df.merge(inter_df_wind, 
        left_on=['DATE','eia_short_name'], 
        right_on=['date','interconnect_short']
    )

    if VERBOSE:
        print('Merging solar...', file=STAT_FILE, flush=True)
        
    inter_df_solar = inter_df[inter_df['data_type']=='Solar']
    solar_df = solar_df.merge(inter_df_solar, 
        left_on=['LST_DATE','eia_short_name'], 
        right_on=['date','interconnect_short']
    )


    return solar_df,wind_df


if __name__ == '__main__':
    import argparse

    #'./data/noaa/uscrn_combined.csv''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_solar_file', help='Input weighted solar data (CSV)')
    parser.add_argument(
        'input_wind_file', help='Input weighted wind data (CSV)')
    parser.add_argument(
        'input_interconnect', help='Interconnect hourly power generation (CSV)')
    parser.add_argument(
        'output_solar', help='Output final solar dataframe (CSV)')
    parser.add_argument(
        'output_wind', help='Output final wind dataframe (CSV)')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='display detailed output',
    )
    args = parser.parse_args()
    
    if args.verbose:
        VERBOSE = True

    if VERBOSE:
        print('Begining UPDATE_FINAL...', file=STAT_FILE, flush=True)
        start = time.time()
    solar_df, wind_df = join_datasets(
        args.input_solar_file,
        args.input_wind_file,
        args.input_interconnect,
    )
    solar_df.to_csv(args.output_solar, index=False)
    wind_df.to_csv(args.output_wind, index=False)
    if VERBOSE:
        duration = time.time() - start
        print('join_datasets total time: {:.2f} seconds'.format(duration), file=STAT_FILE, flush=True)
