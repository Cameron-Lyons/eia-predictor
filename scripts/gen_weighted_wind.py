from shapely.ops import nearest_points
from sklearn.neighbors import BallTree
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import sys
import time

WEATHER_VARS = [
    'TMAX',
    'TMIN',
    'PRCP',
    'SNOW',
    'AWND'
]

VERBOSE = False
STAT_FILE = sys.stdout

def find_wind_stations(wind_plant_data, required_data=[]):
    """
    Takes input list of power plants and finds
    nearest weather station to each power plant
    with the required_data.

    Returns: Station, Associated Power Plant
    """

    station_url = 'https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt'
    # weather historicals API https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation
    # Data descriptions https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt
    # Data import and formatting with headers
    station_list = pd.read_fwf(station_url, header=None)
    station_list.columns = [
        'ID',
        'LATITUDE',
        'LONGITUDE',
        'DATA',
        'START_YEAR',
        'END_YEAR'
    ]


    station_list = station_list.pivot(
        index=['ID','LATITUDE','LONGITUDE'], 
        columns='DATA', 
        values='END_YEAR').reset_index()
    #Filtering list to items of interest and available data in 2021

    station_list = station_list[
        (station_list['TMAX']==2021) & 
        (station_list['TMIN']==2021) & 
        (station_list['PRCP']==2021) & 
        (station_list['SNOW']==2021) & 
        (station_list['AWND']==2021) 
    ]

    station_list.to_csv('data/noaa/tmp_station_list.csv')

    station_list = station_list[[
        'ID', 
        'LATITUDE',
        'LONGITUDE',
        'TMAX',
        'TMIN',
        'PRCP',
        'SNOW',
        'AWND',
    ]]

    # Now we have a list of stations, need to find nearest
    # to each power station

    station_gpd = gpd.GeoDataFrame(
        station_list, 
        geometry=gpd.points_from_xy(station_list.LONGITUDE, station_list.LATITUDE)
    )

    plant_gpd = gpd.GeoDataFrame(
        wind_plant_data, 
        geometry=gpd.points_from_xy(wind_plant_data.Longitude, wind_plant_data.Latitude)
    )

    # Create a BallTree 
    tree = BallTree(station_gpd[['LONGITUDE', 'LATITUDE']].values, leaf_size=2)

    # Query the BallTree on each feature
    plant_gpd['distance_nearest'], plant_gpd['index_nearest'] = tree.query(
        plant_gpd[['Longitude', 'Latitude']].values, # The input array for the query
        k=1, # The number of nearest neighbors
    )

    #Get actual IDs
    plant_gpd['ID_nearest'] = 0
    for index, row in plant_gpd.iterrows():
        #print(station_gpd.iloc[row['index_nearest']].ID)
        plant_gpd.loc[index,'ID_nearest']= station_gpd.iloc[row['index_nearest']].ID

    return plant_gpd

def query_wind_stations(station_listing, start_date, end_date, required_data=[]):
    """
    Queries NOAA for requested weather stations
    and requested data

    Returns: station, [required_data]
    """
    req_list = ",".join(required_data)

    max_i = len(station_listing)
    i = 0
    for station in station_listing:
        if VERBOSE:
            i += 1
            print('Querying station #'+str(i)+'/'+str(max_i), file=STAT_FILE, flush=True)
        station_query = ('https://www.ncei.noaa.gov/access/services/data/v1?'+
                        'dataset=daily-summaries&'+
                        'dataTypes='+req_list+"&"+
                        'stations='+station+'&'+
                        'startDate='+start_date+'&'+
                        'endDate='+end_date+'&'+
                        'includeAttributes=0&'+
                        'format=json')
        #print(station_query)
        response = requests.get(station_query)
        #print(response)
        l_weather_df = pd.DataFrame(response.json())
        #print(l_weather_df.shape)
        if station == station_listing[0]:
            weather_df = l_weather_df
        else:
            weather_df = pd.concat([weather_df,l_weather_df])

        #break
        
    return weather_df

def weight_wind_data(power_plant_data, weather_df):
    """
    Takes requested weather data and power plant data
    and returns a dataframe with weather data weighted
    by nameplate capacity across an interconnect then
    outputs to file

    Returns: 0
    """

    weather_df = weather_df.merge(
        power_plant_data, 
        left_on='STATION', 
        right_on='ID_nearest'
    )

    #calculate capacity weighted wind speed
    weather_df['gen_awnd'] = pd.to_numeric(weather_df['Gen_MW'])*pd.to_numeric(weather_df['AWND'])
    weather_df['gen_tmax'] = pd.to_numeric(weather_df['Gen_MW'])*pd.to_numeric(weather_df['TMAX'])
    weather_df['gen_tmin'] = pd.to_numeric(weather_df['Gen_MW'])*pd.to_numeric(weather_df['TMIN'])
    weather_df['gen_prcp'] = pd.to_numeric(weather_df['Gen_MW'])*pd.to_numeric(weather_df['PRCP'])
    weather_df['gen_snow'] = pd.to_numeric(weather_df['Gen_MW'])*pd.to_numeric(weather_df['SNOW'])

    #Sum calculated weighted variables
    weather_calc_df = weather_df[[
        'DATE',
        'eia_short_name',
        'Gen_MW',
        'gen_awnd',
        'gen_tmax',
        'gen_tmin',
        'gen_prcp',
        'gen_snow']].groupby(['DATE','eia_short_name']).sum().reset_index()
    weather_calc_df['DATE'] = pd.to_datetime(weather_calc_df['DATE'])
    #Normalize by installed capacity p_cap
    weather_calc_df['Weighted_AWND'] = weather_calc_df['gen_awnd']/weather_calc_df['Gen_MW']
    weather_calc_df['Weighted_TMAX'] = weather_calc_df['gen_tmax']/weather_calc_df['Gen_MW']
    weather_calc_df['Weighted_TMIN'] = weather_calc_df['gen_tmin']/weather_calc_df['Gen_MW']
    weather_calc_df['Weighted_PRCP'] = weather_calc_df['gen_prcp']/weather_calc_df['Gen_MW']
    weather_calc_df['Weighted_SNOW'] = weather_calc_df['gen_snow']/weather_calc_df['Gen_MW']
 
    #Drop columns no longer needed
    weather_calc_df = weather_calc_df.drop(columns=[
        'gen_awnd',
        'gen_tmax',
        'gen_tmin',
        'gen_prcp',
        'gen_snow'
        ])
    
    weather_calc_df = weather_calc_df.rename(columns={'Gen_MW':'Installed_MW'})
    #weather_calc_df.to_csv('./data/noaa/wind_weighted_power.csv', index=False)

    return weather_calc_df

if __name__ == '__main__':
    import argparse

    #'./data/noaa/uscrn_combined.csv''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_power_plant', help='Input wind power plants (CSV)')
    parser.add_argument(
        'output_wind_file', help='Output weighted wind data (CSV)')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='display detailed output',
    )
    args = parser.parse_args()
    
    if args.verbose:
        VERBOSE = True

    if VERBOSE:
        print('Begining UPDATE_WIND...', file=STAT_FILE, flush=True)
        start = time.time()
    wind_plant_data = pd.read_csv(args.input_power_plant)
    wind_station_listing = find_wind_stations(wind_plant_data,WEATHER_VARS)
    if VERBOSE:
        duration = time.time() - start
        print('find_wind_stations total time: {:.2f} seconds'.format(duration), file=STAT_FILE, flush=True)

    if VERBOSE:
        print('Begining query_wind_stations...', file=STAT_FILE, flush=True)
        start = time.time()
    wind_weather_data = query_wind_stations(
        wind_station_listing['ID_nearest'].unique(),
        '2018-07-01',
        '2021-04-14',
        WEATHER_VARS)
    if VERBOSE:
        duration = time.time() - start
        print('query_wind_stations total time: {:.2f} seconds'.format(duration), file=STAT_FILE, flush=True)
    #print(weather_data)
    if VERBOSE:
        print('Begining weight_wind_data...', file=STAT_FILE, flush=True)
        start = time.time()
    weight_wind_df = weight_wind_data(wind_station_listing,wind_weather_data)
    weight_wind_df.to_csv(args.output_wind_file, index=False)
    if VERBOSE:
        duration = time.time() - start
        print('weight_wind_data total time: {:.2f} seconds'.format(duration), file=STAT_FILE, flush=True)
