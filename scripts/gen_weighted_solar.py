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

def sunlight_calculation(julian_day, latitude):
    #https://www.esrl.noaa.gov/gmd/grad/solcalc/calcdetails.html

    julian_century =(julian_day-2451545)/36525

    geom_mean_anom_sun =  357.52911+julian_century*(
        35999.05029 - 0.0001537*julian_century
    )

    sun_eq_of_ctr = np.sin(
        np.radians(geom_mean_anom_sun)
    )*(
        1.914602-julian_century*(
            0.004817+0.000014*julian_century
        )
    )+np.sin(
        np.radians(2*geom_mean_anom_sun)
    )*(
        0.019993-0.000101*julian_century
    )+np.sin(
        np.radians(3*geom_mean_anom_sun)
    )*0.000289

    geom_mean_long_sun=np.mod(
        280.46646+julian_century*(
            36000.76983 + julian_century*0.0003032
        ),360)

    sun_true_long = sun_eq_of_ctr+geom_mean_long_sun

    sun_app_long = sun_true_long-0.00569-0.00478*np.sin(
        np.radians(
                125.04-1934.136*julian_century
            )
        )

    mean_obliq_ecliptic = 23+(
        26+(
            (21.448-julian_century*(
                46.815+julian_century*(0.00059-julian_century*0.001813)
            ))
        )/60
    )/60

    obliq_corr = mean_obliq_ecliptic+0.00256*np.cos(
        np.radians(125.04-1934.136*julian_century)
    )

    sun_declin =np.degrees(
        np.arcsin(np.sin(np.radians(obliq_corr))*np.sin(np.radians(sun_app_long)))
    )

    ha_sunrise = np.degrees(
        np.arccos(
            np.cos(
                np.radians(90.833)
            )/(
                np.cos(np.radians(latitude))*np.cos(np.radians(sun_declin))
            )-np.tan(np.radians(latitude))*np.tan(np.radians(sun_declin))
        )
    )

    sunlight_duration = ha_sunrise*8

    return sunlight_duration


def find_solar_stations(solar_plant_data, uscrn_csv):
    """
    Returns stations for solar power plants
    using data from 
    ftp://ftp.ncei.noaa.gov/pub/data/uscrn/products/daily01
    More details on the dataset available from
    https://www.ncdc.noaa.gov/crn/qcdatasets.html
    """

    weather_data = pd.read_csv(uscrn_csv)

    station_list = weather_data.groupby(['WBANNO','LONGITUDE','LATITUDE']).count().reset_index()

    station_list = station_list[['WBANNO','LONGITUDE','LATITUDE']]

    station_gpd = gpd.GeoDataFrame(
        station_list, 
        geometry=gpd.points_from_xy(station_list.LONGITUDE, station_list.LATITUDE)
    )

    plant_gpd = gpd.GeoDataFrame(
        solar_plant_data, 
        geometry=gpd.points_from_xy(solar_plant_data.Longitude, solar_plant_data.Latitude)
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
        plant_gpd.loc[index,'ID_nearest']= station_gpd.iloc[row['index_nearest']].WBANNO

    unique_stations = plant_gpd['ID_nearest'].unique()
    weather_data = weather_data[weather_data['WBANNO'].isin(unique_stations)]

    return plant_gpd, weather_data


def weight_solar_data(power_plant_data, weather_df):

    solar_weather_list = [
        'T_DAILY_MAX',
        'T_DAILY_MIN',
        'T_DAILY_MEAN',
        'T_DAILY_AVG',
        'P_DAILY_CALC',
        'SOLARAD_DAILY',
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
        'SUNLIGHT_MIN'
    ]

    weather_df = weather_df.merge(
        power_plant_data, 
        left_on='WBANNO', 
        right_on='ID_nearest'
    )

    weather_df['SUNLIGHT_MIN'] = sunlight_calculation(
        pd.DatetimeIndex(weather_df['LST_DATE']).to_julian_date(), 
        weather_df['LATITUDE']
    )

    filter_list = ['LST_DATE',
        'eia_short_name',
        'Gen_MW']

    for s_item in solar_weather_list:
        #print(s_item)
        s_name = 'Weighted_'+s_item
        #print(s_name)
        filter_list.append(s_name)
        #print(weather_df['Gen_MW'],weather_df[s_item])
        weather_df[s_name] = pd.to_numeric(weather_df['Gen_MW'])*pd.to_numeric(weather_df[s_item])

    weather_calc_df = weather_df[filter_list].groupby(
            ['LST_DATE','eia_short_name']
        ).sum().reset_index()

    norm_list = [x for x in filter_list if x not in [
        'LST_DATE',
        'eia_short_name',
        'Gen_MW']
    ]

    for s_item in norm_list:
        weather_calc_df[s_item] = weather_calc_df[s_item]/weather_calc_df['Gen_MW']

    weather_calc_df = weather_calc_df.rename(columns={'Gen_MW':'Installed_MW'})
    #weather_calc_df.to_csv('./data/noaa/solar_weighted_power.csv', index=False)

    return weather_calc_df


if __name__ == '__main__':
    import argparse

    #'./data/noaa/uscrn_combined.csv''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_power_plant', help='Input solar power plants (CSV)')
    parser.add_argument(
        'input_uscrn', help='USCRN weather data (CSV)')
    parser.add_argument(
        'output_solar_file', help='Output weighted solar data (CSV)')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='display detailed output',
    )
    args = parser.parse_args()
    
    if args.verbose:
        VERBOSE = True

    if VERBOSE:
        print('Begining UPDATE_SOLAR...', file=STAT_FILE, flush=True)
        start = time.time()
    solar_plant_data = pd.read_csv(args.input_power_plant)
    solar_station_listing, solar_weather_data= find_solar_stations(solar_plant_data,args.input_uscrn)
    if VERBOSE:
        duration = time.time() - start
        print('find_solar_stations total time: {:.2f} seconds'.format(duration), file=STAT_FILE, flush=True)
    if VERBOSE:
        print('Begining weight_solar_data...', file=STAT_FILE, flush=True)
        start = time.time()
    weight_solar_df = weight_solar_data(solar_station_listing, solar_weather_data)
    weight_solar_df.to_csv(args.output_solar_file, index=False)
    if VERBOSE:
        duration = time.time() - start
        print('weight_solar_data total time: {:.2f} seconds'.format(duration), file=STAT_FILE, flush=True)

