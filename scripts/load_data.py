import pandas as pd
import altair as alt
import numpy as np
from altair import datum
import geopandas as gpd
import json
import requests
import pytz
from shapely.ops import nearest_points
from sklearn.neighbors import BallTree
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
import sys
import time
import fsspec
import ftplib
from ftp_walk import FTPWalk
import io


UPDATE_INTERCONNECT = False 
UPDATE_WEATHER = False
UPDATE_WIND = False
UPDATE_SOLAR = False
UPDATE_FINAL = False
DEBUG_SUNLIGHT = False
BUILD_USCRN = False
USCRN_FTP_DEBUG = True

VERBOSE = True

WEATHER_VARS = [
    'TMAX',
    'TMIN',
    'PRCP',
    'SNOW',
    'AWND'
]

STAT_FILE = sys.stdout

def get_interconnect():
    """
    Function to import interconnect data from EIA and save to file
    
    Returns: 0
    """

    interconnects_long = [
        'Alcoa Power Generating, Inc. - Yadkin Division',
        'Arizona Public Service Company',
        'Arlington Valley, LLC',
        'Associated Electric Cooperative, Inc.',
        'Avangrid Renewables, LLC',
        'Avista Corporation',
        'Balancing Authority of Northern California',
        'Bonneville Power Administration',
        'California Independent System Operator',
        'City of Homestead',
        'City of Tacoma, Department of Public Utilities, Light Division',
        'City of Tallahassee',
        'Duke Energy Carolinas',
        'Duke Energy Florida, Inc.',
        'Duke Energy Progress East',
        'Duke Energy Progress West',
        'El Paso Electric Company',
        'Electric Energy, Inc.',
        'Electric Reliability Council of Texas, Inc.',
        'Florida Municipal Power Pool',
        'Florida Power & Light Co.',
        'Gainesville Regional Utilities',
        'GridLiance',
        'Gridforce Energy Management, LLC',
        'Griffith Energy, LLC',
        'ISO New England',
        'Idaho Power Company',
        'Imperial Irrigation District',
        'JEA',
        'Los Angeles Department of Water and Power',
        'Louisville Gas and Electric Company and Kentucky Utilities Company',
        'Midcontinent Independent System Operator, Inc.',
        'NaturEner Power Watch, LLC',
        'NaturEner Wind Watch, LLC',
        'Nevada Power Company',
        'New Harquahala Generating Company, LLC',
        'New York Independent System Operator',
        'NorthWestern Corporation',
        'Ohio Valley Electric Corporation',
        'PJM Interconnection, LLC',
        'PUD No. 1 of Douglas County',
        'PacifiCorp East',
        'PacifiCorp West',
        'Portland General Electric Company',
        'PowerSouth Energy Cooperative',
        'Public Service Company of Colorado',
        'Public Service Company of New Mexico',
        'Public Utility District No. 1 of Chelan County',
        'Public Utility District No. 2 of Grant County, Washington',
        'Puget Sound Energy, Inc.',
        'Salt River Project Agricultural Improvement and Power District',
        'Seattle City Light',
        'Seminole Electric Cooperative',
        'South Carolina Electric & Gas Company',
        'South Carolina Public Service Authority',
        'Southeastern Power Administration',
        'Southern Company Services, Inc. - Trans',
        'Southwest Power Pool',
        'Southwestern Power Administration',
        'Tampa Electric Company',
        'Tennessee Valley Authority',
        'Tucson Electric Power',
        'Turlock Irrigation District',
        'Utilities Commission of New Smyrna Beach',
        'Western Area Power Administration - Desert Southwest Region',
        'Western Area Power Administration - Rocky Mountain Region',
        'Western Area Power Administration - Upper Great Plains West'
    ]

    interconnects_short = [
        'YAD',
        'AZPS',
        'DEAA',
        'AECI',
        'AVRN',
        'AVA',
        'BANC',
        'BPAT',
        'CISO',
        'HST',
        'TPWR',
        'TAL',
        'DUK',
        'FPC',
        'CPLE',
        'CPLW',
        'EPE',
        'EEI',
        'ERCO',
        'FMPP',
        'FPL',
        'GVL',
        'GLHB',
        'GRID',
        'GRIF',
        'ISNE',
        'IPCO',
        'IID',
        'JEA',
        'LDWP',
        'LGEE',
        'MISO',
        'GWA',
        'WWA',
        'NEVP',
        'HGMA',
        'NYIS',
        'NWMT',
        'OVEC',
        'PJM',
        'DOPD',
        'PACE',
        'PACW',
        'PGE',
        'AEC',
        'PSCO',
        'PNM',
        'CHPD',
        'GCPD',
        'PSEI',
        'SRP',
        'SCL',
        'SEC',
        'SCEG',
        'SC',
        'SEPA',
        'SOCO',
        'SWPP',
        'SPA',
        'TEC',
        'TVA',
        'TEPC',
        'TIDC',
        'NSB',
        'WALC',
        'WACM',
        'WAUW'
    ]

    data_types = ['Solar', 'Wind']

    data_names = ['SUN','WND']

    search_list = []

    for i in range(len(interconnects_long)):
        for j in range(len(data_types)):
            search_list.append([
                interconnects_long[i],
                interconnects_short[i],
                data_types[j],
                'EBA.{}-ALL.NG.{}.HL'.format(interconnects_short[i],data_names[j])
            ])

    #print(search_list)

    return_df = pd.DataFrame(columns = ['interconnect_long', 'interconnect_short', 'data_type','datetime','generation'])

    for inter_data in search_list:
        eia_url = (
            "http://api.eia.gov/series/"
            "?api_key=c99361b620bca82f26eac2b287432744"
            "&series_id={}"
        ).format(inter_data[3])
        #print(doe_url)
        response = requests.get(eia_url)
        response_df = response.json()
        try:
            response_df = response_df['series'][0]['data']
            #print(response_df[0])
            l_df = pd.DataFrame(response_df, 
                columns=['datetime','generation']
            )
            l_df["datetime"] = pd.to_datetime(l_df["datetime"])
            l_df['interconnect_long'] = inter_data[0]
            l_df['interconnect_short'] = inter_data[1]
            l_df['data_type'] = inter_data[2]
            return_df = return_df.append(l_df)
            #return_df
        except Exception as e:
            print('No data for ',inter_data[3])
            print(e)

    return_df.to_csv('./data/eia/esod_data.csv', index=False)
    return 0


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

    #return_df = pd.DataFrame(columns=header_list)

    #np_data = np.frombuffer(l_read,dts)
    l_station = pd.read_fwf(io.StringIO(l_read), header=None)
    l_station.columns = header_list
    #l_station.columns = header_list
    #return_df = pd.concat([return_df,l_station])
    
    
    
    return l_station

def uscrn_ftp():
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

    years = ['2018','2019','2020','2021']

    with ftplib.FTP("ftp.ncei.noaa.gov") as ftp:
        ftp.login()
        ftp.cwd('/pub/data/uscrn/products/daily01')
        #print()

        ftpwalk = FTPWalk(ftp)
        
        for i in ftpwalk.walk(path='/pub/data/uscrn/products/daily01'):
            print(i[0])
            if any(year in i[0] for year in years) and (not 'updates' in i[0]):
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
        
    return_df.to_csv('./data/noaa/uscrn_combined.csv', index=False)
    
    return 0

def get_power_plant_data():
    """
    Load power plant shapefile and balancing authority
    shapefile

    Returns: ID, Lat, Long, Interconnect, Type, Capacity 
    """
    plant_path = "simplecache::https://www.eia.gov/maps/map_data/PowerPlants_US_EIA.zip"
    control_path = "simplecache::https://opendata.arcgis.com/datasets/02602aecc68d4e0a90bf65e818155f60_0.zip"
    
    with fsspec.open(plant_path) as file:
        plant_shapefile = gpd.read_file(file)
    
    plant_shapefile = plant_shapefile[
        (plant_shapefile['Solar_MW']>0) | 
        (plant_shapefile['Wind_MW']>0)
    ]

    with fsspec.open(control_path) as file:
        control_shapefile = gpd.read_file(file).to_crs("EPSG:4326")

    combined_shapefile = gpd.sjoin(plant_shapefile, control_shapefile, how="inner", op='intersects')
    
    authority_df = pd.read_csv('./data/eia/balancing_authority_reference.csv')

    combined_shapefile = combined_shapefile.merge(
        authority_df,
        left_on='NAME',
        right_on='hlifd_name'
    )

    combined_shapefile = combined_shapefile[[
        'Plant_Code', 
        'Latitude', 
        'Longitude',
        'Solar_MW', 
        'Wind_MW',
        'eia_short_name'
    ]]

    wind_shapefile = combined_shapefile[combined_shapefile['Wind_MW']>0]
    wind_shapefile = wind_shapefile.rename(columns={'Wind_MW':'Gen_MW'})
    wind_shapefile = wind_shapefile[[
        'Plant_Code', 
        'Latitude', 
        'Longitude',
        'Gen_MW', 
        'eia_short_name'
    ]]

    solar_shapefile = combined_shapefile[combined_shapefile['Solar_MW']>0]
    solar_shapefile = solar_shapefile.rename(columns={'Solar_MW':'Gen_MW'})
    solar_shapefile = solar_shapefile[[
        'Plant_Code', 
        'Latitude', 
        'Longitude',
        'Gen_MW', 
        'eia_short_name'
    ]]


    return wind_shapefile, solar_shapefile

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

    station_list.to_csv('./data/noaa/tmp_station_list.csv')

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

def find_solar_stations(solar_plant_data):
    """
    Returns stations for solar power plants
    using data from 
    ftp://ftp.ncei.noaa.gov/pub/data/uscrn/products/daily01
    More details on the dataset available from
    https://www.ncdc.noaa.gov/crn/qcdatasets.html
    """

    weather_data = pd.read_csv('./data/noaa/uscrn_combined.csv')

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
    weather_calc_df.to_csv('./data/noaa/wind_weighted_power.csv', index=False)

    return 0

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
    weather_calc_df.to_csv('./data/noaa/solar_weighted_power.csv', index=False)

    return 0

def join_datasets():
    """
    Joins interconnect data with weather data
    by interconnect and outputs to file

    Returns: 0
    """

    return 0

if __name__ == '__main__':
    
    
    if UPDATE_INTERCONNECT:
        if VERBOSE:
            print('Begining UPDATE_INTERCONNECT...', file=STAT_FILE, flush=True)
            start = time.time()
        get_interconnect()
        if VERBOSE:
            duration = time.time() - start
            print('UPDATE_INTERCONNECT total time: {:.2f} seconds'.format(duration), file=STAT_FILE, flush=True)

    if BUILD_USCRN:
        #add step to download USCRN FTP
        if VERBOSE:
            print('Begining BUILD_USCRN...', file=STAT_FILE, flush=True)
            start = time.time()
        combine_uscrn()
        if VERBOSE:
            duration = time.time() - start
            print('BUILD_USCRN total time: {:.2f} seconds'.format(duration), file=STAT_FILE, flush=True)

    
    if UPDATE_WEATHER:
        # Get powerplant data
        #add step to get shapefiles
        if VERBOSE:
            print('Begining UPDATE_WEATHER...', file=STAT_FILE, flush=True)
            start = time.time()
        wind_plant_data, solar_plant_data = get_power_plant_data()
        if VERBOSE:
            duration = time.time() - start
            print('get_power_plant_data total time: {:.2f} seconds'.format(duration), file=STAT_FILE, flush=True)

        # Wind data calculation
        if UPDATE_WIND:
            if VERBOSE:
                print('Begining UPDATE_WIND...', file=STAT_FILE, flush=True)
                start = time.time()
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
            weight_wind_data(wind_station_listing,wind_weather_data)
            if VERBOSE:
                duration = time.time() - start
                print('weight_wind_data total time: {:.2f} seconds'.format(duration), file=STAT_FILE, flush=True)


        if UPDATE_SOLAR:
            if VERBOSE:
                print('Begining UPDATE_SOLAR...', file=STAT_FILE, flush=True)
                start = time.time()
            solar_station_listing, solar_weather_data= find_solar_stations(solar_plant_data)
            if VERBOSE:
                duration = time.time() - start
                print('find_solar_stations total time: {:.2f} seconds'.format(duration), file=STAT_FILE, flush=True)
            if VERBOSE:
                print('Begining weight_solar_data...', file=STAT_FILE, flush=True)
                start = time.time()
            weight_solar_data(solar_station_listing, solar_weather_data)
            if VERBOSE:
                duration = time.time() - start
                print('weight_solar_data total time: {:.2f} seconds'.format(duration), file=STAT_FILE, flush=True)

        #Solar data collection

    if UPDATE_FINAL:
        if VERBOSE:
            print('Begining UPDATE_FINAL...', file=STAT_FILE, flush=True)
            start = time.time()
        join_datasets()
        if VERBOSE:
            duration = time.time() - start
            print('join_datasets total time: {:.2f} seconds'.format(duration), file=STAT_FILE, flush=True)


    if DEBUG_SUNLIGHT:
        df = pd.DataFrame({'dates':['2017-01-01 03:15:00','2017-01-01 03:15:00'], 'latitude':[50.0,40.0]})
        df['dates'] = pd.to_datetime(df['dates'])

        df['jul1'] = pd.DatetimeIndex(df['dates']).to_julian_date()
        #if need remove times
        df['jul2'] = pd.DatetimeIndex(df['dates']).floor('d').to_julian_date()
        #print(df)
        print(sunlight_calculation(df['jul1'].to_numpy(), df['latitude'].to_numpy()))

    if USCRN_FTP_DEBUG:
        uscrn_ftp()