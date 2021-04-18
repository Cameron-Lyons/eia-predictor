import pandas as pd
import numpy as np
import sys
import time
import geopandas as gpd
import fsspec

VERBOSE = False
STAT_FILE = sys.stdout

def power_plant_data(authority_reference):
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
    
    authority_df = pd.read_csv('data/balancing_authority_reference.csv')

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

if __name__ == '__main__':
    import argparse

    #'./data/noaa/uscrn_combined.csv''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_authority_ref', help='Reference between EIA and other (CSV)')
    parser.add_argument(
        'output_solar_file', help='Solar power plant data (CSV)')
    parser.add_argument(
        'output_wind_file', help='Wind power plant data (CSV)')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='display detailed output',
    )
    args = parser.parse_args()

    if args.verbose:
        VERBOSE = True

    if VERBOSE:
            print('Begining power_plant_data...', file=STAT_FILE, flush=True)
            start = time.time()
    wind_plant_data, solar_plant_data = power_plant_data(args.input_authority_ref)
    wind_plant_data.to_csv(args.output_wind_file, index=False)
    solar_plant_data.to_csv(args.output_solar_file, index=False)
    if VERBOSE:
        duration = time.time() - start
        print('power_plant_data total time: {:.2f} seconds'.format(duration), file=STAT_FILE, flush=True)
