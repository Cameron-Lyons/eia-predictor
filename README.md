## Overview

Pipeline setup to take data from NOAA and EIA for generation of a machine learning model.


## Setup

Process can be run through generation of final dataframes for machine learning by running the following command:

  - `dvc repro`

## Data Sources

Actual generation data by balancing authority from [EIA](https://www.eia.gov/opendata/).

Weather data for use with solar power taken from [USCRN](https://www.ncdc.noaa.gov/crn/qcdatasets.html).

Weather data for us with wind power taken from [NCEI](https://www.ncei.noaa.gov/support/access-data-service-api-user-documentation).

Listing of power plants in the United States from [EIA](https://www.eia.gov/maps/map_data/PowerPlants_US_EIA.zip).

Mapping of balancing authorities from [HIFLD](https://opendata.arcgis.com/datasets/02602aecc68d4e0a90bf65e818155f60_0.zip)

Mapping of the balacing authority names between EIA and HIFLD provided in data folder.

