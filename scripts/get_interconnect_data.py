import pandas as pd
import json
import requests
import sys
import time

VERBOSE = False
STAT_FILE = sys.stdout


def get_interconnect(out_csv):
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

    i=0
    max_i = len(search_list)
    for inter_data in search_list:
        if VERBOSE:
            i += 1
            print('Querying interconnect #'+str(i)+'/'+str(max_i), file=STAT_FILE, flush=True)
      
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
            if VERBOSE:
                print('No data for ',inter_data[3],e)

    return_df.to_csv(out_csv, index=False)
    return 0

if __name__ == '__main__':
    import argparse

    #'./data/eia/esod_data.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'output_file', help='interconnect hourly power generation (CSV)')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='display detailed output',
    )
    args = parser.parse_args()

    if args.verbose:
        VERBOSE = True

    if VERBOSE:
            print('Begining get_interconnect...', file=STAT_FILE, flush=True)
            start = time.time()
    get_interconnect(args.output_file)
    if VERBOSE:
        duration = time.time() - start
        print('get_interconnect total time: {:.2f} seconds'.format(duration), file=STAT_FILE, flush=True)
