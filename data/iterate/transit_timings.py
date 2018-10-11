import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from scipy import stats
import googlemaps
from datetime import datetime
from pandas.io.json import json_normalize
from pprint import pprint
import json

"""
Read in data
"""

# sold data
#sold = pd.read_json('sold2.jl',lines=True)

# scrapy sale data
sale = pd.read_json('sale_allscrapes.jl',lines=True)

income = pd.read_csv('income_zip.csv')

"""
clean and format data
"""

### trulia data ###

# list of filtering parameters
clean = ['url','address','city_state','price','address','bedrooms','bathrooms','area','year_built']#,'lot_size']

# drop duplicate rows
sale = sale.drop_duplicates(clean)

# drop data with missing data in clean
sale = sale.dropna(subset = clean)

# separate out zip code from city/state
sale['Zip'] = sale.city_state.str.split().str[2]

# convert zip to integer
sale.Zip = pd.to_numeric(sale.Zip, downcast='integer', errors='coerce')

# remove any missing zip codes
sale = sale.dropna(subset = ['Zip'])

### census data ###

# convert from strings to integers and remove commas
income.Median = pd.to_numeric(income.Median.str.replace(',',''), downcast = 'integer', errors='coerce')
income.Mean = pd.to_numeric(income.Mean.str.replace(',',''), downcast = 'integer', errors='coerce')
income.Pop = pd.to_numeric(income.Pop.str.replace(',',''), downcast = 'integer', errors='coerce')

# merge in income data
sale = pd.merge(sale,income,on='Zip')

# rename columns for consistancy
sale.rename(index=str, columns={'Zip':'zipcode'}, inplace=True)
sale.rename(index=str, columns={'Median':'median_income'}, inplace=True)
sale.rename(index=str, columns={'Mean':'mean_income'}, inplace=True)
sale.rename(index=str, columns={'Pop':'population'}, inplace=True)

# add indexing column
sale['ID'] = sale.index

#resulting size
print(str(sale.shape[0]) + ' viable house sale data points')

"""
testing google distance matrix
"""

# initialize google maps client
gmaps = googlemaps.Client(key='AIzaSyBK1EC3HJQaQWVWB_x-h6ffkr-nA7lD5lE')

# ending address - Picasso Statue
end_address = '50 W Washington St'
end_city_state = 'Chicago, IL 60603'

# set travel time for arrival at 9am on Monday, 19 November 2018
arrival_time = datetime.now()
arrival_time = arrival_time.replace(minute=30, hour=8, second=0, year=2018, month=11, day=19)

#subset size
diff =84
subsets = list(range(8000,8100,diff))

for Q in subsets:
    
    """
    Create subsample for querying google api
    """
    Y = Q
    Z = Q+diff
    # create randome sample
    sample = sale.iloc[Y:Z]
    sample = sample.reset_index()

    # add rows to be filled
    sample['distance_steps'] = sample.apply(lambda x: [], axis=1)
    sample['distance_trip'] = sample.apply(lambda x: [], axis=1)
    sample['duration_steps'] = sample.apply(lambda x: [], axis=1)
    sample['duration_trip'] = sample.apply(lambda x: [], axis=1)
    sample['mode_steps'] = sample.apply(lambda x: [], axis=1)
    sample['vehicle_steps'] = sample.apply(lambda x: [], axis=1)
    sample['line_steps'] = sample.apply(lambda x: [], axis=1)
    sample['latitude'] = sample.apply(lambda x: [], axis=1)
    sample['longitude'] = sample.apply(lambda x: [], axis=1)

    """
    Parse each google directions api query and format and input into the dataframe.

    First, query the Google directions API to get transit direction details

    The API output is a series of nested 'legs' and 'steps' which may be walking or transit steps
    The nesting is unpredictable and complex.  This code loops through each 'leg' and subsequent 'step'.
    Each time it happens upon a variable of interest, it appends it to an output list.
    These output lists are then fed back into the data structure.

    Note: be careful, directions API queries are $0.01/per

    Note: this is an incredibly gross set of 8 nearly identical nested 'for' loops.
    It is, however, relatively fast, very effective at fully parsing the data, and handles errors well.

    Let she who is without sin cast the first stone.
    """

    for G in range(len(sample)):


        ### Retrieve directions data from Google directions API ###

        directions_result = gmaps.directions(origin = sample.address.iloc[G] + sale.city_state.iloc[G],
                                             destination = end_address + end_city_state,
                                             mode='transit',
                                             units='metric',
                                             transit_mode='rail',
                                             arrival_time=arrival_time)



        # initialize variables to be parsed
        distance_trip = []
        duration_trip = []
        latitude = []
        longitude = []
        distance_steps = []
        duration_steps = []
        mode_steps = []
        vehicle_steps = []
        line_steps = []

        # maximum number of sequential steps in single direction step order
        N = 20


        ### Parse API data ###
        # loop through legs
        for i in range(5):
            try:
                distance_trip.append(directions_result[0]['legs'][i]['distance']['text'])
            except:
                continue
            try:
                duration_trip.append(directions_result[0]['legs'][i]['duration']['text'])
            except:
                continue
            try:
                latitude.append(directions_result[0]['legs'][i]['start_location']['lat'])
            except:
                continue
            try:
                longitude.append(directions_result[0]['legs'][i]['start_location']['lng'])
            except:
                continue

            # loop through first order steps
            for j in range(N):
                try:
                    distance_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                          ['distance']['text'])
                except:
                    continue
                try:
                    duration_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                          ['duration']['text'])
                except:
                    continue
                try:
                    mode_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                      ['travel_mode'])
                except:
                    continue
                try:
                    vehicle_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                      ['transit_details']['line']['vehicle']['type'])
                except:
                    continue
                try:
                    line_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                      ['transit_details']['line']['name'])
                except:
                    #vehicle_steps.append('WALK')
                    continue

                # loop through second order steps
                for k in range(N):
                    try:
                        distance_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                              ['steps'][k]['distance']['text'])
                    except:
                        continue
                    try:
                        duration_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                              ['steps'][k]['duration']['text'])
                    except:
                        continue
                    try:
                        mode_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                              ['steps'][k]['travel_mode'])
                    except:
                        continue
                    try:
                        vehicle_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                              ['steps'][k]['transit_details']['line']['vehicle']['type'])
                    except:
                        #vehicle_steps.append('WALK')
                        continue
                    try:
                        line_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                              ['steps'][k]['transit_details']['line']['name'])
                    except:
                        #vehicle_steps.append('WALK')
                        continue



                    # loop through third order steps
                    for m in range(N):
                        try:
                            distance_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                  ['steps'][k]['steps'][m]['distance']['text'])
                        except:
                            continue
                        try:
                            duration_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                  ['steps'][k]['steps'][m]['duration']['text'])
                        except:
                            continue
                        try:
                            mode_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                  ['steps'][k]['steps'][m]['travel_mode'])
                        except:
                            continue
                        try:
                            vehicle_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                  ['steps'][k]['steps'][m]['transit_details']['line']['vehicle']['type'])
                        except:
                            #vehicle_steps.append('WALK')
                            continue
                        try:
                            line_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                  ['steps'][k]['steps'][m]['transit_details']['line']['name'])
                        except:
                            #vehicle_steps.append('WALK')
                            continue

                        # loop through fourth order steps
                        for n in range(N):
                            try:
                                distance_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                      ['steps'][k]['steps'][m]['steps'][n]['distance']['text'])
                            except:
                                continue
                            try:
                                duration_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                      ['steps'][k]['steps'][m]['steps'][n]['duration']['text'])
                            except:
                                continue
                            try:
                                mode_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                      ['steps'][k]['steps'][m]['steps'][n]['travel_mode'])
                            except:
                                continue
                            try:
                                vehicle_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                      ['steps'][k]['steps'][m]['steps'][n]\
                                                     ['transit_details']['line']['vehicle']['type'])
                            except:
                                #vehicle_steps.append('WALK')
                                continue
                            try:
                                line_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                      ['steps'][k]['steps'][m]['steps'][n]\
                                                     ['transit_details']['line']['name'])
                            except:
                                #vehicle_steps.append('WALK')
                                continue

                            # loop through fifth order steps
                            for o in range(N):
                                try:
                                    distance_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                          ['steps'][k]['steps'][m]['steps'][n]['steps'][o]['distance']['text'])
                                except:
                                    continue
                                try:
                                    duration_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                          ['steps'][k]['steps'][m]['steps'][n]['steps'][o]['duration']['text'])
                                except:
                                    continue
                                try:
                                    mode_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                          ['steps'][k]['steps'][m]['steps'][n]['steps'][o]['travel_mode'])
                                except:
                                    continue
                                try:
                                    vehicle_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                          ['steps'][k]['steps'][m]['steps'][n]['steps'][o]\
                                                         ['transit_details']['line']['vehicle']['type'])
                                except:
                                    #vehicle_steps.append('WALK')
                                    continue
                                try:
                                    line_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                          ['steps'][k]['steps'][m]['steps'][n]['steps'][o]\
                                                         ['transit_details']['line']['name'])
                                except:
                                    #vehicle_steps.append('WALK')
                                    continue

                                # loop through sixth order steps
                                for p in range(N):
                                    try:
                                        distance_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                              ['steps'][k]['steps'][m]['steps'][n]['steps'][o]['steps'][p]\
                                                              ['distance']['text'])
                                    except:
                                        continue
                                    try:
                                        duration_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                              ['steps'][k]['steps'][m]['steps'][n]['steps'][o]['steps'][p]\
                                                              ['duration']['text'])
                                    except:
                                        continue
                                    try:
                                        mode_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                              ['steps'][k]['steps'][m]['steps'][n]['steps'][o]['steps'][p]\
                                                              ['travel_mode'])
                                    except:
                                        continue
                                    try:
                                        vehicle_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                              ['steps'][k]['steps'][m]['steps'][n]['steps'][o]['steps'][p]\
                                                             ['transit_details']['line']['vehicle']['type'])
                                    except:
                                        #vehicle_steps.append('WALK')
                                        continue
                                    try:
                                        line_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                              ['steps'][k]['steps'][m]['steps'][n]['steps'][o]['steps'][p]\
                                                             ['transit_details']['line']['name'])
                                    except:
                                        #vehicle_steps.append('WALK')
                                        continue

                                    # loop through seventh order steps
                                    for q in range(N):
                                        try:
                                            distance_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                                  ['steps'][k]['steps'][m]['steps'][n]['steps'][o]['steps'][p]\
                                                                  ['steps'][q]['distance']['text'])
                                        except:
                                            continue
                                        try:
                                            duration_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                                  ['steps'][k]['steps'][m]['steps'][n]['steps'][o]['steps'][p]\
                                                                  ['steps'][q]['duration']['text'])
                                        except:
                                            continue
                                        try:
                                            mode_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                                  ['steps'][k]['steps'][m]['steps'][n]['steps'][o]['steps'][p]\
                                                                  ['steps'][q]['travel_mode'])
                                        except:
                                            continue
                                        try:
                                            vehicle_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                                  ['steps'][k]['steps'][m]['steps'][n]['steps'][o]['steps'][p]\
                                                                  ['steps'][q]['transit_details']['line']['vehicle']['type'])
                                        except:
                                            #vehicle_steps.append('WALK')
                                            continue
                                        try:
                                            line_steps.append(directions_result[0]['legs'][i]['steps'][j]\
                                                                  ['steps'][k]['steps'][m]['steps'][n]['steps'][o]['steps'][p]\
                                                                  ['steps'][q]['transit_details']['line']['name'])
                                        except:
                                            #vehicle_steps.append('WALK')
                                            continue

        # write parsed data variables to dataframe
        try:
            sample.latitude.iloc[G] = latitude
        except:
            continue
        try:
            sample.longitude.iloc[G] = longitude
        except:
            continue
        try:
            sample.line_steps.iloc[G] = line_steps
        except:
            continue
        try:
            sample.vehicle_steps.iloc[G] = vehicle_steps
        except:
            continue
        try:
            sample.mode_steps.iloc[G] = mode_steps
        except:
            continue
        try:
            sample.duration_trip.iloc[G] = duration_trip
        except:
            continue
        try:
            sample.duration_steps.iloc[G] = duration_steps
        except:
            continue
        try:
            sample.distance_trip.iloc[G] = distance_trip
        except:
            continue
        try:
            sample.distance_steps.iloc[G] = distance_steps
        except:
            continue
            
    # save sample
    sample.to_csv('./APIreturns/sale_{}.csv'.format(Y))