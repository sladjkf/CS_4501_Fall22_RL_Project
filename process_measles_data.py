#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:29:17 2022

@author: nick
"""

#%%

import pandas as pd
import matplotlib.pyplot as plt
import datetime as datetime

#%%
base_path ="/run/media/nick/9D47-AA7C/Summer 2022 (C4GC with BII)/measles_metapop/"


measles_data = pd.read_csv(
    "/run/media/nick/9D47-AA7C/Summer 2022 (C4GC with BII)/measles_metapop/EW_cities.txt",
    delim_whitespace=True
)

#aggregate data into biweeks?

# first make a date-time column
def get_date(row):
    return datetime.datetime(
        day = row["#DD"],
        month = row['MM'],
        year = row['YY']+1900
    )
measles_data['date'] = measles_data.apply(func=get_date,axis=1)
measles_data.drop(["#DD","MM","YY"],axis=1,inplace=True)

measles_data = measles_data.replace("*",0)

#%%

initial_date = measles_data['date'][0]

#measles_data_agg = pd.DataFrame()
#measles_data_agg.columns = ["start","end","London","Bristol","Liverpool",
                            #"Manchester","Newcastle","Birmingham","Sheffield"]
measles_data_agg = None
i = 0
#for i in range(0,6):
while initial_date < measles_data['date'].iloc[-1]:
    interval = datetime.timedelta(days=14)
    
    interval_enddate = initial_date + interval
    
    row = measles_data[measles_data['date'].between(initial_date,interval_enddate,inclusive="left")]\
        .loc[:,measles_data.columns != "date"].astype("int64").sum()
    row['start'] = initial_date
    row['end'] = interval_enddate
    
    
    row = pd.DataFrame(dict(row),index=[i])
    print(row)
    initial_date = interval_enddate
    if type(measles_data_agg) == type(None):
        measles_data_agg = row
    else:
        measles_data_agg = pd.concat([measles_data_agg,row],axis=0)
    
    i +=1
    #measles_data_agg = pd.concat([measles_data_agg,pd.Datarow],axis=1)

#%%

# plot the new time series data
measles_data_agg.loc[:,~measles_data_agg.columns.isin(['start','end'])].plot(legend=False)

# but note vaccination era began around '67? so let's filter that out.
measles_data_agg[measles_data_agg['end'] < datetime.datetime(day=1,month=1,year=1968)]\
        .loc[:,~measles_data_agg.columns.isin(['start','end'])]\
        .plot(legend=False)

#%%

# getting location data
# https://simplemaps.com/data/gb-cities
gb_latlng = pd.read_csv("/run/media/nick/9D47-AA7C/Summer 2022 (C4GC with BII)/measles_metapop/gb.csv")

measles_cities = gb_latlng[gb_latlng['city'].isin(measles_data_agg.columns)].reset_index(drop=True)

#%%

# getting distances between locations...
# maybe it's fine since they just need to be on the same scale?
# idk.
# https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula
from math import cos, asin, sqrt, pi
from scipy.special import binom
import itertools

def distance(lat1, lon1, lat2, lon2):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 12742 * asin(sqrt(a)) #2*R*asin...

#%%
distance_matrix = np.zeros([len(measles_cities.index)]*2)
matrix_name_vec = np.array(measles_cities.city)
for city1, city2 in itertools.combinations(measles_cities.loc[:,['city','lat','lng']].iterrows(),2):
    city1 = city1[1]
    city2 = city2[1]
    print(city1['city'],city2['city'])
    distance_matrix[city1.name][city2.name] = distance(city1['lat'],city1['lng'],city2['lat'],city2['lng'])
    distance_matrix[city2.name][city1.name] = distance_matrix[city1.name][city2.name] 
    print(distance_matrix[city1.name][city2.name]) 

# pretty print matrix
with np.printoptions(precision=3,suppress=True):
    print(distance_matrix)

#%%

# load the population data
pop = pd.read_csv("/run/media/nick/9D47-AA7C/Summer 2022 (C4GC with BII)/measles_metapop/EW_pop.txt",delim_whitespace=True)
pop = pop.T

#%%

# load the birth rate data
# https://www.macrotrends.net/countries/GBR/united-kingdom/birth-rate

birth_rate = pd.read_csv("/run/media/nick/9D47-AA7C/Summer 2022 (C4GC with BII)/measles_metapop/gb_birth_rate.csv",index_col=False)
birth_rate['per_cap'] = birth_rate.iloc[:,1]/1000

#%%
# could try to align birth rate data with the measles case data?

def get_birth_rates(row):
    year = row['start'].year
    rate = birth_rate[birth_rate['year']==year].per_cap
    if rate.empty:
        # assume this as birth rate?
        return 0.017
    return np.float64(rate)

measles_data_agg['birth_per_cap'] = measles_data_agg.apply(func=get_birth_rates,axis=1)
measles_data_agg['birth_per_cap'] = measles_data_agg['birth_per_cap']/26

#%%

# save cleaned measles ts data with per capita birth rate
measles_data_agg.to_csv(base_path+"EW_cases_cleaned.csv")

