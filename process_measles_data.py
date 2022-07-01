#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:29:17 2022

@author: nick
"""

#%%

import pandas as pd
import matplotlib.pyplot as plt

#%%
base_path ="/run/media/nicholasw/9D47-AA7C/Summer 2022 (C4GC with BII)/measles_metapop/"


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