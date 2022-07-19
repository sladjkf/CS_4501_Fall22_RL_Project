# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 16:35:01 2022

@author: nrw5cq
"""

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# windows machine at the office
project_path = "D:/Summer 2022 (C4GC with BII)/measles_metapop/{}"
# home desktop
project_path = "/run/media/nick/9D47-AA7C/Summer 2022 (C4GC with BII)/measles_metapop/{}"

#%% load and clean commuter data (might take a second)

commuter = pd.read_excel(
        project_path.format('data/va_zipcodes_raw/commuting_flows_county_2015.xlsx'),
        skiprows=6)
commuter.columns=['state_code1','county_code1','state_name1','county_name1',
                  'state_code2','county_code2','state_name2','county_name2',
                  'flow','error']
va_commuter = commuter[np.logical_and(commuter['state_name1'] == "Virginia",commuter['state_name2']=='Virginia')]\
    .reset_index(drop=True)

#%% load data and modules needed for gravity module
import sys
sys.path.append(project_path.format("scripts/"))

from spatial_tsir import *

# vaccination data and population data from sifat
vacc_df = pd.read_csv(project_path.format("data/VA_zipcodes_cleaned/ZC_immunization_sifat.csv"))
# drop 0 population entries, they won't affect the simulation
vacc_df = vacc_df[vacc_df['population'] > 0].reset_index(drop=True)
vacc_df.rename({'population':'pop'},axis=1,inplace=True)

# load distance matrix computed from nominatim and geopy distance function
dist_df = pd.read_csv(project_path.format("data/VA_zipcodes_cleaned/ZC_distance_sifat_nom_geopy.csv"))
# need to replace 0's in distance matrix to avoid divide by zero in gravity formula
default_dist = 0.5
dist_df.loc[dist_df[np.isclose(dist_df['distKM'],0)].index,'distKM']=default_dist
# convert to matrix
dist_mat = dist_df.pivot(index='zipcode1',columns='zipcode2',values='distKM')
dist_mat = dist_mat.replace(np.nan,0)

# align matrix
dist_mat = dist_mat.loc[vacc_df['zipcode'],vacc_df['zipcode']]

#%% load data needed for aggregation methods

zip_county_df = pd.read_csv(project_path.format("data/VA_zipcodes_cleaned/VA_zips_latlong_nominatim.csv"))
zip_county_df['display_name']

# try to get county or locality name
# basic logic: search for one that has county, otherwise maybe try taking the first index?
localities = [x.split(',') for x in zip_county_df['display_name']]

localities_list = []
for name_list in localities:
    check_county = ['County' in name for name in name_list]
    if any(check_county):
        localities_list.append(pd.Series(name_list)[check_county].item().strip())
    else:
        localities_list.append(name_list[0].strip())

#%% check if each locality in the list is actually in the flow data

locality_names = set(list(va_commuter['county_name1']) + list(va_commuter['county_name2']))

for locality in localities_list:
    # some locations won't match ('City' vs 'city'...)
    locality = ' '.join([word for word in locality.split() if word.lower() not in ['county','city']])
    if not any([locality in x for x in locality_names]): # if a match was not found, print it.
        print(locality)

# output
#runcell('check if each locality in the list is actually in the flow data', 'D:/Summer 2022 (C4GC with BII)/measles_metapop/scripts/VA/gravity_calibration.py')
#Wayne...
# So 'Wayne' is a locality not in the flow data (we'll probably need to drop it)

#%% combine locality list and remove 'Wayne'

# assume that no county and no major city share a name
# god I hope
#zip_county_df['locality'] = [locality.strip("County").strip("City") for locality in localities_list]
zip_county_df['locality'] = [' '.join([word for word in locality.split() if word.lower() not in ['county','city']]) for locality in localities_list]

#zip_county_df = zip_county_df[~zip_county_df['locality'].str.contains('Wayne')]

#%%
# TODO: now filter and align pop df + dist_mat to match
vacc_df = vacc_df[vacc_df['zipcode'].isin(zip_county_df['zipcode'])]
dist_mat = dist_mat.loc[zip_county_df['zipcode'],zip_county_df['zipcode']]

assert all(vacc_df['zipcode'] == dist_mat.index)
assert all(vacc_df['zipcode'] == dist_mat.columns)

#%% modify gravity module to use formulation not requiring I vector

def grav_mod(distances,population,tau1,tau2,rho,theta):
    params = {"tau1":tau1, "tau2":tau2, "rho":rho, "theta":theta}
    # if I is replaced by N, it should be equivalent.
    return gravity(population,distances,np.array(population['pop']),params)

# test
grav_mod(np.array(dist_mat),vacc_df,1,1,1,0.0015)

# TODO: aggregate data up to county level and compare to actual flows
# Already one obvious limitation - the gravity formulation is symmetric in terms of actual flows while real-life need not be.

#%% aggregate zipcode level flows up to county level

# easier to do in long format?

zip_county = zip_county_df[['locality']]
zip_county.index = zip_county_df['zipcode']

def agg_flow(grav_output,zips,zip_county,theta):
    '''
    grav_output: output from gravity model code
    zips: the zipcode labeling vector for the grav_output
    zip_county_df: dataframe containing mapping from zipcode -> 'locality'
    theta: constant to scale flows by (should be same theta as in original model?)
    '''
    grav_matrix = grav_output['matrix']
    grav_influx = grav_output['influx']

    # convert to dataframe to go to long form
    grav_matrix = pd.DataFrame(grav_matrix)
    grav_matrix.index = zips
    grav_matrix.columns = zips
    grav_matrix['from_zc'] = grav_matrix.index

    grav_matrix_list = grav_matrix.melt(id_vars='from_zc')
    grav_matrix_list.columns = ['from_zc','to_zc','flow']

    # definitely going to need to be faster if to be used for calibration
    renamed = []
    for row in grav_matrix_list.itertuples():
        county_from = zip_county.loc[row.from_zc,'locality']
        county_to = zip_county.loc[row.to_zc,'locality']
        renamed.append((county_from,county_to,row.flow))

    renamed = pd.DataFrame(renamed)
    renamed.columns = ['from','to','flow']

    renamed = renamed.groupby(['from','to']).sum()
    renamed['flow'] = renamed['flow']*theta

    return renamed

#%% get predictions and actual data in similar forms

va_commuter = va_commuter[['county_name1','county_name2','flow','error']]
va_commuter['county_name1'] = [' '.join([word for word in locality.split() if word.lower() not in ['county','city']]) for locality in va_commuter['county_name1']]
va_commuter['county_name2'] = [' '.join([word for word in locality.split() if word.lower() not in ['county','city']]) for locality in va_commuter['county_name2']]
va_commuter.columns = ['from','to','flow','error']


#%% are there localities in commuter network not in the gravity prediction (yes?)

va_commuter_localities = set(list(va_commuter['from'])+list(va_commuter['to']))
agg_flow_localities = set(zip_county['locality'])
print(va_commuter_localities - agg_flow_localities)
# OUTPUT: {'Colonial Heights', 'Covington', 'Hopewell', 'Sussex', 'Lexington'}
# what to do?
#  1. omit
#  2. get location data from nominatim (and reincporate)
#%%

# get the flow vector?
def compare(agg_flow_output,va_commuter,return_not_found=False):
    # use numpy arrays for speed
    va_commuter_mat = np.array(va_commuter)
    obs = va_commuter_mat[:,2]
    pred= np.zeros(len(obs))
    not_found = []
    for i in range(va_commuter_mat.shape[0]):
        from_county = va_commuter_mat[i,0]
        to_county = va_commuter_mat[i,1]
        try:
            flow = agg_flow_output.loc[(from_county,to_county)].item()
            pred[i] = flow
        except KeyError:
            print((from_county,to_county),"not found")
            not_found.append((from_county,to_county))

    if return_not_found:
        return pd.DataFrame({'obs':obs,'pred':pred}),not_found
    else:
        return pd.DataFrame({'obs':obs,'pred':pred})

#%% test (omit missing locations for now)
to_filter = ['Colonial Heights', 'Covington', 'Hopewell', 'Sussex', 'Lexington']
va_commuter_filtrd = va_commuter[np.logical_and(~va_commuter['from'].isin(to_filter),~va_commuter['to'].isin(to_filter))]

zip_grav_output = grav_mod(np.array(dist_mat),vacc_df,1,1,1,1.5e-5)
agg_flow_output = agg_flow(zip_grav_output,zip_county.index,zip_county,1.5e-5)
to_compare = compare(agg_flow_output,va_commuter_filtrd)

#%%
plt.figure()
plt.scatter(data=to_compare, x='pred', y='obs',s=7,alpha=0.25)
plt.axline((0,0),(1,1))
plt.show()
