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

#%% load and clean commuter data (might take a second)

commuter = pd.read_excel('data/va_zipcodes_raw/commuting_flows_county_2015.xlsx',skiprows=6)
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
    locality = locality.strip("County").strip("City")
    if not any([locality in x for x in locality_names]): # if a match was not found, print it.
        print(locality)

# output
#runcell('check if each locality in the list is actually in the flow data', 'D:/Summer 2022 (C4GC with BII)/measles_metapop/scripts/VA/gravity_calibration.py')
#Wayne...
# So 'Wayne' is a locality not in the flow data (we'll probably need to drop it)

#%% combine locality list and remove 'Wayne'

# assume that no county and no major city share a name
# god I hope
zip_county_df['locality'] = [locality.strip("County").strip("City") for locality in localities_list]
zip_county_df = zip_county_df[~zip_county_df['locality'].str.contains('Wayne')]

#%%
# TODO: now filter and align pop df + dist_mat to match
vacc_df = vacc_df[vacc_df['zipcode'].isin(zip_county_df['zipcode'])]
dist_mat = dist_mat.loc[zip_county_df['zipcode'],zip_county_df['zipcode']]

assert all(vacc_df['zipcode'] == dist_mat.index)
assert all(vacc_df['zipcode'] == dist_mat.columns)

#%% modify gravity module to use formulation not requiring I vector

def grav_mod(distances,population,tau1,tau2,rho,theta):
    params = {"tau1":tau1, "tau2":tau2, "rho":rho, "theta":theta}
    return gravity(population,distances,np.array(population['pop']),params)

grav_mod(np.array(dist_mat),vacc_df,1,1,1,0.0015)

# TODO: aggregate data up to county level and compare to actual flows
# Already one obvious limitation - the gravity formulation is symmetric in terms of actual flows while real-life need not be..