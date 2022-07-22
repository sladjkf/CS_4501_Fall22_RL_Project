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

# output currently blank.. good

#%% combine locality list 

# assume that no county and no major city share a name
# NOTE: this assumption is incorrect. there is a Fairfax city and a Fairfax County
#zip_county_df['locality'] = [' '.join([word for word in locality.split() if word.lower() not in ['county','city']]) for locality in localities_list]

# let's use 'name_county' and 'name_city' format instead.
#zip_county_df['locality'] = ['_'.join([word.lower() for word in locality.split()]) for locality in localities_list]

locality_list_cleaned = []
for locality in localities_list:
    if 'county' in locality.lower():
        loc = '_'.join([word.lower() for word in locality.split()])
    else:
        loc = '_'.join([word.lower() for word in locality.split()]+['city'])
    locality_list_cleaned.append(loc)
zip_county_df['locality'] = locality_list_cleaned
    

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
    return gravity(population,distances,np.array(population['pop']),params,variant='orig')

# test
grav_mod(np.array(dist_mat),vacc_df,1,1,1,0.0015)

# TODO: aggregate data up to county level and compare to actual flows
# Already one obvious limitation - the gravity formulation is symmetric in terms of actual flows while real-life need not be.

#%% aggregate zipcode level flows up to county level

# easier to do in long format?

zip_county = zip_county_df[['locality']]
zip_county.index = zip_county_df['zipcode']
zip_county_dict = {row[0]:row[1] for row in zip_county.itertuples()}

def agg_flow(grav_output,zips,zip_county,theta):
    '''
    grav_output: output from gravity model code
    zips: the zipcode labeling vector for the grav_output
    zip_county_df: dictionary containing mapping from zipcode -> 'locality'
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
    # convert to numpy data structures

    grav_matrix_list_np = np.array(grav_matrix_list)
    renamed = []
    #for row in grav_matrix_list.itertuples():
    for i in range(grav_matrix_list_np.shape[0]):
        row = grav_matrix_list_np[i,:]
        county_from = zip_county[row[0]]
        county_to = zip_county[row[1]]
        #print(county_from,county_to)
        renamed.append((county_from,county_to,row[2]))

    renamed = pd.DataFrame(renamed)
    renamed.columns = ['from','to','flow']

    renamed = renamed.groupby(['from','to']).sum()
    renamed['flow'] = renamed['flow']*theta

    return renamed

#%% get predictions and actual data in similar forms

va_commuter = va_commuter[['county_name1','county_name2','flow','error']]
#va_commuter['county_name1'] = [' '.join([word for word in locality.split() if word.lower() not in ['county','city']]) for locality in va_commuter['county_name1']]
#va_commuter['county_name2'] = [' '.join([word for word in locality.split() if word.lower() not in ['county','city']]) for locality in va_commuter['county_name2']]
va_commuter['county_name1'] = ['_'.join([word.lower() for word in locality.split()]) for locality in va_commuter['county_name1']]
va_commuter['county_name2'] = ['_'.join([word.lower() for word in locality.split()]) for locality in va_commuter['county_name2']]
va_commuter.columns = ['from','to','flow','error']


#%% are there localities in commuter network not in the gravity prediction (yes?)

va_commuter_localities = set(list(va_commuter['from'])+list(va_commuter['to']))
agg_flow_localities = set(zip_county['locality'])
print(va_commuter_localities - agg_flow_localities)
# Output: {'lexington_city', 'colonial_heights_city', 'radford_city', 'covington_city', 'sussex_county', 'portsmouth_city', 'hopewell_city'}
# what to do?
#  1. omit
#  2. get location data from nominatim (and reincporate)
#%%

# get the flow vector?
def compare(agg_flow_output,va_commuter,return_not_found=False):
    # use numpy arrays for speed
    va_commuter_mat = np.array(va_commuter)
    loc_to = []
    loc_from = []
    obs = va_commuter_mat[:,2]
    pred= np.zeros(len(obs))
    not_found = []
    for i in range(va_commuter_mat.shape[0]):
        from_county = va_commuter_mat[i,0]
        to_county = va_commuter_mat[i,1]
        try:
            flow = agg_flow_output.loc[(from_county,to_county)].item()
            pred[i] = flow
            loc_from.append(from_county)
            loc_to.append(to_county)
        except KeyError:
            print((from_county,to_county),"not found")
            not_found.append((from_county,to_county))

    if return_not_found:
        return pd.DataFrame({'from':loc_from,'to':loc_to,'obs':obs,'pred':pred}),not_found
    else:
        return pd.DataFrame({'from':loc_from,'to':loc_to,'obs':obs,'pred':pred})

#%% test (omit missing locations for now)
to_filter = ['lexington_city', 'colonial_heights_city', 'radford_city', 'covington_city', 'sussex_county', 'portsmouth_city', 'hopewell_city']
va_commuter_filtrd = va_commuter[np.logical_and(~va_commuter['from'].isin(to_filter),~va_commuter['to'].isin(to_filter))]

#%%
zip_grav_output = grav_mod(np.array(dist_mat),vacc_df,params[0],params[1],params[2],params[3])
agg_flow_output = agg_flow(zip_grav_output,zip_county.index,zip_county_dict,params[3])

to_compare = compare(agg_flow_output,va_commuter_filtrd)
plt.scatter(data=to_compare, y='pred', x='obs',s=7,alpha=0.25)
plt.axline((0,0),(1,1))


#%% aggregate into outflow and see which zips are the most connected according to gravity
outflow = pd.DataFrame({'zip':dist_mat.index,'outflow':np.sum(zip_grav_output['matrix'],axis=0)})
outflow = outflow.sort_values(by='outflow',ascending=False).reset_index()

#%% for illustration purposes, get zip_grav_output in long format

zip_grav_long = pd.DataFrame(zip_grav_output['matrix'])
zip_grav_long.columns = zip_county_df['zipcode']
zip_grav_long['zc_from'] = zip_county_df['zipcode']
zip_grav_long = zip_grav_long.melt(id_vars='zc_from')
zip_grav_long.columns = ['zc_from','zc_to','flow']
zip_grav_long['flow'] = zip_grav_long['flow']*3.5e-6

zip_grav_long

#%% calibration test?
from sklearn.metrics import mean_squared_error as mse
from scipy.optimize import minimize,differential_evolution
from scipy.stats import pearsonr

#def objective(tau1,tau2,rho,theta):
def objective1(x): # mse
    tau1, tau2, rho, theta = x
    zip_grav_output = grav_mod(np.array(dist_mat),vacc_df,tau1,tau2,rho,theta)
    agg_flow_output = agg_flow(zip_grav_output,zip_county.index,zip_county_dict,theta)
    to_compare = compare(agg_flow_output,va_commuter_filtrd)
    to_return = mse(to_compare['obs'],to_compare['pred'])
    print(x,to_return)
    return to_return

def objective1_alt(x): # mse rescaled
    tau1, tau2, rho, theta = np.exp(x)
    zip_grav_output = grav_mod(np.array(dist_mat),vacc_df,tau1,tau2,rho,theta)
    agg_flow_output = agg_flow(zip_grav_output,zip_county.index,zip_county_dict,theta)
    to_compare = compare(agg_flow_output,va_commuter_filtrd)
    to_return = mse(to_compare['obs'],to_compare['pred'])
    print(x,to_return)
    return to_return

def objective1_ll(x): #likelihood based estimation/poisson regression



def objective2(x): # correlation
    tau1, tau2, rho, theta = x
    zip_grav_output = grav_mod(np.array(dist_mat),vacc_df,tau1,tau2,rho,theta)
    agg_flow_output = agg_flow(zip_grav_output,zip_county.index,zip_county_dict,theta)
    to_compare = compare(agg_flow_output,va_commuter_filtrd)
    to_return = pearsonr(to_compare['obs'],to_compare['pred'])
    print(x,to_return)
    return to_return[0]

def plot_obj(x):
    tau1, tau2, rho, theta = x
    zip_grav_output = grav_mod(np.array(dist_mat),vacc_df,tau1,tau2,rho,theta)
    agg_flow_output = agg_flow(zip_grav_output,zip_county.index,zip_county_dict,theta)
    to_compare = compare(agg_flow_output,va_commuter_filtrd)
    fig,ax = plt.subplots()
    ax.scatter(data=to_compare, y='pred', x='obs',s=7,alpha=0.25)
    ax.axline((0,0),(1,1))
    return (fig,ax)

minimize(objective1,
        np.array([1,1,1,3.5e-6]),
        bounds=[
            (1e-2,1.5),
            (1e-2,1.5),
            (1e-2,1.5),
            (0,1e-2)]
        )


# x: array([  0.79516011,  -0.22805439,   0.44530395, -21.70887529])
minimize(objective1_alt,
        np.log(np.array([1,1,1,3.5e-6])),
        bounds=[
            (None,None),
            (None,None),
            (None,None),
            (None,None)]
        )
# x: array([  0.79516011,  -0.22805439,   0.44530395, -21.70887529])
minimize(objective1_alt,
        np.log(params),
        bounds=[
            (None,None),
            (None,None),
            (None,None),
            (None,None)]
        )

minimize(objective1_alt,
        Out[147]['x'],
        bounds=[
            (None,None),
            (None,None),
            (None,None),
            (None,None)]
        )




# [1.02360428 0.32901247 1.61855813 0.00999999]
# fitting w/ flow < 30000
minimize(objective1,
        np.array([0.99,0.339,1.5,0.00866901]),
        bounds=[
            (1e-2,1.5),
            (1e-2,1.5),
            (1e-2,1.7),
            (0,1e-2)]
        )

# fitting w/ flow < 7000
#[0.01       1.28475751 1.52038396 0.00999999]

# fitting w/ flow < 3000a
# array([0.18458969, 1.0641397 , 1.43245603, 0.01      ])
minimize(objective1,
        Out[43]['x'],
        bounds=[
            (1e-3,1.5),
            (1e-3,1.5),
            (1e-3,1.7),
            (0,1e-2)]
        )




differential_evolution(objective,
        bounds=[
            (0.5,1.5),
            (0.5,1.5),
            (0.5,1.5),
            (0,1e-5)
            ])

