"""

Load and process the Virginia zip code data from the US Census.

"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

#%%
project_path = "/run/media/nick/9D47-AA7C/Summer 2022 (C4GC with BII)/measles_metapop/{}"

# https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.2021.html
us_zipcode_geodata = gpd.read_file(project_path.format("data/tl_2021_us_zcta520/tl_2021_us_zcta520.shp"))

# I literally went to the US census data website
# and copied the list of zctas from the search box...
va_zcta_list = pd.read_csv(project_path.format("data/VA_zcta_list.csv"),header=None, delim_whitespace=True)

#%%

# filter out VA zip codes
va_zipcodes_geodata = us_zipcode_geodata[us_zipcode_geodata['GEOID20'].astype('int').isin(va_zcta_list[1])]


# plot it to check
va_zipcodes_geodata.plot()

# alternate strat: max/min? seem to be missing some

us_zipcode_geodata[us_zipcode_geodata['GEOID20'].astype('int').between(20105,24657)].plot()

#%%
# compute the distance matrix between all the VA zip codes?
# use the centroids? 


# TODO: something about map projections??
# https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.to_crs.html
centroids = va_zipcodes_geodata.to_crs(crs=3857).centroid.to_crs(crs=4326)
rep_points = va_zipcodes_geodata.representative_point()

ax1 = va_zipcodes_geodata.plot()
centroids.plot(color="red",ax=ax1,markersize=1)

ax2 = va_zipcodes_geodata.plot()
rep_points.plot(color="red",ax=ax2,markersize=1)


#%%
# get the population of all VA zctas
# 2010 data.. 
us_zcta_pop = pd.read_csv(project_path.format("data/DECENNIALSF12010.P1_2022-07-11T172722/DECENNIALSF12010.P1_data_with_overlays_2022-07-11T163259.csv"),skiprows=1)
# get zip codes
us_zcta_pop.id = pd.Series([x[9:] for x in us_zcta_pop.id])


# better population data at
# https://data.census.gov/cedsci/table?q=B01001%20SEX%20BY%20AGE&g=0400000US51%248600000&tid=ACSDT5Y2020.B01001
#  https://www.virginia-demographics.com/zip_codes_by_population
va_zcta_pop = pd.read_csv(project_path.format("data/VA_zcta_pop.csv")).dropna()
#%%

# write population counts and centroids
# or compute distances now?

# I think this gives the distance in meters.
# TODO: maybe this computations is good enough for a first start, but
# it can probably be refined I guess...
dists = rep_points.to_crs(3857)
dists = np.array([x.distance(y) for x in dists for y in dists]).reshape(len(rep_points.index),len(rep_points.index))
labels = rep_points.index

dists = pd.DataFrame(dists)
dists.index = labels
dists.columns = labels

us_zcta_pop = us_zcta_pop[['id','Total']]
us_zcta_pop.columns = ['patch_id','pop']

us_zcta_pop = us_zcta_pop[us_zcta_pop['patch_id'].isin(va_zcta_list[1].astype('str'))]

# basic validation: did it sum to the population estimate of va in 2010?
# no.
sum(us_zcta_pop['pop'])

#%%

# discrepancy between number of zipcodes in the geoindex set and from the population data from US census?
# what's going on?
# maybe use the population data 
# to select the relevant zctas from the geodata

# maybe only some are geographically distinct?
# in which case... assign the excess population to the closest centroid?
va_zcta_pop[~va_zcta_pop.patch_id.astype('int').astype('str').isin(us_zipcode_geodata['ZCTA5CE20'])]

#%%

# try a python package
from uszipcode import SearchEngine
search = SearchEngine()
va_zipcodes = va_zcta_pop['patch_id'].astype('int').astype('str')
results = [search.by_zipcode(code).to_dict() for code in va_zipcodes]
latlng = [(x["lat"],x["lng"]) for x in results]
latlng = pd.DataFrame(latlng)

#%%

# plot to check?
# looks legit..
import geopandas
gdf = geopandas.GeoDataFrame(latlng, geometry=geopandas.points_from_xy(latlng[1],latlng[0]))
gdf.plot(markersize=2)

#%%

# directly compute distances using haversine formula?
from math import cos, asin, sqrt, pi
#def distance(lat1, lon1, lat2, lon2):
def distance(loc1, loc2):
    lat1, lon1 = loc1
    lat2, lon2 = loc2
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 12742 * asin(sqrt(a)) #2*R*asin...

num_locations = len(latlng.index)
dist_matrix = np.zeros((num_locations,num_locations))
for i in range(0,num_locations):
    for j in range(0,i):
        loc_i = tuple(latlng.iloc[i,0:2])
        loc_j = tuple(latlng.iloc[j,0:2])
        dist_matrix[i][j] = distance(loc_i, loc_j)
        dist_matrix[j][i] = dist_matrix[i][j]

#%% save the haversine distance matrix

haversine_dist_df = pd.DataFrame(dist_matrix, index=list(va_zipcodes), columns=list(va_zipcodes))
haversine_dist_df.to_csv(project_path.format("data/VA_zipcodes_cleaned/VA_zipcodes_dist_haversine.csv"))

#%%

# compared to geopandas + projection method
gdf_projected = gdf.set_crs('4326').to_crs('3857')

gdf_projected = gdf.set_crs('4269').to_crs('3857')

gdf_dist_matrix = np.zeros((num_locations,num_locations))

for i in range(0,num_locations):
    for j in range(0,i):
        gdf_dist_matrix[i,j] = gdf_projected.iloc[i,2].distance(gdf_projected.iloc[j,2])
        gdf_dist_matrix[j,i] = gdf_dist_matrix[i,j] 


########## ------------------ #################3

#%% get VA zip code latlong from Nominatim

va_vax_schedule = pd.read_csv(project_path.format("data/VA_zipcodes_cleaned/ZC_immunization_sifat.csv"))

# although this df has 876 rows that correspond to the acquired zcta data from us census,
# we won't be able to use all of them so let's jsut drop 0 rows first
# they won't affect the simulation
va_vax_schedule = va_vax_schedule[va_vax_schedule['population'] > 0 ]
# it's easier if we modify the distance matrix, too
# but some entries not in there??
# may need to recompute from scratch
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
locator = Nominatim(user_agent="get_VA_zipcodes")
coder = RateLimiter(locator.geocode,min_delay_seconds=1)
queries = ["{} VA United States".format(x) for x in va_vax_schedule['zipcode']]
locations = [coder(query) for query in queries]

result = pd.DataFrame([x.raw for x in locations])
result.drop('licence',axis=1,inplace=True)
result['zipcode'] = va_vax_schedule.reset_index()['zipcode']

result.to_csv(project_path.format("data/VA_zipcodes_cleaned/VA_zips_latlong_nominatim.csv"))
#%%


