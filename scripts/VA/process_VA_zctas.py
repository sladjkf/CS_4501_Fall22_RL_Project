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

#%%
# compute the distance matrix between all the VA zip codes?
# use the centroids? 


# TODO: something about map projections??
# https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.to_crs.html
centroids = va_zipcodes_geodata.to_crs(crs=3857).centroid.to_crs(crs=4326)

plt.figure()
ax = va_zipcodes_geodata.plot()
centroids.plot(color="red",ax=ax,markersize=1)
plt.show()

#%%
# get the population of all VA zctas
# 2010 data.. 
us_zcta_pop = pd.read_csv(project_path.format("data/DECENNIALSF12010.P1_2022-07-11T172722/DECENNIALSF12010.P1_data_with_overlays_2022-07-11T163259.csv"),skiprows=1)
# get zip codes
us_zcta_pop.id = pd.Series([x[9:] for x in us_zcta_pop.id])

#%%

# write population counts and centroids
