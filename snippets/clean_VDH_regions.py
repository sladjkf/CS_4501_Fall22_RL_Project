import pandas as pd
import numpy as np

VA_gov_region = pd.read_csv("data/VA_Gov_Regions.csv")
VA_county_city = pd.read_csv("data/VA_zipcodes_cleaned/city_county_names_no_bedford_w_greensville_no_redundancies.csv")

VDH_vector = []
for index, row in VA_county_city.iterrows():
    match = VA_gov_region[VA_gov_region['NAMELSAD'].str.match(row['name'], case=False)]
    if len(match) != 1:
        print(index, row['name'])
        VDH_vector.append(-1)
    else:
        VDH_vector.append(match['VDH_Region'].array[0])

# 11 Bedford City
# 23 Charlotte
# 38 Fairfax
# 44 Franklin
# 46 Frederick
# 104 Richmond
# 106 Roanoke


# --
# 22 Charlotte
# 35 Fairfax
# 41 Franklin
# 43 Frederick
# 100 Richmond
# 102 Roanoke

# missing_regions_indices = [11,23,38,44,46,104,106]
# missing_region_labels = [3,4,2,3,1,5,3]
missing_regions_indices = [22, 35, 41, 43, 100, 102]
missing_region_labels = [4,2,3,1,5,3]
for index,label in zip(missing_regions_indices,missing_region_labels):
    VDH_vector[index]=label

assert -1 not in VDH_vector

pd.Series(np.array(VDH_vector)-1, name='surv_mapping').to_csv("data/VA_zipcodes_cleaned/VDH_surv_mapping.csv", index=False)
