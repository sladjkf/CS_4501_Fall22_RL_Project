import os
import re
import dateutil.parser
import datetime
import pandas as pd

run_dir = "../research_project_data/measles_random_opt_out_2/"
up_to =100 
folder_pattern = r'rand_([\w]+)'
subfolders = ["0_05","0_06","0_07","0_08","0_09","0_10"]
run_csv_pattern = "vacc_sim_rand_10_0_05_2023-06-06T12:24:07.983023_samples.csv"
date_string_index = 6
line_to_select = 1500
out_csv = "../research_project_data/measles_random_opt_out_2/test.csv"
county_names = "data/VA_zipcodes_cleaned/city_county_names_no_bedford_w_greensville_no_redundancies.csv"


files = os.listdir(run_dir)
print(files)
dirs_filtered = []
for file in files:
    match_obj = re.search(folder_pattern, file)
    if match_obj:
        number = int(match_obj.group(1))
        if number < up_to:
            dirs_filtered.append(file)

dirs_filtered = sorted(dirs_filtered, key = lambda s: int(re.search(folder_pattern,s).group(1)))

print(dirs_filtered)

## build the dataframe ##

county_df = pd.read_csv(county_names)
names = list(county_df['name'])
with open(out_csv,"w") as output_file:
    print("run_num","constr","optimum",",".join(names),sep="," ,file=output_file)
    for folder in dirs_filtered:
        for sfold in subfolders:
            fils = os.listdir(run_dir+folder+"/"+sfold)
            fils = [fil for fil in fils if ".csv" in fil]
            dates = []
            for fil in fils:
                split_fil = fil.split("_")
                #print(split_fil)
                try:
                    dates.append(dateutil.parser.isoparse(split_fil[date_string_index]))
                except ValueError:
                    continue
            try:
                latest_date = max(dates)
            except ValueError:
                continue
            latest_date_iso = latest_date.isoformat()
            # target_csvs = [fil for fil in fils if latest_date_iso in fil]
            # print(target_csvs)
            # find best trace and samples with latest date
            best_trace_name = [fil for fil in fils if latest_date_iso in fil and 'best_trace' in fil][0]
            with open(run_dir+folder+"/"+sfold+"/"+best_trace_name,'r') as run_file:
                last_line = ""
                for i in range(line_to_select):
                    last_line = run_file.readline()
                if last_line == "":
                    print(folder,sfold,"is incomplete")
                    print(folder,sfold,sep=",",file=output_file)
                else:
                    last_line = last_line.split(",")
                    opt_value = last_line.pop().strip()
                    print(folder,sfold,opt_value,",".join(last_line),sep=",",file=output_file)


