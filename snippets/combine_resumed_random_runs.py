import os
import re
import dateutil.parser
import datetime
import pandas as pd
import numpy as np

run_dir = "../research_project_data/measles_random_opt_out_2/"
up_to = 7
folder_pattern = r'rand_([\w]+)'
subfolders = ["0_05","0_06","0_07","0_08","0_09","0_10"]
run_csv_pattern = "vacc_sim_{run_num}_{constr}_{isodate}_merged_best_trace.csv"
date_string_index = 6
merge_last_n = 2

# filter directories up to var 'up_to'
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
date_now = datetime.datetime.now().isoformat()
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
        if len(dates) < merge_last_n:
            continue
        dates = sorted(list(set(dates)))
        dates = dates[-merge_last_n:]
        merged_run_name = run_csv_pattern.format(run_num=folder, constr=sfold, isodate=date_now)
        print("write to: ", run_dir+folder+"/"+sfold+"/"+merged_run_name)
        with open(run_dir+folder+"/"+sfold+"/"+merged_run_name,"w") as merged_out_file:
            line_counter = 0 
            for date in dates:
                date_str = date.isoformat()
                print(date_str)
                this_best_trace_name = [fil for fil in fils if date_str in fil and 'best_trace' in fil][0]
                last_opt_value = -np.inf
                last_opt_soln_str = None
                print("open: ",run_dir+folder+"/"+sfold+"/"+this_best_trace_name)
                with open(run_dir+folder+"/"+sfold+"/"+this_best_trace_name,'r') as run_file:
                    for this_line in run_file:
                        this_line = np.array([np.float64(x) for x in this_line.strip().split(",")])
                        this_opt = this_line[-1]
                        this_soln = this_line[:-1]
                        if this_opt > last_opt_value:
                            last_opt_value = this_opt
                            last_opt_soln_str = ",".join([str(x) for x in this_soln])
                        print(last_opt_soln_str+","+str(last_opt_value),file=merged_out_file)
                        line_counter += 1
            print("total lines:", line_counter)
