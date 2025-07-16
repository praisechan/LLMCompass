import os
import json
import pandas as pd
                  

# output_dir_base = "/home/juchanlee/LLMcompass/DRAMsim3/output/SOM/coarse_interleave/tRCD_sweep/DDR4_seqrd_bank1/"
output_dir_base = "/home/juchan.lee/DRAMsim3_juchan/output/SOM/LPDDR4_tRCD_sweep_BL_8/"
req_size_list = [4, 16, 64 ,256, 1024, 4096, 16384, 65536]
dir_name_list = ["RCD25", "RCD50", "RCD100", "RCD200"]
# dir_name_list = ["RCD25", "RCD50", "RCD75", "RCD100", "RCD125", "RCD150", "RCD175", "RCD200"]
# dir_name_list = ["RCDSOM", "RCDPCM", "RCD15"]

file_path = output_dir_base + "total_result.csv"

for dir_name in dir_name_list:
  for req_size in req_size_list:
    output_file = dir_name + '/' + "num_req_"+ str(req_size) +".json"
    json_file_name = output_dir_base + output_file
    with open(json_file_name, 'r') as file:
        data = json.load(file)

    max_latency = 0
    for channel, channel_data in data.items():
        latency = channel_data.get("trans_finish_time") #latency in nanosecond(ns)
        if latency > max_latency:
          max_latency = latency

    # If the combination doesn't exist, append it directly to the CSV in append mode
    new_row = pd.DataFrame({"RCD": [dir_name], "req_size": [req_size], "Latency":[max_latency]})
    new_row.to_csv(file_path, mode='a', index=False, header=False)
    print(f"Combination {dir_name},{req_size} added.")