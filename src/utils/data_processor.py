import pandas as pd
import json

uber_data = "./data/uber-raw-data-janjune-15.csv"
df = pd.read_csv(uber_data,
                 header=0,
                 usecols=["Pickup_date", "locationID"],
                 index_col=0)
df.sort_index()
df.index = pd.to_datetime(df.index)
df = df.resample(rule='1D').mean()
df.locationID = df.locationID.astype('int')
start_time = pd.Timestamp(df.index[0]).strftime('%Y-%m-%d %X')
format_dict = {"start": start_time, "target": df.locationID.values.tolist()}
with open('./data/uber-data.json', 'w') as output_file:
    json.dump(format_dict, output_file)
