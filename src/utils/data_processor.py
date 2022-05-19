import pandas as pd
import json

# TODO(Hongqing): Add it to a configuration file and load later.
prediction_length = 10

uber_data = "./data/uber-raw-data-janjune-15.csv"
time_series_of_locations = pd.read_csv(uber_data,
                                       header=0,
                                       usecols=["Pickup_date", "locationID"],
                                       index_col=0)
# We divide the raw data according to locationID. Each json line represents a
# time series of a loacationID. The targets are numbers of pickup-events during
# a day.
time_series_of_locations = time_series_of_locations.groupby(by="locationID")
train_path = './data/train-dataset/uber-data.json'
test_path = './data/test-dataset/uber-data.json'
with open(train_path, 'w') as o_train, open(test_path, 'w') as o_test:
    for locationID, df in time_series_of_locations:
        df.sort_index()
        df.index = pd.to_datetime(df.index)
        count_series = df.resample(rule='1D').size()
        start_time = pd.Timestamp(df.index[0]).strftime('%Y-%m-%d %X')
        format_dict = {
            "start": start_time,
            "target": count_series.values.tolist()
        }
        test_json_line = json.dumps(format_dict)
        o_test.write(test_json_line)
        o_test.write('\n')
        format_dict['target'] = format_dict['target'][:-prediction_length]
        train_json_line = json.dumps(format_dict)
        o_train.write(test_json_line)
        o_train.write('\n')
