import os
import pickle
import pandas as pd
from datetime import datetime

### Creates one data structure containing timestamps, external ECG data, and LVP data from all log files
def create_data_df(folder_path, subject):
    raw_hd_data = pd.DataFrame()
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        file_df = pd.read_csv(file_path, usecols=[0, 49, 46]).dropna()

        raw_hd_data = pd.concat([raw_hd_data, file_df])

    raw_hd_data.to_pickle(f"raw_hd_data_{subject}.pkl")

def create_raw_phase_data(animal_summary_file, raw_data, subject):
    # open raw data file - contains timestamps, external ecg , lvp - and split into column dataframes
    f = open(raw_data, 'rb')
    data = pickle.load(f)
    timestamps, lvp, ecg = split_into_columns(data)
    timestamps = timestamps.apply(lambda x: datetime.strptime(x, "%Y%m%d %H:%M:%S.%f").time())

    # open animal summary file & create list of animal's TD timestamps
    start_times, end_times, labels = timestamps_from_animal(animal_summary_file)

    for i in range(len(start_times)):
        start = start_times[i]
        end = end_times[i]
        label = f"{subject}_{labels[i]}"

        time_data, lvp_data, ecg_data = filter_data(start, end, lvp, timestamps, ecg)

        lvp_df = pd.DataFrame({
            'time' : time_data,
            'lvp' : lvp_data
        })

        ecg_df = pd.DataFrame({
            'time' : time_data,
            'ecg' : ecg_data
        })

        lvp_df.to_pickle(f'{label}_lvp_raw.pkl')
        ecg_df.to_pickle(f'{label}_ecg_raw.pkl')

def split_into_columns(df):
    '''
    takes large dataframe and returns a series of column dataframes -- timestamp, lvp, and ecg
    '''
    return df["RTlog_Timestamp"], df["LVP"], df["Ext_ECG"]

def timestamps_from_animal(s):
    subject = pd.read_csv(s, usecols=[4, 5, 6]).dropna()
    start_times = []
    end_times = []

    for phase in subject["number"].unique():
        phase_data = subject[subject["number"] == phase]
        start = datetime.strptime(phase_data["Time.1"].iloc[0], "%H:%M:%S").time()
        end = datetime.strptime(phase_data["Time.1"].iloc[-1], "%H:%M:%S").time()
        
        start_times.append(start)
        end_times.append(end)

    labels = []
    label_df = pd.read_csv(s, usecols=[9, 10, 11]).dropna()
    for i in label_df.index:
        label = str(label_df.iloc[i]["Med"]) + "_" + str(label_df.iloc[i]["Dose"] + "_" + str(label_df.iloc[i]["P"]))
        labels.append(label)

    return start_times, end_times, labels

def filter_data(start, end, hd, timestamps, ecg):
    filtered = [(t, v1, v2) for t, v1, v2 in zip(timestamps, hd, ecg) if start <= t <= end]
    time_data, hd_data, ecg_data = zip(*filtered)

    return list(time_data), list(hd_data), list(ecg_data)