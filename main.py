import os
import pickle
import pandas as pd
from datetime import datetime, time, timedelta

### Creates one data structure containing timestamps, external ECG data, and LVP data from all log files

def create_data_df(folder_path="C:/Users/saran/Massachusetts Institute of Technology/Maryam Alsharqi - UROP - Sara Nath/VBU_221_2022_01_21/VBU00221Calibrated_LVV"):
    raw_hd_data = pd.DataFrame()
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        file_df = pd.read_csv(file_path, usecols=[0, 48, 45]).dropna()

        raw_hd_data = pd.concat([raw_hd_data, file_df])

    raw_hd_data.to_pickle("raw_hd_data.pkl")

