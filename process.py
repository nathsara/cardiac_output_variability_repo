import os
import pickle
import pandas as pd
from datetime import datetime, time, timedelta

def arrythmia_removal(label, ecg_df, lvp_df):
    # certain subjects/metrics/phases require arrhythmia extraction and then a set of extra processing steps to clean up the data.
    if (label == "203_Nitro_high_P3"):
        ecg_df = ecg_df[~((ecg_df["time"] > time(10, 58, 36, 000000)) & (ecg_df["time"] < time(10, 58, 42, 000000)))]
        lvp_df = lvp_df[~((lvp_df["time"] > time(10, 58, 36, 000000)) & (lvp_df["time"] < time(10, 58, 42, 000000)))]
        ecg_df = ecg_df[~((ecg_df["time"] > time(10, 59, 14, 000000)) & (ecg_df["time"] < time(10, 59, 16, 000000)))]
        lvp_df = lvp_df[~((lvp_df["time"] > time(10, 59, 14, 000000)) & (lvp_df["time"] < time(10, 59, 16, 000000)))]

    elif (label == "203_Nitro_low_P3"):
        ecg_df = ecg_df[~((ecg_df["time"] > time(10, 32, 29, 000000)) & (ecg_df["time"] < time(10, 32, 30, 000000)))]
        lvp_df = lvp_df[~((lvp_df["time"] > time(10, 32, 29, 000000)) & (lvp_df["time"] < time(10, 32, 30, 000000)))]
        ecg_df = ecg_df[~((ecg_df["time"] > time(10, 32, 38, 000000)) & (ecg_df["time"] < time(10, 32, 41, 000000)))]
        lvp_df = lvp_df[~((lvp_df["time"] > time(10, 32, 38, 000000)) & (lvp_df["time"] < time(10, 32, 41, 000000)))]
        ecg_df = ecg_df[~((ecg_df["time"] > time(10, 32, 52, 000000)) & (ecg_df["time"] < time(10, 32, 54, 000000)))]
        lvp_df = lvp_df[~((lvp_df["time"] > time(10, 32, 52, 000000)) & (lvp_df["time"] < time(10, 32, 54, 000000)))]
        ecg_df = ecg_df[~((ecg_df["time"] > time(10, 33, 9, 000000)) & (ecg_df["time"] < time(10, 33, 10, 000000)))]
        lvp_df = lvp_df[~((lvp_df["time"] > time(10, 33, 9, 000000)) & (lvp_df["time"] < time(10, 33, 10, 000000)))]
        ecg_df = ecg_df[~((ecg_df["time"] > time(10, 33, 12, 000000)) & (ecg_df["time"] < time(10, 33, 14, 000000)))]
        lvp_df = lvp_df[~((lvp_df["time"] > time(10, 33, 12, 000000)) & (lvp_df["time"] < time(10, 33, 14, 000000)))]
        ecg_df = ecg_df[~((ecg_df["time"] > time(10, 33, 22, 000000)) & (ecg_df["time"] < time(10, 33, 24, 000000)))]
        lvp_df = lvp_df[~((lvp_df["time"] > time(10, 33, 22, 000000)) & (lvp_df["time"] < time(10, 33, 24, 000000)))]
        ecg_df = ecg_df[~((ecg_df["time"] > time(10, 33, 41, 000000)) & (ecg_df["time"] < time(10, 33, 42, 000000)))]
        lvp_df = lvp_df[~((lvp_df["time"] > time(10, 33, 41, 000000)) & (lvp_df["time"] < time(10, 33, 42, 000000)))]

    elif (label == "202_Phen_0_P3"):
        ecg_df = ecg_df[~((ecg_df["time"] > time(12, 59, 46, 000000)) & (ecg_df["time"] < time(12, 59, 55, 000000)))]
        lvp_df = lvp_df[~((lvp_df["time"] > time(12, 59, 46, 000000)) & (lvp_df["time"] < time(12, 59, 55, 000000)))]
        ecg_df = ecg_df[~((ecg_df["time"] > time(12, 59, 57, 000000)) & (ecg_df["time"] < time(13, 00, 24, 000000)))]
        lvp_df = lvp_df[~((lvp_df["time"] > time(12, 59, 57, 000000)) & (lvp_df["time"] < time(13, 00, 24, 000000)))]
        ecg_df = ecg_df[~((ecg_df["time"] > time(13, 0, 47, 000000)) & (ecg_df["time"] < time(13, 0, 49, 000000)))]
        lvp_df = lvp_df[~((lvp_df["time"] > time(13, 0, 47, 000000)) & (lvp_df["time"] < time(13, 0, 49, 000000)))]
        ecg_df = ecg_df[~((ecg_df["time"] > time(13, 1, 6, 000000)) & (ecg_df["time"] < time(13, 1, 10, 000000)))]
        lvp_df = lvp_df[~((lvp_df["time"] > time(13, 1, 6, 000000)) & (lvp_df["time"] < time(13, 1, 10, 000000)))]
        ecg_df = ecg_df[~((ecg_df["time"] > time(13, 1, 56, 000000)) & (ecg_df["time"] < time(13, 2, 13, 000000)))]
        lvp_df = lvp_df[~((lvp_df["time"] > time(13, 1, 56, 000000)) & (lvp_df["time"] < time(13, 2, 13, 000000)))]
        ecg_df = ecg_df[~((ecg_df["time"] > time(13, 2, 49, 000000)) & (ecg_df["time"] < time(13, 2, 50, 000000)))]
        lvp_df = lvp_df[~((lvp_df["time"] > time(13, 2, 49, 000000)) & (lvp_df["time"] < time(13, 2, 50, 000000)))]

    elif (label == "202_Washout_1_P3"):
        ecg_df = ecg_df[~((ecg_df["time"] > time(13, 36, 00, 000000)) & (ecg_df["time"] < time(13, 36, 21, 000000)))]
        lvp_df = lvp_df[~((lvp_df["time"] > time(13, 36, 00, 000000)) & (lvp_df["time"] < time(13, 36, 21, 000000)))]

    return ecg_df, lvp_df

def pipeline(ecg_data, lvp_data):
    pass