import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from ecgdetectors import Detectors
import seaborn as sns
from datetime import time
from scipy.signal import find_peaks

### Creates one data structure containing timestamps, external ECG data, and LVP data from all log files

def create_data_df(folder_path="C:/Users/saran/Massachusetts Institute of Technology/Maryam Alsharqi - UROP - Sara Nath/VBU_221_2022_01_21/VBU00221Calibrated_LVV"):
    raw_hd_data = pd.DataFrame()
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        file_df = pd.read_csv(file_path, usecols=[0, 48, 45]).dropna()

        raw_hd_data = pd.concat([raw_hd_data, file_df])

    raw_hd_data.to_pickle("raw_hd_data.pkl")

############ Script ##############

def script(file="None", animal="221"):
    # open raw data file - contains timestamps, external ecg , lvp - and split into column dataframes
    f = open("raw_hd_data.pkl", 'rb')
    data = pickle.load(f)
    timestamps, lvp, ecg = split_into_columns(data)
    timestamps = timestamps.apply(lambda x: datetime.strptime(x, "%Y%m%d %H:%M:%S.%f").time())

    # open animal summary file & create list of animal's TD timestamps
    start_times, end_times, labels = timestamps_from_animal(file)

    # for later
    dpdt_max_means = []
    dpdt_max_stds = []
    dpdt_min_means = []
    dpdt_min_stds = []
    lvedp_means = []
    lvedp_stds = []

    for i in range(len(start_times)):
        start = start_times[i]
        end = end_times[i]
        label = animal + "_" + labels[i]

        time_data, lvp_data, ecg_data = filter_data(start, end, lvp, timestamps, ecg)

        lvp_df = pd.DataFrame({
            'time' : time_data,
            'lvp' : lvp_data
        })

        ecg_df = pd.DataFrame({
            'time' : time_data,
            'ecg' : ecg_data
        })

        # certain subjects/metrics/phases require arrhythmia extraction and then a set of extra processing steps to clean up the data.
        if (label == "203_Nitro_high_P3"):

            ecg_df = ecg_df[~((ecg_df["time"] > time(10, 58, 36, 000000)) & (ecg_df["time"] < time(10, 58, 42, 000000)))]
            lvp_df = lvp_df[~((lvp_df["time"] > time(10, 58, 36, 000000)) & (lvp_df["time"] < time(10, 58, 42, 000000)))]
            ecg_df = ecg_df[~((ecg_df["time"] > time(10, 59, 14, 000000)) & (ecg_df["time"] < time(10, 59, 16, 000000)))]
            lvp_df = lvp_df[~((lvp_df["time"] > time(10, 59, 14, 000000)) & (lvp_df["time"] < time(10, 59, 16, 000000)))]

            max_dpdt, min_dpdt, lvedp, r_peaks_timestamps, lvp_peaks_timestamps, dpdt = extra_processing_pipeline(ecg_df, lvp_df, label)

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

            max_dpdt, min_dpdt, lvedp, r_peaks_timestamps, lvp_peaks_timestamps, dpdt = extra_processing_pipeline(ecg_df, lvp_df, label)

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

            max_dpdt, min_dpdt, lvedp, r_peaks_timestamps, lvp_peaks_timestamps, dpdt = extra_processing_pipeline(ecg_df, lvp_df, label, dpdt_res=20, lvedp_res=20)

        elif (label == "202_Washout_1_P3"):

            ecg_df = ecg_df[~((ecg_df["time"] > time(13, 36, 00, 000000)) & (ecg_df["time"] < time(13, 36, 21, 000000)))]
            lvp_df = lvp_df[~((lvp_df["time"] > time(13, 36, 00, 000000)) & (lvp_df["time"] < time(13, 36, 21, 000000)))]

            max_dpdt, min_dpdt, lvedp, r_peaks_timestamps, lvp_peaks_timestamps, dpdt = extra_processing_pipeline(ecg_df, lvp_df, label, dpdt_res=20, lvedp_res=20)

        else:
            dpdt = calc_deriv_lvp(list(lvp_df['lvp']), list(lvp_df['time']))
            r_peaks = christov_beat_timestamps(list(ecg_df['ecg']))
            r_peaks_timestamps = [list(ecg_df['time'])[r] for r in r_peaks]
            lvp_peaks = find_peaks(list(lvp_df['lvp']), height=50)
            lvp_peaks_timestamps = [list(lvp_df['time'])[r] for r in list(lvp_peaks[0])]
            r_peaks_timestamps = r_peak_corrector(r_peaks_timestamps, lvp_peaks_timestamps)

            # use r-peak timestamps to extract lvedp
            lvedp_data = []
            reference_indices = [list(lvp_df['time']).index(x) for x in r_peaks_timestamps]
            for ref_idx in reference_indices:
                lvedp_data.append(lvp_data[ref_idx])

            '''for i in reversed(range(len(lvedp_data))):
                if lvedp_data[i] > 30:
                    del lvedp_data[i]
                    del r_peaks_timestamps[i]'''

            lvedp = pd.DataFrame({
                'time' : r_peaks_timestamps,
                'lvedp': lvedp_data})
            
            max_dpdt, min_dpdt, max_timestamps = dpdt_minmax_extractor(dpdt, r_peaks, list(ecg_df['time']))

            # checks values near suppsoed dpdt_maxes to see if nearby values are more accurate
            dpdt_max_list, dpdt_max_time_list = dpdt_max_finetuner(dpdt, max_dpdt["dpdt_max"].tolist(), max_dpdt["time"].tolist(), lvp_df['time'][:-1].tolist())

            # makes sure the time between maxes makes physiological sense -- if two maxes are less than 0.4 seconds apart
            # (time between peaks if HR=150), then the latter max is deleted.
            dpdt_max_list, dpdt_max_time_list = finetune_signal(dpdt_max_list, dpdt_max_time_list)

            max_dpdt = pd.DataFrame({ 'time' : dpdt_max_time_list, 'dpdt_max': dpdt_max_list})

            # repeat for dpdt min
            dpdt_min_list, dpdt_min_time_list = dpdt_min_finetuner(dpdt, min_dpdt["dpdt_min"].tolist(), min_dpdt["time"].tolist(), lvp_df['time'][:-1].tolist())
            dpdt_min_list, dpdt_min_time_list = finetune_signal(dpdt_min_list, dpdt_min_time_list)
            min_dpdt = pd.DataFrame({'time' : dpdt_min_time_list, 'dpdt_min': dpdt_min_list})

        pd.DataFrame({'time': lvp_df['time'][:-1],
                     'dpdt': dpdt}).to_pickle("dpdt_"+label+".pkl")
        max_dpdt.to_pickle("dpdt_max_"+label+".pkl")
        min_dpdt.to_pickle("dpdt_min_"+label+".pkl")
        lvedp.to_pickle("lvedp_"+label+".pkl")

        '''anchored_time_data = [datetime.combine(datetime.today(), t) for t in lvp_df['time']] #for graphing purposes
        anchored_rpeak_timestamps = [datetime.combine(datetime.today(), t) for t in r_peaks_timestamps] #for graphing purposes
        anchored_mdpdt_timestamps = [datetime.combine(datetime.today(), t) for t in max_dpdt["time"]] #for graphing purposes
        anchored_mindpdt_timestamps = [datetime.combine(datetime.today(), t) for t in min_dpdt["time"]] #for graphing purposes
        #anchored_lvppeak_timestamps = [datetime.combine(datetime.today(), t) for t in lvp_peaks_timestamps] #for graphing purposes
'''
        '''_, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        ax1.scatter(anchored_rpeak_timestamps, lvedp["lvedp"], color='black')
        time_start_index = anchored_time_data.index(anchored_rpeak_timestamps[0])
        time_end_index = anchored_time_data.index(anchored_rpeak_timestamps[-1])
        ax2.plot(anchored_time_data[time_start_index:time_end_index], ecg_df["ecg"][time_start_index:time_end_index], color='gray')
        ax3.plot(anchored_time_data[time_start_index:time_end_index], lvp_df["lvp"][time_start_index:time_end_index], color='gray')
        for rpeak in anchored_rpeak_timestamps:
            ax1.axvline(x=rpeak, color='r', linestyle='--')
            ax2.axvline(x=rpeak, color='r', linestyle='--')
            ax3.axvline(x=rpeak, color='r', linestyle='--')
        plt.title("lvedp+ecg+lvp_"+label)     
        plt.show()'''

        '''_, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        time_start_index = anchored_time_data.index(anchored_rpeak_timestamps[0])
        time_end_index = anchored_time_data.index(anchored_rpeak_timestamps[-1])
        ax1.plot(anchored_time_data[time_start_index:time_end_index], ecg_df["ecg"][time_start_index:time_end_index], color='gray')
        for rpeak in anchored_rpeak_timestamps:
            ax1.axvline(x=rpeak, color='r', linestyle='--')

        ax2.plot(anchored_time_data[time_start_index:time_end_index], dpdt[time_start_index:time_end_index], color='gray')
        ax2.scatter(anchored_mdpdt_timestamps, max_dpdt["dpdt_max"], color="blue")
        ax2.scatter(anchored_mindpdt_timestamps, min_dpdt["dpdt_min"], color="green")

        ax3.plot(anchored_time_data[time_start_index:time_end_index], lvp_df["lvp"][time_start_index:time_end_index], color='gray')
        ax3.scatter(anchored_rpeak_timestamps, lvedp["lvedp"], color='black')
        #for lvppeak in anchored_lvppeak_timestamps:
        #    ax3.axvline(x=lvppeak, color='blue', linestyle='--')
        
        plt.title("ecg/rpeaks_dpdt-max/dpdt_lvedp/lvp_"+label)
        plt.show()'''

        #outlier detection
        dpdt_max_outliers = detect_outliers(max_dpdt["time"], max_dpdt["dpdt_max"], label+"dpdt_max1")
        dpdt_min_outliers = detect_outliers(min_dpdt["time"], min_dpdt["dpdt_min"], label+"dpdt_min1")
        lvedp_outliers = detect_outliers(lvedp["time"], lvedp["lvedp"], label+"lvedp1")

        #outlier cleaning -- 1
        dpdt_max_cleaned = remove_outliers(max_dpdt, dpdt_max_outliers)
        dpdt_min_cleaned = remove_outliers(min_dpdt, dpdt_min_outliers)
        lvedp_cleaned = remove_outliers(lvedp, lvedp_outliers)

        # save files after first round of outlier extraction:
        dpdt_max_cleaned.to_pickle("dpdt_max_"+label+"_cleaned.pkl")
        dpdt_min_cleaned.to_pickle("dpdt_min_"+label+"_cleaned.pkl")
        lvedp_cleaned.to_pickle("lvedp_"+label+"_cleaned.pkl")

        #outlier detection -- 2
        dpdt_max_outliers2 = detect_outliers(dpdt_max_cleaned["time"], dpdt_max_cleaned["dpdt_max"], label+"dpdt_max2")
        dpdt_min_outliers2 = detect_outliers(dpdt_min_cleaned["time"], dpdt_min_cleaned["dpdt_min"], label+"dpdt_min2")
        lvedp_outliers2 = detect_outliers(lvedp_cleaned['time'], lvedp_cleaned["lvedp"], label+"lvedp2")

        #outlier cleaning -- 2
        dpdt_max_cleaner = remove_outliers(dpdt_max_cleaned, dpdt_max_outliers2)
        dpdt_min_cleaner = remove_outliers(dpdt_min_cleaned, dpdt_min_outliers2)
        lvedp_cleaner = remove_outliers(lvedp_cleaned, lvedp_outliers2)

        # debugging plot -- plotting lvedp, max_dpdt, and r_peaks on some plot.

        '''anchored_time_data = [datetime.combine(datetime.today(), t) for t in lvp_df['time']] #for graphing purposes
        anchored_rpeak_timestamps = [datetime.combine(datetime.today(), t) for t in r_peaks_timestamps] #for graphing purposes
        anchored_mdpdt_timestamps = [datetime.combine(datetime.today(), t) for t in dpdt_max_cleaner["time"]] #for graphing purposes
        anchored_mindpdt_timestamps = [datetime.combine(datetime.today(), t) for t in dpdt_min_cleaner["time"]] #for graphing purposes
        anchored_lvedp_timestamps = [datetime.combine(datetime.today(), t) for t in lvedp_cleaner['time']]
        #anchored_lvppeak_timestamps = [datetime.combine(datetime.today(), t) for t in lvp_peaks_timestamps] #for graphing purposes

        _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        time_start_index = anchored_time_data.index(anchored_rpeak_timestamps[0])
        time_end_index = anchored_time_data.index(anchored_rpeak_timestamps[-1])
        ax1.plot(anchored_time_data[time_start_index:time_end_index], ecg_df["ecg"][time_start_index:time_end_index], color='gray')
        for rpeak in anchored_rpeak_timestamps:
            ax1.axvline(x=rpeak, color='r', linestyle='--')

        ax2.plot(anchored_time_data[time_start_index:time_end_index], dpdt[time_start_index:time_end_index], color='gray')
        ax2.scatter(anchored_mdpdt_timestamps, dpdt_max_cleaner["dpdt_max"], color="blue")
        ax2.scatter(anchored_mindpdt_timestamps, dpdt_min_cleaner["dpdt_min"], color="green")

        ax3.plot(anchored_time_data[time_start_index:time_end_index], lvp_df["lvp"][time_start_index:time_end_index], color='gray')
        ax3.scatter(anchored_lvedp_timestamps, lvedp_cleaner["lvedp"], color='black')
        #for lvppeak in anchored_lvppeak_timestamps:
        #    ax3.axvline(x=lvppeak, color='blue', linestyle='--')
        
        plt.title("ecg/rpeaks_dpdt-max/dpdt_lvedp/lvp_"+label)
        plt.show()'''

        # save files after second round of outlier extraction:
        dpdt_max_cleaner.to_pickle("dpdt_max_"+label+"_cleaner.pkl")
        dpdt_min_cleaner.to_pickle("dpdt_min_"+label+"_cleaner.pkl")
        lvedp_cleaner.to_pickle("lvedp_"+label+"_cleaner.pkl")

        #compute mean and std of each file and add to running list (defined earlier):
        dpdt_max_mean = dpdt_max_cleaner["dpdt_max"].mean()
        dpdt_max_std = dpdt_max_cleaner["dpdt_max"].std()
        dpdt_max_means.append(dpdt_max_mean)
        dpdt_max_stds.append(dpdt_max_std)

        dpdt_min_mean = dpdt_min_cleaner["dpdt_min"].mean()
        dpdt_min_std = dpdt_min_cleaner["dpdt_min"].std()
        dpdt_min_means.append(dpdt_min_mean)
        dpdt_min_stds.append(dpdt_min_std)

        lvedp_mean = lvedp_cleaner["lvedp"].mean()
        lvedp_std = lvedp_cleaner["lvedp"].std()
        lvedp_means.append(lvedp_mean)
        lvedp_stds.append(lvedp_std)

    # save files:
    dpdt_max_means_df = pd.DataFrame(dpdt_max_means, columns=["dpdt_max_mean"])
    dpdt_min_means_df = pd.DataFrame(dpdt_min_means, columns=["dpdt_min_mean"])
    lvedp_means_df = pd.DataFrame(lvedp_means, columns=["lvedp_mean"])

    dpdt_max_stds_df = pd.DataFrame(dpdt_max_stds, columns=["dpdt_max_std"])
    dpdt_min_stds_df = pd.DataFrame(dpdt_min_stds, columns=["dpdt_min_std"])
    lvedp_stds_df = pd.DataFrame(lvedp_stds, columns=["lvedp_std"])

    dpdt_max_means_df.to_pickle("dpdt_max_means_"+animal+".pkl")
    dpdt_min_means_df.to_pickle("dpdt_min_means_"+animal+".pkl")
    lvedp_means_df.to_pickle("lvedp_means_"+animal+".pkl")

    dpdt_max_stds_df.to_pickle("dpdt_max_stds"+animal+".pkl")
    dpdt_min_stds_df.to_pickle("dpdt_min_stds"+animal+".pkl")
    lvedp_stds_df.to_pickle("lvedp_stds"+animal+".pkl")

    #generate_221_plot(dpdt_max_means, dpdt_max_stds, animal, "dp/dt max")
    #generate_221_plot(dpdt_min_means, dpdt_min_stds, animal, "dp/dt min")
    #generate_221_plot(lvedp_means, lvedp_stds, animal, "lvedp")

    #generate_203_plot(dpdt_max_means, dpdt_max_stds, animal, "dp/dt max")
    #generate_203_plot(dpdt_min_means, dpdt_min_stds, animal, "dp/dt min")
    #generate_203_plot(lvedp_means, lvedp_stds, animal, "lvedp")

    #generate_205_plot(dpdt_max_means, dpdt_max_stds, animal, "dp/dt max")
    #generate_205_plot(dpdt_min_means, dpdt_min_stds, animal, "dp/dt min")
    #generate_205_plot(lvedp_means, lvedp_stds, animal, "lvedp")

    generate_202_plot(dpdt_max_means, dpdt_max_stds, animal, "dp/dt max")
    generate_202_plot(dpdt_min_means, dpdt_min_stds, animal, "dp/dt min")
    generate_202_plot(lvedp_means, lvedp_stds, animal, "lvedp")

# ONCE WE HAVE ALL OF THIS FOR ALL ANIMALS:
# graph: TD-AIC (saved data) against dpdt_max, dpdt_min means


############ Helper Functions #############3

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

def calc_deriv_lvp(df, time):

    x = list(df)
    dxdt = []

    for i in range(len(x)-1):
        diffx = x[i+1] - x[i]
        dt1 = datetime.combine(datetime.today(), time[i+1])
        dt2 = datetime.combine(datetime.today(), time[i]) #anchor to a date for easy arithmetic
        difft = (dt1 - dt2).total_seconds() * 1000
        dxdt.append(diffx/difft)

    return dxdt


def christov_beat_timestamps(ext_ecg):

    detectors = Detectors(250)
    detector = detectors.christov_detector
    r_peaks = detector(ext_ecg)
    
    return r_peaks

def pan_tompkins_beat_timestamps(ext_ecg):

    detectors = Detectors(250)
    detector = detectors.pan_tompkins_detector
    r_peaks = detector(ext_ecg)
    
    return r_peaks

def hamilton_beat_timestamps(ext_ecg):

    detectors = Detectors(250)
    detector = detectors.hamilton_detector
    r_peaks = detector(ext_ecg)
    
    return r_peaks

def two_average_beat_timestamps(ext_ecg):

    detectors = Detectors(250)
    detector = detectors.two_average_detector
    r_peaks = detector(ext_ecg)
    
    return r_peaks

def r_peak_corrector(r_peak_timestamps, lvp_peaks_timestamps):
    r_peaks = []

    for rts_index in range(len(r_peak_timestamps)-1):
        true_r_peak = False
        curr_r_peak = r_peak_timestamps[rts_index]
        next_r_peak = r_peak_timestamps[rts_index+1]

        for lvpp_index in range(len(lvp_peaks_timestamps)):
            curr_lvp_peak = lvp_peaks_timestamps[lvpp_index]
            if curr_r_peak <= curr_lvp_peak <= next_r_peak:
                true_r_peak = True

        if true_r_peak:
            r_peaks.append(r_peak_timestamps[rts_index])

    return r_peaks

def dpdt_minmax_extractor(dpdt, peak_indices, time_data):

    max_per_beat = []
    min_per_beat = []
    max_times = []
    min_times = []

    for i in range(len(peak_indices)-1):
        start = peak_indices[i]
        end = peak_indices[i+1]

        dpdt_segment = dpdt[start:end]
        time_segment = time_data[start:end]

        max_dpdt = max(dpdt_segment)
        min_dpdt = min(dpdt_segment)

        max_index = dpdt_segment.index(max_dpdt)
        min_index = dpdt_segment.index(min_dpdt)
        max_time = time_segment[max_index]
        min_time = time_segment[min_index]

        max_per_beat.append(max_dpdt)
        min_per_beat.append(min_dpdt)
        max_times.append(max_time)
        min_times.append(min_time)

    max_df = pd.DataFrame({
        'time' : max_times,
        'dpdt_max': max_per_beat})
    min_df = pd.DataFrame({
        'time' : min_times,
        'dpdt_min' : min_per_beat})

    return max_df, min_df, max_times

def detect_outliers(dataset_x, dataset_y, label=None):

    outliers = []
    ds_mean = dataset_y.mean()
    ds_std = dataset_y.std()
    upper = ds_mean + 2 * ds_std
    lower = ds_mean - 2 * ds_std

    for i in range(len(dataset_y)):
        point_x = dataset_x[i]
        point_y = dataset_y[i]

        if point_y > upper or point_y < lower:
            outliers.append((point_x, point_y))

    outliers_df = pd.DataFrame(outliers, columns=["time", "data"])

    return outliers_df

def remove_outliers(data, outliers):
    data_col = data.columns[1]
    outliers_removed = data[~data[data_col].isin(outliers.data)]
    outliers_removed.reset_index(drop=True, inplace=True)

    return outliers_removed

def generate_221_plot(means, stds, subject, metric):
    '''
    Generates scatterplot tracking mean of a metric for a given animal across all pharmacologically-induced states..
    '''

    # BASE PLOT - STAYS THE SAME FOR ALL DATA

    state_change_times = [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5]
    state_labels = ['Baseline', 'Nitroprusside', 'Washout', 'Phenylephrine', 'Washout',
                    'Dobutamine']


    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    plt.axvspan(1.5, 3.5, color='lightblue', alpha=0.5, label='Low Dose')
    plt.axvspan(7.5, 9.5, color='lightblue', alpha=0.5)
    plt.axvspan(9.5, 11.5, color='lightcoral', alpha=0.5, label='High Dose')
    plt.axvspan(3.5, 5.5, color='lightcoral', alpha=0.5)
    plt.axvspan(13.5, 16, color='lightgray', alpha=0.5, label='Uniform Dose')

    for i, t in enumerate(state_change_times):
        plt.axvline(x=t, color='gray', linestyle='--', linewidth=1)

    plt.text(0.5, 3.75, state_labels[0], horizontalalignment='center', verticalalignment='top', fontsize=9)
    plt.text(3.5, 3.75, state_labels[1], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(6.5, 3.75, state_labels[2], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(9.5, 3.75, state_labels[3], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(12.5, 3.75, state_labels[4], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(15, 3.75, state_labels[5], horizontalalignment='center', verticalalignment='top',  fontsize=9)

    # DATA SPECIFIC PLOT MODIFICATIONS + GRAPHING OF DATA

    y_low = min([means[i] - stds[i] for i in range(len(means))])
    y_high = max([means[i] + stds[i] for i in range(len(means))])

    plt.ylim(round(y_low-1), round(y_high+1))
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(list(range(0, 16)))

    xaxis = list(range(len(means)))

    p3_mean = [mean for mean in means if means.index(mean) % 2 == 1]
    p6_mean = [mean for mean in means if means.index(mean) % 2 == 0]
    p3_std = [std for std in stds if stds.index(std) % 2 == 1]
    p6_std = [std for std in stds if stds.index(std) % 2 == 0]
    xaxis_p3 = [x for x in xaxis if xaxis.index(x) % 2 == 1]
    xaxis_p6 = [x for x in xaxis if xaxis.index(x) % 2 == 0]

    plt.errorbar(xaxis_p6, p6_mean, yerr=p6_std, marker='o', capsize=5, linestyle="None", linewidth=1, label='P6')
    plt.errorbar(xaxis_p3, p3_mean, yerr=p3_std, marker='o', capsize=5, linestyle="None", linewidth=1, color='green', label='P3')

    plt.title(metric + ' (Subject ' + subject + ")")
    plt.ylabel('mmHG/s')
    plt.legend()
    plt.grid(axis="x")
    plt.show()

def generate_203_plot(means, stds, subject, metric):
    '''
    Generates scatterplot tracking mean of a metric for a given animal across all pharmacologically-induced states..
    '''

    # BASE PLOT - STAYS THE SAME FOR ALL DATA

    state_change_times = [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.5, 19.5]
    state_labels = ['Baseline', 'Nitroprusside', 'Washout', 'Phenylephrine', 'Washout',
                    'Dobutamine', 'Washout', 'Esmolol']


    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    plt.axvspan(1.5, 3.5, color='lightblue', alpha=0.5, label='Low Dose')
    plt.axvspan(7.5, 9.5, color='lightblue', alpha=0.5)
    plt.axvspan(13.5, 15.5, color='lightblue', alpha=0.5)
    plt.axvspan(9.5, 11.5, color='lightcoral', alpha=0.5, label='High Dose')
    plt.axvspan(3.5, 5.5, color='lightcoral', alpha=0.5)
    plt.axvspan(15.5, 17.5, color='lightcoral', alpha=0.5)
    plt.axvspan(19.5, 21.5, color='lightgray', alpha=0.5, label='Uniform Dose')

    for i, t in enumerate(state_change_times):
        plt.axvline(x=t, color='gray', linestyle='--', linewidth=1)

    plt.text(0.5, -4.5, state_labels[0], horizontalalignment='center', verticalalignment='top', fontsize=9)
    plt.text(3.5, -4.5, state_labels[1], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(6.5, -4.5, state_labels[2], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(9.5, -4.5, state_labels[3], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(12.5, -4.5, state_labels[4], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(15.5, -4.5, state_labels[5], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(18.5, -4.5, state_labels[6], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(20.5, -4.5, state_labels[7], horizontalalignment='center', verticalalignment='top',  fontsize=9)

    # DATA SPECIFIC PLOT MODIFICATIONS + GRAPHING OF DATA

    y_low = min([means[i] - stds[i] for i in range(len(means))])
    y_high = max([means[i] + stds[i] for i in range(len(means))])

    plt.ylim(round(y_low-1), round(y_high+1))
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(list(range(0, 22)))

    xaxis = list(range(len(means)))

    p3_mean = [mean for mean in means if means.index(mean) % 2 == 1]
    p6_mean = [mean for mean in means if means.index(mean) % 2 == 0]
    p3_std = [std for std in stds if stds.index(std) % 2 == 1]
    p6_std = [std for std in stds if stds.index(std) % 2 == 0]
    xaxis_p3 = [x for x in xaxis if xaxis.index(x) % 2 == 1]
    xaxis_p6 = [x for x in xaxis if xaxis.index(x) % 2 == 0]

    plt.errorbar(xaxis_p6, p6_mean, yerr=p6_std, marker='o', capsize=5, linestyle="None", linewidth=1, label='P6')
    plt.errorbar(xaxis_p3, p3_mean, yerr=p3_std, marker='o', capsize=5, linestyle="None", linewidth=1, color='green', label='P3')

    plt.title(metric + ' (Subject ' + subject + ")")
    plt.ylabel('mmHG/s')
    plt.legend()
    plt.grid(axis="x")
    plt.show()

def generate_205_plot(means, stds, subject, metric):
    '''
    Generates scatterplot tracking mean of a metric for a given animal across all pharmacologically-induced states..
    '''

    # BASE PLOT - STAYS THE SAME FOR ALL DATA

    state_change_times = [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5]
    state_labels = ['Baseline', 'Nitroprusside', 'Washout', 'Phenylephrine', 'Washout',
                    'Dobutamine']


    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    plt.axvspan(1.5, 3.5, color='lightblue', alpha=0.5, label='Low Dose')
    plt.axvspan(7.5, 9.5, color='lightblue', alpha=0.5)
    plt.axvspan(13.5, 15.5, color='lightblue', alpha=0.5)
    plt.axvspan(9.5, 11.5, color='lightcoral', alpha=0.5, label='High Dose')
    plt.axvspan(3.5, 5.5, color='lightcoral', alpha=0.5)
    plt.axvspan(15.5, 17.5, color='lightcoral', alpha=0.5)

    for i, t in enumerate(state_change_times):
        plt.axvline(x=t, color='gray', linestyle='--', linewidth=1)

    plt.text(0.5, -1.5, state_labels[0], horizontalalignment='center', verticalalignment='top', fontsize=9)
    plt.text(3.5, -1.5, state_labels[1], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(6.5, -1.5, state_labels[2], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(9.5, -1.5, state_labels[3], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(12.5, -1.5, state_labels[4], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(15.5, -1.5, state_labels[5], horizontalalignment='center', verticalalignment='top',  fontsize=9)

    # DATA SPECIFIC PLOT MODIFICATIONS + GRAPHING OF DATA

    y_low = min([means[i] - stds[i] for i in range(len(means))])
    y_high = max([means[i] + stds[i] for i in range(len(means))])

    plt.ylim(round(y_low-1), round(y_high+1))
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(list(range(0, 17)))

    xaxis = list(range(len(means)))

    p3_mean = [mean for mean in means if means.index(mean) % 2 == 1]
    p6_mean = [mean for mean in means if means.index(mean) % 2 == 0]
    p3_std = [std for std in stds if stds.index(std) % 2 == 1]
    p6_std = [std for std in stds if stds.index(std) % 2 == 0]
    xaxis_p3 = [x for x in xaxis if xaxis.index(x) % 2 == 1]
    xaxis_p6 = [x for x in xaxis if xaxis.index(x) % 2 == 0]

    plt.errorbar(xaxis_p6, p6_mean, yerr=p6_std, marker='o', capsize=5, linestyle="None", linewidth=1, label='P6')
    plt.errorbar(xaxis_p3, p3_mean, yerr=p3_std, marker='o', capsize=5, linestyle="None", linewidth=1, color='green', label='P3')

    plt.title(metric + ' (Subject ' + subject + ")")
    plt.ylabel('mmHG/s')
    plt.legend()
    plt.grid(axis="x")
    plt.show()

def generate_202_plot(means, stds, subject, metric):
    '''
    Generates scatterplot tracking mean of a metric for a given animal across all pharmacologically-induced states..
    '''

    # BASE PLOT - STAYS THE SAME FOR ALL DATA

    state_change_times = [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.5]
    state_labels = ['Baseline', 'Nitroprusside', 'Washout', 'Phenylephrine', 'Washout',
                    'Dobutamine', 'Washout', 'Esmolol']


    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    plt.axvspan(1.5, 3.5, color='lightblue', alpha=0.5, label='Low Dose')
    plt.axvspan(3.5, 5.5, color='lightcoral', alpha=0.5, label='High Dose')
    plt.axvspan(7.5, 9.5, color='lightgray', alpha=0.5, label='Uniform Dose')
    plt.axvspan(11.5, 13.5, color='lightgray', alpha=0.5)
    plt.axvspan(11.5, 13.5, color='lightgray', alpha=0.5)
    plt.axvspan(15.5, 17.5, color='lightgray', alpha=0.5)

    for i, t in enumerate(state_change_times):
        plt.axvline(x=t, color='gray', linestyle='--', linewidth=1)

    plt.text(0.5, -4.5, state_labels[0], horizontalalignment='center', verticalalignment='top', fontsize=9)
    plt.text(3.5, -4.5, state_labels[1], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(6.5, -4.5, state_labels[2], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(8.5, -4.5, state_labels[3], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(10.5, -4.5, state_labels[4], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(12.5, -4.5, state_labels[5], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(14.5, -4.5, state_labels[6], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(16.5, -4.5, state_labels[7], horizontalalignment='center', verticalalignment='top',  fontsize=9)

    # DATA SPECIFIC PLOT MODIFICATIONS + GRAPHING OF DATA

    y_low = min([means[i] - stds[i] for i in range(len(means))])
    y_high = max([means[i] + stds[i] for i in range(len(means))])

    plt.ylim(round(y_low-1), round(y_high+1))
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(list(range(0, 18)))

    xaxis = list(range(len(means)))

    p3_mean = [mean for mean in means if means.index(mean) % 2 == 1]
    p6_mean = [mean for mean in means if means.index(mean) % 2 == 0]
    p3_std = [std for std in stds if stds.index(std) % 2 == 1]
    p6_std = [std for std in stds if stds.index(std) % 2 == 0]
    xaxis_p3 = [x for x in xaxis if xaxis.index(x) % 2 == 1]
    xaxis_p6 = [x for x in xaxis if xaxis.index(x) % 2 == 0]

    plt.errorbar(xaxis_p6, p6_mean, yerr=p6_std, marker='o', capsize=5, linestyle="None", linewidth=1, label='P6')
    plt.errorbar(xaxis_p3, p3_mean, yerr=p3_std, marker='o', capsize=5, linestyle="None", linewidth=1, color='green', label='P3')

    plt.title(metric + ' (Subject ' + subject + ")")
    plt.ylabel('mmHG/s')
    plt.legend()
    plt.grid(axis="x")
    plt.show()

def pipeline_segment(ecg_segment, lvp_segment):
    '''
    This function is a segment of the pipeline outlined in the main script -- it includes calculating dp/dt, identifying and fine-tuning
    the R-peaks, identifying LVP peaks, calculating LVEDP, and calculating dp/dt_max and dp/dt_min. This function accepts small
    continuous segments of ECG and LVP data, and returns the metrics described above for these small segments of data.
    The rest of the pipeline (outlier detection/extraction, graph generation) is applied to the data as a whole, once these segments
    have been stitched together.
    '''
    dpdt_segment = calc_deriv_lvp(list(lvp_segment['lvp']), list(lvp_segment['time']))
    r_peaks_segment = christov_beat_timestamps(list(ecg_segment['ecg']))
    r_peaks_timestamps_segment = [list(ecg_segment['time'])[r] for r in r_peaks_segment]
    lvp_peaks_segment = find_peaks(list(lvp_segment['lvp']), height=50)
    lvp_peaks_timestamps_segment = [list(lvp_segment['time'])[r] for r in list(lvp_peaks_segment[0])]
    r_peaks_timestamps_segment = r_peak_corrector(r_peaks_timestamps_segment, lvp_peaks_timestamps_segment)

    lvedp_data_segment = []
    reference_indices = [list(lvp_segment['time']).index(x) for x in r_peaks_timestamps_segment]
    for ref_idx in reference_indices:
        lvedp_data_segment.append(lvp_segment["lvp"][ref_idx])

    lvedp_segment = pd.DataFrame({
            'time' : r_peaks_timestamps_segment,
            'lvedp': lvedp_data_segment})
    
    dpdt_segment_df = pd.DataFrame({'time': lvp_segment['time'][:-1],
                     'dpdt': dpdt_segment})
        
    max_dpdt_segment, min_dpdt_segment, max_timestamps_segment = dpdt_minmax_extractor(dpdt_segment, r_peaks_segment, list(ecg_segment['time']))

    return max_dpdt_segment, min_dpdt_segment, lvedp_segment, r_peaks_timestamps_segment, lvp_peaks_timestamps_segment, dpdt_segment_df

def stitch_segments(list_of_segments):
    '''
    Given a list of data segments -- each segment should be a DataFrame -- stitches segments together and returns one DataFrame
    containing all data.
    '''
    if type(list_of_segments[0]) == list:
        full = []

        for segment in list_of_segments:
            full.extend(segment)
    else:
        full = pd.DataFrame()

        for segment in list_of_segments:
            full = pd.concat([full, segment], ignore_index=True)

    return full

def split_data_into_segments(data, max_segment_length=10, SAMPLING_FREQUENCY=250):
    '''
    Given a DataFrame for ECG or LVP, split it into continuous segments. Segments should be no longer than the max_segment_length
    specified (default is 10 seconds). When we have a gap of more than 4 milliseconds between a data point and the data point after it, that
    means we've hit a discontinuity; end the segment and start a new segment.
    '''

    time_data = data["time"]
    anchored_time = [datetime.combine(datetime.today(), t) for t in time_data]
    list_of_segments = []
    current_segment = pd.DataFrame()
    max_data_points = max_segment_length * SAMPLING_FREQUENCY

    for i in range(len(data)-1):
        curr_row = data.iloc[[i]]
        curr_time = anchored_time[i]
        next_time = anchored_time[i+1]
        diff = (next_time - curr_time).total_seconds()

        if diff > 0.01 or len(current_segment) >= max_data_points:
            current_segment = pd.concat([current_segment, curr_row], ignore_index=True)
            list_of_segments.append(current_segment)
            current_segment = pd.DataFrame()
        else:
            current_segment = pd.concat([current_segment, curr_row], ignore_index=True)

    return list_of_segments

def finetune_signal(signal_data, signal_time_axis):
    '''
    removes extraneous values detected due to noise by controlling for minimum period (heart rate)
    '''

    selected = []
    selected_timestamps = []

    idx=0 # starting index

    while idx < len(signal_data) - 1:
        curr = signal_data[idx]

        selected.append(curr)
        selected_timestamps.append(signal_time_axis[idx])

        anchored_time_curr = datetime.combine(datetime.today(), signal_time_axis[idx])
        anchored_time_next = datetime.combine(datetime.today(), signal_time_axis[idx+1])

        time_diff = anchored_time_next - anchored_time_curr

        if time_diff > timedelta(milliseconds=400): # =0.4 seconds -- period for 150bpm HR
            idx += 1
        else:
            idx += 2

    return selected, selected_timestamps

def lvedp_finetuner(lvedp, lvedp_timestamps, dpdt, lvp, lvp_timestamps, resolution=10):
    '''
    resolution = number of data points examined before and after identified r-peak timestamp. 10 data points = 40 ms of data.
    '''
    
    for i in range(len(lvedp_timestamps)):
        index_on_lvp = lvp_timestamps.index(lvedp_timestamps[i])
        real_lvedp = lvedp[i]
        real_lvedp_timestamp = lvedp_timestamps[i]
        corr_dpdt = dpdt[index_on_lvp]

        if real_lvedp > 25:
            for j in range(index_on_lvp-resolution, index_on_lvp+resolution+1):
                if dpdt[j] < corr_dpdt and dpdt[j] > 0 and lvp[j] <= 25:
                    try:
                        corr_dpdt = dpdt[j]
                        real_lvedp = lvp[j]
                        real_lvedp_timestamp = lvp_timestamps[j]
                    except KeyError:
                        j += 1

            lvedp[i] = real_lvedp
            lvedp_timestamps[i] = real_lvedp_timestamp

    return lvedp, lvedp_timestamps

def dpdt_max_finetuner(dpdt, dpdt_max, dpdt_max_timestamps, dpdt_timestamps, resolution=15):
    '''
    resolution = number of data points examined before and after identified r-peak timestamp. 15 data points = 60 ms of data.
    '''

    for i in range(len(dpdt_max_timestamps)):
        index_on_dpdt = dpdt_timestamps.index(dpdt_max_timestamps[i])
        real_dpdt_max = dpdt_max[i]
        real_dpdt_max_timestamp = dpdt_max_timestamps[i]

        for j in range(index_on_dpdt-resolution, index_on_dpdt+resolution+1):
            if j >= len(dpdt):
                break
            if dpdt[j] > real_dpdt_max:
                real_dpdt_max = dpdt[j]
                real_dpdt_max_timestamp = dpdt_timestamps[j]

        dpdt_max[i] = real_dpdt_max
        dpdt_max_timestamps[i] = real_dpdt_max_timestamp

    return dpdt_max, dpdt_max_timestamps

def dpdt_min_finetuner(dpdt, dpdt_min, dpdt_min_timestamps, dpdt_timestamps, resolution=15):
    '''
    resolution = number of data points examined before and after identified r-peak timestamp. 15 data points = 60 ms of data.
    '''

    for i in range(len(dpdt_min_timestamps)):
        index_on_dpdt = dpdt_timestamps.index(dpdt_min_timestamps[i])
        real_dpdt_min = dpdt_min[i]
        real_dpdt_min_timestamp = dpdt_min_timestamps[i]

        for j in range(index_on_dpdt-resolution, index_on_dpdt+resolution+1):
            if dpdt[j] < real_dpdt_min:
                real_dpdt_min = dpdt[j]
                real_dpdt_min_timestamp = dpdt_timestamps[j]

        dpdt_min[i] = real_dpdt_min
        dpdt_min_timestamps[i] = real_dpdt_min_timestamp

    return dpdt_min, dpdt_min_timestamps

def extra_processing_pipeline(ecg_df, lvp_df, label, lvedp_finetuning=True, dpdt_finetuning=True, dpdt_res=15, lvedp_res=10):
    # after removing the arrhythmias, we now have gaps in our data -- and we only want continuous segments of data being fed into
    # the pipeline from here on out.
    ecg_segments = split_data_into_segments(ecg_df)
    lvp_segments = split_data_into_segments(lvp_df)

    max_dpdt_segments = []
    min_dpdt_segments = []
    lvedp_segments = []
    r_peaks_timestamps_segments = []
    lvp_peaks_timestamps_segments = []
    dpdt_segments = []
            

    for i in range(len(ecg_segments)):
        ecg_segment = ecg_segments[i]
        lvp_segment = lvp_segments[i]

        max_dpdt_segment, min_dpdt_segment, lvedp_segment, r_peaks_timestamps_segment, lvp_peaks_timestamps_segment, dpdt_segment = pipeline_segment(ecg_segment, lvp_segment)

        max_dpdt_segments.append(max_dpdt_segment)
        min_dpdt_segments.append(min_dpdt_segment)
        lvedp_segments.append(lvedp_segment)
        r_peaks_timestamps_segments.append(r_peaks_timestamps_segment)
        lvp_peaks_timestamps_segments.append(lvp_peaks_timestamps_segment)
        dpdt_segments.append(dpdt_segment)

        anchored_time_data = [datetime.combine(datetime.today(), t) for t in lvp_segment['time']] #for graphing purposes
        anchored_lvedp_timestamps = [datetime.combine(datetime.today(), t) for t in lvedp_segment['time']]
        anchored_rpeak_timestamps = [datetime.combine(datetime.today(), t) for t in r_peaks_timestamps_segment] #for graphing purposes
        time_start_index = anchored_time_data.index(anchored_rpeak_timestamps[0])
        time_end_index = anchored_time_data.index(anchored_rpeak_timestamps[-1])

        plt.plot(anchored_time_data[time_start_index:time_end_index], lvp_segment["lvp"][time_start_index:time_end_index], color='gray')
        plt.scatter(anchored_lvedp_timestamps, lvedp_segment["lvedp"], color='black')
        
        plt.title(f"lvedp/lvp_{label}_{i}")
        plt.show()

    max_dpdt = stitch_segments(max_dpdt_segments)
    min_dpdt = stitch_segments(min_dpdt_segments)
    lvedp = stitch_segments(lvedp_segments)
    r_peaks_timestamps = stitch_segments(r_peaks_timestamps_segments)
    lvp_peaks_timestamps = stitch_segments(lvp_peaks_timestamps_segments)
    dpdt_df = stitch_segments(dpdt_segments)
    dpdt = dpdt_df["dpdt"]

    if dpdt_finetuning:
        # checks values near suppsoed dpdt_maxes to see if nearby values are more accurate
        dpdt_max_list, dpdt_max_time_list = dpdt_max_finetuner(dpdt, max_dpdt["dpdt_max"].tolist(), max_dpdt["time"].tolist(), dpdt_df["time"].tolist(), resolution=dpdt_res)

        # makes sure the time between maxes makes physiological sense -- if two maxes are less than 0.4 seconds apart
        # (time between peaks if HR=150), then the latter max is deleted.
        dpdt_max_list, dpdt_max_time_list = finetune_signal(dpdt_max_list, dpdt_max_time_list)

        max_dpdt = pd.DataFrame({
        'time' : dpdt_max_time_list,
        'dpdt_max': dpdt_max_list})

        # do same for dp/dt min:
        # checks values near suppsoed dpdt_maxes to see if nearby values are more accurate
        dpdt_min_list, dpdt_min_time_list = dpdt_min_finetuner(dpdt, min_dpdt["dpdt_min"].tolist(), min_dpdt["time"].tolist(), dpdt_df["time"].tolist())
        dpdt_min_list, dpdt_min_time_list = finetune_signal(dpdt_min_list, dpdt_min_time_list)

        min_dpdt = pd.DataFrame({
        'time' : dpdt_min_time_list,
        'dpdt_min': dpdt_min_list})

    if lvedp_finetuning:
        lvedp_list, lvedp_time_list = lvedp_finetuner(lvedp["lvedp"].tolist(), lvedp["time"].tolist(), dpdt, lvp_df["lvp"].tolist(), lvp_df["time"].tolist(), resolution=lvedp_res)

        lvedp = pd.DataFrame({
        'time' : lvedp_time_list,
        'lvedp' : lvedp_list
    })

    return max_dpdt, min_dpdt, lvedp, r_peaks_timestamps, lvp_peaks_timestamps, dpdt

##################### MAIN #######################

#create_data_df(folder_path=""C:/Users/saran/Massachusetts Institute of Technology/Maryam Alsharqi - UROP - Sara Nath/VBU_221_2022_01_21/VBU00221Calibrated_LVV"")
#script("VBU_221_2022_01_21/221_AICvTD.csv", "221")

#create_data_df(folder_path="C:/Users/saran/Massachusetts Institute of Technology/Maryam Alsharqi - UROP - Sara Nath/VBU_203_2021_10_05/VBU00203Calibrated_LVV")
#script("VBU_203_2021_10_05/203_TDvAIC.csv", "203")

#create_data_df(folder_path="C:/Users/saran/Massachusetts Institute of Technology/Maryam Alsharqi - UROP - Sara Nath/VBU_205_2021_12_13/VBU00205CalibratedLVV")
#script("VBU_205_2021_12_13/205_TDvAIC.csv", "205")

#create_data_df(folder_path="C:/Users/saran/Massachusetts Institute of Technology/Maryam Alsharqi - UROP - Sara Nath/VBU_202_2021_09_23/VBU00202Calibrated_LVV")
script("VBU_202_2021_09_23/202_TDvAIC.csv", "202")

################## OLD CODE ######################

'''#OUTLIER DETECTION

        #dpdt_max outlier detection
        dpdt_max_mean = max_dpdt["dpdt_max"].mean()
        dpdt_max_std = max_dpdt["dpdt_max"].std()
        upper_max = dpdt_max_mean + 2 * dpdt_max_std
        lower_max = dpdt_max_mean - 2 * dpdt_max_std

        #dpdt_min outlier detection
        dpdt_min_mean = min_dpdt["dpdt_min"].mean()
        dpdt_min_std = min_dpdt["dpdt_min"].std()
        upper_min = dpdt_min_mean + 2 * dpdt_min_std
        lower_min = dpdt_min_mean - 2 * dpdt_min_std

        #lvedp outlier detection
        lvedp_mean = lvedp["lvedp"].mean()
        lvedp_std = lvedp["lvedp"].std()
        upper = lvedp_mean + 2 * lvedp_std
        lower = lvedp_mean - 2 * lvedp_std

        # plots
        anchored_time_data = [datetime.combine(datetime.today(), t) for t in time_data] #for graphing purposes
        anchored_max_timestamps = [datetime.combine(datetime.today(), t) for t in max_timestamps] #for graphing purposes (repeated)
        anchored_rpeak_timestamps = [datetime.combine(datetime.today(), t) for t in r_peaks_timestamps] #for graphing purposes
        anchored_min_timestamps = [datetime.combine(datetime.today(), t) for t in min_dpdt["time"]] #for graphing purposes

        dpdt_max_outliers = []
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for i in range(len(max_dpdt["dpdt_max"])):
            point_x = anchored_max_timestamps[i]
            point_y = max_dpdt["dpdt_max"][i]

            if point_y > upper_max or point_y < lower_max:
                dpdt_max_outliers.append((point_x, point_y))
                ax1.scatter(point_x, point_y, color='blue')
            else:
                ax1.scatter(point_x, point_y, color='black')
            
            plt.title("dpdt_max_"+label)
        time_start_index = anchored_time_data.index(anchored_max_timestamps[0])
        time_end_index = anchored_time_data.index(anchored_max_timestamps[-1])
        ax2.plot(anchored_time_data[time_start_index:time_end_index], lvp_data[time_start_index:time_end_index], color='gray')     
        plt.show()
        print(label+"_dpdt_max: "+str(dpdt_max_outliers))

        dpdt_min_outliers = []
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for i in range(len(min_dpdt["dpdt_min"])):
            point_x = anchored_min_timestamps[i]
            point_y = min_dpdt["dpdt_min"][i]

            if point_y > upper_min or point_y < lower_min:
                dpdt_min_outliers.append((point_x, point_y))
                ax1.scatter(point_x, point_y, color='blue')
            else:
                ax1.scatter(point_x, point_y, color='black')
            
            plt.title("dpdt_min_"+label)
        time_start_index = anchored_time_data.index(anchored_min_timestamps[0])
        time_end_index = anchored_time_data.index(anchored_min_timestamps[-1])
        ax2.plot(anchored_time_data[time_start_index:time_end_index], lvp_data[time_start_index:time_end_index], color='gray')
        plt.show()
        print(label+"_dpdt_min: "+str(dpdt_min_outliers))

        lvedp_outliers = []
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for i in range(len(lvedp["lvedp"])):
            point_x = anchored_rpeak_timestamps[i]
            point_y = lvedp["lvedp"][i]

            if point_y > upper or point_y < lower:
                lvedp_outliers.append((point_x, point_y))
                ax1.scatter(point_x, point_y, color='blue')
            else:
                ax1.scatter(point_x, point_y, color='black')
            
            plt.title("lvedp_"+label)
        time_start_index = anchored_time_data.index(anchored_rpeak_timestamps[0])
        time_end_index = anchored_time_data.index(anchored_rpeak_timestamps[-1])
        ax2.plot(anchored_time_data[time_start_index:time_end_index], lvp_data[time_start_index:time_end_index], color='gray')
        plt.show()
        print(label+"_lvedp: "+str(lvedp_outliers))
        
        print("ONE"+label, len(dpdt_min_cleaned), len(dpdt_min_outliers), len(min_dpdt))
        print("TWO"+label, len(dpdt_max_cleaner), len(dpdt_max_outliers2), len(dpdt_max_cleaned))

        # anchored version of time datasets (only used for graphing)
        anchored_time_data = [datetime.combine(datetime.today(), t) for t in time_data] #for graphing purposes
        anchored_max_timestamps = [datetime.combine(datetime.today(), t) for t in dpdt_max_cleaned["time"]] #for graphing purposes (repeated)
        anchored_max_timestamps_outliers = [datetime.combine(datetime.today(), t) for t in dpdt_max_outliers2["time"]] #for graphing purposes (repeated)
        anchored_min_timestamps_outliers = [datetime.combine(datetime.today(), t) for t in dpdt_min_outliers2["time"]] #for graphing purposes (repeated)
        anchored_lvedp_timestamps_outliers = [datetime.combine(datetime.today(), t) for t in lvedp_outliers2["time"]] #for graphing purposes (repeated)
        anchored_rpeak_timestamps = [datetime.combine(datetime.today(), t) for t in lvedp_cleaned["time"]] #for graphing purposes
        anchored_min_timestamps = [datetime.combine(datetime.today(), t) for t in dpdt_min_cleaned["time"]] #for graphing purposes

        # graph dpdt_max w outliers and lvp:
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.scatter(anchored_max_timestamps, dpdt_max_cleaned["dpdt_max"], color='black')
        if len(dpdt_max_outliers2) > 0:
            ax1.scatter(anchored_max_timestamps_outliers, dpdt_max_outliers2["data"], color='blue')
        time_start_index = anchored_time_data.index(anchored_max_timestamps[0])
        time_end_index = anchored_time_data.index(anchored_max_timestamps[-1])
        ax2.plot(anchored_time_data[time_start_index:time_end_index], lvp_data[time_start_index:time_end_index], color='gray')
        plt.title("dpdt_max_"+label)     
        plt.show()

        # graph dpdt_min w outliers and lvp:
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.scatter(anchored_min_timestamps, dpdt_min_cleaned["dpdt_min"], color='black')
        if len(dpdt_min_outliers2) > 0:
            ax1.scatter(anchored_min_timestamps_outliers, dpdt_min_outliers2["data"], color='blue')
        time_start_index = anchored_time_data.index(anchored_min_timestamps[0])
        time_end_index = anchored_time_data.index(anchored_min_timestamps[-1])
        ax2.plot(anchored_time_data[time_start_index:time_end_index], lvp_data[time_start_index:time_end_index], color='gray')
        plt.title("dpdt_min_"+label)     
        plt.show()

        # graph lvedp w outliers and lvp:
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.scatter(anchored_rpeak_timestamps, lvedp_cleaned["lvedp"], color='black')
        if len(lvedp_outliers2) > 0:
            ax1.scatter(anchored_lvedp_timestamps_outliers, lvedp_outliers2["data"], color='blue')
        time_start_index = anchored_time_data.index(anchored_rpeak_timestamps[0])
        time_end_index = anchored_time_data.index(anchored_rpeak_timestamps[-1])
        ax2.plot(anchored_time_data[time_start_index:time_end_index], lvp_data[time_start_index:time_end_index], color='gray')
        plt.title("lvedp_"+label)     
        plt.show()

        #sanity check! plot lvedp on top of lvp
        plt.plot(anchored_time_data, lvp_data, color='black')
        plt.scatter(anchored_rpeak_timestamps, lvedp_data, color='blue')
        plt.title(label)
        plt.show()

        '''
