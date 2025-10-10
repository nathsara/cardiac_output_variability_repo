import pickle
import pandas as pd
from datetime import datetime, time, timedelta
from ecgdetectors import Detectors
from scipy.signal import find_peaks
import plots

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

# GENERAL PIPELINE (SINGLE PHASE INPUT)
def pipeline(label, plot=True):

    animal = label[:3]
    f1 = open(f"{animal}_data/{label}_ecg_raw.pkl", 'rb')
    f2 = open(f"{animal}_data/{label}_lvp_raw.pkl", 'rb')
    ecg_df = pickle.load(f1)
    lvp_df = pickle.load(f2)
    
    ecg_df, lvp_df = arrythmia_removal(label, ecg_df, lvp_df)

    if label in ["203_Nitro_high_P3", "203_Nitro_low_P3", "202_Phen_0_P3", "202_Washout_1_P3"]:
        max_dpdt, min_dpdt, lvedp, r_peaks_timestamps, lvp_peaks_timestamps, dpdt = extra_processing_pipeline(ecg_df, lvp_df, lvedp_finetuning=True)
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
            lvedp_data.append(lvp_df['lvp'][ref_idx])

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
                 'dpdt': dpdt}).to_pickle(f"{animal}_data/dpdt_{label}.pkl")
    max_dpdt.to_pickle(f"{animal}_data/dpdt_max_{label}.pkl")
    min_dpdt.to_pickle(f"{animal}_data/dpdt_min_{label}.pkl")
    lvedp.to_pickle(f"{animal}_data/lvedp_{label}.pkl")

    #outlier detection
    dpdt_max_outliers = detect_outliers(max_dpdt["time"], max_dpdt["dpdt_max"], label+"dpdt_max1")
    dpdt_min_outliers = detect_outliers(min_dpdt["time"], min_dpdt["dpdt_min"], label+"dpdt_min1")
    lvedp_outliers = detect_outliers(lvedp["time"], lvedp["lvedp"], label+"lvedp1")

    #outlier cleaning -- 1
    dpdt_max_cleaned = remove_outliers(max_dpdt, dpdt_max_outliers)
    dpdt_min_cleaned = remove_outliers(min_dpdt, dpdt_min_outliers)
    lvedp_cleaned = remove_outliers(lvedp, lvedp_outliers)

    # save files after first round of outlier extraction:
    dpdt_max_cleaned.to_pickle(f"{animal}_data/dpdt_max_{label}_cleaned.pkl")
    dpdt_min_cleaned.to_pickle(f"{animal}_data/dpdt_min_{label}_cleaned.pkl")
    lvedp_cleaned.to_pickle(f"{animal}_data/lvedp_{label}_cleaned.pkl")

    #outlier detection -- 2
    dpdt_max_outliers2 = detect_outliers(dpdt_max_cleaned["time"], dpdt_max_cleaned["dpdt_max"], label+"dpdt_max2")
    dpdt_min_outliers2 = detect_outliers(dpdt_min_cleaned["time"], dpdt_min_cleaned["dpdt_min"], label+"dpdt_min2")
    lvedp_outliers2 = detect_outliers(lvedp_cleaned['time'], lvedp_cleaned["lvedp"], label+"lvedp2")

    #outlier cleaning -- 2
    dpdt_max_cleaner = remove_outliers(dpdt_max_cleaned, dpdt_max_outliers2)
    dpdt_min_cleaner = remove_outliers(dpdt_min_cleaned, dpdt_min_outliers2)
    lvedp_cleaner = remove_outliers(lvedp_cleaned, lvedp_outliers2)

    # save files after second round of outlier extraction:
    dpdt_max_cleaner.to_pickle(f"{animal}_data/dpdt_max_{label}_cleaner.pkl")
    dpdt_min_cleaner.to_pickle(f"{animal}_data/dpdt_min_{label}_cleaner.pkl")
    lvedp_cleaner.to_pickle(f"{animal}_data/lvedp_{label}_cleaner.pkl")

    if plot:
        plots.phase_data_plot(ecg_df, lvp_df, r_peaks_timestamps, dpdt_max_cleaner, dpdt_min_cleaner, lvedp_cleaner, dpdt, label)

    #compute mean and std of each file and add to running list (defined earlier):
    dpdt_max_mean = dpdt_max_cleaner["dpdt_max"].mean()
    dpdt_max_std = dpdt_max_cleaner["dpdt_max"].std()

    dpdt_min_mean = dpdt_min_cleaner["dpdt_min"].mean()
    dpdt_min_std = dpdt_min_cleaner["dpdt_min"].std()

    lvedp_mean = lvedp_cleaner["lvedp"].mean()
    lvedp_std = lvedp_cleaner["lvedp"].std()

    return dpdt_max_mean, dpdt_max_std, dpdt_min_mean, dpdt_min_std, lvedp_mean, lvedp_std

# PIPELINE FOR ADDITIONAL PROCESSING (SINGLE PHASE INPUT)
def extra_processing_pipeline(ecg_df, lvp_df, lvedp_finetuning=True, dpdt_finetuning=True, dpdt_res=15, lvedp_res=15):
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

# SUMMARY DATA PER ANIMAL (SINGLE ANIMAL INPUT)
def combined_phase_data(animal, plot=True):

    print(f"Starting {animal}...")
    if (animal==221):
        phases = ["221_Baseline_0_P6", "221_Baseline_0_P3", "221_Nitro_low_P6", "221_Nitro_low_P3", "221_Nitro_high_P6", "221_Nitro_high_P3",
                  "221_Washout_0_P6", "221_Washout_0_P3", "221_Phen_low_P6", "221_Phen_low_P3", "221_Phen_high_P6", "221_Phen_high_P3",
                  "221_Washout_1_P6", "221_Washout_1_P3", "221_Dobu_low_P6", "221_Dobu_low_P3"]
    elif (animal==205):
        phases = ["205_Baseline_0_P6", "205_Baseline_0_P3", "205_Nitro_low_P6", "205_Nitro_low_P3", "205_Nitro_high_P6", "205_Nitro_high_P3",
                  "205_Washout_1_P6", "205_Washout_1_P3", "205_Phen_low_P6", "205_Phen_low_P3", "205_Phen_high_P6", "205_Phen_high_P3",
                  "205_Washout_2_P6", "205_Washout_2_P3", "205_Dobu_low_P6", "205_Dobu_low_P3", "205_Dobu_high_P6"]
    elif (animal == 203):
        phases = ["203_Baseline_0_P6", "203_Baseline_0_P3", "203_Nitro_low_P6", "203_Nitro_low_P3", "203_Nitro_high_P6", "203_Nitro_high_P3",
                  "203_Washout_0_P6", "203_Washout_0_P3", "203_Phen_low_P6", "203_Phen_low_P3", "203_Phen_high_P6", "203_Phen_high_P3",
                  "203_Washout_1_P6", "203_Washout_1_P3", "203_Dobu_low_P6", "203_Dobu_low_P3", "203_Dobu_high_P6", "203_Dobu_high_P3", 
                  "203_Washout_2_P6", "203_Washout_2_P3", "203_Esmo_low_P6", "203_Esmo_low_P3"]
    elif (animal == 202):
        phases = ["202_Baseline_0_P6", "202_Baseline_0_P3", "202_Nitro_low_P6", "202_Nitro_low_P3", "202_Nitro_high_P6", "202_Nitro_high_P3",
                  "202_Washout_0_P6", "202_Washout_0_P3", "202_Phen_0_P6", "202_Phen_0_P3",
                  "202_Washout_1_P6", "202_Washout_1_P3", "202_Dobu_0_P6", "202_Dobu_0_P3",
                  "202_Washout_2_P6", "202_Washout_2_P3", "202_Esmo_0_P6", "202_Esmo_0_P3"]
    dpdt_max_means = []
    dpdt_max_stds = []
    dpdt_min_means = []
    dpdt_min_stds = []
    lvedp_means = []
    lvedp_stds = []
        
    for phase in phases:
        print(f"Starting on {phase}...")
        dpdt_max_mean, dpdt_max_std, dpdt_min_mean, dpdt_min_std, lvedp_mean, lvedp_std = pipeline(phase, plot=plot)

        dpdt_max_means.append(dpdt_max_mean)
        dpdt_max_stds.append(dpdt_max_std)
        dpdt_min_means.append(dpdt_min_mean)
        dpdt_min_stds.append(dpdt_min_std)
        lvedp_means.append(lvedp_mean)
        lvedp_stds.append(lvedp_std)
        print(f"Finished {phase}")

    dpdt_max_means_df = pd.DataFrame(dpdt_max_means, columns=["dpdt_max_mean"])
    dpdt_min_means_df = pd.DataFrame(dpdt_min_means, columns=["dpdt_min_mean"])
    lvedp_means_df = pd.DataFrame(lvedp_means, columns=["lvedp_mean"])

    dpdt_max_stds_df = pd.DataFrame(dpdt_max_stds, columns=["dpdt_max_std"])
    dpdt_min_stds_df = pd.DataFrame(dpdt_min_stds, columns=["dpdt_min_std"])
    lvedp_stds_df = pd.DataFrame(lvedp_stds, columns=["lvedp_std"])

    dpdt_max_means_df.to_pickle(f"{animal}_data/dpdt_max_means_{animal}.pkl")
    dpdt_min_means_df.to_pickle(f"{animal}_data/dpdt_min_means_{animal}.pkl")
    lvedp_means_df.to_pickle(f"{animal}_data/lvedp_means_{animal}.pkl")

    dpdt_max_stds_df.to_pickle(f"{animal}_data/dpdt_max_stds_{animal}.pkl")
    dpdt_min_stds_df.to_pickle(f"{animal}_data/dpdt_min_stds_{animal}.pkl")
    lvedp_stds_df.to_pickle(f"{animal}_data/lvedp_stds_{animal}.pkl")
    print(f"Finished with {animal}. Now generating summary plots...")

    if (animal==221):
        plots.generate_221_plot(dpdt_max_means, dpdt_max_stds, animal, "dp/dt max")
        plots.generate_221_plot(dpdt_min_means, dpdt_min_stds, animal, "dp/dt min")
        plots.generate_221_plot(lvedp_means, lvedp_stds, animal, "lvedp")
    elif (animal == 205):
        plots.generate_205_plot(dpdt_max_means, dpdt_max_stds, animal, "dp/dt max")
        plots.generate_205_plot(dpdt_min_means, dpdt_min_stds, animal, "dp/dt min")
        plots.generate_205_plot(lvedp_means, lvedp_stds, animal, "lvedp")
    elif (animal==203):
        plots.generate_203_plot(dpdt_max_means, dpdt_max_stds, animal, "dp/dt max")
        plots.generate_203_plot(dpdt_min_means, dpdt_min_stds, animal, "dp/dt min")
        plots.generate_203_plot(lvedp_means, lvedp_stds, animal, "lvedp")
    elif (animal==202):
        plots.generate_202_plot(dpdt_max_means, dpdt_max_stds, animal, "dp/dt max")
        plots.generate_202_plot(dpdt_min_means, dpdt_min_stds, animal, "dp/dt min")
        plots.generate_202_plot(lvedp_means, lvedp_stds, animal, "lvedp")

### HELPER FUNCTIONS
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
        print("BREAK")
        print("INITIAL: ", real_lvedp)
        if real_lvedp > 25:
            for j in range(index_on_lvp-resolution, index_on_lvp+resolution+1):
                print(j, lvp[j])
                #if dpdt[j] < corr_dpdt and dpdt[j] > 0 and lvp[j] <= 30:
                if lvp[j] <= 30:
                    try:
                        print("ENTERED")
                        corr_dpdt = dpdt[j]
                        real_lvedp = lvp[j]
                        real_lvedp_timestamp = lvp_timestamps[j]
                    except KeyError:
                        j += 1
            print("DECIDED: ", real_lvedp)
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