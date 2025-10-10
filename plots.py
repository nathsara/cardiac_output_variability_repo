import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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

    # DATA SPECIFIC PLOT MODIFICATIONS + GRAPHING OF DATA

    y_low = min([means[i] - stds[i] for i in range(len(means))])
    y_high = max([means[i] + stds[i] for i in range(len(means))])

    plt.ylim(round(y_low-1), round(y_high+1))
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(list(range(0, 16)))

    plt.text(0.5, y_low-1, state_labels[0], horizontalalignment='center', verticalalignment='top', fontsize=9)
    plt.text(3.5, y_low-1, state_labels[1], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(6.5, y_low-1, state_labels[2], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(9.5, y_low-1, state_labels[3], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(12.5, y_low-1, state_labels[4], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(15, y_low-1, state_labels[5], horizontalalignment='center', verticalalignment='top',  fontsize=9)

    xaxis = list(range(len(means)))

    p3_mean = [mean for mean in means if means.index(mean) % 2 == 1]
    p6_mean = [mean for mean in means if means.index(mean) % 2 == 0]
    p3_std = [std for std in stds if stds.index(std) % 2 == 1]
    p6_std = [std for std in stds if stds.index(std) % 2 == 0]
    xaxis_p3 = [x for x in xaxis if xaxis.index(x) % 2 == 1]
    xaxis_p6 = [x for x in xaxis if xaxis.index(x) % 2 == 0]

    plt.errorbar(xaxis_p6, p6_mean, yerr=p6_std, marker='o', capsize=5, linestyle="None", linewidth=1, label='P6')
    plt.errorbar(xaxis_p3, p3_mean, yerr=p3_std, marker='o', capsize=5, linestyle="None", linewidth=1, color='green', label='P3')

    plt.title(f"{metric} (Subject {subject})")
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

    # DATA SPECIFIC PLOT MODIFICATIONS + GRAPHING OF DATA

    y_low = min([means[i] - stds[i] for i in range(len(means))])
    y_high = max([means[i] + stds[i] for i in range(len(means))])

    plt.ylim(round(y_low-1), round(y_high+1))
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(list(range(0, 22)))

    plt.text(0.5, y_low-1, state_labels[0], horizontalalignment='center', verticalalignment='top', fontsize=9)
    plt.text(3.5, y_low-1, state_labels[1], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(6.5, y_low-1, state_labels[2], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(9.5, y_low-1, state_labels[3], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(12.5, y_low-1, state_labels[4], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(15.5, y_low-1, state_labels[5], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(18.5, y_low-1, state_labels[6], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(20.5, y_low-1, state_labels[7], horizontalalignment='center', verticalalignment='top',  fontsize=9)

    xaxis = list(range(len(means)))

    p3_mean = [mean for mean in means if means.index(mean) % 2 == 1]
    p6_mean = [mean for mean in means if means.index(mean) % 2 == 0]
    p3_std = [std for std in stds if stds.index(std) % 2 == 1]
    p6_std = [std for std in stds if stds.index(std) % 2 == 0]
    xaxis_p3 = [x for x in xaxis if xaxis.index(x) % 2 == 1]
    xaxis_p6 = [x for x in xaxis if xaxis.index(x) % 2 == 0]

    plt.errorbar(xaxis_p6, p6_mean, yerr=p6_std, marker='o', capsize=5, linestyle="None", linewidth=1, label='P6')
    plt.errorbar(xaxis_p3, p3_mean, yerr=p3_std, marker='o', capsize=5, linestyle="None", linewidth=1, color='green', label='P3')

    plt.title(f"{metric} (Subject {subject})")
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

    # DATA SPECIFIC PLOT MODIFICATIONS + GRAPHING OF DATA

    y_low = min([means[i] - stds[i] for i in range(len(means))])
    y_high = max([means[i] + stds[i] for i in range(len(means))])

    plt.ylim(round(y_low-1), round(y_high+1))
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(list(range(0, 17)))

    plt.text(0.5, y_low-1, state_labels[0], horizontalalignment='center', verticalalignment='top', fontsize=9)
    plt.text(3.5, y_low-1, state_labels[1], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(6.5, y_low-1, state_labels[2], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(9.5, y_low-1, state_labels[3], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(12.5, y_low-1, state_labels[4], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(15.5, y_low-1, state_labels[5], horizontalalignment='center', verticalalignment='top',  fontsize=9)

    xaxis = list(range(len(means)))

    p3_mean = [mean for mean in means if means.index(mean) % 2 == 1]
    p6_mean = [mean for mean in means if means.index(mean) % 2 == 0]
    p3_std = [std for std in stds if stds.index(std) % 2 == 1]
    p6_std = [std for std in stds if stds.index(std) % 2 == 0]
    xaxis_p3 = [x for x in xaxis if xaxis.index(x) % 2 == 1]
    xaxis_p6 = [x for x in xaxis if xaxis.index(x) % 2 == 0]

    plt.errorbar(xaxis_p6, p6_mean, yerr=p6_std, marker='o', capsize=5, linestyle="None", linewidth=1, label='P6')
    plt.errorbar(xaxis_p3, p3_mean, yerr=p3_std, marker='o', capsize=5, linestyle="None", linewidth=1, color='green', label='P3')

    plt.title(f"{metric} (Subject {subject})")
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

    # DATA SPECIFIC PLOT MODIFICATIONS + GRAPHING OF DATA

    y_low = min([means[i] - stds[i] for i in range(len(means))])
    y_high = max([means[i] + stds[i] for i in range(len(means))])

    plt.ylim(round(y_low-1), round(y_high+1))
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(list(range(0, 18)))

    plt.text(0.5, y_low-1, state_labels[0], horizontalalignment='center', verticalalignment='top', fontsize=9)
    plt.text(3.5, y_low-1, state_labels[1], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(6.5, y_low-1, state_labels[2], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(8.5, y_low-1, state_labels[3], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(10.5, y_low-1, state_labels[4], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(12.5, y_low-1, state_labels[5], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(14.5, y_low-1, state_labels[6], horizontalalignment='center', verticalalignment='top',  fontsize=9)
    plt.text(16.5, y_low-1, state_labels[7], horizontalalignment='center', verticalalignment='top',  fontsize=9)

    xaxis = list(range(len(means)))

    p3_mean = [mean for mean in means if means.index(mean) % 2 == 1]
    p6_mean = [mean for mean in means if means.index(mean) % 2 == 0]
    p3_std = [std for std in stds if stds.index(std) % 2 == 1]
    p6_std = [std for std in stds if stds.index(std) % 2 == 0]
    xaxis_p3 = [x for x in xaxis if xaxis.index(x) % 2 == 1]
    xaxis_p6 = [x for x in xaxis if xaxis.index(x) % 2 == 0]

    plt.errorbar(xaxis_p6, p6_mean, yerr=p6_std, marker='o', capsize=5, linestyle="None", linewidth=1, label='P6')
    plt.errorbar(xaxis_p3, p3_mean, yerr=p3_std, marker='o', capsize=5, linestyle="None", linewidth=1, color='green', label='P3')

    plt.title(f"{metric} (Subject {subject})")
    plt.ylabel('mmHG/s')
    plt.legend()
    plt.grid(axis="x")
    plt.show()

def phase_data_plot(ecg_df, lvp_df, r_peaks_timestamps_list, dpdt_max_df, dpdt_min_df, lvedp_df, dpdt_list, label):
    #x-axis must be a DateTime object, not a DateTime.time object, so we anchor all of our DateTime.time objects to a date.
    anchored_time_data = [datetime.combine(datetime.today(), t) for t in lvp_df['time']]
    anchored_rpeak_timestamps = [datetime.combine(datetime.today(), t) for t in r_peaks_timestamps_list]
    anchored_mdpdt_timestamps = [datetime.combine(datetime.today(), t) for t in dpdt_max_df["time"]]
    anchored_mindpdt_timestamps = [datetime.combine(datetime.today(), t) for t in dpdt_min_df["time"]]
    anchored_lvedp_timestamps = [datetime.combine(datetime.today(), t) for t in lvedp_df['time']]

    _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    time_start_index = anchored_time_data.index(anchored_rpeak_timestamps[0])
    time_end_index = anchored_time_data.index(anchored_rpeak_timestamps[-1])
    ax1.plot(anchored_time_data[time_start_index:time_end_index], ecg_df["ecg"][time_start_index:time_end_index], color='gray')
    for rpeak in anchored_rpeak_timestamps:
        ax1.axvline(x=rpeak, color='r', linestyle='--')

    ax2.plot(anchored_time_data[time_start_index:time_end_index], dpdt_list[time_start_index:time_end_index], color='gray')
    ax2.scatter(anchored_mdpdt_timestamps, dpdt_max_df["dpdt_max"], color="blue")
    ax2.scatter(anchored_mindpdt_timestamps, dpdt_min_df["dpdt_min"], color="green")

    ax3.plot(anchored_time_data[time_start_index:time_end_index], lvp_df["lvp"][time_start_index:time_end_index], color='gray')
    ax3.scatter(anchored_lvedp_timestamps, lvedp_df["lvedp"], color='black')
        
    plt.title("ecg/rpeaks_dpdt-max/dpdt_lvedp/lvp_"+label)
    plt.show()