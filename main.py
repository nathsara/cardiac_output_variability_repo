import os
import pickle
import pandas as pd
from datetime import datetime, time, timedelta
import data_init

'''
RUN THE BELOW SECTION ONLY ONCE:
May need to update the create_data_df function for each subject because the ECG and LVP are
not at the same index for each subject.
'''
#221
#data_init.create_data_df("VBU_221_2022_01_21/VBU00221Calibrated_LVV", "221")
#data_init.create_raw_phase_data("VBU_221_2022_01_21/221_AICvTD.csv", "raw_hd_data_221.pkl", "221")

#202
#data_init.create_data_df("VBU_202_2021_09_23/VBU00202Calibrated_LVV", "202")
#data_init.create_raw_phase_data("VBU_202_2021_09_23/202_TDvAIC.csv", "raw_hd_data_202.pkl", "202")

#203
#data_init.create_data_df("VBU_203_2021_10_05/VBU00203Calibrated_LVV", "203")
#data_init.create_raw_phase_data("VBU_203_2021_10_05/203_TDvAIC.csv", "raw_hd_data_203.pkl", "203")

#205
#data_init.create_data_df("VBU_205_2021_12_13/VBU00205CalibratedLVV", "205")
#data_init.create_raw_phase_data("VBU_205_2021_12_13/205_TDvAIC.csv", "raw_hd_data_205.pkl", "205")