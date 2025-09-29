import os
import pickle
import pandas as pd
from datetime import datetime, time, timedelta
import data_init

'''
RUN THE BELOW SECTION ONLY ONCE:
'''
#221
#data_init.create_data_df("VBU_221_2022_01_21/VBU00221Calibrated_LVV", "221")
data_init.create_raw_phase_data("VBU_221_2022_01_21/221_AICvTD.csv", "raw_hd_data_221.pkl", "221")