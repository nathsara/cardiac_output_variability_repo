import data_init
import process
import plots

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



#RUN A SINGLE PHASE:
#process.pipeline("202_Phen_0_P3")

# RUN A SINGLE ANIMAL: (plot=False) means a plot won't be generated for each phase for the animal; the summary plots containing ALL phase data are always generated.
#process.combined_phase_data(202, plot=False)

# RUN EVERYTHING
#process.combined_phase_data(221, plot=False)
#process.combined_phase_data(205, plot=False)
#process.combined_phase_data(203, plot=False)
#process.combined_phase_data(202, plot=False)