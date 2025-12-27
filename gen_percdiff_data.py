import pandas as pd
import pickle

def gen_percdiffs(file, file_name, type="AIC"):
    
    f = open(file, "rb")
    df = pickle.load(f)

    if type=="AIC":
        ls = list(df["CO_trend"])
    elif type=="TD":
        ls = list(df["TD"])
    elif type=="dpdtmax":
        ls = list(df["dpdt_max_mean"])
    elif type=="dpdtmin":
        ls = list(df["dpdt_min_mean"])
    elif type=="lvedp":
        ls = list(df["lvedp_mean"])

    percdiff_ls = []
    
    for i in range(len(ls)):
        if i % 2 == 0:
            percdiff_ls.append((ls[i] - ls[0])/ls[0] * 100)
        else:
            percdiff_ls.append((ls[i] - ls[1])/ls[1] * 100)

    df = pd.DataFrame(percdiff_ls)

    df.to_pickle(f'{file_name}_percdiff_full.pkl')

'''gen_percdiffs("AIC_mean_DF202.pkl", "aic202", type="AIC")
gen_percdiffs("AIC_mean_DF203.pkl", "aic203", type="AIC")
gen_percdiffs("TD_mean_DF202.pkl", "td202", type="TD")
gen_percdiffs("TD_mean_DF203.pkl", "td203", type="TD")'''

gen_percdiffs("202_data/dpdt_max_means_202.pkl", "dpdt_max_202", type="dpdtmax")
gen_percdiffs("202_data/dpdt_min_means_202.pkl", "dpdt_min_202", type="dpdtmin")
gen_percdiffs("202_data/lvedp_means_202.pkl", "lvedp_202", type="lvedp")

gen_percdiffs("203_data/dpdt_max_means_203.pkl", "dpdt_max_203", type="dpdtmax")
gen_percdiffs("203_data/dpdt_min_means_203.pkl", "dpdt_min_203", type="dpdtmin")
gen_percdiffs("203_data/lvedp_means_203.pkl", "lvedp_203", type="lvedp")

gen_percdiffs("205_data/dpdt_max_means_205.pkl", "dpdt_max_205", type="dpdtmax")
gen_percdiffs("205_data/dpdt_min_means_205.pkl", "dpdt_min_205", type="dpdtmin")
gen_percdiffs("205_data/lvedp_means_205.pkl", "lvedp_205", type="lvedp")

gen_percdiffs("221_data/dpdt_max_means_221.pkl", "dpdt_max_221", type="dpdtmax")
gen_percdiffs("221_data/dpdt_min_means_221.pkl", "dpdt_min_221", type="dpdtmin")
gen_percdiffs("221_data/lvedp_means_221.pkl", "lvedp_221", type="lvedp")
