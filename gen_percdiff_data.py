import pandas as pd
import pickle

def gen_percdiffs(file, file_name, type="AIC"):
    
    f = open(file, "rb")
    df = pickle.load(f)

    if type=="AIC":
        ls = list(df["CO_trend"])
    else:
        ls = list(df["TD"])

    percdiff_ls = []
    
    for i in range(len(ls)):
        if i % 2 == 0:
            percdiff_ls.append((ls[i] - ls[0])/ls[0] * 100)
        else:
            percdiff_ls.append((ls[i] - ls[1])/ls[1] * 100)

    df = pd.DataFrame(percdiff_ls)

    df.to_pickle(f'{file_name}_percdiff_incl_dobu.pkl')

gen_percdiffs("AIC_mean_DF202.pkl", "aic202", type="AIC")
gen_percdiffs("AIC_mean_DF203.pkl", "aic203", type="AIC")
gen_percdiffs("TD_mean_DF202.pkl", "td202", type="TD")
gen_percdiffs("TD_mean_DF203.pkl", "td203", type="TD")
