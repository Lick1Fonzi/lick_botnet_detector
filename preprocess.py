import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def save_to_csvfile(data,csv_name):
    save_dir = "./data/raw/"
    data.to_csv(save_dir + csv_name,index=None)  

# https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/detailed-bidirectional-flow-labels/capture20110810.binetflow 
data = pd.read_csv("data/rbot.binetflow")

#print("Dropping unused columns...")
data = data.drop(columns=["StartTime"])
data.drop(data[data.Proto != 'tcp'].index, inplace=True)
data['Proto'].replace({'tcp': 0}, inplace=True)
data = data.drop(columns=["Proto"])

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data['sTos'].replace(np.nan, 0, inplace=True)
data['dTos'].replace(np.nan, 0, inplace=True)

# Drop samples with NaN values
data = data.dropna(how='any', axis=0)

# Add Derived Features
#print("Computing derived features...")
data['BytesPerPkt'] = data['TotBytes'] / data['TotPkts']

data['PktsPerSec'] = data['TotPkts'] / data['Dur']
max_value = data.loc[data['PktsPerSec'] != np.inf, 'PktsPerSec'].max()
# #print(max_value)
data['PktsPerSec'].replace(np.inf, max_value, inplace=True)

data['RatioOutIn'] = (data['TotBytes'] - data['SrcBytes']) / data['SrcBytes']
max_value = data.loc[data['RatioOutIn'] != np.inf, 'RatioOutIn'].max()
# #print(max_value)
data['RatioOutIn'].replace(np.inf, max_value, inplace=True)

data['DstBytes'] = data['TotBytes'] - data['SrcBytes']

"""     # 95-th Percentile threshold
#print("Filtering outliers...")
data.drop(data[data.Dur > 300].index, inplace=True)
data.drop(data[data.SrcBytes >= 60000].index, inplace=True)
data.drop(data[data.TotPkts >= 100].index, inplace=True)
data.drop(data[data.PktsPerSec >= 10000].index, inplace=True) 
"""

# Encoding Src/Dst ports
#print("Encoding ports...")
data.Sport = data.Sport.astype(np.int64)
data.Dport = data.Dport.astype(np.int64)

data['SrcPort'] = np.where(data.Sport.between(0, 1023, inclusive='both'),
                    'SrcPortWellKnown',
                    np.where(data.Sport.between(1024, 49151, inclusive='both'),
                            'SrcPortRegistered', 'SrcPortPrivate'))

data['DstPort'] = np.where(data.Dport.between(0, 1023, inclusive='both'),
                    'DstPortWellKnown',
                    np.where(data.Dport.between(1024, 49151, inclusive='both'),
                            'DstPortRegistered', 'DstPortPrivate'))

data = pd.get_dummies(data, columns=["SrcPort", "DstPort"], prefix="", prefix_sep="")

# One Hot Encoding
new_cols = pd.get_dummies(data.Dir, prefix='Dir')
data[new_cols.columns] = new_cols
data = data.drop('Dir', axis=1)

#print("One hot encoding states")
state_flags = ['P', 'A', 'S', 'C', 'F',  'R', 'E','U']  # state 'U' seems all nulls
data.loc[:, "State_Split"] = data.State.str.split("_")
state_l = data["State_Split"].to_list()
src_state = []
dst_state = []
for i in state_l:
    src_state.append(i[0])
    dst_state.append(i[1])
    
data.loc[:, "SrcState"] = src_state
data.loc[:, "DstState"] = dst_state

for s in state_flags:
    data["SrcState_" + s] = data["SrcState"].apply(lambda x: 1 if s in x else 0)
    data["DstState_" + s] = data["DstState"].apply(lambda x: 1 if s in x else 0)
#Riempio valori che vengono fuori null, ovvero DstState_E e entrambi U
data = data.fillna(0)  
data = data.drop(columns=["State", "State_Split", "SrcState", "DstState"])

#print("Encoding explicit IPAddresses...")
data['IPSrcType'] = np.where(data.SrcAddr.str.startswith("147.32."), 1, 0)
data['IPDstType'] = np.where(data.DstAddr.str.startswith("147.32."), 1, 0)

data = data.drop(columns=["IPSrcType"])
data = data.drop(columns=["IPDstType"])

# Concatenate IP Addr and Port data for graph nodes
#print("Generating Node identifiers...")
data['Src IP'] = data['SrcAddr'].astype(str) + ":" + data["Sport"].astype(str)
data['Dst IP'] = data['DstAddr'].astype(str) + ":" + data["Dport"].astype(str)
data.drop(columns=["Sport", "Dport", "SrcAddr", "DstAddr"], inplace=True)

label = np.where(data.Label.str.contains("Botnet"), 1, 0)
data = data.drop(columns = ["Label"])
data["Label"] = label



'''
FINE PREPROCESSING, SALVO DATASET ORIGINALE PREPROCESSATO IN VARIABILE data
'''



field_to_ignore = ["Src IP","Dst IP"]
#data=data.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x, axis=0)
COL_ORDER = data.columns.values.tolist()
#minmax scaler scikit per normalizzare, escludo campo SRC IP
norm_data = data.copy().drop(columns=field_to_ignore)

scaler = MinMaxScaler()
scaler.fit(norm_data)

data_nonadv = pd.DataFrame(scaler.transform(norm_data),columns=[x for x in COL_ORDER if x not in field_to_ignore])
data_nonadv['Src IP'] = data["Src IP"]
data_nonadv["Dst IP"] = data["Dst IP"]
data_nonadv = data_nonadv[COL_ORDER]
data_nonadv = data_nonadv.dropna(axis=0, subset=field_to_ignore)


#print(len(data))
ben_data = data_nonadv[data_nonadv.Label == 0]
mal_data = data_nonadv[data_nonadv.Label == 1]

if len(mal_data) > 20000:
    mal_data = mal_data.sample(20000)

#print("Generating 20:1 ben:mal dataset")
if len(ben_data) >= 20*len(mal_data):
    ben_data = ben_data.sample(20*len(mal_data))
else:
    mal_data = mal_data.sample(len(ben_data)//20)
data_nonadv = pd.concat([ben_data,mal_data])
print("Saving Preprocessed Regulare csv...")
save_to_csvfile(data_nonadv,"Rbot.csv")


'''
INIZIO GENERAZIONE CTU ADVERSARIAL CON PERTURBAZIONI
'''

groups = {
    '1a': ['Dur'],
    '1b': ['SrcBytes'],
    '1c': ['DstBytes'],
    '1d': ['TotPkts'],
    '2a': ['Dur','SrcBytes'],
    '2b': ['Dur','DstBytes'],
    '2c': ['Dur','TotPkts'],
    '2e': ['SrcBytes','TotPkts'],
    '2d': ['SrcBytes','DstBytes'],
    '2f': ['DstBytes','TotPkts'],
    '3a': ['Dur','SrcBytes','DstBytes'],
    '3b': ['Dur','SrcBytes','TotPkts'],
    '3c': ['Dur','DstBytes','TotPkts'],
    '3d': ['SrcBytes','DstBytes','TotPkts'],
    '4a': ['Dur','SrcBytes','DstBytes','TotPkts']
}

group_step = {
    'Dur' : 0,
    'SrcBytes' : 1,
    'DstBytes' : 2,
    'TotPkts' : 3,
}

steps = {
    '1': [1,1,1,1],
    '2': [2,2,2,2],
    '3': [5,8,8,5],
    '4': [10,16,16,10],
    '5': [15,64,64,15],
    '6': [30,128,128,20],
    '7': [45,256,256,30],
    '8': [60,512,512,50],
    '9': [120,1024,1024,100]
}
GEN_ADVERSARIAL = 0
if(GEN_ADVERSARIAL):
    for step in steps:
        for group in groups:
            df = data.copy()
            df['DstBytes'] = df['TotBytes'].astype(np.float64) - df['SrcBytes'].astype(np.float64)

            bot_label = 1.0
            mal_data = df[df["Label"] == bot_label]
            ben_df = df[df["Label"] != bot_label]
            df = mal_data

            for field in groups[group]:
                print("Changing field {} at step {} of value {}".format(field,step,steps[step]))
                df[field] = df[field].astype(np.float64) + (np.float64(steps[step][group_step[field]])).astype(np.float64)

            df['BytesPerPkt'] = df['TotBytes'] / df['TotPkts']
            df['PktsPerSec'] = df['TotPkts'] / df['Dur']
            max_value = df.loc[df['PktsPerSec'] != np.inf, 'PktsPerSec'].max()
            # #print(max_value)
            df['PktsPerSec'].replace(np.inf, max_value, inplace=True)
            df['RatioOutIn'] = (df['TotBytes'] - df['SrcBytes']) / df['SrcBytes']
            max_value = df.loc[df['RatioOutIn'] != np.inf, 'RatioOutIn'].max()
            # #print(max_value)
            df['RatioOutIn'].replace(np.inf, max_value, inplace=True)
            df['DstBytes'] = df['TotBytes'] - df['SrcBytes']

            if len(df) > 20000:
                df = df.sample(20000)

            #print("Generating 20:1 ben:mal dataset")
            if len(ben_data) >= 20*len(df):
                ben_data = ben_data.sample(20*len(df))
            else:
                df = df.sample(len(ben_data)//20)
            
            df = pd.concat([ben_df,df])
            if COL_ORDER != None:
                df = df[COL_ORDER]

            d = df.copy()
            d = d.drop(columns=field_to_ignore)
            d = pd.DataFrame(scaler.transform(d),columns=[x for x in COL_ORDER if x not in field_to_ignore])
            d['Src IP'] = df['Src IP']
            d['Dst IP'] = df['Dst IP']
            df = d[COL_ORDER]
            df = df.dropna(axis=0,subset=field_to_ignore)
            
            print("Saving file: Rbot_adversarial-{}-{}.csv".format(group, step))
            df.to_csv("data/raw/Rbot_adversarial-{}-{}.csv".format(group, step), index=False)