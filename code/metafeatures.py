# %%
import os
from pymfe.mfe import MFE
import pandas as pd

# %%
deepl = ['TVAE', 'CopulaGAN', 'CTGAN']
city = ['pategan', 'dpgan']

def meta_features(file):
    #try:
    if any(word in file for word in deepl):
        data = pd.read_csv(f'{os.path.dirname(os.getcwd())}/data/deep_learning/{file}')
        X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
        #pass
    # print(file)
    #print(any(word in file for word in city))
    if 'privateSMOTE' in file:
        #print("ENTREI")
        data = pd.read_csv(f'{os.path.dirname(os.getcwd())}/data/PrivateSMOTE/{file}')
        X, y = data.iloc[:, :-2].values, data.iloc[:, -2].values
        #pass
    if any(word in file for word in city):
        data = pd.read_csv(f'{os.path.dirname(os.getcwd())}/data/synthcity/{file}')
        X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    
    mfe = MFE()
    mfe.fit(X, y)
    ft = mfe.extract()
    # print(len(ft[0]))
    ftdf = pd.DataFrame(ft[1:], columns=ft[0])

    return ftdf
    # except: pass

# %% get only the best optimisation type and store linkability and roc auc results
performance_results = pd.read_csv(f'{os.path.dirname(os.getcwd())}/output_analysis/results_test.csv')
linkability_results = pd.read_csv(f'{os.path.dirname(os.getcwd())}/output_analysis/anonymeterk3.csv')
all_metaft = []

for idx in performance_results.index:
    if performance_results.opt_type[idx] == 'Hyperband':
        metaft = meta_features(performance_results.ds_complete[idx])
        try:
            linkability_file = linkability_results.loc[(linkability_results.ds_complete == performance_results.ds_complete[idx])].reset_index(drop=True)
            metaft['linkability_score'] = linkability_file.value.iloc[0]
            metaft['test_roc_auc'] = performance_results.test_roc_auc[idx]
            metaft['ds_complete'] = performance_results.ds_complete[idx]
        
            all_metaft.append(metaft)
        except: pass
# %%
# Concatenate all meta feature DataFrames in the list
all_metaft_ = pd.concat(all_metaft, ignore_index=True)
  
# %%
metaft_df = all_metaft_.copy().reset_index(drop=True)
# %%
for idx in metaft_df.index:
    technique = metaft_df.ds_complete[idx].split('_')[1]
    metaft_df.at[idx, 'technique'] = technique
    #print(metaft_df.ds_complete[idx])
    if technique in (deepl + city):
        metaft_df.at[idx, 'QI'] = ''.join(filter(str.isdigit, metaft_df.ds_complete[idx].split('_')[2]))
        metaft_df.at[idx, 'epochs'] = ''.join(filter(str.isdigit, metaft_df.ds_complete[idx].split('_')[3]))
        metaft_df.at[idx, 'batch'] = ''.join(filter(str.isdigit, metaft_df.ds_complete[idx].split('_')[4].split('.')[0]))
        
        if technique in city:
            metaft_df.at[idx, 'epsilon'] = ''.join(filter(str.isdigit, metaft_df.ds_complete[idx].split('_')[5].split('.')[0]))

    if 'privateSMOTE' in metaft_df.ds_complete[idx]:
        metaft_df.at[idx, 'QI'] = ''.join(filter(str.isdigit, metaft_df.ds_complete[idx].split('_')[2]))
        metaft_df.at[idx, 'knn'] = ''.join(filter(str.isdigit, metaft_df.ds_complete[idx].split('_')[3]))
        metaft_df.at[idx, 'per'] = ''.join(filter(str.isdigit, metaft_df.ds_complete[idx].split('_')[4].split('.')[0]))

# %% change columns positions
end_cols = ['linkability_score', 'test_roc_auc'] 
other_cols = [col for col in metaft_df.columns if col not in end_cols]
metaft_df = metaft_df[other_cols + end_cols]
# %%
# metaft_df.to_csv('../output_analysis/metaftk3.csv', index=False)

# TODO: metadf have 4505 should have 4770: validate this! if any is missing, run ONLY the remaining
# %%

# allmfe = MFE(groups="all", summary="all")
# allmfe.fit(X, y)
# allft = allmfe.extract()
# print(len(allft[0]))
# allftdf = pd.DataFrame(allft[1:], columns=allft[0])
# allftdf.to_csv('output/allmetaft.csv', index=False)
