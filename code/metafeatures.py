# %%
import os
from pymfe.mfe import MFE
import pandas as pd

# %%
deepl = ['TVAE', 'CopulaGAN', 'CTGAN']
city = ['pategan', 'dpgan']

def meta_features(X,y, file):
    mfe = MFE()
    mfe.fit(X, y)
    ft = mfe.extract()
    # print(len(ft[0]))
    ftdf = pd.DataFrame(ft[1:], columns=ft[0])
    ftdf['ds_complete'] = file

    return ftdf

# %%
all_metaft = []
files_dpl = os.listdir(f'{os.path.dirname(os.getcwd())}/data/deep_learning/')
for file in files_dpl:
    data = pd.read_csv(f'{os.path.dirname(os.getcwd())}/data/deep_learning/{file}')
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    all_metaft.append(meta_features(X,y, file))
# %%
files_prsmote = os.listdir(f'{os.path.dirname(os.getcwd())}/data/PrivateSMOTE/')
for file in files_prsmote:
    data = pd.read_csv(f'{os.path.dirname(os.getcwd())}/data/PrivateSMOTE/{file}')
    X, y = data.iloc[:, :-2].values, data.iloc[:, -2].values
    all_metaft.append(meta_features(X,y, file))
# %%
files_city = os.listdir(f'{os.path.dirname(os.getcwd())}/data/synthcity/')
for file in files_city:
    data = pd.read_csv(f'{os.path.dirname(os.getcwd())}/data/synthcity/{file}')
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    all_metaft.append(meta_features(X,y, file))
# %% get only the best optimisation type and store linkability and roc auc results
performance_results = pd.read_csv(f'{os.path.dirname(os.getcwd())}/output_analysis/results_test.csv')
linkability_results = pd.read_csv(f'{os.path.dirname(os.getcwd())}/output_analysis/anonymeterk3.csv')
all_metaft_df = pd.DataFrame(all_metaft)
# %% not efficient: read all data in each folder and merge to accuracy and linkability
metafeatures = performance_results.copy()
for idx in performance_results.index:
    if performance_results.opt_type[idx] == 'Halving':
        linkability_file = linkability_results.loc[(linkability_results.ds_complete == performance_results.ds_complete[idx])].reset_index(drop=True)
        #meta_file = all_metaft.loc[(all_metaft.ds_complete == performance_results.ds_complete[idx])].reset_index(drop=True)
        metafeatures['linkability_score'] = linkability_file.value.iloc[0]

# %%
metafeatures = pd.merge(metafeatures, all_metaft_df, how='left', on='ds_complete')

# %%
for idx in metafeatures.index:
    technique = metafeatures.ds_complete[idx].split('_')[1]
    metafeatures.at[idx, 'technique'] = technique
    #print(metaft_df.ds_complete[idx])
    if technique in (deepl + city):
        metafeatures.at[idx, 'QI'] = ''.join(filter(str.isdigit, metafeatures.ds_complete[idx].split('_')[2]))
        metafeatures.at[idx, 'epochs'] = ''.join(filter(str.isdigit, metafeatures.ds_complete[idx].split('_')[3]))
        metafeatures.at[idx, 'batch'] = ''.join(filter(str.isdigit, metafeatures.ds_complete[idx].split('_')[4].split('.')[0]))
        
        if technique in city:
            metafeatures.at[idx, 'epsilon'] = ''.join(filter(str.isdigit, metafeatures.ds_complete[idx].split('_')[5].split('.')[0]))

    if 'privateSMOTE' in metafeatures.ds_complete[idx]:
        metafeatures.at[idx, 'QI'] = ''.join(filter(str.isdigit, metafeatures.ds_complete[idx].split('_')[2]))
        metafeatures.at[idx, 'knn'] = ''.join(filter(str.isdigit, metafeatures.ds_complete[idx].split('_')[3]))
        metafeatures.at[idx, 'per'] = ''.join(filter(str.isdigit, metafeatures.ds_complete[idx].split('_')[4].split('.')[0]))

# %% change columns positions
end_cols = ['linkability_score', 'test_roc_auc'] 
other_cols = [col for col in metafeatures.columns if col not in end_cols]
metaft_df = metafeatures[other_cols + end_cols]
# %%
metaft_df.to_csv('../output_analysis/metaftk3_halving.csv', index=False)
# %%

# allmfe = MFE(groups="all", summary="all")
# allmfe.fit(X, y)
# allft = allmfe.extract()
# print(len(allft[0]))
# allftdf = pd.DataFrame(allft[1:], columns=allft[0])
# allftdf.to_csv('output/allmetaft.csv', index=False)
