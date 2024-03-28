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

def create_dataframe(meta_features_list, columns):
    df = pd.DataFrame(columns=columns)
    for i, inner_row in enumerate(meta_features_list):
        if set(inner_row.columns)==set(columns):
            df.loc[i] = inner_row.values[0]

        else:
            pass # filter synthcity errors
    return df

# %%
dpl_metaft = []
files_dpl = os.listdir(f'{os.path.dirname(os.getcwd())}/data/deep_learning/')
for file in files_dpl:
    data = pd.read_csv(f'{os.path.dirname(os.getcwd())}/data/deep_learning/{file}')
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    dpl_metaft.append(meta_features(X,y, file))
# %%
dpl_metaft_df = create_dataframe(dpl_metaft, columns=dpl_metaft[0].columns)
# %%
prsmote_metaft = []
files_prsmote = os.listdir(f'{os.path.dirname(os.getcwd())}/data/PrivateSMOTE/')
for file in files_prsmote:
    data = pd.read_csv(f'{os.path.dirname(os.getcwd())}/data/PrivateSMOTE/{file}')
    X, y = data.iloc[:, :-2].values, data.iloc[:, -2].values
    prsmote_metaft.append(meta_features(X,y, file))

# %%
prsmote_metaft_df = create_dataframe(prsmote_metaft, columns=prsmote_metaft[0].columns)
# %%
city_metaft = []
files_city = os.listdir(f'{os.path.dirname(os.getcwd())}/data/synthcity/')
for file in files_city:
    data = pd.read_csv(f'{os.path.dirname(os.getcwd())}/data/synthcity/{file}')
    X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    city_metaft.append(meta_features(X,y, file))

# %%
city_metaft_df = create_dataframe(city_metaft, columns=city_metaft[0].columns)
# %%
all_metaft_df = pd.concat([dpl_metaft_df,prsmote_metaft_df,city_metaft_df]).reset_index(drop=True)

# %% add technique details for meta features
for idx in all_metaft_df.index:
    technique = all_metaft_df.ds_complete[idx].split('_')[1]
    #print(metaft_df.ds_complete[idx])
    if technique in (deepl + city):
        all_metaft_df.at[idx, 'QI'] = ''.join(filter(str.isdigit, all_metaft_df.ds_complete[idx].split('_')[2]))
        all_metaft_df.at[idx, 'epochs'] = ''.join(filter(str.isdigit, all_metaft_df.ds_complete[idx].split('_')[3]))
        all_metaft_df.at[idx, 'batch'] = ''.join(filter(str.isdigit, all_metaft_df.ds_complete[idx].split('_')[4].split('.')[0]))
        all_metaft_df.at[idx, 'technique'] = technique
        if technique in city:
            all_metaft_df.at[idx, 'epsilon'] = ''.join(filter(str.isdigit, all_metaft_df.ds_complete[idx].split('_')[5].split('.')[0]))
            all_metaft_df.at[idx, 'technique'] = technique.upper()

    if 'privateSMOTE' in all_metaft_df.ds_complete[idx]:
        all_metaft_df.at[idx, 'QI'] = ''.join(filter(str.isdigit, all_metaft_df.ds_complete[idx].split('_')[2]))
        all_metaft_df.at[idx, 'knn'] = ''.join(filter(str.isdigit, all_metaft_df.ds_complete[idx].split('_')[3]))
        all_metaft_df.at[idx, 'per'] = ''.join(filter(str.isdigit, all_metaft_df.ds_complete[idx].split('_')[4].split('.')[0]))
        all_metaft_df.at[idx, 'technique'] = r'$\epsilon$-PrivateSMOTE'
# %% change names
all_metaft_df["technique"]=all_metaft_df["technique"].str.replace('CopulaGAN', 'Copula GAN')

# %% get only the best optimisation type and store linkability and roc auc results
performance_results = pd.read_csv(f'{os.path.dirname(os.getcwd())}/output_analysis/results_test.csv')
linkability_results = pd.read_csv(f'{os.path.dirname(os.getcwd())}/output_analysis/anonymeterk3.csv')

# %% merge predcitive performance and linakbility
performance_results = pd.merge(performance_results, linkability_results, how='left', left_on=['ds_complete', 'ds', 'technique'], right_on=['ds_complete', 'dsn', 'technique'])
performance_results['value'].isna().sum()
rows_to_remove = performance_results[performance_results['value'].isna()].index
performance_results = performance_results.drop(rows_to_remove)
# %% merge meta features with predictive performance and linkability
all_metaft_df_ = performance_results.merge(all_metaft_df,how='left',on=['ds_complete', 'technique'])

# %% change columns positions
end_cols = ['ds','ds_complete','opt_type','technique','value', 'test_roc_auc'] 
other_cols = [col for col in all_metaft_df_.columns if col not in end_cols]
metaft_df = all_metaft_df_[other_cols + end_cols]

# %%
metaft_df = metaft_df.drop(columns=['params', 'model', 'test_roc_auc_oracle', 'roc_auc_perdif', 'ci', 'dsn'])
# %%
metaft_df.to_csv('../output_analysis/metaftk3.csv', index=False)
# %%

# allmfe = MFE(groups="all", summary="all")
# allmfe.fit(X, y)
# allft = allmfe.extract()
# print(len(allft[0]))
# allftdf = pd.DataFrame(allft[1:], columns=allft[0])
# allftdf.to_csv('output/allmetaft.csv', index=False)
