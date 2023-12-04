# %%
from pymfe.mfe import MFE
import pandas as pd

#%%
def meta_features(file):
    if any(word in file for word in deepl):
        data = pd.read_csv(f'../data/deep_learningk2/{file}')
    if 'transf' in file:
        data = pd.read_csv(f'../data/PPT_transformed/PPT_train/{file}')
    if 'privateSMOTE' in file:
        data = pd.read_csv(f'../data/PrivateSMOTEk2/{file}')
    if any(word in file for word in city):
        data = pd.read_csv(f'../data/synthcityk2/{file}')
    
    if 'privateSMOTE' in file:
        X, y = data.iloc[:, :-2].values, data.iloc[:, -2].values
    else:    
        X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    
    mfe = MFE()
    mfe.fit(X, y)
    ft = mfe.extract()
    print(len(ft[0]))
    ftdf = pd.DataFrame(ft[1:], columns=ft[0])

    return ftdf

# %%
performance_results = pd.read_csv('../output/resultsCV.csv')
linkability_results = pd.read_csv('../output/anonymeterk2.csv')
deepl = ['TVAE', 'CopulaGAN', 'CTGAN']
city = ['PATEGAN', 'DPGAN']
all_metaft = []
c=0
for idx in performance_results.index:
    if performance_results.opt_type[idx] == 'Hyperband':
        print(performance_results.ds_complete[idx])
        metaft = meta_features(performance_results.ds_complete[idx])

        linkability_file = linkability_results.loc[(linkability_results.ds_complete == performance_results.ds_complete[idx])].reset_index(drop=True)
        metaft['linkability_score'] = linkability_file.value.iloc[0]
        metaft['mean_test_score'] = performance_results.mean_test_score[idx]
    
        # concat each meta feature result
        if c == 0:
            all_metaft = metaft
            c += 1
        else:  
            all_metaft = pd.concat([all_metaft, metaft])
        
# %%
all_metaft.to_csv('../output/metaft.csv', index=False)

# %%

# allmfe = MFE(groups="all", summary="all")
# allmfe.fit(X, y)
# allft = allmfe.extract()
# print(len(allft[0]))
# allftdf = pd.DataFrame(allft[1:], columns=allft[0])
# allftdf.to_csv('output/allmetaft.csv', index=False)


