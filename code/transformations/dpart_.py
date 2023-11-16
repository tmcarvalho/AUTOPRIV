# %%
from os import sep, walk
import re
import ast
import pandas as pd
import numpy as np
from dpart.engines import DPsynthpop, Independent
# %%

def keep_numbers(data):
    """mantain correct data types according to the data"""
    for col in data.columns:
        # transform strings to digits
        if isinstance(data[col].iloc[0], str) and data[col].iloc[0].isdigit():
            data[col] = data[col].astype(float)
        # remove trailing zeros
        if isinstance(data[col].iloc[0], (int, float)):
            if int(data[col].iloc[0]) == float(data[col].iloc[0]):
                data[col] = data[col].astype(int)
    return data

def aux_singleouts(key_vars, dt):
    """create single out variable based on k-anonymity"""
    k = dt.groupby(key_vars)[key_vars[0]].transform(len)
    dt['single_out'] = np.where(k < 2 , 1, 0)
    return dt


def synt_dpart(original_folder, file, technique):
    output_interpolation_folder = '../../data/dpartk2'
    data = pd.read_csv(f'{original_folder}/{file}')

    # get 80% of data to synthesise
    indexes = np.load('../../indexes.npy', allow_pickle=True).item()
    indexes = pd.DataFrame.from_dict(indexes)

    f = list(map(int, re.findall(r'\d+', file.split('_')[0])))
    index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]
    data_idx = list(set(list(data.index)) - set(index))
    data = data.iloc[data_idx, :]
    data = keep_numbers(data)

    list_key_vars = pd.read_csv('../../list_key_vars.csv')
    set_key_vars = ast.literal_eval(
        list_key_vars.loc[list_key_vars['ds']==f[0], 'set_key_vars'].values[0])

    # transform target to string because integer targets are not well synthesised
    data[data.columns[-1]] = data[data.columns[-1]].astype(str)

    # limit to 3 set of QIs
    for idx in range(3):
        keys = set_key_vars[idx]
        data = aux_singleouts(keys, data)
        print(data.shape)
        protected_data = data.loc[data['single_out'] == 0].reset_index(drop=True)
        unprotected_data = data.loc[data['single_out'] == 1].reset_index(drop=True)

        del protected_data['single_out']
        del unprotected_data['single_out']
        print(protected_data.shape)
        print(unprotected_data.shape)

        col_ = unprotected_data.columns
        X_bounds = {}
        for col in col_:
            if unprotected_data[col].dtype == np.object_:
                if unprotected_data[col].str.contains("/").any() or unprotected_data[col].str.contains("").any():
                    unprotected_data[col] = unprotected_data[col].apply(lambda x: x.replace("/", "-"))
                col_stats = unprotected_data[col].unique().tolist()
                col_stats_dict = {'categories': col_stats}
                
            else:
                col_stats_dict = {'min': unprotected_data[col].min(),
                                'max': unprotected_data[col].max()}
            X_bounds.update({col: col_stats_dict})

        epsilon = [0.1, 0.25, 0.5, 0.75, 1.0]
        

        for ep in epsilon:
            try:
                if technique == 'independent':
                    dpart_dpsp = Independent(epsilon=ep, bounds=X_bounds)
                else:
                    dpart_dpsp = DPsynthpop(epsilon=ep, bounds=X_bounds)
                
                # Fit the model to the data
                dpart_dpsp.fit(unprotected_data)
                synth_df = dpart_dpsp.sample(len(unprotected_data))
                
                new_data = pd.concat([synth_df, protected_data])
                print(new_data.shape)    
                # save synthetic data
                new_data.to_csv(
                    f'{output_interpolation_folder}{sep}ds{file.split(".csv")[0]}_{technique}_QI{idx}_ep{ep}.csv',
                    index=False)
            
            except Exception:
                with open('../../output/failed_file_synth.txt', 'a') as failed_file:
                    #  Save the name of the failed file to a text file
                    failed_file.write(f'{file} --- QI{idx} --- tech: {technique}\n')


# %%
original_folder = '../../data/original'
_, _, input_files = next(walk(f'{original_folder}'))

not_considered_files = [0,1,3,13,23,28,34,36,40,48,54,66,87, 100,43]
for idx,file in enumerate(input_files):
    if int(file.split(".csv")[0]) not in not_considered_files:
        for technique in ['synthpop', 'independent']:
            print(idx)
            print(file)
            synt_dpart(original_folder, file, technique)
            
# %%
