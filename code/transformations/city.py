# %%
#!/usr/bin/env python
from os import sep, walk
import re
import ast
import pandas as pd
import numpy as np
from synthcity.plugins import Plugins
from numba import jit, cuda

# %%

def aux_singleouts(key_vars, dt):
    """create single out variable based on k-anonymity"""
    k = dt.groupby(key_vars)[key_vars[0]].transform(len)
    dt['single_out'] = np.where(k < 3 , 1, 0)
    return dt

def synth_city(msg, args):
    """Synthesize data using a deep learning model

    Args:
        original_folder (str): Path to the original data folder.
        file (str): Name of the original data file.
        technique (str): Deep learning technique to use. Valid options are 'TVAE', 'CTGAN', and 'CopulaGAN'.

    Returns:
        None
    """
    cuda.select_device(int(args.id))
    print(msg)
    output_interpolation_folder = 'data/synthcity/'

    f = list(map(int, re.findall(r'\d+', msg.split('_')[0])))
    data = pd.read_csv(f'data/original/{str(f[0])}.csv')
    print(f)
    # get 80% of data to synthesise
    indexes = np.load('indexes.npy', allow_pickle=True).item()
    indexes = pd.DataFrame.from_dict(indexes)

    index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]
    data_idx = list(set(list(data.index)) - set(index))
    data = data.iloc[data_idx, :]

    # transform target to string because integer targets are not well synthesised
    data[data.columns[-1]] = data[data.columns[-1]].astype(str)
    
    list_key_vars = pd.read_csv('list_key_vars.csv')
    set_key_vars = ast.literal_eval(
        list_key_vars.loc[list_key_vars['ds']==f[0], 'set_key_vars'].values[0])

    technique = msg.split('_')[1]
    
    keys_nr = list(map(int, re.findall(r'\d+', msg.split('_')[2])))[0]
    keys = set_key_vars[keys_nr]
    if (f[0] == 37) and ('code_number' in keys): # because list of key vars have the original names (before the change due to SDV)
        keys[keys.index("code_number")] = "state"
    if (f[0] == 37) and ('phone_number' in keys):
        keys[keys.index("phone_number")] = "number"
    if (f[0] == 37) and ('voice_mail_plan' in keys):
        keys[keys.index("voice_mail_plan")] = "voice_plan"
    if (f[0] == 55) and ('code_number' in keys):
        keys[keys.index("code_number")] = "state"
    print(keys)
    data = aux_singleouts(keys, data)
    protected_data = data.loc[data['single_out'] == 0].reset_index(drop=True)
    unprotected_data = data.loc[data['single_out'] == 1].reset_index(drop=True)
    del protected_data['single_out']
    del unprotected_data['single_out']
    try: 
        if technique == 'dpgan':
            epo = list(map(int, re.findall(r'\d+', msg.split('_')[3])))[0]
            bs = list(map(int, re.findall(r'\d+', msg.split('_')[4])))[0]
            epi = list(map(float, re.findall(r'\d+\.\d+', msg.split('_')[5])))[0]
            model = Plugins().get("dpgan", n_iter=epo, batch_size=bs, epsilon=epi)

        elif technique == 'pategan':
            epo = list(map(int, re.findall(r'\d+', msg.split('_')[3])))[0]
            bs = list(map(int, re.findall(r'\d+', msg.split('_')[4])))[0]
            epi = list(map(float, re.findall(r'\d+\.\d+', msg.split('_')[5])))[0]
            model = Plugins().get("pategan", n_iter=epo, batch_size=bs, epsilon=epi)


        new_data = modeling(model, unprotected_data)
        new_data_ = pd.concat([new_data, protected_data])

        # Save the synthetic data
        new_data_.to_csv(
            f'{output_interpolation_folder}{sep}{msg}.csv',
            index=False)
    except:
        with open('output/failed_file_synth.txt', 'a') as failed_file:
            #  Save the name of the failed file to a text file
            failed_file.write(f'{msg} --- city\n')

# function optimized to run on gpu 
@jit(target_backend='cuda')
def modeling(model, data):
    # Fit the model to the data
    model.fit(data)
    # Generate synthetic data
    new_data = model.generate(count=len(data))
    new_data = new_data.dataframe()
    return new_data
