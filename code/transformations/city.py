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
# Plugins(categories=["generic", "privacy"]).list()
# from sklearn.datasets import load_diabetes
# X, y = load_diabetes(return_X_y=True, as_frame=True)
# X["target"] = y
# syn_model = Plugins().get("ctgan", n_iter=100, batch_size=50)
# syn_model.fit(X)
# new_data = syn_model.generate(count = len(X))
# from synthcity.metrics import Metrics
# Metrics().list()
# Metrics().evaluate(X_gt=X, X_syn=new_data, metrics={'privacy':['k-anonymization', 'distinct l-diversity', 'identifiability_score']})


def aux_singleouts(key_vars, dt):
    """create single out variable based on k-anonymity"""
    k = dt.groupby(key_vars)[key_vars[0]].transform(len)
    dt['single_out'] = np.where(k < 2 , 1, 0)
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
    output_interpolation_folder = 'data/synthcityk2/'

    f = list(map(int, re.findall(r'\d+', msg.split('_')[0])))
    print(str(f[0]))
    data = pd.read_csv(f'data/original/{str(f[0])}.csv')

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
    print(technique)
    
    if (technique in ['dpgan', 'pategan']):
        keys_nr = list(map(int, re.findall(r'\d+', msg.split('_')[2])))[0]
        print(keys_nr)
        keys = set_key_vars[keys_nr]

        data = aux_singleouts(keys, data)
        protected_data = data.loc[data['single_out'] == 0].reset_index(drop=True)
        unprotected_data = data.loc[data['single_out'] == 1].reset_index(drop=True)
        del protected_data['single_out']
        del unprotected_data['single_out']
        print(protected_data.shape)
        print(unprotected_data.shape)
        
        if (keys_nr <3):
            if technique == 'dpgan':
                epo = list(map(int, re.findall(r'\d+', msg.split('_')[3])))[0]
                bs = list(map(int, re.findall(r'\d+', msg.split('_')[4])))[0]
                epi = list(map(float, re.findall(r'\d+\.\d+', msg.split('_')[5])))[0]
                if epi not in [0.2, 0.01]:
                    print(epo)
                    print(bs)
                    print(epi)
                    model = Plugins().get("dpgan", n_iter=epo, batch_size=bs, epsilon=epi)

            elif technique == 'pategan':
                epo = list(map(int, re.findall(r'\d+', msg.split('_')[3])))[0]
                bs = list(map(int, re.findall(r'\d+', msg.split('_')[4])))[0]
                epi = list(map(float, re.findall(r'\d+\.\d+', msg.split('_')[5])))[0]
                if epi not in [0.2, 0.001]:
                    print(epo)
                    print(bs)
                    print(epi)
                    model = Plugins().get("pategan", n_iter=epo, batch_size=bs, epsilon=epi)

            elif technique=='privbayes':
                epi = list(map(float, re.findall(r'\d+\.\d+', msg.split('_')[3])))[0]
                if epi not in [0.2, 0.001]:
                    print(epi)
                    model = Plugins().get("privbayes", epsilon=epi)
            
            elif technique == 'tvae':
                epo = list(map(int, re.findall(r'\d+', msg.split('_')[3])))[0]
                bs = list(map(int, re.findall(r'\d+', msg.split('_')[4])))[0]
                print(epo)
                print(bs)
                model = Plugins().get("tvae", n_iter=epo, batch_size=bs)
            
            elif technique == 'ctgan':
                epo = list(map(int, re.findall(r'\d+', msg.split('_')[3])))[0]
                bs = list(map(int, re.findall(r'\d+', msg.split('_')[4])))[0]
                print(epo)
                print(bs)
                model = Plugins().get("ctgan", n_iter=epo, batch_size=bs)
            
            else:
                li = list(map(int, re.findall(r'\d+', msg.split('_')[3])))[0]
                print(li)
                model = Plugins().get("bayesian_network", struct_learning_n_iter=li)

            try:
                new_data = modeling(model, unprotected_data)
                new_data_ = pd.concat([new_data, protected_data])

                # Save the synthetic data
                new_data_.to_csv(
                    f'{output_interpolation_folder}{sep}{msg}.csv',
                    index=False)
            except Exception:
                pass

# function optimized to run on gpu 
@jit(target_backend='cuda')
def modeling(model, data):
    # Fit the model to the data
    model.fit(data)
    # Generate synthetic data
    new_data = model.generate(count=len(data))
    new_data = new_data.dataframe()
    return new_data
