#!/usr/bin/env python
from os import sep
import re
import ast
import pandas as pd
import numpy as np
import gc
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer
from numba import jit, cuda

def aux_singleouts(key_vars, dt):
    """create single out variable based on k-anonymity"""
    k = dt.groupby(key_vars)[key_vars[0]].transform(len)
    dt['single_out'] = np.where(k < 2 , 1, 0)
    return dt


def synth(msg, args):
    """Synthesize data using a deep learning model

    Args:
        msg (str): name of the file, technique and parameters.
        
    Returns:
        None
    """
    cuda.select_device(int(args.id))
    print(msg)
    output_interpolation_folder = 'data/deep_learningk2/'
    if msg.split('_')[0] not in ['ds100']:
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

        technique = msg.split('_')[1]
        print(technique)
        ep = list(map(int, re.findall(r'\d+', msg.split('_')[3])))[0]
        bs = list(map(int, re.findall(r'\d+', msg.split('_')[4])))[0]
        
        if technique=='CTGAN':
            # check nr of singleouts
            with open('output/singleouts.txt', 'a') as info:
                #  Save the name of the failed file to a text file
                info.write(f'{msg.split("_")[0]} --- {msg.split("_")[2]} --- lendata: {data.shape[0]} ---- nsing: {unprotected_data.shape[0]}\n')

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=unprotected_data)

        print("epochs: ", ep)
        print("batch_size: ", bs)
        if technique=='CTGAN':
            model = CTGANSynthesizer(metadata, epochs=ep, batch_size=bs, verbose=True)
        elif technique=='TVAE':
            model = TVAESynthesizer(metadata, epochs=ep, batch_size=bs)
        else:
            model = CopulaGANSynthesizer(metadata, epochs=ep, batch_size=bs, verbose=True)

        # Generate synthetic data
        new_data = modeling(model, unprotected_data)                
        new_data_ = pd.concat([new_data, protected_data])

        # Save the synthetic data
        new_data_.to_csv(
            f'{output_interpolation_folder}{sep}{msg}.csv',
            index=False)
        gc.collect()

# function optimized to run on gpu 
@jit(target_backend='cuda')
def modeling(model, data):
    # Fit the model to the data
    model.fit(data)
    # Generate synthetic data
    new_data = model.sample(num_rows=len(data))
    return new_data
