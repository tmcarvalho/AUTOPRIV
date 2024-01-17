""" 
This script will modeling data
"""
import os
import re
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from modelingBO import evaluate_model_bo
from modelingSH import evaluate_model_sh
from modelingHB import evaluate_model_hb
from modelingGS import evaluate_model_gs
from modelingRS import evaluate_model_rs

warnings.filterwarnings(action='ignore', category=FutureWarning)

def save_results(file, args, results):
    """Create a folder if doesn't exist and save results

    Args:
        file (string): file name
        args (args): command line arguments
        results (list of Dataframes): results for cross validation and out of sample
    """
    output_folder_val = (
        f'{args.output_folder}/validation')
    output_folder_test = (
        f'{args.output_folder}/test')
    if not os.path.exists(output_folder_val):
        os.makedirs(output_folder_val)
    if not os.path.exists(output_folder_test):
        os.makedirs(output_folder_test)

    results[0].to_csv(f'{output_folder_val}/{file}.csv', index=False)
    results[1].to_csv(f'{output_folder_test}/{file}.csv', index=False)


def modeling(file, args):
    """Apply predictive performance.

    Args:
        file (string): input file
    """
    print(f'{args.input_folder}/{file}')

    indexes = np.load('indexes.npy', allow_pickle=True).item()
    indexes = pd.DataFrame.from_dict(indexes)

    f = list(map(int, re.findall(r'\d+', file.split('_')[0])))
    index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]

    orig_folder = 'original'
    _, _, orig_files = next(os.walk(f'{orig_folder}'))
    orig_file = [fl for fl in orig_files if list(map(int, re.findall(r'\d+', fl.split('.')[0])))[0] == f[0]]
    print(orig_file)
    orig_data = pd.read_csv(f'{orig_folder}/{orig_file[0]}')
    data = pd.read_csv(f'{args.input_folder}/{file}')

    if args.type == 'original':
        data_idx = list(set(list(data.index)) - set(index))
        data = orig_data.iloc[data_idx, :]
        print(data.shape)

    # if f[0] == 37: # because SDV models returns real states instead of numbers as in the original data
    #     data.rename(columns = {'state':'code_number','phone_number':'number', 'voice_mail_plan':'voice_plan'}, inplace = True)
    
    # if f[0] == 55:
    #     data.rename(columns = {'state':'code_number'}, inplace = True) 

    # prepare data to modeling
    orig_data = orig_data.apply(LabelEncoder().fit_transform)
    data = data.apply(LabelEncoder().fit_transform)

    # prepare data to modeling
    if args.type == 'PrivateSMOTE':
        x_train, y_train = data.iloc[:, :-2], data.iloc[:, -2]
    else:
        x_train, y_train = data.iloc[:, :-1], data.iloc[:, -1]

    x_test = orig_data.iloc[index, :-1]
    y_test = orig_data.iloc[index, -1]

    try:
        if args.opt == 'BO':
            results = evaluate_model_bo(x_train, x_test, y_train, y_test)
        elif args.opt == 'HB':
            results = evaluate_model_hb(x_train, x_test, y_train, y_test)
        elif args.opt == 'SH':
            results = evaluate_model_sh(x_train, x_test, y_train, y_test)
        elif args.opt == 'GS':
            results = evaluate_model_gs(x_train, x_test, y_train, y_test)
        else:
            results = evaluate_model_rs(x_train, x_test, y_train, y_test)
        save_results(file, args, results)
    
    except Exception:
        with open('output/failed_files.txt', 'a') as failed_file:
            #  Save the name of the failed file to a text file
            failed_file.write(f'{args.input_folder}/{file} --- {args.opt}\n')
