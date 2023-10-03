""" 
This script will modeling data
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from modeling import evaluate_model
import re


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

    results[0].to_csv(f'{output_folder_val}/{file}', index=False)
    results[1].to_csv(f'{output_folder_test}/{file}', index=False)

def modeling_ppt(file, args):
    """Apply predictive performance.

    Args:
        file (string): input file

    Raises:
        Exception: failed to apply smote when single outs
        class is great than non single outs.
        exc: failed to writing the results.
    """
  
    print(f'{args.input_folder}/{file}')

    test_folder = 'PPT_transformed/PPT_test'
    _, _, test_files = next(os.walk(f'{test_folder}'))
    ff = file.split('.')[0]
    f1 = ff.split('_')[0]
    f2 = ff.split('_')[2]
    f = f1+'_'+f2
    print(f)
    test_file = [fl for fl in test_files if fl.split('.')[0] == f]
    print(test_file)
    test_data = pd.read_csv(f'{test_folder}/{test_file[0]}')
    data = pd.read_csv(f'{args.input_folder}/{file}')

    # prepare data to modeling
    test_data = test_data.apply(LabelEncoder().fit_transform)
    data = data.apply(LabelEncoder().fit_transform)

    # split data 80/20
    x_train, y_train = data.iloc[:, :-1], data.iloc[:, -1]

    x_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    # predictive performance
    results = evaluate_model(x_train, x_test, y_train, y_test)
    
    # save validation and test results
    save_results(file, args, results)

# %%
def modeling_privatesmote_and_gans(file, args):
    """Apply predictive performance.

    Args:
        file (string): input file
    """
    print(f'{args.input_folder}/{file}')

    indexes = np.load('indexes.npy', allow_pickle=True).item()
    indexes = pd.DataFrame.from_dict(indexes)

    f = list(map(int, re.findall(r'\d+', file.split('_')[0])))
    index = indexes.loc[indexes['ds']==str(f[0]), 'indexes'].values[0]

    orig_folder = 'data/original'
    _, _, orig_files = next(os.walk(f'{orig_folder}'))
    orig_file = [fl for fl in orig_files if list(map(int, re.findall(r'\d+', fl.split('.')[0])))[0] == f[0]]
    print(orig_file)
    orig_data = pd.read_csv(f'{orig_folder}/{orig_file[0]}')
    data = pd.read_csv(f'{args.input_folder}/{file}')

    # prepare data to modeling
    orig_data = orig_data.apply(LabelEncoder().fit_transform)
    data = data.apply(LabelEncoder().fit_transform)

    if args.type == 'PrivateSMOTE':
        x_train, y_train = data.iloc[:, :-2], data.iloc[:, -2]
    else:
        x_train, y_train = data.iloc[:, :-1], data.iloc[:, -1]

    x_test = orig_data.iloc[index, :-1]
    y_test = orig_data.iloc[index, -1]

    #if (y_train.value_counts().nunique() != 1):
    results = evaluate_model(x_train, x_test, y_train, y_test)
    save_results(file, args, results)
