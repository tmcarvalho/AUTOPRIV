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


def modelling(file, args):
    """Apply predictive performance.

    Args:
        file (string): input file
    """
    print(f'{args.input_folder}/{file}')

    data = pd.read_csv(f'{args.input_folder}/{file}.csv')

    # extract the same 80% of the original data
    if len(file) <= 6:
        indexes = np.load('indexes.npy', allow_pickle=True).item()
        indexes = pd.DataFrame.from_dict(indexes)

        index = indexes.loc[indexes['ds']==file, 'indexes'].values[0]
        
        data_idx = list(set(list(data.index)) - set(index))
        data = data.iloc[data_idx, :]
        print(data.shape)
    data = data.apply(LabelEncoder().fit_transform)

    # prepare data to modeling
    if args.type == 'PrivateSMOTE':
        x_train, y_train = data.iloc[:, :-2], data.iloc[:, -2]
    else:
        x_train, y_train = data.iloc[:, :-1], data.iloc[:, -1]

    # Split the training data into a training set and a validation set for early stop in XGBClassifier
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

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
