# Importing necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LinearRegression, BayesianRidge,RidgeCV
from sklearn.preprocessing import StandardScaler
import itertools
import re
import shap
from pymfe.mfe import MFE
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Function to prepare data
def prepare_data(opt_type):
    # Read data
    training_data = pd.read_csv(f'{os.getcwd()}/output_analysis/metaftk3.csv')
    testing_data = pd.read_csv(f'{os.getcwd()}/43.csv')
    
    # get 80% of data
    indexes = np.load('indexes.npy', allow_pickle=True).item()
    indexes = pd.DataFrame.from_dict(indexes)

    index = indexes.loc[indexes['ds']==str(43), 'indexes'].values[0]
    data_idx = list(set(list(testing_data.index)) - set(index))
    testing_data = testing_data.iloc[data_idx, :]
    
    training_data = training_data.loc[training_data['opt_type']==opt_type].reset_index(drop=True)
    print(training_data.shape)
    return training_data, testing_data

# Function to encode categorical data
def encode(data, encoder):
    # Label encode 'technique'
    data['technique'] = encoder.fit_transform(data['technique'])

# Function to generate pipeline parameters
def generate_pipeline_params(params_dict):
    keys, values = zip(*params_dict.items())
    param_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return pd.DataFrame(param_dicts)

def create_testing_pipelines():
    # Define parameters for different techniques
    city_params = {'technique': ['DPGAN', 'PATEGAN'], 'epochs':[100, 200], 'batch':[50, 100], 'epsilon':[0.1, 0.5, 1.0, 5.0]}
    deep_learning_params = {'technique': ['CopulaGAN', 'CTGAN', 'TVAE'], 'epochs':[100, 200], 'batch':[50, 100]}
    privateSMOTE_params = {'technique':['privateSMOTE'], 'knn':[1,3,5], 'per': [1,2,3], 'epsilon':[0.1, 0.5, 1.0, 5.0, 10.0]}
    
    # Generate parameter combinations for each technique
    city = generate_pipeline_params(city_params)
    deep_learning = generate_pipeline_params(deep_learning_params)
    privatesmote = generate_pipeline_params(privateSMOTE_params)
    return city, deep_learning, privatesmote

# Function to generate unseen data
def generate_unseen_data(ftdf, technique_params):
    unseen_data = pd.DataFrame(np.repeat(ftdf.values, technique_params.shape[0], axis=0), columns=ftdf.columns)
    unseen_data = pd.concat([unseen_data, technique_params], axis=1)
    return unseen_data

# Function to extract metafeatures from an unseen dataset
def extract_metafeatures(unseen_dataset):
    mfe = MFE()
    mfe.fit(unseen_dataset.iloc[:, :-1].values, unseen_dataset.iloc[:, -1].values)
    ft = mfe.extract()
    unseen_metafeatures = pd.DataFrame(ft[1:], columns=ft[0])
    return unseen_metafeatures

def calculate_rank(predictions_performance, predictions_linkability):
    # Calculate ranks for performance and linkability
    rank_performance = pd.Series(predictions_performance).rank(pct=False, ascending=False) # Set ascending=False for inverse ranking
    rank_linkability = pd.Series(predictions_linkability).rank(pct=False, ascending=True)  # lower values, lower rank -> rank 1 is the best

    # Calculate mean rank of performance and linkability
    mean_rank = (rank_performance + rank_linkability) / 2
    return mean_rank

def main():
    nan_to_keep = ['QI', 'epochs', 'batch', 'knn', 'per', 'epsilon']
    label_encoder = LabelEncoder()

    opt_type='Halving'
    # Prepare data
    training_data, testing_data = prepare_data(opt_type)
    # Replace NaN in privacy parameters
    training_data[nan_to_keep] = training_data[nan_to_keep].fillna(0)
    encode(training_data, label_encoder)

    # Remove NaN columns
    columns_to_drop = training_data.columns[training_data.isnull().any()]
    training_data = training_data.drop(columns=columns_to_drop)

    # remove unwanted columns
    training_data = training_data.drop(columns=['ds_complete','ds','opt_type','QI'])
    # Two possible ways: predict considering the QIs, or general solution (without QIS), we chose the second approach
    # del training_data['QI']
    
    # Generate pipeline parameters for testing
    city_params, deep_learning_params, privateSMOTE_params = create_testing_pipelines()

    # Extract metafeatures from unseen data
    ftdf = extract_metafeatures(testing_data)

    # Generate unseen data
    unseen_data_deep = generate_unseen_data(ftdf, deep_learning_params)
    unseen_data_city = generate_unseen_data(ftdf, city_params)
    unseen_data_privatesmote = generate_unseen_data(ftdf, privateSMOTE_params)

    # Concatenate all unseen_data DataFrames
    unseen_data = pd.concat([unseen_data_deep, unseen_data_city, unseen_data_privatesmote], ignore_index=True)

    # Replace NaN in privacy parameters with 0 in unseen data
    unseen_data[list(set(nan_to_keep)-set(['QI']))] = unseen_data[list(set(nan_to_keep)-set(['QI']))].fillna(0)
    unseen_data = unseen_data.drop(columns=columns_to_drop)
    # del unseen_data['QI']
    encode(unseen_data, label_encoder)

    # change columns position
    end_cols = ['epochs', 'batch', 'knn', 'per', 'epsilon', 'technique'] 
    other_cols = [col for col in unseen_data.columns if col not in end_cols]
    unseen_data = unseen_data[other_cols + end_cols]

    # Define x_train, y_train_accuracy, y_train_linkability
    train_metafeatures, roc_auc = training_data.iloc[:,:-2].values, training_data.iloc[:,-1].values
    linkability = training_data.iloc[:,-2].values

    scaler = StandardScaler()
    train_metafeatures = scaler.fit_transform(train_metafeatures)
    unseen_data_scaled = scaler.transform(unseen_data)

    # Train linear regression model for predictive performance
    lr_performance = LinearRegression()
    lr_performance.fit(train_metafeatures, roc_auc)
    predictions_performance = lr_performance.predict(unseen_data_scaled)

    # Train linear regression model for linkability
    lr_linkability = LinearRegression()
    lr_linkability.fit(train_metafeatures, linkability)
    predictions_linkability = lr_linkability.predict(unseen_data_scaled)

    #  SHAP Explanations
    print("Calculating SHAP values for performance model...")
    explainer_perf = shap.LinearExplainer(lr_performance, train_metafeatures, feature_perturbation="interventional")
    shap_values_perf = explainer_perf.shap_values(unseen_data_scaled)
    print("Calculating SHAP values for linkability model...")
    explainer_link = shap.LinearExplainer(lr_linkability, train_metafeatures, feature_perturbation="interventional")
    shap_values_link = explainer_link.shap_values(unseen_data_scaled)

    # SHAP Visualization
    shap.summary_plot(shap_values_perf, training_data.columns, show=True)
    shap.summary_plot(shap_values_link, training_data.columns, show=True)

    # Create DataFrame with predictions
    predict_columns = ['epochs','batch','knn','per','epsilon','technique']
    pred_performance = pd.DataFrame(predictions_performance, columns=['Predictions Performance'])
    pred_linkability = pd.DataFrame(predictions_linkability, columns=['Predictions Linkability'])
    output = pd.concat([unseen_data.loc[:,predict_columns], pred_performance, pred_linkability], axis=1)

    output['technique'] = label_encoder.inverse_transform(output['technique'])
    # print(output.sort_values(by=['Predictions Performance'], ascending=False))

    output['rank'] = calculate_rank(pred_performance['Predictions Performance'].values, pred_linkability['Predictions Linkability'].values)
    print(output.sort_values(by=['rank'], ascending=True))
    output = output.sort_values(by=['rank'], ascending=True)
    output.to_csv(f'{os.getcwd()}/output_analysis/predictions_{opt_type}.csv', index=False)

if __name__ == "__main__":
    main()

# exec(open('code/stackml.py').read())