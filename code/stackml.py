# Importing necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.linear_model import Ridge
import itertools
from pymfe.mfe import MFE

# Function to prepare data
def prepare_data():
    # Read data
    training_data = pd.read_csv(f'{os.getcwd()}/output_analysis/metaftk3_hyperband.csv')
    testing_data = pd.read_csv(f'{os.getcwd()}/data/original/3.csv')

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
    city_params = {'technique': ['DPGAN', 'PATEGAN'], 'QI':[0,1,2], 'epochs':[100, 200], 'batch':[50, 100], 'epsilon':[0.1, 0.5, 1.0, 5.0]}
    deep_learning_params = {'technique': ['CopulaGAN', 'CTGAN', 'TVAE'], 'QI':[0,1,2], 'epochs':[100, 200], 'batch':[50, 100]}
    privateSMOTE_params = {'technique':['privateSMOTE'], 'QI':[0,1,2], 'knn':[1,3,5], 'per': [1,2,3], 'epsilon':[0.1, 0.5, 1.0, 5.0, 10.0]}
    
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
    rank_performance = pd.Series(predictions_performance).rank(pct=True)
    rank_linkability = pd.Series(predictions_linkability).rank(pct=True, ascending=False)  # Set ascending=False for inverse ranking

    # Calculate mean rank of performance and linkability
    mean_rank = (rank_performance + rank_linkability) / 2
    return mean_rank

def main():
    nan_to_keep = ['QI', 'epochs', 'batch', 'knn', 'per', 'epsilon']
    label_encoder = LabelEncoder()

    # Prepare data
    training_data, testing_data = prepare_data()
    # Replace NaN in privacy parameters
    training_data[nan_to_keep] = training_data[nan_to_keep].fillna(0)
    encode(training_data, label_encoder)

    # Remove NaN columns
    columns_to_drop = training_data.columns[training_data.isnull().any()]
    training_data = training_data.drop(columns=columns_to_drop)
    del training_data['ds_complete']
    del training_data['QI']

    # Define x_train, y_train_accuracy, y_train_linkability
    train_metafeatures, roc_auc = training_data.iloc[:,:-2].values, training_data.iloc[:,-1].values
    linkability = training_data.iloc[:,-2].values

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
    unseen_data[nan_to_keep] = unseen_data[nan_to_keep].fillna(0)
    unseen_data = unseen_data.drop(columns=columns_to_drop)
    del unseen_data['QI']
    encode(unseen_data, label_encoder)

    # Train linear regression model for predictive performance
    # Set random seed for reproducibility
    # np.random.seed(42)
    lr_performance = Ridge()
    lr_performance.fit(train_metafeatures, roc_auc)
    predictions_performance = lr_performance.predict(unseen_data.values)

    # Train linear regression model for linkability
    lr_linkability = Ridge()
    lr_linkability.fit(train_metafeatures, linkability)
    predictions_linkability = lr_linkability.predict(unseen_data.values)

    # Create DataFrame with predictions
    pred_performance = pd.DataFrame(predictions_performance, columns=['Predictions Performance'])
    pred_linkability = pd.DataFrame(predictions_linkability, columns=['Predictions Linkability'])
    output = pd.concat([unseen_data.iloc[:,92:], pred_performance, pred_linkability], axis=1)
    output['technique'] = label_encoder.inverse_transform(output['technique'])
    print(output.sort_values(by=['Predictions Performance'], ascending=False))

    output['rank'] = calculate_rank(pred_performance['Predictions Performance'].values, pred_linkability['Predictions Linkability'].values)
    print(output)
    # output.to_csv(f'{os.getcwd()}/output_analysis/predictions.csv', index=False)

if __name__ == "__main__":
    main()


# SVR, kneighbour, normal LR