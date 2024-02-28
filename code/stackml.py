# %%
import os
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import itertools
from pymfe.mfe import MFE

# %%
# Read data
training_data = pd.read_csv(f'{os.path.dirname(os.getcwd())}/output_analysis/metaftk3.csv')
testing_data = pd.read_csv(f'{os.path.dirname(os.getcwd())}/data/original/3.csv')

# %%
# Replace NaN in privacy parameters with 99999
nan_to_keep = ['QI', 'epochs', 'batch', 'knn', 'per', 'epsilon']
training_data[nan_to_keep] = training_data[nan_to_keep].fillna(0)

columns_to_drop = training_data.columns[training_data.isnull().any()]
training_data = training_data.drop(columns=columns_to_drop)

# Remove ds_complete
del training_data['ds_complete']
del training_data['QI']
# %%
# Label encode 'technique'
label_encoder = LabelEncoder()
training_data['technique'] = label_encoder.fit_transform(training_data['technique'])

# Define x_train, y_train_accuracy, y_train_linkability
train_metafeatures, roc_auc = training_data.iloc[:,:-2].values, training_data.iloc[:,-1].values
linkability = training_data.iloc[:,-2].values
# %%
# Create pipeline for testing data: each private configuration corresponds to a different pipeline
city_params = {'technique': ['DPGAN', 'PATEGAN'], 'QI':[0,1,2], 'epochs':[100, 200], 'batch':[50, 100], 'epsilon':[0.1, 0.5, 1.0, 5.0]}
deep_learning_params = {'technique': ['CopulaGAN', 'CTGAN', 'TVAE'], 'QI':[0,1,2], 'epochs':[100, 200], 'batch':[50, 100]}
privateSMOTE_params = {'technique':['privateSMOTE'], 'QI':[0,1,2], 'knn':[1,3,5], 'per': [1,2,3], 'epsilon':[0.1, 0.5, 1.0, 5.0, 10.0]}

keys_city, values_city = zip(*city_params.items())
city_dicts = [dict(zip(keys_city, v)) for v in itertools.product(*values_city)]
city = pd.DataFrame(city_dicts)

keys_deep, values_deep = zip(*deep_learning_params.items())
deep_dicts = [dict(zip(keys_deep, v)) for v in itertools.product(*values_deep)]
deep_learning = pd.DataFrame(deep_dicts)

keys_psmote, values_psmote = zip(*privateSMOTE_params.items())
psmote_dicts = [dict(zip(keys_psmote, v)) for v in itertools.product(*values_psmote)]
privatesmote = pd.DataFrame(psmote_dicts)
# %%    
# Function to extract metafeatures from an unseen dataset
def extract_metafeatures(unseen_dataset):
    mfe = MFE()
    mfe.fit(unseen_dataset.iloc[:, :-1].values, unseen_dataset.iloc[:, -1].values)
    ft = mfe.extract()
    unseen_metafeatures = pd.DataFrame(ft[1:], columns=ft[0])
    return unseen_metafeatures


# %%
# Add the private configurations to the metafeatures of the unseen data
ftdf = extract_metafeatures(testing_data)
# Generate unseen_data for deep_learning
unseen_data_deep = pd.DataFrame(np.repeat(ftdf.values, deep_learning.shape[0], axis=0), columns=ftdf.columns)
unseen_data_deep = pd.concat([unseen_data_deep, deep_learning], axis=1)

# Generate unseen_data for city
unseen_data_city = pd.DataFrame(np.repeat(ftdf.values, city.shape[0], axis=0), columns=ftdf.columns)
unseen_data_city = pd.concat([unseen_data_city, city], axis=1)

# Generate unseen_data for privatesmote
unseen_data_privatesmote = pd.DataFrame(np.repeat(ftdf.values, privatesmote.shape[0], axis=0), columns=ftdf.columns)
unseen_data_privatesmote = pd.concat([unseen_data_privatesmote, privatesmote], axis=1)

# Concatenate all unseen_data DataFrames
unseen_data = pd.concat([unseen_data_deep, unseen_data_city, unseen_data_privatesmote], ignore_index=True)
# %%
# Replace NaN in privacy parameters with 99999 in unseen data
unseen_data[nan_to_keep] = unseen_data[nan_to_keep].fillna(0)

unseen_data = unseen_data.drop(columns=columns_to_drop)
del unseen_data['QI']
# Label encode 'technique'
unseen_data['technique'] = label_encoder.fit_transform(unseen_data['technique'])

print(unseen_data.shape)
# %%
# Train logistic regression model on the training data for predictive performance
#lr_performance = LinearRegression()
#lr_performance.fit(train_metafeatures, roc_auc)
#predictions_performance = lr_performance.predict(unseen_data.values)
neigh = KNeighborsRegressor(n_neighbors=10, weights='distance', leaf_size=30)
neigh.fit(train_metafeatures, roc_auc)
# Predict using the logistic regression model
predictions_performance = neigh.predict(unseen_data.values)
print(predictions_performance)

#clf = SVR(kernel='rbf',C=1.0,gamma='auto',coef0=0.0, epsilon=0.1, max_iter=3).fit(train_metafeatures, roc_auc)
#predictions_performance = clf.predict(unseen_data.values)
#print(predictions_performance)

neigh_linkability = KNeighborsRegressor(n_neighbors=10, weights='distance', leaf_size=30)
neigh_linkability.fit(train_metafeatures, linkability)
# Predict using the logistic regression model
predictions_linkability = neigh_linkability.predict(unseen_data.values)
#print(predictions_linkability)

# Train logistic regression model on the training data for predictive performance
#lr_linkability = LinearRegression()
#lr_linkability.fit(train_metafeatures, linkability)

# Predict using the logistic regression model
# predictions_linkability = lr_linkability.predict(unseen_data.values)

# print("Predictions:", predictions_performance)
pred_performance = pd.DataFrame(predictions_performance, columns=['Predictions Performance'])
pred_linkability = pd.DataFrame(predictions_linkability, columns=['Predictions Linkability'])

output = pd.concat([unseen_data.iloc[:,92:], pred_performance, pred_linkability], axis=1)

output['technique'] = label_encoder.inverse_transform(output['technique'])

# output = output.sort_values(by=['Predictions Performance'], ascending=False)
# Rank the predictions based on the predicted values
# print(output.head(10))

# %%
output = output.sort_values(by=['Predictions Performance'], ascending=False)
# %%
