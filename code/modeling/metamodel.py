import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LinearRegression
from pymfe.mfe import MFE
import itertools

training_data = pd.read_csv('output_analysis/metaft.csv')
testing_data = pd.read_csv('data/original/3.csv')

print(training_data.columns[training_data.isnull().any()])

columns_to_drop = ['can_cor.sd', 'cor.mean', 'cor.sd', 'g_mean.mean', 'g_mean.sd',
       'h_mean.mean', 'h_mean.sd', 'kurtosis.mean', 'kurtosis.sd',
       'linear_discr.mean', 'linear_discr.sd', 'num_to_cat', 'sd_ratio',
       'skewness.mean', 'skewness.sd']
training_data = training_data.drop(columns=columns_to_drop)

# Replace NaN in privacy parameters with 99999
training_data = training_data.fillna(99999)

# Remove ds_complete
del training_data['ds_complete']

# Label encode 'technique'
label_encoder = LabelEncoder()
training_data['technique'] = label_encoder.fit_transform(training_data['technique'])

# Define x_train and y_train
x_train, y_train = training_data.iloc[:,:-2].values, training_data.iloc[:,-1].values
linkability = training_data.iloc[:,-2].values

# Create pipeline for testing data: each private configuration corresponds to a different pipeline
city_params = {'technique': ['DPGAN', 'PATEGAN'], 'QI':[0,1,2], 'epochs':[100, 200], 'batch':[50, 100], 'epsilon':[0.1, 0.5, 0.25, 0.75, 1]}
deep_learning_params = {'technique': ['CopulaGAN', 'CTGAN', 'TVAE'], 'QI':[0,1,2], 'epochs':[100, 200], 'batch':[50, 100]}
arx_params = {'technique':['transf', 'transf', 'transf'], 'n_transf':[0,1,2], 'QI':[0,1,2]}
privateSMOTE_params = {'technique':['privateSMOTE'], 'QI':[0,1,2], 'knn':[], 'per': []}

keys_city, values_city = zip(*city_params.items())
city_dicts = [dict(zip(keys_city, v)) for v in itertools.product(*values_city)]
city=pd.DataFrame(city_dicts)
keys_deep, values_deep = zip(*deep_learning_params.items())
deep_dicts = [dict(zip(keys_deep, v)) for v in itertools.product(*values_deep)]
deep_learning=pd.DataFrame(deep_dicts)
keys_arx, values_arx = zip(*arx_params.items())
arx_dicts = [dict(zip(keys_arx, v)) for v in itertools.product(*values_arx)]
arx=pd.DataFrame(arx_dicts)
keys_psmote, values_psmote = zip(*privateSMOTE_params.items())
psmote_dicts = [dict(zip(keys_psmote, v)) for v in itertools.product(*values_psmote)]
privatesmote=pd.DataFrame(psmote_dicts)
     
# Extract features from testing data
mfe = MFE()
mfe.fit(testing_data.iloc[:, :-1].values, testing_data.iloc[:, -1].values)
ft = mfe.extract()
ftdf = pd.DataFrame(ft[1:], columns=ft[0])
ftdf = ftdf.drop(columns=columns_to_drop)

# Add the private configurations to the metafeatures of the unseen data
unseen_data = pd.DataFrame(np.repeat(ftdf.values, deep_learning.shape[0], axis=0), columns=ftdf.columns)

unseen_data = pd.concat([unseen_data, deep_learning], axis=1)

# TODO: merge other PPTS to unseen data

# Replace NaN in privacy parameters with 99999
unseen_data = unseen_data.fillna(99999)

# Label encode 'technique'
unseen_data['technique'] = label_encoder.fit_transform(unseen_data['technique'])

# Train logistic regression model on the scaled training data
lr = LinearRegression()
lr.fit(x_train, y_train)

# Predict using the logistic regression model
predictions = lr.predict(unseen_data.values)

print("Predictions:", predictions)
pred = pd.DataFrame(predictions, columns=['Predictions'])

output = pd.concat([unseen_data.iloc[:,96:], pred], axis=1)

output['technique'] = label_encoder.inverse_transform(output['technique'])

output = output.sort_values(by=['Predictions'], ascending=False)
# Rank the predictions based on the predicted values
print(output.head(10))