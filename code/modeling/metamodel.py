import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pymfe.mfe import MFE
from sklearn.metrics.pairwise import euclidean_distances

training_data = pd.read_csv('output_analysis/metaft.csv')
testing_data = pd.read_csv('data/original/3.csv')

#training_data_scaled = np.clip(training_data, -1e4, 1e4)
print(training_data.columns[training_data.isnull().any()])

columns_to_drop = ['can_cor.sd', 'cor.mean', 'cor.sd', 'g_mean.mean', 'g_mean.sd',
       'h_mean.mean', 'h_mean.sd', 'kurtosis.mean', 'kurtosis.sd',
       'linear_discr.mean', 'linear_discr.sd', 'num_to_cat', 'sd_ratio',
       'skewness.mean', 'skewness.sd']
training_data = training_data.drop(columns=columns_to_drop)

x_train, y_train = training_data.iloc[:,:-3].values, training_data.iloc[:,-2].values

# Extract features from testing data
mfe = MFE()
mfe.fit(testing_data.iloc[:, :-1].values, testing_data.iloc[:, -1].values)
ft = mfe.extract()
ftdf = pd.DataFrame(ft[1:], columns=ft[0])

ftdf = ftdf.drop(columns=columns_to_drop)

# Transform testing data using the same scaler
#X_test_scaled = scaler.transform(ftdf)
print(x_train)
# Train logistic regression model on the scaled training data
lr = LinearRegression()
lr.fit(x_train, y_train)

# Predict using the logistic regression model
predictions = lr.predict(ftdf.values)

# Print or use the predictions as needed
print("Predictions:", predictions)

# Calculate the Euclidean distances between the test example and all training examples
distances = euclidean_distances(x_train, ftdf.values.reshape(1, -1))

# Get the indices of the top 10 training examples with the smallest distances
top_10_indices = np.argsort(distances.flatten())[:10]
print(top_10_indices)

print(training_data.iloc[top_10_indices,-1])