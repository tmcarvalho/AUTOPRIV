import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

training_data = pd.read_csv('output/metaft.csv')
testing_data = pd.read_csv('data/original/3.csv')


lr = LogisticRegression(random_state=1)

#training_data_scaled = np.clip(training_data, -1e4, 1e4)
print(training_data.columns[training_data.isnull().any()])
#training_data_scaled=training_data_scaled.dropna()

X, y = training_data.iloc[:,:-2], training_data.iloc[:,-1]
print(X)
#print(y)
# prepare data to modeling
testing_data = testing_data.apply(LabelEncoder().fit_transform)
# testing_data = testing_data.dropna()
X_test, y_test = testing_data.iloc[:,:-1], testing_data.iloc[:,-1]

lr.fit(X, y)
# use the best estimated hyperparameter
predictions = lr.predict(X_test)

# Print or use the predictions as needed
print("Predictions:", predictions)