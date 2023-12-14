import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LinearRegression
import itertools
from pymfe.mfe import MFE
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns


# Define the accuracy and linkability_risk functions
# def accuracy(x):
#     print(x)
#     return x[0] ** 2

# def linkability_risk(x):
#     return (x[0] - 2) ** 2

# Read data
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
     
# Extract features from holdout data set
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

# Train logistic regression model on the training data for predictive performance
lr_performance = LinearRegression()
lr_performance.fit(x_train, y_train)

# Predict using the logistic regression model
predictions_performance = lr_performance.predict(unseen_data.values)

# Train logistic regression model on the training data for predictive performance
lr_linkability = LinearRegression()
lr_linkability.fit(x_train, linkability)

# Predict using the logistic regression model
predictions_linkability = lr_linkability.predict(unseen_data.values)

# print("Predictions:", predictions_performance)
pred_performance = pd.DataFrame(predictions_performance, columns=['Predictions Performance'])
pred_linkability = pd.DataFrame(predictions_linkability, columns=['Predictions Linkability'])

output = pd.concat([unseen_data.iloc[:,96:], pred_performance, pred_linkability], axis=1)

output['technique'] = label_encoder.inverse_transform(output['technique'])

# output = output.sort_values(by=['Predictions Performance'], ascending=False)
# Rank the predictions based on the predicted values
print(output.head(10))


# Define the multi-objective problem for optimization
class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=output.shape[0], n_obj=2, xl=0, xu=5, type_var=np.float)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = output['Predictions Performance'].values
        f2 = -output['Predictions Linkability'].values
        out["F"] = np.column_stack([f1, f2])

# Instantiate the problem
problem = MyProblem()

# Define the optimization algorithm (NSGA-II)
algorithm = NSGA2(pop_size=output.shape[0])

# Optimize the objectives
result = minimize(problem, algorithm, ('n_gen', 200), seed=1, verbose=True)

# Get the optimal solutions
optimal_solutions = result.X
optimal_accuracy = result.F[:, 0]  # Negate to get the actual accuracy values
optimal_linkability = -result.F[:, 1]  # Negate to get the actual linkability values

# Print or use the optimal solutions as needed
for i, solution in enumerate(optimal_solutions):
    print(f"Solution {i + 1}: Accuracy={optimal_accuracy[i]}, Linkability={optimal_linkability[i]}")

# Extract the Pareto front solutions
pareto_front = result.F
pareto_front_df = pd.DataFrame(pareto_front, columns=['Performance', 'Linkability'])
print(len(pareto_front_df))
pareto_front_df.Linkability = -pareto_front_df.Linkability
print(pareto_front_df)
# Plot the Pareto front
sns.scatterplot(data=pareto_front_df, x='Performance', y='Linkability')
plt.show()