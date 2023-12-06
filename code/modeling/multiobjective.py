import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from pymfe.mfe import MFE
from sklearn.metrics.pairwise import euclidean_distances
from pymoo.algorithms.moo import nsga2
from pymoo.problems import get_problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from sklearn.metrics import mean_squared_error


# Define the accuracy and linkability_risk functions
def accuracy(x):
    return x[0] ** 2

def linkability_risk(x):
    return (x[0] - 2) ** 2

# Read data
training_data = pd.read_csv('output/metaft.csv')
holdout_data = pd.read_csv('data/original/3.csv')

# Drop columns
columns_to_drop = ['can_cor.sd', 'cor.mean', 'cor.sd', 'g_mean.mean', 'g_mean.sd',
                   'h_mean.mean', 'h_mean.sd', 'kurtosis.mean', 'kurtosis.sd',
                   'linear_discr.mean', 'linear_discr.sd', 'num_to_cat', 'sd_ratio',
                   'skewness.mean', 'skewness.sd']
training_data = training_data.drop(columns=columns_to_drop)

# Extract features from training data
x_train, y_train = training_data.iloc[:, :-2].values, training_data.iloc[:, -1].values

# Extract features from holdout set data
mfe = MFE()
mfe.fit(holdout_data.iloc[:, :-1].values, holdout_data.iloc[:, -1].values)
ft = mfe.extract()
ftdf_holdout = pd.DataFrame(ft[1:], columns=ft[0])
ftdf_holdout = ftdf_holdout.drop(columns=columns_to_drop)

# Define the multi-objective problem for optimization
problem = get_problem("dtlz1", n_var=1, n_obj=2, xl=0, xu=5, func=lambda x, _: [accuracy(x), linkability_risk(x)])

# Perform the optimization using NSGA-II
algorithm = nsga2(pop_size=100)
result = minimize(problem, algorithm, ('n_gen', 100), seed=1, verbose=True)

# Extract the Pareto front solutions
pareto_front = result.F

# Print the Pareto front solutions
print("Pareto Front:")
for solution in pareto_front:
    print(solution)

# Select the best solution based on a preference for privacy and predictive performance
best_solution_index = np.argmin(pareto_front[:, 1])  # Choose based on minimizing linkability_risk

# Get the decision variable value for the best solution
best_solution_x = result.X[best_solution_index]

# Train a model using the best solution
lr_best = LinearRegression()
lr_best.fit(x_train, y_train)

# Predict using the best solution on the holdout set
predictions_holdout = lr_best.predict(best_solution_x.reshape(1, -1))

# Print or use the predictions as needed
print("Best Solution Predictions on Holdout Set:", predictions_holdout)
