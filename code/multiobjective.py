# %%
import os
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LinearRegression
import itertools
from pymfe.mfe import MFE
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Read data
training_data = pd.read_csv(f'{os.path.dirname(os.getcwd())}/output_analysis/metaftk3.csv')
testing_data = pd.read_csv(f'{os.path.dirname(os.getcwd())}/data/original/3.csv')

# %%
# Replace NaN in privacy parameters with 99999
training_data[['QI', 'epochs', 'batch', 'knn', 'per', 'epsilon']] = training_data[['QI', 'epochs', 'batch', 'knn', 'per', 'epsilon']].fillna(99999)

columns_to_drop = training_data.columns[training_data.isnull().any()]
training_data = training_data.drop(columns=columns_to_drop)

# Remove ds_complete
del training_data['ds_complete']
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
# Replace NaN in privacy parameters with 99999
unseen_data[['QI', 'epochs', 'batch', 'knn', 'per', 'epsilon']] = unseen_data[['QI', 'epochs', 'batch', 'knn', 'per', 'epsilon']].fillna(99999)

unseen_data = unseen_data.drop(columns=columns_to_drop)

# Label encode 'technique'
unseen_data['technique'] = label_encoder.fit_transform(unseen_data['technique'])

print(unseen_data.shape)
# %%
# Define the optimization problem
class PrivacyConfigurationProblem(Problem):
    def __init__(self, metafeatures, roc_auc_values, linkability_values, unseen_metafeatures):
        super().__init__(n_var=len(unseen_metafeatures), n_obj=2, n_constr=0, xl=0, xu=1)
        #print(unseen_metafeatures.shape[1])
        self.metafeatures = metafeatures
        self.roc_auc_values = roc_auc_values
        self.linkability_values = linkability_values
        self.unseen_metafeatures = unseen_metafeatures

    def _evaluate(self, X, out, *args, **kwargs):
        # Evaluate each privacy configuration
        n_configs = len(X)
        objectives = np.zeros((n_configs, 2))
        # Predict accuracy based on the relationship learned from training data
        performance_model = predict_accuracy(self.metafeatures, self.roc_auc_values)
        # Estimate privacy risk based on the metafeatures of the unseen data
        privacy_model = predict_privacy(self.metafeatures, self.linkability_values)

        for i in range(n_configs):
            configuration = self.unseen_metafeatures[i].reshape(1, -1)
            # Predict for unseen data
            predicted_performance = performance_model.predict(configuration)
            predicted_linkability = privacy_model.predict(configuration)
            print(predicted_performance)
            print(predicted_linkability)
            #accuracy_distance = np.abs(predicted_performance - X[i, 0])
            #linkability_distance = np.abs(predicted_linkability - X[i, 1])

            objectives[i] = [predicted_performance, predicted_linkability]
        
        out["F"] = objectives

# Define functions for predicting accuracy and estimating privacy risk
def predict_accuracy(metafeatures_train, accuracy_values_train):
    # Train logistic regression model on the training data for predictive performance
    lr_performance = LinearRegression()
    lr_performance.fit(metafeatures_train, accuracy_values_train)
    return lr_performance

def predict_privacy(metafeatures_train, likability_values):
    # Implement your privacy risk estimation model here
    # Train logistic regression model on the training data for linkability
    lr_linkability = LinearRegression()
    lr_linkability.fit(metafeatures_train, likability_values)
    return lr_linkability

# Initialize the problem with training data and unseen data metafeatures
problem = PrivacyConfigurationProblem(train_metafeatures, roc_auc, linkability, unseen_data.values)

# Choose an Optimization Algorithm (NSGA-II)
algorithm = NSGA2()

# Specify Termination Criteria
termination = get_termination("n_gen", 100)

# Run the Optimization
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               #verbose=True
               )
# %%
#print(res.algorithm.n_gen)
# Plot all solutions
plt.figure(figsize=(8, 6))
plt.scatter(res.F[:,0], res.F[:,1], color='blue')
plt.title('All Solutions')
plt.xlabel('Accuracy Distance')
plt.ylabel('Linkability Distance')
plt.grid(True)

# Plot the non-dominated solutions
#non_dominated = res.F[np.where(res.F[:, 0] <= np.min(res.F[:, 0]))]
#plt.scatter(non_dominated[:,0], non_dominated[:,1], color='red', label='Non-dominated Solutions')

plt.legend()
plt.show()



# %%
# Get the optimal solution
optimal_solution = res.X[np.argmin(res.F[:, 0])]  # Select the solution with the minimum accuracy distance

# Print the optimal solution
print("Optimal Accuracy:", optimal_solution[0])
print("Optimal Privacy Risk:", optimal_solution[1])


# %%

# Extract the Pareto front solutions
pareto_front = res.F
pareto_front_df = pd.DataFrame(pareto_front, columns=['Performance', 'Linkability'])
print(len(pareto_front_df))
#pareto_front_df.Linkability = -pareto_front_df.Linkability
print(pareto_front_df)
# Plot the Pareto front
sns.scatterplot(data=pareto_front_df, x='Performance', y='Linkability')
plt.show()
# %%
# Print the best solutions (non-dominated fronts)
print("Best solutions:")
for i, sol in enumerate(res.X):
    print(f"Solution {i+1}: {sol}")
    print(f"Objectives: {res.F[i]}")
    print("-------------------------")

# Select the desired solution based on your preference (e.g., highest accuracy with acceptable privacy)
best_params = res.X[0]  # Replace with your selection criteria
# %%
