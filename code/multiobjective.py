# %%
import os
import pandas as pd
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Read data
predictions = pd.read_csv(f'{os.path.dirname(os.getcwd())}/output_analysis/predictions.csv')

# %%
# Define the optimization problem
class PrivacyConfigurationProblem(Problem):
    def __init__(self, preds):
        super().__init__(n_var=len(preds), n_obj=2, n_constr=0, xl=0, xu=1)
        self.preds = preds

    def _evaluate(self, X, out, *args, **kwargs):
        X = X[:len(self.preds)]
        print(len(X))
        # Assign the solutions to the appropriate privacy parameters
        self.preds['technique'] = X[:, 0]
        self.preds['epochs'] = X[:, 1]
        self.preds['batch'] = X[:, 2]
        self.preds['epsilon'] = X[:, 3]
        self.preds['knn'] = X[:, 4]
        self.preds['per'] = X[:, 5]

        objectives = [self.preds['Predictions Performance'].values, -self.preds['Predictions Linkability'].values]
        out["F"] = np.column_stack(objectives)
        #out["F"] = objectives


# Initialize the problem with training data and unseen data metafeatures
problem = PrivacyConfigurationProblem(predictions)

# Choose an Optimization Algorithm (NSGA-II)
algorithm = NSGA2()

# Specify Termination Criteria
termination = get_termination("n_gen", 100)

# Run the Optimization
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True
               )
# DOES NOT WORK!!
# %%
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

# %% ########################################
#           PLOT SIMPLE PARETO FRONT        #
#############################################
# data is ordered by ranking
# Identify Pareto-optimal solutions
pareto_front = []
sorted_predictions = predictions.sort_values(by=['Predictions Linkability'], ascending=True)

max_performance = float('-inf')
for index, row in sorted_predictions.iterrows():
    if row['Predictions Performance'] >= max_performance:
        max_performance = row['Predictions Performance']
        pareto_front.append(row)
pareto_front_df = pd.DataFrame(pareto_front)
# %%
# Create a scatter plot for Pareto front
sns.set_style("darkgrid")
plt.figure(figsize=(11,7))
sns.scatterplot(data=predictions, x="Predictions Performance", y="Predictions Linkability", color='#0083FF', s=70, label='All Solutions', alpha=0.65)
sns.regplot(data=pareto_front_df, x="Predictions Performance", y="Predictions Linkability", color='red', label='Pareto Front')
sns.set(font_scale=1.5)
plt.xlabel('Performance Predictions')
plt.ylabel('Linkability Predictions')
# plt.title('Pareto Front Analysis')
plt.legend()
plt.grid(True)
plt.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/pareto.pdf', bbox_inches='tight')

#plt.xticks(np.arange(min(predictions['Predictions Performance']), max(predictions['Predictions Performance'])+0.001, 0.001))
#plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))  # Adjust decimal places as needed
# %%
pareto_rank = predictions.head(15)
sns.set_style("darkgrid")
plt.figure(figsize=(11,7))
sns.scatterplot(data=predictions, x="Predictions Performance", y="Predictions Linkability", color='#0083FF', s=70, label='All Solutions', alpha=0.65)
sns.regplot(data=pareto_rank, x="Predictions Performance", y="Predictions Linkability", color='red', label='Best Rank')
sns.set(font_scale=1.5)
plt.xlabel('Performance Predictions')
plt.ylabel('Linkability Predictions')
plt.legend()
plt.grid(True)
# plt.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/pareto_rank.pdf', bbox_inches='tight')

# %%
