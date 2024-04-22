# %%
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Read data
predictions = pd.read_csv(f'{os.path.dirname(os.getcwd())}/output_analysis/predictions_Halving.csv')

# %% ########################################
#           PLOT SIMPLE PARETO FRONT        #
#############################################
# data is ordered by ranking
# Identify Pareto-optimal solutions
pareto_front = []
sorted_predictions = predictions.sort_values(by=['Predictions Performance'], ascending=False)

min_linkability = float('inf')
for index, row in sorted_predictions.iterrows():
    if row['Predictions Linkability'] <= min_linkability:
        min_linkability = row['Predictions Linkability']
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
# plt.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/pareto_rank_hyperband.pdf', bbox_inches='tight')

# %%
