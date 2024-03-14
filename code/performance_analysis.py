"""Predictive performance analysis
This script will analyse the predictive performance in the out-of-sample.
"""
# %%
import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# %% 
# %%
results_cv = pd.read_csv('../output_analysis/resultsCV.csv')
results_test = pd.read_csv('../output_analysis/results_test.csv')

# %% remove datasets that failed to produce synthcity variants due to a low number of singleouts
# remove_ds = ['ds8', 'ds32', 'ds24', 'ds2', 'ds59']
# remove_ds = ['ds2', 'ds59', 'ds56', 'ds55', 'ds51', 'ds50', 'ds38', 'ds37', 'ds33']
#results_cv = results_cv[~results_cv['ds'].isin(remove_ds)]
# results_test = results_test[~results_test['ds'].isin(remove_ds)]
# %%
PROPS = {
    'boxprops':{'facecolor':'#00BFC4', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}
color_techniques = ['#26C6DA', '#AB47BC', '#FFA000', '#FFEB3B', '#9CCC65', '#E91E63']
order_technique = ['Copula GAN', 'TVAE', 'CTGAN', 'DPGAN', 'PATEGAN', r'$\epsilon$-PrivateSMOTE']
order_optype = ['GridSearch', 'RandomSearch', 'Bayes', 'Halving', 'Hyperband']
# %% ROC AUC in Cross Validation
sns.set_style("darkgrid")
plt.figure(figsize=(18,10))
ax = sns.boxplot(data=results_cv, x='opt_type', y='roc_auc_perdif', hue='technique',
                 order=order_optype, hue_order=order_technique, palette=color_techniques)
sns.set(font_scale=2.2)
plt.xticks(rotation=45)
plt.xlabel("")
# plt.yscale('symlog')
plt.ylabel("Percentage difference of \n predictive performance (AUC)")
sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Transformations', borderaxespad=0., frameon=False)
plt.show()
figure = ax.get_figure()
figure.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/performanceCV_optypek3.pdf', bbox_inches='tight')

# %% ROC AUC in out of sample
sns.set_style("darkgrid")
plt.figure(figsize=(18,10))
ax = sns.boxplot(data=results_test, x='opt_type', y='roc_auc_perdif', hue='technique',
                 order=order_optype, hue_order=order_technique, palette=color_techniques)
sns.set(font_scale=2.2)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Percentage difference of \n predictive performance (AUC)")
sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Transformations', borderaxespad=0., frameon=False)
plt.show()
figure = ax.get_figure()
figure.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/performancetest_optypek3.pdf', bbox_inches='tight')

# %% ROC AUC in out of sample -- BEST
results_test['time'] = results_cv[['time']]
results_test_best = results_test.loc[results_test.groupby(['ds', 'technique', 'opt_type'])['roc_auc_perdif'].idxmax()]
sns.set_style("darkgrid")
plt.figure(figsize=(18,10))
ax = sns.boxplot(data=results_test_best, x='opt_type', y='roc_auc_perdif', hue='technique',
                 order=order_optype, hue_order=order_technique, palette=color_techniques)
sns.set(font_scale=2.2)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Percentage difference of \n predictive performance (AUC)")
sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Transformations', borderaxespad=0., frameon=False)
plt.show()
figure = ax.get_figure()
figure.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/performancetest_optypek3_best.pdf', bbox_inches='tight')
# %% best in time during CV
sns.set_style("darkgrid")
plt.figure(figsize=(18,10))
ax = sns.boxplot(data=results_test_best, x='opt_type', y='time', hue='technique',
                 order=order_optype, hue_order=order_technique, palette=color_techniques)
sns.set(font_scale=2.2)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Time (min)")
sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Transformations', borderaxespad=0., frameon=False)
plt.show()
figure = ax.get_figure()
figure.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/performancetest_optypek3_best_time.pdf', bbox_inches='tight')

# %% fit time in CV for all
sns.set_style("darkgrid")
plt.figure(figsize=(15,8))
ax = sns.boxplot(data=results_cv, x='opt_type', y='fit_time_sum',
                 order=order_optype, **PROPS)
sns.set(font_scale=1.8)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Time (min)")
plt.show()
figure = ax.get_figure()
figure.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/fittimeCV_optypek3.pdf', bbox_inches='tight')

# %% time during all processes in CV
sns.set_style("darkgrid")
plt.figure(figsize=(15,8))
ax = sns.boxplot(data=results_cv, x='opt_type', y='time',
                 order=order_optype, **PROPS)
sns.set(font_scale=1.7)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Time (min)")
plt.show()
figure = ax.get_figure()
figure.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/timeCV_optypek3.pdf', bbox_inches='tight')

# %% time during all processes in CV per each technique
sns.set_style("darkgrid")
plt.figure(figsize=(15,8))
ax = sns.boxplot(data=results_cv, x='opt_type', y='time', hue='technique',
                 order=order_optype, hue_order=order_technique, palette=color_techniques)
sns.set(font_scale=1.7)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Time (min)")
sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Transformations', borderaxespad=0., frameon=False)
plt.show()
figure = ax.get_figure()
figure.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/timeCV_optypek3_tech.pdf', bbox_inches='tight')

# %%
