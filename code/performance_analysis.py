"""Predictive performance analysis
This script will analyse the predictive performance in the out-of-sample.
"""
# %%
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# %%
results_cv = pd.read_csv('../output_analysis/resultsCV.csv')
results_test = pd.read_csv('../output_analysis/results_test.csv')
priv_results = pd.read_csv('../output_analysis/anonymeterk3.csv')
# %%
results_cv["technique"]=results_cv["technique"].str.replace('PATEGAN', 'PATE-GAN')
results_test["technique"]=results_test["technique"].str.replace('PATEGAN', 'PATE-GAN')
priv_results["technique"]=priv_results["technique"].str.replace('PATEGAN', 'PATE-GAN')

results_cv["opt_type"]=results_cv["opt_type"].str.replace('RandomSearch', 'Random Search')
results_test["opt_type"]=results_test["opt_type"].str.replace('RandomSearch', 'Random Search')
results_cv["opt_type"]=results_cv["opt_type"].str.replace('GridSearch', 'Grid Search')
results_test["opt_type"]=results_test["opt_type"].str.replace('GridSearch', 'Grid Search')
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
order_technique = ['Copula GAN', 'TVAE', 'CTGAN', 'DPGAN', 'PATE-GAN', r'$\epsilon$-PrivateSMOTE']
order_optype = ['Grid Search', 'Random Search', 'Bayes', 'Halving', 'Hyperband']
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
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/performanceCV_optypek3.pdf', bbox_inches='tight')

# %% ROC AUC in out of sample
sns.set_style("darkgrid")
plt.figure(figsize=(18,10))
ax = sns.boxenplot(data=results_test, x='opt_type', y='roc_auc_perdif', hue='technique',
                 order=order_optype, hue_order=order_technique, palette=color_techniques)
sns.set(font_scale=2.2)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Percentage difference of \n predictive performance (AUC)")
sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Transformations', borderaxespad=0., frameon=False)
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/performancetest_optypek3.pdf', bbox_inches='tight')
# %%
g = sns.catplot(data=results_test, x="technique", y="roc_auc_perdif", hue='technique',
            order=order_technique, hue_order=order_technique,col="opt_type", height=7.3,aspect=.48, 
            palette=color_techniques, col_order=order_optype, legend=True,
            kind='boxen')
g.set_titles(template='{col_name}')
g.set_xticklabels('')
g.set_xlabels('')
g.set_ylabels("Percentage difference of \n predictive performance (AUC)")
plt.subplots_adjust(wspace = 0.1)
g.legend.set_title('Transformation')
sns.set(font_scale=2)
#g.fig.tight_layout()
# plt.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/performancetest_optypek3_grid.pdf', bbox_inches='tight')

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
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/performancetest_optypek3_best.pdf', bbox_inches='tight')
# %%
g = sns.catplot(data=results_test_best, x="technique", y="roc_auc_perdif", hue='technique',
            order=order_technique, hue_order=order_technique,col="opt_type", height=7.3,aspect=.48, 
            palette=color_techniques, col_order=order_optype, legend=True,
            kind='box')
g.set_titles(template='{col_name}')
g.set_xticklabels('')
g.set_xlabels('')
g.set_ylabels("Percentage difference of \n predictive performance (AUC)")
plt.subplots_adjust(wspace = 0.04)
g.legend.set_title('Transformation')
sns.set(font_scale=2)
# plt.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/performancetest_optypek3_best_grid.pdf', bbox_inches='tight')

# %% best in time during CV per technique
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
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/performancetest_optypek3_tech_best_time.pdf', bbox_inches='tight')

# %% best in time during CV per opt
sns.set_style("darkgrid")
plt.figure(figsize=(18,10))
ax = sns.boxplot(data=results_test_best, x='opt_type', y='time',
                 order=order_optype, **PROPS)
sns.set(font_scale=2.2)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Time (min)")
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/performancetest_optypek3_best_time.pdf', bbox_inches='tight')

# %% fit time (from sklearn) in CV for all
sns.set_style("darkgrid")
plt.figure(figsize=(15,8))
ax = sns.boxplot(data=results_cv, x='opt_type', y='fit_time_sum',
                 order=order_optype, **PROPS)
sns.set(font_scale=1.8)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Time (min)")
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/fittimeCV_optypek3.pdf', bbox_inches='tight')

# %% time during all processes (our time) in CV for all
sns.set_style("darkgrid")
plt.figure(figsize=(15,8))
ax = sns.boxplot(data=results_cv, x='opt_type', y='time',
                 order=order_optype, **PROPS)
sns.set(font_scale=1.7)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Time (min)")
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/timeCV_optypek3.pdf', bbox_inches='tight')

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
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/timeCV_optypek3_tech.pdf', bbox_inches='tight')

# %% best in out of sample + time in CV
sns.set_style("darkgrid")
fig, axes = plt.subplots(2, 1, figsize=(16,17))
sns.boxplot(ax=axes[0], data=results_test_best,x='opt_type',y='roc_auc_perdif', hue='technique',
            order=order_optype, hue_order=order_technique, palette=color_techniques)
sns.boxplot(ax=axes[1], data=results_test_best,x='opt_type',y='time', hue='technique',
    order=order_optype, hue_order=order_technique, palette=color_techniques)
sns.set(font_scale=2.4)
# sns.light_palette("seagreen", as_cmap=True)
axes[0].set_ylabel("Percentage difference of \n predictive performance (AUC)")
axes[0].set_xlabel("")
axes[0].set_xticklabels("")
#axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=60)
axes[1].set_ylabel("Time (min)")
axes[1].set_xlabel("")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=60)
sns.move_legend(axes[0], bbox_to_anchor=(1.35,0),loc='center right', title='Transformations', borderaxespad=0., frameon=False)
axes[1].get_legend().set_visible(False)
axes[0].use_sticky_edges = False
axes[1].use_sticky_edges = False
# plt.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/performance_time.pdf', bbox_inches='tight')

# %% best in out of sample + risk + time in CV
results_test_best_risk = results_test_best.merge(priv_results, how='left', on=['ds_complete', 'technique'])
sns.set_style("darkgrid")
fig, axes = plt.subplots(3, 1, figsize=(15.5,20))
sns.boxplot(ax=axes[0], data=results_test_best_risk,x='opt_type',y='roc_auc_perdif', hue='technique',
            order=order_optype, hue_order=order_technique, palette=color_techniques)
sns.boxplot(ax=axes[1], data=results_test_best_risk,x='opt_type',y='time', hue='technique',
    order=order_optype, hue_order=order_technique, palette=color_techniques)
sns.boxplot(ax=axes[2], data=results_test_best_risk,x='opt_type',y='value', hue='technique',
    order=order_optype, hue_order=order_technique, palette=color_techniques)
sns.set(font_scale=2)
axes[0].set_ylabel("Percentage difference of \n predictive performance (AUC)")
axes[0].set_xlabel("")
axes[0].set_xticklabels("")
axes[1].set_ylabel("Time (min)")
axes[1].set_xlabel("")
axes[1].set_xticklabels("")
axes[2].set_ylabel("Privacy Risk (Linkability)")
axes[2].set_xlabel("")
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=60)
sns.move_legend(axes[1], bbox_to_anchor=(1.35,0.5), loc='right', title='Transformations', borderaxespad=0., frameon=False)
axes[0].get_legend().set_visible(False)
axes[2].get_legend().set_visible(False)
axes[0].use_sticky_edges = False
axes[1].use_sticky_edges = False
axes[2].use_sticky_edges = False
# plt.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/performance_time_risk.pdf', bbox_inches='tight')

# %% best in out of sample + Risk
sns.set_style("darkgrid")
fig, axes = plt.subplots(2, 1, figsize=(16,17))
sns.boxplot(ax=axes[0], data=results_test_best_risk,x='opt_type',y='roc_auc_perdif', hue='technique',
            order=order_optype, hue_order=order_technique, palette=color_techniques)
sns.boxplot(ax=axes[1], data=results_test_best_risk,x='opt_type',y='value', hue='technique',
    order=order_optype, hue_order=order_technique, palette=color_techniques)
sns.set(font_scale=2.4)
axes[0].set_ylabel("Percentage difference of \n predictive performance (AUC)")
axes[0].set_xlabel("")
axes[0].set_xticklabels("")
#axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=60)
axes[1].set_ylabel("Privacy Risk (Linkability)")
axes[1].set_xlabel("")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=60)
sns.move_legend(axes[0], bbox_to_anchor=(1.35,0),loc='center right', title='Transformations', borderaxespad=0., frameon=False)
axes[1].get_legend().set_visible(False)
axes[0].use_sticky_edges = False
axes[1].use_sticky_edges = False
# plt.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/performance_risk.pdf', bbox_inches='tight')

# %%
