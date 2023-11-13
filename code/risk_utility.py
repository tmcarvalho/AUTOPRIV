# %%
import os
import pandas as pd
import numpy as np
import seaborn as sns
import re
from matplotlib import pyplot as plt


# %%
priv_results = pd.read_csv('../output/anonymeter.csv')
# priv_results = results.copy()
predictive_results = pd.read_csv('../output/results_test.csv')

# %% remove ds38, ds43, ds100
#priv_results = priv_results.loc[~priv_results.dsn.isin(['ds38', 'ds43', 'ds100'])]
#predictive_results = predictive_results.loc[~predictive_results.ds.isin(['ds38', 'ds43', 'ds100'])]
# %%
# predictive_max = predictive_results.loc[predictive_results.groupby(['ds', 'technique', 'opt_type'])['test_roc_auc'].idxmax()].reset_index(drop=True)
# %% remove "qi" from privacy results file to merge the tables correctly
priv_results['ds_complete_pred'] = priv_results['ds_complete']
priv_results['ds_complete_pred'] = priv_results['ds_complete_pred'].apply(lambda x: re.sub(r'_qi[0-9]','', x) if (('TVAE' in x) or ('CTGAN' in x) or ('copulaGAN' in x) or ('dpart' in x) or ('synthpop' in x)) else x)

# %%
priv_performance = pd.merge(priv_results, predictive_results, left_on=['technique', 'ds_complete_pred'], right_on=['technique', 'ds_complete'])

# %%
#####################################
#         PERFORMANCE FIRST         #
#####################################
performance_best = priv_performance.loc[priv_performance.groupby(['ds', 'technique','opt_type'])['test_roc_auc'].idxmax()].reset_index(drop=True)

# %%
PROPS = {
    'boxprops':{'facecolor':'#00BFC4', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}
order_technique = ['PPT', 'Copula GAN', 'TVAE', 'CTGAN', r'$\epsilon$-PrivateSMOTE']
order_optype = ['GridSearch', 'RandomSearch', 'Bayes', 'Halving', 'Hyperband']

# %%  BEST PERFORMANCE
sns.set_style("darkgrid")
plt.figure(figsize=(15,8))
ax = sns.boxplot(data=performance_best, x='technique', y='test_roc_auc',
    hue='opt_type', palette='Spectral_r', order=order_technique, hue_order=order_optype)
sns.set(font_scale=1.6)
ax.margins(y=0.02)
ax.margins(x=0.03)
ax.use_sticky_edges = False
ax.autoscale_view(scaley=True)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Predictive Performance (AUC) \n in out of sample")
sns.move_legend(ax, bbox_to_anchor=(0.5,1.15), loc='upper center', title='Optimization', borderaxespad=0., ncol=5, frameon=False)
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/bestperformance_test_optype.pdf', bbox_inches='tight')

# %% Best performance with Halving and the respective linkability
performance_best_halving = performance_best.loc[performance_best.opt_type=='Halving'].reset_index(drop=True)

sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(18,8))
sns.boxplot(ax=axes[0], data=performance_best_halving,
    x='technique', y='test_roc_auc', order=order_technique,**PROPS)
sns.boxplot(ax=axes[1], data=performance_best_halving,
    x='technique', y='value', order=order_technique, **PROPS)
sns.set(font_scale=1)
axes[0].set_ylabel("Predictive performance (AUC)")
axes[0].set_xlabel("")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
axes[1].set_ylabel("Privacy Risk (linkability)")
axes[1].set_xlabel("")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
sns.set(font_scale=1.55)
axes[0].set_ylim(0.4,1.02)
axes[1].set_ylim(-0.02,1.02)
axes[0].margins(y=0.2)
axes[0].autoscale_view(scaley=True)
axes[1].autoscale_view(scaley=True)
fig.suptitle("Best predictive performance with Halving and respective linkability")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/halving_performance_priv.pdf', bbox_inches='tight')

# %% 
#####################################
#           PRIVACY FIRST           #
#####################################
# Best linkability with the respective predictive performance w.r.t Halving
halving = priv_performance.loc[priv_performance.opt_type=='Halving'].reset_index(drop=True)
privacy_best = halving.loc[halving.groupby(['ds', 'technique'])['value'].idxmin()].reset_index(drop=True)

sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(18,8))
sns.boxplot(ax=axes[0], data=privacy_best,
    x='technique', y='value', order=order_technique,**PROPS)
sns.boxplot(ax=axes[1], data=privacy_best,
    x='technique', y='test_roc_auc', order=order_technique, **PROPS)
sns.set(font_scale=1)
axes[1].set_ylabel("Predictive performance (AUC)")
axes[1].set_xlabel("")
axes[1].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
axes[0].set_ylabel("Privacy Risk (linkability)")
axes[0].set_xlabel("")
axes[0].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
sns.set(font_scale=1.65)
#axes[1].set_ylim(0.40,1.02)
axes[0].set_ylim(-0.02,1.02)
axes[0].margins(y=0.2)
axes[0].autoscale_view(scaley=True)
axes[1].autoscale_view(scaley=True)
fig.suptitle("Best linkability with the respective predictive performance w.r.t Halving")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/priv_halving_performance.pdf', bbox_inches='tight')


# %%
#####################################
#       RANK between the two        #
#####################################

halving['rank_auc'] = halving['test_roc_auc'].rank()
halving['rank_priv'] = halving['value'].rank(ascending=False)
halving['best_rank'] = halving[['rank_auc', 'rank_priv']].mean(axis=1)
# %%
rank_best = halving.loc[halving.groupby(['ds', 'technique'])['best_rank'].idxmin()].reset_index(drop=True)

# %%
sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(18,8))
sns.boxplot(ax=axes[0], data=rank_best,
    x='technique', y='test_roc_auc', order=order_technique,**PROPS)
sns.boxplot(ax=axes[1], data=rank_best,
    x='technique', y='value', order=order_technique, **PROPS)
sns.set(font_scale=1)
axes[0].set_ylabel("Predictive performance (AUC)")
axes[0].set_xlabel("")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
axes[1].set_ylabel("Privacy Risk (linkability)")
axes[1].set_xlabel("")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
sns.set(font_scale=1.55)
#axes[0].set_ylim(0.4,1.02)
axes[1].set_ylim(-0.02,1.02)
axes[0].margins(y=0.2)
axes[0].autoscale_view(scaley=True)
axes[1].autoscale_view(scaley=True)
fig.suptitle("Best rank")
# plt.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/meanrank.pdf', bbox_inches='tight')

# %%
