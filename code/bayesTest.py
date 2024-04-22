# %%
import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %% 
# percentage difference in out of sample setting
results_test = pd.read_csv(f'{os.path.dirname(os.getcwd())}/output_analysis/results_test.csv')
# %%
results_test.loc[results_test['technique']=='PATEGAN', 'technique'] = 'PATE-GAN'
results_test["opt_type"]=results_test["opt_type"].str.replace('RandomSearch', 'RS')
results_test["opt_type"]=results_test["opt_type"].str.replace('GridSearch', 'GS')
results_test["opt_type"]=results_test["opt_type"].str.replace('Bayes', 'BO')
results_test["opt_type"]=results_test["opt_type"].str.replace('Halving', 'SH')
results_test["opt_type"]=results_test["opt_type"].str.replace('Hyperband', 'HB')

# %%

def bayesian_sign_test(diff_vector, rope_min, rope_max):
    prob_left = np.mean(diff_vector < rope_min)
    prob_rope = np.mean((diff_vector >= rope_min) & (diff_vector <= rope_max))
    prob_right = np.mean(diff_vector > rope_max)
    alpha = [prob_left, prob_rope, prob_right]
    alpha = [a + 0.0001 for a in alpha]
    res = np.random.dirichlet(alpha, 30000).mean(axis=0)
    return res

def assign_hyperband(df, transfs_name):
    solution_res = pd.DataFrame(columns=['Solution', 'Result', 'Probability'])

    c = 0
    for j in range(3):
        for i in range(len(df)):
            c += 1
            if j == 0:
                solution_res.loc[c] = [transfs_name[i], 'Lose', df[i][j]]
            elif j == 1:
                solution_res.loc[c] = [transfs_name[i], 'Draw', df[i][j]]
            else:
                solution_res.loc[c] = [transfs_name[i], 'Win', df[i][j]]
    return solution_res    

def apply_test(candidates, metric, type_):
    solutions_f1 = [bayesian_sign_test(candidate, -1, 1) for candidate in candidates[metric]]
    solutions_names = candidates[type_].tolist()

    solution_res = assign_hyperband(solutions_f1, solutions_names)
    return solution_res

def custom_palette(df):
    custom_palette = {'Win': '#229954', 'Draw': '#FBC02D', 'Lose': '#2471A3'}
    return {q: custom_palette[q] for q in set(df['Result'])}

def solutions_concat(candidates, metric, type_):
    solutions = apply_test(candidates, metric, type_)
    solutions = solutions[solutions['Probability'] > 0.005]

    palette = custom_palette(solutions)   
    return solutions, palette

def sorter(column):
    reorder = [
        'Copula GAN', 'TVAE', 'CTGAN', 'DPGAN', 'PATE-GAN', r'$\epsilon$-PrivateSMOTE'
    ]
    cat = pd.Categorical(column, categories=reorder, ordered=True)
    return pd.Series(cat)

def sorter_optype(column):
    reorder = [
        'GS', 'RS', 'BO', 'SH', 'HB'
    ]
    cat = pd.Categorical(column, categories=reorder, ordered=True)
    return pd.Series(cat)

# %% oracle percentage difference in out of sample for each optimisation type
def filter_data(results, opt_type):
    """Filter data to focus on optimisation type."""
    optsearch = results.loc[results.opt_type==opt_type].reset_index(drop=True)
    return optsearch

def calculate_performance_difference(optsearch):
    """Calculate performance difference compared to the best overall performance."""
    best_optype_performance = optsearch.loc[optsearch.groupby(['ds', 'technique'])['test_roc_auc_oracle'].idxmax()].reset_index(drop=True)
    oracle_performance = best_optype_performance.loc[best_optype_performance.groupby(['ds'])["test_roc_auc_oracle"].idxmax()].reset_index(drop=True)

    best_optype_performance['test_roc_auc_perdif_oracle'] = None
    for i in range(len(best_optype_performance)):
        ds_oracle = oracle_performance.loc[(best_optype_performance.at[i,'ds'] == oracle_performance.ds),:].reset_index(drop=True)
        best_optype_performance.at[i, 'test_roc_auc_perdif_oracle'] = 100 * (best_optype_performance.at[i, 'test_roc_auc_oracle'] - ds_oracle.at[0, 'test_roc_auc_oracle']) / ds_oracle.at[0, 'test_roc_auc_oracle']

    best_optype_performance['test_roc_auc_perdif_oracle'] = best_optype_performance['test_roc_auc_perdif_oracle'].astype(np.float)
    return best_optype_performance


def visualize_data(best_optype_performance, metric_column, type_, plot_name):
    """Combine solutions and assign palettes based on performance metric."""
    solutions_org_candidates, palette_candidates = solutions_concat(best_optype_performance, metric_column, type_)   
    solutions_org_candidates = solutions_org_candidates.reset_index(drop=True)
    solutions_org_candidates = solutions_org_candidates.sort_values(by="Solution", key=sorter)

    sns.set_style("darkgrid")
    fig, ax= plt.subplots(figsize=(8.5, 3))
    sns.histplot(data=solutions_org_candidates, stat='probability', multiple='fill', x='Solution', hue='Result', edgecolor='none',
                palette = palette_candidates, shrink=0.9, hue_order=['Lose', 'Draw'])
    ax.axhline(0.5, linewidth=0.5, color='lightgrey')
    ax.margins(x=0.2)
    ax.set_xlabel("")
    ax.set_ylabel('Proportion of probability')
    sns.move_legend(ax, bbox_to_anchor=(0.5,1.23), loc='upper center', borderaxespad=0., ncol=3, frameon=False, title="")         
    sns.set(font_scale=1.35)
    plt.xticks(rotation=30)
    figure = ax.get_figure()
    figure.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/{plot_name}.pdf', bbox_inches='tight')

# %% # Is there any solution that is always better?
# Grid Search
gridsearch = filter_data(results_test, 'GS')
best_performance = calculate_performance_difference(gridsearch)
visualize_data(best_performance,'test_roc_auc_perdif_oracle', 'technique', 'bayes_gridsearch')
# %% Halving
halving = filter_data(results_test, 'SH')
best_performance = calculate_performance_difference(halving)
visualize_data(best_performance,'test_roc_auc_perdif_oracle', 'technique', 'bayes_halving')
# %% Hyperband
hyperband = filter_data(results_test, 'HB')
best_performance = calculate_performance_difference(hyperband)
visualize_data(best_performance,'test_roc_auc_perdif_oracle', 'technique','bayes_hyperband')
# %%
def calculate_performance_difference_optype(optsearch):
    """Calculate performance difference compared to the best overall performance."""
    best_optype_performance = optsearch.loc[optsearch.groupby(['ds', 'opt_type'])['test_roc_auc_oracle'].idxmax()].reset_index(drop=True)
    # gridsearch = filter_data(optsearch, 'Halving')
    oracle_performance = best_optype_performance.loc[best_optype_performance.groupby(['ds'])["test_roc_auc_oracle"].idxmax()].reset_index(drop=True)
    # print(oracle_performance)
    # print(best_optype_performance)
    best_optype_performance['test_roc_auc_perdif_oracle'] = None
    for i in range(len(best_optype_performance)):
        ds_oracle = oracle_performance.loc[(best_optype_performance.at[i,'ds'] == oracle_performance.ds),:].reset_index(drop=True)
        best_optype_performance.at[i, 'test_roc_auc_perdif_oracle'] = 100 * (best_optype_performance.at[i, 'test_roc_auc_oracle'] - ds_oracle.at[0, 'test_roc_auc_oracle']) / ds_oracle.at[0, 'test_roc_auc_oracle']

    best_optype_performance['test_roc_auc_perdif_oracle'] = best_optype_performance['test_roc_auc_perdif_oracle'].astype(np.float)
    return best_optype_performance

def visualize_data(best_optype_performance, metric_column, type_, plot_name):
    """Combine solutions and assign palettes based on performance metric."""
    solutions_org_candidates, palette_candidates = solutions_concat(best_optype_performance, metric_column, type_)   
    solutions_org_candidates = solutions_org_candidates.reset_index(drop=True)
    solutions_org_candidates = solutions_org_candidates.sort_values(by="Solution", key=sorter_optype)
    sns.set_style("darkgrid")
    fig, ax= plt.subplots(figsize=(6.8, 2.8))
    sns.histplot(data=solutions_org_candidates, stat='probability', multiple='fill', x='Solution', hue='Result', edgecolor='none',
                palette = palette_candidates, shrink=0.9, hue_order=['Lose', 'Draw',])
    ax.axhline(0.5, linewidth=0.5, color='lightgrey')
    ax.margins(x=0.2)
    ax.set_xlabel("")
    ax.set_ylabel('Proportion of probability')
    sns.move_legend(ax, bbox_to_anchor=(0.5,1.23), loc='upper center', borderaxespad=0., ncol=3, frameon=False, title="")         
    sns.set(font_scale=1.2)
    # plt.xticks(rotation=45)
    figure = ax.get_figure()
    figure.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/{plot_name}.pdf', bbox_inches='tight')

# %%
best_performance = calculate_performance_difference_optype(results_test)
# best_performance = best_performance.loc[best_performance.opt_type!='Halving']
visualize_data(best_performance,'test_roc_auc_perdif_oracle', 'opt_type', 'bayes_optype')
# %%