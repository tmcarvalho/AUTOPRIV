"""Predictive performance analysis
This script will analyse the predictive performance in the out-of-sample.
"""
# %%
import os
from os import walk
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# %% 
# process predictive results
def join_allresults(folder, technique):
    concat_results_cv = []
    concat_results_test = []
    c=0
    _,_,result_files = next(walk(f'{folder}/test/'))

    for file in result_files:
        result_cv_train = pd.read_csv(f'{folder}/validation/{file}')
        result_test = pd.read_csv(f'{folder}/test/{file}')

        # guaranteeing that we have the best model in the grid search instead of 3 models
        best_cv = result_cv_train.loc[result_cv_train['rank_test_score'] == 1,:].reset_index(drop=True)
        # for some datasets, more than one model has rank==1, keep the first
        best_cv = best_cv.iloc[:1]

        # add optimzation type to the train results
        best_cv.loc[:, 'opt_type'] = result_test['opt_type'][0]
        # get technique
        if technique=='resampling':
            best_cv.loc[:, 'technique'] = file.split('_')[1].title()
            result_test.loc[:, 'technique'] = file.split('_')[1].title()
        elif technique=='deep_learning':
            best_cv.loc[:, 'technique'] = file.split('_')[1]
            result_test.loc[:, 'technique'] = file.split('_')[1]
        else:
            best_cv.loc[:, 'technique'] = technique
            result_test.loc[:, 'technique'] = technique

        # get dataset number
        best_cv['ds'] = file.split('_')[0]
        best_cv['ds_complete'] = file
        result_test['ds'] = file.split('_')[0]
        result_test['ds_complete'] = file

        # concat each test result
        if c == 0:
            concat_results_cv = best_cv
            concat_results_test = result_test
            c += 1
        else:     
            concat_results_cv = pd.concat([concat_results_cv, best_cv])
            concat_results_test = pd.concat([concat_results_test, result_test])

    return concat_results_cv, concat_results_test    

# %% deep learning
deeplearnBO_folder = '../output/modelingBO/deep_learning/'
deeplearnCVBO, deeplearn_testBO = join_allresults(deeplearnBO_folder, 'deep_learning')

# %%
deeplearnHB_folder = '../output/modelingHB/deep_learning/'
deeplearnCVHB, deeplearn_testHB = join_allresults(deeplearnHB_folder, 'deep_learning')

# %%
deeplearnSH_folder = '../output/modelingSH/deep_learning/'
deeplearnCVSH, deeplearn_testSH = join_allresults(deeplearnSH_folder, 'deep_learning')
# %%
deeplearnGS_folder = '../output/modelingGS/deep_learning/'
deeplearnCVGS, deeplearn_testGS = join_allresults(deeplearnGS_folder, 'deep_learning')
# %%
deeplearnRS_folder = '../output/modelingRS/deep_learning/'
deeplearnCVRS, deeplearn_testRS = join_allresults(deeplearnRS_folder, 'deep_learning')


# %% concat all techniques
results_cv = pd.concat([deeplearnCVBO, deeplearnCVHB, deeplearnCVSH, deeplearnCVGS, deeplearnCVRS]).reset_index(drop=True)
results_test = pd.concat([deeplearn_testBO, deeplearn_testHB, deeplearn_testSH, deeplearn_testGS, deeplearn_testRS]).reset_index(drop=True)

# %%
results_cv.loc[results_cv['technique']=='copulaGAN', 'technique'] = 'Copula GAN'
results_test.loc[results_test['technique']=='copulaGAN', 'technique'] = 'Copula GAN'

# %% remove ds38 because it failed for the majority of opt type
results_cv = results_cv.loc[results_cv.ds != 'ds38']
results_test = results_test.loc[results_test.ds != 'ds38']
# %%
# results_cv.to_csv('../output/resultsCV.csv', index=False)
# results_test.to_csv('../output/results_test.csv', index=False)
# %%
# results_cv = pd.read_csv('../output/resultsCV.csv')

# %%
order_technique = ['Copula GAN', 'TVAE', 'CTGAN']
order_optype = ['GridSearch', 'RandomSearch', 'Bayes', 'Halving', 'Hyperband']
# %%
sns.set_style("darkgrid")
plt.figure(figsize=(15,8))
ax = sns.boxplot(data=results_cv, x='technique', y='mean_test_score', hue='opt_type', 
                 order=order_technique, hue_order=order_optype, palette="Set3")
sns.set(font_scale=1.5)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("ROC AUC")
# sns.color_palette("Paired")
sns.move_legend(ax, bbox_to_anchor=(0.5,1.15), loc='upper center', title='Optimization', borderaxespad=0., ncol=5, frameon=False)
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/performanceCV_optype.pdf', bbox_inches='tight')

# %%
sns.set_style("darkgrid")
plt.figure(figsize=(15,8))
ax = sns.boxplot(data=results_cv, x='opt_type', y='time',
                 order=order_optype, palette="Set3")
sns.set(font_scale=1.5)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Time (min)")
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/timeCV_optype.pdf', bbox_inches='tight')

# %%
sns.set_style("darkgrid")
plt.figure(figsize=(15,8))
ax = sns.boxplot(data=results_test, x='technique', y='test_roc_auc', hue='opt_type', 
                 order=order_technique, hue_order=order_optype, palette="Set3")
sns.set(font_scale=1.5)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("ROC AUC")
# sns.color_palette("Paired")
sns.move_legend(ax, bbox_to_anchor=(0.5,1.15), loc='upper center', title='Optimization', borderaxespad=0., ncol=5, frameon=False)
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/performancetest_optype.pdf', bbox_inches='tight')

# %%