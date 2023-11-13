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
    _,_,result_files = next(walk(f'{folder}/{technique}/test/'))

    for file in result_files:
        result_cv_train = pd.read_csv(f'{folder}/{technique}/validation/{file}')
        result_test = pd.read_csv(f'{folder}/{technique}/test/{file}')

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
        best_cv['fit_time_sum'] = result_cv_train.mean_fit_time.sum() / 60
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
BO_folder = '../output/modelingBO/'
deeplearnCVBO, deeplearn_testBO = join_allresults(BO_folder, 'deep_learning')
pptCVBO, ppt_testBO = join_allresults(BO_folder, 'PPT_ARX')

HB_folder = '../output/modelingHB/'
deeplearnCVHB, deeplearn_testHB = join_allresults(HB_folder, 'deep_learning')
# pptCVHB, ppt_testHB = join_allresults(HB_folder, 'PPT_ARX')

SH_folder = '../output/modelingSH/'
deeplearnCVSH, deeplearn_testSH = join_allresults(SH_folder, 'deep_learning')
pptCVSH, ppt_testSH = join_allresults(SH_folder, 'PPT_ARX')

GS_folder = '../output/modelingGS/'
deeplearnCVGS, deeplearn_testGS = join_allresults(GS_folder, 'deep_learning')
pptCVGS, ppt_testGS = join_allresults(GS_folder, 'PPT_ARX')

RS_folder = '../output/modelingRS/'
deeplearnCVRS, deeplearn_testRS = join_allresults(RS_folder, 'deep_learning')
pptCVRS, ppt_testRS = join_allresults(RS_folder, 'PPT_ARX')

# %% concat all techniques
results_cv = pd.concat([deeplearnCVBO, deeplearnCVHB, deeplearnCVSH, deeplearnCVGS, deeplearnCVRS,
                        pptCVBO, #pptCVHB,
                        pptCVSH, pptCVGS, pptCVRS]).reset_index(drop=True)

results_test = pd.concat([deeplearn_testBO, deeplearn_testHB, deeplearn_testSH, deeplearn_testGS, deeplearn_testRS,
                          ppt_testBO, #ppt_testHB,
                          ppt_testSH, ppt_testGS, ppt_testRS]).reset_index(drop=True)

# %%
results_cv.loc[results_cv['technique']=='PPT_ARX', 'technique'] = 'PPT'
results_cv.loc[results_cv['technique']=='copulaGAN', 'technique'] = 'Copula GAN'
results_test.loc[results_test['technique']=='PPT_ARX', 'technique'] = 'PPT'
results_test.loc[results_test['technique']=='copulaGAN', 'technique'] = 'Copula GAN'

# %% remove some datasets because it failed for certain opt types
remove_ds = ['ds38', 'ds43']
results_cv = results_cv[~results_cv['ds'].isin(remove_ds)]
results_test = results_test[~results_test['ds'].isin(remove_ds)]
# %%
# results_cv.to_csv('../output/resultsCV.csv', index=False)
# results_test.to_csv('../output/results_test.csv', index=False)
# %%
# results_cv = pd.read_csv('../output/resultsCV.csv')

# %%
PROPS = {
    'boxprops':{'facecolor':'#00BFC4', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}
order_technique = ['PPT', 'Copula GAN', 'TVAE', 'CTGAN']
order_optype = ['GridSearch', 'RandomSearch', 'Bayes', 'Halving', 'Hyperband']
# %% ROC AUC in Cross Validation
sns.set_style("darkgrid")
plt.figure(figsize=(15,8))
ax = sns.boxplot(data=results_cv, x='technique', y='mean_test_score', hue='opt_type',
                 order=order_technique, hue_order=order_optype, palette="Spectral_r")
sns.set(font_scale=1.5)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Predictive Performance (AUC) in CV")
sns.move_legend(ax, bbox_to_anchor=(0.5,1.15), loc='upper center', title='Optimization', borderaxespad=0., ncol=5, frameon=False)
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/performanceCV_optype.pdf', bbox_inches='tight')

# %% fit time in CV
sns.set_style("darkgrid")
plt.figure(figsize=(15,8))
ax = sns.boxplot(data=results_cv, x='opt_type', y='fit_time_sum',
                 order=order_optype, **PROPS)
sns.set(font_scale=1.5)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Time (min)")
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/fittimeCV_optype.pdf', bbox_inches='tight')

# %% time during all process in CV
sns.set_style("darkgrid")
plt.figure(figsize=(15,8))
ax = sns.boxplot(data=results_cv, x='opt_type', y='time',
                 order=order_optype, **PROPS)
sns.set(font_scale=1.5)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Time (min)")
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/timeCV_optype.pdf', bbox_inches='tight')

# %% ROC AUC in out of sample
sns.set_style("darkgrid")
plt.figure(figsize=(15,8))
ax = sns.boxplot(data=results_test, x='technique', y='test_roc_auc', hue='opt_type',
                 order=order_technique, hue_order=order_optype, palette="Set3")
sns.set(font_scale=1.5)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Predictive performance (AUC) \n in out of sample")
# sns.color_palette("Paired")
sns.move_legend(ax, bbox_to_anchor=(0.5,1.15), loc='upper center', title='Optimization', borderaxespad=0., ncol=5, frameon=False)
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/performancetest_optype.pdf', bbox_inches='tight')

# %%