"""Predictive performance analysis
This script will analyse the predictive performance in the out-of-sample.
"""
# %%
import os
from os import walk
import pandas as pd
import numpy as np
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
        try:
            result_cv_train = pd.read_csv(f'{folder}/{technique}/validation/{file}')
            result_test = pd.read_csv(f'{folder}/{technique}/test/{file}')
            # select the best model in CV
            best_cv = result_cv_train.iloc[[result_cv_train['mean_test_score'].idxmax()]]
            # use the best model in CV to get the results of it in out of sample
            best_test = result_test.iloc[best_cv.index,:]
            # TODO: validate models in best_test and best_cv 
            # add optimzation type to the train results
            best_cv['opt_type'] = result_test['opt_type']

            # get technique
            if technique=='deep_learningk2':
                best_cv['technique'] = file.split('_')[1]
                best_test['technique'] = file.split('_')[1]
            elif technique=='synthcityk2':
                best_cv['technique'] = file.split('_')[1].upper()
                best_test['technique'] = file.split('_')[1].upper()
            elif technique == 'PrivateSMOTEk2':
                best_cv['technique'] = r'$\epsilon$-PrivateSMOTE'
                best_test['technique'] = r'$\epsilon$-PrivateSMOTE'
            else:
                best_cv['technique'] = technique
                best_test['technique'] = technique

            # get dataset number
            best_cv['fit_time_sum'] = result_cv_train.mean_fit_time.sum() / 60
            best_cv['ds'] = file.split('_')[0]
            best_cv['ds_complete'] = file
            best_test['ds'] = file.split('_')[0]
            best_test['ds_complete'] = file

            # concat each test result
            if c == 0:
                concat_results_cv = best_cv
                concat_results_test = best_test
                c += 1
            else:  
                concat_results_cv = pd.concat([concat_results_cv, best_cv])
                concat_results_test = pd.concat([concat_results_test, best_test])
        except Exception:
            with open('../output/modeling_failed.txt', 'a', encoding='utf-8') as failed_file:
                # Save the name of the failed file to a text file
                failed_file.write(f'{file} --- {technique} --- {result_test.opt_type[0]}\n')

    return concat_results_cv, concat_results_test

# %% Baeys optimization
BO_folder = '../output/modelingBO/'
deeplearnCVBO, deeplearn_testBO = join_allresults(BO_folder, 'deep_learningk2')
pptCVBO, ppt_testBO = join_allresults(BO_folder, 'PPT_ARX')
#privatesmoteCVBO, privatesmote_testBO = join_allresults(BO_folder, 'PrivateSMOTEk2')
#cityCVBO, city_testBO = join_allresults(BO_folder, 'synthcityk2')
origCVBO, orig_testBO = join_allresults(BO_folder, 'original')

# %% Hyperband
HB_folder = '../output/modelingHB/'
#deeplearnCVHB, deeplearn_testHB = join_allresults(HB_folder, 'deep_learningk2')
pptCVHB, ppt_testHB = join_allresults(HB_folder, 'PPT_ARX')
#privatesmoteCVHB, privatesmote_testHB = join_allresults(HB_folder, 'PrivateSMOTEk2')
#cityCVHB, city_testHB = join_allresults(HB_folder, 'synthcityk2')
origCVHB, orig_testHB = join_allresults(HB_folder, 'original')

# %% Sussessive Halving
SH_folder = '../output/modelingSH/'
deeplearnCVSH, deeplearn_testSH = join_allresults(SH_folder, 'deep_learningk2')
pptCVSH, ppt_testSH = join_allresults(SH_folder, 'PPT_ARX')
#privatesmoteCVSH, privatesmote_testSH = join_allresults(SH_folder, 'PrivateSMOTEk2')
#cityCVSH, city_testSH = join_allresults(SH_folder, 'synthcityk2')
origCVSH, orig_testSH = join_allresults(SH_folder, 'original')
# %% Grid Search
GS_folder = '../output/modelingGS/'
deeplearnCVGS, deeplearn_testGS = join_allresults(GS_folder, 'deep_learningk2')
pptCVGS, ppt_testGS = join_allresults(GS_folder, 'PPT_ARX')
#privatesmoteCVGS, privatesmote_testGS = join_allresults(GS_folder, 'PrivateSMOTEk2')
#cityCVGS, city_testGS = join_allresults(GS_folder, 'synthcityk2')
origCVGS, orig_testGS = join_allresults(GS_folder, 'original')
# %% Random Search
RS_folder = '../output/modelingRS/'
deeplearnCVRS, deeplearn_testRS = join_allresults(RS_folder, 'deep_learningk2')
pptCVRS, ppt_testRS = join_allresults(RS_folder, 'PPT_ARX')
#privatesmoteCVRS, privatesmote_testRS = join_allresults(RS_folder, 'PrivateSMOTEk2')
#cityCVRS, city_testRS = join_allresults(RS_folder, 'synthcityk2')
origCVRS, orig_testRS = join_allresults(RS_folder, 'original')

# %% concat all techniques
results_cv = pd.concat([deeplearnCVBO, deeplearnCVRS, deeplearnCVSH, deeplearnCVGS, #deeplearnCVHB,
                        pptCVBO, pptCVHB, pptCVSH, pptCVGS, pptCVRS,
                        #privatesmoteCVBO, privatesmoteCVHB, privatesmoteCVSH, privatesmoteCVRS, privatesmoteCVGS,
                        #cityCVBO, cityCVHB, cityCVGS, cityCVRS, cityCVSH
                        origCVBO, origCVHB, origCVSH, origCVGS, origCVRS,
                        ]).reset_index(drop=True)

results_test = pd.concat([deeplearn_testBO, deeplearn_testRS, deeplearn_testSH, deeplearn_testGS, #deeplearn_testHB,
                          ppt_testBO, ppt_testHB, ppt_testSH, ppt_testGS, ppt_testRS,
                          #privatesmote_testBO, privatesmote_testHB, privatesmote_testSH, privatesmote_testRS, privatesmote_testGS,
                          #city_testBO, city_testHB, city_testGS, city_testRS, city_testSH
                          orig_testBO, orig_testHB, orig_testSH, orig_testGS, orig_testRS,
                          ]).reset_index(drop=True)

# %%
results_cv.loc[results_cv['technique']=='PPT_ARX', 'technique'] = 'PPT'
results_cv.loc[results_cv['technique']=='CopulaGAN', 'technique'] = 'Copula GAN'
results_test.loc[results_test['technique']=='PPT_ARX', 'technique'] = 'PPT'
results_test.loc[results_test['technique']=='CopulaGAN', 'technique'] = 'Copula GAN'
# %% prepare to calculate percentage difference
original_results_cv = results_cv.loc[results_cv['technique']=='original'].reset_index(drop=True)
original_results_test = results_test.loc[results_test['technique']=='original'].reset_index(drop=True)
results_cv = results_cv.loc[results_cv['technique'] != 'original'].reset_index(drop=True)
results_test = results_test.loc[results_test['technique'] != 'original'].reset_index(drop=True)

# %% match ds name with transformed files
original_results_cv['ds'] = original_results_cv['ds'].apply(lambda x: f'ds{x.split(".")[0]}')
original_results_test['ds'] = original_results_test['ds'].apply(lambda x: f'ds{x.split(".")[0]}')

# %% percentage difference
results_cv['roc_auc_perdif'] = np.NaN
results_test['roc_auc_perdif'] = np.NaN
for idx in results_cv.index:
    orig_file_cv = original_results_cv.loc[(original_results_cv.ds == results_cv.ds[idx]) & (original_results_cv.opt_type == results_cv.opt_type[idx])].reset_index(drop=True)
    orig_file_test = original_results_test.loc[(original_results_test.ds == results_test.ds[idx]) & (original_results_test.opt_type == results_test.opt_type[idx])].reset_index(drop=True)

    # calculate the percentage difference
    # 100 * (Sc - Sb) / Sb
    results_cv['roc_auc_perdif'][idx] = (results_cv['mean_test_score'][idx] - orig_file_cv['mean_test_score'].iloc[0]) / orig_file_cv['mean_test_score'].iloc[0] * 100
    results_test['roc_auc_perdif'][idx] = (results_test['test_roc_auc'][idx] - orig_file_test['test_roc_auc'].iloc[0]) / orig_file_test['test_roc_auc'].iloc[0] * 100

# %%
# results_cv.to_csv('../output_analysis/resultsCV_new.csv', index=False)
# results_test.to_csv('../output_analysis/results_test_new.csv', index=False)
# %%
# results_cv = pd.read_csv('../output_analysis/resultsCV_new.csv')
# results_test = pd.read_csv('../output_analysis/results_test_new.csv')

# %% remove datasets that failed to produce synthcity variants due to a low number of singleouts
# remove_ds = ['ds8', 'ds32', 'ds24', 'ds2', 'ds59']
remove_ds = ['ds2', 'ds59', 'ds56', 'ds55', 'ds51', 'ds50', 'ds38', 'ds37', 'ds33']
results_cv = results_cv[~results_cv['ds'].isin(remove_ds)]
results_test = results_test[~results_test['ds'].isin(remove_ds)]
# %%
PROPS = {
    'boxprops':{'facecolor':'#00BFC4', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}
order_technique = ['PPT', 'Copula GAN', 'TVAE', 'CTGAN', r'$\epsilon$-PrivateSMOTE', 'DPGAN', 'PATEGAN']
order_optype = ['GridSearch', 'RandomSearch', 'Bayes', 'Halving', 'Hyperband']
# %% ROC AUC in Cross Validation
sns.set_style("darkgrid")
plt.figure(figsize=(18,10))
ax = sns.boxplot(data=results_cv, x='opt_type', y='roc_auc_perdif', hue='technique',
                 order=order_optype, hue_order=order_technique, palette="Set3")
sns.set(font_scale=2)
plt.xticks(rotation=45)
plt.xlabel("")
# plt.yscale('symlog')
plt.ylabel("Percentage difference of \n predictive performance (AUC) in CV")
sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Transformations', borderaxespad=0., frameon=False)
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/performanceCV_optypek2.pdf', bbox_inches='tight')

# %% ROC AUC in out of sample
sns.set_style("darkgrid")
plt.figure(figsize=(18,10))
ax = sns.boxplot(data=results_test, x='opt_type', y='roc_auc_perdif', hue='technique',
                 order=order_optype, hue_order=order_technique, palette="Set3")
sns.set(font_scale=2)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Predictive performance (AUC) \n in out of sample")
sns.move_legend(ax, bbox_to_anchor=(1,0.5), loc='center left', title='Transformations', borderaxespad=0., frameon=False)
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/performancetest_optype.pdf', bbox_inches='tight')

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
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/fittimeCV_optypek2.pdf', bbox_inches='tight')

# %% time during all processes in CV
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
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output_analysis/plots/timeCV_optypek2.pdf', bbox_inches='tight')

# %%
sns.set_style("darkgrid")
plt.figure(figsize=(15,8))
ax = sns.boxplot(data=results_cv, x='opt_type', y='mean_test_score')
sns.set(font_scale=1.5)
plt.xticks(rotation=45)
plt.xlabel("")
#plt.ylabel("Time (min)")
plt.show()
# %%
sns.set_style("darkgrid")
plt.figure(figsize=(15,8))
ax = sns.boxplot(data=results_test, x='opt_type', y='test_roc_auc')
sns.set(font_scale=1.5)
plt.xticks(rotation=45)
plt.xlabel("")
#plt.ylabel("Time (min)")
plt.show()
