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
            best_cv = result_cv_train.iloc[[result_cv_train['mean_test_score'].idxmax()]].reset_index(drop=True)

            # add optimzation type to the train results
            best_cv['opt_type'] = result_test['opt_type']

            # get technique
            if technique=='deep_learningk2':
                best_cv['technique'] = file.split('_')[1]
                result_test['technique'] = file.split('_')[1]
            elif technique=='synthcityk2':
                best_cv['technique'] = file.split('_')[1].upper()
                result_test['technique'] = file.split('_')[1].upper()
            elif technique == 'PrivateSMOTEk2':
                best_cv['technique'] = r'$\epsilon$-PrivateSMOTE'
                result_test['technique'] = r'$\epsilon$-PrivateSMOTE'
            else:
                best_cv['technique'] = technique
                result_test['technique'] = technique

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
        except Exception:
            with open('../output/modeling_failed.txt', 'a', encoding='utf-8') as failed_file:
                # Save the name of the failed file to a text file
                failed_file.write(f'{file} --- {technique} --- {result_test.opt_type[0]}\n')

    return concat_results_cv, concat_results_test

# %% Baeys optimization
BO_folder = '../output/modelingBO/'
deeplearnCVBO, deeplearn_testBO = join_allresults(BO_folder, 'deep_learningk2')
pptCVBO, ppt_testBO = join_allresults(BO_folder, 'PPT_ARX')
privatesmoteCVBO, privatesmote_testBO = join_allresults(BO_folder, 'PrivateSMOTEk2')
origCVBO, orig_testBO = join_allresults(BO_folder, 'original')

# %% Hyperband
HB_folder = '../output/modelingHB/'
deeplearnCVHB, deeplearn_testHB = join_allresults(HB_folder, 'deep_learningk2')
pptCVHB, ppt_testHB = join_allresults(HB_folder, 'PPT_ARX')
privatesmoteCVHB, privatesmote_testHB = join_allresults(HB_folder, 'PrivateSMOTEk2')
cityCVHB, city_testHB = join_allresults(HB_folder, 'synthcityk2')
origCVHB, orig_testHB = join_allresults(HB_folder, 'original')

# %% Sussessive Halving
SH_folder = '../output/modelingSH/'
deeplearnCVSH, deeplearn_testSH = join_allresults(SH_folder, 'deep_learningk2')
pptCVSH, ppt_testSH = join_allresults(SH_folder, 'PPT_ARX')
privatesmoteCVSH, privatesmote_testSH = join_allresults(SH_folder, 'PrivateSMOTEk2')
origCVSH, orig_testSH = join_allresults(SH_folder, 'original')
# %% Grid Search
GS_folder = '../output/modelingGS/'
deeplearnCVGS, deeplearn_testGS = join_allresults(GS_folder, 'deep_learningk2')
pptCVGS, ppt_testGS = join_allresults(GS_folder, 'PPT_ARX')
privatesmoteCVGS, privatesmote_testGS = join_allresults(GS_folder, 'PrivateSMOTEk2')
cityCVGS, city_testGS = join_allresults(GS_folder, 'synthcityk2')
origCVGS, orig_testGS = join_allresults(GS_folder, 'original')
# %% Random Search
RS_folder = '../output/modelingRS/'
deeplearnCVRS, deeplearn_testRS = join_allresults(RS_folder, 'deep_learningk2')
pptCVRS, ppt_testRS = join_allresults(RS_folder, 'PPT_ARX')
privatesmoteCVRS, privatesmote_testRS = join_allresults(RS_folder, 'PrivateSMOTEk2')
origCVRS, orig_testRS = join_allresults(RS_folder, 'original')

# %% concat all techniques
results_cv = pd.concat([deeplearnCVBO, deeplearnCVHB, deeplearnCVSH, deeplearnCVGS, deeplearnCVRS,
                        pptCVBO, pptCVHB, pptCVSH, pptCVGS, pptCVRS,
                        privatesmoteCVBO, privatesmoteCVHB, privatesmoteCVSH, privatesmoteCVRS, privatesmoteCVGS,
                        cityCVHB, cityCVGS,
                        origCVBO, origCVHB, origCVSH, origCVGS, origCVRS]).reset_index(drop=True)

results_test = pd.concat([deeplearn_testBO, deeplearn_testHB, deeplearn_testSH, deeplearn_testGS, deeplearn_testRS,
                          ppt_testBO, ppt_testSH, ppt_testGS, ppt_testRS, ppt_testHB,
                          privatesmote_testBO, privatesmote_testHB, privatesmote_testSH, privatesmote_testRS, privatesmote_testGS,
                          city_testHB, city_testGS,
                          orig_testBO, orig_testHB, orig_testSH, orig_testGS, orig_testRS]).reset_index(drop=True)

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
    orig_file_cv = original_results_cv.loc[(original_results_cv.ds == results_cv.ds[idx]) & (original_results_cv.opt_type == results_cv.opt_type[idx])]
    orig_file_test = original_results_test.loc[(original_results_test.ds == results_test.ds[idx]) & (original_results_test.opt_type == results_test.opt_type[idx])]
    print(orig_file_cv)
    print()
    # calculate the percentage difference
    # 100 * (Sc - Sb) / Sb
    results_cv['roc_auc_perdif'][idx] = 100 * (results_cv['mean_test_score'][idx] - original_results_cv['mean_test_score'].values[0]) / original_results_cv['mean_test_score'].values[0]
    results_test['roc_auc_perdif'][idx] = 100 * (results_test['test_roc_auc'][idx] - original_results_test['test_roc_auc'].values[0]) / original_results_test['test_roc_auc'].values[0]


# %% remove datasets that failed to produce synthcity variants due to a low number of singleouts
# remove_ds = ['ds8', 'ds32', 'ds24', 'ds2', 'ds59']
# results_cv = results_cv[~results_cv['ds'].isin(remove_ds)]
# results_test = results_test[~results_test['ds'].isin(remove_ds)]
# %%
# results_cv_filtered.to_csv('../output/resultsCV.csv', index=False)
# results_test.to_csv('../output/results_test.csv', index=False)
# %%
# results_cv_filtered = pd.read_csv('../output/resultsCV.csv')

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
plt.figure(figsize=(15,8))
ax = sns.boxplot(data=results_cv, x='technique', y='roc_auc_perdif', hue='opt_type',
                 order=order_technique, hue_order=order_optype, palette="Spectral_r")
sns.set(font_scale=1.5)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Percentage difference of \n predictive performance (AUC)")
sns.move_legend(ax, bbox_to_anchor=(0.5,1.15), loc='upper center', title='Optimization', borderaxespad=0., ncol=5, frameon=False)
plt.show()
# figure = ax.get_figure()
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/performanceCV_optypek2.pdf', bbox_inches='tight')

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
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/fittimeCV_optypek2.pdf', bbox_inches='tight')

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
# figure.savefig(f'{os.path.dirname(os.getcwd())}/output/plots/timeCV_optypek2.pdf', bbox_inches='tight')

# %% ROC AUC in out of sample
sns.set_style("darkgrid")
plt.figure(figsize=(15,8))
ax = sns.boxplot(data=results_test, x='technique', y='roc_auc_perdif', hue='opt_type',
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