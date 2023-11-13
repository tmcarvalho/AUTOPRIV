# %%
from os import walk
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# %%
def concat_each_file(folder):
    """Join all results of anonymeter

    Args:
        folder (string): name of the folder

    Returns:
        pd.DataFrame: all results
    """
    _, _, input_files = next(walk(f'{folder}'))
    concat_results = pd.DataFrame()

    for file in input_files:
        risk = pd.read_csv(f'{folder}/{file}')
        risk['ds_complete']=file
        concat_results = pd.concat([concat_results, risk])

    return concat_results

# %%
risk_ppt = concat_each_file('../output/anonymeter/PPT_ARX')
# %%
risk_deeplearning = concat_each_file('../output/anonymeter/deep_learning')
# %%
risk_privateSMOTE = concat_each_file('../output/anonymeter/PrivateSMOTE')
# %%
risk_dpart = concat_each_file('../output/anonymeter/dpart')
# %%
risk_ppt = risk_ppt.reset_index(drop=True)
risk_deeplearning = risk_deeplearning.reset_index(drop=True)
risk_privateSMOTE = risk_privateSMOTE.reset_index(drop=True)
risk_dpart = risk_dpart.reset_index(drop=True)
# %%
risk_ppt['technique'] = 'PPT'
risk_deeplearning['technique'] = risk_deeplearning['ds_complete'].apply(lambda x: x.split('_')[1])
risk_privateSMOTE['technique'] = 'PrivateSMOTE'
risk_dpart['technique'] = risk_dpart['ds_complete'].apply(lambda x: x.split('_')[1])
# %%
results = []
results = pd.concat([risk_ppt, risk_deeplearning, risk_privateSMOTE, risk_dpart])
results = results.reset_index(drop=True)
# %%
results['dsn'] = results['ds_complete'].apply(lambda x: x.split('_')[0])
# %%
results.loc[results['technique']=='dpart', 'technique'] = 'Independent'
results.loc[results['technique']=='synthpop', 'technique'] = 'Synthpop'
results.loc[results['technique']=='copulaGAN', 'technique'] = 'Copula GAN'
results.loc[results['technique']=='PrivateSMOTE', 'technique'] = r'$\epsilon$-PrivateSMOTE'
# %%
# results.to_csv('../output/anonymeter.csv', index=False)
# %%
results_risk_max = results.copy()
results_risk_max = results.loc[results.groupby(['dsn', 'technique'])['value'].idxmin()].reset_index(drop=True)

# %%  BETTER IN PRIVACY
order = ['PPT', 'Copula GAN', 'TVAE', 'CTGAN', 'Independent', 'Synthpop', r'$\epsilon$-PrivateSMOTE']

sns.set_style("darkgrid")
plt.figure(figsize=(20,10))
ax = sns.boxplot(data=results_risk_max,
    x='technique', y='value', order=order, color='c')
sns.set(font_scale=2.5)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3, title='', borderaxespad=0., frameon=False)
#ax.set_yscale("symlog")
#ax.set_ylim(-0.2,150)
plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("Re-identification Risk")
plt.show()

# %%
