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
risk_deeplearning = concat_each_file('../output/anonymeter/deep_learningk2')
# %%
risk_privateSMOTE = concat_each_file('../output/anonymeter/PrivateSMOTEk2')
# %%
risk_city = concat_each_file('../output/anonymeter/synthcityk2')
# %%
risk_deeplearning = risk_deeplearning.reset_index(drop=True)
risk_privateSMOTE = risk_privateSMOTE.reset_index(drop=True)
risk_city = risk_city.reset_index(drop=True)
# %%
risk_deeplearning['technique'] = risk_deeplearning['ds_complete'].apply(lambda x: x.split('_')[1])
risk_privateSMOTE['technique'] = 'PrivateSMOTE'
risk_city['technique'] = risk_city['ds_complete'].apply(lambda x: x.split('_')[1].upper())
# %%
results = []
results = pd.concat([risk_deeplearning, risk_privateSMOTE, risk_city])
results = results.reset_index(drop=True)
# %%
results['dsn'] = results['ds_complete'].apply(lambda x: x.split('_')[0])
# %%
results.loc[results['technique']=='CopulaGAN', 'technique'] = 'Copula GAN'
results.loc[results['technique']=='PrivateSMOTE', 'technique'] = r'$\epsilon$-PrivateSMOTE'
# %%
results.to_csv('../output_analysis/anonymeterk3.csv', index=False)
# %%
results_risk_max = results.copy()
results_risk_max = results.loc[results.groupby(['dsn', 'technique'])['value'].idxmin()].reset_index(drop=True)

# %%  BETTER IN PRIVACY
order = ['Copula GAN', 'TVAE', 'CTGAN', 'DPGAN', 'PATEGAN', r'$\epsilon$-PrivateSMOTE']

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
plt.ylabel("Linkability Risk")
plt.show()

# %%
