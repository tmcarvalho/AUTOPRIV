# %%
from os import walk
_,_,result_files = next(walk('../data/synthcityk2'))
# %%
ds = [file.split('_')[0] for file in result_files]
# %%
from collections import Counter
x = Counter([element for element in ds])
# %%
_,_,files = next(walk('../data/PrivateSMOTEk2'))
import pandas as pd
for file in files:
    dss = pd.read_csv(f'../data/PrivateSMOTEk2/{file}')
    print(dss.shape, file)
# %%