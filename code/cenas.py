# %%
from os import walk
import pandas as pd
# %%
_,_,result_files = next(walk('../data/synthcityk2'))
# %%
ds = [file.split('_')[0] for file in result_files]
# %%
from collections import Counter
x = Counter([element for element in ds])
# %%
_,_,files = next(walk('../data/deep_learningk2'))
for file in files:
    dsss = pd.read_csv(f'../data/deep_learningk2/{file}')
    print(dsss.shape, file)
    print(dsss.shape[0]*0.8)
# %%
# %%
_,_,files = next(walk('../data/original'))
import pandas as pd
for file in files:
    dss = pd.read_csv(f'../data/original/{file}')
    print(dss.shape, file)
    print(dss.shape[0]*0.8)
# %%
