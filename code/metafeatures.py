from pymfe.mfe import MFE
import pandas as pd

data = pd.read_csv('data/original/2.csv')
X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
mfe = MFE()
mfe.fit(X, y)
ft = mfe.extract()
print(len(ft[0]))
ftdf = pd.DataFrame(ft[1:], columns=ft[0])
ftdf.to_csv('output/metaft.csv', index=False)



allmfe = MFE(groups="all", summary="all")
allmfe.fit(X, y)
allft = allmfe.extract()
print(len(allft[0]))
allftdf = pd.DataFrame(allft[1:], columns=allft[0])
allftdf.to_csv('output/allmetaft.csv', index=False)

