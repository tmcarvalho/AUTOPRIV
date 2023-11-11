# %%
from synthcity.plugins import Plugins

# %%
from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True, as_frame=True)
X["target"] = y

# %%
Plugins(categories=["generic", "privacy"]).list()
# %%
syn_model = Plugins().get("ctgan", n_iter=100, batch_size=50)

# %%
syn_model.fit(X)
# %%
new_data = syn_model.generate(count = len(X))
# %%
new_data
# %%
X
# %%
Plugins().list()

# %%
from synthcity.metrics import Metrics
# %%
Metrics().list()
# %%
Metrics().evaluate(X_gt=X, X_syn=new_data, metrics={'privacy':['k-anonymization', 'distinct l-diversity', 'identifiability_score']})
