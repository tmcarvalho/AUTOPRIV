"""Predictive performance
This script will test the predictive performance of each data set with Hyperband hyperparameter optimization.
"""
import warnings
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.experimental import enable_halving_search_cv #noqa
from sklearn.model_selection import HalvingRandomSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings(action='ignore', category=FutureWarning)

# %% evaluate a model
def evaluate_model_hb(x_train, x_test, y_train, y_test):
    """Evaluatation

    Args:
        x_train (pd.DataFrame): dataframe for train
        x_test (pd.DataFrame): dataframe for test
        y_train (np.int64): target variable for train
        y_test (np.int64): target variable for test
    Returns:
        tuple: dictionary with validation, train and test results
    """

    seed = np.random.seed(1234)

    # initiate models
    rfc = RandomForestClassifier(random_state=seed)
    booster = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=seed)

    reg = LogisticRegression(random_state=seed)
    nnet = MLPClassifier(random_state=seed)

    n_feat = x_train.shape[1]

    # set parameterisation
    # param1 = {}
    # param1['classifier__n_estimators'] = [100, 250, 500, 750, 1000]
    # param1['classifier__max_depth'] = [6, 8, 10, 12]
    # param1['classifier__learning_rate'] = [0.01, 0.1, 0.3, 0.5]
    # param1['classifier'] = [xgb]

    param2 = {}
    param2['classifier__alpha'] = [1e-3, 1e-4, 1e-5]
    param2['classifier__max_iter'] = [1000, 10000, 1000000]
    param2['classifier__eta0'] = [0, 0.01, 0.1, 0.5, 1]
    param2['classifier'] = [sdg]
    
    param3 = {}
    param3['classifier__n_estimators'] = [100, 250, 500, 750, 1000]
    param3['classifier__max_depth'] = [6, 8, 10, 12]
    param3['classifier__learning_rate'] = [0.01, 0.1, 0.3, 0.5]
    param3['classifier'] = [grb]

    param4 = {}
    param4['classifier__hidden_layer_sizes'] = [[n_feat // 2], [int(n_feat * (2 / 3))], [n_feat]]
    param4['classifier__alpha'] = [1e-2, 1e-3, 1e-4]
    param4['classifier__max_iter'] = [1000, 10000, 1000000]
    param4['classifier'] = [nnet]

    # define metric functions -- doens't accept multi measures
    scoring = make_scorer(roc_auc_score, max_fpr=0.001, needs_proba=True)

    pipeline = Pipeline([('classifier', sdg)])
    params = [param2, param3, param4]

    print("Start modeling with CV")
    training = time.time()

    # Train the grid search model
    grid = HalvingRandomSearchCV(
        pipeline,
        param_distributions=params,
        cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=42),
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1).fit(x_train, y_train)

    print(f'It takes {(time.time() - training)/60} minutes')

    score_cv = {
    'params':[], 'model':[], 'test_roc_auc':[]
    }
    # Store results from grid search
    validation = pd.DataFrame(grid.cv_results_)
    validation['model'] = validation['param_classifier']
    # validation['model'] = validation['model'].apply(lambda x: 'XGBoost' if 'XGB' in str(x) else x)
    validation['model'] = validation['model'].apply(lambda x: 'Stochastic Gradient Descent' if 'SGDClassifier' in str(x) else x)
    validation['model'] = validation['model'].apply(lambda x: 'Gradient Boosting' if 'GradientBoostingClassifier' in str(x) else x)
    validation['model'] = validation['model'].apply(lambda x: 'Neural Network' if 'MLPClassifier' in str(x) else x)
    validation['time'] = (time.time() - training)/60

    print("Start modeling in out of sample")
    outofsample = time.time()
    for i in range(len(validation)):
        # set each model for prediction on test
        clf_best = grid.best_estimator_.set_params(**grid.cv_results_['params'][i]).fit(x_train, y_train)
        clf = clf_best.predict(x_test)
        score_cv['params'].append(str(grid.cv_results_['params'][i]))
        score_cv['model'].append(validation.loc[i, 'model'])
        score_cv['test_roc_auc'].append(roc_auc_score(y_test, clf))
        score_cv['time'] = (time.time() - outofsample)/60

    score_cv = pd.DataFrame(score_cv)

    return [validation, score_cv]
