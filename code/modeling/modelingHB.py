"""Predictive performance
This script will test the predictive performance of each data set with Hyperband hyperparameter optimization.
"""
import warnings
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.experimental import enable_halving_search_cv #noqa
from sklearn.model_selection import HalvingRandomSearchCV, RepeatedKFold
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
            #early_stopping_rounds=10)

    reg = LogisticRegression(random_state=seed)
    nnet = MLPClassifier(random_state=seed,
                         early_stopping=True,
                         n_iter_no_change=10)

    n_feat = x_train.shape[1]

    # set parameterisation
    param1 = {}
    param1['classifier__n_estimators'] = [100, 250, 500]
    param1['classifier__max_depth'] = [4, 7, 10]
    param1['classifier'] = [rfc]

    param2 = {}
    param2['classifier__n_estimators'] = [100, 250, 500]
    param2['classifier__max_depth'] = [4, 7, 10]
    param2['classifier__learning_rate'] = [0.01, 0.1]
    param2['classifier'] = [booster]

    param3 = {}
    param3['classifier__C'] = np.logspace(-4, 4, 3)
    param3['classifier__max_iter'] = [1000000, 100000000]
    param3['classifier'] = [reg]

    param4 = {}
    param4['classifier__hidden_layer_sizes'] = [[n_feat // 2], [int(n_feat * (2 / 3))], [n_feat]]
    param4['classifier__alpha'] = [1e-2, 1e-3, 1e-4]
    param4['classifier__max_iter'] = [10000, 100000]
    param4['classifier'] = [nnet]

    # define metric functions -- doens't accept multi measures
    scoring = make_scorer(roc_auc_score, max_fpr=0.001, needs_proba=True)

    pipeline = Pipeline([('classifier', rfc)])
    params = [param1, param2, param3, param4]

    print("Start modeling with CV")
    start = time.time()

    # Train the grid search model
    grid = HalvingRandomSearchCV(
        pipeline,
        param_distributions=params,
        cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=1),
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1).fit(x_train, y_train)

    print(f'It takes {(time.time() - start)/60} minutes')

    score_cv = {
    'params':[], 'model':[], 'test_roc_auc':[]
    }
    # Store results from grid search
    validation = pd.DataFrame(grid.cv_results_)
    validation['model'] = validation['param_classifier']
    validation['model'] = validation['model'].apply(lambda x: 'Random Forest' if 'RandomForest' in str(x) else x)
    validation['model'] = validation['model'].apply(lambda x: 'XGBoost' if 'XGB' in str(x) else x)
    validation['model'] = validation['model'].apply(lambda x: 'Logistic Regression' if 'Logistic' in str(x) else x)
    validation['model'] = validation['model'].apply(lambda x: 'Neural Network' if 'MLPClassifier' in str(x) else x)
    validation['time'] = (time.time() - start)/60

    print("Start modeling in out of sample")

    for i in range(len(validation)):
        # set each model for prediction on test
        clf_best = grid.best_estimator_.set_params(**grid.cv_results_['params'][i]).fit(x_train, y_train)
        clf = clf_best.predict(x_test)
        score_cv['params'].append(str(grid.cv_results_['params'][i]))
        score_cv['model'].append(validation.loc[i, 'model'])
        score_cv['test_roc_auc'].append(roc_auc_score(y_test, clf))

    score_cv = pd.DataFrame(score_cv)

    return [validation, score_cv]
