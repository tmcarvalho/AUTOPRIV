"""Predictive performance
This script will test the predictive performance of each data set with Bayes hyperparameter optimization.
"""
import warnings
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, roc_auc_score
from skopt import BayesSearchCV
<<<<<<< HEAD
from skopt.space import Real, Integer
from sklearn.model_selection import RepeatedKFold
=======
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import RepeatedStratifiedKFold
>>>>>>> 3b16ca4 (stratified kfold)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings(action='ignore', category=FutureWarning)

# %% evaluate a model
def evaluate_model_bo(x_train, x_test, y_train, y_test):
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
            random_state=seed, early_stopping_rounds=10)

    reg = LogisticRegression(random_state=seed)
    nnet = MLPClassifier(random_state=seed)

    n_feat = x_train.shape[1]

    # set parameterisation
    param1 = {}
    param1['classifier__n_estimators'] = Integer(100, 500)
    param1['classifier__max_depth'] = Integer(4, 10)
    param1['classifier'] = [rfc]

    param2 = {}
    param2['classifier__n_estimators'] = Integer(100, 500)
    param2['classifier__max_depth'] = Integer(4, 10)
    param2['classifier__learning_rate'] = Real(1e-2, 1e-1)
    param2['classifier'] = [booster]

    param3 = {}
    param3['classifier__C'] = Real(1.e-4, 1.e4)
    param3['classifier__max_iter'] = Integer(1000000, 100000000)
    param3['classifier'] = [reg]

    param4 = {}
    param4['classifier__hidden_layer_sizes'] = Integer(int(n_feat // 2), n_feat)
    param4['classifier__alpha'] = Real(1e-4, 1e-2)
    param4['classifier__max_iter'] = Integer(10000, 100000)
    param4['classifier'] = [nnet]

    # define metric functions -- doens't accept multi measures
    scoring = make_scorer(roc_auc_score, max_fpr=0.001, needs_proba=True)

    pipeline = Pipeline([('classifier', rfc)])
    params = [param1, param2, param3, param4]

    print("Start modeling with CV")
    start = time.time()

    # Train the grid search model
    grid = BayesSearchCV(
        pipeline,
        search_spaces=params,
<<<<<<< HEAD
        n_iter=50,
        cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=42),
=======
        n_iter=32,
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1),
>>>>>>> 3b16ca4 (stratified kfold)
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
