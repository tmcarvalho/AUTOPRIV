"""Predictive performance
This script will test the predictive performance of each data set with Bayes hyperparameter optimization.
"""
import multiprocessing
import time
import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# evaluate a model
def evaluate_model_gs(x_train, x_test, y_train, y_test):
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
    _, x_valid, _, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
    # initiate models
    xgb = XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            random_state=seed)

    grb = GradientBoostingClassifier(n_iter_no_change=10,
                                    random_state=seed)

    sdg = SGDClassifier(early_stopping=True,
                        n_iter_no_change=10,
                        random_state=seed)

    nnet = MLPClassifier(early_stopping=True,
                         n_iter_no_change=10,
                        random_state=seed)

    n_feat = x_train.shape[1]

    # set parameterisation
    param1 = {}
    param1['classifier__n_estimators'] = [100, 250, 500]
    param1['classifier__max_depth'] = [4, 7, 10]
    param1['classifier__learning_rate'] = [0.01, 0.1]
    param1['classifier'] = [xgb]

    xgb_fit_params = {
    'eval_metric': 'logloss',
    'early_stopping_rounds': 10,  # Early stopping
    'eval_set': [(x_valid, y_valid)]  # Evaluation set
    }

    param2 = {}
    param2['classifier__alpha'] = [0.01, 0.1]
    param2['classifier__max_iter'] = [10000, 100000]
    param2['classifier__eta0'] = [0.01, 0.1]
    param2['classifier'] = [sdg]
    
    param3 = {}
    param3['classifier__n_estimators'] = [100, 250, 500]
    param3['classifier__max_depth'] = [4, 7, 10]
    param3['classifier__learning_rate'] = [0.01, 0.1]
    param3['classifier'] = [grb]

    param4 = {}
    param4['classifier__hidden_layer_sizes'] = [[int(n_feat // 2)], [int(n_feat * (2 / 3))], [n_feat]]
    param4['classifier__alpha'] = [0.01, 0.1]
    param4['classifier__max_iter'] = [10000, 100000]
    param4['classifier'] = [nnet]

    # define metric functions
    scoring = make_scorer(roc_auc_score, max_fpr=0.001, needs_proba=False)

    pipeline = Pipeline([('classifier', XGBClassifier(**xgb_fit_params))])
    params = [param1, param2, param3, param4]

    print("Start modeling with CV")
    training = time.time()

    # Train the grid search model
    grid = GridSearchCV(
        pipeline,
        param_grid=params,
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1),
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1).fit(x_train, y_train)

    print(f'It takes {(time.time() - training)/60} minutes')

    # Store results from grid search
    validation = pd.DataFrame(grid.cv_results_)
    validation['model'] = validation['param_classifier']
    validation['model'] = validation['model'].apply(lambda x: 'XGBClassifier' if 'XGBClassifier' in str(x) else x)
    validation['model'] = validation['model'].apply(lambda x: 'SGDClassifier' if 'SGDClassifier' in str(x) else x)
    validation['model'] = validation['model'].apply(lambda x: 'GradientBoostingClassifier' if 'GradientBoostingClassifier' in str(x) else x)
    validation['model'] = validation['model'].apply(lambda x: 'MLPClassifier' if 'MLPClassifier' in str(x) else x)
    validation['time'] = (time.time() - training)/60

    print("Start modeling in out of sample")

    # Use joblib to parallelize the loop
    num_jobs = min(multiprocessing.cpu_count(), len(validation))
    results = joblib.Parallel(n_jobs=num_jobs)(
        joblib.delayed(predict_model)(args) for args in
        [(i, grid.cv_results_['params'][i], x_train, y_train, x_test, y_test, pipeline, validation) for i in range(len(validation))]
    )

    score_cv = pd.DataFrame(results)

    return [validation, score_cv]


def predict_model(args):
    i, params, x_train, y_train, x_test, y_test, pipeline, validation = args

    clf_best = pipeline.set_params(**params).fit(x_train, y_train)
    clf = clf_best.predict(x_test)

    return {
        'params': str(params),
        'model': validation.loc[i, 'model'],
        'test_roc_auc': roc_auc_score(y_test, clf, max_fpr=0.001),
        'opt_type': "GridSearch"
    }