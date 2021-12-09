import itertools
import numpy as np

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor


def holdout_grid_search(clf, X_train, y_train, X_val, y_val, hyperparams, fixed_hyperparams={}, smaller_is_better=True):
    """
    Conduct hyperparameter grid search on hold out validation set. Use holdout validation.
    Hyperparameters are input as a dictionary mapping each hyperparameter name to the
    range of values they should iterate over. Use the cindex function as your evaluation
    function.

    Input:
        clf: sklearn classifier
        X_train (dataframe): dataframe for training set input variables
        y_train (dataframe): dataframe for training set targets
        X_val (dataframe): dataframe for validation set input variables
        y_val (dataframe): dataframe for validation set targets
        hyperparams (dict): hyperparameter dictionary mapping hyperparameter
                            names to range of values for grid search
        fixed_hyperparams (dict): dictionary of fixed hyperparameters that
                                  are not included in the grid search

    Output:
        best_estimator (sklearn classifier): fitted sklearn classifier with best performance on
                                             validation set
        best_hyperparams (dict): hyperparameter dictionary mapping hyperparameter
                                 names to values in best_estimator
    """
    best_estimator = None
    best_hyperparams = {}
    
    # hold best running score
    best_score = np.inf if smaller_is_better else 0.0

    # get list of param values
    lists = hyperparams.values()
    
    # get all param combinations
    param_combinations = list(itertools.product(*lists))
    total_param_combinations = len(param_combinations)

    print("(holdout_grid_search) Searching for the best hyperparameters ...")

    # iterate through param combinations
    for i, params in enumerate(param_combinations, 1):   # 1: start index, i starts with this number
        # fill param dict with params
        param_dict = {}
        for param_index, param_name in enumerate(hyperparams):
            param_dict[param_name] = params[param_index]
            
        # create estimator with specified params
        estimator = clf(**param_dict, **fixed_hyperparams)

        # fit estimator
        estimator.fit(X_train, y_train)
        
        # get predictions on validation set
        y_pred = estimator.predict(X_val)
        
        # compute RMSE
        estimator_score = RMSE(y_val, y_pred)

        print(f'... [{i}/{total_param_combinations}] {param_dict}')
        print(f'... Val RMSE: {estimator_score}\n')

        # if new high score, update high score, best estimator
        # and best params 
        if (smaller_is_better and estimator_score < best_score) or (not smaller_is_better and estimator_score >= best_score):
            best_score = estimator_score
            best_estimator = estimator
            best_hyperparams = param_dict

    # add fixed hyperparamters to best combination of variable hyperparameters
    best_hyperparams.update(fixed_hyperparams)
    
    return best_estimator, best_hyperparams

def random_forest_grid_search(X_train, y_train, X_val, y_val, hyperparams=None):

    if hyperparams is None: 
        # Define ranges for the chosen random forest hyperparameters 
        hyperparams = {
            
            # how many trees should be in the forest (int)
            'n_estimators': [50, 100, 500, 700],

            # the maximum depth of trees in the forest (int)
            
            'max_depth': [4, 8, None, ], # [4, 5, 8, 10, 25, ],
            
            # the minimum number of samples in a leaf as a fraction
            # of the total number of samples in the training set
            # Can be int (in which case that is the minimum number)
            # or float (in which case the minimum is that fraction of the
            # number of training set samples)
            'min_samples_leaf': [2, 4, 8, ] # [4, 8, 0.01, 0.05, 0.1, 0.2, 0.25, ],
        }
    assert isinstance(hyperparams, dict)

    fixed_hyperparams = {
        'random_state': 10,
    }
    
    rf = RandomForestRegressor

    best_rf, best_hyperparams = holdout_grid_search(rf, X_train, y_train,
                                                    X_val, y_val, hyperparams,
                                                    fixed_hyperparams)

    print(f"> Best hyperparameters:\n{best_hyperparams}")

    y_train_best = best_rf.predict(X_train)
    print(f"> Train RMSE: {RMSE(y_train, y_train_best)}")

    y_val_best = best_rf.predict(X_val)
    print(f"> Val RMSE: {RMSE(y_val, y_val_best)}")
    
    # Add fixed hyperparamters to best combination of variable hyperparameters
    best_hyperparams.update(fixed_hyperparams)
    
    return best_rf, best_hyperparams


def RMSE(y, y_pred, **kargs): 
    return np.sqrt(mean_squared_error(y, y_pred))

def evaluate(model, X, y): 
    from sklearn.metrics import make_scorer

    # Evaluate the model
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X, y, scoring=make_scorer(RMSE), cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('RMSE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    return 

def demo_rf_regression():
    from sklearn.datasets import make_regression # synthetic data
    from data_pipeline import fraction_rows_missing, load_data, toXY, toDF, make_training_data

    ### 1. Synthetic data
    # Define dataset
    X, y = make_regression(n_samples=7500, n_features=6, n_informative=5, noise=0.1, random_state=2)

    # Define the model
    model = RandomForestRegressor()
    print(f"> Paramters: {model.get_params()}")

    evaluate(model, X, y)
    print('\n\n')

    ### 2. User data (e.g. radiology cost data)

    # Load data
    X, y, feature_set, target = make_training_data(input_file="radiology_costs.csv", input_dir='.', verbose=1)
    evaluate(model, X, y)

    return

def demo_model_selection(): 
    from sklearn.model_selection import train_test_split
    from data_pipeline import fraction_rows_missing, load_data, toXY, toDF, make_training_data

    # Load data
    X, y, feature_set, target = make_training_data(input_file="radiology_costs.csv", input_dir='.')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    best_rf, best_hyperparams = random_forest_grid_search(X_train, y_train, X_test, y_test)

    # Now, retrain the model and use CV to estimate the RMSE via CV
    evaluate(best_rf, X, y)

    return best_rf, best_hyperparams

def test(): 

    # Compare RF regression on synthetic data and user data (e.g. radiology_costs.csv)
    # demo_rf_regression()

    # Model selection
    demo_model_selection()

    return

if __name__ == "__main__": 
    test()






