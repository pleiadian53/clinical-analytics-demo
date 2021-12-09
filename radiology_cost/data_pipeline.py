import os
import ast
import collections
import pandas as pd 
from pandas import DataFrame, Series
import numpy as np
from utils import highlight
import nltk


########################################################

def find_outliers(seq, m=2):
    if isinstance(seq, list): 
        seq = np.array(seq) 
        return list(np.where( abs(seq - np.mean(seq)) > m * np.std(seq) )[0])
    return np.where( abs(seq - np.mean(seq)) > m * np.std(seq) )[0]

def reject_outliers(seq, m=2):
    if isinstance(seq, list): 
        return list(np.array(seq)[abs(seq - np.mean(seq)) < m * np.std(seq)] )
    return seq[abs(seq - np.mean(seq)) < m * np.std(seq)]

def fraction_rows_missing(df, verbose=False, n=None):
    '''
    Return percent of rows with any missing
    data in the dataframe. 
    
    Input:
        df (dataframe): a pandas dataframe with potentially missing data
    Output:
        frac_missing (float): fraction of rows with missing data
    '''
    rows_null = df.isnull().any(axis=1)
    n_null = sum(rows_null)
    
    if verbose: # show rows with null values
        if n_null > 0: 
            print(f"(fraction_rows_missing) Rows with nulls (n={n_null}):\n{df[rows_null].head(n=n).to_string(index=False)}\n")
    
    return sum(rows_null)/df.shape[0]


def save_data(df, output_file="news_data.csv", output_dir=None, verbose=False): 
    if output_dir is None: output_dir = os.path.join(os.getcwd(), 'data')
    assert os.path.exists(output_dir)
    
    output_path = os.path.join(output_dir, output_file) 
    df.to_csv(output_path, sep=',', index=False, header=True)

    if verbose: 
        print(f"(save_data) Dim(df): {df.shape}")

    return

def load_data(input_file="radiology_costs.csv", input_dir=None, dropna=False, subset=False, 
                n=None, shuffle=False, how='any', verbose=False): 
    # np.random.seed(53)

    if input_dir is None: input_dir = os.path.join(os.getcwd(), 'data')
    assert os.path.exists(input_dir)
    
    input_path = os.path.join(input_dir, input_file) 
    df = pd.read_csv(input_path)
    N0 = N = df.shape[0]
    
    # drop rows with null values 
    if dropna:
        df = df.dropna(how=how)
        r_missing = fraction_rows_missing(df, verbose=True)
        assert r_missing == 0, "Found missing values. R(missing)={}".format(r_missing)
        N = df.shape[0]
        if verbose: 
            print(f"> size(df): {N0} -> {N}; dropped {N0-N} rows")
    
    # subsetting and shuffling
    if subset and (n is not None and n > 0): 
        if N > n: 
            df = df.sample(n=n)
        # else no-op
    if shuffle: 
        df = df.sample(frac=1.0)
        
    if verbose: 
        print(f"(load_data) Dim(df): {df.shape}")
    
    return df

def make_training_data(input_file="radiology_costs.csv", input_dir=None, cols_x=None, cols_y=None, 
        verbose=1, include_holdout=False): 
    """
    Create training data out of the given CSV files. 

    Input
    -----
        input_file:
        input_dir: 
        cols_x: 
            The columns corresponding to the covariates (explanatory variables); if not provided, it'll 
            be deduced from cols_y
        cols_y: The columns corresponding to the target (response variables)

        include_holdout: If set to True, will also return the holdout dataset (unlabeled data)

    Output
    ------
        Either 4-tuple if `include_holdout` is set to False,

            (X, y, covariates, target)

        or optinally return the holdout dataset, and hence a 5-tuple) if `include_holdout` is set to True. I.e. 

            (X, y, covariates, target, X_holdout)

    Memo
    ----
        Alternatively, we could design this function such that it turns a 5-tuple consistently. 

    """
    from sklearn import preprocessing

    df = load_data(input_file="radiology_costs.csv", input_dir=input_dir, dropna=False, shuffle=False) 
    target = 'downstream_cost'

    df_labeled = df.dropna(subset=[target, ], how='any')
    df_holdout = df_holdout_test = df[df[target].isnull()]  
    N_labeled, N_holdout = df_labeled.shape[0], df_holdout.shape[0]

    if cols_y is None: cols_y = [target, ]
    if cols_x is None: cols_x = list(df.drop(cols_y, axis=1).columns)
    feature_set = cols_x

    df = df_labeled # Use only labeled data
    minmax_scaler = preprocessing.MinMaxScaler()
    df[feature_set] = minmax_scaler.fit_transform(df[feature_set])
    X, y, feature_set, target = toXY(df, cols_y=cols_y, untracked=[])

    # Create holdout data for future test if desirable
    if include_holdout: 
        df_holdout[feature_set] = minmax_scaler.transform(df_holdout[feature_set])
        X_holdout, *rest = toXY(df_holdout, cols_y=cols_y, untracked=[])
    
    # y is a 2D array at this point 
    y = np.squeeze(y, -1)

    print(f"[debug] type(y): {type(y)}, shape(y): {y.shape}")
    assert X.shape[0] == N_labeled, f"size(X): {X.shape[0]} =?= {N_labeled}"

    if verbose: 
        print(f"(make_training_data) Feature set:\n{feature_set}\n")
        print(f"... Target: {target}")
        print(f"... Size (labeled set): {N_labeled}")
        print(f"... Size (holdout set): {N_holdout}")
        print(f"... Shape(X): {X.shape}, Shape(y): {y.shape}")
        print(f"... Downstream cost: avg: {np.mean(y)}, std:{np.std(y)}, median={np.median(y)}, max={max(y)}, min={min(y)}")

    if include_holdout: 
        return (X, y, feature_set, target, X_holdout)
    return (X, y, feature_set, target)

def save_to_npz(X, y=None, output_dir=None, suffix=None, verbose=0):
    # save numpy array as npz file
    from numpy import asarray
    from numpy import savez_compressed

    hasY = False
    if y is not None and len(y) > 0: 
        assert np.array(y).shape[0] == X.shape[0]
        hasY = True
    
    # save to npy file
    if output_dir is None: output_dir = os.getcwd()
    if suffix is not None: 
        Xfn = "X-%s.npz" % suffix
        yfn = "y-%s.npz" % suffix
    else: 
        Xfn = "X.npz"
        yfn = "y.npz"
    
    X_path = os.path.join(output_dir, Xfn)
    y_path = os.path.join(output_dir, yfn)
    savez_compressed(X_path, X)
    if hasY: 
        savez_compressed(y_path, y)

    if verbose: 
        print(f"[I/O] Saving X to:\n{X_path}\n")
        if hasY: print(f"...   Saving y to:\n{y_path}\n")

    return X_path, y_path
### Alias 
save_XY = save_to_npz 

def load_from_npz(input_dir=None, suffix=None): 
    from numpy import load 

    if input_dir is None: input_dir = os.getcwd()
    if suffix is not None: 
        Xfn = "X-%s.npz" % suffix
        yfn = "y-%s.npz" % suffix
    else: 
        Xfn = "X.npz"
        yfn = "y.npz"

    X_path = os.path.join(input_dir, Xfn)
    y_path = os.path.join(input_dir, yfn)
    
    X = load(X_path)['arr_0']
    try: 
        y = load(y_path)['arr_0']
    except: 
        y = []

    return (X, y)
### Alias
load_XY = load_from_npz

def scale(X, scaler=None, **kargs):
    from sklearn import preprocessing
    if scaler is None: 
        return X 

    if isinstance(scaler, str): 
        if scaler.startswith(('stand', 'z')): # standardize, z-score
            std_scale = preprocessing.StandardScaler().fit(X)
            X = std_scale.transform(X)
        elif scaler.startswith('minmax'): 
            minmax_scale = preprocessing.MinMaxScaler().fit(X)
            X = minmax_scale.transform(X)
        elif scaler.startswith("norm"): # normalize
            norm = kargs.get('norm', 'l2')
            copy = kargs.get('copy', False)
            X = preprocessing.Normalizer(norm=norm, copy=copy).fit_transform(X)
    else: 
        try: 
            X = scaler.transform(X)
        except Exception as e: 
            msg = "(scale) Invalid scaler: {}".format(e)
            raise ValueError(msg)
    return X

def toDF(X, cols_x, y=None, cols_y=None):
    import pandas as pd
    dfX = DataFrame(X, columns=cols_x)

    if y is not None: 
        if cols_y is None: cols_y = ['target', ]
        dfY = DataFrame(y, columns=cols_y)
        return pd.concat([dfX, dfY], axis=1)
    
    return dfX 

def toXY(df, cols_x=[], cols_y=[], untracked=[], **kargs): 
    """
    Convert a dataframe in to the (X, y)-format, where 
       X is an n x m numpy array with n instances and m variables
       y is an n x 1 numpy array, representing class labels

    Inupt
    -----
        cols_x: explanatory variables
        cols_y: target/dependent variable(s)
        untracked: meta-data

    """
    verbose = kargs.get('verbose', 1)

    # optional operations
    scaler = kargs.pop('scaler', None) # used when scaler is not None (e.g. "standardize")
    
    X = y = None
    if len(untracked) > 0: # untracked variables
        df = df.drop(untracked, axis=1)
    
    if isinstance(cols_y, str): cols_y = [cols_y, ]
    if len(cols_x) > 0:  
        X = df[cols_x].values
        
        cols_y = list(df.drop(cols_x, axis=1).columns)
        y = df[cols_y].values

    else: 
        if len(cols_y) > 0:
            cols_x = list(df.drop(cols_y, axis=1).columns)
            X = df[cols_x].values
            y = df[cols_y].values
        else: 
            if verbose: 
                print("(toXY) Both cols_x and cols_y are empty => Assuming all attributes are variables (n={})".format(df.shape[1]))
            X = df.values
            y = None

    if scaler is not None:
        if verbose: print("(toXY) Scaling X using method:\"{}\"".format(scaler))
        X = scale(X, scaler=scaler, **kargs)
    
    return (X, y, cols_x, cols_y)


def demo_load_data(input_file='news_data.csv'):
    """

    Memo
    ----
    1. n(rows) of news_data.csv: ~381K
    """
    n_samples = 1000

    df = load_data(input_file, subset=True, n=n_samples, verbose=True)
    assert len(df) <= n_samples

    # load the entire data set
    df = load_data(input_file, subset=False, verbose=True)

    return df

def run_data_pipeline(test_mode=False): 
    from sklearn.model_selection import train_test_split

    # Load data
    X, y, feature_set, target = make_training_data(input_file="radiology_costs.csv", input_dir='.')

    # Make training and test splits out of the labeled data
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2)
  
    return

if __name__ == "__main__": 
    run_data_pipeline(test_mode=False)

