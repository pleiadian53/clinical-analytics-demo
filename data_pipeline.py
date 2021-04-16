import os
import pandas as pd 
from pandas import DataFrame, Series
import numpy as np

diagnosis_file = "Diagnosis.csv"
treatment_file = "Prescriptions.csv"
resource_file = "ccs.csv"
input_dir = os.getcwd()

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

def load_data(input_dir=None, n=None, dropna=True, subset=True, how='any', verbose=False): 
    if input_dir is None: input_dir = os.getcwd()
    assert os.path.exists(input_dir)
    diagnosis_file = "Diagnosis.csv"
    treatment_file = "Prescriptions.csv"
    resource_file = "ccs.csv"

    diagnosis_path = os.path.join(input_dir, diagnosis_file)  
    treatment_path = os.path.join(input_dir, treatment_file) 
    resource_path = os.path.join(input_dir, resource_file) 
    
    df_diag = pd.read_csv(diagnosis_path) # ~644K
    df_treat = pd.read_csv(treatment_path) # ~1000K
    df_res = pd.read_csv(resource_path) # ~70K
    
    Nd0, Nt0, Nr0 = df_diag.shape[0], df_treat.shape[0], df_res.shape[0]
    
    # drop rows with null values 
    if dropna:
        df_diag = df_diag.dropna(how=how)
        r_missing = fraction_rows_missing(df_diag, verbose=True)
        assert r_missing == 0, "Found missing values. R(missing)={}".format(r_missing)
        df_treat = df_treat.dropna(how=how) # we already know from EDA that treatment table doesn't have nulls
        
        if verbose: 
            print(f"> size(df_diag): {Nd0} -> {df_diag.shape[0]}; dropped {Nd0-df_diag.shape[0]} rows")
            print(f"> size(df_treat): {Nt0} -> {df_treat.shape[0]}; dropped {Nt0-df_treat.shape[0]} rows")
    
    if subset and (n is not None and n > 0): 
        df_diag, df_treat = subsample(df_diag, df_treat, n_samples=n, col_id='Patient_id')
        
    if verbose: 
        print(f"> size of diagnosis table: {df_diag.shape[0]}")
        print(f"> size of treatment table: {df_treat.shape[0]}")
        print(f"> size of ccs lookup     : {df_res.shape[0]}")
    
    return df_diag, df_treat, df_res

def intersection(df_diag, df_treat, on='Patient_id'):
    col_key = on
    uniq_patient_ids_diag = df_diag[col_key].unique()
    uniq_patient_ids_treat = df_treat[col_key].unique()
    n_uniq_pids_diag, n_uniq_pids_treat = len(uniq_patient_ids_diag), len(uniq_patient_ids_treat)
    return list(set(uniq_patient_ids_diag).intersection(uniq_patient_ids_treat))
    
def subsample(df_diag, df_treat, n_samples, col_id='Patient_id'): 
    """
    Subsample patients that appear in both diagnosis and treatment. 
    """
    uniq_patient_ids_diag = df_diag[col_id].unique()
    uniq_patient_ids_treat = df_treat[col_id].unique()
    n_uniq_pids_diag, n_uniq_pids_treat = len(uniq_patient_ids_diag), len(uniq_patient_ids_treat)

    set_patients_diag_treat = list(set(uniq_patient_ids_diag).intersection(uniq_patient_ids_treat))
    N = len(set_patients_diag_treat)
    
    if n_samples < N: 
        set_patients_diag_treat = np.random.choice(set_patients_diag_treat, n_samples, replace=False)
        # note: np.random.choice(), sampling with replacement is True by default

        df_diag_sub = df_diag.loc[df_diag[col_id].isin(set_patients_diag_treat)]
        df_treat_sub = df_treat.loc[df_treat[col_id].isin(set_patients_diag_treat)]

        assert len(df_diag_sub[col_id].unique()) == n_samples, f"n(patients): {len(df_diag_sub[col_id].unique())} =?= {n_samples}"
        assert len(df_treat_sub[col_id].unique()) == n_samples, f"n(patients): {len(df_treat_sub[col_id].unique())} =?= {n_samples}"
        
    return df_diag_sub, df_treat_sub 


def save_to_npz(X, y, input_dir=None, suffix=None):
    # save numpy array as npz file
    from numpy import asarray
    from numpy import savez_compressed
    
    # save to npy file
    if input_dir is None: input_dir = os.getcwd()
    if suffix is not None: 
        Xfn = "X-%s.npz" % suffix
        yfn = "y-%s.npz" % suffix
    else: 
        Xfn = "X.npz"
        yfn = "y.npz"
    
    X_path = os.path.join(input_dir, Xfn)
    y_path = os.path.join(input_dir, yfn)
    savez_compressed(X_path, X)
    savez_compressed(y_path, y)

    return 
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
    y = load(y_path)['arr_0']

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

def toDF(X, y, cols_x, cols_y):
    import pandas as pd
    dfX = DataFrame(X, columns=cols_x)
    dfY = DataFrame(y, columns=cols_y)
    return pd.concat([dfX, dfY], axis=1)

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

def analyze_diagnosis(verbose=1): 
    from utils import Diagnosis
    from icd_utils import encode, decode, is_valid_icd, is_disease_specific

    p = Diagnosis.properties
    col_key = p['id']
    col_date = p['date']
    col_code = p['code']

    # df_diag = Diagnosis.load(dtype='in', subset=False, verbose=verbose > 1)
    df_diag, df_treat, df_res = load_data(input_dir=os.getcwd(), subset=False, verbose=verbose > 1)

    codebook = {} 
    codebook['codes'] = codebook['unique'] = codes = df_diag['ICD10'].unique()
    print(f"(1) Number of unique codes: {len(codes)}")

    # number of ill-formatted ICD10 codes? 
    codebook['invalid'] = codes_invalid = [code for code in codes if not is_valid_icd(code)]
    codebook['valid'] = codes_valid = list(set(codes) - set(codes_invalid))
    print(f"(2) Invalid codes (n={len(codes_invalid)}):\n{codes_invalid}\n")

    precision = None
    codes_encoded = encode(codes, precision=precision)
    # print(f"> example codes:\n{codes_encoded[:50]}\n")

    # Can we find the definition for all codes? 
    codes_missed_encoded = set(codes_encoded) - set(df_res['diag'].values)
    codebook['missed'] = codes_missed = decode(codes_missed_encoded) # [decode(code) for code in codes_missed_encoded]
    print(f"(3) Num of codes not found in CCS (n={len(codes_missed)}):\n{codes_missed}\n")

    print("(3.1) Codes missed by CCS must be invalid? {}".format(set(codes_invalid) < set(codes_missed)))
    codes_hit_invalid = set(codes_invalid) - set(codes_missed) 
    if len(codes_hit_invalid) > 0: 
        print(f"(3.2) Codes found in CCS but do not conform to ICD10 format:\n{codes_hit_invalid}\n")
        
    for r, row in df_res[df_res['diag'].isin( decode(codes_hit_invalid) )].iterrows(): 
        print(f"(3.3) {row['diag']}: {row['diag_desc']}")

    codebook['disease'] = codes_disease = [code for code in codes_valid if is_disease_specific(code)]

    # build ICD10 lookup table
    # codebook['lookup'] = dict(zip(decode(df_res['diag'].values), des_res['diag_desc'].values))

    # For each patient, list his/her diagnoses and their associated dates; 
    # doing so allows us to see patients' diagnoses as a temporal sequence
    print("(5) List example ICDs across sessions ...\n")
    n_samples = 5

    # We could perhaps gain more insights by looking at patients who went through multiple clinical visits
    pids = []
    for pid, dfi in df_diag.groupby(col_key): 
        if len(dfi[col_date].unique()) >= 2: 
            pids.append(pid)
        if len(pids) > n_samples: break
    df_diag_sub = df_diag[df_diag[col_key].isin(pids)]
    for pid, dfi in df_diag_sub.groupby([col_key, ]):
        print(f"... PID={pid}")
        print(f"... date: {col_date}")
        print(f"... dfi:\n{dfi}\n")

    return df_diag, codebook

def analyze_treatment(verbose=1): 
    """

    Memo
    ----
    1. Output: 
       Found n=89 drug category
       Found n=415 drug group
       Found n=604 drug class
    """
    from utils import Treatment

    p = Treatment.properties
    col_category = p['category'] # 'drug_category'
    col_group = p['group'] # 'drug_group'
    col_class = p['class'] # 'drug_class'
    col_code = p['code']  # derived from after encode_prescriptions()

    codebook = {}
    df_treat = Treatment.load(dtype='in', verbose=verbose > 1)

    # treatment table attributes: Patient_id, Prescription_date, drug_category, drug_group, drug_class
    cols = [(col_category, 'categories'), (col_group, 'groups'), (col_class, 'classes')]
    for i, (col, unit) in enumerate(cols): 
        counts = df_treat[col].value_counts()
        print(f"(1.{i+1}) Found n={len(counts)} drug {unit}, and their counts are as follows:\n{counts.nlargest(n=20)}\n")

    if col_code in df_treat.columns: 
        codebook = dict(zip(df_treat[col_class], df_treat[col_code]))
    return df_treat, codebook

def demo_load_data():
    col_key = 'Patient_id'
    col_date = 'Diag_date'
    col_code = 'ICD10'

    n_samples = 1000

    # load a subset of the patient data (by default only randomly select from patients that appear in both
    # diagnosis and treatment tables)
    df_diag, df_treat, df_res = load_data(input_dir=os.getcwd(), n=n_samples, verbose=False)
    assert len(df_diag[col_key].unique()) == n_samples
    assert set(df_treat[col_key].unique()) == set(df_diag[col_key].unique())

    # load the entire data set
    df_diag, df_treat, df_res = load_data(input_dir=os.getcwd(), subset=False, verbose=True)

    return 

def test(): 
    col_key = 'Patient_id'
    col_date = 'Diag_date'
    col_code = 'ICD10'

    # different ways of loading the data sets
    # demo_load_data()

    analyze_treatment()
   
    return

if __name__ == "__main__": 
    test()