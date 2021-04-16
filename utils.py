import sys, os
import numpy as np
import collections
import pandas as pd 
from pandas import DataFrame, Series
import feature_extractor as fext

class Diagnosis(object): 
    input_dir = os.getcwd()
    input_file = 'Diagnosis.csv'
    seq_file = 'Diagnosis-sequenced.csv'

    properties = {'id': 'Patient_id', 'date': 'Diag_date',
                  'code': 'ICD10',

                  # derived
                  'history': 'History', 
                  'primary': 'Topn', 'status': 'Topn'
                }

    def __init__(self, df=None): 
        self.df = df

    def init(self, **kargs):
        if self.df is not None: 
            return # no-op 
        self.df = Diagnosis.load(dtype='input', **kargs)

    def importance_weights(self): 
        pass

    def sequence(self, **kargs): 
        if self.df is None or self.df.empty: 
            raise ValueError("[sequence] Missing the diagnosis table to compute diagnosis sequences.")

        if 'n_days_lookback' in kargs: 
            if kargs['n_days_lookback'] is None: kargs['tFilterByLength'] = False
        return transform_diagnosis(self.df, **kargs)

    def eval_topn_diagnoses(self, df_seq=None, topn=3, topn_global=10, overwrite=False, 
                                col_key='Patient_id', col_code='ICD10', col_topn='Topn', **kargs):

        verbose = kargs.get('verbose', 1)
        
        # load sequenced diagnosis file
        if df_seq is None: 
            try:  
                df_seq = Diagnosis.load(dtype='seq') # load the pre-computed sequence data, because it's take a minute or two to sequence the data
            except: 
                msg = "[Diagnosis] No sequenced data found. Computing sequenced data."
                raise ValueError(msg)

        # drop pre-existing topn-related columns if they exist
        has_topn = True
        if col_topn in df_seq.columns: 
            if verbose: "[Diagnosis] Found pre-computed primary diagnoses."
            if overwrite: 
                if verbose: "[Diagnosis] Overwriting pre-computed primary diagnoses ..."
                df_seq = df_seq.drop(col_topn, axis=1)
        else: 
            has_topn = False
        
        df_topn_local, df_topn_global = \
                fext.eval_topn_features(df_seq, topn=topn, topn_global=topn_global, 
                                            col_key=col_key, col_doc=col_code, col_topn=col_topn, **kargs)
        
        # append to the sequenced dataframe (df_seq) the topn ICD10 codes as the primary diagnosis
        if overwrite or not has_topn: 
            Np = df_seq.shape[0]
            df_seq = pd.merge(df_seq, df_topn_local, on=col_key)
            assert Np == df_seq.shape[0]

            Diagnosis.save(df_seq, dtype='seq', verbose=verbose > 1)
            if verbose: print("[Diagnosis] Saved primary diagnoses.")

        return df_topn_local, df_topn_global

    @staticmethod
    def get_path(dtype='input'):
        if dtype.startswith('in'): 
            return os.path.join(Diagnosis.input_dir, Diagnosis.input_file)
        return os.path.join(Diagnosis.input_dir, Diagnosis.seq_file)

    @staticmethod
    def save(df, dtype='seq', verbose=1):
        if dtype.startswith('in'):
            raise NotImplementedError("[save] Overwriting the source data set is not supported at the moment.")

        df.to_csv(Diagnosis.get_path(dtype='seq'), sep='|', index=False, header=True)
        if verbose: print("[Diagnosis] Saved sequenced diagnosis file to:\n{}\n".format(Diagnosis.get_path(dtype='seq')))
        return

    @staticmethod
    def load(dtype='seq', **kargs):
        """
        Load data set.

        Todo
        ----

        """
        if dtype.startswith('in'):
            from data_pipeline import load_data

            # only support default behavior for now
            df_diag, _, _ = load_data(**kargs)
            return df_diag
        
        print("[Diagnosis] Loading sequenced diagnosis file from:\n{}\n".format(Diagnosis.get_path(dtype='seq')))
        return pd.read_csv(Diagnosis.get_path(dtype='seq'), sep='|', header=0, index_col=None)

class Treatment(object): 
    input_dir = os.getcwd()
    input_file = 'Prescriptions.csv'
    seq_file = 'Prescriptions-sequenced.csv'

    properties = {'id': 'Patient_id', 'date': 'Prescription_date',
                  'category': 'drug_category', 'group': 'drug_group', 'class': 'drug_class', 

                   ### derived attributes
                   'code': 'drug_code',  # encode_prescriptions
                   'history': 'History', 
                   'primary': 'Topn'

                }

    def __init__(self, df=None): 
        self.df = df

    def init(self):
        if self.df is not None: 
            return # no-op 
        self.df = Treatment.load(dtype='input', **kargs)

    @staticmethod
    def get_path(dtype='input'):
        if dtype.startswith('in'): 
            return os.path.join(Treatment.input_dir, Treatment.input_file)
        return os.path.join(Treatment.input_dir, Treatment.seq_file)

    @staticmethod
    def load(dtype='seq', **kargs):
        """
        Load data set.

        Todo
        ----

        """
        if dtype.startswith('in'):
            from data_pipeline import load_data

            # only support default behavior for now
            _, df_treat, _ = load_data(**kargs)
            return df_treat
        
        print("[Treatment] Loading sequenced treatment/prescriptions file from:\n{}\n".format(Treatment.get_path(dtype='seq')))
        return pd.read_csv(Treatment.get_path(dtype='seq'), sep='|', header=0, index_col=None)

    @staticmethod
    def save(df, dtype='seq', verbose=1):
        col_code = Treatment.properties['code']
        if dtype.startswith('in'):
            # safe guard 
            assert col_code in df.columns

            df.to_csv(Treatment.get_path(dtype='in'), index=False, header=True)
            if verbose: print(f"[Treatment] Saved input file back to:\n{Treatment.get_path(dtype='in')}\n")
        else: 
            df.to_csv(Treatment.get_path(dtype='seq'), sep='|', index=False, header=True)
            if verbose: print("[Treatment] Saved sequenced treatment file to:\n{}\n".format(Treatment.get_path(dtype='seq')))
        return

    def sequence(self, **kargs): 
        if self.df is None or self.df.empty: 
            raise ValueError("[sequence] Missing the treatment table to compute prescription sequences.")

        if 'n_days_lookback' in kargs: 
            if kargs['n_days_lookback'] is None: kargs['tFilterByLength'] = False
        return transform_treatment(self.df, **kargs)

    def eval_topn_prescriptions(self, df_seq=None, topn=5, topn_global=10, overwrite=False,
                                        sort_topn_by_time=True,
                                        col_topn='Primary', **kargs):

        p = Treatment.properties
        col_key = p['id']
        col_code = p['code']
        verbose = kargs.get('verbose', 1)
        
        # load sequenced diagnosis file
        if df_seq is None: 
            try:  
                df_seq = Treatment.load(dtype='seq') # load the pre-computed sequence data, because it's take a minute or two to sequence the data
            except: 
                msg = "[Treatment] No sequenced data found. Computing sequenced data."
                raise ValueError(msg)

        # drop pre-existing topn-related columns if they exist
        has_topn = True
        if col_topn in df_seq.columns: 
            if verbose: "[Treatment] Found pre-computed prime prescriptions."
            if overwrite: 
                if verbose: "[Treatment] Overwriting pre-computed prime prescriptions ..."
                df_seq = df_seq.drop(col_topn, axis=1)
        else: 
            has_topn = False
        
        df_topn_local, df_topn_global = \
                fext.eval_topn_features(df_seq, topn=topn, topn_global=topn_global, 
                                            col_key=col_key, col_doc=col_code, col_topn=col_topn, **kargs)
        
        # re-order topn prescriptions according to time? 
        if sort_topn_by_time: 
            pass  # [todo] as it first appears or last? 

        # append to the sequenced dataframe (df_seq) the topn ICD10 codes as the primary diagnosis
        if overwrite or not has_topn: 
            Np = df_seq.shape[0]
            df_seq = pd.merge(df_seq, df_topn_local, on=col_key)
            assert Np == df_seq.shape[0]

            Treatment.save(df_seq, dtype='seq', verbose=verbose)
            if verbose: print("[Treatment] Saved primary prescritpions.")

        return df_topn_local, df_topn_global
    
    def encode(self, prefix='D', sep='-', save=False, verbose=True):
        col = Treatment.properties
        col_category = col['category'] # 'drug_category'
        col_group = col['group'] # 'drug_group'
        col_class = col['class']  # 'drug_class'
        col_encoded = col['code'] # 'drug_code' 
        assert set([col_category, col_group, col_class]) <= set(self.df.columns)

        drug_classes = {}
        for drug_class in sorted(self.df[col_class].unique()):
            drug_classes[drug_class] = len(drug_classes)

        dfx = []
        for dc, dfi in self.df.groupby(col_class): 
            categories = dfi[col_category].unique()
            assert len(categories) == 1, f"Drug class {dc} is associated with multiple categories (n={len(categories)}) > 1)?\n{categories}\n"
            dfi[col_encoded] = '%s%s%s' % (prefix, sep, drug_classes[dc]) # NOTE: this is not sufficient to retain the change
            dfx.append(dfi)
        self.df = pd.concat(dfx, ignore_index=True)

        if save: 
            Treatment.save(self.df, dtype='in', verbose=verbose > 1)
            if verbose: 
                print(f"[Treatment] Added new column {col_encoded} and saved new dataframe to:\n{Treatment.get_path(dtype='in')}\n")
        
        return

def encode_prescriptions(df_treat, prefix='D', sep='-', **kargs): 
    col = Treatment.properties
    col_id = col['id']
    col_date = col['date']
    col_category = col['category'] # 'drug_category'
    col_group = col['group'] # 'drug_group'
    col_class = col['class']  # 'drug_class'
    col_encoded = col['code'] # 'drug_code' 

    drug_classes = {}
    for drug_class in sorted(df_treat[col_class].unique()):
        drug_classes[drug_class] = len(drug_classes)
    
    dfx = []
    cols_prime = [col_id, col_date, col_encoded]
    for dc, dfi in df_treat.groupby(col_class): 
        categories = dfi[col_category].unique()
        assert len(categories) == 1, f"Drug class {dc} is associated with multiple categories (n={len(categories)}) > 1)?\n{categories}\n"
        
        dfi_new = DataFrame(columns=cols_prime)

        # those patients on these dates all had this prescription drug
        dfi_new[col_id] = dfi[col_id]
        dfi_new[col_date] = dfi[col_date]
        dfi_new[col_encoded] = '%s%s%s' % (prefix, sep, drug_classes[dc]) # NOTE: this is not sufficient to retain the change
        dfx.append(dfi_new)
        # s_indices = [str(j) for j in range(dfi.shape[0])]
        # dfi[col_encoded] = map(sep.join, zip(dfi[col_category].values, s_indices))
    df_derived = pd.concat(dfx, ignore_index=True)
    assert col_encoded in df_derived.columns
    
    return df_derived

def time_to_event(df, col1='Diag_date', col2='Prescription_date', strftime='%Y-%m-%d'):
    return (pd.to_datetime(df[col1], format=strftime)-pd.to_datetime(df[col2], format=strftime)).dt.days

def length_of_history(df, col='Diag_date', strftime='%Y-%m-%d'):
    if df.empty: return 0

    min_date = df[col].min()
    max_date = df[col].max()
    if min_date == max_date: 
        return 1
    return (pd.to_datetime(max_date, format=strftime)-pd.to_datetime(min_date, format=strftime)).days+1

def time_window(date, bounds=(1, 1), constraints=(None, None), strftime='%Y-%m-%d'): 
    n_days_minus, n_days_plus = bounds
    if n_days_minus is None: n_days_minus = 0
    if n_days_plus is None: n_days_plus = 0
    t_minus = pd.to_datetime(date, format=strftime) - pd.Timedelta(days=abs(n_days_minus))
    t_plus = pd.to_datetime(date, format=strftime) + pd.Timedelta(days=abs(n_days_plus))

    # constraints can come in the form of a dataframe or a user-defined tuple (representing a lowerbound and upperbound)
    first_date, last_date = constraints
    if first_date is not None: 
        first_date = pd.to_datetime(first_date, format=strftime)
        if t_minus < first_date: t_minus = first_date

    if last_date is not None:
        last_date = pd.to_datetime(last_date, format=strftime)
        if t_plus > last_date: t_plus = last_date

    return (t_minus.strftime(strftime), t_plus.strftime(strftime))

def create_time_windows(df, col_key='Patient_id', col_date='Diag_date', 
        col_lower='Diag_min', col_upper='Diag_max',
        bounds=(1, 1), constraints=(None, None), strftime='%Y-%m-%d'): 
    def center_at(x): 
        lower, upper = constraints
        if lower is None: lower = df[col_date].min()
        if upper is None: upper = df[col_date].max() 
        return time_window(x, bounds=bounds, constraints=(lower, upper), strftime=strftime)

    new_cols = [col_lower, col_upper]
    # df_new = DataFrame(columns=[col_key, col_date]+new_cols)
    # df_new[col_key] = df[col_key]
    # df_new[col_date] = df[col_date]

    s = df[col_date].apply(center_at)
    for i, col in enumerate(new_cols):
        df[col] = s.apply(lambda x: x[i])
    return df

def filter_by_CCS(df_diag, df_res, col_code='ICD10', col_diag='diag', verbose=True): 
    from icd_utils import encode, decode
    
    codes_encoded = encode(df_diag[col_code].unique(), precision=None)
    # print(f"> example codes:\n{codes_encoded[:50]}\n")

    # Can we find the definition for all codes? 
    codes_missed_encoded = set(codes_encoded) - set(df_res[col_diag].values)
    codes_missed = decode(codes_missed_encoded) # [decode(code) for code in codes_missed_encoded]
    if verbose: print(f"> Num of codes not found in CCS (n={len(codes_missed)}):\n{codes_missed}\n")

    return df_diag[~df_diag[col_code].isin(codes_missed)]

def filter_by_ICD(df_diag, col_code='ICD10', disease_specific=True): 
    from icd_utils import is_valid_icd, is_disease_specific

    if disease_specific: 
        return df_diag[df_diag[col_code].apply(is_disease_specific)]  # which implies that the code is valid
    
    return df_diag[df_diag[col_code].apply(is_valid_icd)]

def subsample_seq(df_seq, n=10, col_key='Patient_id', col_topn='Topn',verbose=False): 
    df_seq_sub = df_seq[df_seq[col_key].isin(np.random.choice(df_seq[col_key].unique(), n, replace=False))]
    has_topn = col_topn in df_seq_sub.columns
    if verbose: 
        p = Diagnosis.properties
        col_code = p['code']
        col_intv = p['history']
        for r, row in df_seq_sub.iterrows(): 
            print(f"> ID:      {row[col_key]}")
            print(f"> history: {row[col_intv]} days")
            print(f"> sequence:\n{row[col_code]}\n")
            if has_topn: 
                print(f"> topn:    {row[col_topn]}")
    return df_seq_sub

def transform_treatment(df_treat, col_key='Patient_id', col_date='Prescription_date', col_code='drug_code', 
                            col_intv='History', 

                            # predicates 
                            tIncludeTime=False, 
                            tFilterByLength=True,

                            n_days_lookback = 180, 
                            verbose=1,

                            **kargs
                        ):
    """
    This function defines a "sequencing transformation" on the input dataframe `df_treat`, a treatment table
    comprising drug prescritpions. 

    """
    # import collections

    if tFilterByLength: 
        assert n_days_lookback is not None

    assert col_code in df_treat.columns, "Input table has not been encoded yet; see encode_prescriptions()."

    adict = {k: [] for k in [col_key, col_intv, col_code]}

    group_size = 1000
    acc = 0
    for i, (pid, dfi) in enumerate(df_treat.groupby(col_key)):  # foreach patient and his/her record ...
        adict[col_key].append(pid)

        if tFilterByLength: 
            first_date, last_date = time_window(dfi[col_date].max(), bounds=(n_days_lookback, None), 
                                        constraints=(dfi[col_date].min(), dfi[col_date].max()), strftime='%Y-%m-%d')

            assert last_date == dfi[col_date].max(), f"{last_date} =?= {dfi[col_date].max()}"
            dfi = dfi[dfi[col_date] >= first_date]
            size_history = length_of_history(dfi, col=col_date)
            assert size_history <= n_days_lookback+1, f"length of history: {size_history} <=? n_days: {n_days_lookback}"

        n_days = length_of_history(dfi, col=col_date)
        
        adict[col_intv].append(n_days)

        history = []
        for date, dfj in dfi.groupby(col_date):  # for the same person, group by the date
            codes = dfj[col_code].values  # drug codes based on drug classes

            # Unlike ICD10, it's possible to find duplicate prescriptions on the same date
            # assert len(codes) == len(set(codes)), f"Dup drug codes found in the same prescription date (date={date}): {codes}"
            if verbose: 
                if len(codes) != len(set(codes)) and (i < group_size):  
                    print(f"Dup drug codes found in the same prescription date (date={date}): {collections.Counter(codes)}")

            codes = list(dfj[col_code].unique())  # redundancy, should have been unique

            if tIncludeTime:
                history.extend( ['%s/%s' % (c, date) for c in codes] )
            else: 
                history.extend( codes )

        adict[col_code].append( ' '.join(history) )

        if i > 0 and i % group_size == 0: 
            print(f"> [{i}] " + "#" * (acc+1))
            acc += 1
    return DataFrame(adict)

def make_lookup_table(input_dir=None): 
    from data_pipeline import load_data
    from icd_utils import encode, decode

    df_diag, df_treat, df_res = load_data(input_dir=input_dir, subset=False, verbose=True)
    return dict(zip(decode(df_res['diag'].values), df_res['diag_desc'].values))

def transform_diagnosis(df_diag, col_key='Patient_id', col_date='Diag_date', col_code='ICD10', 
                          
                          # derived attributes
                          col_intv='History',  
                      
                          # predicates
                          tIncludeTime=False,
                          tFilterByLength=True, 

                          tDocumentedOnly=False,
                          df_res=None, 
                          
                          tFilterByICD=True, 
                          disease_specific_only=False,
                          
                          # filtering parameters
                          n_days_lookback = 180,

                          verbose=True
                     ): 
    """
    This function defines a "sequencing transformation" on the input dataframe `df_diag`, a diagnosis table, 
    meaning that it takes an input dataframe (df_diag) and produces another dataframe containing a 
    sequenced representation of the diagnoses (in terms of ICD10 codes) as specified by the column `col_code`. 
    
    Input
        df_diag: diagnosis table with at least the following attributes as defined by
                `col_key`, `col_date`, `col_code` 
        
        tIncludeTime: if True, the corresponding diagnosis date is appended to each ICD10 code in the 
                      diagnostic temporal sequence

                    e.g. without timestamp 

                         R05 I10 J45.40 K76.0 R10.13

                         with timestamp appended

                         R05/2016-01-23 I10/2016-02-11 J45.40/2016-05-28 K76.0/2016-05-28 R10.13/2016-11-01

        tFilterByLength: if True, discard the ICD10 codes documented more than 'n_days_lookback' days ago;
                         only retain the ICD10 codes from the most recent 'n_days_lookback' days counting 
                         from the last session (inclusive)

        tFilterByICD: if True, then only retain valid ICD10 codes
                      
       
    Output
       A transformed dataframe with the following attributes: 
       
       col_key (`Patient_id` by default): Patient ID
       col_intv (`History` by default): Length of the medical record i.e. 
       
                 date of last session - date of first session + 1 
                 
       col_code (`ICD10` by default): A string of temporal sequene of ICD10 codes 
    """
    if tFilterByICD: 
        df_diag = filter_by_ICD(df_diag, disease_specific=disease_specific_only) # retain rows with valid ICD10 codes 

    if tDocumentedOnly and df_res is not None: 
        # assert df_res is not None, "CCS table is not provided"
        df_diag = filter_by_CCS(df_diag, df_res, col_code='ICD10', col_diag='diag', verbose=True) 

    if tFilterByLength: 
        assert n_days_lookback is not None

    adict = {k: [] for k in [col_key, col_intv, col_code]}
    group_size = 1000
    acc = 0
    for i, (pid, dfi) in enumerate(df_diag.groupby(col_key)):
        adict[col_key].append(pid)

        if tFilterByLength: 
            # filter by time (e.g. only get diagnoses from n days ago until now)

            first_date, last_date = time_window(dfi[col_date].max(), bounds=(n_days_lookback, None), 
                                        constraints=(dfi[col_date].min(), dfi[col_date].max()), strftime='%Y-%m-%d')

            assert last_date == dfi[col_date].max(), f"{last_date} =?= {dfi[col_date].max()}"
            dfi = dfi[dfi[col_date] >= first_date]
            size_history = length_of_history(dfi, col=col_date)
            assert size_history <= n_days_lookback+1, f"length of history: {size_history} <=? n_days: {n_days_lookback}"

        n_days = length_of_history(dfi, col=col_date)
        # if i < 2: print(f"> ID: {pid}, n_days: {n_days}")
        adict[col_intv].append(n_days)

        history = []
        for date, dfj in dfi.groupby(col_date): 
            codes = dfj[col_code].values
            assert len(codes) == len(set(codes)), f"Dup ICDs found in the same session (date={date}): {codes}"

            codes = list(dfj[col_code].unique())  # redundancy, should have been unique

            if tIncludeTime:
                history.extend( ['%s/%s' % (c, date) for c in codes] )
            else: 
                history.extend( codes )

        adict[col_code].append( ' '.join(history) )

        if i > 0 and i % group_size == 0: 
            print(f"> [{i}] " + "#" * (acc+1))
            acc += 1
    return DataFrame(adict)

def match(df_diag, df_treat, target_code, window=(60, 60), col_window=('Diag_min', 'Diag_max'), verbose=1): 
    """
    Match diagnosis and treatment table by dates. 

    Given a diagnosis date d, a treatment date (dt) is said to 
    fall within a date window W=[x, y] iff the treatment date (dt) is between 
    date x and date y. 

    E.g. W = (30, 60), and d = '2017-06-18'
         then dt = '2017-05-29' falls within W because d-30 < dt < d+60
              dt = '2017-07-29' also falls within W because d-30 < dt < d+60
         but  dt = '2018-02-11' does not fall within W because dt > d+60

    Given an ICD10 code, representing (partial) health status, find related 
    diagnosis dates for each patient and then find the matching prescription dates
    that fall within a given window W (for each of the diagnosis date).
    """
    from tabulate import tabulate
    from data_pipeline import intersection

    dp = Diagnosis.properties
    col_key = dp['id']
    col_date = dp['date']
    col_code = dp['code']

    tp = Treatment.properties
    col_date_t = tp['date']
    col_class_t = tp['class']  # drug class
    col_code_t = tp['code']

    w_min, w_max = 'Diag_min', 'Diag_max' 

    # ensure that df_diag and df_treat share the same set of patients 
    set_patients = intersection(df_diag, df_treat, on='Patient_id') 
    n_samples = len(set_patients)
    if verbose: print(f"(match) Total number of patients: {n_samples}")
    df_diag = df_diag[df_diag[col_key].isin(set_patients)]
    df_treat = df_treat[df_treat[col_key].isin(set_patients)]
    ##########################################
    
    # check if the treatment table is encoded
    is_prescription_encoded = col_code_t in df_treat.columns
    
    df_diag_subset = df_diag[df_diag[col_code]==target_code]
    target_patients = df_diag_subset[col_key].unique()
    Nd = len(target_patients)
    if verbose: print(f"(match) Number of patients with diagnosis {target_code}: {Nd}")
    
    # subset all patients with this diagnosis 
    df_treat_subset = df_treat[df_treat[col_key].isin(target_patients)]
    Nt = len(df_treat_subset[col_key].unique())

    if verbose: 
        print(f"(match) Number of patients who had been given prescriptions for diagnosis {target_code}: {Nt}")
        print(f"... n(diagnosed): {Nd} >=? n(treated): {Nt}")

    # compute the allowable dates given W 
    w_min, w_max = col_window
    # foreach diagnosis date in the joined table, find the time window W, centered on the given diagnosis date (d), 
    # where the lowerbound of W is given by d-W[0] days and, 
    #       the upperbound of W is given by d+W[1] days
    nrows0 = df_diag_subset.shape[0]
    df_diag_subset = create_time_windows(df_diag_subset, col_date=col_date, col_lower=w_min, col_upper=w_max,
                        bounds=window, constraints=(None, None), strftime='%Y-%m-%d')
    assert df_diag_subset.shape[0] == nrows0
    
    if verbose: 
        print(f"(match) diagnosis table with the window of dates determined (W={window}):\n{df_diag_subset.columns}\n")
        print(f"... sampled dataframe:\n{tabulate(df_diag_subset.head(10), headers='keys', tablefmt='psql', showindex=False)}\n")

    acc, group_size = 0, 100 # used to track progress
    n_matched = 0
    df_treat_new = []
    for i, (pid, dfi) in enumerate(df_diag_subset.groupby(col_key)): # foreach patient
        assert len(dfi[col_code].unique()) == 1
        dfi_t = df_treat_subset[df_treat_subset[col_key]==pid]
        assert not dfi_t.empty, f"We should have already taken intersection on {col_key}"
        
        for date, dfj in dfi.groupby(col_date): # foreach each date
            
            assert len(dfj[w_min].unique()) == 1, \
                  f"lower end of W should be consistent for the same date:\n{dfj[w_min].unique()}\n"
            d_minus = dfj[w_min].unique()[0]
            assert len(dfj[w_max].unique()) == 1, \
                  f"upper end of W should be consistent for the same date:\n{dfj[w_max].unique()}\n"
            d_plus = dfj[w_max].unique()[0]

            # find all prescription dates that fall within the allowed window (d_minus, d_plus)
            dfj_t = dfi_t[(dfi_t[col_date_t] >= d_minus) & (dfi_t[col_date_t] <= d_plus)] 
            if not dfj_t.empty: 

                cols = [col_key, col_date_t, col_class_t]
                if is_prescription_encoded: cols += [col_code_t, ]

                dfk = DataFrame(columns=cols)
                dfk[cols] = dfj_t[cols]

                # all of these prescription dates share the same diagnosis date
                dfk[col_date] = date 
                n_matched += 1
                
                if (verbose > 1) and (n_matched > 0 and n_matched % 20 == 0): 
                    header = [col_key, col_date_t, col_class_t]
                    print(f"... matched prescriptions by date={date}, W=({d_minus}, {d_plus}):\n{dfk[header]}\n")
                
                # per_patient_t.update(df[col_class_t].values)  
                df_treat_new.append(dfk)
        if (verbose > 1) and (i > 0 and i % group_size == 0): 
            print(f"> [{i}] " + "#" * (acc+1))
            acc += 1
    
    df_match = DataFrame()
    if len(df_treat_new) > 0: 
        df_match = pd.concat(df_treat_new, ignore_index=True)
        df_match[col_code] = target_code
        if verbose: print(f"(match) Found n={len(df_match[col_key].unique())} patients with matching prescription records")
    else: 
        if verbose: print(f"(match) Found n=0 patients with matching prescription records")

    return df_match

def find_similar_diagnosis(df_diag, target_code, n_base=2): 
    def is_similar(x): 
        x.startswith()

    p = Diagnosis.properties
    col_key = p['id']
    col_code = p['code']

    # similar diagnosis codes should fall into the same range
    df_subset = df_diag[df_diag[col_code]!=target_code]
    prefix = target_code[:n_base]
    return df_subset[ df_subset[col_code].apply(lambda x: x.startswith(prefix)) ]

def make_training_data(df_diag, df_treat, target_code, n_samples=None, W=(60, 60), 
        pos_label=1, neg_label=0, shuffle=True):
    from sklearn.utils import shuffle

    dp = Diagnosis.properties
    col_key = dp['id']
    col_code = dp['code']

    tp = Treatment.properties
    col_code_t = tp['code']

    # df_diag, df_treat, df_res = load_data(input_dir=os.getcwd(), subset=False, verbose=True)

    # by default, use all prescriptions (drug_class) as variables
    assert col_code_t in df_treat.columns, "Drugs have not been encoded. See demo_encode_treatment()."
    vars_prescription = sorted(df_treat[col_code_t].unique())
    vars_to_index = {vp:i for i, vp in enumerate(vars_prescription)}
    Nf = len(vars_prescription)  # feature dimension

    # positive classe
    df_diag_pos = df_diag[df_diag[col_code]==target_code]
    df_diag_neg = df_diag[df_diag[col_code]!=target_code]
    target_patients = df_diag_pos[col_key].unique()  # patients with given diagnosis

    N0 = len(target_patients)
    df_treat_pos = df_treat[df_treat[col_key].isin(target_patients)]

    df_match = match(df_diag_pos, df_treat_pos, target_code, window=W, verbose=1)
    target_patients_pos = df_match[col_key].unique()
    N_pos = len(target_patients_pos)

    # each patient constitutes a training example 
    Xp = []
    n_actives = []  # check how many non-zero values in a feature vector
    for pid, dfi in df_match.groupby(col_key): 

        x = np.zeros(Nf)
        nf = 0
        for v, c in dfi[col_code_t].value_counts().items():
            x[vars_to_index[v]]=c
            nf += 1
        Xp.append(x)
        n_actives.append(nf)
    Xp = np.array(Xp)
    assert Xp.shape[0] == N_pos

    # ... now we have positive examples
    print(f"(make_training_data) 1. Average number of active variables: {np.mean(n_actives)}")
    print(f"... n_active<min>: {min(n_actives)}, n_active<max>: {max(n_actives)}")
    print(f"... N(pos): {Xp.shape[0]}")

    # Now, we find appropriate negative examples; we could make it harder to train by 
    # identiying "similar patients" but with different diagnoses
    N_neg = N_pos * 3 
    # N_similar_max = N_pos * 2

    similar_codes = set()
    for n_base in [3, 2, 1]:
        dfs = find_similar_diagnosis(df_diag_neg, target_code, n_base=n_base)
        
        sc = dfs[col_code].unique()
        similar_codes.update(sc)
        example_codes = np.random.choice(sc, min(5, len(sc)), replace=False)
        print(f"... example similar diagnoses (prefix len: {n_base}): {example_codes}")

        # subset = dfs[col_key].unique()
        # Nn = len(subset)
        # if n_base > 1: 
        #     if Nn >= N_similar_max:  # then the job is done, we have enough similar patients
        #         target_subset = np.random.choice(subset, N_similar_max, replace=False)
        #         df_similar = dfs[dfs[col_key].isin(target_subset)]
        #         print(f"... stopped at n_base={n_base}")
        #         break 
        # else: # n_base == 1, the last case of the same diagnostic category, we just have to choose either way
        #     target_subset = np.random.choice(subset, min(Nn, N_similar_max), replace=False)
        #     df_similar = dfs[dfs[col_key].isin(target_subset)]

    similar_codes = list(similar_codes)
    dissimilar_codes = list(set(df_diag_neg[col_code].unique())-set(similar_codes))
    n_codes_sim = len(similar_codes)
    n_codes_dis = len(dissimilar_codes)
    print(f"... n(similar): {n_codes_sim} <<? n(dissimilar): {n_codes_dis}")

    # chooose 50% similar and 50% dissimilar 
    ################################
    n_codes_ctrl = 200
    sim_ratio = 0.5
    dis_ratio = 1 - sim_ratio
    n_codes_ctrl_sim = int(n_codes_ctrl * sim_ratio)
    n_codes_ctrl_dis = n_codes_ctrl - n_codes_ctrl_sim
    # print(f"(debug) {n_codes_ctrl_sim}, {n_codes_ctrl_dis}, {min(n_codes_ctrl_sim, n_codes_sim)}, {min(n_codes_ctrl_dis, n_codes_dis)}")
    # print(f"(debug) {similar_codes}, {list(dissimilar_codes)[:5]}")
    similar_codes = np.random.choice(similar_codes, min(n_codes_ctrl_sim, n_codes_sim), replace=False) # NOTE: a set cannot be sampled
    dissimilar_codes = np.random.choice(dissimilar_codes, min(n_codes_ctrl_dis, n_codes_dis), replace=False)
    ################################

    target_codes = set(similar_codes).union(dissimilar_codes)
    n_codes_mixed = len(target_codes)
    print(f"(make_training_data) 2. Choose negative examples from n={n_codes_mixed} target_codes and their associated data.")

    df_diag_ctrl = df_diag_neg[df_diag_neg[col_code].isin(target_codes)]
    target_patients = df_diag_ctrl[col_key].unique()
    df_treat_ctrl = df_treat[df_treat[col_key].isin(target_patients)]
    # ... Now we have our finalized negative example candidates with a mixture of similar and dissimilar patients

    # we need this many negative examples per code
    ################################
    n_codes_per_target = int(N_neg/n_codes_mixed) * 2
    ################################

    df_ctrl = []
    for target_code in target_codes: 

        print(f"... (2.1) making control data, matching code: {target_code}")
        df_match = match(df_diag_ctrl, df_treat_ctrl, target_code=target_code, window=W, verbose=0)
        if df_match.empty: 
            print(f"... (2.2) found n=0 matched patients within code: {target_code}")
            continue
        else: 
            assert col_key in df_match

        target_patients = df_match[col_key].unique()
        n_matched = len(target_patients)
        print(f"... (2.2) found n={n_matched} matched patients within code: {target_code}")

        # if n_matched > n_codes_per_target: 
        #     subset = np.random.choice(target_patients, n_codes_per_target, replace=False)
        #     df_match = df_match[df_match[col_key].isin(subset)]

        df_ctrl.append(df_match)
    df_ctrl = pd.concat(df_ctrl, ignore_index=True)
    target_patients_ctrl = df_ctrl[col_key].unique()

    N_neg_ref = len(target_patients_ctrl) # number of patients in candidate control group
    if N_neg_ref > N_neg: 
        subset = np.random.choice(target_patients_ctrl, N_neg, replace=False)
        df_ctrl = df_ctrl[df_ctrl[col_key].isin(subset)]
        target_patients_ctrl = df_ctrl[col_key].unique()

    print(f"(make_training_data) 3. Combined control data comprising mixture of n={n_codes_mixed} codes.")
    print(f"... n(patients) in df_ctrl: {N_neg_ref} <?> {N_neg}\n... Finally we get n(patients) in ctrl group: {len(target_patients_ctrl)}")
    
    # Again, each patient constitutes a training example 
    Xn = []
    n_actives_neg = []  # check how many non-zero values in a feature vector
    for pid, dfi in df_ctrl.groupby(col_key): 
        x = np.zeros(Nf)
        nf = 0
        for v, c in dfi[col_code_t].value_counts().items():
            x[vars_to_index[v]]=c
            nf += 1
        Xn.append(x)
        n_actives_neg.append(nf)
    Xn = np.array(Xn)
    N_neg = Xn.shape[0]
    # assert Xn.shape[0] == N_neg  # this is not guaranteed, should only be approximately equal

    # ... now we have positive examples
    print(f"(make_training_data) 4. Average number of active variables in the control group: {np.mean(n_actives_neg)}")
    print(f"... n_active<min>: {min(n_actives_neg)}, n_active<max>: {max(n_actives_neg)}")
    print(f"... N(neg): {Xn.shape[0]}")

    y = np.hstack( (np.repeat(pos_label, Xp.shape[0]), np.repeat(neg_label, Xn.shape[0])) )
    X = np.vstack((Xp, Xn))

    if shuffle: 
        X, y = shuffle(X, y, random_state=53)

    return (X, y) 

def summarize_dict(d, topn=15, sort_=True, prefix=''): 
    if topn != 0 or sort_: 
        import operator
        d = sorted(d.items(), key=operator.itemgetter(1))
    for k, v in d[:topn]: 
        print(f"{prefix}[{k}] -> {v}")
    return

# Demo codes below
###############################################################################

def demo_filter(): 
    from data_pipeline import load_data

    col_key = 'Patient_id'
    col_date = 'Diag_date'
    col_code = 'ICD10'

    df_diag, df_treat, df_res = load_data(input_dir=os.getcwd(), subset=False, verbose=True)
    
    Np0 = len(df_diag[col_key].unique())
    df_diag_sub = filter_by_ICD(df_diag, disease_specific=True) # retain rows with valid and disease-specific ICD10 codes
    print(f"Number of (unique) patients prior to filtering by disease-specific ICD: {Np0}")
    print(f"Number of patients after filtering filtering by disease-specific ICD  : {len(df_diag_sub[col_key].unique())}")

    df_diag_sub = filter_by_ICD(df_diag, disease_specific=False) # retain rows with valid ICD10 codes 
    print(f"Number of patients after filtering by validity of ICD: {len(df_diag_sub[col_key].unique())}")
    
    return

def demo_sequencing(dtype='diag', verbose=1): 
    from data_pipeline import load_data

    col_key = 'Patient_id'
    df_diag, df_treat, df_res = load_data(input_dir=os.getcwd(), subset=False, verbose=True)

    horizon = 180

    if dtype.startswith('diag'):
        print("> Sequencing diagnosis data ...")
        Np = len(df_diag[col_key].unique())

        diag = Diagnosis(df_diag)
        # df_seq = transform_diagnosis(df_diag, df_res=df_res, tIncludeTime=False, 
        #               tFilterByLength=True, tFilterByICD=True, tDocumentedOnly=False, n_days_lookback=180)
        df_seq = diag.sequence(tFilterByICD=True, tFilterByLength=True, n_days_lookback=horizon)
        Ns = len(df_seq[col_key].unique())

        print(f"Number of (unique) patients in original data: {Np}, in sequenced file: {Ns} | {Np} >=? {Ns}")

        # save to disk 
        diag.save(df_seq, dtype='seq', verbose=True)

        # load from disk 
        df_seq = diag.load(dtype='seq', verbose=True)
        Ns_prime = len(df_seq[col_key].unique())
        assert Ns_prime == Ns
    else: 
        print("> Sequencing treatment data ...")
        Np = len(df_treat[col_key].unique())

        treatment = Treatment(df_treat)
        df_seq = treatment.sequence(tFilterByLength=True, n_days_lookback=horizon)
        Ns = len(df_seq[col_key].unique())

        print(f"Number of (unique) patients in original data: {Np}, in sequenced file: {Ns} | {Np} >=? {Ns}")

        # save to disk 
        treatment.save(df_seq, dtype='seq', verbose=True)
 
        # load from disk 
        df_seq = treatment.load(dtype='seq', verbose=True)
        Ns_prime = len(df_seq[col_key].unique())
        assert Ns_prime == Ns

    if verbose: 
        col_intv = Diagnosis.properties['history']  # Diagnosis and Treatment shares the same sequencing properties
        M, m = df_seq[col_intv].max(), df_seq[col_intv].min()
        msg = f"(demo) Length distribution of medical history in days (dtype={dtype}):\n"
        msg += f"... min: {m}, max: {M}\n"
        msg += f"... mean: {df_seq[col_intv].mean()}, median: {df_seq[col_intv].median()}\n"
        msg += f"... std: {df_seq[col_intv].std()}\n"
        print(msg)

        pids = df_seq[col_key].sample(n=5).values
        seq_type = 'd' if dtype.startswith('diag') else 'p'
        print(f"> Patients vs {seq_type}-sequences ({len(pids)}/{len(df_seq[col_key].unique())} patients) ...")
        for r, row in df_seq[df_seq[col_key].isin(pids)].iterrows(): 
            print(f"> ID:      {row[col_key]}")
            print(f"> history: {row[col_intv]} days")
            print(f"> sequence:\n{row[col_code]}\n")

    return df_seq

def demo_topn_diagnoses(n=None, topn=15): 
    from data_pipeline import load_data
    from tabulate import tabulate
    from icd_utils import encode

    p = Diagnosis.properties
    col_key = p['id']
    col_code = p['code']

    horizon = 180
    diag = Diagnosis()
    try: 
        df_seq = diag.load(dtype='seq', verbose=True) # assuming that the sequence file has be generated
    except: 
        diag.init()
        df_seq = diag.sequence(tFilterByICD=True, tFilterByLength=True, n_days_lookback=horizon)

    if n is not None: # focus only a subset of patients
        subset = np.random.choice(df_seq[col_key].unique(), n, replace=False)
        df_seq = df_seq[df_seq[col_key].isin(subset)]
        print(f"> Focus only on a subset of patients (n={len(subset)})")

    col_topn = 'Topn' # health status
    df_topn_local, df_topn_global = \
        diag.eval_topn_diagnoses(df_seq=df_seq, topn=3, topn_global=10, overwrite=False, 
                                    col_key=col_key, col_code=col_code, col_topn=col_topn)

    counter = collections.Counter()
    for status in df_topn_local[col_topn].values: 
        counter.update( status.split() )

    print(f"> There are n={len(counter)} (unique) primary diagnosis codes in total.")

    sorted_key_value = counter.most_common()
    most_common_target, most_common_count = sorted_key_value[0][0], sorted_key_value[0][1]
    least_common_target, least_common_count = sorted_key_value[-1][0], sorted_key_value[-1][1]
 
    print(f"> Most common primary diagnosis: {most_common_target}, n={most_common_count}")
    print(f"> Least common primary diagnosis: {least_common_target}, n={least_common_count}")

    # most common disease-specific ICD10 codes? 
    df = DataFrame(counter.items(), columns=['code', 'count'])
    df_topn = df[df['code'] <= 'O00'].sort_values('count', ascending=False).iloc[:topn]
    most_common_conditions = df_topn['code'].values

    codebook = make_lookup_table()  # maps from diag (ICD10 in regular format) to code_desc
    df_topn['code_desc'] = [codebook[code] for code in df_topn['code']]
    print(f"> Most most common conditions:\n{most_common_conditions}\n")
    print(tabulate(df_topn, headers='keys', tablefmt='psql', showindex=False))

    df_botn = df[df['code'] <= 'O00'].sort_values('count', ascending=True).iloc[:topn]
    least_common_conditions = df_botn['code'].values
    df_botn['code_desc'] = [codebook.get(code, 'n/a') for code in df_botn['code']]

    print(f"> Least common conditions:\n{least_common_conditions}\n")
    print(tabulate(df_botn, headers='keys', tablefmt='psql', showindex=False))

    return df_topn, df_botn

def demo_topn_prescriptions(n=None, topn=15): 
    from data_pipeline import load_data 

    p = Treatment.properties
    col_key = p['id']
    col_code = p['code']
    col_class = p['class']

    horizon = 180
    treatment = Treatment()
    try: 
        df_seq = treatment.load(dtype='seq', verbose=True) # assuming that the sequence file has be generated
    except: 
        treat.init()
        df_seq = treatment.sequence(tFilterByLength=True, n_days_lookback=horizon)

    if n is not None: # focus only a subset of patients
        subset = np.random.choice(df_seq[col_key].unique(), n, replace=False)
        df_seq = df_seq[df_seq[col_key].isin(subset)]
        print(f"> Focus only on a subset of patients (n={len(subset)})")

    col_topn = 'Topn' # Prime prescriptions
    df_topn_local, df_topn_global = \
        treatment.eval_topn_prescriptions(df_seq=df_seq, topn=5, topn_global=10, overwrite=False, col_topn=col_topn)

    counter = collections.Counter()
    for status in df_topn_local[col_topn].values: 
        counter.update( status.split() )


    df_treat = Treatment.load(dtype='in', verbose=False)   
    df_treat = df_treat[df_treat[col_key].isin(df_seq[col_key].values)]
    name_dict = dict(zip(df_treat[col_code], df_treat[col_class]))

    sorted_key_value = counter.most_common()
    most_common_target, most_common_count = sorted_key_value[0][0], sorted_key_value[0][1]
    least_common_target, least_common_count = sorted_key_value[-1][0], sorted_key_value[-1][1]
 
    print(f"> Most common primary prescritpions: {most_common_target}({name_dict[most_common_target]}), n={most_common_count}")
    print(f"> Least common primary prescritpions: {least_common_target}({name_dict[least_common_target]}), n={least_common_count}")
    print(f"> There are n={len(counter)} prescritpion codes in total.")    

    return

def demo_time_arithmatic():
    from data_pipeline import load_data

    col_key = 'Patient_id'
    col_date = 'Diag_date'
    col_code = 'ICD10'

    n_samples = 1000
    # load a subset of the patient data (by default only randomly select from patients that appear in both
    # diagnosis and treatment tables)
    df_diag, df_treat, df_res = load_data(input_dir=os.getcwd(), n=n_samples, verbose=False)

    # compute boundaries of a time window centered on a target date 
    target_date = '2017-02-21' # df_diag[col_date].median()
    print(f"Example diag date: {target_date}")

    t1, t2 = time_window(target_date, bounds=(100, 100), 
                            constraints=(df_diag[col_date].min(), df_diag[col_date].max()), strftime='%Y-%m-%d')

    print(f"{t1} < {target_date} < {t2}")

    return

def demo_encode_treatment(): 
    from data_pipeline import load_data

    p = Treatment.properties
    col_key = p['id']
    col_code = p['code']

    df_diag, df_treat, df_res = load_data(input_dir=os.getcwd(), subset=False, verbose=False)
    df_derived = encode_prescriptions(df_treat) # by default: prefix='D', sep='-'

    treatment = Treatment(df_treat)
    treatment.encode(save=True)

    assert df_derived[col_key].equals(treatment.df[col_key])
    assert df_derived[col_code].equals(treatment.df[col_code])

    return

def demo_match(target_codes=[], W=(60, 60), n_samples=None, verbose=1): 
    from data_pipeline import load_data

    # Get diagnosis and treatment table (subsampling if n_samples is given)
    ########################################################
    df_diag, df_treat, df_res = load_data(input_dir=os.getcwd(), n=n_samples, verbose=verbose > 1)
    ########################################################

    if not target_codes: target_codes = ['J06.9', ]
    
    for target_code in target_codes: 
        df_match = match(df_diag, df_treat, target_code, window=W, verbose=1)

    return

def demo_make_classification(target_codes=[], W=(60, 60), verbose=0):
    from data_pipeline import load_data, save_XY, load_XY
    # save training data numpy array as NPZ
    from numpy import asarray
    from numpy import savez_compressed
    from numpy import load as loadz_compressed

    # Get diagnosis and treatment table (subsampling if n_samples is given)
    ########################################################
    df_diag, df_treat, df_res = load_data(input_dir=os.getcwd(), verbose=verbose)
    ########################################################

    if not target_codes: 
        target_codes = ['I10', 'N39.0', 'E78.2', 'F41.9', 'K21.9', 'J20.9' ] # [ 'M54.5', 'L70.0', 'J06.9', 'E11.9']
    
    for target_code in target_codes: 
        X, y = make_training_data(df_diag, df_treat, target_code, n_samples=None, W=W) 

        save_XY(X, y, suffix=target_code)  # specify input_path to save to a different directory other than current dir

        Xp, yp = load_XY(suffix=target_code)

        assert Xp.shape == X.shape
        assert yp.shape == y.shape

    return

def test():

    # demo_time_arithmatic()
    # demo_filter()
    # demo_sequencing(dtype='diag')
    demo_topn_diagnoses()

    # demo_encode_treatment()
    # demo_sequencing(dtype='treat')
    # demo_topn_prescriptions()

    # demo_match()

    # demo_make_classification()

    return

if __name__ == "__main__": 
    test()