import string, sys
import math
import pandas as pd
from pandas import DataFrame
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)
 
def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)
 
def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)
 
def augmented_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
    return (0.5 + ((0.5 * term_frequency(term, tokenized_document))/max_count))
 
def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values
 
def tfidf(documents, tokenize):
    tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents

def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude


##########################################################
# --- interpret TF-IDF

def top_features_in_doc(Xtr, features, row_id, top_n=25):
    """
    Top tfidf features in specific document (matrix row)

    Memo
    ----
    1. np.squeeze()

       Remove single-dimensional entries from the shape of an array.
    """
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_features(row, features, top_n)

def top_tfidf_features(row, features, top_n=25):
    """
    Get top n tfidf values in row and return their corresponding feature names.

    Memo
    ----
    1. np.argsort() returns the indices of the ordering
    """
    topn_ids = np.argsort(row)[::-1][:top_n]  # x[::-1] puts x in reversed order
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'score']
    return df

def top_mean_features(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    """

    Input
        Xtr: TF-IDF-transformed feature representation where each row corresponds to 
             a document ID and each column corresponds to a token (or n-grams in general)
        features: the set of vocabulatry over which a TF-IDF score is computed for each 
                  document

    Return the top n features that on average are most important amongst documents in rows
    indentified by indices in grp_ids. 
    """
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_features(tfidf_means, features, top_n)
def top_median_features(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.median(D, axis=0)
    return top_tfidf_features(tfidf_means, features, top_n)

def top_features_by_class(Xtr, y, features, min_tfidf=0.1, top_n=25):
    """
    Return a list of dfs, where each df holds top_n features and their mean tfidf value
    calculated across documents with the same class label. 
    """
    dfs = []
    labels = np.unique(y)
    for label in labels:
        ids = np.where(y==label)
        feats_df = top_mean_features(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)
        feats_df.label = label
        dfs.append(feats_df)
    return dfs

def plot_tfidf_classfeats_h(dfs):
    """
    Plot the data frames returned by the function top_features_by_class(). 
    """
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("label = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.barh(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([-1, x[-1]+1])
        yticks = ax.set_yticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()

##########################################################

def eval_topn_features(df_seq, topn=3, **kargs):
    
    col_key = kargs.get('col_key', 'Patient_id')
    col_doc = kargs.get('col_doc', 'ICD10')  # options: 'drug_code'
    assert set([col_key, col_doc]) <= set(df_seq.columns)

    col_topn = kargs.get('col_topn', 'Primary') 
    # place_holder = kargs.pop('place_holder', 'X')
    sep = kargs.pop('sep', " ") # top N feature/code separator
    n_examples = kargs.pop('n_test_examples', 10)
    topn_global = kargs.pop('topn_global', 10) # show most important features in global scope (i.e. across documents)
    verbose = kargs.get('verbose', 1)

    pids = df_seq[col_key].values
    docs = df_seq.set_index(col_key)[col_doc].to_dict()
    corpus = np.array([docs[pid] for pid in pids])
    Nd, Np = len(docs), len(pids)

    ngram_range = (1, 1)
    stop_words = []

    tokenizer = lambda doc: doc.split(" ")
    model = TfidfVectorizer(analyzer='word', tokenizer=tokenizer, ngram_range=ngram_range, 
                min_df=0, smooth_idf=True, lowercase=False, stop_words=stop_words) 

    Xtr = model.fit_transform(corpus)
    assert Xtr.shape[0] == len(pids), f"number of rows in Xtr {Xtr.shape[0]} not equal to num of (unique) patients {len(pids)}"
    if verbose: print(f"(eval_topn_features) Nd: {Nd}, dim(Xtr): {Xtr.shape}, size(vocab): {len(model.vocabulary_)}")

    fset = model.get_feature_names()
    if verbose: print(f"... example feature names (n={len(fset)}=?={len(model.vocabulary_)}):\n{fset[:50]}\n{fset[-50:]}\n")

    test_indices = np.random.choice(range(Nd), n_examples, replace=False)
    top_features = []
    for i in range(Xtr.shape[0]):
        df_doc = top_features_in_doc(Xtr, features=fset, row_id=i, top_n=topn) 

        # "remove" features whose scores are 0 (and replace them with a placeholder) 
        df_doc = df_doc[df_doc['score'] > 0]
        assert not df_doc.empty, f"top N({topn}) features all have zero scores?\n{df_doc}\n"

        # columns: feature, score
        topfs = sep.join(df_doc['feature'].values)
        top_features.append( topfs )
        if verbose and (i in test_indices): 
            print("--- doc #{} ---\n{}".format(i, df_doc.to_string(index=True)))
            print("...... top N({}) features: {}\n".format(topn, topfs))
    
    df_topn_local = DataFrame({col_key: pids, col_topn: top_features}, columns=[col_key, col_topn])
    df_topn_global = top_mean_features(Xtr, fset, grp_ids=None, min_tfidf=0.1, top_n=topn_global)
    if verbose: 
        print("(eval_topn_features) top N features overall across all docs ...")
        print("... doc(mean):\n{}\n".format(df_topn_global.to_string(index=True)))

    return df_topn_local, df_topn_global

def demo_tfidf_diagnosis(df_seq=None):
    import os
    from data_pipeline import load_data
    from utils import Diagnosis

    col_key = 'Patient_id'
    col_date = 'Diag_date'
    col_code = 'ICD10'
    col_intv = 'History'

    n_samples = 1000
    tLoad = True

    if df_seq is None: 
        df_diag, df_treat, df_res = load_data(input_dir=os.getcwd(), verbose=False)
        diag = Diagnosis(df_diag) # create a Diagnosis object
        if tLoad: 
            df_seq = diag.load(dtype='seq') # load the pre-computed sequence data, because it's take a minute or two to sequence the data
        else: 
            df_seq = diag.sequence(tFilterByICD=True, tFilterByLength=False)
            # note: set tFilterByICD to True to only include valid (well-formatted) ICD10 codes
            #       set tFilterByLength to False to include the entire d-sequence for each patient
            #       Say you want to focus on only the most recent 100 days of diagnoses, then pass 
            #       n_days_lookback=100
    else: 
        assert not df_seq.empty

    pids = df_seq[col_key].values
    docs = df_seq.set_index(col_key)[col_code].to_dict()
    corpus = np.array([docs[pid] for pid in pids])
    Nd, Np = len(docs), len(pids)

    ngram_range = (1, 1)
    stop_words = []

    tokenizer = lambda doc: doc.split(" ")
    model = TfidfVectorizer(analyzer='word', tokenizer=tokenizer, ngram_range=ngram_range, 
                min_df=0, smooth_idf=True, lowercase=False, stop_words=stop_words) 

    Xtr = model.fit_transform(corpus)
    analyzer = model.build_analyzer()
    print("(demo_tfidf_diagnosis) ngram_range: {} => {}".format(ngram_range, analyzer("D64.9")))
    print(f"... Nd: {Nd}, dim(Xtr): {Xtr.shape}, size(vocab): {len(model.vocabulary_)}")
    assert Xtr.shape[0] == len(pids), f"number of rows in Xtr {Xtr.shape[0]} not equal to num of (unique) patients {len(pids)}"

    fset = model.get_feature_names()
    assert sum(1 for w in stop_words if w in fset) == 0, "Found stop words in the feature set!"
    print(f"... example feature names (n={len(fset)}=?={len(model.vocabulary_)}):\n{fset[:50]}\n{fset[-50:]}\n")

    n_examples = 10
    test_indices = np.random.choice(range(Nd), n_examples, replace=False)
    for i, dvec in enumerate(Xtr):
        # if i in test_indices: 
        #     print("...... doc #[{}]:\n{}\n".format(i, dvec.toarray()))
        assert np.sum(dvec) > 0

    print("... size(ICD10 codes): {}".format( len(model.vocabulary_) ))

    # --- interpretation 
    print("(demo_tfidf_diagnosis) Interpreting the TF-IDF model")
    topn = 3
    for i in range(Xtr.shape[0]):
        df_doc = top_features_in_doc(Xtr, features=fset, row_id=i, top_n=topn)
        if i in test_indices: 
            print("...... doc #{}:\n{}\n".format(i, df_doc.to_string(index=True)))

    topn = 10
    print("... top N ICD10 codes overall across all docs")
    df_topn = top_mean_features(Xtr, fset, grp_ids=None, min_tfidf=0.1, top_n=top_n)
    print("... doc(avg):\n{}\n".format(df_topn.to_string(index=True)))

    # --- interface
    # a. get the scores of individual tokens or n-grams in a given document? 
    print("(demo_tfidf_diagnosis) Get the scores of individual ICD10s in a given d-sequence")
    df = pd.DataFrame(Xtr.toarray(), columns = model.get_feature_names())
    print(df.head())

    # df.to_csv(Diagnosis.get_path(dtype='tfidf'), sep='|', index=False, header=True)

    return df


def demo_tfidf_transform(**kargs):
    """

    Memo
    ----

    """
    # from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    # ... compute dot product

    docs = {}
    docs[0] = "SMN1 GENE MUTATION ANALYSIS BLOOD TISSUE MOLECULAR GENETICS METHOD NARRATIVE"
    docs[1] = "SMN1 GENE TARGETED MUTATION ANALYSIS BLOOD TISSUE MOLECULAR GENETICS METHOD"
    docs[2] = "SALMON IGE AB SERUM"
    docs[3] = "SCALLOP IGE AB RAST CLASS SERUM"
    docs[4] = "SJOGRENS SYNDROME A EXTRACTABLE NUCLEAR AB SERUM"
    docs[5] = "MYELOCYTES BLOOD"

    dtest = {}
    dtest[0] = "SCALLOP IGE AB RAST CLASS SERUM"

    corpus = np.array([docs[i] for i in range(len(docs))])
    vectorizer = CountVectorizer(decode_error="replace")
    vec_train = vectorizer.fit_transform(corpus)

    # -- model persistance
    # # Save vectorizer.vocabulary_
    # pickle.dump(vectorizer.vocabulary_,open("feature.pkl","wb"))

    # # Load it later
    # transformer = TfidfTransformer()
    # loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
    # tfidf = transformer.transform(loaded_vec.fit_transform(np.array(["aaa ccc eee"])))

    # vec = TfidfVectorizer()
    # tfidf = vec.fit_transform()

    ngram_range = (1,3)
    stop_words = ['METHOD', 'CLASS']
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=ngram_range, min_df=0, smooth_idf=True, stop_words=stop_words) 
    # sublinear_tf=True? it's unlikely to observe repeated tokens in the LOINC long name or MTRT

    Xtr = tfidf.fit_transform(corpus)
    
    analyzer = tfidf.build_analyzer()
    print("... ngram_range: {} => {}".format(ngram_range, analyzer("RHEUMATOID FACTOR IGA SERUM")))

    # --- get feature index
    part_sent = "CLASS SERUM"
    feature_index = tfidf.vocabulary_.get("CLASS SERUM".lower())  # lowercase: True by default
    print("... phrase: {} => {}".format(part_sent, feature_index))

    # > size of the vocab
    # tfidf.vocabulary_: a dictionary
    print("... size(vocab): {}".format( len(tfidf.vocabulary_) ))

    # -- doc vectors
    # print("... d2v(train):\n{}\n".format( tfidf.to_array() ))
    fset = tfidf.get_feature_names()
    assert sum(1 for w in stop_words if w in fset) == 0, "Found stop words in the feature set!"
    print("> feature names:\n{}\n".format(fset))
    for i, dvec in enumerate(Xtr):
        print("> doc #[{}]:\n{}\n".format(i, dvec.toarray()))


    # --- predicting new data
    corpus_test = np.array([doc for i, doc in dtest.items()])
    doc_vec_test = tfidf.transform(corpus_test)
    print("... d2v(test):\n{}\n".format( doc_vec_test.toarray() ))

    # --- interpretation 
    print("(demo_predict) Interpreting the TF-IDF model")
    for i, dvec in enumerate(Xtr):
        # top_tfidf_features(dvec, features=tfidf.get_feature_names(), top_n=10)
        df = top_features_in_doc(Xtr, features=fset, row_id=i, top_n=10)
        print("... doc #{}:\n{}\n".format(i, df.to_string(index=True)))

    print("... top n features overall across all docs")
    df = top_mean_features(Xtr, fset, grp_ids=None, min_tfidf=0.1, top_n=10)
    print("... doc(avg):\n{}\n".format(df.to_string(index=True)))

    # --- interface
    # a. get the scores of individual tokens or n-grams in a given document? 
    print("> Get the scores of individual tokens or n-grams in a given document? ")
    df = pd.DataFrame(Xtr.toarray(), columns = tfidf.get_feature_names())
    vocab = ['salmon ige ab', 'salmon']
    print(df.head())

    return

def demo_tfidf(**kargs):
    """


    Memo
    ----
    TfidfVectorizer is equivalent to CountVectorizer followed by TfidfTransformer, where

    CountVectorizer: Transforms text into a sparse matrix of n-gram counts.
    TfidfTransformer: Performs the TF-IDF transformation from a provided matrix of counts.
    """
    # import string, sys
    # import math
    # from sklearn.feature_extraction.text import TfidfVectorizer

    tokenizer = lambda doc: doc.upper().split(" ")
 
    document_0 = "SMN1 GENE MUTATION ANALYSIS BLOOD TISSUE MOLECULAR GENETICS METHOD NARRATIVE"
    document_1 = "SMN1 GENE TARGETED MUTATION ANALYSIS BLOOD TISSUE MOLECULAR GENETICS METHOD"
    document_2 = "SALMON IGE AB SERUM"
    document_3 = "SCALLOP IGE AB RAST CLASS SERUM"
    document_4 = "SJOGRENS SYNDROME A EXTRACTABLE NUCLEAR AB SERUM"
    document_5 = "MYELOCYTES BLOOD"
    document_6 = "KAPPA LIGHT CHAINS FREE 24 HOUR URINE"
 
    all_documents = [document_0, document_1, document_2, document_3, document_4, document_5, document_6]
     
    sklearn_tfidf = TfidfVectorizer(norm='l2', min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenizer)
    # sublinear_tf: Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf)
    # smooth_idf: Smooth idf weights by adding one to document frequencies, as if an extra document 
    #             was seen containing every term in the collection exactly once. Prevents zero divisions.

    tfidf_representation = tfidf(all_documents, tokenizer)
    sklearn_representation = sklearn_tfidf.fit_transform(all_documents) 
    # sklearn_representation: a sparse matrix

    # print(tfidf_representation[0])
    # print(sklearn_representation.toarray()[0].tolist())

    my_tfidf_comparisons = []
    for count_0, doc_0 in enumerate(tfidf_representation):
        for count_1, doc_1 in enumerate(tfidf_representation):
            my_tfidf_comparisons.append((cosine_similarity(doc_0, doc_1), count_0, count_1))

    skl_tfidf_comparisons = []
    for count_0, doc_0 in enumerate(sklearn_representation.toarray()):
        for count_1, doc_1 in enumerate(sklearn_representation.toarray()):
            skl_tfidf_comparisons.append((cosine_similarity(doc_0, doc_1), count_0, count_1))

    for x in zip(sorted(my_tfidf_comparisons, reverse = True), sorted(skl_tfidf_comparisons, reverse = True)):
        print(x)

    return


def test(): 

    # --- TF-IDF encoding
    # demo_tfidf()

    # --- prediction using the vectors produced by TF-IDF encoding 
    # demo_tfidf_transform()

    # --- TF-IDF scores for d-sequences 
    demo_tfidf_diagnosis()

    return

if __name__ == "__main__": 
    test()