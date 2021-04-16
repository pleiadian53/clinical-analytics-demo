
import os, sys
from pandas import DataFrame, Series
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator # used to force yticks to assume integer values
plt.style.use('ggplot')

def plot_sessions(df, key='Patient_id', col='Diag_date', title='Diagnosis', 
                      show_key=False, ascending=False, figsize=(16, 8), pids=[]):
    '''
    Plot histogram of patient vs number of {diagnoses, treatments} 
    '''
    def transform_date(x, sep='-'): 
        fields = x.split(sep)
        assert len(fields) == 3
        return "%s-%s" % (fields[0], fields[1])
    
    if col is None: 
        # heuristic: automatic search for date-specific attribute 
        col = [c for c in df.columns if c.lower().find('date') > 0][0]
         
    df['dt'] = df[col].astype("datetime64")
    df['dt'] = df['dt'].apply(lambda x: "%s-%s" % (x.year, x.month))
    # df['dt'] = df[col].apply(transform_date)
    
    if not pids: 
        patients_to_nsessions = df.groupby(key)['dt'].nunique().sort_values(ascending=ascending)
        # ~> a series: Patient_id as index and number of sessions as values
    else: 
        # only focus on the given patient IDs
        patients_to_nsessions = df[df[key].isin(pids)].groupby(key)['dt'].nunique().reindex(pids)
    
    patients_to_nsessions.plot(kind="bar", rot=75, 
                                   figsize=figsize, use_index=show_key, color='skyblue') # e.g. PatientID vs *Date
        
    df.drop(['dt',], axis=1, inplace=True)
    
    plt.title(title)
    plt.xlabel(f"Patient IDs")
    plt.ylabel(f"Number of sessions")
    plt.show()
    
    return list(patients_to_nsessions.index)

def plot_roc(cv_data, **kargs):
    """
    
    Params
    ------
    cv_data: a list of (y_true, y_score) obtained from a completed CV process
    """
    from scipy import interp
    from sklearn.metrics import roc_curve, auc

    method = kargs.get("method", "test")
    prefix_id = kargs.get('prefix', '')
    show_plot = kargs.get("show", False)
    verbose = kargs.get("verbose", True)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    n_fold = len(cv_data)
    if not n_fold: 
        print('(plot_roc) No CV data. Aborting ...')
        return

    plt.clf()
    for i, (y_true, y_score) in enumerate(cv_data):

        fold_num = i+1
        if (n_fold > 5 and fold_num % 2 == 0) or n_fold <= 5: 
            # probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y_true, y_score) # roc_curve(y[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
     
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                        label='ROC fold %d (AUC = %0.2f)' % (fold_num, roc_auc))

    ### plotting
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    if prefix_id: 
        plt.title('ROC for {}: method={}, nfold={})'.format(prefix_id, method, n_fold)) 
    else: 
        plt.title('ROC for {}, nfold={})'.format(method, n_fold))
    plt.legend(loc="lower right")
    
    filename = kargs.get('filename', 'roc')
    if prefix_id: 
        filename = f"{prefix_id}-roc"
    output_dir = kargs.get("output_dir", "plot")
    output_path = os.path.join(output_dir, filename)  # example path: System.analysisPath
    if verbose: 
        print(f"[plot] Saving ROC curve to:\n{output_path}\n")
    saveFig(plt, output_path, dpi=300, message="[output] ROC curve", verbose=verbose)

    if show_plot: 
        plt.show()

    return


def saveFig(plt, fpath, ext='tif', dpi=300, message='', verbose=True):
    """
    Save pyplot figure. 

    Params
    ------
    fpath: output path of the plot
    message: stdout message for debugging/test purposes

    Memo
    ----
    1. supported graphing format: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff 

    """
    import os
    outputdir, fname = os.path.dirname(fpath), os.path.basename(fpath) 

    # [todo] abstraction
    supported_formats = ['eps', 'jpeg', 'jpg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff', ]  

    # supported graphing format: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
    if not outputdir: outputdir = os.getcwd() # sys_config.read('DataExpRoot') # ./bulk_training/data-learner)
    # assert os.path.exists(outputdir), "Invalid output path: %s" % outputdir
    if not os.path.exists(outputdir):
        if verbose: print("(savefig) Creating directory: {}".format(outputdir))
        os.mkdir(outputdir)

    ext_plot = ext  # eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff.
    if not fname: fname = 'generic-test.%s' % ext_plot
    fbase, fext = os.path.splitext(fname)
    # if fext[1:] in supported_formats, "Unsupported graphic format: %s" % fname
    if not fext: fname = fbase + '.tif' 

    fpath = os.path.join(outputdir, fname)
    if verbose: print('(saveFig) Saving plot to:\n%s\n... description: %s' % (fpath, 'n/a' if not message else message))
    
    # pylab leaves a generous, often undesirable, whitespace around the image. Remove it by setting bbox_inches to tight
    plt.savefig(fpath, bbox_inches='tight', dpi=dpi)   
    return

# some helpful utilities for demo
def indent(message, nfill=6, char=' ', mode='r'): 
    if mode.startswith('r'): # left padding 
        return message.rjust(len(message)+nfill, char)
    return message.ljust(len(message)+nfill, char)

def div(message=None, symbol='=', prefix=None, n=80, adaptive=False, border=0, offset=5, stdout=True): 
    output = ''
    if border is not None: output = '\n' * border
    # output += symbol * n + '\n'
    if isinstance(message, str) and len(message) > 0:
        if prefix: 
            line = '%s: %s\n' % (prefix, message)
        else: 
            line = '%s\n' % message
        if adaptive: n = len(line)+offset 

        output += symbol * n + '\n' + line + symbol * n
    elif message is not None: 
        # message is an unknown object of some class
        if prefix: 
            line = '%s: %s\n' % (prefix, str(message))
        else: 
            line = '%s\n' % str(message)
        if adaptive: n = len(line)+offset 
        output += symbol * n + '\n' + line + symbol * n
    else: 
        output += symbol * n
        
    if border is not None: 
        output += '\n' * border
    if stdout: print(output)
    return output
### alias 
highlight = div