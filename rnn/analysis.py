from joblib.parallel import delayed
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from scipy.optimize import curve_fit
import pingouin as pg
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as sch
import math
from tensorly.decomposition import parafac

def run_pca(hs):
    trials, timesteps, batch_size, hidden_dim = hs.shape
    pca = PCA(n_components=3)
    low_x = pca.fit_transform(hs.reshape(trials*timesteps*batch_size, hidden_dim)).reshape(trials, timesteps, batch_size, 3)
    return low_x.reshape(trials, timesteps, batch_size, 3)

def run_tca(ws):
    trials, timesteps, batch_size, hidden_dim, hidden_dim = ws.shape
    factors = parafac(ws.reshape(trials*timesteps*batch_size, hidden_dim, hidden_dim), rank=3)
    low_w = factors[0]
    return low_w.reshape(trials, timesteps, batch_size, 3)

def linear_regression(X, y):
    res = pg.linear_regression(X, y)
    return res

def linear_classification_cv(X, y):
    clf = LinearSVC()
    scores = cross_val_score(clf, X, y, cv=5)
    return np.mean(scores)

def forward_encoding(xs, hs, NPROC=16):
    n_time_steps, n_trials, n_hidden = hs.shape
    assert(xs.shape[:2]==(n_time_steps, n_trials))
    results = [[None] * n_hidden] * n_time_steps
    for t in range(n_time_steps):
        jobs = [delayed(linear_regression)(xs[t], hs[t,:,i]) for i in range(n_hidden)]
        results[t] = Parallel(n_jobs=NPROC)(jobs)
    return results

def backward_decoding(hs, ys, NPROC=16):
    n_time_steps, n_trials, n_hidden = hs.shape
    assert(ys.shape[:2]==(n_time_steps, n_trials))
    jobs = [delayed(linear_classification_cv)(hs[t], ys[t]) for t in range(n_time_steps)]
    results = Parallel(n_jobs=NPROC)(jobs)
    return results
    
def representational_similarity_analysis(xs, hs, NPROC=16):
    n_time_steps, n_trials, n_hidden = hs.shape
    hs = hs.reshape((n_time_steps*n_trials, n_hidden))
    rnn_sim = pdist(hs, metric='euclidean')
    input_sim = []
    for x in xs: # for each input measure
        x = x.reshape((n_time_steps*n_trials, 1))
        input_sim.append(pdist(x, metric='euclidean'))
    input_sim = np.stack(input_sim).transpose(1,0)
    result = linear_regression(input_sim, rnn_sim)
    return rnn_sim, input_sim, result

def fit_exp(y,x=None):
    if x is None:
        x = np.arange(y.shape[0])
    pred = lambda p: p[0]*(1-math.exp(-x/p[1]))
    p0 = [np.rand()*10, np.rand()*1e4]
    lb = [0, 0]
    ub = [10, 1e4]
    popt, _ = curve_fit(pred, x, y, bound=(lb, ub), p0=p0)
    return popt[0]*(1-math.exp(-x/popt[1]))

def hierarchical_clustering(x):
    dmat = pdist(x)
    Z = sch.linkage(dmat, method='centroid')
    return Z