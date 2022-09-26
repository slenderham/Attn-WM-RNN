import math

import numpy as np
import pingouin as pg
import scipy.cluster.hierarchy as sch
from joblib import Parallel, delayed
from joblib.parallel import delayed
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from tensorly.decomposition import parafac, CP
from dPCA import dPCA

def run_pca(hs, rank=3):
    trials, timesteps, batch_size, hidden_dim = hs.shape
    pca = PCA(n_components=rank)
    low_x = pca.fit_transform(hs.reshape(trials*timesteps*batch_size, hidden_dim)).reshape(trials, timesteps, batch_size, 3)
    return low_x.reshape(trials, timesteps, batch_size, 3)

def run_tca(ws, ranks=9, num_reps=5, rank_err_tol):
    trials, timesteps, batch_size, post_dim, pre_dim = ws.shape
    results = {}
    for r in ranks:
        results[r] = []
        for n in num_reps:
            cp = CP(rank=r)
            cp.fit_transform(ws.reshape(trials*timesteps*batch_size, post_dim, pre_dim), return_errors=True)
            results[r].append(cp)
    
    for r in ranks:
        idx = np.argsort([rr.errors_[-1] for rr in results[r]])
        


    low_w = factors[0]
    post_factor = factors[1]
    pre_factor = factors[1]
    return low_w.reshape(trials, timesteps, batch_size, rank), post_factor, pre_factor

def linear_regression(X, y):
    res = pg.linear_regression(X, y)
    return res

def anova(X, dv):
    res = pg.anova(X, dv)
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

def cluster(x, max_clusters=20):
    silhouettes = []
    for k in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=k).fit(x)
        silhouettes.append(silhouette_score(x, kmeans.labels_, metric='euclidean'))
    argmax_k = np.argmax(silhouettes)
    kmeans = KMeans(n_clusters=argmax_k+1).fit(x)
    return kmeans.labels_

def targeted_dimensionality_reduction(hs, xs, ortho=False):
    n_time_steps, n_trials, n_hidden = hs.shape
    assert(xs.shape[0]==n_trials)
    n_vars = xs.shape[-1]

    # z-score the hidden activity
    hs = zscore(hs.reshape(n_time_steps*n_trials, n_hidden), axis=0).reshape(n_time_steps, n_trials, n_hidden)

    # denoise with PCA
    # U, S, Vh = np.linalg.svd(hs) # n_time_steps*n_trials x n_hidden, n_hidden X n_hidden, n_hidden X n_hidden
    # npca = 0
    # while np.cumsum(S[:npca]**2)/np.sum(S**2) < 0.9:
    #     npca += 1
    
    # D = Vh[:npca].T@Vh[:npca]

    hat = xs@np.linalg.inv((xs@xs.T)) # n_trials X n_feats
    betas = (hs.transpose(0,2,1)@hat.reshape(1, n_trials, n_vars)) # n_time_steps X n_hidden X n_feats
    
    coeff_norms = np.linalg.norm(betas, axis=1) # n_time_steps X n_feats
    coeff_max = betas[np.argmax(coeff_norms, axis=0), :, np.arange(n_vars)] # argmax has size n_feats
    assert coeff_max.shape==(n_hidden, n_vars)

    if ortho:
        Q, R = np.linalg.qr(coeff_max)
        coeff_max = Q

    return betas, coeff_max