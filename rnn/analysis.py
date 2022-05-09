import math
from warnings import WarningMessage

import numpy as np
import pingouin as pg
import scipy.cluster.hierarchy as sch
from joblib import Parallel, delayed
from joblib.parallel import delayed
from scipy.optimize import curve_fit, linear_sum_assignment
from scipy.spatial.distance import pdist
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import LinearSVC
from tensorly.decomposition import parafac, non_negative_parafac
import tensorly as tl
from tensorly.tenalg import mode_dot

def run_pca(hs, rank=3):
    trials, timesteps, batch_size, hidden_dim = hs.shape
    pca = PCA(n_components=rank)
    low_x = pca.fit_transform(hs.reshape(trials*timesteps*batch_size, hidden_dim)).reshape(trials, timesteps, batch_size, 3)
    return low_x.reshape(trials, timesteps, batch_size, 3)

def kronecker_mat_ten(matrices, X):
    for k in range(len(matrices)):
        M = matrices[k]
        Y = mode_dot(X, M, k)
        X = Y
        X = tl.moveaxis(X, [0, 1, 2], [2, 1, 0])
    return Y

def corcondia(X, k):
    # https://gist.github.com/willshiao/2c0d7cc1133d8fa31587e541fef480fb

    rank_X = len(X)
    Us = [];
    Ss = [];
    Vs = [];

    for F in X:
        U, S, V = np.linalg.svd(F, )
        Us.append(U.T)
        Ss.append(S)
        Vs.append(V.T)

    for i, S in enumerate(Ss):
        Ss[i] = np.diag(1/S)

    part1 = kronecker_mat_ten(Us, X)
    part2 = kronecker_mat_ten(Ss, part1)
    G = kronecker_mat_ten(Vs, part2)

    T = np.zeros(tuple([k]*rank_X))
    T[np.diag_indices(k, rank_X)]=1

    return (1 - ((G-T)**2).sum() / float(k))

def run_tca(xs, ranks=[1, 27], num_reps=5):
    results = {}
    for r in range(ranks[0], ranks[1]+1):
        print(f'fitting model of rank {r}')
        results[r] = []
        for n in range(num_reps):
            cp, errs = non_negative_parafac(xs, rank=r, return_errors=True)
            results[r].append({'cp_tensor': cp, 'errors': errs})
    
    print('sorting errors')
    for r in range(ranks[0], ranks[1]+1):
        idx = np.argsort([rr['errors'][-1] for rr in results[r]])
        results[r] = [results[r][i] for i in idx]

    print('adding corcondia')
    # For each rank, align everything to the best model
    for r in range(ranks[0], ranks[1]+1):
        # align lesser fit models to best models
        for i, res in enumerate(results[r]):
            results[r][i]['corcondia'] = corcondia(res['cp_tensor'].factors, k=r)
    
    return results

def linear_regression(X, y):
    res = LinearRegression().fit(X, y)
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

def targeted_dimensionality_reduction(hs, xs, do_zscore=False, denoise=False, ortho=False, n_jobs=1):
    n_time_steps, n_trials, n_hidden = hs.shape
    assert(xs.shape[0]==n_trials)
    n_vars = xs.shape[-1]

    # z-score the hidden activity
    if do_zscore:
        hs = zscore(hs.reshape(n_time_steps*n_trials, n_hidden), axis=0).reshape(n_time_steps, n_trials, n_hidden)

    # denoise with PCA
    if denoise:
        U, S, Vh = np.linalg.svd(hs) # n_time_steps*n_trials x n_hidden, n_hidden X n_hidden, n_hidden X n_hidden
        npca = 0
        while np.cumsum(S[:npca]**2)/np.sum(S**2) < 0.95:
            npca += 1
        D = Vh[:npca].T@Vh[:npca]

    hs_flat = hs.reshape(n_time_steps*n_trials, n_hidden)
    xs_flat = np.repeat(xs, n_time_steps, axis=0)

    lr = LinearRegression(fit_intercept=not do_zscore, n_jobs=n_jobs).fit(xs_flat, hs_flat)
    betas = lr.coef_.reshape(n_time_steps, n_trials, n_hidden)
    
    if ortho:
        raise NotImplementedError('Not using orthogonal functionality')
        coeff_norms = np.linalg.norm(betas, axis=1) # n_time_steps X n_feats
        coeff_max = betas[np.argmax(coeff_norms, axis=0), :, np.arange(n_vars)] # argmax has size n_feats
        assert coeff_max.shape==(n_hidden, n_vars)
        Q, R = np.linalg.qr(coeff_max)
        coeff_max = Q
    
    return lr, betas

def get_CPD(hs, xs, full_model, channel_groups):
    n_time_steps, n_trials, n_hidden = hs.shape
    assert(xs.shape[0]==n_trials)
    n_vars = xs.shape[1]

    hs_flat = hs.reshape(n_time_steps*n_trials, n_hidden)
    xs_flat = np.repeat(xs, n_time_steps, axis=0)

    sse = np.sum((full_model.predict(xs_flat) - hs_flat)**2, axis=0)

    assert(np.sum(channel_groups)==n_vars)
    channel_groups = np.cumsum(channel_groups)
    channel_groups = np.insert(channel_groups, 0, 0)
    # manually calculate SSE for categorical IVs

    cpd = np.empty(n_time_steps*n_trials, len(channel_groups))

    for i in range(1, len(channel_groups)):
        X_i = xs_flat.copy()
        X_i[:,channel_groups[i-1]:channel_groups[i]] = 0 # remove one factor
        sse_X_i = np.sum((full_model.predict(X_i) - hs_flat)**2, axis=0)
        cpd[:,i]=(sse_X_i-sse)/sse_X_i

    cpd = cpd.reshape(n_time_steps, n_trials, len(channel_groups))

    return cpd