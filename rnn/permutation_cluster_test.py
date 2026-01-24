"""
Standalone simplified version of MNE's permutation_cluster_test.

This is a simplified implementation that performs cluster-based permutation testing
for one-way ANOVA (F-test) without requiring the full MNE package.

Based on the algorithm from:
- Maris & Oostenveld (2007). Nonparametric statistical testing of EEG- and MEG-data.
  Journal of Neuroscience Methods, 164(1), 177-190.

Usage:
    Replace: from mne.stats import permutation_cluster_test
    With: from permutation_cluster_test import permutation_cluster_test
    
    Or: import permutation_cluster_test as mne_stats
        mne_stats.permutation_cluster_test(...)

Requirements:
    - numpy
    - scipy (for f_oneway)

Limitations (compared to full MNE version):
    - Only supports 1D time series (no spatial/topographic clustering)
    - No connectivity matrices
    - No parallel processing (n_jobs ignored)
    - Simplified threshold selection
"""

import numpy as np
from scipy.stats import f_oneway
from scipy.stats import f

def permutation_cluster_test(X, threshold=None, n_permutations=1000, 
                            tail=0, stat_fun=None,
                            seed=None, max_step=1, 
                            exclude=None, t_power=1, out_type='indices',
                            check_disjoint=False, buffer_size=1000, 
                            verbose=None):
    """
    Non-parametric cluster-level test for differences between conditions.
    
    This function performs a cluster-based permutation test for one-way ANOVA
    (F-test) across multiple conditions. It identifies clusters of adjacent
    significant test statistics and assesses their significance through permutation.
    
    Parameters
    ----------
    X : list of array, shape (n_observations, n_times)
        List of arrays, one per condition. Each array has shape
        (n_observations, n_times) where n_observations can vary per condition.
    threshold : float | dict | None
        The threshold value for cluster forming. If None, will use a t-value
        corresponding to p < 0.05 for the given number of observations.
        If dict, should have keys 'start' and 'step' for threshold-free
        cluster enhancement (TFCE). Optional keys: 'E' (default 0.5) and 
        'H' (default 2.0) for TFCE parameters.
    n_permutations : int
        Number of permutations to compute the p-value. Must be >= 0. If 0,
        only the observed statistic is returned.
    tail : int
        If tail is 0, two-sided test. If tail is 1, one-sided test for
        positive values. If tail is -1, one-sided test for negative values.
        Default is 0 (two-sided).
    stat_fun : callable | None
        Function to compute the test statistic. If None, uses F-statistic
        for one-way ANOVA.
    seed : int | None
        Random seed for reproducibility.
    max_step : int
        Maximum number of steps between samples to be considered connected
        (default: 1, meaning only adjacent samples).
    exclude : bool | None
        Whether to exclude masked points (not implemented).
    t_power : float
        Power to raise the test statistic before summing (default: 1).
    out_type : str
        Type of output ('mask' or 'indices', default: 'indices').
    check_disjoint : bool
        Whether to check if clusters are disjoint (default: False).
    buffer_size : int
        Buffer size for permutations (default: 1000).
    verbose : bool | None
        Whether to print progress (not implemented).
    
    Returns
    -------
    T_obs : array, shape (n_times,)
        T-statistic observed for all time points.
    clusters : list
        List of arrays, each containing the indices of points in a cluster.
    cluster_p_values : array
        P-value for each cluster.
    H0 : array, shape (n_permutations,)
        Max cluster statistic for each permutation.
    
    Notes
    -----
    This is a simplified implementation. Some features from the full MNE
    version are not implemented:
    - Multi-dimensional clustering (spatial, time-frequency)
    - Connectivity matrices
    - Parallel processing
    - Advanced masking options
    
    TFCE (Threshold-Free Cluster Enhancement):
    !!!!Not supported yet, need validation!!!!
    When threshold is a dict with 'start' and 'step', TFCE is used instead
    of threshold-based clustering. TFCE integrates cluster statistics over
    all possible thresholds, providing enhanced sensitivity without requiring
    threshold selection.
    
    Examples
    --------
    >>> import numpy as np
    >>> X = [np.random.randn(10, 100), np.random.randn(10, 100), 
    ...      np.random.randn(10, 100)]
    >>> T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
    ...     X, n_permutations=1000)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert to list if single array
    if not isinstance(X, list):
        X = [X]

    # Assert that each array is 2D (observation X time)
    X_flat = []
    for i, x in enumerate(X):
        # Assert that each array has exactly 2 dimensions
        assert x.ndim == 2, f"Array {i} must have 2 dimensions (observation X time), got {x.ndim} dimensions with shape {x.shape}"
        X_flat.append(x)
    
    # Get dimensions
    n_conditions = len(X)
    n_times = X[0].shape[-1]
    
    # Check all arrays have same number of time points
    for i, x in enumerate(X):
        if x.shape[-1] != n_times:
            raise ValueError(f"All conditions must have same number of time points. "
                           f"Condition {i} has {x.shape[-1]}, expected {n_times}.")
    
    # Compute observed statistic
    if stat_fun is None:
        # Use F-statistic for one-way ANOVA
        T_obs = _compute_f_statistic(X_flat)
    else:
        T_obs = stat_fun(X_flat)
    
    # Check if TFCE is requested
    use_tfce = isinstance(threshold, dict)
    
    if use_tfce:
        raise NotImplementedError("TFCE is not implemented yet")
        # # TFCE mode
        # tfce_start = threshold.get('start', 0.0)
        # tfce_step = threshold.get('step', 0.1)
        # tfce_E = threshold.get('E', 0.5)  # Extent parameter
        # tfce_H = threshold.get('H', 2.0)  # Height parameter
        
        # # Compute TFCE for observed data
        # tfce_obs = _compute_tfce(T_obs, tfce_start, tfce_step, tfce_E, tfce_H, 
        #                          tail, max_step)
        
        # # Find clusters based on TFCE values (use a threshold on TFCE)
        # # For TFCE, we typically use a threshold based on the distribution
        # # Use a small threshold to find meaningful clusters
        # if np.max(np.abs(tfce_obs)) > 0:
        #     tfce_threshold = np.max(np.abs(tfce_obs)) * 0.01  # Use 1% of max as threshold
        # else:
        #     tfce_threshold = 0.0
        
        # clusters_obs, cluster_stats_obs = _find_clusters_from_tfce(
        #     tfce_obs, tfce_threshold, tail, max_step
        # )
        
        # # If no clusters found, return empty results
        # if len(clusters_obs) == 0:
        #     return T_obs, [], np.array([]), np.array([])
        
        # # Permutation testing for TFCE
        # if n_permutations == 0:
        #     H0 = np.array([])
        #     cluster_p_values = np.ones(len(clusters_obs))
        # else:
        #     H0 = _permutation_test_tfce(
        #         X_flat, tfce_start, tfce_step, tfce_E, tfce_H, tail, 
        #         n_permutations, stat_fun, max_step, buffer_size
        #     )
            
        #     # Compute p-values based on max TFCE per cluster
        #     cluster_p_values = np.zeros(len(clusters_obs))
        #     for i, cluster in enumerate(clusters_obs):
        #         cluster_tfce_max = np.max(np.abs(tfce_obs[cluster]))
        #         cluster_p_values[i] = np.mean(H0 >= cluster_tfce_max)
    else:
        # Standard threshold-based clustering
        # Set threshold if not provided
        if threshold is None:
            # Use approximate t-value for p < 0.05
            # For F-test, we'll use a simple heuristic
            n_total = sum(x.shape[0] for x in X_flat)
            # Approximate F-critical value (simplified)
            n_groups = len(X_flat)
            n_total = sum(x.shape[0] for x in X_flat)
            df_between = n_groups - 1
            df_within = n_total - n_groups
            alpha = 0.05
            threshold = f.ppf(1 - alpha, df_between, df_within)
        
        # Find clusters in observed data
        clusters_obs, cluster_stats_obs = _find_clusters(
            T_obs, threshold, tail, max_step, t_power
        )
        
        # If no clusters found, return empty results
        if len(clusters_obs) == 0:
            return T_obs, [], np.array([]), np.array([])
        
        # Permutation testing
        if n_permutations == 0:
            H0 = np.array([])
            cluster_p_values = np.ones(len(clusters_obs))
        else:
            H0 = _permutation_test(
                X_flat, threshold, tail, n_permutations, 
                stat_fun, max_step, t_power, buffer_size
            )
            
            # Compute p-values
            cluster_p_values = np.zeros(len(clusters_obs))
            for i, cs in enumerate(cluster_stats_obs):
                # Two-sided p-value
                cluster_p_values[i] = np.mean(np.abs(H0) >= np.abs(cs))
    
    # Convert clusters to requested output type
    if out_type == 'indices':
        clusters = [np.array(c) for c in clusters_obs]
    else:  # 'mask'
        clusters = []
        for c in clusters_obs:
            mask = np.zeros(n_times, dtype=bool)
            mask[c] = True
            clusters.append(mask)
    
    return T_obs, clusters, cluster_p_values, H0


def _compute_f_statistic(X):
    """
    Compute F-statistic for one-way ANOVA across conditions.
    
    Parameters
    ----------
    X : list of array
        List of arrays, each of shape (n_obs, n_times).
    
    Returns
    -------
    F_stats : array, shape (n_times,)
        F-statistic for each time point.
    """
    n_times = X[0].shape[-1]
    F_stats = np.zeros(n_times)
    
    for t in range(n_times):
        # Extract data for this time point across all conditions
        groups = [x[:, t] for x in X]
        # Compute F-statistic
        F, _ = f_oneway(*groups)
        F_stats[t] = F
    
    return F_stats


def _find_clusters(T_obs, threshold, tail, max_step, t_power):
    """
    Find clusters of significant test statistics.
    
    Parameters
    ----------
    T_obs : array, shape (n_times,)
        Test statistics.
    threshold : float
        Threshold for significance.
    tail : int
        Tail direction (0=two-sided, 1=positive, -1=negative).
    max_step : int
        Maximum step for connectivity.
    t_power : float
        Power to raise statistic before summing.
    
    Returns
    -------
    clusters : list of arrays
        List of cluster indices.
    cluster_stats : array
        Cluster statistics (sum of powered statistics).
    """
    n_times = len(T_obs)
    
    # Create mask based on threshold and tail
    if tail == 0:  # two-sided
        mask = np.abs(T_obs) > threshold
    elif tail == 1:  # positive
        mask = T_obs > threshold
    elif tail == -1:  # negative
        mask = T_obs < -threshold
    else:
        raise ValueError(f"tail must be -1, 0, or 1, got {tail}")
    
    if not np.any(mask):
        return [], np.array([])
    
    # Find connected components (clusters) - simple 1D implementation
    # Two points are connected if they're within max_step of each other
    clusters = []
    cluster_stats = []
    
    # Find all significant points
    sig_indices = np.where(mask)[0]
    
    if len(sig_indices) == 0:
        return [], np.array([])
    
    # Group consecutive points (within max_step) into clusters
    current_cluster = [sig_indices[0]]
    
    for i in range(1, len(sig_indices)):
        # Check if current point is within max_step of previous point
        if sig_indices[i] - sig_indices[i-1] <= max_step:
            current_cluster.append(sig_indices[i])
        else:
            # Start new cluster
            clusters.append(np.array(current_cluster))
            current_cluster = [sig_indices[i]]
    
    # Don't forget the last cluster
    if len(current_cluster) > 0:
        clusters.append(np.array(current_cluster))
    
    # Compute cluster statistics
    for cluster in clusters:
        # Compute cluster statistic (sum of powered statistics)
        cluster_stat = np.sum(np.abs(T_obs[cluster]) ** t_power)
        cluster_stats.append(cluster_stat)
    
    cluster_stats = np.array(cluster_stats)
    
    return clusters, cluster_stats


# def _compute_tfce(T_obs, start, step, E, H, tail, max_step):
#     """
#     Compute Threshold-Free Cluster Enhancement (TFCE) values.
    
#     TFCE integrates cluster statistics over all possible thresholds.
#     For each point, TFCE = sum over thresholds of (cluster_size^E * height^H)
#     where height is the threshold value above the starting threshold.
    
#     Parameters
#     ----------
#     T_obs : array, shape (n_times,)
#         Test statistics.
#     start : float
#         Starting threshold value.
#     step : float
#         Step size for threshold increments.
#     E : float
#         Extent parameter (typically 0.5).
#     H : float
#         Height parameter (typically 2.0).
#     tail : int
#         Tail direction (0=two-sided, 1=positive, -1=negative).
#     max_step : int
#         Maximum step for connectivity.
    
#     Returns
#     -------
#     tfce_map : array, shape (n_times,)
#         TFCE value at each time point.
#     """
#     n_times = len(T_obs)
#     tfce_map = np.zeros(n_times)
    
#     # For TFCE, we work with absolute values to simplify computation
#     # This is the standard approach and works for all tail types
#     T_abs = np.abs(T_obs)
#     max_stat = np.max(T_abs)
    
#     # Generate thresholds from start to max_stat
#     if max_stat <= start:
#         # No statistics above threshold, return zeros
#         return tfce_map
    
#     thresholds = np.arange(start + step, max_stat + step, step)
    
#     # Iterate over thresholds
#     for thresh in thresholds:
#         # Find clusters at this threshold (using absolute value for two-sided)
#         # For one-sided tests, we still use abs but the original T_obs will
#         # determine which clusters are found
#         clusters, _ = _find_clusters(T_obs, thresh, tail, max_step, t_power=1)
        
#         # For each cluster, add TFCE contribution
#         for cluster in clusters:
#             cluster_size = len(cluster)
#             height = thresh - start  # Height above starting threshold
            
#             # TFCE contribution: cluster_size^E * height^H
#             # Multiply by step to approximate the integral
#             contribution = (cluster_size ** E) * (height ** H) * step
#             tfce_map[cluster] += contribution
    
#     return tfce_map


# def _find_clusters_from_tfce(tfce_map, threshold, tail, max_step):
#     """
#     Find clusters from TFCE values using a threshold.
    
#     Parameters
#     ----------
#     tfce_map : array, shape (n_times,)
#         TFCE values.
#     threshold : float
#         Threshold for cluster formation.
#     tail : int
#         Tail direction (0=two-sided, 1=positive, -1=negative).
#     max_step : int
#         Maximum step for connectivity.
    
#     Returns
#     -------
#     clusters : list of arrays
#         List of cluster indices.
#     cluster_stats : array
#         Maximum TFCE value in each cluster.
#     """
#     # Create mask based on threshold and tail
#     if tail == 0:  # two-sided
#         mask = np.abs(tfce_map) > threshold
#     elif tail == 1:  # positive
#         mask = tfce_map > threshold
#     elif tail == -1:  # negative
#         mask = tfce_map < -threshold
#     else:
#         raise ValueError(f"tail must be -1, 0, or 1, got {tail}")
    
#     if not np.any(mask):
#         return [], np.array([])
    
#     # Find all significant points
#     sig_indices = np.where(mask)[0]
    
#     if len(sig_indices) == 0:
#         return [], np.array([])
    
#     # Group consecutive points (within max_step) into clusters
#     clusters = []
#     cluster_stats = []
#     current_cluster = [sig_indices[0]]
    
#     for i in range(1, len(sig_indices)):
#         # Check if current point is within max_step of previous point
#         if sig_indices[i] - sig_indices[i-1] <= max_step:
#             current_cluster.append(sig_indices[i])
#         else:
#             # Start new cluster
#             clusters.append(np.array(current_cluster))
#             current_cluster = [sig_indices[i]]
    
#     # Don't forget the last cluster
#     if len(current_cluster) > 0:
#         clusters.append(np.array(current_cluster))
    
#     # Compute cluster statistics (max TFCE in each cluster)
#     for cluster in clusters:
#         cluster_stat = np.max(np.abs(tfce_map[cluster]))
#         cluster_stats.append(cluster_stat)
    
#     cluster_stats = np.array(cluster_stats)
    
#     return clusters, cluster_stats


# def _permutation_test_tfce(X, tfce_start, tfce_step, tfce_E, tfce_H, 
#                            tail, n_permutations, stat_fun, max_step, buffer_size):
#     """
#     Perform permutation testing with TFCE to build null distribution.
    
#     Parameters
#     ----------
#     X : list of array
#         List of condition arrays.
#     tfce_start : float
#         Starting threshold for TFCE.
#     tfce_step : float
#         Step size for TFCE thresholds.
#     tfce_E : float
#         TFCE extent parameter.
#     tfce_H : float
#         TFCE height parameter.
#     tail : int
#         Tail direction.
#     n_permutations : int
#         Number of permutations.
#     stat_fun : callable | None
#         Statistic function.
#     max_step : int
#         Maximum step for connectivity.
#     buffer_size : int
#         Buffer size (not used).
    
#     Returns
#     -------
#     H0 : array, shape (n_permutations,)
#         Maximum TFCE value for each permutation.
#     """
#     n_conditions = len(X)
#     n_times = X[0].shape[-1]
    
#     # Combine all data
#     all_data = np.concatenate([x for x in X], axis=0)  # (n_total, n_times)
#     n_per_condition = [x.shape[0] for x in X]
    
#     H0 = []
    
#     for perm in range(n_permutations):
#         # Randomly permute condition labels
#         perm_indices = np.random.permutation(all_data.shape[0])
        
#         # Split into conditions
#         X_perm = []
#         start_idx = 0
#         for n in n_per_condition:
#             end_idx = start_idx + n
#             X_perm.append(all_data[perm_indices[start_idx:end_idx]])
#             start_idx = end_idx
        
#         # Compute statistic for permuted data
#         if stat_fun is None:
#             T_perm = _compute_f_statistic(X_perm)
#         else:
#             T_perm = stat_fun(X_perm)
        
#         # Compute TFCE for permuted data
#         tfce_perm = _compute_tfce(T_perm, tfce_start, tfce_step, tfce_E, tfce_H,
#                                   tail, max_step)
        
#         # Store maximum TFCE value
#         H0.append(np.max(np.abs(tfce_perm)))
    
#     return np.array(H0)


def _permutation_test(X, threshold, tail, n_permutations, 
                     stat_fun, max_step, t_power, buffer_size):
    """
    Perform permutation testing to build null distribution.
    
    Parameters
    ----------
    X : list of array
        List of condition arrays.
    threshold : float
        Threshold for cluster formation.
    tail : int
        Tail direction.
    n_permutations : int
        Number of permutations.
    stat_fun : callable | None
        Statistic function.
    max_step : int
        Maximum step for connectivity.
    t_power : float
        Power for cluster statistic.
    buffer_size : int
        Buffer size (not used in simplified version).
    
    Returns
    -------
    H0 : array, shape (n_permutations,)
        Maximum cluster statistic for each permutation.
    """
    n_conditions = len(X)
    n_times = X[0].shape[-1]
    
    # Combine all data
    all_data = np.concatenate([x for x in X], axis=0)  # (n_total, n_times)
    n_per_condition = [x.shape[0] for x in X]
    
    H0 = []
    
    for perm in range(n_permutations):
        # Randomly permute condition labels
        perm_indices = np.random.permutation(all_data.shape[0])
        
        # Split into conditions
        X_perm = []
        start_idx = 0
        for n in n_per_condition:
            end_idx = start_idx + n
            X_perm.append(all_data[perm_indices[start_idx:end_idx]])
            start_idx = end_idx
        
        # Compute statistic for permuted data
        if stat_fun is None:
            T_perm = _compute_f_statistic(X_perm)
        else:
            T_perm = stat_fun(X_perm)
        
        # Find clusters
        clusters_perm, cluster_stats_perm = _find_clusters(
            T_perm, threshold, tail, max_step, t_power
        )
        
        # Store maximum cluster statistic (or 0 if no clusters)
        if len(cluster_stats_perm) > 0:
            H0.append(np.max(np.abs(cluster_stats_perm)))
        else:
            H0.append(0.0)
    
    return np.array(H0)


# Example usage and testing
if __name__ == "__main__":
    # Simple test
    np.random.seed(42)
    
    # Create test data: 3 conditions, 10 observations each, 100 time points
    X = [
        np.random.randn(10, 100),
        np.random.randn(10, 100) + 0.5,  # Slight difference
        np.random.randn(10, 100)
    ]
    
    print("Running permutation_cluster_test with threshold...")
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        X, n_permutations=100, verbose=False
    )
    
    print(f"T_obs shape: {T_obs.shape}")
    print(f"Number of clusters: {len(clusters)}")
    print(f"Cluster p-values: {cluster_p_values}")
    print(f"H0 shape: {H0.shape}")
    print("Test completed successfully!")
