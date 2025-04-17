import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

def extension(Y, R):
    """
    Extend a multi-channel signal Y by an extension factor R
    using Toeplitz matrices.

    Parameters:
        Y (ndarray): Original signal (n_channels x n_samples)
        R (int): Extension factor (number of lags)

    Returns:
        eY (ndarray): Extended signal (n_channels * R x n_samples)
    """
    n_channels, n_samples = Y.shape
    eY = np.zeros((n_channels * R, n_samples))

    for i in range(n_channels):
        col = np.concatenate(([Y[i, 0]], np.zeros(R - 1)))
        row = Y[i, :]
        T = toeplitz(col, row)
        eY[i * R:(i + 1) * R, :] = T

    return eY

def whitening(Y, method='ZCA', backend='eig', regularization='auto', eps=1e-10):
    """
    Adaptive whitening function using ZCA, PCA, or Cholesky.

    Parameters:
        Y (ndarray): Input signal (n_channels x n_samples)
        method (str): Whitening method: 'ZCA', 'PCA', 'Cholesky'
        backend (str): 'eig', or 'svd'
        regularization (str or float): 'auto', float value, or None
        eps (float): Small epsilon for numerical stability

    Returns:
        wY (ndarray): Whitened signal
        Z (ndarray): Whitening matrix
    """
    n_channels, n_samples = Y.shape
    use_svd = backend == 'svd'

    if method == 'Cholesky':
        covariance = Y @ Y.T / n_samples
        R = np.linalg.cholesky(covariance)
        Z = np.linalg.inv(R.T)
        wY = Z @ Y
        return wY, Z

    # Use SVD
    if use_svd:
        U, S, Vt = np.linalg.svd(Y, full_matrices=False)
        if regularization == 'auto':
            reg = np.mean(S[len(S)//2:]**2)
        elif isinstance(regularization, float):
            reg = regularization
        else:
            reg = 0
        S_inv = 1. / np.sqrt(S**2 + reg + eps)

        if method == 'ZCA':
            Z = U @ np.diag(S_inv) @ U.T
        elif method == 'PCA':
            Z = np.diag(S_inv) @ U.T
        else:
            raise ValueError("Unknown method.")
        wY = Z @ Y

    # Use EIG
    else:
        covariance = Y @ Y.T / n_samples
        S, V = np.linalg.eigh(covariance)

        if regularization == 'auto':
            reg = np.mean(S[:len(S)//2])
        elif isinstance(regularization, float):
            reg = regularization
        else:
            reg = 0
        S_inv = 1. / np.sqrt(S + reg + eps)

        if method == 'ZCA':
            Z = V @ np.diag(S_inv) @ V.T
        elif method == 'PCA':
            Z = np.diag(S_inv) @ V.T
        else:
            raise ValueError("Unknown method.")
        wY = Z @ Y

    return wY, Z

def est_spike_times(sig, fsamp, cluster = 'kmeans', a = 2):
    """
    Estimate spike indices given a motor unit source signal and compute
    a silhouette-like metric for source quality quantification

    Parameters:
        sig (np.ndarray): Input signal (motor unit source)
        fsamp (float): Sampling rate in Hz
        cluster (string): Clustering method used to identify the spike indices
        a (float): Exponent of assymetric power law 

    Returns:
        est_spikes (np.ndarray): Estimated spike indices
        sil (float): Silhouette-like score (0 = poor, 1 = strong separation)
    """
    sig = np.asarray(sig)

    # Assymetric power law that can be useful for contrast enhancement
    sig = np.sign(sig) * sig**a

    if cluster == 'kmeans':
    
        # Detect peaks with minimum distance of 10 ms
        min_peak_dist = int(round(fsamp * 0.01))
        peaks, _ = find_peaks(sig, distance=min_peak_dist)

        if len(peaks) == 0:
            return np.array([])

        # Get peak values
        peak_vals = sig[peaks].reshape(-1, 1)

        # K-means clustering to separate signal vs. noise
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = kmeans.fit_predict(peak_vals)
        centroids = kmeans.cluster_centers_.flatten()

        # Spikes are those in the cluster with the higher mean
        spike_cluster = np.argmax(centroids)
        est_spikes = peaks[labels == spike_cluster]

        # Compute within- and between-cluster distances
        D = kmeans.transform(peak_vals)  # Distances to both centroids
        sumd = np.sum(D[labels == spike_cluster, spike_cluster])
        between = np.sum(D[labels == spike_cluster, 1 - spike_cluster])
        
        # Silhouette-inspired score
        denom = max(sumd, between)
        sil = (between - sumd) / denom if denom > 0 else 0.0

    return est_spikes, sil

def gram_schmidt(w, B):
    """
    Stabilized Gram-Schmidt orthogonalization.

    Parameters:
        w (np.ndarray): Input vector to be orthogonalized (shape: [n,])
        B (np.ndarray): Matrix of orthogonal basis vectors in columns (shape: [n, k])

    Returns:
        u (np.ndarray): Orthogonalized vector
    """
    w = np.asarray(w, dtype=float)
    B = np.asarray(B, dtype=float)

    # Remove zero columns from B
    non_zero_cols = ~np.all(B == 0, axis=0)
    B = B[:, non_zero_cols]

    u = w.copy()
    for i in range(B.shape[1]):
        a = B[:, i]
        projection = (np.dot(u, a) / np.dot(a, a)) * a
        u = u - projection

    return u