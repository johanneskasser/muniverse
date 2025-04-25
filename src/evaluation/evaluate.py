import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags

def match_spikes(s1, s2, shift=0, tol=0.001):
    """
    Match spike times of two neurons given time shift and tolerance.

    Args:
        - s1 (ndarray): Spike times of the first neuron (in seconds)
        - s2 (ndarray): Spike times of the second neuron (in seconds) 
        - shift (float): Temporal delay between the spinking activity (in seconds)
        - tol (float): Common spikes are in a window [spike-tol, spike+tol]

    Returns:
        - overlap (float): Maximum cross-correlation
        - best_shift (float): Delay that maximizes the cross-correlation     

    """

    t1 = np.sort(s1)
    t2 = np.sort(s2 + shift)

    matched_1 = np.zeros(len(t1), dtype=bool)
    matched_2 = np.zeros(len(t2), dtype=bool)

    i, j = 0, 0
    while i < len(t1) and j < len(t2):
        dt = t1[i] - t2[j]
        if abs(dt) <= tol:
            matched_1[i] = True
            matched_2[j] = True
            i += 1
            j += 1
        elif dt < -tol:
            i += 1
        else:
            j += 1

    tp = matched_1.sum()
    fn = (~matched_1).sum()
    fp = (~matched_2).sum()
    return tp, fp, fn

def get_bin_spikes(spike_indices, n_samples):
    '''
    Make binary spike trains given a set of spike indices

    Args:
        spike_indices (ndarray): Array of spike indices (i.e, integers) 
        n_samples (int): Number of time samples

    Returns:
        spike_train (ndarray): Binary spike train vector    

    '''

    spike_train = np.zeros(n_samples, dtype=int)
    spike_train[spike_indices] = 1

    return spike_train

def bin_spikes(spike_times, fsamp = 10000, t_start=0, t_end = 60):
    '''
    Make binary spike trains given a set of spike times

    Args:
        spike_times (ndarray): Array of spike times (in seconds) 
        fsamp (float): Sampling rate in Hz of the binary spike train
        t_start (float) : Start of the time window to be considered (in seconds)
        t_end (float): End of the time window to be considered (in seconds)


    Returns:
        spike_train (ndarray): Binary spike train vector

    '''

    step_size = 1/fsamp
    n_samples = int(np.ceil((t_end - t_start) / step_size)) + 1
    spike_train = np.zeros(n_samples, dtype=int)
    spike_indices = np.round((spike_times - t_start) / step_size).astype(int) #(spike_times / step_size).astype(int)
    spike_indices = spike_indices[(spike_indices >= 0) & (spike_indices < n_samples)]
    spike_train[spike_indices] = 1

    return spike_train

def best_time_shift(spikes1, spikes2, tolerance=0.001, max_shift=0.01, shift_step=0.0005):
    """
    Try multiple time shifts and return the one with maximum TP.
    """
    best_tp = 0
    best_shift = 0.0
    best_fp, best_fn = 0, 0

    shifts = np.arange(-max_shift, max_shift + shift_step, shift_step)
    for shift in shifts:
        tp, fp, fn = match_spikes(spikes1, spikes2, shift, tolerance)
        if tp > best_tp:
            best_tp, best_fp, best_fn = tp, fp, fn
            best_shift = shift

    return best_tp, best_fp, best_fn, best_shift

def max_xcorr(sig1, sig2, max_shift=1000):
    '''
    Align two spike trains by finding the delay maximizing their cross-correlation.

    Arguments:
        - a (ndarray): Reference signal
        - b (ndarray): Another signal 
        - max_shift (int): Maximum delay (in samples) between the two signals

    Outputs:
        - overlap (float): Maximum cross-correlation
        - best_shift (float): Delay that maximizes the cross-correlation     

    '''
    
    corr = correlate(sig1, sig2, mode='full')
    lags = correlation_lags(len(sig1), len(sig2), mode='full')
    mask = (lags >= -max_shift) & (lags <= max_shift)
    corr_win = corr[mask]
    lags_win = lags[mask]
    best_idx = np.argmax(corr_win)
    overlap = corr_win[best_idx]
    best_shift = lags_win[best_idx]

    #corr = np.correlate(sig1, sig2, mode='full')
    #mid = len(sig1) - 1
    #shift_range = range(mid - max_shift, mid + max_shift + 1)
    #best_shift = max(shift_range, key=lambda i: corr[i])
    #overlap = corr[best_shift]
    #best_shift = best_shift - mid
    return overlap, best_shift

def label_sources(df, fsamp=10000, t_start=0, t_end=60, threshold=0.3, max_shift=0.1, tol=0.001):
    """
    Find common sources given a set of sources and spike times

    Args:
        df1 (DataFrame): Data Frame containing spiking neuron activities (columns: 'source_id', 'spike_time')
        fsamp (float): Sampling frequecny (in Hz) of the binary spike trains
        t_start (float) : Start of the time window to be considered (in seconds)
        t_end (float): End of the time window to be considered (in seconds)
        theshold (float) : Common sources need to have a matching score higher than the theshold
        max_shift (float): Maximum delay between two sources (in seconds)
        tol (float): Common spikes need to be in the window [spike-tol, spike+tol]


    Returns:
        labels (ndarray): new labels of the sources
        match_matrix (ndarray): matching scores between all pairs of sources

    """
    
    units = sorted(df['source_id'].unique())
    n_source = len(units)
    labels = np.arange(n_source)
    match_matrix = np.identity(n_source)

    for i in np.arange(n_source):
        spikes_1 = df[df['source_id'] == units[i]]['spike_time'].values
        st1 = bin_spikes(spikes_1, fsamp=fsamp, t_start=t_start, t_end=t_end)
        
        for j in np.arange(i+1, n_source):
            spikes_2 = df[df['source_id'] == units[j]]['spike_time'].values
            st2 = bin_spikes(spikes_2, fsamp=fsamp, t_start=t_start, t_end=t_end)
            _ , shift = max_xcorr(st1, st2, max_shift=int(max_shift*fsamp))
            tp, _, _ = match_spikes(spikes_1, spikes_2, shift=shift/fsamp, tol=tol) 
            denom = max(len(spikes_1), len(spikes_2))
            match_score = tp / denom if denom > 0 else 0

            match_matrix[i,j] = match_score
            match_matrix[j,i] = match_score

            if match_score >= threshold:
                labels[j] = i

    return labels, match_matrix


def evaluate_spike_matches(df1, df2, t_start = 0, t_end = 60, tol=0.001, 
                           max_shift=0.1, fsamp = 10000, threshold=0.3):
    """
    Match spiking sources betwee two data sets.

    Args:
        df1 (DataFrame): Data Frame containing spiking neuron activities (columns: 'source_id', 'spike_time')
        df2 (DataFrame): Data Frame containing spiking neuron activities (columns: 'source_id', 'spike_time')
        t_start (float) : Start of the time window to be considered (in seconds)
        t_end (float): End of the time window to be considered (in seconds)
        tol (float): Common spikes need to be in the window [spike-tol, spike+tol]
        max_shift (float): Maximum delay between two sources (in seconds)
        fsamp (float): Sampling rate (in Hz) of the binary spike train
        theshold (float) : Common sources need to have a matching score higher than the theshold

    Returns:
        results (DataFrame): Table of matched units
        

    """
    source_labels_1 = sorted(df1['source_id'].unique())
    source_labels_2 = sorted(df2['source_id'].unique())
    used_labels = set()
    results = []

    for l1 in source_labels_1:
        spikes_1 = df1[df1['source_id'] == l1]['spike_time'].values
        spike_train_1 = bin_spikes(spikes_1, fsamp=fsamp, t_start=t_start, t_end=t_end)
        best_match = None
        best_score = 0

        for l2 in source_labels_2:
            if l2 in used_labels:
                continue

            spikes_2 = df2[df2['source_id'] == l2]['spike_time'].values
            spike_train_2 = bin_spikes(spikes_2, fsamp=fsamp, t_start=t_start, t_end=t_end)
            _ , shift = max_xcorr(spike_train_1, spike_train_2, max_shift=int(max_shift*fsamp))
            tp, fp, fn = match_spikes(spikes_1, spikes_2, shift=shift/fsamp, tol=tol) 
            denom = max(len(spikes_1), len(spikes_2))
            match_score = tp / denom if denom > 0 else 0

            if match_score > best_score:
                best_score = match_score
                best_match = (l1, l2, tp, fp, fn, shift)

        if best_match and best_score >= threshold:
            l1, l2, tp, fp, fn, shift = best_match
            results.append({
                'source_df1': l1,
                'source_df2': l2,
                'match_score': best_score,
                'delay_seconds': shift/fsamp,
                'common_spikes': tp,
                'only_df2': fp,
                'only_df1': fn
            })
            used_labels.add(l2)

    return pd.DataFrame(results)
