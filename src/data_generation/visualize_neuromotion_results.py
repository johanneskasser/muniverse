#!/usr/bin/env python3
"""
Visualization utilities for NeuroMotion EMG simulation results.
This script creates diagnostic plots to verify the correctness of the generated signals.

Usage:
    python visualize_neuromotion_results.py <data_directory>

The data directory should contain the output files from run_neuromotion.py,
including emg.npy, spikes.npy, effort_profile.npy, etc.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse


def load_data(data_dir):
    """Load all relevant data files from the specified directory."""
    files = {
        'emg': os.path.join(data_dir, 'emg.npy'),
        'spikes': os.path.join(data_dir, 'spikes.npy'),
        'effort': os.path.join(data_dir, 'effort_profile.npy'),
        'muaps': os.path.join(data_dir, 'muaps.npy'),
        'metadata': os.path.join(data_dir, 'metadata.json'),
        'config': os.path.join(data_dir, 'config_used.json')
    }
    
    data = {}
    for key, file_path in files.items():
        if os.path.exists(file_path):
            if file_path.endswith('.npy'):
                try:
                    # First try loading with default settings
                    data[key] = np.load(file_path)
                except ValueError:
                    # If that fails, try with allow_pickle=True
                    data[key] = np.load(file_path, allow_pickle=True)
                print(f"Loaded {key} from {file_path}")
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data[key] = json.load(f)
                print(f"Loaded {key} from {file_path}")
        else:
            print(f"Warning: {file_path} not found")
    
    return data


def is_sparse_spike_format(spikes):
    """Check if spikes are in sparse format (lists of indices) or dense format (binary arrays)."""
    if not isinstance(spikes, (list, np.ndarray)):
        return False
    
    # Check the first few elements
    for i in range(min(5, len(spikes))):
        if isinstance(spikes[i], list) and all(isinstance(x, (int, np.integer)) for x in spikes[i][:10]):
            return True
    
    return False


def plot_overview(data, output_dir):
    """
    Create an overview plot showing EMG signals, effort profile, and active motor units.
    
    Args:
        data (dict): Dictionary containing the loaded data
        output_dir (str): Directory to save the plots
    """
    if 'emg' not in data or 'effort' not in data or 'spikes' not in data:
        print("Error: Missing required data for overview plot")
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(4, 3, figure=fig)
    
    # Get metadata for plotting
    fs = 2048  # Default sampling frequency
    if 'metadata' in data and 'simulation_info' in data['metadata']:
        fs = data['metadata']['simulation_info'].get('fs', 2048)
    
    # Time vectors
    time_emg = np.arange(data['emg'].shape[2]) / fs
    time_effort = np.arange(len(data['effort'])) / fs
    
    # Plot EMG signals from multiple channels
    ax1 = fig.add_subplot(gs[0, :])
    emg = data['emg']
    n_rows, n_cols = emg.shape[0], emg.shape[1]
    center_row, center_col = n_rows // 2, n_cols // 2
    
    # Plot a few channels from the center of the array
    channels_to_plot = [
        (center_row, center_col),
        (center_row - 3, center_col),
        (center_row + 3, center_col),
        (center_row, center_col - 2),
        (center_row, center_col + 2)
    ]
    
    for i, (row, col) in enumerate(channels_to_plot):
        if 0 <= row < n_rows and 0 <= col < n_cols:
            ax1.plot(time_emg, emg[row, col, :] + i*0.5, label=f"Channel ({row},{col})")
    
    ax1.set_title("EMG Signals from Selected Channels")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot effort profile
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(time_effort, data['effort'], 'r-', linewidth=2)
    ax2.set_title("Effort Profile")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Effort Level")
    ax2.grid(True, alpha=0.3)
    
    # Check if spikes are in sparse format (lists of spike times) or dense format (binary arrays)
    sparse_format = is_sparse_spike_format(data['spikes'])
    spikes = data['spikes']
    num_mus = len(spikes)
    
    # Plot number of active motor units over time
    ax3 = fig.add_subplot(gs[2, :])
    
    # Calculate active MUs using appropriate method for the spike format
    window_size = int(fs * 0.1)  # 100ms window
    if sparse_format:
        # For sparse format (lists of spike indices)
        active_mus = np.zeros(len(data['effort']) - window_size + 1)
        for t in range(len(active_mus)):
            active_count = 0
            window_start = t
            window_end = t + window_size
            
            for mu_idx in range(num_mus):
                # Check if any spike falls within this window
                if any(window_start <= spike < window_end for spike in spikes[mu_idx] if spike < len(data['effort'])):
                    active_count += 1
            
            active_mus[t] = active_count
    else:
        # For dense format (binary arrays)
        active_mus = np.zeros(len(data['effort']) - window_size + 1)
        for j in range(len(active_mus)):
            active_count = 0
            for i in range(num_mus):
                if np.sum(spikes[i][j:j+window_size]) > 0:
                    active_count += 1
            active_mus[j] = active_count
    
    time_active = np.arange(len(active_mus)) / fs
    ax3.plot(time_active, active_mus, 'b-', linewidth=2)
    ax3.set_title("Number of Active Motor Units")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Active MUs")
    ax3.set_ylim(0, num_mus * 1.1)  # Give some headroom
    ax3.grid(True, alpha=0.3)
    
    # Plot spike trains (raster plot)
    ax4 = fig.add_subplot(gs[3, :])
    
    # Plot subset of spike trains as raster plot
    step = max(1, num_mus // 20)  # Show at most ~20 spike trains
    for i, mu_idx in enumerate(range(0, num_mus, step)):
        if mu_idx < num_mus:
            if sparse_format:
                # For sparse format, spike times are already available
                spike_times = np.array(spikes[mu_idx]) / fs
                valid_spikes = spike_times[spike_times < max(time_emg)]
                if len(valid_spikes) > 0:
                    ax4.plot(valid_spikes, np.ones_like(valid_spikes) * i, '|', markersize=3)
            else:
                # For dense format, convert to spike times
                spike_times = np.where(spikes[mu_idx])[0] / fs
                ax4.plot(spike_times, np.ones_like(spike_times) * i, '|', markersize=3)
    
    ax4.set_title("Spike Raster Plot (subset of MUs)")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Motor Unit Index")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'overview_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved overview plot to {output_path}")
    plt.close(fig)


def plot_emg_heatmap(data, output_dir):
    """
    Create a heatmap of the EMG signal across the electrode array.
    
    Args:
        data (dict): Dictionary containing the loaded data
        output_dir (str): Directory to save the plots
    """
    if 'emg' not in data:
        print("Error: EMG data missing for heatmap plot")
        return
    
    emg = data['emg']
    n_rows, n_cols, n_samples = emg.shape
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Get interesting time points (near peak effort)
    if 'effort' in data:
        peak_idx = np.argmax(data['effort'])
    else:
        peak_idx = n_samples // 2
    
    time_points = [
        peak_idx - 1000,  # Before peak
        peak_idx,         # At peak
        peak_idx + 1000,  # After peak
        n_samples - 2000  # Near end
    ]
    
    # Ensure time points are valid
    time_points = [max(0, min(t, n_samples-1)) for t in time_points]
    
    # Compute RMS of EMG in windows around each time point
    window_size = 500  # samples
    heatmaps = []
    
    for t in time_points:
        start = max(0, t - window_size//2)
        end = min(n_samples, t + window_size//2)
        window_data = emg[:, :, start:end]
        rms = np.sqrt(np.mean(window_data**2, axis=2))
        heatmaps.append(rms)
    
    titles = ["Before Peak", "At Peak", "After Peak", "Near End"]
    
    # Plot heatmaps
    for i, (ax, heatmap, title) in enumerate(zip(axs.flat, heatmaps, titles)):
        im = ax.imshow(heatmap, cmap='jet', interpolation='nearest', aspect='auto')
        ax.set_title(f"EMG RMS Heatmap - {title}")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        fig.colorbar(im, ax=ax, label="RMS Amplitude")
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'emg_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved EMG heatmap to {output_path}")
    plt.close(fig)


def plot_muap_examples(data, output_dir):
    """
    Plot examples of MUAPs for different motor units.
    
    Args:
        data (dict): Dictionary containing the loaded data
        output_dir (str): Directory to save the plots
    """
    if 'muaps' not in data:
        print("Error: MUAP data missing for MUAP example plot")
        return
    
    muaps = data['muaps']
    num_mus, steps, n_rows, n_cols, time_length = muaps.shape
    
    # Select a few MUs to plot
    mus_to_plot = [
        0,              # First MU (typically smallest)
        num_mus // 4,
        num_mus // 2,
        3 * num_mus // 4,
        num_mus - 1     # Last MU (typically largest)
    ]
    
    # Select center of electrode array for visualization
    center_row, center_col = n_rows // 2, n_cols // 2
    
    # Create grid of subplots
    fig, axs = plt.subplots(len(mus_to_plot), 3, figsize=(15, 12))
    
    for i, mu_idx in enumerate(mus_to_plot):
        if mu_idx < num_mus:
            # Plot MUAP for different positions (steps): beginning, middle, end
            step_indices = [0, steps//2, steps-1]
            
            for j, step_idx in enumerate(step_indices):
                ax = axs[i, j]
                muap = muaps[mu_idx, step_idx, center_row, center_col, :]
                ax.plot(muap)
                
                if j == 0:
                    ax.set_ylabel(f"MU {mu_idx}")
                
                if i == 0:
                    if j == 0:
                        ax.set_title("Beginning Position")
                    elif j == 1:
                        ax.set_title("Middle Position")
                    else:
                        ax.set_title("End Position")
                
                if i == len(mus_to_plot) - 1:
                    ax.set_xlabel("Samples")
    
    plt.suptitle("Example MUAPs at Different Joint Positions (Center Electrode)", fontsize=16)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'muap_examples.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved MUAP examples to {output_path}")
    plt.close(fig)


def plot_frequency_analysis(data, output_dir):
    """
    Perform frequency domain analysis of the EMG signals.
    
    Args:
        data (dict): Dictionary containing the loaded data
        output_dir (str): Directory to save the plots
    """
    if 'emg' not in data:
        print("Error: EMG data missing for frequency analysis")
        return
    
    emg = data['emg']
    n_rows, n_cols, n_samples = emg.shape
    
    # Get sampling frequency
    fs = 2048  # Default sampling frequency
    if 'metadata' in data and 'simulation_info' in data['metadata']:
        fs = data['metadata']['simulation_info'].get('fs', 2048)
    
    # Select center and peripheral channels
    center_row, center_col = n_rows // 2, n_cols // 2
    
    # Create figure
    fig = plt.figure(figsize=(15, 8))
    
    # Top plot: PSD in log scale
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    
    # Compute PSD for a few channels
    channels = [
        (center_row, center_col),
        (center_row - 2, center_col),
        (center_row + 2, center_col),
        (center_row, center_col - 2),
        (center_row, center_col + 2)
    ]
    
    # Compute FFT and PSD
    for row, col in channels:
        if 0 <= row < n_rows and 0 <= col < n_cols:
            # Get the signal
            signal = emg[row, col, :]
            
            # Compute the FFT
            fft = np.fft.rfft(signal)
            freq = np.fft.rfftfreq(n_samples, d=1/fs)
            
            # Compute the power spectrum
            psd = np.abs(fft)**2 / n_samples
            
            # Plot the PSD (up to 1000 Hz or Nyquist, whichever is smaller)
            max_freq_idx = np.searchsorted(freq, min(1000, fs/2))
            
            # Log scale plot
            ax1.semilogy(freq[:max_freq_idx], psd[:max_freq_idx], label=f"Channel ({row},{col})")
    
    ax1.set_title("Power Spectral Density (Log Scale)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Power/Frequency (dB/Hz)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Bottom-left: Alternative to spectrogram: Time-frequency representation using short windows
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    signal = emg[center_row, center_col, :]
    
    # Compute PSD for different time segments
    segment_length = 4096  # Length of each segment
    hop_length = segment_length // 4  # 75% overlap
    
    num_segments = (len(signal) - segment_length) // hop_length + 1
    if num_segments > 0:
        time_segments = np.zeros((num_segments, max_freq_idx))
        segment_times = np.zeros(num_segments)
        
        for i in range(num_segments):
            start_idx = i * hop_length
            end_idx = start_idx + segment_length
            
            if end_idx <= len(signal):
                segment = signal[start_idx:end_idx]
                segment_fft = np.fft.rfft(segment)
                segment_psd = np.abs(segment_fft)**2 / len(segment)
                
                segment_freq = np.fft.rfftfreq(len(segment), d=1/fs)
                segment_max_freq_idx = np.searchsorted(segment_freq, min(500, fs/2))
                
                if segment_max_freq_idx <= max_freq_idx:
                    time_segments[i, :segment_max_freq_idx] = segment_psd[:segment_max_freq_idx]
                segment_times[i] = (start_idx + segment_length/2) / fs
        
        # Plot as a color mesh
        extent = [segment_times[0], segment_times[-1], 0, 500]
        im = ax2.imshow(time_segments.T, aspect='auto', origin='lower', 
                       extent=extent, cmap='viridis', interpolation='nearest')
        fig.colorbar(im, ax=ax2, label='Power')
    
    ax2.set_title(f"Time-Frequency Analysis (Center Channel)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    
    # Bottom-right: Cumulative PSD (how much power is below each frequency)
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    
    for row, col in channels[:3]:  # Use fewer channels for clarity
        if 0 <= row < n_rows and 0 <= col < n_cols:
            signal = emg[row, col, :]
            fft = np.fft.rfft(signal)
            freq = np.fft.rfftfreq(n_samples, d=1/fs)
            psd = np.abs(fft)**2 / n_samples
            
            # Calculate cumulative PSD
            max_freq_idx = np.searchsorted(freq, 500)  # Only up to 500 Hz
            cumulative_psd = np.cumsum(psd[:max_freq_idx]) / np.sum(psd[:max_freq_idx])
            
            ax3.plot(freq[:max_freq_idx], cumulative_psd, label=f"Channel ({row},{col})")
    
    # Add horizontal lines at 50% and 95% power
    ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label="50% Power")
    ax3.axhline(y=0.95, color='g', linestyle='--', alpha=0.7, label="95% Power")
    
    ax3.set_title("Cumulative Power Distribution")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Cumulative Power (%)")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    ax3.legend()
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'frequency_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved frequency analysis to {output_path}")
    plt.close(fig)


def plot_mu_recruitment(data, output_dir):
    """
    Plot motor unit recruitment patterns against effort level.
    
    Args:
        data (dict): Dictionary containing the loaded data
        output_dir (str): Directory to save the plots
    """
    if 'spikes' not in data or 'effort' not in data:
        print("Error: Missing spike or effort data for recruitment plot")
        return
    
    spikes = data['spikes']
    effort = data['effort']
    
    # Get sampling frequency
    fs = 2048  # Default sampling frequency
    if 'metadata' in data and 'simulation_info' in data['metadata']:
        fs = data['metadata']['simulation_info'].get('fs', 2048)
    
    # Check if spikes are in sparse format (lists of spike times) or dense format (binary arrays)
    sparse_format = is_sparse_spike_format(spikes)
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [1, 1, 2]})
    
    # Plot 1: Effort profile
    time_effort = np.arange(len(effort)) / fs
    axs[0].plot(time_effort, effort, 'r-', linewidth=2)
    axs[0].set_title("Effort Profile")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Effort Level")
    axs[0].grid(True, alpha=0.3)
    
    # Plot 2: Number of active motor units over time
    num_mus = len(spikes)
    window_size = int(fs * 0.1)  # 100ms window
    
    if sparse_format:
        # For sparse format (lists of spike indices)
        active_mus = np.zeros(len(effort) - window_size + 1)
        for t in range(len(active_mus)):
            active_count = 0
            window_start = t
            window_end = t + window_size
            
            for mu_idx in range(num_mus):
                # Check if any spike falls within this window
                if any(window_start <= spike < window_end for spike in spikes[mu_idx] if spike < len(effort)):
                    active_count += 1
            
            active_mus[t] = active_count
    else:
        # For dense format (binary arrays)
        active_mus = np.zeros(len(effort) - window_size + 1)
        for j in range(len(effort) - window_size + 1):
            active_count = 0
            for i in range(num_mus):
                if np.sum(spikes[i][j:j+window_size]) > 0:
                    active_count += 1
            active_mus[j] = active_count
    
    time_active = np.arange(len(active_mus)) / fs
    axs[1].plot(time_active, active_mus, 'b-', linewidth=2)
    axs[1].set_title("Number of Active Motor Units")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Active MUs")
    axs[1].set_ylim(0, num_mus * 1.1)  # Give some headroom
    axs[1].grid(True, alpha=0.3)
    
    # Plot 3: Detailed recruitment onset visualization
    # Calculate first spike time for each MU
    first_spike_times = np.full(num_mus, np.nan)
    
    if sparse_format:
        # For sparse format (lists of spike indices)
        for i in range(num_mus):
            if len(spikes[i]) > 0:
                first_spike_times[i] = spikes[i][0] / fs
    else:
        # For dense format (binary arrays)
        for i in range(num_mus):
            spike_indices = np.where(spikes[i])[0]
            if len(spike_indices) > 0:
                first_spike_times[i] = spike_indices[0] / fs
    
    # Sort MUs by recruitment time
    recruitment_order = np.argsort(first_spike_times)
    sorted_times = first_spike_times[recruitment_order]
    
    # Filter out NaNs (MUs that never fired)
    valid_indices = ~np.isnan(sorted_times)
    sorted_times = sorted_times[valid_indices]
    sorted_mus = recruitment_order[valid_indices]
    
    # Plot recruitment times as scatter plot
    axs[2].scatter(sorted_times, np.arange(len(sorted_times)), marker='|', s=100, color='b')
    
    # Add effort profile as background reference
    ax_effort = axs[2].twinx()
    ax_effort.plot(time_effort, effort, 'r-', alpha=0.3)
    ax_effort.set_ylabel("Effort Level", color='r')
    ax_effort.tick_params(axis='y', labelcolor='r')
    
    axs[2].set_title("Motor Unit Recruitment Order")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("MU Index (sorted by recruitment time)")
    axs[2].grid(True, alpha=0.3)
    
    # Create additional figure for recruitment threshold analysis
    fig2, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate recruitment threshold (effort level at first spike)
    recruitment_thresholds = np.full(num_mus, np.nan)
    
    if sparse_format:
        # For sparse format (lists of spike indices)
        for i in range(num_mus):
            if len(spikes[i]) > 0:
                first_spike_idx = spikes[i][0]
                if first_spike_idx < len(effort):
                    recruitment_thresholds[i] = effort[first_spike_idx]
    else:
        # For dense format (binary arrays)
        for i in range(num_mus):
            spike_indices = np.where(spikes[i])[0]
            if len(spike_indices) > 0:
                first_spike_idx = spike_indices[0]
                if first_spike_idx < len(effort):
                    recruitment_thresholds[i] = effort[first_spike_idx]
    
    # Filter out NaNs
    valid_indices = ~np.isnan(recruitment_thresholds)
    thresholds = recruitment_thresholds[valid_indices]
    mu_indices = np.arange(num_mus)[valid_indices]
    
    # Plot recruitment thresholds
    ax.plot(mu_indices, thresholds, 'o-')
    ax.set_title("Motor Unit Recruitment Thresholds")
    ax.set_xlabel("Motor Unit Index")
    ax.set_ylabel("Effort Level at Recruitment")
    ax.grid(True)
    
    plt.tight_layout()
    
    # Save plots
    output_path1 = os.path.join(output_dir, 'mu_recruitment.png')
    fig.savefig(output_path1, dpi=300, bbox_inches='tight')
    
    output_path2 = os.path.join(output_dir, 'recruitment_thresholds.png')
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
    
    print(f"Saved MU recruitment plots to {output_path1} and {output_path2}")
    plt.close(fig)
    plt.close(fig2)


def plot_all(data_dir):
    """Generate all plots for a given data directory."""
    # Create plots directory
    plots_dir = os.path.join(data_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load data
    data = load_data(data_dir)
    
    # Generate all plots
    plot_overview(data, plots_dir)
    plot_emg_heatmap(data, plots_dir)
    plot_muap_examples(data, plots_dir)
    plot_frequency_analysis(data, plots_dir)
    plot_mu_recruitment(data, plots_dir)
    
    print(f"All plots saved to {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize NeuroMotion EMG simulation results")
    parser.add_argument("data_dir", help="Directory containing the simulation results")
    args = parser.parse_args()
    
    plot_all(args.data_dir)