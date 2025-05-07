#!/usr/bin/env python3

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Ensure proper path setup
sys.path.append('.')
if os.path.exists('./src'):
    sys.path.append('./src')

# Import evaluation function if available, otherwise define it here
try:
    from evaluation.evaluate import evaluate_spike_matches
except ImportError:
    # Define a simple spike matching function if the import fails
    def evaluate_spike_matches(df1, df2, t_start, t_end, tol, max_shift, fsamp, threshold):
        """
        Simplified spike matching function that identifies matches between two sets of spikes.
        """
        results = []
        for source1 in df1['source_id'].unique():
            s1_spikes = df1[df1['source_id'] == source1]['spike_time'].values
            
            best_match = None
            best_score = 0
            best_shift = 0
            
            for source2 in df2['source_id'].unique():
                s2_spikes = df2[df2['source_id'] == source2]['spike_time'].values
                
                # Try different time shifts within max_shift range
                shifts = np.linspace(-max_shift, max_shift, 21)  # 21 points for testing shifts
                
                for shift in shifts:
                    # Apply shift to s2_spikes
                    shifted_s2 = s2_spikes + shift
                    
                    # Count matches
                    match_count = 0
                    for s1 in s1_spikes:
                        # Find closest s2 spike
                        if len(shifted_s2) > 0:
                            closest_idx = np.argmin(np.abs(shifted_s2 - s1))
                            if np.abs(shifted_s2[closest_idx] - s1) <= tol:
                                match_count += 1
                    
                    # Calculate score
                    total_spikes = len(s1_spikes) + len(s2_spikes)
                    if total_spikes > 0:
                        score = 2 * match_count / total_spikes
                        
                        if score > best_score:
                            best_score = score
                            best_match = source2
                            best_shift = shift
            
            # If score is above threshold, add to results
            if best_score >= threshold and best_match is not None:
                s2_spikes = df2[df2['source_id'] == best_match]['spike_time'].values
                shifted_s2 = s2_spikes + best_shift
                
                # Count common and exclusive spikes
                common = 0
                for s1 in s1_spikes:
                    if len(shifted_s2) > 0:
                        closest_idx = np.argmin(np.abs(shifted_s2 - s1))
                        if np.abs(shifted_s2[closest_idx] - s1) <= tol:
                            common += 1
                
                only_df1 = len(s1_spikes) - common
                only_df2 = len(s2_spikes) - common
                
                results.append({
                    'source_df1': source1,
                    'source_df2': best_match,
                    'match_score': best_score,
                    'common_spikes': common,
                    'only_df1': only_df1,
                    'only_df2': only_df2,
                    'delay_seconds': best_shift
                })
        
        return pd.DataFrame(results)


def find_original_data(decomp_result_dir, original_data_base):
    """
    Find the corresponding original data folder for a decomposition result folder.
    
    Args:
        decomp_result_dir (Path): Path to decomposition result directory
        original_data_base (Path): Base path to original data directories
        
    Returns:
        Path: Path to matching original data directory, or None if not found
    """
    # Extract run ID, subject ID, and muscle from decomp folder name
    # Example: decomposed_run_20250506_164456_0_PL
    folder_name = decomp_result_dir.name
    parts = folder_name.split('_')
    
    if len(parts) < 5 or not folder_name.startswith('decomposed_run_'):
        print(f"Warning: Unexpected folder name format: {folder_name}")
        return None
    
    # Extract run ID (date and time)
    run_id = "_".join(parts[1:4])  # e.g., run_20250506_164456
    
    # Extract subject ID and muscle
    subject_id = parts[4]  # e.g., 0
    muscle = parts[5]      # e.g., PL
    
    # Find the original data directory
    original_dir = original_data_base / run_id
    
    if not original_dir.exists():
        print(f"Warning: Original data directory not found: {original_dir}")
        return None
    
    # Check if this folder contains the expected files for this subject and muscle
    emg_file = original_dir / f"subject_{subject_id}_{muscle}_emg.npz"
    spikes_file = original_dir / f"subject_{subject_id}_{muscle}_spikes.npz"
    config_file = original_dir / f"subject_{subject_id}_{muscle}_config_used.json"
    
    if not (emg_file.exists() and spikes_file.exists() and config_file.exists()):
        print(f"Warning: Missing required files in {original_dir}")
        return None
        
    return original_dir, subject_id, muscle


def extract_config_parameters(config_file, subject_id, muscle):
    """
    Extract relevant parameters from the configuration JSON file.
    
    Args:
        config_file (Path): Path to the configuration JSON file
        subject_id (str): Subject ID
        muscle (str): Muscle name
        
    Returns:
        dict: Dictionary of extracted parameters
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            
        params = {
            'subject_id': subject_id,
            'muscle': muscle
        }
        
        # Extract Movement Configuration parameters
        movement_config = config.get('MovementConfiguration', {})
        params['movement_type'] = movement_config.get('MovementType', 'Unknown')
        params['movement_dof'] = movement_config.get('MovementDOF', 'Unknown')
        
        # Extract Movement Profile parameters
        movement_profile = movement_config.get('MovementProfileParameters', {})
        params['effort_level'] = movement_profile.get('EffortLevel', 0)
        params['effort_profile'] = movement_profile.get('EffortProfile', 'Unknown')
        params['angle_profile'] = movement_profile.get('AngleProfile', 'Unknown')
        params['target_angle'] = movement_profile.get('TargetAngle', 0)
        params['movement_duration'] = movement_profile.get('MovementDuration', 0)
        params['repetitions'] = movement_profile.get('NRepetitions', 1)
        params['rest_duration'] = movement_profile.get('RestDuration', 0)
        params['ramp_duration'] = movement_profile.get('RampDuration', 0)
        params['hold_duration'] = movement_profile.get('HoldDuration', 0)
        
        # Extract Recording Configuration parameters
        recording_config = config.get('RecordingConfiguration', {})
        params['sampling_frequency'] = recording_config.get('SamplingFrequency', 2048)
        params['noise_level_db'] = recording_config.get('NoiseLeveldb', 0)
        params['noise_seed'] = recording_config.get('NoiseSeed', 0)
        
        # Extract Electrode Configuration
        electrode_config = recording_config.get('ElectrodeConfiguration', {})
        params['inter_electrode_distance'] = electrode_config.get('InterElectrodeDistance', 0)
        params['n_electrodes'] = electrode_config.get('NElectrodes', 0)
        params['n_rows'] = electrode_config.get('NRows', 0)
        params['n_cols'] = electrode_config.get('NCols', 0)
        
        # Extract Filter Properties
        filter_props = recording_config.get('FilterProperties', {})
        params['filter_cutoff'] = filter_props.get('CutoffFrequency', 0)
        params['filter_order'] = filter_props.get('FilterOrder', 0)
        
        # Extract Subject Configuration parameters
        subject_config = config.get('SubjectConfiguration', {})
        params['subject_seed'] = subject_config.get('SubjectSeed', 0)
        params['fibre_density'] = subject_config.get('FibreDensity', 0)
        
        # Get motor unit count for this specific muscle
        muscle_labels = subject_config.get('MuscleLabels', [])
        motor_unit_counts = subject_config.get('MuscleMotorUnitCounts', [])
        
        if muscle in muscle_labels and len(muscle_labels) == len(motor_unit_counts):
            idx = muscle_labels.index(muscle)
            params['motor_unit_count'] = motor_unit_counts[idx]
        else:
            params['motor_unit_count'] = 0
            
        return params
        
    except Exception as e:
        print(f"Error extracting config parameters: {str(e)}")
        return {}


def evaluate_decomposition(decomp_result_dir, original_data_dir, subject_id, muscle, 
                          tol=0.001, max_shift=0.1, threshold=0.3):
    """
    Evaluate decomposition performance by comparing estimated spikes with true spikes.
    
    Args:
        decomp_result_dir (Path): Path to decomposition result directory
        original_data_dir (Path): Path to original data directory
        subject_id (str): Subject ID
        muscle (str): Muscle name
        tol (float): Tolerance for spike matching (in seconds)
        max_shift (float): Maximum time shift allowed when matching spikes (in seconds)
        threshold (float): Minimum match score required to consider a match
        
    Returns:
        dict: Dictionary containing metrics and results
    """
    print(f"Evaluating decomposition for subject {subject_id}, muscle {muscle}")
    
    # Load decomposition results
    decomp_results_file = decomp_result_dir / "decomposition_results.pkl"
    if not decomp_results_file.exists():
        print(f"Warning: Decomposition results file not found: {decomp_results_file}")
        return None
    
    try:
        with open(decomp_results_file, 'rb') as f:
            decomp_results = pickle.load(f)
        
        # Extract spikes from decomposition results
        # Based on the decompose_upperbound function, spikes are directly in the results dict
        estimated_spikes = decomp_results.get('spikes')
        silhouette_scores = decomp_results.get('silhouette', [])
        
        if estimated_spikes is None:
            print("Warning: No estimated spikes found in decomposition results")
            return None
            
    except Exception as e:
        print(f"Error loading decomposition results: {str(e)}")
        return None
    
    # Load true spikes
    spikes_file = original_data_dir / f"subject_{subject_id}_{muscle}_spikes.npz"
    try:
        true_spikes_data = np.load(spikes_file, allow_pickle=True)
        # Try different keys that might contain the spikes
        if 'spikes' in true_spikes_data:
            true_spikes = true_spikes_data['spikes']
        elif 'arr_0' in true_spikes_data:
            true_spikes = true_spikes_data['arr_0']
        else:
            # If no keys match, try to get the first array directly
            true_spikes = true_spikes_data[list(true_spikes_data.keys())[0]]
    except Exception as e:
        print(f"Error loading true spikes: {str(e)}")
        return None
    
    # Load config to get parameters
    config_file = original_data_dir / f"subject_{subject_id}_{muscle}_config_used.json"
    config_params = extract_config_parameters(config_file, subject_id, muscle)
    
    # Get sampling frequency from config parameters
    fsamp = config_params.get('sampling_frequency', 2048)
    
    # Load processing metadata if available
    metadata_file = decomp_result_dir / "processing_metadata.json"
    processing_metadata = None
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                processing_metadata = json.load(f)
        except Exception:
            pass
    
    # Convert true spikes to dictionary format
    true_spikes_dict = {}
    for i, unit_spikes in enumerate(true_spikes):
        if len(unit_spikes) > 0:  # Only include units with spikes
            true_spikes_dict[i] = unit_spikes
    
    # Convert spikes to dataframes for evaluation
    def spikes_to_dataframe(spikes_dict, fsamp):
        data = []
        for source_id, spike_indices in spikes_dict.items():
            for spike_idx in spike_indices:
                # Convert sample index to time in seconds
                spike_time = spike_idx / fsamp
                data.append({'source_id': source_id, 'spike_time': spike_time})
        return pd.DataFrame(data)
    
    true_spikes_df = spikes_to_dataframe(true_spikes_dict, fsamp)
    
    # Ensure estimated_spikes is a dictionary as expected
    if not isinstance(estimated_spikes, dict):
        print(f"Warning: estimated_spikes is not a dictionary: {type(estimated_spikes)}")
        # Try to convert to dictionary if it's a list or array
        if isinstance(estimated_spikes, (list, np.ndarray)):
            estimated_spikes_dict = {}
            for i, spikes in enumerate(estimated_spikes):
                if len(spikes) > 0:
                    estimated_spikes_dict[i] = spikes
            estimated_spikes = estimated_spikes_dict
        else:
            return None
    
    discovered_spikes_df = spikes_to_dataframe(estimated_spikes, fsamp)
    
    print(f"True spikes: {len(true_spikes_df)} spikes across {len(true_spikes_dict)} units")
    print(f"Discovered spikes: {len(discovered_spikes_df)} spikes across {len(estimated_spikes)} units")
    
    # Get total recording duration from EMG data or estimate it
    try:
        emg_file = original_data_dir / f"subject_{subject_id}_{muscle}_emg.npz"
        emg_data = np.load(emg_file)
        # Look for the EMG data in the npz file
        if 'emg' in emg_data:
            emg = emg_data['emg']
        elif 'arr_0' in emg_data:
            emg = emg_data['arr_0']
        else:
            emg = emg_data[list(emg_data.keys())[0]]
        
        total_duration = emg.shape[0] / fsamp if emg.shape[0] > emg.shape[1] else emg.shape[1] / fsamp
    except Exception:
        # If EMG data can't be loaded, estimate duration from spikes or use config
        if config_params.get('movement_duration', 0) > 0:
            total_duration = config_params['movement_duration']
        else:
            total_duration = max([max(spikes) if len(spikes) > 0 else 0 
                                for spikes in true_spikes]) / fsamp
    
    # Run the evaluation
    results = evaluate_spike_matches(
        true_spikes_df, 
        discovered_spikes_df,
        t_start=0,
        t_end=total_duration,
        tol=tol,
        max_shift=max_shift, 
        fsamp=fsamp,
        threshold=threshold
    )
    
    # Calculate summary metrics
    num_matches = len(results)
    avg_score = results['match_score'].mean() if num_matches > 0 else 0
    avg_precision = (results['common_spikes'].sum() / (results['common_spikes'].sum() + results['only_df2'].sum()) 
                     if num_matches > 0 and (results['common_spikes'].sum() + results['only_df2'].sum()) > 0 else 0)
    avg_recall = (results['common_spikes'].sum() / (results['common_spikes'].sum() + results['only_df1'].sum()) 
                 if num_matches > 0 and (results['common_spikes'].sum() + results['only_df1'].sum()) > 0 else 0)
    avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    metrics = {
        'run_id': original_data_dir.name,
        'num_matched_units': num_matches,
        'match_percentage': num_matches / len(true_spikes_dict) * 100 if len(true_spikes_dict) > 0 else 0,
        'avg_match_score': avg_score,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1_score': avg_f1,
        'total_true_spikes': len(true_spikes_df),
        'total_discovered_spikes': len(discovered_spikes_df),
        'avg_spike_rate_true': len(true_spikes_df) / (total_duration * len(true_spikes_dict)) if len(true_spikes_dict) > 0 else 0,
        'avg_spike_rate_disc': len(discovered_spikes_df) / (total_duration * len(estimated_spikes)) if len(estimated_spikes) > 0 else 0,
        'avg_delay_ms': results['delay_seconds'].mean() * 1000 if num_matches > 0 else 0,
        'recording_duration': total_duration,
        'num_true_units': len(true_spikes_dict),
        'num_discovered_units': len(estimated_spikes),
        'avg_silhouette': np.mean(silhouette_scores) if len(silhouette_scores) > 0 else np.nan,
        'min_silhouette': np.min(silhouette_scores) if len(silhouette_scores) > 0 else np.nan,
        'max_silhouette': np.max(silhouette_scores) if len(silhouette_scores) > 0 else np.nan,
    }
    
    # Add config parameters to metrics
    metrics.update(config_params)
    
    # Add processing metadata if available
    if processing_metadata:
        if 'processing_time' in processing_metadata:
            metrics['processing_time'] = processing_metadata['processing_time']
        if 'algorithm_params' in processing_metadata:
            for key, value in processing_metadata['algorithm_params'].items():
                metrics[f'param_{key}'] = value
    
    print("\nDecomposition Performance Metrics:")
    print(f"Matched {num_matches} out of {len(true_spikes_dict)} motor units ({metrics['match_percentage']:.1f}%)")
    print(f"Average match score: {avg_score:.3f}")
    print(f"Precision: {avg_precision:.3f}, Recall: {avg_recall:.3f}, F1 Score: {avg_f1:.3f}")
    print(f"Movement Type: {metrics['movement_type']}, Effort Level: {metrics['effort_level']}%, Effort Profile: {metrics['effort_profile']}")
    
    return metrics


def batch_evaluate_decompositions(decomp_results_dir, original_data_dir, output_csv=None, 
                                 tol=0.001, max_shift=0.1, threshold=0.3):
    """
    Evaluate all decomposition results and generate a summary CSV.
    
    Args:
        decomp_results_dir (str): Directory containing decomposition result folders
        original_data_dir (str): Directory containing original data folders
        output_csv (str): Path to output CSV file (default: results_summary_{timestamp}.csv)
        tol, max_shift, threshold: Parameters for evaluate_spike_matches
        
    Returns:
        pd.DataFrame: DataFrame with evaluation results
    """
    decomp_results_dir = Path(decomp_results_dir)
    original_data_dir = Path(original_data_dir)
    
    if not output_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"results_summary_{timestamp}.csv"
    
    # Find all decomposition result directories
    decomp_dirs = [d for d in decomp_results_dir.iterdir() 
                  if d.is_dir() and d.name.startswith('decomposed_run_')]
    
    print(f"Found {len(decomp_dirs)} decomposition result directories")
    
    all_results = []
    
    # Process each decomposition result directory
    for i, decomp_dir in enumerate(decomp_dirs):
        print(f"\nProcessing {i+1}/{len(decomp_dirs)}: {decomp_dir.name}")
        
        # Find matching original data directory
        match_result = find_original_data(decomp_dir, original_data_dir)
        
        if match_result is None:
            print(f"Skipping {decomp_dir.name}: No matching original data found")
            continue
            
        original_dir, subject_id, muscle = match_result
        
        # Evaluate decomposition
        metrics = evaluate_decomposition(
            decomp_dir, 
            original_dir, 
            subject_id, 
            muscle,
            tol=tol, 
            max_shift=max_shift, 
            threshold=threshold
        )
        
        if metrics:
            all_results.append(metrics)
    
    # Create DataFrame and save to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Calculate overall statistics
        avg_match_percentage = results_df['match_percentage'].mean()
        avg_f1_score = results_df['avg_f1_score'].mean()
        
        print("\nOverall Statistics:")
        print(f"Average match percentage across all runs: {avg_match_percentage:.2f}%")
        print(f"Average F1 score across all runs: {avg_f1_score:.3f}")
        
        # Save to CSV
        results_df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")
        
        # Create summary figures
        create_summary_plots(results_df, output_prefix=output_csv.replace('.csv', ''))
        
        return results_df
    else:
        print("No results to save")
        return None


def create_summary_plots(results_df, output_prefix):
    """
    Create summary plots for the evaluation results.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing evaluation results
        output_prefix (str): Prefix for output plot files
    """
    # Set plot style
    plt.style.use('ggplot')
    sns.set(style="whitegrid")
    
    # Create output directory for plots
    output_dir = Path(f"{output_prefix}_plots")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Match percentage by muscle type
    plt.figure(figsize=(10, 6))
    muscle_groups = results_df.groupby('muscle')['match_percentage'].mean().sort_values(ascending=False)
    
    ax = muscle_groups.plot(kind='bar', color='skyblue')
    plt.title('Average Motor Unit Match Percentage by Muscle', fontsize=14)
    plt.xlabel('Muscle', fontsize=12)
    plt.ylabel('Match Percentage (%)', fontsize=12)
    plt.ylim(0, 100)
    
    # Add value labels
    for i, v in enumerate(muscle_groups):
        ax.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "match_by_muscle.png")
    plt.close()
    
    # 2. F1 score by muscle type
    plt.figure(figsize=(10, 6))
    f1_groups = results_df.groupby('muscle')['avg_f1_score'].mean().sort_values(ascending=False)
    
    ax = f1_groups.plot(kind='bar', color='lightgreen')
    plt.title('Average F1 Score by Muscle', fontsize=14)
    plt.xlabel('Muscle', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.ylim(0, 1)
    
    # Add value labels
    for i, v in enumerate(f1_groups):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "f1_by_muscle.png")
    plt.close()
    
    # 3. Performance by effort level
    if 'effort_level' in results_df.columns:
        plt.figure(figsize=(10, 6))
        
        # Group by effort level
        effort_groups = results_df.groupby('effort_level')['avg_f1_score'].mean().reset_index()
        
        # Create scatter plot with size proportional to count
        counts = results_df.groupby('effort_level').size()
        sizes = [50 * (count / max(counts)) + 20 for count in counts]
        
        plt.scatter(effort_groups['effort_level'], effort_groups['avg_f1_score'], 
                   s=sizes, alpha=0.7, c='purple')
        
        # Add regression line
        if len(effort_groups) >= 3:  # Only add trendline if we have enough points
            x = effort_groups['effort_level']
            y = effort_groups['avg_f1_score']
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), "r--", alpha=0.8)
        
        plt.title('F1 Score vs Effort Level', fontsize=14)
        plt.xlabel('Effort Level (%)', fontsize=12)
        plt.ylabel('Average F1 Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "f1_by_effort_level.png")
        plt.close()
    
    # 4. Performance by effort profile and movement type
    if 'effort_profile' in results_df.columns and 'movement_type' in results_df.columns:
        plt.figure(figsize=(12, 8))
        
        # Create a grouped bar chart
        effort_movement_groups = results_df.groupby(['effort_profile', 'movement_type'])['avg_f1_score'].mean().unstack()
        
        if not effort_movement_groups.empty:
            ax = effort_movement_groups.plot(kind='bar', width=0.7)
            plt.title('F1 Score by Effort Profile and Movement Type', fontsize=14)
            plt.xlabel('Effort Profile', fontsize=12)
            plt.ylabel('Average F1 Score', fontsize=12)
            plt.ylim(0, 1)
            plt.legend(title='Movement Type')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "f1_by_effort_and_movement.png")
            plt.close()
    
    # 5. Silhouette score vs Noise Level
    if 'avg_silhouette' in results_df.columns and 'noise_level_db' in results_df.columns:
        plt.figure(figsize=(10, 6))
        
        # Create a scatter plot
        plt.scatter(results_df['noise_level_db'], results_df['avg_silhouette'],
                   alpha=0.7, c=results_df['avg_f1_score'], cmap='viridis', s=80)
        
        plt.colorbar(label='F1 Score')
        plt.title('Silhouette Score vs Noise Level', fontsize=14)
        plt.xlabel('Noise Level (dB)', fontsize=12)
        plt.ylabel('Average Silhouette Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add muscle labels
        for i, row in results_df.iterrows():
            plt.annotate(row['muscle'], 
                        (row['noise_level_db'], row['avg_silhouette']),
                        xytext=(5, 0), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_dir / "silhouette_vs_noise.png")
        plt.close()
    
    # 6. Match percentage vs Motor Unit Count
    if 'motor_unit_count' in results_df.columns:
        plt.figure(figsize=(10, 6))
        
        # Create a scatter plot
        plt.scatter(results_df['motor_unit_count'], results_df['match_percentage'],
                   alpha=0.7, c=results_df['effort_level'] if 'effort_level' in results_df.columns else 'blue', 
                   cmap='plasma', s=80)
        
        if 'effort_level' in results_df.columns:
            plt.colorbar(label='Effort Level (%)')
            
        plt.title('Match Percentage vs Motor Unit Count', fontsize=14)
        plt.xlabel('Motor Unit Count', fontsize=12)
        plt.ylabel('Match Percentage (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add regression line
        x = results_df['motor_unit_count']
        y = results_df['match_percentage']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(output_dir / "match_vs_mu_count.png")
        plt.close()
    
    # 7. Heatmap of F1 scores by muscle and effort profile
    if 'effort_profile' in results_df.columns:
        plt.figure(figsize=(12, 8))
        
        # Create a pivot table
        heat_data = results_df.pivot_table(
            index='muscle', 
            columns='effort_profile',
            values='avg_f1_score',
            aggfunc='mean'
        )
        
        if not heat_data.empty and heat_data.size > 1:
            # Plot heatmap
            sns.heatmap(heat_data, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt=".3f")
            
            plt.title('F1 Score by Muscle and Effort Profile', fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / "f1_heatmap_muscle_effort.png")
            plt.close()
    
    # 8. Box plot of F1 scores by effort profile
    if 'effort_profile' in results_df.columns:
        plt.figure(figsize=(12, 6))
        
        # Create grouped box plot
        sns.boxplot(x='effort_profile', y='avg_f1_score', data=results_df, palette='Set3')
        
        plt.title('Distribution of F1 Scores by Effort Profile', fontsize=14)
        plt.xlabel('Effort Profile', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "f1_boxplot_by_effort.png")
        plt.close()
    
    # 9. Scatter plot of match percentage vs spike rates
    plt.figure(figsize=(10, 6))
    
    # Create scatter plot
    plt.scatter(results_df['avg_spike_rate_true'], results_df['match_percentage'],
               alpha=0.7, c=results_df['muscle'].astype('category').cat.codes, s=80)
    
    plt.title('Match Percentage vs Average Spike Rate', fontsize=14)
    plt.xlabel('Average True Spike Rate (spikes/s/MU)', fontsize=12)
    plt.ylabel('Match Percentage (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Create a custom legend for muscles
    muscle_codes = {code: muscle for code, muscle in 
                   zip(results_df['muscle'].astype('category').cat.codes.unique(), 
                      results_df['muscle'].unique())}
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=plt.cm.tab10(code % 10), markersize=10,
                             label=muscle) for code, muscle in muscle_codes.items()]
    
    plt.legend(handles=legend_elements, title='Muscle', loc='best')
    
    plt.tight_layout()
    plt.savefig(output_dir / "match_vs_spike_rate.png")
    plt.close()
    
    # 10. Bar chart of average processing times by muscle (if available)
    if 'processing_time' in results_df.columns:
        plt.figure(figsize=(10, 6))
        
        # Group by muscle
        time_groups = results_df.groupby('muscle')['processing_time'].mean().sort_values(ascending=False)
        
        ax = time_groups.plot(kind='bar', color='salmon')
        plt.title('Average Processing Time by Muscle', fontsize=14)
        plt.xlabel('Muscle', fontsize=12)
        plt.ylabel('Processing Time (s)', fontsize=12)
        
        # Add value labels
        for i, v in enumerate(time_groups):
            ax.text(i, v + 0.1, f"{v:.1f}s", ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / "processing_time_by_muscle.png")
        plt.close()
    
    # 11. Combined visualization of critical factors
    plt.figure(figsize=(15, 10))
    
    # Select key parameters
    x = 'effort_level' if 'effort_level' in results_df.columns else 'motor_unit_count'
    y = 'match_percentage'
    size = 'motor_unit_count' if 'motor_unit_count' in results_df.columns else 'num_true_units'
    color = 'muscle'
    
    # Create bubble chart
    scatter = plt.scatter(
        results_df[x], 
        results_df[y],
        s=results_df[size] * 2,  # Scale size for visibility
        c=results_df[color].astype('category').cat.codes,
        alpha=0.7,
        cmap='tab10'
    )
    
    plt.title('Multi-Dimensional Analysis of Decomposition Performance', fontsize=16)
    plt.xlabel(x.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(y.replace('_', ' ').title(), fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Create legend for muscle colors
    legend1 = plt.legend(scatter.legend_elements()[0], 
                       results_df[color].unique(),
                       title="Muscle", loc="upper left")
    plt.gca().add_artist(legend1)
    
    # Create legend for sizes
    size_range = results_df[size].max() - results_df[size].min()
    if size_range > 0:
        handles, labels = [], []
        min_val, max_val = results_df[size].min(), results_df[size].max()
        mid_val = (min_val + max_val) / 2
        
        for val in [min_val, mid_val, max_val]:
            handles.append(plt.scatter([], [], s=val*2, color='gray', alpha=0.7))
            labels.append(f'{int(val)}')
            
        plt.legend(handles, labels, title=size.replace('_', ' ').title(), 
                  loc='upper right', frameon=True, scatterpoints=1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "multi_dimensional_analysis.png")
    plt.close()
    
    # 12. Correlation matrix of key parameters
    plt.figure(figsize=(14, 12))
    
    # Select key metrics for correlation analysis
    key_metrics = [
        'match_percentage', 'avg_f1_score', 'avg_precision', 'avg_recall',
        'avg_silhouette', 'effort_level', 'motor_unit_count', 'noise_level_db',
        'avg_spike_rate_true', 'num_matched_units', 'total_true_spikes'
    ]
    
    # Filter to metrics actually in the DataFrame
    corr_metrics = [m for m in key_metrics if m in results_df.columns]
    
    # Calculate correlation matrix
    corr_matrix = results_df[corr_metrics].corr()
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
    
    plt.title('Correlation Matrix of Key Performance Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrix.png")
    plt.close()

    print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate EMG decomposition results')
    parser.add_argument('--decomp_dir', type=str, required=True,
                       help='Directory containing decomposition result folders')
    parser.add_argument('--original_dir', type=str, required=True,
                       help='Directory containing original data folders')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output CSV file')
    parser.add_argument('--tol', type=float, default=0.001,
                       help='Tolerance for spike matching (seconds)')
    parser.add_argument('--max_shift', type=float, default=0.1,
                       help='Maximum time shift for spike matching (seconds)')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Minimum score threshold for considering a match')
    
    args = parser.parse_args()
    
    # Run the batch evaluation
    results = batch_evaluate_decompositions(
        args.decomp_dir,
        args.original_dir,
        args.output,
        tol=args.tol,
        max_shift=args.max_shift,
        threshold=args.threshold
    )