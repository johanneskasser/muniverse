# Imports
# -------
import numpy as np
import scipy
from scipy import signal
import os
from copy import copy
import argparse

from NeuroMotion.MNPoollib.MNPool import MotoneuronPool
from NeuroMotion.MNPoollib.mn_params import mn_default_settings
from NeuroMotion.MNPoollib.mn_utils import ensure_spikes_in_range, spikes_to_bin, generate_emg
from NeuroMotion.loaders import load_gen_data, save_sim_emg

# Functions
# ---------
def compute_rms(emg, timestamps, win_len_s, win_step_s, fs=2048):

    # Initialise variables
    win_len = np.round( win_len_s * fs ).astype(int)
    win_step = np.round( win_step_s * fs ).astype(int)

    samples, chs = emg.shape
    win_num = np.round( (samples - win_len)/win_step ).astype(int) +1 

    timestamps_aux = np.linspace( timestamps[0], timestamps[-1], win_num)
    rms_aux = np.zeros((win_num, chs))

    # Compute RMS
    for win in range(win_num):
        mask = np.arange( 0 + win_step * win, np.amin( [win_len + win_step * win, samples] ) ).astype(int)
        rms_aux[win] = np.sqrt( np.mean( emg[mask]**2, axis=0) )
    
    # Interpolate RMS to match EMG signals
    rms = np.zeros_like(emg)
    for ch in range(chs):
        rms[:, ch] = scipy.interpolate.Akima1DInterpolator( timestamps_aux, rms_aux[:, ch] )( timestamps ) 

    return rms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate EMG signals from movements')
    
    parser.add_argument('--num_mus', default=100, type=int, help='number of motor units')
    parser.add_argument('--muscle', default='FDSI', type=str, help='ECRB|ECRL|PL|FCU|ECU|EDCI|FDSI')
    parser.add_argument('--mov', default='flx_ext', type=str, help='flx_ext | rad_uln')
    parser.add_argument('--mode', default='sample', type=str, help='sample or morph')
    parser.add_argument('--sampling', default='default', type=str, help='distribution of motor unit properties')
    parser.add_argument('--iter', default=0, type=int, help='bootstrapping iteration')
    parser.add_argument('--angle', default=0, type=int, help='wrist angle')
    parser.add_argument('--dur_s', default=20, type=float, help='contraction duration in seconds')
    parser.add_argument('--mvc_level', default=30, type=int, help='maximum voluntary contraction level')
    parser.add_argument('--snr_level', default=30, type=int, help='signal-to-noise ratio in dB')
    parser.add_argument('--path_save', default='./', type=str, help='path to save the simulated contraction')
    args = parser.parse_args()

    args.iter -= 1 # Fix for JOB_ARRAY_INDEX (always > 0)
    np.random.seed(args.iter)

    # Load data
    # ---------
    data_path = os.path.join(
        os.environ['HOME'], 
        'NeuroMotion', 'res',
        f'{args.muscle}_{args.mov}_{args.num_mus}mu_bs{args.iter}',
        f'sim_muaps_{args.sampling}_{args.muscle}_{args.num_mus}_{args.mov}_{args.mode}.hdf5'
    )
    sim = load_gen_data(data_path)
    print(f"{args.muscle}: {sim['muaps'].shape}")

    units, morphs, ch_rows, ch_cols, win = sim['muaps'].shape
    chs =  ch_rows * ch_cols
    ch_map = np.arange(chs).astype(int).reshape(ch_rows, ch_cols)
    fs = sim['neural_params']['fs_spikes']
    print('Generated data loaded correctly')

    # Generate MUAP angle labels
    # --------------------------
    angles = {
        'flex': -65,
        'default': 0,
        'ext': 65,
    }
    muap_angle_labels = np.linspace(angles['flex'], angles['ext'], sim['muaps'].shape[1]).astype(int)
    idx_sel_angle = np.nonzero(muap_angle_labels == args.angle)[0][0]
    muaps_at_angle = sim['muaps'][:,idx_sel_angle]

    # Generate force and angle profiles
    # ---------------------------------
    muscle_force = np.ones(args.dur_s * fs) * args.mvc_level
    angle_profile = np.ones(args.dur_s * fs) * args.angle
    timestamps = np.linspace(0, args.dur_s, args.dur_s * fs)
    samples = len(timestamps)
    print('Contraction dynamics generated correctly')

    # Generate spike trains based on simulated properties
    # ---------------------------------------------------
    # Initialise motor neuron pool and assign properties (common across all muscles)
    mn_pool = MotoneuronPool(args.num_mus, args.muscle, **mn_default_settings)

    # Initialise properties with those of the selected units
    mn_pool.properties = sim['neural_params']
    muscle_properties = sim['neural_params']

    # Generate spike trains based on force activations
    mn_pool.init_twitches(fs)
    mn_pool.init_quisistatic_ef_model()
    ext_new, spikes, fr, ipis = mn_pool.generate_spike_trains(muscle_force/100, fit=False)
    spikes = ensure_spikes_in_range(spikes,len(timestamps))

    # Check number of active units
    active_mu = 0
    for sp in spikes:
        if len(sp) > 0:
            active_mu += 1
    print(f'{args.muscle} - number of active mu: ', active_mu)

    # Store them
    muscle_mn = {
        'spikes': spikes,
        'bin_spikes': spikes_to_bin(spikes, samples),
        'fr': fr,
    }
    print('Spike trains generated correctly')

    # Generate EMG signals
    # --------------------
    emg_raw = generate_emg(sim['muaps'], muscle_mn['spikes'], muap_angle_labels, angle_profile)

    # Center EMG
    emg_raw -= emg_raw.mean(-1)[:,:,None]
    emg_raw = emg_raw.reshape(chs, samples).T

    # Generate noise
    std_emg = emg_raw.std(0)
    std_noise = std_emg * 10 ** (-args.snr_level/20)
    noise = np.random.normal(loc=np.zeros(chs), scale=std_noise, size=emg_raw.shape)
    print(f'{args.snr_level} dB - std noise: {noise.std()}')

    # Apply noise
    emg = copy(emg_raw) + noise

    # Preprocess EMG
    sos = signal.butter(2, [20, 500], 'bandpass', fs=fs, output='sos')
    emg_filt = signal.sosfiltfilt(sos, emg, axis=0)  
    emg_filt -= emg_filt.mean(0)[None,:]
    rms_emg_filt = compute_rms(emg_filt, timestamps, 0.1, 0.1, fs)
    print('EMG signals generated correctly')

    # Save data
    # ---------
    os.makedirs(args.path_save, exist_ok=True)
    file_save = os.path.join( args.path_save, f'semg_{args.muscle}_static_prepro_{args.mvc_level}mvc_{args.snr_level}dB_bs{args.iter}.hdf5')

    data = {
        'emg': emg_filt,
        'spikes': muscle_mn['bin_spikes'],
        'spikes_muscles': [args.muscle] * args.num_mus,
        'rms': rms_emg_filt.mean(-1),
        'noise': noise,
        'fs': fs,
        'angle_profile': angle_profile,
        'force_profile': muscle_force,
        'timestamps': timestamps,
        'ch_map': ch_map,
    }
    save_sim_emg(file_save, data)
    print('EMG data saved correctly')