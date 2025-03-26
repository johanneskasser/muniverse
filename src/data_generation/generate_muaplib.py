import argparse
import os
import torch
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from easydict import EasyDict as edict
from scipy.signal import butter, filtfilt
from copy import copy

# Import from installed packages
from NeuroMotion.MSKlib.MSKpose import MSKModel
from NeuroMotion.MNPoollib.MNPool import MotoneuronPool
from NeuroMotion.MNPoollib.mn_utils import plot_spike_trains, generate_emg_mu, normalise_properties
from NeuroMotion.MNPoollib.mn_params import DEPTH, ANGLE, MS_AREA, NUM_MUS, mn_default_settings
from BioMime.models.generator import Generator
from BioMime.utils.basics import update_config, load_generator
from BioMime.utils.plot_functions import plot_muaps
from NeuroMotion.loaders import save_gen_data

NM_DATAPATH = '/Users/pm1222/Work/py-home/muniverse-demo/data/models/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate EMG signals from movements')
    parser.add_argument('--cfg', type=str, default='config.yaml', help='Name of configuration file')
    parser.add_argument('--model_pth', default='model_linear.pth', type=str, help='path of best pretrained BioMime model')
    parser.add_argument('--res_path', default='./', type=str, help='path of result folder')
    parser.add_argument('--device', default='cpu', type=str, help='cuda|cpu')
    parser.add_argument('--mode', default='sample', type=str, help='sample or morph')
    parser.add_argument('--muap_file', default='muap_examples.pkl', type=str, help='initial labelled muaps')
    
    parser.add_argument('--muscle', required=True, type=str, help='ECRB|ECRL|PL|FCU|ECU|EDCI|FDSI')
    parser.add_argument('--mov', required=True, type=str, help='flx_ext | rad_uln')
    parser.add_argument('--num_mus', default=100, type=int, help='number of motor units')

    args = parser.parse_args()
    cfg = update_config(NM_DATAPATH + args.cfg)
    num_mus = args.num_mus

    # Results path
    # ------------
    if not os.path.exists(args.res_path):
        os.mkdir(args.res_path)

    # User defined parameters
    # -----------------------    
    fs_mov = 50      # temporal frequency in Hz

    if args.mov == 'flx_ext':
        poses = ['flex', 'default', 'ext']
        r_flex = -65
        r_ext = 65
        durations = np.abs( [ r_flex, r_ext ] )/ fs_mov     # Note that len(durations) should be one less than len(poses) as it represents the intervals.
    
    elif args.mov == 'rad_uln':
        poses = ['rdev', 'default', 'udev']
        r_rad = -10
        r_uln = 25
        durations = np.abs( [ r_rad, r_uln ] )/ fs_mov

    duration = np.sum(durations)
    steps = np.round( np.sum(durations) * fs_mov).astype(int)

    if args.mode == 'morph':
        with open(NM_DATAPATH + args.muap_file, 'rb') as fl:
            db = pickle.load(fl)
        num_mus = len(db['iz'])
        ms_label = 'FDSI'
    else:
        ms_label = copy(args.muscle)

    ms_labels = [ms_label]

    # Define anatomical properties (initial conditions for BioMime)
    # -------------------------------------------------------------
    mn_pool = MotoneuronPool(num_mus, ms_label, **mn_default_settings)
    fibre_density = 200     # 200 fibres per mm^2
    num_fb = np.round(MS_AREA[ms_label] * fibre_density)
    config = edict({
        'num_fb': num_fb,           # Exponential
        'depth': DEPTH[ms_label],   # Uniform
        'angle': ANGLE[ms_label],   # Uniform
        'iz': [0.5, 0.1],         # Normal from default 0.5 ± 0.1
        'len': [1, 0.05],         # Normal from default 1 ± 0.05
        'cv': [4, 0.4]            # Normal from default 4 ± 0.4
    })
    
    if args.mode == 'morph':
        num, depth, angle, iz, cv, length, base_muaps = normalise_properties(db, num_mus, steps)
    else:
        properties = mn_pool.assign_properties(config, normalise=True)
        num = torch.from_numpy(properties['num']).reshape(num_mus, 1).repeat(1, steps)
        depth = torch.from_numpy(properties['depth']).reshape(num_mus, 1).repeat(1, steps)
        angle = torch.from_numpy(properties['angle']).reshape(num_mus, 1).repeat(1, steps)
        iz = torch.from_numpy(properties['iz']).reshape(num_mus, 1).repeat(1, steps)
        cv = torch.from_numpy(properties['cv']).reshape(num_mus, 1).repeat(1, steps)
        length = torch.from_numpy(properties['len']).reshape(num_mus, 1).repeat(1, steps)

    # OpenSim modulation
    # ------------------
    # Get the NeuroMotion package directory
    # import NeuroMotion
    # neuro_motion_dir = os.path.dirname(NeuroMotion.__file__)
    msk = MSKModel(
        model_path = NM_DATAPATH + 'ARMS_Wrist_Hand_Model_4.3/',
        model_name = 'Hand_Wrist_Model_for_development.osim',
        default_pose_path = NM_DATAPATH + 'poses.csv',
    )
    msk.sim_mov(fs_mov, poses, durations)

    # Load joint angles from file
    # file_path = './data/joint_angle.pkl'
    # with open(file_path, 'rb') as file:
    #     joint_angles = pickle.load(file)        # pd.dataframe or np.array
    # duration = 10       # seconds
    # fs_mov = 5
    # msk.load_mov(joint_angles)

    ms_labels = ['ECRB', 'ECRL', 'PL', 'FCU', 'ECU', 'EDCI', 'FDSI']
    ms_lens = msk.mov2len(ms_labels=ms_labels)
    changes = msk.len2params()
    steps = changes['steps']

    # Format changes
    if ms_label == 'FCU_u' or ms_label == 'FCU_h':
        tgt_ms_labels = ['FCU'] * num_mus
    else:
        tgt_ms_labels = [ms_label] * num_mus

    ch_depth = changes['depth'].loc[:, tgt_ms_labels]
    ch_cv = changes['cv'].loc[:, tgt_ms_labels]
    ch_len = changes['len'].loc[:, tgt_ms_labels]

    # Apply BioMime model
    # -------------------
    # Model
    generator = Generator(cfg.Model.Generator)
    generator = load_generator(args.model_pth, generator, args.device)
    generator.eval()

    # Device
    if args.device == 'cuda':
        assert torch.cuda.is_available()
        generator.cuda()

    # Define MUAP low-pass filter
    time_length = 96
    fs = 2048
    T = time_length / fs
    cutoff = 800
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    order = 4
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    if args.mode == 'sample':
        zi = torch.randn(num_mus, cfg.Model.Generator.Latent)

    start_time = time.time()

    muaps = []
    for sp in tqdm(range(steps), dynamic_ncols=True, desc='Simulating MUAPs during dynamic movement...'):
        cond = torch.vstack((
            num[:, sp],
            depth[:, sp] * ch_depth.iloc[sp, :].values,
            angle[:, sp],
            iz[:, sp],
            cv[:, sp] * ch_cv.iloc[sp, :].values,
            length[:, sp] * ch_len.iloc[sp, :].values,
        )).transpose(1, 0)

        if args.mode == 'sample':
            if args.device == 'cuda':
                zi = zi.cuda()
        else:
            if args.device == 'cuda':
                base_muaps = base_muaps.cuda()

        if args.device == 'cuda':
            cond = cond.cuda()

        if args.mode == 'sample':
            sim = generator.sample(num_mus, cond.float(), cond.device, zi)
        else:
            sim = generator.generate(base_muaps, cond.float())

        if args.device == 'cuda':
            sim = sim.permute(0, 2, 3, 1).cpu().detach().numpy()
        else:
            sim = sim.permute(0, 2, 3, 1).detach().numpy()

        num_mu_dim, n_row_dim, n_col_dim, n_time_dim = sim.shape
        sim = filtfilt(b, a, sim.reshape(-1, n_time_dim))
        muaps.append(sim.reshape(num_mu_dim, n_row_dim, n_col_dim, n_time_dim).astype(np.float32))

    muaps = np.array(muaps)
    muaps = np.transpose(muaps, (1, 0, 2, 3, 4))
    print('--- %s seconds ---' % (time.time() - start_time))
    
    # Plot MUAPs
    plot_muaps(muaps, args.res_path, np.arange(0, num_mus, 20), np.arange(0, steps, 5), suffix=ms_label)

    # Save MUAPs and parameters
    # -------------------------
    file_name = os.path.join( os.getcwd(), args.res_path, f'sim_muaps_default_{args.muscle}_{num_mus}_{args.mov}_{args.mode}.hdf5')
    save_gen_data(file_name, muaps, args.mode, args.muscle, num_mus, fs_mov, poses, durations, changes, fs, num, depth, angle, iz, cv, length)