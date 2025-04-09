import h5py
import mne
from mne_bids import BIDSPath, write_raw_bids
import numpy as np

def make_electrodes_sidecar():
    return None

def make_channels_sidecar():
    return None

def convert_to_bids_simple(filename):
    # example filename I used: data/micro_dataset/FDSI Static Prepro 10dB.hdf5
    # which had the following "keys": ["emg", "fs", "noise", ""]
    with h5py.File(filename, 'r') as f:
        emg = f['emg'][()]
        sfreq = f['fs'][()]

    ch_names = list(map(str, np.arange(1, emg.shape[1]+1)))
    info = mne.create_info(ch_names, sfreq)
    rawarray = mne.io.RawArray(emg.T, info)

    channel_types = {i: 'eeg' for i in ch_names}
    rawarray = rawarray.set_channel_types(mapping=channel_types)

    # completely hard coded for now, but easy to read from config file #To-Do
    bids_path = BIDSPath(subject='01', session='01', run='01', task='static', datatype='eeg', root='./data/micro_dataset')
    write_raw_bids(rawarray, bids_path, allow_preload=True, format='EDF', overwrite=True, verbose=False)

    # simple test to confirm that everything worked --
    # Read the EDF file (16bit int precision)
    # raw = mne.io.read_raw_edf('sub-01/ses-01/eeg/sub-01_ses-01_task-ballistic_run-01_eeg.edf', preload=True)
    # emg_data = raw.get_data(picks='eeg')
    # 
    # # Convert to numpy array
    # emg_numpy = np.array(emg_data)
    # emg_numpy.shape
    # 
    # # Load the true file (64bit float)
    # with h5py.File('data/micro_dataset/FDSI Ballistic Prepro 40mV 25dB.hdf5', 'r') as f:
    #     emg_hdf5 = f['emg'][()]
    # 
    # np.isclose(emg_hdf5.T, emg_numpy, atol=1e-4) (tolerance is paltry, but that is the tradeoff w.r.t edf)

    return None
