# src/algorithms/run_upper_bound.py
import numpy as np
from scipy.io import loadmat
from .decomposition_methods import upper_bound, basic_cBSS
import matplotlib.pyplot as plt
#import sys
#sys.path.append('../evaluation/')
from ..evaluation.evaluate import *
from pathlib import Path

datapath = str(Path.home()) + '/Documents/CBM/simulation_data/RESULTS_2023_05/'

muap_data = loadmat(datapath + 'motor_unit_responses/muscle_2_f5mm.mat', struct_as_record=False, squeeze_me=True)
emg_data  = loadmat(datapath + 'mvc_experiments/muscle_2_f5mm_low_mvc.mat', struct_as_record=False, squeeze_me=True)

MUAPs = muap_data['motor_unit_responses'].HD_sEMG.MUEP_data
SIG = emg_data['data'].HD_sEMG.emg_data
SIG = np.concatenate((SIG,SIG,SIG,SIG,SIG,SIG,SIG,SIG,SIG,SIG),axis=1)
fsamp = 2000

UB = upper_bound()
UB.ext_fact = 40
ipts, predicted_spikes, sil = UB.decompose(SIG, MUAPs[:70,:,:], fsamp)

cBSS = basic_cBSS()
ipts, predicted_spikes, sil = cBSS.decompose(SIG, fsamp)



print('done')
