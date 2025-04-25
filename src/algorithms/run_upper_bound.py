import numpy as np
from scipy.io import loadmat
from decomposition_methods import upper_bound, basic_cBSS
import matplotlib.pyplot as plt
import sys
sys.path.append('../evaluation/')
from evaluate import *


muap_data = loadmat('../../../../simulation_data/RESULTS_2023_05/motor_unit_responses/muscle_2_f5mm.mat', struct_as_record=False, squeeze_me=True)
emg_data  = loadmat('../../../../simulation_data/RESULTS_2023_05/mvc_experiments/muscle_2_f5mm_low_mvc.mat', struct_as_record=False, squeeze_me=True)

MUAPs = muap_data['motor_unit_responses'].HD_sEMG.MUEP_data
SIG = emg_data['data'].HD_sEMG.emg_data
fsamp = 2000

cBSS = basic_cBSS()
ipts, predicted_spikes, sil = cBSS.decompose(SIG, fsamp)

UB = upper_bound()
ipts, predicted_spikes, sil = UB.decompose(SIG, MUAPs[:70,:,:], fsamp)

print('done')
