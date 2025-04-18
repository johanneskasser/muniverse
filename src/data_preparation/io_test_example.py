import numpy as np
import pandas as pd
from edfio import *
from data2bids import emg_bids_io
from otb_io import open_otb, format_otb_channel_metadata, format_subject_metadata
from sidecar_templates import emg_sidecar_template, dataset_sidecar_template

# Initalize bids dataset
bids_dataset = emg_bids_io(subject=1)
# Add some non-sense to it
some_data = np.random.randn(10,3)
bids_dataset.set_raw_data(mydata=some_data,fsamp=1)
# Add some metadata
subject_info = {'name': bids_dataset.subject_id, 'age': 43}
bids_dataset.add_subject_metadata(subject_info)

channel_info = {'name': ['1', '2', '3'], 
                'type': ['EMG', 'EMG', 'EMG'], 
                'unit': ['V', 'V', 'V']}
bids_dataset.add_channel_metadata(channel_info)

dataset_info = {'Name': 'Just a simple toy dataset'}
bids_dataset.add_dataset_sidecar_metadata(dataset_info)

# Save results
bids_dataset.write()

# Make an other bids dataset by loading what we have just generated
another_bids_dataset = emg_bids_io(subject=1)
another_bids_dataset.read()

print(another_bids_dataset.channels)
print(another_bids_dataset.dataset_sidecar)
print(another_bids_dataset.subject)

print('Finished')




