import argparse
import numpy as np
import pandas as pd
import json
import os
from edfio import *
from muniverse.data_preparation.data2bids import *
from pathlib import Path
import glob

def list_files(root, extension):
    files = list(root.rglob(f'*{extension}'))
    return files

def get_recording_info(source_file_name):
    splitname = source_file_name.split('_')

    if splitname[1].split('-')[0] == 'ses':
        sub = int(splitname[0].split('-')[1])
        ses = int(splitname[1].split('-')[1])
        task = splitname[2].split('-')[1]
        run = int(splitname[3].split('-')[1])
        data_type = splitname[4].split('.')[0]
    else:        
        sub = int(splitname[0].split('-')[1])
        task = splitname[1].split('-')[1]
        run = int(splitname[2].split('-')[1])
        ses = -1
        data_type = splitname[3].split('.')[0]

    return sub, ses, task, run, data_type

def main():
    parser = argparse.ArgumentParser(description='Convert the output of a decomposition in BIDS format')
    parser.add_argument('-d', '--dataset_name', help='Name of the dataset to process')
    parser.add_argument('-a', '--algorithm', choices=['scd', 'cbss'], help='Algorithm to use for decomposition')
    parser.add_argument('-r', '--bids_root',  default='/Users/thomi/Documents/muniverse-data', help='Path to the muniverse datasets')
    parser.add_argument('-s', '--source_root', help='Path to the raw decomposition outputs')
    
    args = parser.parse_args()

    # Datset and Pipeline information
    datasetname = args.dataset_name
    pipelinename = args.algorithm
    root = args.bids_root
    source_root = args.source_root

    # Link to the source dataset
    source_dataset = bids_dataset(datasetname=datasetname, root=root + '/Datasets/')
    source_dataset.read()
    source_file_list = source_dataset.list_all_file('_emg.edf')

    # Get all decomposition log files
    files = list_files(Path(source_root),'log.json')

    for i in np.arange(len(files)):
        with open(str(files[i]), 'r') as f:
            pipeline_sidecar = json.load(f)
        
        # Find recording in dataset
        source_file_name = pipeline_sidecar['InputData']['FileName']
        matches = source_file_list[source_file_list['file_name'].str.endswith(source_file_name)]
        match_idx = matches.index.to_list()[0]

        # Link to BIDS recording 
        emg_recording = bids_emg_recording(data_obj=source_dataset)
        emg_recording.read_data_frame(source_file_list,match_idx)
        emg_recording.read()

        fsamp = emg_recording.channels.loc[0, 'sampling_frequency']
        n_samples = emg_recording.emg_data.signals[0].data.shape[0]

        # Extract time configuration
        start_time = pipeline_sidecar['AlgorithmConfiguration']['Config']['start_time']
        end_time = pipeline_sidecar['AlgorithmConfiguration']['Config']['end_time']
        start_idx = int(start_time * fsamp)
        end_idx = int(end_time * fsamp)

        # Initalize BIDS deivatives class
        my_derivative = bids_decomp_derivatives(pipelinename=pipelinename, 
                                                root=root + '/Benchmarks/', 
                                                datasetname=datasetname, 
                                                subject=emg_recording.subject_id, 
                                                task=emg_recording.task.split('-')[1], 
                                                session=emg_recording.session, 
                                                run=emg_recording.run, 
                                                datatype=emg_recording.datatype)
        
        # Set pipeline sidecar file
        my_derivative.set_metadata('pipeline_sidecar', str(files[i]))
        
        # Get the preicted spikes
        spikes_file = str(list_files(files[i].parent, '.tsv')[0])
        spikes_df = pd.read_csv(spikes_file, delimiter='\t')
        spikes_df['spike_time'] = spikes_df['timestamp'] / fsamp + start_time
        spikes_df['timestamp'] = spikes_df['timestamp'] + int(start_time*fsamp)
        my_derivative.set_metadata('spikes', spikes_df)

        # Get the predicted sources 
        predicted_sources = str(list_files(files[i].parent, '.npz')[0])
        predicted_sources = np.load(predicted_sources)
        predicted_sources = predicted_sources['predicted_sources']

        shifted_sources = np.zeros((n_samples, predicted_sources.shape[1]))

        for j in np.arange(predicted_sources.shape[1]):
            shifted_sources[start_idx:end_idx,j] = predicted_sources[:,j] 

        my_derivative.set_data('source', shifted_sources, fsamp) 

        # Write your results
        my_derivative.write() 

if __name__ == '__main__':
    main()        

