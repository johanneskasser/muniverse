import numpy as np
import os
import tarfile as tf
import xml.etree.ElementTree as ET
import json
import pandas as pd
from edfio import *
#import mne
#from mne_bids import BIDSPath, write_raw_bids

def openOTB(inputname,ngrid):
    # Extract data and metadata from OTB+ File

    # 
    filename = inputname.split('/')[1]
    temp_dir = os.path.join('./', 'temp_tarholder')
    # make a temporary directory to store the data of the otb file if it doesn't exist yet
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    # Open the .tar file and extract all data
    with tf.open(inputname, 'r') as emg_tar:
        emg_tar.extractall(temp_dir)

    # Extract file names from .tar directory
    sig_files = [f for f in os.listdir(temp_dir) if f.endswith('.sig')]
    trial_label_sig = sig_files[0]  # only one .sig so can be used to get the trial name (0 index list->string)
    trial_label_xml = trial_label_sig.split('.')[0] + '.xml'
    trial_label_sig = os.path.join(temp_dir, trial_label_sig)
    trial_label_xml = os.path.join(temp_dir, trial_label_xml)
    sip_files = [f for f in os.listdir(temp_dir) if f.endswith('.sip')]

    # read the metadata xml file 
    with open(trial_label_xml, encoding='utf-8') as file:
        xml=ET.fromstring(file.read())

    # Get the device info
    device_info = xml.attrib

    # Get the adapter info 
    adapter_info = xml.findall('.//Adapter')

    nADbit = int(device_info['ad_bits'])
    nchans  = int(device_info['DeviceTotalChannels'])
    # read in the EMG trial data
    emg_data = np.fromfile(open(trial_label_sig),dtype='int'+ str(nADbit)) 
    emg_data = np.transpose(emg_data.reshape(int(len(emg_data)/nchans),nchans)) # need to reshape because it is read as a stream
    emg_data = emg_data.astype(float)

    # initalize data vector
    data = np.zeros((emg_data.shape[1], ngrid*64+len(sip_files)))

    # initalize vector of recorded units
    ch_units = []

    # convert the data from bits to microvolts
    for i in range(ngrid*64):
        data[:,i] = ((np.dot(emg_data[i,:],5000))/(2**float(nADbit)))
        ch_units.append('uV')

    # Get data and metadata from the aux input channels
    aux_info = dict()

    for i in range(len(sip_files)):
        # Get metadata
        tmp = sip_files[i]
        tmp = tmp.split('.')[0] + '.pro'
        tmp = os.path.join(temp_dir, tmp)
        with open(tmp, encoding='utf-8') as file:
            xml=ET.fromstring(file.read())

        aux_info[i] = {child.tag: child.text for child in xml}
        ch_units.append(aux_info[i]['unity_of_measurement'])
        
        # get data
        trial_label_sip = os.path.join(temp_dir, sip_files[i])
        aux_data = np.fromfile(open(trial_label_sip),dtype='float64')
        aux_data = aux_data[0:data.shape[0]]
        data[:,i+ngrid*64] = aux_data

    # Get the subject info
    with open(os.path.join(temp_dir, 'patient.xml'), encoding='utf-8') as file:
        xml=ET.fromstring(file.read())  

    subject_info = {child.tag: child.text for child in xml}      

    # Remove .tar folder
    for filename in os.listdir(temp_dir):
        file = os.path.join(temp_dir, filename)
        if os.path.isfile(file):
            os.remove(file)

    os.rmdir(temp_dir)

    metadata = {
                'device_info': device_info, 'adapter_info': adapter_info,
                'aux_info': aux_info, 'subject_info': subject_info, 'units': ch_units
    }

    return (data, metadata)

def make_channel_tsv(bids_path, name, data_type, units, **kwargs):
    # Make *_channels.tsv file
    # 
    # Essentials: 
    #   - name (string) 
    #   - type (string)
    #   - units (string) 

    # Set the essential columns of the _channel.tsv file in the correct order
    essentials = {'name': name, 'type': data_type, 'unit': units}
    # Add additional metadata columns to the _channels.tsv file
    other = {k: v for k, v in kwargs.items()}
        
    # Save channel metadata as tsv
    channel_metadata = {**essentials, **other}
    df_meta = pd.DataFrame(data=channel_metadata)
    df_meta.to_csv(bids_path, sep='\t', index=False, header=True)

    return()

def make_emg_json(bids_path, EMGPlacemnetScheme, EMGReference, 
                  SamplingFrequency, PowerLineFrequency, SoftwareFilters, **kwargs):
    # Make *_emg.json file
    # 
    # Essentials: 
    #   - EMGPlacemnetScheme (string) 
    #   - EMGReference (string)
    #   - SamplingFrequency (number) 
    #   - PowerLineFrequency (number or "n/a"), 
    #   - SoftwareFilters (object of objects or "n/a")

    essentials = {'EMGPlacemnetScheme': EMGPlacemnetScheme, 
                  'EMGReference': EMGReference,
                  'SamplingFrequency': SamplingFrequency,
                  'PowerLineFrequency': PowerLineFrequency,
                  'SoftwareFilters': SoftwareFilters
                  }
    
    other = {k: v for k, v in kwargs.items()}

    metadata = {**essentials, **other}

    with open(bids_path, 'w') as f:
        json.dump(metadata, f)
    return()

# def writeEDF(data, device_info, bids_path):

#     fsamp = int(device_info['SampleFrequency'])
#     seconds = np.ceil(data.shape[0]/fsamp)
#     signal = np.zeros([int(seconds*fsamp), data.shape[1]])
#     signal[0:data.shape[0],:] = data

#     edf = Edf(
#     [
#         EdfSignal(signal, sampling_frequency=fsamp),
#     ]
#     )
#     edf.write(bids_path)
#     return None


