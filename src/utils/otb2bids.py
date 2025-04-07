import numpy as np
import os
import tarfile as tf
import xml.etree.ElementTree as ET
import json
import pandas as pd


inputfile = './MVC_30MVC.otb+'
print(inputfile)

n_electrodes = 64*4
n_aux        = 3

file_name = inputfile.split('/')[1]
temp_dir = os.path.join('./', 'temp_tarholder')

# make a temporary directory to store the data of the otb file if it doesn't exist yet
if not os.path.isdir(temp_dir):
    os.mkdir(temp_dir)

# Open the .tar file and extract all data
with tf.open(inputfile, 'r') as emg_tar:
    emg_tar.extractall(temp_dir)

#os.chdir(temp_dir)
sig_files = [f for f in os.listdir(temp_dir) if f.endswith('.sig')]
trial_label_sig = sig_files[0]  # only one .sig so can be used to get the trial name (0 index list->string)
trial_label_xml = trial_label_sig.split('.')[0] + '.xml'
trial_label_sig = os.path.join(temp_dir, trial_label_sig)
trial_label_xml = os.path.join(temp_dir, trial_label_xml)

# read the contents of the trial xml file
with open(trial_label_xml, encoding='utf-8') as file:
    xml=ET.fromstring(file.read())

# get sampling frequency, no. bits of AD converter, no. channels, electrode names and muscle names
fsamp = int(xml.find('.').attrib['SampleFrequency'])
nADbit = int(xml.find('.').attrib['ad_bits'])
nchans = int(xml.find('.').attrib['DeviceTotalChannels'])
electrode_names = [child[0].attrib['ID'] for child in xml.find('./Channels')]  # the channel description is a nested 'child' of the adapter description
muscle_names = [child[0].attrib['Muscle'] for child in xml.find('./Channels')]
all_channel_elements = xml.findall('.//Channel')
all_adapter_elements = xml.findall('.//Adapter')
       

# read in the EMG trial data
emg_data = np.fromfile(open(trial_label_sig),dtype='int'+ str(nADbit)) 
emg_data = np.transpose(emg_data.reshape(int(len(emg_data)/nchans),nchans)) # need to reshape because it is read as a stream
emg_data = emg_data.astype(float) # needed otherwise you just get an integer from the bits to microvolt division
data_matrix = np.zeros((emg_data.shape[1], n_electrodes+n_aux ))

# convert the data from bits to microvolts
#for i in range(n_electrodes):
#    data[i,:] = ((np.dot(emg_data[i,:],5000))/(2**float(nADbit))) # np.dot is faster than *

name = []
signal_el = []
unit = []
reference = []
target = []
group = []
data_type = []

for i in range(n_electrodes):
    name.append('EMG'+str(i+1))
    data_type.append('EMG')
    unit.append('micro volt')
    reference.append('R1')
    target.append('tibialis anterior')
    group_name = all_channel_elements[i].attrib['Prefix']
    group_name = group_name.split(" (")[0]
    #group.append(all_channel_elements[i].attrib['Prefix'])
    group.append(group_name)
    signal_el.append('E' + all_channel_elements[i].attrib['Index'])
    data_matrix[:,i] = ((np.dot(emg_data[i,:],5000))/(2**float(nADbit)))

for i in range(n_aux):
    sip_files = [f for f in os.listdir(temp_dir) if f.endswith('.sip')]
    tmp = sip_files[i]
    tmp = tmp.split('.')[0] + '.pro'
    tmp = os.path.join(temp_dir, tmp)
    with open(tmp, encoding='utf-8') as file:
        xml2=ET.fromstring(file.read())

    trace_type = xml2.findall('./title')
    trace_type = trace_type[0].text
    aux_unit = xml2.findall('./unity_of_measurement')
    aux_unit = aux_unit[0].text

    name.append(trace_type)
    data_type.append('Force')
    unit.append(aux_unit)
    reference.append('None')
    target.append('ankle torque')
    group.append(all_channel_elements[i+n_electrodes].attrib['Prefix'])
    signal_el.append('F' + str(i+1))

    trial_label_sip = os.path.join(temp_dir, sip_files[i])
    aux_data = np.fromfile(open(trial_label_sip),dtype='float64')
    #aux_data = np.transpose(aux_data)
    aux_data = aux_data[0:data_matrix.shape[0]]

    data_matrix[:,i+n_electrodes] = aux_data

# Save metadata as json file 
metadata = {'SamplingFrequency': fsamp, 'EMGChannelCount': n_electrodes, 'EMGChannelCount': n_electrodes, 
            'EMGReference': 'placed on the ankle (proximal)', 'EMGground': 'placed on the ankle (distal)',
            'EMGElectrodeGroups': ['Grid1', 'Grid2', 'Grid3', 'Grid4']}
with open('emg.json', 'w') as f:
    json.dump(metadata, f)
# Save time series data as csv file
df = pd.DataFrame ( data=data_matrix, columns = name, index=np.arange(data_matrix.shape[0]))
df.to_csv('emg.csv')
# Save channel metadata as tsv
channel_metadata = {'name': name, 'type': data_type, 'unit': unit, 'reference': reference, 'target_muscle': target, 'group': group, 'signal_electrode': signal_el}
df_meta = pd.DataFrame(data=channel_metadata)
df_meta.to_csv('channels.tsv', sep='\t', index=False, header=True)

# delete the temp_tarholder directory since everything we need has been taken out of it
for file_name in os.listdir(temp_dir):
    file = os.path.join(temp_dir, file_name)
    if os.path.isfile(file):
       os.remove(file)

os.rmdir(temp_dir)


