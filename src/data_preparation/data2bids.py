import numpy as np
import os
import json
import pandas as pd
from edfio import *

class emg_bids_generator:

    def __init__(self, subject, task, datatype, root, session = -1, id_options=2):
      
        # Check if the function arguments are valid
        if type(subject) is not int or subject > 10**id_options-1:
            raise ValueError('invlaid subject ID')
        
        if type(session) is not int or session > 10**id_options-1:
            raise ValueError('invlaid session ID')
        
        if datatype not in ['emg']:
            raise ValueError('datatype must be emg')

        # Process name and session input
        if session < 0:
            sub_name = 'sub' + '-' + str(subject).zfill(id_options)
            datapath = root + '/' + sub_name + '/' + datatype + '/'
        else:
            sub_name = 'sub' + '-' + str(subject).zfill(id_options)
            ses_name = 'ses' + '-' + str(session).zfill(id_options)
            datapath = root + '/' + sub_name + '/' + ses_name + '/' + datatype + '/'
        
        # Store essential information for BIDS compatible folder structure in a dictonary
        self.root = root
        self.datapath = datapath
        self.task = 'task-' + task
        self.subject = sub_name
        self.datatype = datatype

        # Generate an empty set of folders for hosting your BIDS dataset
        if not os.path.exists(datapath):
            os.makedirs(datapath)  

    def make_channel_tsv(self, channel_metadata):
        """
        Generate a BIDS compatible *_channels.tsv file

        Args:
            bids_path (dict): filename and filepath information
            channel_metadata (dict): Channel metadata with essential keys
                - name (string)
                - type (string)
                - units (string)

        """

        # Check if the essential keys exist (Note: ordering matters)
        keys = list(channel_metadata.keys())[0:3]    
        if not keys == ['name', 'type', 'unit']:
            raise ValueError('essential keys are missing or incorrectly ordered')
    
        # Get a BIDS compatible path and filename
        path = self.datapath
        name = self.subject + '_' + self.task + '_' + 'channels'

        # Convert metadata into a pandas data frame and save tsv-file
        df = pd.DataFrame(data=channel_metadata)
        df.to_csv(path + name + '.tsv', sep='\t', index=False, header=True)

        return()

    def make_electrode_tsv(self, el_metadata):
        """
        Generate a a BIDS compatible *_electrodes.tsv file

        Args:
            bids_path (dict): filename and filepath information
            el_metadata (dict): electrode metadata with essential keys
                - name (string)
                - x (float)
                - y (float)
                - z (float)
                - coordinate_system (string)

        """

        # Check if the essential keys exist (ordering matters)
        if 'z' in el_metadata:
            keys = list(el_metadata.keys())[0:5]    
            if not keys == ['name', 'x', 'y', 'z', 'coordinate_system']:
                raise ValueError('essential keys are missing or incorrectly ordered')
        else:    
            keys = list(el_metadata.keys())[0:4]    
            if not keys == ['name', 'x', 'y', 'coordinate_system']:
                raise ValueError('essential keys are missing or incorrectly ordered')

        # Get a BIDS compatible path and filename
        path = self.datapath 
        name = self.subject + '_' + self.task + '_' + 'electrodes'

        # Convert metadata into a pandas data frame and save tsv-file
        df = pd.DataFrame(data=el_metadata)
        df.to_csv(path + name + '.tsv', sep='\t', index=False, header=True)

        return()

    def make_emg_json(self, emg_metadata):
        """
        Generate a a BIDS compatible *_emg.json file

        Args:
            bids_path (dict): filename and filepath information
            emg_metadata (dict): metadata with essential keys
                - EMGPlacementScheme (str)
                - EMGReference (str)
                - SamplingFrequency (float)
                - PowerLineFrequency (float or "n/a")
                - SoftwareFilters (dict or "n/a")
                - TaskName (str)

        """

        # Check if the essential keys exist
        essentials = ['EMGPlacementScheme', 'EMGReference', 'SamplingFrequency',
                    'PowerLineFrequency', 'SoftwareFilters', 'TaskName']
        
        for i in np.arange(len(essentials)):
            if essentials[i] not in emg_metadata:
                raise ValueError('essential keys are missing')

        # Get a BIDS compatible path and filename
        path = self.datapath
        name = self.subject + '_' + self.task + '_' + self.datatype

        # Store the metadata in a json file
        with open(path + name + '.json', 'w') as f:
            json.dump(emg_metadata, f)

        return()

    def make_coordinate_system_json(self, coordsystem_metadata):
        """
        Generate a a BIDS compatible *_coordsystem.json file

        Args:
            bids_path (dict): filename and filepath information
            coordsystem_metadata (dict): metadata with essential keys
                - EMGCoordinateSystem (str)
                - EMGCoordinateUnits (str)

        """

        # Check if the essential keys exist
        essentials = ['EMGCoordinateSystem', 'EMGCoordinateUnits']
        
        for i in np.arange(len(essentials)):
            if essentials[i] not in coordsystem_metadata:
                raise ValueError('essential keys are missing')

        # Get a BIDS compatible path and filename
        path = self.datapath
        name = self.subject + '_' + self.task + '_' + 'coordsystem'

        # Store the metadata in a json file
        with open(path + name + '.json', 'w') as f:
            json.dump(coordsystem_metadata, f)

        return()

    def make_participant_tsv(self, subject_metadata):
        """
        Generate a a BIDS compatible participants.tsv file

        Args:
            bids_path (dict): filename and filepath information
            subject_metadata (dict): metadata with essential keys
                - name (str) 
                - age (float or "n/a")
                - sex (str)
                - hand (str)
                - weight (float or "n/a")
                - height (float or "n/a") 
        """

        # Check if the essential keys exist
        essentials = ['name', 'age', 'sex', 'hand', 'weight', 'height']
        
        for i in np.arange(len(essentials)):
            if essentials[i] not in subject_metadata:
                raise ValueError('essential keys are missing')

        # Get a BIDS compatible path and filename
        filename = self.root + '/' + 'participants.tsv'
     
        if os.path.isfile(filename):
            df1 = pd.read_table(filename)
            df2 = pd.DataFrame(data=subject_metadata, index=[0])
            frames = [df1, df2]
            df = pd.concat(frames)
            df = df.drop_duplicates(subset='name', keep='first')
            df.to_csv(filename, sep='\t', index=False, header=True)
        else:
            df = pd.DataFrame(data=subject_metadata, index=[0])
            df.to_csv(filename, sep='\t', index=False, header=True)

        return()

    def make_participant_json(self,data_type):
        """
        Generate a a BIDS compatible participants.json file

        Args:
            bids_path (dict): filename and filepath information
            data_type (str): 'simulation' or 'experimental'

        """

        # Hardcoded dictonary 
        if data_type == 'simulation':
            metadata = {'name': {'Description': 'Unique subject identifier'},
                        'generated by': {'Description': 'This data set contains simulated data',
                                        'string': 'Software used to generate the data'}  
                        }
        else:    
            metadata = {'name': {'Description': 'Unique subject identifier'},
                        'age': {'Description': 'Age of the participant at time of testing', 
                                'Unit': 'years'},
                        'sex': {'Description': 'Biological sex of the participant',
                                'Levels': {'F': 'female', 'M': 'male', 'O': 'other'}},
                        'handedness': {'Description': 'handedness of the participant as reported by the participant',
                                'Levels': {'L': 'left', 'R': 'right'}},        
                        'weight': {'Description': 'Body weight of the participant', 
                                'Unit': 'kg'},
                        'height': {'Description': 'Body height of the participant', 
                                'Unit': 'm'}                
                        }
            
            # Get a BIDS compatible path and filename
            filename = self.root + '/' + 'participants.json'

            # Store the metadata in a json file
            with open(filename, 'w') as f:
                json.dump(metadata, f)
        
        return()

    def make_dataset_description_json(self, metadata):
        """
        Generate a a BIDS compatible dataset_description.tsv file

        Args:
            bids_path (dict): filename and filepath information
            subject_metadata (dict): metadata with essential keys
                - Name (str) 
                - BIDSversion (str)

        """

        # Check if the essential keys exist
        essentials = ['Name', 'BIDSversion']
        
        for i in np.arange(len(essentials)):
            if essentials[i] not in metadata:
                raise ValueError('essential keys are missing')

        # Get a BIDS compatible path and filename
        filename = self.root + '/' + 'dataset_description.json'

        # Store the metadata in a json file
        with open(filename, 'w') as f:
            json.dump(metadata, f)

        return()

    def make_dataset_readme():
        # ToDo

        return()
    
    def emg_to_edf(self, data, fsamp, ch_names):
        # basic version, one could add more metadata, e.g., see https://edfio.readthedocs.io/en/stable/examples.html

        # Get duration of the signal in seconds
        seconds = np.ceil(data.shape[0]/fsamp)
        # Add zeros to the signal such that the total length is in full seconds
        signal = np.zeros([int(seconds*fsamp), data.shape[1]])
        signal[0:data.shape[0],:] = data

        edf = Edf([EdfSignal(signal[:,0], sampling_frequency=fsamp, label=ch_names[0])])

        for i in np.arange(1,signal.shape[1]):
            new_signal = EdfSignal(signal[:,i], sampling_frequency=fsamp, label=ch_names[i])
            edf.append_signals(new_signal)

        # Get a BIDS compatible path and filename
        path = self.datapath  
        name = self.subject + '_' + self.task + '_' + self.datatype

        edf.write(path + name + '.edf')
        return()




