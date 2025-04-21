import numpy as np
import os
import json
import pandas as pd
from edfio import *
import h5py

class emg_bids_io:

    def __init__(self, 
                 subject=1, 
                 task = 'isometric', 
                 datatype = 'emg', 
                 datasetname = 'dataset-name',
                 root = './', 
                 session = -1, 
                 n_digits = 2):
      
        # Check if the function arguments are valid
        if type(subject) is not int or subject > 10**n_digits-1:
            raise ValueError('invalid subject ID')
        
        if type(session) is not int or session > 10**n_digits-1:
            raise ValueError('invalid session ID')
        
        if datatype not in ['emg']:
            raise ValueError('datatype must be emg')

        # Process name and session input
        sub_name = 'sub' + '-' + str(subject).zfill(n_digits)
        if session < 0:
            datapath = root + datasetname + '/' + sub_name + '/' + datatype + '/'
        else:
            ses_name = 'ses' + '-' + str(session).zfill(n_digits)
            datapath = root + datasetname + '/' + sub_name + '/' + ses_name + '/' + datatype + '/'
        
        # Store essential information for BIDS compatible folder structure in a dictonary
        self.root = root + datasetname
        self.datapath = datapath
        self.task = 'task-' + task
        self.subject_id = sub_name
        self.datatype = datatype
        self.data = Edf([EdfSignal(np.zeros(1), sampling_frequency=1)])
        self.channels = pd.DataFrame(columns=['name', 'type', 'unit'])
        self.electrodes = pd.DataFrame(columns=['name','x','y','z', 'coordinate_system'])
        self.subject = pd.DataFrame(columns=['name', 'age', 'sex', 'hand', 'weight', 'height'])
        self.emg_sidecar = {'EMGPlacementScheme': [], 'EMGReference': [], 'SamplingFrequency': [],
                    'PowerLineFrequency': [], 'SoftwareFilters': [], 'TaskName': []}
        self.coord_sidecar = {'EMGCoordinateSystem': [], 'EMGCoordinateUnits': []}
        self.dataset_sidecar = {'Name': datasetname, 'BIDSversion': 'unpublished'}
        self.subject_sidecar = {'name': []} 

  

    def write(self):
        """
        Save dataset in BIDS format

        """
        # Generate an empty set of folders for your BIDS dataset
        if not os.path.exists(self.datapath):
            os.makedirs(self.datapath)
        # write *_channels.tsv
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'channels' 
        self.channels.to_csv(name + '.tsv', sep='\t', index=False, header=True)
        # write *_electrode.tsv
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'electrodes' 
        self.electrodes.to_csv(name + '.tsv', sep='\t', index=False, header=True)
        # write *_emg.json  
        name = self.datapath + self.subject_id + '_' + self.task + '_' + self.datatype
        with open(name + '.json', 'w') as f:
            json.dump(self.emg_sidecar, f)
        # write *_coordsystem.json     
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'coordsystem'
        with open(name + '.json', 'w') as f:
            json.dump(self.coord_sidecar, f)
        # write participant.tsv
        name = self.root + '/' + 'participants.tsv'
        self.subject.to_csv(name, sep='\t', index=False, header=True)
        # write participant.json
        name = self.root + '/' + 'participants.json'
        with open(name, 'w') as f:
            json.dump(self.subject_sidecar, f)  
        # write dataset.json
        name = self.root + '/' + 'dataset.json'
        with open(name, 'w') as f:
            json.dump(self.dataset_sidecar, f) 
        # write edf file 
        name = self.datapath + self.subject_id + '_' + self.task + '_' + self.datatype
        self.data.write(name + '.edf')  

        return()

    def read(self):
        """
        Import data from BIDS dataset

        """
        # read *_channels.tsv
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'channels.tsv' 
        if os.path.isfile(name):
            self.channels = pd.read_table(name)
        # read *_electrodes.tsv    
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'electrodes.tsv'
        if os.path.isfile(name):
            self.electrodes = pd.read_table(name)  
        # read *_emg.json  
        name = self.datapath + self.subject_id + '_' + self.task + '_' + self.datatype + '.json'
        if os.path.isfile(name):
            with open(name, 'r') as f:
                self.emg_sidecar = json.load(f)
        # read *_coordsystem.json     
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'coordsystem.json'
        if os.path.isfile(name):
            with open(name, 'r') as f:
                self.coord_sidecar = json.load(f)
        # read participant.tsv
        name = self.root + '/' + 'participants.tsv'
        if os.path.isfile(name):
            self.subject = pd.read_table(name)
        # read participant.json
        name = self.root + '/' + 'participants.json'
        if os.path.isfile(name):
            with open(name, 'r') as f:
                self.subject_sidecar = json.load(f) 
        # read dataset.json
        name = self.root + '/' + 'dataset.json'
        if os.path.isfile(name):
            with open(name, 'r') as f:
                self.dataset_sidecar = json.load(f) 
        # read edf file
        name = self.datapath + self.subject_id + '_' + self.task + '_' + self.datatype + '.edf'
        if os.path.isfile(name):
            self.data = read_edf(name)      
              
        return()  
                      
    def add_channel_metadata(self, new_metadata):
        """
        Add channel metadata

        Args:
            new_metadata (dict or pd.core.frame.DataFrame): Metadata

            List of essential keys (ordering matters):
                - name (string)
                - type (string)
                - units (string)

        """
    
        # If an electrode already exists overwrite existing metadata
        if type(new_metadata) == dict:
            df_new = pd.DataFrame(data=new_metadata)
            frames = [self.channels, df_new]
        elif type(new_metadata) == pd.core.frame.DataFramect:
            frames = [self.channels, new_metadata]
        else:
            raise ValueError('input has incorrect datatype')
        
        self.channels = pd.concat(frames)
        self.channels = self.channels.drop_duplicates(subset='name', keep='last')

        return()

    def add_electrode_metadata(self, new_metadata):
        """
        Add electrode metadata

        Args:
            new_metadata (dict or pd.core.frame.DataFrame): Metadata

            List of essential keys (ordering matters): 
                - name (string)
                - x (float)
                - y (float)
                - z (float)
                - coordinate_system (string)

        """

        # If an electrode already exists overwrite existing metadata
        if type(new_metadata) == dict:
            df_new = pd.DataFrame(data=new_metadata)
            frames = [self.electrodes, df_new]
        elif type(new_metadata) == pd.core.frame.DataFramect:
            frames = [self.electrodes, new_metadata]
        else:
            raise ValueError('input has incorrect datatype')
        
        self.electrodes = pd.concat(frames)
        self.electrodes = self.subject.drop_duplicates(subset=['name', 'coordinate_system'], keep='last')

        # If the z coordinate exist make sure the ordering is correct
        if 'z' in self.electrodes.columns:
            col = self.electrodes.pop('z')
            self.electrodes.insert(3, 'z', col)    

        return()

    def add_emg_sidecar_metadata(self, new_metadata):
        """
        Update metadata of the emg sidecar file 

        Args:
            new_metadata (dict or path): metadata 

            List of essential keys: 
                - EMGPlacementScheme (str)
                - EMGReference (str)
                - SamplingFrequency (float)
                - PowerLineFrequency (float or "n/a")
                - SoftwareFilters (dict or "n/a")
                - TaskName (str)

        """

        if type(new_metadata) == dict:
            self.emg_sidecar.update(new_metadata)
        elif type(new_metadata) == str and os.path.isfile(new_metadata):
            with open(new_metadata, 'r') as f:
                tmp = json.load(f)
            self.emg_sidecar.update(tmp)
        else:
            raise ValueError('input has incorrect datatype')    

        return()

    def add_coordsystem_sidecar_metadata(self, new_metadata):
        """
        Add metadata that should be stored in coordsystem.json 

        Args:
            new_metadata (dict or path): metadata 

            List of essential keys:
                - EMGCoordinateSystem (str)
                - EMGCoordinateUnits (str)

        """

        if type(new_metadata) == dict:
            self.coord_sidecar.update(new_metadata)
        elif type(new_metadata) == str and os.path.isfile(new_metadata):
            with open(new_metadata, 'r') as f:
                tmp = json.load(f)
            self.coord_sidecar.update(tmp)
        else:
            raise ValueError('input has incorrect datatype')      

        return()

    def add_subject_metadata(self, subject_metadata):
        """
        Add new subject 

        Args:
            subject_metadata (dict): metadata with essential keys
                - name (str) 
                - age (float or "n/a")
                - sex (str)
                - hand (str)
                - weight (float or "n/a")
                - height (float or "n/a") 
        """

        # If a subject already exist only overwrite existing metadata
        df_new = pd.DataFrame(data=subject_metadata, index=[0])
        frames = [self.subject, df_new]
        self.subject = pd.concat(frames)
        self.subject = self.subject.drop_duplicates(subset='name', keep='last')

        return()

    def add_participant_sidecar_metadata_from_template(self,data_type):
        """
        Add metadata for a a BIDS compatible participants.json file from a predined template

        Args:
            data_type (str): select from predefined template 'simulation' or 'experimental'

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
            
            self.subject_sidecar.update(metadata)
        
        return()

    def add_dataset_sidecar_metadata(self, new_metadata):
        """
        Add metadata to dataset_sidecar 

        Args:
            new_metadata (dict or path): metadata

            List of essential keys
                - Name (str) 
                - BIDSversion (str)

        """

        if type(new_metadata) == dict:
            self.dataset_sidecar.update(new_metadata)
        elif type(new_metadata) == str and os.path.isfile(new_metadata):
            with open(new_metadata, 'r') as f:
                tmp = json.load(f)
            self.dataset_sidecar.update(tmp)
        else:
            raise ValueError('input has incorrect datatype')

        return()

    def make_dataset_readme():
        # ToDo

        return()
    
    def set_raw_data(self, mydata, fsamp):
        """
        Add raw data and convert it into edf format

        Args:
            data (np.ndarry): emg_data (n_samples x n_channels)
            fsamp (float): Sampling frequency in Hz

        """

        # Add zeros to the signal such that the total length is in full seconds 
        seconds = np.ceil(mydata.shape[0]/fsamp)
        signal = np.zeros([int(seconds*fsamp), mydata.shape[1]])
        signal[0:mydata.shape[0],:] = mydata

        # Initalize
        edf = Edf([EdfSignal(signal[:,0], sampling_frequency=fsamp)])

        for i in np.arange(1,signal.shape[1]):
            new_signal = EdfSignal(signal[:,i], sampling_frequency=fsamp)
            edf.append_signals(new_signal)

        self.data = edf

        return()
    
    def edf_to_numpy(self, idx):
        """
        Output data of selcetd channels as numpy array

        Args:
            idx (ndarray): Indices of the channels to be stored

        """

        data_out = np.zeros((self.data.signals[idx[0]].data.shape[0], len(idx)))
        for i in np.arange(len(idx)):
            data_out[:,i] = self.data.signals[idx[i]].data

        return(data_out)  


class simulated_emg_bids_io(emg_bids_io):
    def __init__(self, config_path, hdf5_path, root='./data', datasetname='simulated-dataset'):
        # Parse config file first
        self.config = self._parse_config(config_path)
        self.hdf5_path = hdf5_path
        
        # Initialize the parent class with BIDS-compatible parameters
        subject_id = self._generate_subject_id()
        task_name = self._generate_task_name()
        
        super().__init__(
            subject=subject_id,
            task=task_name,
            datatype='emg',
            datasetname=datasetname,
            root=root
        )
        
        # Initialize additional data structures for simulated data
        self.spikes = pd.DataFrame(columns=['source_id', 'spike_time'])
        self.motor_units = pd.DataFrame(columns=['unit_id', 'recruitment_threshold', 'depth', 'innervation_zone', 
                                               'fibre_density', 'fibre_length', 'conduction_velocity', 'angle'])
        self.internals = Edf([EdfSignal(np.zeros(1), sampling_frequency=1)])
        # self.spikes_sidecar = {'source_id': [], 'spike_time': []}
        # self.motor_units_sidecar = {'unit_id': [], 'recruitment_threshold': [], 'depth': [], 'innervation_zone': [], 'fibre_density': [], 'fibre_length': [], 'conduction_velocity': [], 'angle':[]}
        self.internals_sidecar = {'SamplingFrequency': [], 'Description': 'Simulated internal variables'}
        
    def _parse_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
            
    def _generate_subject_id(self):
        return f"sim{self.config['SubjectConfiguration']['SubjectSeed']}"
        
    def _generate_task_name(self):
        # Generate a BIDS-compatible task name
        movement = self.config['MovementConfiguration']
        profile = movement['MovementProfileParameters']
        muscle = movement['TargetMuscle']
        movement_type = movement['MovementType']
        effort_level = profile['EffortLevel']
        if movement_type == 'Isometric':
            effort_profile = profile['EffortProfile']
        else:
            effort_profile = profile['AngleProfile']
        
        # Example: "FDSIisometrictrapezoid10pct"
        return f"{muscle.upper()}{movement_type.lower()}{effort_profile.lower()}{effort_level}pct"
        
    def _setup_electrode_metadata(self):
        # Generate electrode metadata based on RecordingConfiguration
        elec_config = self.config['RecordingConfiguration']['ElectrodeConfiguration']
        
        # Generate electrode names and positions: TO-DO
        names = []
        x_pos = []
        y_pos = []
        coordinate_system = []
        
        for i in range(elec_config['NElectrodes']):
            row = i // elec_config['NCols']
            col = i % elec_config['NCols']
            
            names.append(f"E{i+1}")
            x_pos.append(col * elec_config['InterElectrodeDistance'])
            y_pos.append(row * elec_config['InterElectrodeDistance'])
            coordinate_system.append("Grid1")
            
        return {
            'name': names,
            'x': x_pos,
            'y': y_pos,
            'coordinate_system': coordinate_system
        }
        
    def _setup_channel_metadata(self):
        # Generate channel metadata
        elec_config = self.config['RecordingConfiguration']['ElectrodeConfiguration']
        
        names = []
        types = []
        units = []
        
        for i in range(elec_config['NElectrodes']):
            names.append(f"E{i+1}")
            types.append("EMG")
            units.append("mV")
            
        return {
            'name': names,
            'type': types,
            'unit': units
        }
        
    def _setup_coordsystem_metadata(self):
        return {
            'EMGCoordinateSystem': 'local',
            'EMGCoordinateUnits': 'mm'
        }
        
    def _setup_spikes_data(self):
        """Read and format spike times from HDF5 file"""
        with h5py.File(self.hdf5_path, 'r') as f:
            spike_times = f['spikes'][:]
            unit_ids = f['unit_ids'][:]
            
        # Convert to long format DataFrame
        rows = []
        for unit_id, times in zip(unit_ids, spike_times):
            for t in times:
                rows.append({'source_id': unit_id, 'spike_time': t})
                
        self.spikes = pd.DataFrame(rows)
        
    def _setup_motor_units_data(self):
        """Read and format motor unit properties from HDF5 file"""
        with h5py.File(self.hdf5_path, 'r') as f:
            unit_ids = f['unit_ids'][:]
            recruitment_thresholds = f['recruitment_thresholds'][:]
            unit_properties = f['unit_properties'][:]

        # Handle individual properties: TO-DO
        self.motor_units = pd.DataFrame({
            'unit_id': unit_ids,
            'recruitment_threshold': recruitment_thresholds,
            'depth': unit_properties[:,0],
            'innervation_zone': unit_properties[:,1],
            'fibre_density': unit_properties[:,2],
            'fibre_length': unit_properties[:,3],
            'conduction_velocity': unit_properties[:,4],
            'angle': unit_properties[:,5]
        })
        
    def _setup_internals_data(self):
        """Read and format internal simulation variables from HDF5 file"""
        with h5py.File(self.hdf5_path, 'r') as f:
            internals_data = f['_internals'][:]
            fsamp = self.config['RecordingConfiguration']['SamplingFrequency']
            
        # Convert to EDF format
        edf = Edf([EdfSignal(internals_data[:,0], sampling_frequency=fsamp)])
        for i in range(1, internals_data.shape[1]):
            new_signal = EdfSignal(internals_data[:,i], sampling_frequency=fsamp)
            edf.append_signals(new_signal)
            
        self.internals = edf
        self.internals_sidecar['SamplingFrequency'] = fsamp
        
    def _write_simulated_files(self):
        """Write additional simulated data files"""
        # Write spikes data
        spikes_name = os.path.join(self.datapath, f"{self.subject_id}_{self.task}_spikes")
        self.spikes.to_csv(spikes_name + '.tsv', sep='\t', index=False)
        
        # Write motor units data
        mu_name = os.path.join(self.datapath, f"{self.subject_id}_{self.task}_motorunits")
        self.motor_units.to_csv(mu_name + '.tsv', sep='\t', index=False)
        
        # Write internals data
        internals_name = os.path.join(self.datapath, f"{self.subject_id}_{self.task}_internals")
        self.internals.write(internals_name + '.edf')
        with open(internals_name + '.json', 'w') as f:
            json.dump(self.internals_sidecar, f)
            
    def write(self):
        """Override write method to include simulated data"""
        # First set up all the simulated data
        self._setup_spikes_data()
        self._setup_motor_units_data()
        self._setup_internals_data()
        
        # Call parent's write method to handle standard BIDS files
        super().write()
        
        # Write additional simulated files
        self._write_simulated_files()
        
    def read(self):
        """Override read method to include simulated data"""
        # Call parent's read method first
        super().read()
        
        # Read simulated data files if they exist
        spikes_name = os.path.join(self.datapath, f"{self.subject_id}_{self.task}_spikes.tsv")
        if os.path.isfile(spikes_name):
            self.spikes = pd.read_table(spikes_name)
            
        mu_name = os.path.join(self.datapath, f"{self.subject_id}_{self.task}_motorunits.tsv")
        if os.path.isfile(mu_name):
            self.motor_units = pd.read_table(mu_name)
            
        internals_name = os.path.join(self.datapath, f"{self.subject_id}_{self.task}_internals.edf")
        if os.path.isfile(internals_name):
            self.internals = read_edf(internals_name)
            with open(internals_name.replace('.edf', '.json'), 'r') as f:
                self.internals_sidecar = json.load(f)


class decomp_derivatives_bids_io:

    def __init__(self, 
                 subject=1, 
                 task = 'isometric', 
                 datatype = 'emg', 
                 datasetname = 'dataset-name',
                 pipelinename = 'pipeline-name',
                 root = './', 
                 session = -1, 
                 n_digits = 2):
      
        # Check if the function arguments are valid
        if type(subject) is not int or subject > 10**n_digits-1:
            raise ValueError('invlaid subject ID')
        
        if type(session) is not int or session > 10**n_digits-1:
            raise ValueError('invlaid session ID')
        
        if datatype not in ['emg']:
            raise ValueError('datatype must be emg')

        # Process name and session input
        sub_name = 'sub' + '-' + str(subject).zfill(n_digits)
        if session < 0:
            datapath = sub_name + '/' + datatype + '/'
        else:
            ses_name = 'ses' + '-' + str(session).zfill(n_digits)
            datapath = sub_name + '/' + ses_name + '/' + datatype + '/'
        
        # Store essential information for BIDS compatible folder structure in a dictonary
        self.pipelinename = pipelinename
        self.root = root + datasetname + '_' + pipelinename
        self.datapath = self.root + '/' + datapath
        self.task = 'task-' + task
        self.subject_id = sub_name
        self.datatype = datatype
        self.source = Edf([EdfSignal(np.zeros(1), sampling_frequency=1)])
        self.spikes = pd.DataFrame(columns=['source_id', 'spike_time'])
        self.pipeline_sidecar = {'PipelineName': pipelinename, 
                                 'PipelineParameters': [],
                                 'PipelineDescription': [], 
                                 'SamplingFrequency': []}
        self.dataset_sidecar = {'Name': datasetname + '_' + pipelinename, 
                                'BIDSversion': 'unpublished', 
                                'GeneratedBy': pipelinename}
        

    def write(self):
        """
        Save dataset in BIDS format

        """
        # Generate an empty set of folders for your BIDS dataset
        if not os.path.exists(self.datapath):
            os.makedirs(self.datapath)
        # write *predictedspikes.tsv
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'predictedspikes' 
        self.spikes.to_csv(name + '.tsv', sep='\t', index=False, header=True)
        # write *_pipeline.json  
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'pipeline'
        with open(name + '.json', 'w') as f:
            json.dump(self.pipeline_sidecar, f)
        # write dataset.json
        name = self.root + '/' + 'dataset.json'
        with open(name, 'w') as f:
            json.dump(self.dataset_sidecar, f) 
        # write edf file 
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'predictedsources'
        self.source.write(name + '.edf') 

        return()
    
    def read(self):
        """
        Import data from BIDS dataset

        """
        # read *_predictedspikes.tsv
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'predictedspikes.tsv' 
        if os.path.isfile(name):
            self.spikes = pd.read_table(name)
        # read *_pipeline.json  
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'pipeline.json'
        if os.path.isfile(name):
            with open(name, 'r') as f:
                self.pipeline_sidecar = json.load(f)
        # read dataset.json
        name = self.root + '/' + 'dataset.json'
        if os.path.isfile(name):
            with open(name, 'r') as f:
                self.dataset_sidecar = json.load(f) 
        # read edf file
        name = self.datapath + self.subject_id + '_' + self.task + '_' + 'predictedsources.edf'
        if os.path.isfile(name):
            self.source = read_edf(name)    
              
        return()
    
    def add_spikes(self,spikes):
        """
        Convert a dictionary of spike times to long-format TSV-style DataFrame.

        Parameters:
            spike_dict (dict): {source_id: list of spike times}

        """
        rows = []
        for unit_id, spike_times in spikes.items():
            for t in spike_times:
                rows.append({'source_id': unit_id, 'spike_time': t})

        frames = [self.spikes, pd.DataFrame(rows)]
        self.spikes = pd.concat(frames, ignore_index=True)
        self.spikes = self.spikes.drop_duplicates(subset=['source_id', 'spike_time'])

        return()
    
    def add_dataset_sidecar_metadata(self, new_metadata):
        """
        Add metadata to dataset_sidecar 

        Args:
            new_metadata (dict or path): metadata

            List of essential keys
                - Name (str) 
                - BIDSversion (str)
                - GeneratedBy (str)

        """

        if type(new_metadata) == dict:
            self.dataset_sidecar.update(new_metadata)
        elif type(new_metadata) == str and os.path.isfile(new_metadata):
            with open(new_metadata, 'r') as f:
                tmp = json.load(f)
            self.dataset_sidecar.update(tmp)
        else:
            raise ValueError('input has incorrect datatype')

        return()
    
    def add_dataset_sidecar_metadata(self, new_metadata):
        """
        Add metadata to pipeline_sidecar 

        Args:
            new_metadata (dict or path): metadata

            List of essential keys
                - ...
                - ...

        """

        if type(new_metadata) == dict:
            self.pipeline_sidecar.update(new_metadata)
        elif type(new_metadata) == str and os.path.isfile(new_metadata):
            with open(new_metadata, 'r') as f:
                tmp = json.load(f)
            self.pipeline_sidecar.update(tmp)
        else:
            raise ValueError('input has incorrect datatype')

        return()
    
    def set_source_data(self, mysources, fsamp):
        """
        Add raw data and convert it into edf format

        Args:
            data (np.ndarry): emg_data (n_samples x n_channels)
            fsamp (float): Sampling frequency in Hz

        """

        # Add zeros to the signal such that the total length is in full seconds 
        seconds = np.ceil(mysources.shape[0]/fsamp)
        signal = np.zeros([int(seconds*fsamp), mysources.shape[1]])
        signal[0:mysources.shape[0],:] = mysources

        # Initalize
        edf = Edf([EdfSignal(signal[:,0], sampling_frequency=fsamp)])

        for i in np.arange(1,signal.shape[1]):
            new_signal = EdfSignal(signal[:,i], sampling_frequency=fsamp)
            edf.append_signals(new_signal)

        self.source = edf

        return()






