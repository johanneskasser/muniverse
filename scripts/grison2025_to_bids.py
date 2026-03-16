import numpy as np
import pandas as pd
import json
import os
from edfio import *
from muniverse.utils.data2bids import *
from muniverse.utils.otb_io import open_otb, format_otb_channel_metadata
from pathlib import Path

# ------------------------------------------ #
# ---------  Helper functions -------------- #
# ------------------------------------------ #

def get_grid_coordinates(grid_name):
    """
    Helper funcion to extract electrode coordinates given a grid type.
    
    """

    if grid_name == 'GR04MM1305':
        x = np.zeros(64)
        y = np.zeros(64)
        y[0:12]  = 0
        x[0:12]  = np.linspace(11*4,0,12)
        y[12:25] = 4
        x[12:25] = np.linspace(0,12*4,13)
        y[25:38] = 8
        x[25:38] = np.linspace(12*4,0,13)
        y[38:51] = 12
        x[38:51] = np.linspace(0,12*4,13)
        y[51:64] = 16
        x[51:64] = np.linspace(12*4,0,13)
           
    else:
        raise ValueError('The given grid_name has no reference')

    return(x,y)

def make_electrode_metadata(
        ngrids, 
        gridname='GR04MM1305'
    ):
    """
    Helper function to curate the electrode metadata
    
    """

    # Define the columns of the electrode.tsv file
    columns = ["name", "x", "y", "coordinate_system"]

    # Init dataframe containing the electrode metadata
    df = pd.DataFrame(np.nan, index=range(64*ngrids + 2), columns=columns)
    df = df.astype({
        "name": "string", 
        "x": "float", 
        "y": "float",
        "coordinate_system": "string", 
    })


    # Loop over each electrode (of the four grids) and set metadata
    elecorode_idx = 0
    for i in np.arange(ngrids):
        (xg, yg) = get_grid_coordinates(gridname)
        for j in np.arange(64):
            df.loc[elecorode_idx, "name"] = f"E{str(elecorode_idx+1).zfill(3)}" # f'E' + str(elecorode_idx))
            # Map all electrode coordinates into the grid1 coordinate system
            df.loc[elecorode_idx, "coordinate_system"] = "grid1"
            if i==0: # Lateral-Proximal
                df.loc[elecorode_idx, "x"] = xg[j]
                df.loc[elecorode_idx, "y"] = yg[j]
            elif i==3: # Medial-Proximal
                y_shift = 20 if gridname == "GR04MM1305" else 40
                df.loc[elecorode_idx, "x"] = xg[j]
                df.loc[elecorode_idx, "y"] = yg[j] + y_shift
            elif i==1: # Lateral-Distal
                x_shift = 100 if gridname == "GR04MM1305" else 200
                y_shift = 16 if gridname == "GR04MM1305" else 32
                df.loc[elecorode_idx, "x"] = x_shift - xg[j]
                df.loc[elecorode_idx, "y"] = y_shift - yg[j]
            elif i==2: # Medial-Distal
                x_shift = 100 if gridname == "GR04MM1305" else 200
                y_shift = 36 if gridname == "GR04MM1305" else 72
                df.loc[elecorode_idx, "x"] = x_shift - xg[j]
                df.loc[elecorode_idx, "y"] = y_shift - yg[j]
            # Take care of the electrode index    
            elecorode_idx += 1    
   
   # Add the reference electrodes
    df.loc[ngrids*64+0, "name"] = "R1"
    df.loc[ngrids*64+1, "name"] = "R2"

    df.loc[ngrids*64+0, "coordinate_system"] = "lowerLeg"
    df.loc[ngrids*64+1, "coordinate_system"] = "lowerLeg"

    df.loc[ngrids*64+0, "x"] = 90
    df.loc[ngrids*64+1, "x"] = 95

    df.loc[ngrids*64+0, "y"] = 0
    df.loc[ngrids*64+1, "y"] = 0

    return df

def get_events_tsv(requested_path, fsamp, mvc_level, mvc_rate):
    """
    Helper function to convert the requested path
    into a events.tsv file
    
    """

    columns = ["onset", "duration", "sample", "mvc_rate", "mvc_level", "event_type", "description"]
    df = pd.DataFrame(columns=columns)
    df = df.astype({
        "onset": "float", 
        "duration": "float", 
        "sample": "int",
        "mvc_rate": "float", 
        "mvc_level": "float",
        "event_type": "string", 
        "description": "string"
    })

    delta = 0.5
    path_0 = requested_path[0]
    path_max = np.max(requested_path)

    l_ramp = mvc_level / mvc_rate

    if mvc_level >= 70:
        l_plateau = 10
    elif mvc_level >= 50:
        l_plateau = 15
    else:
        l_plateau = 20

    idx_1 = np.argwhere(requested_path>path_0+delta).squeeze()[0]
    idx_2 = np.argwhere(requested_path>path_max-delta).squeeze()[0]
    idx_3 = np.argwhere(requested_path>path_max-delta).squeeze()[-1]
    idx_4 = np.argwhere(requested_path>path_0+delta).squeeze()[-1]

    mask1 = np.arange(idx_1,idx_2)
    m1, b1 = np.polyfit(mask1, requested_path[mask1], 1)
    mask2 = np.arange(idx_3,idx_4)
    m2, b2 = np.polyfit(mask2, requested_path[mask2], 1)

    idx_1 = int((0 - b1) / m1)
    idx_2 = int((mvc_level - b1) / m1)
    idx_3 = int((mvc_level - b2) / m2)
    idx_4 = int((0 - b2) / m2)
    #t_4 = (path_0 - b2) / m2

    df.loc[len(df)] = [
        np.round(idx_1/fsamp,6), 0, 
        idx_1, np.nan, np.nan, 
        "muscle_on",
        "Time at which the muscle is activated."
    ]
    df.loc[len(df)] = [
        np.round(idx_1/fsamp,6), l_ramp, 
        idx_1, mvc_rate, 0, 
        "linear_isometric_ramp",
        f"Linear ramp (rate: {mvc_rate} % MVC per s; duration: {l_ramp} s) of the isometric torque starting at 0 % MVC."
    ]
    df.loc[len(df)] = [
        np.round(idx_2/fsamp,6), l_plateau, 
        idx_2, 0, mvc_level, 
        "steady_isometric",
        f"Steady isometric torque at {mvc_level}% MVC for {l_plateau} s"
    ]
    df.loc[len(df)] = [
        np.round(idx_3/fsamp,6), l_ramp, 
        idx_3, -mvc_rate, mvc_level, 
        "linear_isometric_ramp",
        f"Linear ramp (rate: {-mvc_rate} % MVC per s; duration: {l_ramp} s) of the isometric torque starting at {mvc_level} % MVC."
    ]
    df.loc[len(df)] = [
        np.round(idx_4/fsamp,6), 0, 
        idx_4, np.nan, np.nan, 
        "muscle_off",
        "Time at which the muscle is deactivated."
    ]

    return df  

# ------------------------------------------ #
# --------  Dataset-level metadata --------- #
# ------------------------------------------ #

metadatapath = str(Path(__file__).parent.parent) + '/bids_metadata/' 

with open(metadatapath + 'grison_et_al_2025.json', 'r') as f:
    manual_metadata = json.load(f)

# Sampling rate
fsamp = 10240
# Number of subjects
n_sub = 1
# Number of trials
n_mvc = 1

mvc_levels = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70]

# Prepare the dataset-level README.md file
readme = """
# Caillet et al 2023: HDsEMG recordings

BIDS-formatted version of the HDsEMG dataset published in *[Caillet et al. 2023](https://doi.org/10.1523/ENEURO.0064-23.2023)*. 
Six healthy male subjects performed two submaximal (30 and 50 percent MVC) 
isometric ankle  dorsiflexions. EMG signals were recorded from the right tibialis anterior 
using four arrays  of 64 surface electrodes for a total of 256 electrodes.

# Missing data
There is no 50 % MVC ramp-and-hold contraction for the second subject.

# Coordinate systems
All electrode coordinates (reported in mm) have been converted to a common reference 
frame corresponding to the first EMG-array (*space-grid1*). 
The positions of the reference and ground electrodes are reported in a seperate 
coordinate system (*space-lowerLeg*) reported in percent of the lower leg length. 

# Conversion
The dataset has been converted semi-automatically using the *MUniverse* software.
See *dataset_description.json* for further details.

"""

# Prepare a events sidecar file
events_sidecar = {
    "onset": {
        "Description": "Onset time of the event in seconds from recording start.", 
        "Unit": "s"
    }, 
    "duration": {
        "Description": "Duration of the event in seconds. A value of zero means that the event is a dirac pulse", 
        "Unit": "s"
    }, 
    "sample": {
        "Description": "Sample index of the event onset (zero-indexing)", 
        "Unit": "samples"
    },
    "mvc_rate": {
        "Description": "Rate at which the torque changes in percent MVC per second",
        "Unit": "% MVC / s"
    }, 
    "mvc_level": {
        "Description": "MVC (maximum voluntary contraction) level at the onset of the event",
        "Unit": "% MVC"
    },
    "event_type": {
        "Description": "Event label.",
        "Levels": {
            "muscle_on": "The muscle is activated.",
            "muscle_off": "The muscle is deactivated.",
            "linear_isometric_ramp": "The isometric torque changes linearly over time with a fixed rate.",
            "steady_isometric": "Steady isometric contraction at a fixed MVC level."
        }
    },
    "description": {
        "Description": "Free text event description."
    }
}

sourcepath = str(Path.home()) + '/Downloads/S1/'

subjects_data = {'participant_id': ['sub-01'], 
            'sex': ['n/a']}
dataset_sidecar = manual_metadata["DatasetDescription"] #dataset_sidecar_template(ID='Caillet2023')

Grison_2025 = bids_dataset(datasetname='Grison_et_al_2025', root=str(Path.home()) + '/Downloads/')
Grison_2025.set_metadata(field_name='subjects_data', source=subjects_data)
Grison_2025.set_metadata(field_name='dataset_sidecar', source=dataset_sidecar)
Grison_2025.readme = readme
Grison_2025.write()

# ------------------------------------------ #
# -------  Loop over all recordings -------- #
# ------------------------------------------ #

for i in np.arange(n_sub):

    print(f"Bidsifying data of sub-{str(i+1).zfill(2)}")

    for j in np.arange(len(mvc_levels)):

        mvc_level = mvc_levels[j]
        task_label = f"isometric{str(mvc_levels[j])}percentmvc"

        print(f"    - Recording {j}: task-{task_label}")

        # Import daata from otb+ file
        ngrids = 9
        fname =  sourcepath + str(mvc_levels[j]) + '/' + str(mvc_levels[j]) + 'mvc_semg.otb+'
        (data, metadata) = open_otb(fname, ngrids)

        # Get and write channel metadata
        ch_metadata = format_otb_channel_metadata(data,metadata,ngrids)

        df = pd.DataFrame(ch_metadata)
        filtered_df = df[df['target_muscle'].str.contains('Tibialis Anterior|n/a')]

        idx = np.asarray(filtered_df.index.to_list(),dtype=int)

        # Get electrode metadata
        el_metadata = make_electrode_metadata(ngrids=2)

        # Make the coordinate system sidecar file (here just a placeholder)
        coordsystem_metadata = manual_metadata["CoordSystemSidecar"] # {'EMGCoordinateSystem': 'local', 'EMGCoordinateUnits': 'mm'}

        # Make the emg sidecar file
        emg_sidecar = manual_metadata["EMGSidecar"] #emg_sidecar_template('Caillet2023')
        emg_sidecar['SamplingFrequency'] =  int(metadata['device_info']['SampleFrequency'])
        emg_sidecar['SoftwareVersions'] = metadata['subject_info']['software_version']
        emg_sidecar['ManufacturerModelName'] = metadata['device_info']['Name']
        emg_sidecar["TaskName"] = task_label
        emg_sidecar["RecordingDuration"] = data.shape[1]/fsamp

        # events metadata
        indices = [i for i, s in enumerate(ch_metadata["description"]) if "requested path" in s]
        target = data[indices[0], :]
        events = get_events_tsv(target, fsamp, mvc_level, mvc_rate=5)


        # Make a recording and add data and metadata
        emg_recording = bids_emg_recording(
            data_obj=Grison_2025,
            subject_label=str(i+1).zfill(2), 
            task=task_label, 
            datatype='emg'
        )
        emg_recording.set_metadata(field_name='channels', source=filtered_df)
        emg_recording.set_metadata(field_name='electrodes', source=el_metadata) 
        emg_recording.set_metadata(field_name='emg_sidecar', source=emg_sidecar)
        emg_recording.set_metadata(field_name='coord_sidecar', source=coordsystem_metadata)
        emg_recording.set_data(field_name='emg_data', mydata=data[:,idx],fsamp=emg_sidecar['SamplingFrequency'])
        emg_recording.set_metadata(field_name="events_sidecar", source=events_sidecar)
        emg_recording.set_metadata(field_name="events", source=events)

        emg_recording.write()

# ------------------------------------------ #
# ---------  Validate outputs -------------- #
# ------------------------------------------ #

err, warn, _ = Grison_2025.validate(
    print_errors=True,
    print_warnings=True,
    ignored_codes=["TSV_COLUMN_TYPE_REDEFINED"],
    ignored_fields=["HEDVersion", "StimulusPresentation", "DeviceSerialNumber"],
    ignored_files=[]
)

print("The BIDS conversion has completed")
print(f"Your BIDS dataset contains {len(err)} errors and {len(warn)} warnings")



