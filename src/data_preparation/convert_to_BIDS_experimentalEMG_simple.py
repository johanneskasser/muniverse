from data2bids import *
from sidecar_templates import emg_sidecar_template
from edfio import *
import numpy as np

bids_path = make_bids_path(subject=1, task='isometric_30_percent_mvc', datatype='emg', root='./data')

ngrids = 4
(data, metadata) = open_otb('./../utils/MVC_30MVC.otb+',ngrids)

ch_metadata = format_otb_channel_metadata(data,metadata,ngrids)
make_channel_tsv(bids_path,ch_metadata)

emg_sidecar = emg_sidecar_template('Caillet2023')
make_emg_json(bids_path,emg_sidecar)

subject = {}
subject['name'] = bids_path['subject']
subject['age']  = 'n/a'
subject['sex'] = 'M'
subject['hand'] = 'n/a'
subject['weight'] = 'n/a'
subject['height'] = 'n/a'

make_participant_tsv(bids_path, subject)
make_participant_json(bids_path,'exp')

write_edf(data, 2048, ch_metadata['name'], bids_path)


