from slab import InstrumentManager
from slab.instruments.awg import write_Tek5014_file
import numpy as np
import os


class Experiment:
    def __init__(self, cfg):
        im = InstrumentManager()

        self.tek = im['TEK']


    def run_experiment(self, sequences, path, name):
        print(self.tek.get_id())

        tek_waveform_channels_num = 4

        tek_waveform_channels = ['hetero1_I', 'hetero1_Q', 'hetero2_I', 'hetero2_Q']
        tek_marker_channels = ['m8195a_trig', 'readout1_trig', 'readout2_trig', 'alazar_trig', None, None, None, None]

        tek_waveforms = []
        for channel in tek_waveform_channels:
            if not channel == None:
                tek_waveforms.append(sequences[channel])
            else:
                tek_waveforms.append(np.zeros_like(sequences[tek_waveform_channels[0]]))

        tek_markers = []
        for channel in tek_marker_channels:
            if not channel == None:
                tek_markers.append(sequences[channel])
            else:
                tek_markers.append(np.zeros_like(sequences[tek_marker_channels[0]]))

        write_Tek5014_file(tek_waveforms, tek_markers, os.path.join(path, 'sequences/tek.awg'), name)

        self.tek.pre_load()
        self.tek.load_sequence_file(os.path.join(path, 'sequences/tek.awg'), force_reload=True)
        self.tek.prep_experiment()

