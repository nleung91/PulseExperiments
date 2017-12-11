from slab import InstrumentManager
from slab.instruments.awg import write_Tek5014_file
from slab.instruments.awg.M8195A import upload_M8195A_sequence
from slab.instruments.awg import M8195A
import numpy as np
import os


class Experiment:
    def __init__(self, cfg):
        im = InstrumentManager()

        self.tek = im['TEK']
        self.m8195a = im['M8195A']


    def initiate_tek(self, name, path, sequences):
        print(self.tek.get_id())
        tek_waveform_channels_num = 4
        tek_waveform_channels = ['hetero1_I', 'hetero1_Q', 'hetero2_I', 'hetero2_Q']
        tek_marker_channels = ['alazar_trig', 'readout1_trig', 'readout2_trig', 'm8195a_trig', None, None, None, None]
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



    def initiate_m8195a(self, path, sequences):
        print(self.m8195a.get_id())
        waveform_channels = ['charge1', 'flux1', 'charge2', 'flux2']
        waveform_matrix = [sequences[channel] for channel in waveform_channels]

        awg = {"period_us":200, "amplitudes":[1,1,1,1]}

        m8195a = M8195A(address='192.168.14.247:5025')

        upload_M8195A_sequence(m8195a,waveform_matrix, awg, path)


    def run_experiment(self, sequences, path, name):


        self.initiate_tek(name, path, sequences)
        self.initiate_m8195a(path, sequences)

        self.m8195a.start_output()
        self.tek.prep_experiment()
        self.tek.run()

