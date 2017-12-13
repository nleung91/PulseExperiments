from slab import InstrumentManager
from slab.instruments.awg import write_Tek5014_file
from slab.instruments.awg.M8195A import upload_M8195A_sequence
from slab.instruments.awg import M8195A
from slab.instruments.Alazar import Alazar
import numpy as np
import os
import time
from tqdm import tqdm
import visdom
from slab.datamanagement import SlabFile
from slab.dataanalysis import get_next_filename


class Experiment:
    def __init__(self, cfg, hardware_cfg):
        self.cfg = cfg
        self.hardware_cfg = hardware_cfg

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

        awg = {"period_us": 200, "amplitudes": [1, 1, 1, 1]}

        m8195a = M8195A(address='192.168.14.247:5025')

        upload_M8195A_sequence(m8195a, waveform_matrix, awg, path)

    def awg_prep(self):
        self.m8195a.stop_output()
        self.tek.stop()
        self.tek.prep_experiment()

    def awg_run(self):
        self.m8195a.start_output()
        time.sleep(1)
        self.tek.run()


    def initiate_alazar(self, sequence_length):
        self.hardware_cfg['alazar']['samplesPerRecord'] = 2 ** (self.cfg['alazar_readout']['width'] - 1).bit_length()
        self.hardware_cfg['alazar']['recordsPerBuffer'] = sequence_length
        self.hardware_cfg['alazar']['recordsPerAcquisition'] = 100
        print("Prep Alazar Card")
        self.adc = Alazar(self.hardware_cfg['alazar'])

    def run_experiment(self, sequences, path, name):

        self.initiate_tek(name, path, sequences)
        self.initiate_m8195a(path, sequences)

        self.m8195a.start_output()
        self.tek.prep_experiment()
        self.tek.run()

        sequence_length = len(sequences['charge1'])

        self.initiate_alazar(sequence_length)

        averages = self.cfg[name]['averages']

        data_path = os.path.join(path,'data/')
        data_file = os.path.join(data_path,get_next_filename(data_path,name, suffix='.h5'))


        expt_data_ch1 = None
        expt_data_ch2 = None
        for ii in tqdm(np.arange(max(1, int(averages / 100)))):
            tpts, ch1_pts, ch2_pts = self.adc.acquire_avg_data_by_record(prep_function=self.awg_prep,
                                                                         start_function=self.awg_run,
                                                                         excise=self.cfg['alazar_readout']['window'])

            if expt_data_ch1 is None:
                expt_data_ch1 = ch1_pts
                expt_data_ch2 = ch2_pts
            else:
                expt_data_ch1 = (expt_data_ch1 * ii + ch1_pts) / (ii + 1.0)
                expt_data_ch2 = (expt_data_ch2 * ii + ch2_pts) / (ii + 1.0)

            expt_avg_data_ch1 = np.mean(expt_data_ch1, 1)
            expt_avg_data_ch2 = np.mean(expt_data_ch2, 1)


            self.slab_file = SlabFile(data_file)
            with self.slab_file as f:
                f.add('expt_data_ch1', expt_data_ch1)
                f.add('expt_avg_data_ch1', expt_avg_data_ch1)
                f.add('expt_data_ch2', expt_data_ch2)
                f.add('expt_avg_data_ch2', expt_avg_data_ch2)
                # f.add('expt_pts', self.expt_pts)
                f.close()







            # for ii in range(100):
            # time.sleep(10)
            #
            #     self.m8195a.stop_output()
            #     self.tek.stop()
            #     self.tek.prep_experiment()
            #
            #     time.sleep(1)
            #
            #     self.m8195a.start_output()
            #     self.tek.run()


