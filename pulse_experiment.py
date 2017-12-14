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
import json


class Experiment:
    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg):
        self.quantum_device_cfg = quantum_device_cfg
        self.experiment_cfg = experiment_cfg
        self.hardware_cfg = hardware_cfg

        im = InstrumentManager()

        self.tek = im['TEK']
        self.m8195a = im['M8195A']

        self.rf1 = im['RF1']
        self.rf2 = im['RF2']

        self.flux1 = im['YOKO1']
        self.flux2 = im['YOKO3']


    def initiate_tek(self, name, path, sequences):
        print(self.tek.get_id())
        tek_waveform_channels_num = 4
        tek_waveform_channels = self.hardware_cfg['awg_info']['tek5014a']['waveform_channels']
        tek_marker_channels = self.hardware_cfg['awg_info']['tek5014a']['marker_channels']
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
        waveform_channels = self.hardware_cfg['awg_info']['m8195a']['waveform_channels']
        waveform_matrix = [sequences[channel] for channel in waveform_channels]

        awg_info = self.hardware_cfg['awg_info']['m8195a']

        upload_M8195A_sequence(self.m8195a, waveform_matrix, awg_info, path)

    def awg_prep(self):
        self.m8195a.stop_output()
        self.tek.stop()
        self.tek.prep_experiment()

    def awg_run(self):
        self.m8195a.start_output()
        time.sleep(1)
        self.tek.run()


    def initiate_alazar(self, sequence_length, averages):
        self.hardware_cfg['alazar']['samplesPerRecord'] = 2 ** (
        self.quantum_device_cfg['alazar_readout']['width'] - 1).bit_length()
        self.hardware_cfg['alazar']['recordsPerBuffer'] = sequence_length
        self.hardware_cfg['alazar']['recordsPerAcquisition'] = int(
            sequence_length * min(averages, 100))
        print("Prep Alazar Card")
        self.adc = Alazar(self.hardware_cfg['alazar'])

    def initiate_readout_rf(self):
        self.rf1.set_frequency(self.quantum_device_cfg['heterodyne']['1']['lo_freq'] * 1e9)
        self.rf2.set_frequency(self.quantum_device_cfg['heterodyne']['2']['lo_freq'] * 1e9)
        self.rf1.set_power(self.quantum_device_cfg['heterodyne']['1']['lo_power'])
        self.rf2.set_power(self.quantum_device_cfg['heterodyne']['2']['lo_power'])
        self.rf1.set_ext_pulse(mod=True)
        self.rf2.set_ext_pulse(mod=True)

    def initiate_flux(self):
        self.flux1.ramp_current(self.quantum_device_cfg['freq_flux']['1']['current_mA'] * 1e-3)
        self.flux2.ramp_current(self.quantum_device_cfg['freq_flux']['2']['current_mA'] * 1e-3)

    def save_cfg_info(self, f):
        f.attrs['quantum_device_cfg'] = json.dumps(self.quantum_device_cfg)
        f.attrs['experiment_cfg'] = json.dumps(self.experiment_cfg)
        f.attrs['hardware_cfg'] = json.dumps(self.hardware_cfg)
        f.close()

    def run_experiment(self, sequences, path, name, seq_data_file=None):

        self.initiate_readout_rf()
        self.initiate_flux()

        self.initiate_tek(name, path, sequences)
        self.initiate_m8195a(path, sequences)

        self.m8195a.start_output()
        self.tek.prep_experiment()
        self.tek.run()

        sequence_length = len(sequences['charge1'])

        averages = self.experiment_cfg[name]['averages']

        self.initiate_alazar(sequence_length, averages)



        if seq_data_file == None:
            data_path = os.path.join(path, 'data/')
            data_file = os.path.join(data_path, get_next_filename(data_path, name, suffix='.h5'))
            self.slab_file = SlabFile(data_file)
            with self.slab_file as f:
                self.save_cfg_info(f)
        else:
            self.slab_file = SlabFile(seq_data_file)
            with self.slab_file as f:
                self.save_cfg_info(f)

        expt_data_ch1 = None
        expt_data_ch2 = None
        for ii in tqdm(np.arange(max(1, int(averages / 100)))):
            tpts, ch1_pts, ch2_pts = self.adc.acquire_avg_data_by_record(prep_function=self.awg_prep,
                                                                         start_function=self.awg_run,
                                                                         excise=
                                                                         self.quantum_device_cfg['alazar_readout'][
                                                                             'window'])

            if expt_data_ch1 is None:
                expt_data_ch1 = ch1_pts
                expt_data_ch2 = ch2_pts
            else:
                expt_data_ch1 = (expt_data_ch1 * ii + ch1_pts) / (ii + 1.0)
                expt_data_ch2 = (expt_data_ch2 * ii + ch2_pts) / (ii + 1.0)

            expt_avg_data_ch1 = np.mean(expt_data_ch1, 1)
            expt_avg_data_ch2 = np.mean(expt_data_ch2, 1)

            if seq_data_file == None:
                self.slab_file = SlabFile(data_file)
                with self.slab_file as f:
                    f.add('expt_data_ch1', expt_data_ch1)
                    f.add('expt_avg_data_ch1', expt_avg_data_ch1)
                    f.add('expt_data_ch2', expt_data_ch2)
                    f.add('expt_avg_data_ch2', expt_avg_data_ch2)
                    f.close()

        if not seq_data_file == None:
            self.slab_file = SlabFile(seq_data_file)
            with self.slab_file as f:
                f.append_line('expt_avg_data_ch1', expt_avg_data_ch1)
                f.append_line('expt_avg_data_ch2', expt_avg_data_ch2)
                f.close()

