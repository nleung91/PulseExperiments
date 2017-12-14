try:
    from .sequencer import Sequencer
    from .pulse_classes import Gauss, Idle, Ones, Square, DRAG
except:
    from sequencer import Sequencer
    from pulse_classes import Gauss, Idle, Ones, Square, DRAG
# from qutip_experiment import run_qutip_experiment

import numpy as np
import visdom


class PulseSequences:
    # channels and awgs

    def set_parameters(self, quantum_device_cfg ,experiment_cfg, hardware_cfg):
        self.quantum_device_cfg = quantum_device_cfg
        self.experiment_cfg = experiment_cfg
        self.hardware_cfg = hardware_cfg

        self.channels = hardware_cfg['channels']

        self.channels_awg = hardware_cfg['channels_awg']

        self.awg_info = hardware_cfg['awg_info']

        self.channels_delay = hardware_cfg['channels_delay']


        # pulse params

        self.qubit_freq = {"1": self.quantum_device_cfg['qubit']['1']['freq'], "2": self.quantum_device_cfg['qubit']['2']['freq']}

        self.qubit_pi = {"1": Gauss(max_amp=0.5, sigma_len=5, cutoff_sigma=2, freq=self.qubit_freq["1"], phase=0, plot=False),
                         "2": Gauss(max_amp=0.5, sigma_len=5, cutoff_sigma=2, freq=self.qubit_freq["2"], phase=0, plot=False)}

        self.multimodes = {'freq': [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
                           'pi_len': [100, 100, 100, 100, 100, 100, 100, 100]}

    def __init__(self, quantum_device_cfg ,experiment_cfg, hardware_cfg):
        self.set_parameters(quantum_device_cfg , experiment_cfg, hardware_cfg)


    def readout(self, sequencer, on_qubits = None):
        if on_qubits == None:
            on_qubits = ["1", "2"]

        sequencer.sync_channels_time(self.channels)

        readout_time = sequencer.get_time('alazar_trig')

        # get readout time to be integer multiple of 5ns (
        # 5ns is the least common multiple between tek1 dt (1/1.2 ns) and alazar dt (1 ns)
        readout_time_5ns_multiple = np.ceil(readout_time / 5) * 5

        sequencer.append_idle_to_time('alazar_trig', readout_time_5ns_multiple)
        sequencer.sync_channels_time(self.channels)

        heterodyne_cfg = self.quantum_device_cfg['heterodyne']

        for qubit_id in on_qubits:
            sequencer.append('hetero%s_I'%qubit_id,
                             Square(max_amp=heterodyne_cfg[qubit_id]['amp'], flat_len=heterodyne_cfg[qubit_id]['length'],
                                    ramp_sigma_len=20, cutoff_sigma=2, freq=heterodyne_cfg[qubit_id]['freq'], phase=0,
                                    phase_t0=readout_time_5ns_multiple))
            sequencer.append('hetero%s_Q'%qubit_id,
                             Square(max_amp=heterodyne_cfg[qubit_id]['amp'], flat_len=heterodyne_cfg[qubit_id]['length'],
                                    ramp_sigma_len=20, cutoff_sigma=2, freq=heterodyne_cfg[qubit_id]['freq'],
                                    phase=np.pi / 2, phase_t0=readout_time_5ns_multiple))
            sequencer.append('readout%s_trig'%qubit_id, Ones(time=heterodyne_cfg[qubit_id]['length']))


        sequencer.append('alazar_trig', Ones(time=100))

        return readout_time


    def sideband_rabi(self, sequencer):
        # rabi sequences

        for rabi_len in np.arange(0, 100, 10.0):
            sequencer.new_sequence()

            sequencer.append('m8195a_trig', Ones(time=100))
            sequencer.append('charge1', self.qubit_1_pi)
            sequencer.sync_channels_time(self.channels)
            sequencer.append('flux1',
                             Square(max_amp=0.5, flat_len=rabi_len, ramp_sigma_len=5, cutoff_sigma=2, freq=2.0, phase=0,
                                    plot=False))
            self.readout(sequencer)

            sequencer.end_sequence()

        return sequencer.complete(plot=False)

    def pulse_probe(self, sequencer):
        # pulse_probe sequences

        for qubit_freq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence()

            sequencer.append('m8195a_trig', Ones(time=100))
            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s' %qubit_id,
                                 Square(max_amp=self.expt_cfg['pulse_amp'], flat_len=self.expt_cfg['pulse_length'],
                                        ramp_sigma_len=20, cutoff_sigma=2, freq=qubit_freq, phase=0,
                                        phase_t0=0))

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self,plot=True)


    def rabi(self, sequencer):
        # rabi sequences

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence()

            sequencer.append('m8195a_trig', Ones(time=100))
            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s' %qubit_id,
                                 Gauss(max_amp=0.5, sigma_len=rabi_len, cutoff_sigma=2, freq=self.qubit_freq[qubit_id], phase=0,
                                       plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self,plot=True)


    def vacuum_rabi(self, sequencer):
        # vacuum rabi sequences
        heterodyne_cfg = self.quantum_device_cfg['heterodyne']

        for iq_freq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence()

            sequencer.append('m8195a_trig', Ones(time=100))
            sequencer.append('alazar_trig', Ones(time=100))
            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('hetero%s_I'%qubit_id,
                                 Square(max_amp=heterodyne_cfg[qubit_id]['amp'], flat_len=heterodyne_cfg[qubit_id]['length'],
                                        ramp_sigma_len=20, cutoff_sigma=2, freq=iq_freq, phase=0,
                                        phase_t0=0))
                sequencer.append('hetero%s_Q'%qubit_id,
                                 Square(max_amp=heterodyne_cfg[qubit_id]['amp'], flat_len=heterodyne_cfg[qubit_id]['length'],
                                        ramp_sigma_len=20, cutoff_sigma=2, freq=iq_freq,
                                        phase=np.pi / 2, phase_t0=0))
                sequencer.append('readout%s_trig'%qubit_id, Ones(time=heterodyne_cfg[qubit_id]['length']))


            sequencer.end_sequence()

        return sequencer.complete(self,plot=True)

    def histogram(self, sequencer):
        # vacuum rabi sequences
        heterodyne_cfg = self.quantum_device_cfg['heterodyne']

        for iq_freq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):

            # no pi pulse
            sequencer.new_sequence()

            sequencer.append('m8195a_trig', Ones(time=100))
            sequencer.append('alazar_trig', Ones(time=100))
            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('hetero%s_I'%qubit_id,
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=heterodyne_cfg[qubit_id]['length'],
                                        ramp_sigma_len=20, cutoff_sigma=2, freq=iq_freq, phase=0,
                                        phase_t0=0))
                sequencer.append('hetero%s_Q'%qubit_id,
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=heterodyne_cfg[qubit_id]['length'],
                                        ramp_sigma_len=20, cutoff_sigma=2, freq=iq_freq,
                                        phase=np.pi / 2, phase_t0=0))
                sequencer.append('readout%s_trig'%qubit_id, Ones(time=heterodyne_cfg[qubit_id]['length']))


            sequencer.end_sequence()

            # with pi pulse
            sequencer.new_sequence()

            sequencer.append('m8195a_trig', Ones(time=100))
            sequencer.append('alazar_trig', Ones(time=100))
            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s' %qubit_id, self.qubit_pi[qubit_id])
                sequencer.sync_channels_time(self.channels)
                sequencer.append('hetero%s_I'%qubit_id,
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=heterodyne_cfg[qubit_id]['length'],
                                        ramp_sigma_len=20, cutoff_sigma=2, freq=iq_freq, phase=0,
                                        phase_t0=0))
                sequencer.append('hetero%s_Q'%qubit_id,
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=heterodyne_cfg[qubit_id]['length'],
                                        ramp_sigma_len=20, cutoff_sigma=2, freq=iq_freq,
                                        phase=np.pi / 2, phase_t0=0))
                sequencer.append('readout%s_trig'%qubit_id, Ones(time=heterodyne_cfg[qubit_id]['length']))


            sequencer.end_sequence()

        return sequencer.complete(self,plot=True)


    def t1(self, sequencer):
        # t1 sequences

        for t1_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence()

            sequencer.append('m8195a_trig', Ones(time=100))
            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s' %qubit_id, self.qubit_pi[qubit_id])
                sequencer.append('charge%s' %qubit_id, Idle(time=t1_len))
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        pi_calibration_info = {'pi_calibration':True,'expt_cfg':self.expt_cfg, 'qubit_pi':self.qubit_pi,
                               'readout':self.readout}

        return sequencer.complete(self, plot=True)

        # for idle_len in np.arange(0, 100, 20):
        #     sequencer.new_sequence()
        #
        #     sequencer.append('m8195a_trig', Ones(time=100))
        #     sequencer.append('charge1', self.qubit_pi["1"])
        #     sequencer.append('charge1', Idle(time=idle_len))
        #     self.readout(sequencer)
        #
        #     sequencer.end_sequence()
        #
        # return sequencer.complete(plot=True)

    def alazar_test(self, sequencer):
        # drag_rabi sequences

        freq_ge = 4.5  # GHz
        alpha = - 0.125  # GHz

        freq_lambda = (freq_ge + alpha) / freq_ge
        optimal_beta = freq_lambda ** 2 / (4 * alpha)

        for rabi_len in np.arange(0, 50, 5):
            sequencer.new_sequence()

            sequencer.append('m8195a_trig', Ones(time=100))
            self.readout(sequencer)
            # sequencer.append('charge1', Idle(time=100))
            sequencer.append('charge1',
                             Gauss(max_amp=0.5, sigma_len=rabi_len, cutoff_sigma=2, freq=self.qubit_freq, phase=0,
                                   plot=False))

            sequencer.end_sequence()

        return sequencer.complete(self,plot=True)


    def get_experiment_sequences(self, experiment):
        vis = visdom.Visdom()
        vis.close()

        sequencer = Sequencer(self.channels, self.channels_awg, self.awg_info, self.channels_delay)
        self.expt_cfg = self.experiment_cfg[experiment]

        multiple_sequences = eval('self.' + experiment)(sequencer)

        return self.get_sequences(multiple_sequences)

    def get_sequences(self, multiple_sequences):
        seq_num = len(multiple_sequences)

        sequences = {}
        for channel in self.channels:
            channel_waveform = []
            for seq_id in range(seq_num):
                channel_waveform.append(multiple_sequences[seq_id][channel])
            sequences[channel] = np.array(channel_waveform)

        return sequences


    def get_awg_readout_time(self, readout_time_list):
        awg_readout_time_list = {}
        for awg in self.awg_info:
            awg_readout_time_list[awg] = (readout_time_list - self.awg_info[awg]['time_delay'])

        return awg_readout_time_list


if __name__ == "__main__":
    cfg = {
        "qubit": {"freq": 4.5}
    }

    hardware_cfg = {
        "channels": [
            "charge1", "flux1", "charge2", "flux2",
            "hetero1_I", "hetero1_Q", "hetero2_I", "hetero2_Q",
            "m8195a_trig", "readout1_trig", "readout2_trig", "alazar_trig"
        ],
        "channels_awg": {"charge1": "m8195a", "flux1": "m8195a", "charge2": "m8195a", "flux2": "m8195a",
                         "hetero1_I": "tek5014a", "hetero1_Q": "tek5014a", "hetero2_I": "tek5014a",
                         "hetero2_Q": "tek5014a",
                         "m8195a_trig": "tek5014a", "readout1_trig": "tek5014a", "readout2_trig": "tek5014a",
                         "alazar_trig": "tek5014a"},
        "awg_info": {"m8195a": {"dt": 0.0625, "min_increment": 16, "min_samples": 128, "time_delay": 110},
                     "tek5014a": {"dt": 0.83333333333, "min_increment": 16, "min_samples": 128, "time_delay": 0}}
    }

    ps = PulseSequences(cfg, hardware_cfg)

    ps.get_experiment_sequences('rabi')