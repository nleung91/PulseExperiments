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

    def set_parameters(self,cfg):
        self.channels = cfg['channels']

        self.channels_awg = cfg['channels_awg']

        self.awg_info = cfg['awg_info']

        self.channels_delay = {'readout1_trig': -20, 'readout2_trig': -20, 'alazar_trig': -50}


        # pulse params
        self.drag_pi = {'A': 0.0701200429, 'beta': -0.6998354176626167, 'sigma_len': 3.4692014249759544,
                   'freq': 4.4995338309483905}

        self.qubit_freq = cfg['qubit']['freq']

        self.qubit_1_pi = Gauss(max_amp=0.5, sigma_len=7, cutoff_sigma=2, freq=self.qubit_freq, phase=0, plot=False)

        self.multimodes = {'freq': [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9], 'pi_len': [100, 100, 100, 100, 100, 100, 100, 100]}

    def __init__(self,cfg):
        self.set_parameters(cfg)


    def readout(self, sequencer):
        sequencer.sync_channels_time(self.channels)

        readout_time = sequencer.get_time('alazar_trig')

        sequencer.append('hetero1_I',
                         Square(max_amp=0.5, flat_len=200, ramp_sigma_len=20, cutoff_sigma=2, freq=0.2, phase=0,
                                plot=False))
        sequencer.append('hetero1_Q',
                         Square(max_amp=0.5, flat_len=200, ramp_sigma_len=20, cutoff_sigma=2, freq=0.2,
                                phase=np.pi / 2))

        sequencer.append('hetero2_I',
                         Square(max_amp=0.5, flat_len=200, ramp_sigma_len=20, cutoff_sigma=2, freq=0.2, phase=0,
                                plot=False))
        sequencer.append('hetero2_Q',
                         Square(max_amp=0.5, flat_len=200, ramp_sigma_len=20, cutoff_sigma=2, freq=0.2,
                                phase=np.pi / 2))

        sequencer.append('alazar_trig', Ones(time=100))
        sequencer.append('readout1_trig', Ones(time=250))
        sequencer.append('readout2_trig', Ones(time=250))

        return readout_time


    def sideband_rabi(self, sequencer):
        # rabi sequences

        readout_time_list = []

        for freq in np.arange(3.39, 3.41, 0.005):
            sequencer.new_sequence()

            sequencer.append('m8195a_trig', Ones(time=100))
            sequencer.append('charge1', self.qubit_1_pi)
            sequencer.sync_channels_time(self.channels)
            sequencer.append('flux1',
                             Square(max_amp=0.5, flat_len=150, ramp_sigma_len=5, cutoff_sigma=2, freq=freq, phase=0,
                                    plot=False))
            sequencer.append('flux1', Idle(time=200))
            readout_time = readout(sequencer)
            readout_time_list.append(readout_time)

            sequencer.end_sequence()

        return sequencer.complete(plot=True), np.array(readout_time_list)


    def rabi(self, sequencer):
        # rabi sequences

        readout_time_list = []

        for rabi_len in np.arange(0, 10, 1.0):
            sequencer.new_sequence()

            sequencer.append('m8195a_trig', Ones(time=100))
            sequencer.append('charge1',
                             Gauss(max_amp=0.5, sigma_len=rabi_len, cutoff_sigma=2, freq=4.5, phase=0, plot=False))
            readout_time = readout(sequencer)
            readout_time_list.append(readout_time)

            sequencer.end_sequence()

        return sequencer.complete(plot=True), np.array(readout_time_list)


    def t1(self, sequencer):
        # t1 sequences

        readout_time_list = []

        for idle_len in np.arange(0, 100, 20):
            sequencer.new_sequence()

            sequencer.append('m8195a_trig', Ones(time=100))
            sequencer.append('charge1', self.qubit_1_pi)
            sequencer.append('charge1', Idle(time=idle_len))
            readout_time = self.readout(sequencer)
            readout_time_list.append(readout_time)

            sequencer.end_sequence()

        return sequencer.complete(plot=True), np.array(readout_time_list)


    def drag_rabi(self, sequencer):
        # drag_rabi sequences
        readout_time_list = []

        freq_ge = 4.5  # GHz
        alpha = - 0.125  # GHz

        freq_lambda = (freq_ge + alpha) / freq_ge
        optimal_beta = freq_lambda ** 2 / (4 * alpha)

        for rabi_len in np.arange(0, 10, 2):
            sequencer.new_sequence()

            sequencer.append('m8195a_trig', Ones(time=100))
            sequencer.append('charge1',
                             DRAG(A=0.1, beta=optimal_beta, sigma_len=rabi_len, cutoff_sigma=2, freq=4.5, phase=0,
                                  plot=False))
            readout_time = readout(sequencer)
            readout_time_list.append(readout_time)

            sequencer.end_sequence()

        return sequencer.complete(plot=True), np.array(readout_time_list)


    def run_single_experiment(self, experiment):
        vis = visdom.Visdom()
        vis.close()

        sequencer = Sequencer(self.channels, self.channels_awg, self.awg_info, self.channels_delay)

        multiple_sequences, readout_time_list = eval('self.'+experiment)(sequencer)

        awg_readout_time_list = self.get_awg_readout_time(readout_time_list)


    def get_awg_readout_time(self, readout_time_list):
        awg_readout_time_list = {}
        for awg in self.awg_info:
            awg_readout_time_list[awg] = (readout_time_list - self.awg_info[awg]['time_delay'])

        return awg_readout_time_list


if __name__ == "__main__":
    run_single_experiment('sideband_rabi')