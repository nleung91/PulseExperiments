from sequencer import Sequencer
from pulse_classes import Gauss, Idle, Ones, Square, DRAG
#from qutip_experiment import run_qutip_experiment

import numpy as np
import visdom

# channels and awgs

channels = ['charge1', 'flux1', 'charge2', 'flux2',
            'hetero1_I', 'hetero1_Q', 'hetero2_I', 'hetero2_Q',
            'm8195a_trig', 'readout1_trig', 'readout2_trig', 'alazar_trig']

channels_awg = {'charge1': 'm8195a', 'flux1': 'm8195a', 'charge2': 'm8195a', 'flux2': 'm8195a',
                'hetero1_I': 'tek5014a', 'hetero1_Q': 'tek5014a', 'hetero2_I': 'tek5014a', 'hetero2_Q': 'tek5014a',
                'm8195a_trig': 'tek5014a', 'readout1_trig': 'tek5014a', 'readout2_trig': 'tek5014a',
                'alazar_trig': 'tek5014a'}

awg_info = {'m8195a': {'dt': 1. / 16., 'min_increment': 16, 'min_samples': 128, 'time_delay': 110},
            'tek5014a': {'dt': 1. / 1.2, 'min_increment': 16, 'min_samples': 128, 'time_delay': 0}}

channels_delay = {'readout1_trig': -20, 'readout2_trig': -20, 'alazar_trig': -50}


# pulse params
drag_pi = {'A': 0.0701200429, 'beta': -0.6998354176626167, 'sigma_len': 3.4692014249759544,
           'freq': 4.4995338309483905}

qubit_freq = 4.5

qubit_1_pi = Gauss(max_amp=0.5, sigma_len=7, cutoff_sigma=2, freq=qubit_freq, phase=0, plot=False)


def readout(sequencer):
    sequencer.sync_channels_time(channels)

    readout_time = sequencer.get_time('alazar_trig')

    sequencer.append('hetero1_I',
                     Square(max_amp=0.5, flat_len=200, ramp_sigma_len=20, cutoff_sigma=2, freq=0.2, phase=0,
                            plot=False))
    sequencer.append('hetero1_Q',
                     Square(max_amp=0.5, flat_len=200, ramp_sigma_len=20, cutoff_sigma=2, freq=0.2, phase=np.pi / 2))

    sequencer.append('hetero2_I',
                     Square(max_amp=0.5, flat_len=200, ramp_sigma_len=20, cutoff_sigma=2, freq=0.2, phase=0,
                            plot=False))
    sequencer.append('hetero2_Q',
                     Square(max_amp=0.5, flat_len=200, ramp_sigma_len=20, cutoff_sigma=2, freq=0.2, phase=np.pi / 2))

    sequencer.append('alazar_trig', Ones(time=100))
    sequencer.append('readout1_trig', Ones(time=250))
    sequencer.append('readout2_trig', Ones(time=250))

    return readout_time


def sideband_rabi(sequencer):
    # rabi sequences

    readout_time_list = []

    for freq in np.arange(3.39, 3.41, 0.005):
        sequencer.new_sequence()

        sequencer.append('m8195a_trig', Ones(time=100))
        sequencer.append('charge1',qubit_1_pi)
        sequencer.sync_channels_time(channels)
        sequencer.append('flux1',
                         Square(max_amp=0.5, flat_len=150, ramp_sigma_len=5, cutoff_sigma=2, freq=freq, phase=0,
                                plot=False))
        sequencer.append('flux1', Idle(time=200))
        # sequencer.append('flux1',
        # Gauss(max_amp=0.5, sigma_len=rabi_len, cutoff_sigma=2, freq=3.4, phase=0, plot=False))
        readout_time = readout(sequencer)
        readout_time_list.append(readout_time)

        sequencer.end_sequence()

    return sequencer.complete(plot=True), np.array(readout_time_list)


def rabi(sequencer):
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


def t1(sequencer):
    # t1 sequences

    readout_time_list = []

    for idle_len in np.arange(0, 100, 20):
        sequencer.new_sequence()

        sequencer.append('m8195a_trig', Ones(time=100))
        sequencer.append('charge1', qubit_1_pi)
        sequencer.append('charge1', Idle(time=idle_len))
        readout_time = readout(sequencer)
        readout_time_list.append(readout_time)

        sequencer.end_sequence()

    return sequencer.complete(plot=True), np.array(readout_time_list)


def drag_rabi(sequencer):
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


def run_single_experiment(experiment):
    vis = visdom.Visdom()
    vis.close()

    sequencer = Sequencer(channels, channels_awg, awg_info, channels_delay)

    multiple_sequences, readout_time_list = experiment(sequencer)

    awg_readout_time_list = get_awg_readout_time(readout_time_list)
    # data, measured_data, dt, rho_data = run_qutip_experiment(multiple_sequences, awg_readout_time_list['m8195a'], progress_bar=True)
    #
    # win = vis.line(
    #     X=np.arange(0, len(measured_data)),
    #     Y=measured_data,
    #     opts=dict(title='experiment data'))


def get_awg_readout_time(readout_time_list):
    awg_readout_time_list = {}
    for awg in awg_info:
        awg_readout_time_list[awg] = (readout_time_list - awg_info[awg]['time_delay'])

    return awg_readout_time_list


if __name__ == "__main__":
    run_single_experiment(sideband_rabi)