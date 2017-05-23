from sequencer import Sequencer
from pulse_classes import Gauss, Idle, Ones, Square, DRAG
from qutip_experiment import run_qutip_experiment

import numpy as np
import visdom


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


def readout(sequencer):
    sequencer.sync_channels_time(channels)

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


def rabi(sequencer):
    # rabi sequences

    for rabi_len in range(0, 10, 1):
        sequencer.new_sequence()

        sequencer.append('m8195a_trig', Ones(time=100))
        sequencer.append('charge1',
                         Gauss(max_amp=0.5, sigma_len=rabi_len, cutoff_sigma=2, freq=3.9, phase=0, plot=False))
        readout(sequencer)

        sequencer.end_sequence()

    return sequencer.complete(plot=True)


def drag_rabi(sequencer):
    # drag_rabi sequences

    freq_ge = 3.9  # GHz
    alpha = 0.2  # GHz

    freq_lambda = (freq_ge - alpha) / freq_ge
    optimal_beta = freq_lambda ** 2 / (4 * alpha)

    for rabi_len in range(0, 10, 2):
        sequencer.new_sequence()

        sequencer.append('m8195a_trig', Ones(time=100))
        sequencer.append('charge1',
                         DRAG(A=0.3, beta=optimal_beta, sigma_len=rabi_len, cutoff_sigma=2, freq=3.9, phase=0,
                              plot=False))
        readout(sequencer)

        sequencer.end_sequence()

    return sequencer.complete(plot=True)


if __name__ == "__main__":
    vis = visdom.Visdom()
    vis.close()

    sequencer = Sequencer(channels, channels_awg, awg_info, channels_delay)

    multiple_sequences = drag_rabi(sequencer)

    data, measured_data = run_qutip_experiment(multiple_sequences)
