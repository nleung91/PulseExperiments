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


def rabi(sequencer):
    # rabi sequences

    for rabi_len in range(0, 10, 1):
        sequencer.new_sequence()

        sequencer.append('m8195a_trig', Ones(time=100))
        sequencer.append('charge1',
                         Gauss(max_amp=0.5, sigma_len=rabi_len, cutoff_sigma=2, freq=4.5, phase=0, plot=False))
        readout(sequencer)

        sequencer.end_sequence()

    return sequencer.complete(plot=True)


def drag_rabi(sequencer):
    # drag_rabi sequences

    freq_ge = 4.5  # GHz
    alpha = - 0.125  # GHz

    freq_lambda = (freq_ge + alpha) / freq_ge
    optimal_beta = freq_lambda ** 2 / (4 * alpha)

    for rabi_len in range(0, 10, 2):
        sequencer.new_sequence()

        sequencer.append('m8195a_trig', Ones(time=100))
        sequencer.append('charge1',
                         DRAG(A=0.3, beta=optimal_beta, sigma_len=rabi_len, cutoff_sigma=2, freq=4.5, phase=0,
                              plot=False))
        readout(sequencer)

        sequencer.end_sequence()

    return sequencer.complete(plot=True)


def drag_optimization(sequencer, params, deltas, plot=True):
    # drag_rabi sequences

    # using current params
    params_now = params.copy()

    sequencer.new_sequence()

    sequencer.append('m8195a_trig', Ones(time=100))
    sequencer.append('charge1',
                     DRAG(A=params_now['A'], beta=params_now['beta'], sigma_len=params_now['sigma_len'],
                          cutoff_sigma=2,
                          freq=4.5, phase=0,
                          plot=False))
    readout(sequencer)

    sequencer.end_sequence()

    # start perturbation on each params
    for key in params.keys():
        params_now = params.copy()

        # + perturb
        params_now[key] = params[key] + deltas[key]

        sequencer.new_sequence()

        sequencer.append('m8195a_trig', Ones(time=100))
        sequencer.append('charge1',
                         DRAG(A=params_now['A'], beta=params_now['beta'], sigma_len=params_now['sigma_len'],
                              cutoff_sigma=2,
                              freq=4.5, phase=0,
                              plot=False))
        readout(sequencer)

        sequencer.end_sequence()

        # - perturb
        params_now[key] = params[key] - deltas[key]

        sequencer.new_sequence()

        sequencer.append('m8195a_trig', Ones(time=100))
        sequencer.append('charge1',
                         DRAG(A=params_now['A'], beta=params_now['beta'], sigma_len=params_now['sigma_len'],
                              cutoff_sigma=2,
                              freq=4.5, phase=0,
                              plot=False))
        readout(sequencer)

        sequencer.end_sequence()

    return sequencer.complete(plot=plot)


def drag_optimization_neldermead(sequencer, params, plot=True):
    # drag_rabi sequences

    readout_time_list = []

    # using current params
    params_now = params.copy()

    sequencer.new_sequence()

    sequencer.append('m8195a_trig', Ones(time=100))
    sequencer.append('charge1',
                     DRAG(A=params_now['A'], beta=params_now['beta'], sigma_len=params_now['sigma_len'],
                          cutoff_sigma=2,
                          freq=4.5, phase=0,
                          plot=False))
    readout_time = readout(sequencer)
    readout_time_list.append(readout_time)

    sequencer.end_sequence()

    return sequencer.complete(plot=plot), np.array(readout_time_list)


def run_single_experiment():
    vis = visdom.Visdom()
    vis.close()

    sequencer = Sequencer(channels, channels_awg, awg_info, channels_delay)

    multiple_sequences = drag_rabi(sequencer)

    data, measured_data = run_qutip_experiment(multiple_sequences)


def optimize_drag():
    vis = visdom.Visdom()
    vis.close()

    optimization_loop = 1000

    freq_ge = 4.5  # GHz
    alpha = 0.125  # GHz

    freq_lambda = (freq_ge - alpha) / freq_ge
    optimal_beta = freq_lambda ** 2 / (4 * alpha)

    params_init = {'A': 0.16015153996149053, 'beta': 1.0743441859336595, 'sigma_len': 10.004606863527842}
    deltas = {'A': 0.001, 'beta': 0.01 * optimal_beta, 'sigma_len': 0.01}

    params = params_init

    update_step = 0.001

    for ii in range(optimization_loop):
        sequencer = Sequencer(channels, channels_awg, awg_info, channels_delay)

        print("optimization loop: %d" % ii)
        print("params: %s" % params)

        multiple_sequences = drag_optimization(sequencer, params, deltas, plot=True)

        data, measured_data = run_qutip_experiment(multiple_sequences, plot=True)

        Pe_list = measured_data[:, 1]

        grad_list = Pe_list[1::2] - Pe_list[2::2]

        grads = {}

        # update params according to grads
        for key_id, key in enumerate(params.keys()):
            grads[key] = grad_list[key_id] / (2 * deltas[key])
            params[key] = params[key] + update_step * grads[key]

        print("Current value: %s" % Pe_list[0])
        print("gradients: %s" % grads)


def optimize_drag_neldermead():
    vis = visdom.Visdom()
    vis.close()

    import scipy


    freq_ge = 4.5  # GHz
    alpha = - 0.125  # GHz

    freq_lambda = (freq_ge + alpha) / freq_ge
    optimal_beta = freq_lambda ** 2 / (4 * alpha)

    params_init = {'A': 0.10675161505589079, 'beta': -0.93559086069256625, 'sigma_len': 14.054871633775756}

    params_values_init = np.array([v for v in params_init.values()])
    print(params_values_init)

    def opt_fun(params_values):
        sequencer = Sequencer(channels, channels_awg, awg_info, channels_delay)

        params = {}
        for ii, params_key in enumerate(params_init.keys()):
            params[params_key] = params_values[ii]

        print("params: %s" % params)

        multiple_sequences, readout_time_list = drag_optimization_neldermead(sequencer, params, plot=False)

        awg_readout_time_list = get_awg_readout_time(readout_time_list)

        data, measured_data = run_qutip_experiment(multiple_sequences, awg_readout_time_list['m8195a'], plot=False)

        Pe_list = measured_data[:, 1]

        print("Current value: %s" % Pe_list[0])

        return (1 - Pe_list[0])

    scipy.optimize.minimize(opt_fun, params_values_init, args=(), method='Nelder-Mead')


def get_awg_readout_time(readout_time_list):
    awg_readout_time_list = {}
    for awg in awg_info:
        awg_readout_time_list[awg] = (readout_time_list - awg_info[awg]['time_delay'])

    return awg_readout_time_list


if __name__ == "__main__":
    # run_single_experiment()
    optimize_drag_neldermead()