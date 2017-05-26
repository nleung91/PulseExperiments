from sequencer import Sequencer
from pulse_classes import Gauss, Idle, Ones, Square, DRAG, ARB
from qutip_experiment import run_qutip_experiment

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
                          freq=params_now['freq'], phase=0,
                          plot=False))
    readout_time = readout(sequencer)
    readout_time_list.append(readout_time)

    sequencer.end_sequence()

    return sequencer.complete(plot=plot), np.array(readout_time_list)


def arb_optimization_neldermead(sequencer, params, plot=True):
    # drag_rabi sequences

    readout_time_list = []

    # using current params
    params_now = params.copy()

    sequencer.new_sequence()

    sequencer.append('m8195a_trig', Ones(time=100))
    sequencer.append('charge1',
                     ARB(A_list=params_now['A_list'], B_list=params_now['B_list'], len=params_now['len'],
                         freq=params_now['freq'], phase=0,
                         plot=False))
    readout_time = readout(sequencer)
    readout_time_list.append(readout_time)

    sequencer.end_sequence()

    return sequencer.complete(plot=plot), np.array(readout_time_list)


def sideband_optimization_neldermead(sequencer, params, plot=True):
    # drag_rabi sequences

    readout_time_list = []

    # using current params
    params_now = params.copy()

    sequencer.new_sequence()

    sequencer.append('m8195a_trig', Ones(time=100))
    sequencer.append('charge1',
                     DRAG(A=drag_pi['A'], beta=drag_pi['beta'], sigma_len=drag_pi['sigma_len'],
                          cutoff_sigma=2,
                          freq=drag_pi['freq'], phase=0,
                          plot=False))
    sequencer.sync_channels_time(channels)
    sequencer.append('flux1',
                     Square(max_amp=params_now['A'], flat_len=params_now['flat_len'], ramp_sigma_len=5, cutoff_sigma=2,
                            freq=params_now['freq'], phase=0,
                            plot=False))
    sequencer.append('flux1', Idle(time=200))
    readout_time = readout(sequencer)
    readout_time_list.append(readout_time)

    sequencer.end_sequence()

    return sequencer.complete(plot=plot), np.array(readout_time_list)


def arb_sideband_optimization_neldermead(sequencer, params, plot=True):
    # drag_rabi sequences

    readout_time_list = []

    # using current params
    params_now = params.copy()

    sequencer.new_sequence()

    sequencer.append('m8195a_trig', Ones(time=100))
    sequencer.append('charge1',
                     DRAG(A=drag_pi['A'], beta=drag_pi['beta'], sigma_len=drag_pi['sigma_len'],
                          cutoff_sigma=2,
                          freq=drag_pi['freq'], phase=0,
                          plot=False))
    sequencer.sync_channels_time(channels)
    sequencer.append('flux1',
                     ARB(A_list=params_now['A_list'], B_list=params_now['B_list'], len=params_now['len'],
                         freq=params_now['freq'], phase=0,
                         plot=False))
    sequencer.append('flux1', Idle(time=200))
    readout_time = readout(sequencer)
    readout_time_list.append(readout_time)

    sequencer.end_sequence()

    return sequencer.complete(plot=plot), np.array(readout_time_list)


def optimize_arb_neldermead():
    vis = visdom.Visdom()
    vis.close()

    import scipy


    freq_ge = 4.5  # GHz
    alpha = - 0.125  # GHz

    freq_lambda = (freq_ge + alpha) / freq_ge
    optimal_beta = freq_lambda ** 2 / (4 * alpha)

    params_init = {'A_list': [0.14847291, 0.10139152, 0.03838122], 'B_list': [-0.07549997, -0.00026219, 0.0994355],
                   'len': 6.1609238325685594, 'freq': 4.451511875313134}  # 99.94%

    params_values_init_list = []
    # for params_key in params_init.keys():
    # if 'list' in params_key:
    # pass

    for params_key in params_init.keys():
        if 'list' in params_key:
            params_values_init_list.extend(params_init[params_key])
        else:
            params_values_init_list.append(params_init[params_key])

    params_values_init = np.array(params_values_init_list)

    def opt_fun(params_values):
        sequencer = Sequencer(channels, channels_awg, awg_info, channels_delay)

        params = {}

        list_index_acc = 0
        for ii, params_key in enumerate(params_init.keys()):
            index = list_index_acc + ii
            if 'list' in params_key:
                params[params_key] = params_values[index:index + len(params_init[params_key])]
                list_index_acc += len(params_init[params_key]) - 1
            else:
                params[params_key] = params_values[index]

        print("params: %s" % params)

        multiple_sequences, readout_time_list = arb_optimization_neldermead(sequencer, params, plot=True)

        awg_readout_time_list = get_awg_readout_time(readout_time_list)

        data, measured_data = run_qutip_experiment(multiple_sequences, awg_readout_time_list['m8195a'], plot=True)

        Pe_list = measured_data[:, 1]

        print("Current value: %s" % Pe_list[0])

        return (1 - Pe_list[0])

    scipy.optimize.minimize(opt_fun, params_values_init, args=(), method='Nelder-Mead')


def optimize_drag_neldermead():
    vis = visdom.Visdom()
    vis.close()

    import scipy


    freq_ge = 4.5  # GHz
    alpha = - 0.125  # GHz

    freq_lambda = (freq_ge + alpha) / freq_ge
    optimal_beta = freq_lambda ** 2 / (4 * alpha)

    # params_init = {'A': 0.4, 'beta': optimal_beta, 'sigma_len': 4.0}
    params_init = {'A': 0.42563777438563438, 'beta': -0.70673964408726209, 'sigma_len': 3.5783897050108426,
                   'freq': 4.499996809696226}  # 99.9%
    # params: {'A': 0.093370909035440874, 'beta': -10.866475110292235, 'sigma_len': 16.027265455966429} # gaussian pulse

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


def optimize_sideband_neldermead():
    vis = visdom.Visdom()
    vis.close()

    import scipy


    freq_ge = 4.5  # GHz
    alpha = - 0.125  # GHz

    freq_lambda = (freq_ge + alpha) / freq_ge
    optimal_beta = freq_lambda ** 2 / (4 * alpha)

    params_init = {'A': 0.52271155140480463, 'flat_len': 77.921376826335546, 'freq': 3.4018974869929988}

    params_values_init = np.array([v for v in params_init.values()])
    print(params_values_init)

    def opt_fun(params_values):
        sequencer = Sequencer(channels, channels_awg, awg_info, channels_delay)

        params = {}
        for ii, params_key in enumerate(params_init.keys()):
            params[params_key] = params_values[ii]

        print("params: %s" % params)

        multiple_sequences, readout_time_list = sideband_optimization_neldermead(sequencer, params, plot=True)

        awg_readout_time_list = get_awg_readout_time(readout_time_list)

        data, measured_data = run_qutip_experiment(multiple_sequences, awg_readout_time_list['m8195a'], plot=True)

        Pe_list = measured_data[:, 1]

        print("Current value: %s" % Pe_list[0])

        return (Pe_list[0])

    scipy.optimize.minimize(opt_fun, params_values_init, args=(), method='Nelder-Mead')


def optimize_arb_sideband_neldermead():
    vis = visdom.Visdom()
    vis.close()

    import scipy

    params_init = {'A_list': [0.51207953, 0.48719588, 0.50377773], 'B_list': [0.50808107, 0.48305012, 0.50398184],
                   'len': 73.541223556941844, 'freq': 3.4016619220703097}

    params_values_init_list = []
    # for params_key in params_init.keys():
    # if 'list' in params_key:
    # pass

    for params_key in params_init.keys():
        if 'list' in params_key:
            params_values_init_list.extend(params_init[params_key])
        else:
            params_values_init_list.append(params_init[params_key])

    params_values_init = np.array(params_values_init_list)

    def opt_fun(params_values):
        sequencer = Sequencer(channels, channels_awg, awg_info, channels_delay)

        params = {}

        list_index_acc = 0
        for ii, params_key in enumerate(params_init.keys()):
            index = list_index_acc + ii
            if 'list' in params_key:
                params[params_key] = params_values[index:index + len(params_init[params_key])]
                list_index_acc += len(params_init[params_key]) - 1
            else:
                params[params_key] = params_values[index]

        print("params: %s" % params)

        multiple_sequences, readout_time_list = arb_sideband_optimization_neldermead(sequencer, params, plot=False)

        awg_readout_time_list = get_awg_readout_time(readout_time_list)

        data, measured_data = run_qutip_experiment(multiple_sequences, awg_readout_time_list['m8195a'], plot=False)

        Pe_list = measured_data[:, 1]

        print("Current value: %s" % Pe_list[0])

        return (Pe_list[0])

    scipy.optimize.minimize(opt_fun, params_values_init, args=(), method='Nelder-Mead')


def get_awg_readout_time(readout_time_list):
    awg_readout_time_list = {}
    for awg in awg_info:
        awg_readout_time_list[awg] = (readout_time_list - awg_info[awg]['time_delay'])

    return awg_readout_time_list


if __name__ == "__main__":
    # optimize_drag_neldermead()
    # optimize_sideband_neldermead()
    # optimize_arb_neldermead()
    optimize_arb_sideband_neldermead()