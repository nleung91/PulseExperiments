from slab.experiments.PulseExperiments.sequences import PulseSequences
from slab.experiments.PulseExperiments.pulse_experiment import Experiment
import numpy as np
import os
import json
from slab.dataanalysis import get_next_filename
from slab.datamanagement import SlabFile
from slab.dsfit import fitdecaysin

from skopt import Optimizer

import pickle

def histogram(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['histogram']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'histogram', suffix='.h5'))
    on_qubits = expt_cfg['on_qubits']

    lo_freq = {"1": quantum_device_cfg['heterodyne']['1']['lo_freq'],
               "2": quantum_device_cfg['heterodyne']['2']['lo_freq']}

    for amp in np.arange(expt_cfg['amp_start'], expt_cfg['amp_stop'], expt_cfg['amp_step']):
        for qubit_id in on_qubits:
            quantum_device_cfg['heterodyne'][qubit_id]['amp'] = amp
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('histogram')
        update_awg = True
        for lo_freq_delta in np.arange(expt_cfg['lo_freq_delta_start'], expt_cfg['lo_freq_delta_stop'], expt_cfg['lo_freq_delta_step']):
            for qubit_id in on_qubits:
                quantum_device_cfg['heterodyne'][qubit_id]['lo_freq'] = lo_freq[qubit_id] + lo_freq_delta

            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
            exp.run_experiment(sequences, path, 'histogram', seq_data_file, update_awg)

            update_awg = False


def qubit_frequency_flux_calibration(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
    sequences = ps.get_experiment_sequences('ramsey')
    expt_cfg = experiment_cfg['ramsey']
    uncalibrated_qubits = list(expt_cfg['on_qubits'])
    for ii in range(3):
        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'ramsey')
        expt_cfg = experiment_cfg['ramsey']
        on_qubits = expt_cfg['on_qubits']
        expt_pts = np.arange(expt_cfg['start'], expt_cfg['stop'], expt_cfg['step'])
        with SlabFile(data_file) as a:
            for qubit_id in on_qubits:
                data_list = np.array(a['expt_avg_data_ch%s' % qubit_id])
                fitdata = fitdecaysin(expt_pts, data_list, showfit=False)
                qubit_freq = quantum_device_cfg['qubit']['%s' % qubit_id]['freq']
                ramsey_freq = experiment_cfg['ramsey']['ramsey_freq']
                real_qubit_freq = qubit_freq + ramsey_freq - fitdata[1]
                possible_qubit_freq = qubit_freq + ramsey_freq + fitdata[1]
                flux_offset = -(real_qubit_freq - qubit_freq) / quantum_device_cfg['freq_flux'][qubit_id][
                    'freq_flux_slope']
                suggested_flux = round(quantum_device_cfg['freq_flux'][qubit_id]['current_mA'] + flux_offset, 4)
                print('qubit %s' %qubit_id)
                print('original qubit frequency:' + str(qubit_freq) + " GHz")
                print('Decay Time: %s ns' % (fitdata[3]))
                print("Oscillation frequency: %s GHz" % str(fitdata[1]))
                print("Suggested qubit frequency: %s GHz" % str(real_qubit_freq))
                print("possible qubit frequency: %s GHz" % str(possible_qubit_freq))
                print("Suggested flux: %s mA" % str(suggested_flux))
                print("Max contrast: %s" % str(max(data_list)-min(data_list)))

                freq_offset = ramsey_freq - fitdata[1]

                if (abs(freq_offset) < 50e-6):
                    print("Frequency is within expected value. No further calibration required.")
                    if qubit_id in uncalibrated_qubits: uncalibrated_qubits.remove(qubit_id)
                elif (abs(flux_offset) < 0.01):
                    print("Changing flux to the suggested flux: %s mA" % str(suggested_flux))
                    quantum_device_cfg['freq_flux'][qubit_id]['current_mA'] = suggested_flux
                else:
                    print("Large change in flux is required; please do so manually")
                    return

                if uncalibrated_qubits == []:
                    print("All qubits frequency calibrated.")
                    with open(os.path.join(path, 'quantum_device_config.json'), 'w') as f:
                        json.dump(quantum_device_cfg, f)
                    return


def sideband_rabi_sweep(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['sideband_rabi_sweep']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sideband_rabi_sweep', suffix='.h5'))
    on_qubits = expt_cfg['on_qubits']

    for freq in np.arange(expt_cfg['freq_start'], expt_cfg['freq_stop'], expt_cfg['freq_step']):
        experiment_cfg['sideband_rabi']['freq'] = freq
        experiment_cfg['sideband_rabi']['amp'] = expt_cfg['amp']
        experiment_cfg['sideband_rabi']['start'] = expt_cfg['time_start']
        experiment_cfg['sideband_rabi']['stop'] = expt_cfg['time_stop']
        experiment_cfg['sideband_rabi']['step'] = expt_cfg['time_step']
        experiment_cfg['sideband_rabi']['acquisition_num'] = expt_cfg['acquisition_num']
        experiment_cfg['sideband_rabi']['on_qubits'] = expt_cfg['on_qubits']
        experiment_cfg['sideband_rabi']['pi_calibration'] = expt_cfg['pi_calibration']
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('sideband_rabi')
        update_awg = True

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        exp.run_experiment(sequences, path, 'sideband_rabi_sweep', seq_data_file, update_awg)

        update_awg = False


def photon_transfer_optimize(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['photon_transfer_arb']
    data_path = os.path.join(path, 'data/')
    filename = get_next_filename(data_path, 'photon_transfer_optimize', suffix='.h5')
    seq_data_file = os.path.join(data_path, filename)

    iteration_num = 20000

    sequence_num = 2
    expt_num = 2*expt_cfg['repeat']

    A_list_len = 6

    max_a = {"1":0.6, "2":0.6}
    max_len = 1000

    limit_list = []
    limit_list += [(0.0, max_a[expt_cfg['sender_id']])]*A_list_len
    limit_list += [(0.0, max_a[expt_cfg['receiver_id']])]*A_list_len
    limit_list += [(10.0,max_len)] * 2

    ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)

    opt = Optimizer(limit_list, "GP", acq_optimizer="auto")

    gauss_z = np.linspace(-2,2,A_list_len)
    gauss_envelop = np.exp(-gauss_z**2)
    init_send_A_list = list(quantum_device_cfg['communication'][expt_cfg['sender_id']]['pi_amp'] * gauss_envelop)
    init_rece_A_list = list(quantum_device_cfg['communication'][expt_cfg['receiver_id']]['pi_amp'] * gauss_envelop)
    init_send_len = [300]
    init_rece_len = [300]

    init_x = [init_send_A_list + init_rece_A_list + init_send_len + init_rece_len] * sequence_num



    x_array = np.array(init_x)

    send_A_list = x_array[:,:A_list_len]
    rece_A_list = x_array[:,A_list_len:2*A_list_len]
    send_len = x_array[:,-2]
    rece_len = x_array[:,-1]


    sequences = ps.get_experiment_sequences('photon_transfer_arb', sequence_num = sequence_num,
                                                send_A_list = send_A_list, rece_A_list = rece_A_list,
                                                send_len = send_len, rece_len = rece_len)

    exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
    data_file = exp.run_experiment(sequences, path, 'photon_transfer_arb', seq_data_file)

    with SlabFile(data_file) as a:
        f_val_list = list(1-np.array(a['expt_avg_data_ch%s'%expt_cfg['receiver_id']])[-1])
        print(f_val_list)
        print(f_val_list[::2])
        print(f_val_list[1::2])

    f_val_all = []

    for ii in range(sequence_num):
        f_val_all += [np.mean(f_val_list[ii::sequence_num])]

    print(f_val_all)
    opt.tell(init_x, f_val_all)




    for iteration in range(iteration_num):

        next_x_list = opt.ask(sequence_num,strategy='cl_max')

        # do the experiment
        x_array = np.array(next_x_list)

        send_A_list = x_array[:,:A_list_len]
        rece_A_list = x_array[:,A_list_len:2*A_list_len]
        send_len = x_array[:,-2]
        rece_len = x_array[:,-1]
        sequences = ps.get_experiment_sequences('photon_transfer_arb', sequence_num = sequence_num,
                                                send_A_list = send_A_list, rece_A_list = rece_A_list,
                                                send_len = send_len, rece_len = rece_len)

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        data_file = exp.run_experiment(sequences, path, 'photon_transfer_arb', seq_data_file)

        with SlabFile(data_file) as a:
            f_val_list = list(1-np.array(a['expt_avg_data_ch%s'%expt_cfg['receiver_id']])[-1])
            print(f_val_list)
            print(f_val_list[::2])
            print(f_val_list[1::2])

        f_val_all = []

        for ii in range(sequence_num):
            f_val_all += [np.mean(f_val_list[ii::sequence_num])]

        print(f_val_all)

        opt.tell(next_x_list, f_val_all)

        with open(os.path.join(path,'optimizer/%s.pkl' %filename.split('.')[0]), 'wb') as f:
            pickle.dump(opt, f)


        frequency_recalibrate_cycle = 15
        if iteration % frequency_recalibrate_cycle == 0:
            qubit_frequency_flux_calibration(quantum_device_cfg, experiment_cfg, hardware_cfg, path)


def photon_transfer_sweep(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['photon_transfer']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'photon_transfer_sweep', suffix='.h5'))

    sweep = 'sender_a'

    if sweep == 'delay':
        delay_len_start = -100
        delay_len_stop = 100
        delay_len_step = 4.0

        for delay_len in np.arange(delay_len_start, delay_len_stop,delay_len_step):
            experiment_cfg['photon_transfer']['rece_delay'] = delay_len
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences('photon_transfer')

            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
            exp.run_experiment(sequences, path, 'photon_transfer', seq_data_file)

    elif sweep == 'sender_a':
        start = 0.6
        stop = 0.2
        step = -0.01

        for amp in np.arange(start, stop,step):
            quantum_device_cfg['communication'][experiment_cfg['photon_transfer']['sender_id']]['pi_amp'] = amp
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
            sequences = ps.get_experiment_sequences('photon_transfer')

            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
            exp.run_experiment(sequences, path, 'photon_transfer', seq_data_file)


def communication_rabi_amp_sweep(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['communication_rabi']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'communication_rabi_amp_sweep', suffix='.h5'))

    amp_start = 0.7
    amp_stop = 0.0
    amp_step = -0.01

    on_qubit = "2"

    for amp in np.arange(amp_start, amp_stop,amp_step):
        quantum_device_cfg['communication'][on_qubit]['pi_amp'] = amp
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('communication_rabi')

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        exp.run_experiment(sequences, path, 'communication_rabi', seq_data_file)


def sideband_rabi_freq_amp_sweep(quantum_device_cfg, experiment_cfg, hardware_cfg, path):
    expt_cfg = experiment_cfg['sideband_rabi_freq']
    data_path = os.path.join(path, 'data/')
    seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'sideband_rabi_freq_amp_sweep', suffix='.h5'))

    amp_start = 0.75
    amp_stop = 0.0
    amp_step = -0.01

    for amp in np.arange(amp_start, amp_stop,amp_step):
        experiment_cfg['sideband_rabi_freq']['amp'] = amp
        experiment_cfg['sideband_rabi_freq']['pulse_len'] = 100*amp_start/amp
        ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg)
        sequences = ps.get_experiment_sequences('sideband_rabi_freq')

        exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg)
        exp.run_experiment(sequences, path, 'sideband_rabi_freq', seq_data_file)

