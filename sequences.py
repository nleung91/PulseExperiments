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

    def set_parameters(self, quantum_device_cfg, experiment_cfg, hardware_cfg):
        self.quantum_device_cfg = quantum_device_cfg
        self.experiment_cfg = experiment_cfg
        self.hardware_cfg = hardware_cfg

        self.channels = hardware_cfg['channels']

        self.channels_awg = hardware_cfg['channels_awg']

        self.awg_info = hardware_cfg['awg_info']

        self.channels_delay = hardware_cfg['channels_delay']


        # pulse params

        self.qubit_freq = {"1": self.quantum_device_cfg['qubit']['1']['freq'],
                           "2": self.quantum_device_cfg['qubit']['2']['freq']}

        self.qubit_ef_freq = {"1": self.quantum_device_cfg['qubit']['1']['freq']+self.quantum_device_cfg['qubit']['1']['anharmonicity'],
                           "2": self.quantum_device_cfg['qubit']['2']['freq']+self.quantum_device_cfg['qubit']['2']['anharmonicity']}

        self.pulse_info = self.quantum_device_cfg['pulse_info']

        self.qubit_pi = {
        "1": Gauss(max_amp=self.pulse_info['1']['pi_amp'], sigma_len=self.pulse_info['1']['pi_len'], cutoff_sigma=2,
                   freq=self.qubit_freq["1"], phase=0, plot=False),
        "2": Gauss(max_amp=self.pulse_info['2']['pi_amp'], sigma_len=self.pulse_info['2']['pi_len'], cutoff_sigma=2,
                   freq=self.qubit_freq["2"], phase=0, plot=False)}

        self.qubit_half_pi = {
        "1": Gauss(max_amp=self.pulse_info['1']['half_pi_amp'], sigma_len=self.pulse_info['1']['half_pi_len'],
                   cutoff_sigma=2, freq=self.qubit_freq["1"], phase=0, plot=False),
        "2": Gauss(max_amp=self.pulse_info['2']['half_pi_amp'], sigma_len=self.pulse_info['2']['half_pi_len'],
                   cutoff_sigma=2, freq=self.qubit_freq["2"], phase=0, plot=False)}

        self.qubit_ef_pi = {
        "1": Gauss(max_amp=self.pulse_info['1']['pi_ef_amp'], sigma_len=self.pulse_info['1']['pi_ef_len'], cutoff_sigma=2,
                   freq=self.qubit_ef_freq["1"], phase=0, plot=False),
        "2": Gauss(max_amp=self.pulse_info['2']['pi_ef_amp'], sigma_len=self.pulse_info['2']['pi_ef_len'], cutoff_sigma=2,
                   freq=self.qubit_ef_freq["2"], phase=0, plot=False)}

        self.qubit_ef_half_pi = {
        "1": Gauss(max_amp=self.pulse_info['1']['half_pi_ef_amp'], sigma_len=self.pulse_info['1']['half_pi_ef_len'],
                   cutoff_sigma=2, freq=self.qubit_ef_freq["1"], phase=0, plot=False),
        "2": Gauss(max_amp=self.pulse_info['2']['half_pi_ef_amp'], sigma_len=self.pulse_info['2']['half_pi_ef_len'],
                   cutoff_sigma=2, freq=self.qubit_ef_freq["2"], phase=0, plot=False)}

        self.multimodes = self.quantum_device_cfg['multimodes']
        self.communication = self.quantum_device_cfg['communication']
        self.sideband_cooling = self.quantum_device_cfg['sideband_cooling']

    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg):
        self.set_parameters(quantum_device_cfg, experiment_cfg, hardware_cfg)


    def readout(self, sequencer, on_qubits=None):
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
            sequencer.append('hetero%s_I' % qubit_id,
                             Square(max_amp=heterodyne_cfg[qubit_id]['amp'],
                                    flat_len=heterodyne_cfg[qubit_id]['length'],
                                    ramp_sigma_len=20, cutoff_sigma=2, freq=heterodyne_cfg[qubit_id]['freq'], phase=0,
                                    phase_t0=readout_time_5ns_multiple))
            sequencer.append('hetero%s_Q' % qubit_id,
                             Square(max_amp=heterodyne_cfg[qubit_id]['amp'],
                                    flat_len=heterodyne_cfg[qubit_id]['length'],
                                    ramp_sigma_len=20, cutoff_sigma=2, freq=heterodyne_cfg[qubit_id]['freq'],
                                    phase=np.pi / 2 + heterodyne_cfg[qubit_id]['phase_offset'], phase_t0=readout_time_5ns_multiple))
            sequencer.append('readout%s_trig' % qubit_id, Ones(time=heterodyne_cfg[qubit_id]['length']))

        sequencer.append('alazar_trig', Ones(time=self.hardware_cfg['trig_pulse_len']['alazar']))

        return readout_time

    def sideband_rabi(self, sequencer):
        # sideband rabi time domain
        rabi_freq = self.expt_cfg['freq']
        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge1', self.qubit_pi[qubit_id])
                sequencer.sync_channels_time(self.channels)
                sequencer.append('flux%s'%qubit_id,
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len, ramp_sigma_len=5, cutoff_sigma=2, freq=rabi_freq, phase=0,
                                        plot=False))
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self,plot=False)

    def sideband_rabi_freq(self, sequencer):
        # sideband rabi freq sweep
        rabi_len = self.expt_cfg['pulse_len']
        for rabi_freq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge1', self.qubit_pi[qubit_id])
                sequencer.sync_channels_time(self.channels)
                sequencer.append('flux%s'%qubit_id,
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len, ramp_sigma_len=5, cutoff_sigma=2, freq=rabi_freq, phase=0,
                                        plot=False))
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self,plot=False)

    def ef_sideband_rabi_freq(self, sequencer):
        # sideband rabi freq sweep
        rabi_len = self.expt_cfg['pulse_len']
        for rabi_freq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                sequencer.append('charge%s' % qubit_id, self.qubit_ef_pi[qubit_id])
                sequencer.sync_channels_time(self.channels)
                sequencer.append('flux%s'%qubit_id,
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len, ramp_sigma_len=5, cutoff_sigma=2, freq=rabi_freq, phase=0,
                                        plot=False))
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self,plot=False)


    def pulse_probe(self, sequencer):
        # pulse_probe sequences

        for qubit_freq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s' % qubit_id,
                                 Square(max_amp=self.expt_cfg['pulse_amp'], flat_len=self.expt_cfg['pulse_length'],
                                        ramp_sigma_len=20, cutoff_sigma=2, freq=qubit_freq, phase=0,
                                        phase_t0=0))

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def rabi(self, sequencer):
        # rabi sequences

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s' % qubit_id,
                                 Gauss(max_amp=self.expt_cfg['amp'], sigma_len=rabi_len, cutoff_sigma=2,
                                       freq=self.qubit_freq[qubit_id], phase=0,
                                       plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def vacuum_rabi(self, sequencer):
        # vacuum rabi sequences
        heterodyne_cfg = self.quantum_device_cfg['heterodyne']

        for iq_freq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            sequencer.append('alazar_trig', Ones(time=self.hardware_cfg['trig_pulse_len']['alazar']))
            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('hetero%s_I' % qubit_id,
                                 Square(max_amp=heterodyne_cfg[qubit_id]['amp'],
                                        flat_len=heterodyne_cfg[qubit_id]['length'],
                                        ramp_sigma_len=20, cutoff_sigma=2, freq=iq_freq, phase=0,
                                        phase_t0=0))
                sequencer.append('hetero%s_Q' % qubit_id,
                                 Square(max_amp=heterodyne_cfg[qubit_id]['amp'],
                                        flat_len=heterodyne_cfg[qubit_id]['length'],
                                        ramp_sigma_len=20, cutoff_sigma=2, freq=iq_freq,
                                        phase=np.pi / 2, phase_t0=0))
                sequencer.append('readout%s_trig' % qubit_id, Ones(time=heterodyne_cfg[qubit_id]['length']))

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def histogram(self, sequencer):
        # vacuum rabi sequences
        heterodyne_cfg = self.quantum_device_cfg['heterodyne']

        for ii in range(50):

            # no pi pulse
            sequencer.new_sequence(self)

            self.readout(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

            # with pi pulse
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
            self.readout(sequencer, self.expt_cfg['on_qubits'])
            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)


    def t1(self, sequencer):
        # t1 sequences

        for t1_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                sequencer.append('charge%s' % qubit_id, Idle(time=t1_len))
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()
        #
        # pi_calibration_info = {'pi_calibration': True, 'expt_cfg': self.expt_cfg, 'qubit_pi': self.qubit_pi,
        #                        'readout': self.readout}

        return sequencer.complete(self, plot=False)

        # for idle_len in np.arange(0, 100, 20):
        # sequencer.new_sequence(self)
        #
        #     sequencer.append('charge1', self.qubit_pi["1"])
        #     sequencer.append('charge1', Idle(time=idle_len))
        #     self.readout(sequencer)
        #
        #     sequencer.end_sequence()
        #
        # return sequencer.complete(plot=True)

    def ef_t1(self, sequencer):
        # t1 for the e and f level

        for ef_t1_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                sequencer.append('charge%s' % qubit_id, self.qubit_ef_pi[qubit_id])
                sequencer.append('charge%s' % qubit_id, Idle(time=ef_t1_len))
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def ef_rabi(self, sequencer):
        # ef rabi sequences

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                if self.expt_cfg['ge_pi']:
                    sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                sequencer.append('charge%s' % qubit_id,
                                 Gauss(max_amp=self.expt_cfg['amp'], sigma_len=rabi_len,
                   cutoff_sigma=2, freq=self.qubit_ef_freq[qubit_id], phase=0, plot=False))
                # sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def ef_ramsey(self, sequencer):
        # ef ramsey sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                sequencer.append('charge%s' % qubit_id, self.qubit_ef_half_pi[qubit_id])
                sequencer.append('charge%s' % qubit_id, Idle(time=ramsey_len))
                sequencer.append('charge%s' % qubit_id,
                                 Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_ef_amp'], sigma_len=self.pulse_info[qubit_id]['half_pi_ef_len'],
                   cutoff_sigma=2, freq=self.qubit_ef_freq[qubit_id], phase=2*np.pi*ramsey_len*self.expt_cfg['ramsey_freq'], plot=False))
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def ef_echo(self, sequencer):
        # ef echo sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                sequencer.append('charge%s' % qubit_id, self.qubit_ef_half_pi[qubit_id])
                for echo_id in self.expt_cfg['echo_times']:
                    sequencer.append('charge%s' % qubit_id, Idle(time=ramsey_len/(float(2*self.expt_cfg['echo_times']))))
                    if self.expt_cfg['cp']:
                        sequencer.append('charge%s' % qubit_id, self.qubit_ef_pi[qubit_id])
                    elif self.expt_cfg['cpmg']:
                        sequencer.append('charge%s' % qubit_id,
                                 Gauss(max_amp=self.pulse_info[qubit_id]['pi_ef_amp'],
                                       sigma_len=self.pulse_info[qubit_id]['pi_ef_len'], cutoff_sigma=2,
                                       freq=self.qubit_ef_freq[qubit_id], phase=0.5*np.pi, plot=False))
                    sequencer.append('charge%s' % qubit_id, Idle(time=ramsey_len/(float(2*self.expt_cfg['echo_times']))))
                sequencer.append('charge%s' % qubit_id,
                                 Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_ef_amp'], sigma_len=self.pulse_info[qubit_id]['half_pi_ef_len'],
                   cutoff_sigma=2, freq=self.qubit_ef_freq[qubit_id], phase=2*np.pi*ramsey_len*self.expt_cfg['ramsey_freq'], plot=False))
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def ramsey(self, sequencer):
        # ramsey sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s' % qubit_id, self.qubit_half_pi[qubit_id])
                sequencer.append('charge%s' % qubit_id, Idle(time=ramsey_len))
                sequencer.append('charge%s' % qubit_id,
                                 Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_amp'],
                                       sigma_len=self.pulse_info[qubit_id]['half_pi_len'], cutoff_sigma=2,
                                       freq=self.qubit_freq[qubit_id], phase=2*np.pi*ramsey_len*self.expt_cfg['ramsey_freq'], plot=False))
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def echo(self, sequencer):
        # ramsey sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                sequencer.append('charge%s' % qubit_id, self.qubit_half_pi[qubit_id])
                for echo_id in range(self.expt_cfg['echo_times']):
                    sequencer.append('charge%s' % qubit_id, Idle(time=ramsey_len/(float(2*self.expt_cfg['echo_times']))))
                    if self.expt_cfg['cp']:
                        sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                    elif self.expt_cfg['cpmg']:
                        sequencer.append('charge%s' % qubit_id,
                                 Gauss(max_amp=self.pulse_info[qubit_id]['pi_amp'],
                                       sigma_len=self.pulse_info[qubit_id]['pi_len'], cutoff_sigma=2,
                                       freq=self.qubit_freq[qubit_id], phase=0.5*np.pi, plot=False))
                    sequencer.append('charge%s' % qubit_id, Idle(time=ramsey_len/(float(2*self.expt_cfg['echo_times']))))
                sequencer.append('charge%s' % qubit_id,
                                 Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_amp'],
                                       sigma_len=self.pulse_info[qubit_id]['half_pi_len'], cutoff_sigma=2,
                                       freq=self.qubit_freq[qubit_id], phase=2*np.pi*ramsey_len*self.expt_cfg['ramsey_freq'], plot=False))
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def communication_rabi(self, sequencer):
        # mm rabi sequences

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                if qubit_id in self.expt_cfg['pi_pulse']:
                    sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                sequencer.append('flux%s'%qubit_id,
                                 Square(max_amp=self.communication[qubit_id]['pi_amp'], flat_len=rabi_len,
                                        ramp_sigma_len=5, cutoff_sigma=2,
                                        freq=self.communication[qubit_id]['freq'], phase=0,
                                        plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def photon_transfer(self, sequencer):
        # mm rabi sequences

        for rabi_len in np.arange(self.expt_cfg['receiver_len_start'], self.expt_cfg['receiver_len_stop'], self.expt_cfg['receiver_len_step']):
            sequencer.new_sequence(self)

            sender_id = self.expt_cfg['sender_id']
            receiver_id = self.expt_cfg['receiver_id']

            sequencer.append('charge%s' % sender_id, self.qubit_pi[sender_id])
            sequencer.sync_channels_time(['charge%s' % sender_id, 'flux%s' % sender_id, 'flux%s' % receiver_id])
            sequencer.append('flux%s'%sender_id,
                             Square(max_amp=self.communication[sender_id]['pi_amp'],
                                    flat_len=rabi_len,
                                    ramp_sigma_len=5, cutoff_sigma=2,
                                    freq=self.communication[sender_id]['freq'], phase=0,
                                    plot=False))

            sequencer.append('flux%s'%receiver_id,
                             Square(max_amp=self.communication[receiver_id]['pi_amp'], flat_len=rabi_len,
                                    ramp_sigma_len=5, cutoff_sigma=2,
                                    freq=self.communication[receiver_id]['freq'], phase=0,
                                    plot=False))


            self.readout(sequencer, self.expt_cfg.get('on_qubits',["1","2"]))

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def multimode_rabi(self, sequencer):
        # mm rabi sequences

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                mm_id = self.expt_cfg['on_mms'][qubit_id]
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                sequencer.append('flux%s'%qubit_id,
                                 Square(max_amp=self.expt_cfg['amp'], flat_len=rabi_len, ramp_sigma_len=5, cutoff_sigma=2, freq=self.multimodes[qubit_id]['freq'][mm_id], phase=0,
                                        plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def multimode_t1(self, sequencer):
        # multimode t1 sequences

        for t1_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                mm_id = self.expt_cfg['on_mms'][qubit_id]
                sequencer.append('charge%s' % qubit_id, self.qubit_pi[qubit_id])
                sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                sequencer.append('flux%s'%qubit_id,
                                 Square(max_amp=self.multimodes[qubit_id]['pi_amp'][mm_id], flat_len=self.multimodes[qubit_id]['pi_len'][mm_id], ramp_sigma_len=5, cutoff_sigma=2, freq=self.multimodes[qubit_id]['freq'][mm_id], phase=0,
                                        plot=False))
                sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                sequencer.append('charge%s' % qubit_id, Idle(time=t1_len))
                sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                sequencer.append('flux%s'%qubit_id,
                                 Square(max_amp=self.multimodes[qubit_id]['pi_amp'][mm_id], flat_len=self.multimodes[qubit_id]['pi_len'][mm_id], ramp_sigma_len=5, cutoff_sigma=2, freq=self.multimodes[qubit_id]['freq'][mm_id], phase=0,
                                        plot=False))
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def multimode_ramsey(self, sequencer):
        # mm rabi sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                mm_id = self.expt_cfg['on_mms'][qubit_id]
                sequencer.append('charge%s' % qubit_id, self.qubit_half_pi[qubit_id])
                sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                sequencer.append('flux%s'%qubit_id,
                                 Square(max_amp=self.multimodes[qubit_id]['pi_amp'][mm_id], flat_len=self.multimodes[qubit_id]['pi_len'][mm_id], ramp_sigma_len=5, cutoff_sigma=2, freq=self.multimodes[qubit_id]['freq'][mm_id], phase=0,
                                        plot=False))
                sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                sequencer.append('charge%s' % qubit_id, Idle(time=ramsey_len))
                sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                sequencer.append('flux%s'%qubit_id,
                                 Square(max_amp=self.multimodes[qubit_id]['pi_amp'][mm_id], flat_len=self.multimodes[qubit_id]['pi_len'][mm_id], ramp_sigma_len=5, cutoff_sigma=2, freq=self.multimodes[qubit_id]['freq'][mm_id], phase=0*2*np.pi*ramsey_len*(self.expt_cfg['ramsey_freq']+self.quantum_device_cfg['multimodes']['dc_offset'][mm_id]),
                                        plot=False))
                sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                sequencer.append('charge%s' % qubit_id, self.qubit_half_pi[qubit_id])
            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def sideband_dc_offset(self, sequencer):
        # sideband dc offset sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)

            for qubit_id in self.expt_cfg['on_qubits']:
                mm_id = self.expt_cfg['on_mms'][qubit_id]
                if self.expt_cfg['ge']:

                    sequencer.append('charge%s' % qubit_id, self.qubit_half_pi[qubit_id])
                    sequencer.append('charge%s' % qubit_id, self.qubit_ef_pi[qubit_id])
                    sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                    sequencer.append('flux%s'%qubit_id,
                                     Square(max_amp=self.expt_cfg['amp'], flat_len=ramsey_len, ramp_sigma_len=5, cutoff_sigma=2, freq=self.multimodes[qubit_id]['freq'][mm_id], phase=0,
                                            plot=False))
                    sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                    sequencer.append('charge%s' % qubit_id, self.qubit_ef_pi[qubit_id])
                    sequencer.append('charge%s' % qubit_id,
                                     Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_amp'], sigma_len=self.pulse_info[qubit_id]['half_pi_len'],
                   cutoff_sigma=2, freq=self.qubit_freq[qubit_id], phase=2*np.pi*ramsey_len*(self.expt_cfg['ramsey_freq']+self.quantum_device_cfg['multimodes']['dc_offset'][mm_id]), plot=False))

                else:

                    sequencer.append('charge%s' % qubit_id, self.qubit_half_pi[qubit_id])
                    sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                    sequencer.append('flux%s'%qubit_id,
                                     Square(max_amp=self.expt_cfg['amp'], flat_len=ramsey_len, ramp_sigma_len=5, cutoff_sigma=2, freq=self.multimodes[qubit_id]['freq'][mm_id], phase=0,
                                            plot=False))
                    sequencer.sync_channels_time(['charge%s' % qubit_id, 'flux%s' % qubit_id])
                    sequencer.append('charge%s' % qubit_id,
                                     Gauss(max_amp=self.pulse_info[qubit_id]['half_pi_amp'], sigma_len=self.pulse_info[qubit_id]['half_pi_len'],
                   cutoff_sigma=2, freq=self.qubit_freq[qubit_id], phase=2*np.pi*ramsey_len*(self.expt_cfg['ramsey_freq']+self.quantum_device_cfg['multimodes']['dc_offset'][mm_id]), plot=False))

            self.readout(sequencer, self.expt_cfg['on_qubits'])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=False)

    def alazar_test(self, sequencer):
        # drag_rabi sequences

        freq_ge = 4.5  # GHz
        alpha = - 0.125  # GHz

        freq_lambda = (freq_ge + alpha) / freq_ge
        optimal_beta = freq_lambda ** 2 / (4 * alpha)

        for rabi_len in np.arange(0, 50, 5):
            sequencer.new_sequence(self)

            self.readout(sequencer)
            # sequencer.append('charge1', Idle(time=100))
            sequencer.append('charge1',
                             Gauss(max_amp=0.5, sigma_len=rabi_len, cutoff_sigma=2, freq=self.qubit_freq, phase=0,
                                   plot=False))

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)


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