import visdom
import numpy as np

from .pulse_classes import Gauss, Idle, Ones, Square


class Sequencer:
    def __init__(self, channels, channels_awg, awg_info, channels_delay):
        self.channels = channels
        self.channels_awg = channels_awg
        self.awg_info = awg_info
        self.channels_delay = channels_delay

        channels_awg_info = {}

        for channel in channels:
            channels_awg_info[channel] = awg_info[channels_awg[channel]]

        self.channels_awg_info = channels_awg_info

        self.pulse_array_list = {}
        self.multiple_sequences = []

    def new_sequence(self):
        self.pulse_array_list = {}
        for channel in self.channels:
            idle = Idle(time=10, dt=self.channels_awg_info[channel]['dt'])
            idle.generate_pulse_array()
            self.pulse_array_list[channel] = [idle.pulse_array]

    def append(self, channel, pulse):
        pulse.generate_pulse_array(t0=self.get_time(channel), dt=self.channels_awg_info[channel]['dt'])
        pulse_array = pulse.pulse_array
        self.pulse_array_list[channel].append(pulse_array)

    def get_time(self, channel):
        return self.channels_awg_info[channel]['dt'] * len(np.concatenate(self.pulse_array_list[channel])) + \
               self.channels_awg_info[channel]['time_delay']

    def append_idle_to_time(self, channel, time):
        current_time = self.get_time(channel)
        extra_time = time - current_time

        if extra_time > 0:
            self.append(channel, Idle(time=extra_time, dt=self.channels_awg_info[channel]['dt']))

    def sync_channels_time(self, channels):

        max_time = 0

        for channel in channels:
            channel_time = self.get_time(channel)
            if channel_time > max_time:
                max_time = channel_time

        for channel in channels:
            self.append_idle_to_time(channel, max_time)

    def rectify_pulse_len(self, sequence):
        for channel in self.channels:
            channel_len = len(sequence[channel])

            min_samples = self.channels_awg_info[channel]['min_samples']

            if channel_len < min_samples:
                sequence[channel] = np.pad(sequence[channel], (0, min_samples - channel_len), 'constant',
                                           constant_values=(0, 0))

            channel_len = len(sequence[channel])
            min_increment = self.channels_awg_info[channel]['min_increment']

            if channel_len % min_increment != 0:
                sequence[channel] = np.pad(sequence[channel], (0, min_increment - (channel_len % min_increment)),
                                           'constant', constant_values=(0, 0))

        return sequence

    def equalize_sequences(self):
        awg_max_len = {}
        awg_list = set(self.channels_awg.values())
        for awg in awg_list:
            awg_max_len[awg] = 0

        for sequence in self.multiple_sequences:
            for channel in self.channels:
                channel_len = len(sequence[channel])
                if channel_len > awg_max_len[self.channels_awg[channel]]:
                    awg_max_len[self.channels_awg[channel]] = channel_len

        for sequence in self.multiple_sequences:
            for channel in self.channels:
                channel_len = len(sequence[channel])
                if channel_len < awg_max_len[self.channels_awg[channel]]:
                    sequence[channel] = np.pad(sequence[channel],
                                               (0, awg_max_len[self.channels_awg[channel]] - channel_len), 'constant',
                                               constant_values=(0, 0))

    def delay_channels(self, channels_delay):
        for sequence in self.multiple_sequences:
            for channel, delay in channels_delay.items():
                delay_num = int(delay / self.channels_awg_info[channel]['dt'])
                sequence[channel] = np.roll(sequence[channel], delay_num)

    def get_sequence(self):
        sequence = {}
        for channel in self.channels:
            sequence[channel] = np.concatenate(self.pulse_array_list[channel])

        return sequence

    def end_sequence(self):
        sequence = self.get_sequence()
        sequence = self.rectify_pulse_len(sequence)

        self.multiple_sequences.append(sequence)

    def complete(self, plot=True):
        self.equalize_sequences()
        self.delay_channels(self.channels_delay)

        if plot:
            self.plot_sequences()

        return self.multiple_sequences

    def plot_sequences(self):

        sequence_id = 0

        for sequence in self.multiple_sequences:

            sequence_id += 1

            vis = visdom.Visdom()
            win = vis.line(
                X=np.arange(0, 1),
                Y=np.arange(0, 1),
                opts=dict(
                    legend=[self.channels[0]], title='seq %d' % sequence_id, xlabel='Time (ns)'))

            kk = 0
            for channel in self.channels:
                sequence_array = sequence[channel]
                vis.updateTrace(
                    X=np.arange(0, len(sequence_array)) * self.channels_awg_info[channel]['dt'] +
                      self.channels_awg_info[channel]['time_delay'],
                    Y=sequence_array + 2 * (len(self.channels) - kk),
                    win=win, name=channel, append=False)

                kk += 1


def testing_function():
    vis = visdom.Visdom()
    vis.close()

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

    sequencer = Sequencer(channels, channels_awg, awg_info, channels_delay)

    # sequence 1
    sequencer.new_sequence()

    sequencer.append('charge1', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=1, phase=0, plot=False))
    sequencer.append('charge1', Idle(time=10))
    sequencer.append('charge1', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2, phase=np.pi))

    sequencer.append('flux1', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2, phase=np.pi))

    sequencer.append('charge2', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2, phase=np.pi))

    sequencer.append('flux2', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2, phase=np.pi))

    sequencer.end_sequence()

    # sequence 2
    sequencer.new_sequence()

    sequencer.append('charge1', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=1, phase=0, plot=False))
    sequencer.append('charge1', Idle(time=10))
    sequencer.append('charge1', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2, phase=np.pi))

    sequencer.append('flux1', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2, phase=np.pi))

    sequencer.append('charge2', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2, phase=np.pi))

    sequencer.append('flux2', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2, phase=np.pi))

    sequencer.end_sequence()

    # sequence 3
    sequencer.new_sequence()

    sequencer.append('m8195a_trig', Ones(time=100))

    sequencer.append('charge1', Gauss(max_amp=0.5, sigma_len=10, cutoff_sigma=2, freq=4.524, phase=0, plot=False))
    sequencer.append('charge1', Idle(time=10))
    sequencer.append('charge1', Gauss(max_amp=0.5, sigma_len=10, cutoff_sigma=2, freq=4.524, phase=np.pi))

    sequencer.append('flux1', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2.1, phase=np.pi))

    sequencer.append('charge2', Gauss(max_amp=0.5, sigma_len=30.2, cutoff_sigma=2, freq=4.524, phase=0))

    sequencer.append('flux2', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2.1, phase=np.pi))

    sequencer.sync_channels_time(['charge1', 'flux1', 'charge2'])

    sequencer.append('charge1', Gauss(max_amp=0.5, sigma_len=10, cutoff_sigma=2, freq=4.5, phase=0, plot=False))
    sequencer.append('charge1', Idle(time=10))
    sequencer.append('charge1', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=4.524, phase=np.pi))

    sequencer.append('flux1', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2.1, phase=np.pi))

    sequencer.append('charge2', Gauss(max_amp=0.5, sigma_len=20.3, cutoff_sigma=2, freq=4.524, phase=0))

    sequencer.sync_channels_time(channels)

    sequencer.append('flux1', Gauss(max_amp=0.5, sigma_len=15, cutoff_sigma=2, freq=2.1, phase=np.pi))
    sequencer.append('flux2', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2.1, phase=np.pi))
    sequencer.append('flux2', Gauss(max_amp=0.5, sigma_len=10, cutoff_sigma=2, freq=2.1, phase=np.pi))

    sequencer.sync_channels_time(['charge1', 'charge2', 'flux2'])
    sequencer.append('charge1', Gauss(max_amp=0.5, sigma_len=15, cutoff_sigma=2, freq=4.524, phase=np.pi))
    sequencer.append('charge2', Gauss(max_amp=0.5, sigma_len=30, cutoff_sigma=2, freq=4.524, phase=np.pi))
    sequencer.append('flux2', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2.1, phase=np.pi))

    sequencer.sync_channels_time(['flux1', 'flux2'])

    sequencer.append('flux1', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2.1, phase=np.pi))

    sequencer.sync_channels_time(channels)

    sequencer.append('hetero1_I', Square(max_amp=0.5, flat_len= 200, ramp_sigma_len=20, cutoff_sigma=2, freq=0.2, phase=0, plot=False))
    sequencer.append('hetero1_Q', Square(max_amp=0.5, flat_len= 200, ramp_sigma_len=20, cutoff_sigma=2, freq=0.2, phase=np.pi / 2))

    sequencer.append('hetero2_I', Square(max_amp=0.5, flat_len= 200, ramp_sigma_len=20, cutoff_sigma=2, freq=0.2, phase=0, plot=False))
    sequencer.append('hetero2_Q', Square(max_amp=0.5, flat_len= 200, ramp_sigma_len=20, cutoff_sigma=2, freq=0.2, phase=np.pi / 2))

    sequencer.append('alazar_trig', Ones(time=100))
    sequencer.append('readout1_trig', Ones(time=250))
    sequencer.append('readout2_trig', Ones(time=250))

    sequencer.end_sequence()

    sequencer.complete(plot=True)


if __name__ == "__main__":
    testing_function()




