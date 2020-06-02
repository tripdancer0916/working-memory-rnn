"""generating input and target"""

import numpy as np
import torch.utils.data as data


def gauss(x, amp=2, mu=0, sigma=5):
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) * (1 / np.sqrt(2 * np.pi * sigma ** 2))


class SPPSDataset(data.Dataset):
    def __init__(
            self,
            time_length,
            n_in,
            num_positions,
            prob_amp,
            signal_length,
            sigma_in):
        self.time_length = time_length
        self.n_in = n_in
        self.num_positions = num_positions
        self.prob_amp = prob_amp
        self.signal_length = signal_length
        self.sigma_in = sigma_in

    def __len__(self):
        return 200

    def __getitem__(self, item):
        # input signal
        signal = np.zeros((self.time_length + 1, self.n_in))
        first_signal_timing = 0
        second_signal_timing = self.time_length - self.signal_length

        first_signal_position = np.random.randint(self.num_positions)
        first_signal_mu = first_signal_position * self.n_in / (self.num_positions - 1)
        first_prob = gauss(np.arange(0, self.n_in), mu=first_signal_mu, sigma=self.sigma_in)
        first_signal = np.random.binomial(1, p=first_prob, size=(self.signal_length, self.n_in))
        signal[first_signal_timing: first_signal_timing + self.signal_length] = first_signal

        second_signal_position = np.random.randint(self.num_positions)
        second_signal_mu = second_signal_position * self.n_in / (self.num_positions - 1)
        second_prob = gauss(np.arange(0, self.n_in), mu=second_signal_mu, sigma=self.sigma_in)
        second_signal = np.random.binomial(1, p=second_prob, size=(self.signal_length, self.n_in))
        signal[second_signal_timing: second_signal_timing + self.signal_length] = second_signal

        # target
        if first_signal_position > second_signal_position:
            target = np.array(0)
        elif first_signal_position < second_signal_position:
            target = np.array(1)
        else:
            target = np.array(2)

        return signal, target


"""
class SPPSDatasetVariableDelay(data.Dataset):
    def __init__(
            self,
            time_length,
            freq_min,
            freq_max,
            min_interval,
            signal_length,
            sigma_in,
            delay_variable):
        self.time_length = time_length
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.min_interval = min_interval
        self.signal_length = signal_length
        self.sigma_in = sigma_in
        self.delay_variable = delay_variable

    def __len__(self):
        return 200

    def __getitem__(self, item):
        # input signal
        signal = np.zeros(self.time_length + 1)
        first_signal_timing = 0
        v = np.random.randint(-self.delay_variable, self.delay_variable + 1)
        second_signal_timing = self.time_length - self.signal_length + v
        second_signal_length = self.signal_length - v

        first_signal_freq = np.random.rand() * (self.freq_max - self.freq_min) + \
                            self.freq_min
        while True:
            second_signal_freq = np.random.rand() * (self.freq_max - self.freq_min) + \
                                 self.freq_min
            if abs(second_signal_freq - first_signal_freq) > self.min_interval:
                break

        # first signal
        t = np.arange(0, self.signal_length / 4, 0.25)
        phase_shift = np.random.rand() * np.pi
        first_signal = np.sin(first_signal_freq * t + phase_shift) + \
                       np.random.normal(0, self.sigma_in, self.signal_length)
        signal[first_signal_timing: first_signal_timing +
                                    self.signal_length] = first_signal

        # second signal
        t = np.arange(0, second_signal_length / 4, 0.25)
        phase_shift = np.random.rand() * np.pi
        second_signal = np.sin(second_signal_freq * t + phase_shift) + \
                        np.random.normal(0, self.sigma_in, second_signal_length)
        signal[second_signal_timing:second_signal_timing +
                                    second_signal_length] = second_signal

        # target
        if first_signal_freq > second_signal_freq:
            target = np.array(0)
        else:
            target = np.array(1)

        signal = np.expand_dims(signal, axis=1)

        return signal, target
"""
