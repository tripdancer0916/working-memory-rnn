"""generating input and target"""

import numpy as np
import torch.utils.data as data


class FreqDataset(data.Dataset):
    def __init__(
            self,
            time_length,
            time_scale,
            freq_min,
            freq_max,
            min_interval,
            signal_length,
            variable_signal_length,
            sigma_in,
            delay_variable):
        self.time_length = time_length
        self.time_scale = time_scale
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.min_interval = min_interval
        self.signal_length = signal_length
        self.variable_signal_length = variable_signal_length
        self.sigma_in = sigma_in
        self.delay_variable = delay_variable

    def __len__(self):
        return 200

    def __getitem__(self, item):
        # input signal
        signal = np.zeros(self.time_length + 1)
        first_signal_timing = 0
        vs = np.random.randint(-self.variable_signal_length, self.variable_signal_length + 1)
        first_signal_length = self.signal_length + vs
        v = np.random.randint(-self.delay_variable, self.delay_variable + 1)
        second_signal_timing = -self.signal_length + vs + v + self.time_length
        second_signal_length = self.signal_length - v - vs

        first_signal_freq = np.random.rand() * (self.freq_max - self.freq_min) + \
            self.freq_min
        while True:
            second_signal_freq = np.random.rand() * (self.freq_max - self.freq_min) + \
                                 self.freq_min
            if abs(second_signal_freq - first_signal_freq) > self.min_interval:
                break

        # first signal
        t = np.arange(0, first_signal_length * self.time_scale, self.time_scale)
        if len(t) != first_signal_length:
            t = t[:-1]
        phase_shift = np.random.rand() * np.pi
        first_signal = np.sin(first_signal_freq * t + phase_shift) + \
            np.random.normal(0, self.sigma_in, first_signal_length)
        signal[first_signal_timing: first_signal_timing +
               first_signal_length] = first_signal

        # second signal
        t = np.arange(0, second_signal_length * self.time_scale, self.time_scale)
        if len(t) != second_signal_length:
            t = t[:-1]
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
