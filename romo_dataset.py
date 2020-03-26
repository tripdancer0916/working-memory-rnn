"""generating input and target"""

import numpy as np
import torch.utils.data as data


class RomoDataset(data.Dataset):
    def __init__(self, time_length, freq_min, freq_max, min_interval, signal_length):
        self.time_length = time_length
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.min_interval = min_interval
        self.signal_length = signal_length

    def __len__(self):
        return 200

    def __getitem__(self, item):
        # input signal
        signal = np.zeros(self.time_length + 1)
        first_signal_timing = 10
        second_signal_timing = 50

        first_signal_freq = np.random.rand() * (self.freq_max - self.freq_min) + self.freq_min
        while True:
            second_signal_freq = np.random.rand() * (self.freq_max - self.freq_min) + self.freq_min
            if abs(second_signal_freq - first_signal_freq) > self.min_interval:
                break
        t = np.arange(0, self.signal_length)
        signal[first_signal_timing: first_signal_timing+self.signal_length] = np.sin(first_signal_freq * t)
        signal[second_signal_timing:second_signal_timing+self.signal_length] = np.sin(second_signal_freq * t)

        # target
        if first_signal_freq > second_signal_freq:
            target = np.array(0)
        else:
            target = np.array(1)

        signal = np.expand_dims(signal, axis=1)
        # target = np.expand_dims(target, axis=1)

        return signal, target
