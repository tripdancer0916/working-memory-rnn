"""generating input and target"""

import numpy as np
import torch.utils.data as data


class DistDataset(data.Dataset):
    def __init__(
            self,
            time_length,
            sigma_min,
            sigma_max,
            min_interval,
            signal_length,
            freq):
        self.time_length = time_length
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.min_interval = min_interval
        self.signal_length = signal_length
        self.freq = freq

    def __len__(self):
        return 200

    def __getitem__(self, item):
        # input signal
        signal = np.zeros(self.time_length + 1)
        first_signal_timing = 0
        second_signal_timing = self.time_length - self.signal_length

        first_signal_sigma = np.random.rand() * (self.sigma_max - self.sigma_min) + \
            self.sigma_min
        while True:
            second_signal_sigma = np.random.rand() * (self.sigma_max - self.sigma_min) + \
                                  self.sigma_min
            if abs(second_signal_sigma - first_signal_sigma) > self.min_interval:
                break

        # first signal
        t = np.arange(0, self.signal_length / 4, 0.25)
        # phase_shift = np.random.rand() * np.pi
        phase_shift = 0
        first_signal = np.sin(self.freq * t + phase_shift) + \
            np.random.normal(0, first_signal_sigma, self.signal_length)
        signal[first_signal_timing: first_signal_timing +
               self.signal_length] = first_signal

        # second signal
        # phase_shift = np.random.rand() * np.pi
        phase_shift = 0
        second_signal = np.sin(self.freq * t + phase_shift) + \
            np.random.normal(0, second_signal_sigma, self.signal_length)
        signal[second_signal_timing:second_signal_timing +
               self.signal_length] = second_signal

        # target
        if first_signal_sigma > second_signal_sigma:
            target = np.array(0)
        else:
            target = np.array(1)

        signal = np.expand_dims(signal, axis=1)
        # target = np.expand_dims(target, axis=1)

        return signal, target


class DistDatasetVariableDelay(data.Dataset):
    def __init__(
            self,
            time_length,
            sigma_min,
            sigma_max,
            min_interval,
            signal_length,
            freq,
            delay_variable):
        self.time_length = time_length
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.min_interval = min_interval
        self.signal_length = signal_length
        self.freq = freq
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

        first_signal_sigma = np.random.rand() * (self.sigma_max - self.sigma_min) + \
            self.sigma_min
        while True:
            second_signal_sigma = np.random.rand() * (self.sigma_max - self.sigma_min) + \
                self.sigma_min
            if abs(second_signal_sigma - first_signal_sigma) > self.min_interval:
                break

        # first signal
        t = np.arange(0, self.signal_length / 4, 0.25)
        # phase_shift = np.random.rand() * np.pi
        phase_shift = 0
        first_signal = np.sin(self.freq * t + phase_shift) + \
            np.random.normal(0, first_signal_sigma, self.signal_length)
        signal[first_signal_timing: first_signal_timing +
               self.signal_length] = first_signal

        # second signal
        t = np.arange(0, second_signal_length / 4, 0.25)
        # phase_shift = np.random.rand() * np.pi
        phase_shift = 0
        second_signal = np.sin(self.freq * t + phase_shift) + \
            np.random.normal(0, second_signal_sigma, second_signal_length)
        signal[second_signal_timing:second_signal_timing +
               second_signal_length] = second_signal

        # target
        if first_signal_sigma > second_signal_sigma:
            target = np.array(0)
        else:
            target = np.array(1)

        signal = np.expand_dims(signal, axis=1)

        return signal, target
