"""generating input and target"""

import numpy as np
import torch.utils.data as data


class VelocityDataset(data.Dataset):
    def __init__(
            self,
            time_length,
            time_scale,
            a_min,
            a_max,
            min_interval,
            signal_length,
            variable_signal_length,
            sigma_in,
            delay_variable):
        self.time_length = time_length
        self.time_scale = time_scale
        self.a_min = a_min
        self.a_max = a_max
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

        first_signal_a = np.random.rand() * (self.a_max - self.a_min) + self.a_min
        while True:
            second_signal_a = np.random.rand() * (self.a_max - self.a_min) + self.a_min
            if abs(second_signal_a - first_signal_a) > self.min_interval:
                break

        # first signal
        t = np.arange(0, first_signal_length * self.time_scale, self.time_scale)
        if len(t) != first_signal_length:
            t = t[:-1]
        intercept = np.random.rand() * 2 - 1
        first_signal = first_signal_a * t + intercept + np.random.normal(0, self.sigma_in, first_signal_length)
        signal[first_signal_timing: first_signal_timing + first_signal_length] = first_signal

        # second signal
        t = np.arange(0, second_signal_length * self.time_scale, self.time_scale)
        if len(t) != second_signal_length:
            t = t[:-1]
        intercept = np.random.rand() * 2 - 1
        second_signal = second_signal_a * t + intercept + np.random.normal(0, self.sigma_in, second_signal_length)
        signal[second_signal_timing:second_signal_timing + second_signal_length] = second_signal

        # target
        if first_signal_a > second_signal_a:
            target = np.array(0)
        else:
            target = np.array(1)

        signal = np.expand_dims(signal, axis=1)

        return signal, target
