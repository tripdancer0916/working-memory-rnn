"""training models"""

import argparse
import os
import shutil
import sys

import numpy as np
import torch
import torch.optim as optim
import yaml

sys.path.append('../')

from torch.autograd import Variable
import torch.utils.data as data

from model import RecurrentNeuralNetwork


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

    def getitem(self, delay_period):
        time_length = 2 * self.signal_length + delay_period
        # input signal
        signal = np.zeros(time_length + 1)
        first_signal_timing = 0
        second_signal_timing = self.signal_length + delay_period

        first_signal_freq = np.random.rand() * (self.freq_max - self.freq_min) + self.freq_min
        while True:
            second_signal_freq = np.random.rand() * (self.freq_max - self.freq_min) + self.freq_min
            if abs(second_signal_freq - first_signal_freq) > self.min_interval:
                break

        # first signal
        vs = np.random.randint(-self.variable_signal_length, self.variable_signal_length + 1)
        first_signal_length = self.signal_length + vs
        t = np.arange(0, first_signal_length * self.time_scale, self.time_scale)
        if len(t) != first_signal_length:
            t = t[:-1]
        phase_shift = np.random.rand() * np.pi
        first_signal = np.sin(first_signal_freq * t + phase_shift) + \
            np.random.normal(0, self.sigma_in, first_signal_length)
        signal[first_signal_timing: first_signal_timing + first_signal_length] = first_signal

        # second signal
        t = np.arange(0, self.signal_length * self.time_scale, self.time_scale)
        if len(t) != self.signal_length:
            t = t[:-1]
        phase_shift = np.random.rand() * np.pi
        second_signal = np.sin(second_signal_freq * t + phase_shift) + \
            np.random.normal(0, self.sigma_in, self.signal_length)
        signal[second_signal_timing:second_signal_timing + self.signal_length] = second_signal

        # target
        if first_signal_freq > second_signal_freq:
            target = np.array(0)
        else:
            target = np.array(1)

        signal = np.expand_dims(signal, axis=1)

        return signal, target


def main(config_path):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = os.path.splitext(os.path.basename(config_path))[0]

    # save path
    os.makedirs('trained_model', exist_ok=True)
    os.makedirs('trained_model/freq_20210203', exist_ok=True)
    save_path = f'trained_model/freq_20210203/{model_name}'
    os.makedirs(save_path, exist_ok=True)

    # copy config file
    shutil.copyfile(config_path, os.path.join(save_path, os.path.basename(config_path)))

    use_cuda = cfg['MACHINE']['CUDA'] and torch.cuda.is_available()
    torch.manual_seed(cfg['MACHINE']['SEED'])
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

    if 'ALPHA' not in cfg['MODEL'].keys():
        cfg['MODEL']['ALPHA'] = 0.25

    model = RecurrentNeuralNetwork(n_in=1, n_out=2, n_hid=cfg['MODEL']['SIZE'], device=device,
                                   alpha_time_scale=cfg['MODEL']['ALPHA'], beta_time_scale=cfg['MODEL']['BETA'],
                                   activation=cfg['MODEL']['ACTIVATION'],
                                   sigma_neu=cfg['MODEL']['SIGMA_NEU'],
                                   sigma_syn=cfg['MODEL']['SIGMA_SYN'],
                                   use_bias=cfg['MODEL']['USE_BIAS'],
                                   anti_hebbian=cfg['MODEL']['ANTI_HEBB']).to(device)
    # model_path = f'trained_model/freq/20210130_2_1/epoch_2400.pth'
    # model.load_state_dict(torch.load(model_path, map_location=device))

    train_dataset = FreqDataset(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                                time_scale=cfg['MODEL']['ALPHA'],
                                freq_min=cfg['DATALOADER']['FREQ_MIN'],
                                freq_max=cfg['DATALOADER']['FREQ_MAX'],
                                min_interval=cfg['DATALOADER']['MIN_INTERVAL'],
                                signal_length=cfg['DATALOADER']['SIGNAL_LENGTH'],
                                variable_signal_length=cfg['DATALOADER']['VARIABLE_SIGNAL_LENGTH'],
                                sigma_in=cfg['DATALOADER']['SIGMA_IN'],
                                delay_variable=cfg['DATALOADER']['VARIABLE_DELAY'])

    print(model)
    print('Epoch Loss Acc')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg['TRAIN']['LR'], weight_decay=cfg['TRAIN']['WEIGHT_DECAY'])
    correct = 0
    num_data = 0
    base_delay_period = cfg['DATALOADER']['TIME_LENGTH'] - 2*cfg['DATALOADER']['SIGNAL_LENGTH']
    vs = cfg['DATALOADER']['VARIABLE_DELAY']
    batch_size = 50
    for epoch in range(cfg['TRAIN']['NUM_EPOCH'] + 1):
        model.train()
        for i in range(4):
            delay_period = np.random.randint(base_delay_period - vs, base_delay_period + vs + 1)
            time_length = cfg['DATALOADER']['SIGNAL_LENGTH'] * 2 + delay_period
            inputs = np.zeros([batch_size, time_length+1, 1])
            targets = np.zeros(batch_size)
            for j in range(batch_size):
                signal, target = train_dataset.getitem(delay_period)
                inputs[j] = signal
                targets[j] = target

            inputs = torch.from_numpy(inputs)
            targets = torch.from_numpy(targets)
            inputs, targets = inputs.float(), targets.long()
            inputs, targets = Variable(inputs).to(device), Variable(targets).to(device)

            hidden = torch.zeros(batch_size, cfg['MODEL']['SIZE'])
            hidden = hidden.to(device)
            optimizer.zero_grad()
            hidden = hidden.detach()
            hidden_list, output, hidden, new_j = model(inputs, hidden)
            # print(output)

            loss = torch.nn.CrossEntropyLoss()(output[:, -1], targets)
            dummy_zero = torch.zeros([cfg['TRAIN']['BATCHSIZE'],
                                      time_length + 1,
                                      cfg['MODEL']['SIZE']]).float().to(device)
            active_norm = torch.nn.MSELoss()(hidden_list, dummy_zero)

            loss += cfg['TRAIN']['ACTIVATION_LAMBDA'] * active_norm
            loss.backward()
            optimizer.step()
            correct += \
                (np.argmax(output[:, -1].cpu().detach().numpy(), axis=1) == targets.cpu().detach().numpy()).sum().item()
            num_data += targets.cpu().detach().numpy().shape[0]

        if epoch % cfg['TRAIN']['DISPLAY_EPOCH'] == 0:
            acc = correct / num_data
            print(f'{epoch}, {loss.item():.6f}, {acc:.6f}')
            print(active_norm)
            # print('w_hh: ', model.w_hh.weight.cpu().detach().numpy()[:4, :4])
            # print('new_j: ', new_j.cpu().detach().numpy()[0, :4, :4])
            correct = 0
            num_data = 0
        if epoch > 0 and epoch % cfg['TRAIN']['NUM_SAVE_EPOCH'] == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    print(args)
    main(args.config_path)
