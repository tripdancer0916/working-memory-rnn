import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.append('../')

from model import RecurrentNeuralNetwork

plt.style.use('seaborn-darkgrid')
import seaborn as sns

sns.set_palette('Dark2')


def romo_signal(batch_size, signal_length, sigma_in, time_length=400, alpha=0.25):
    signals = np.zeros([batch_size, time_length + 1, 1])
    omega_1_list = np.random.rand(batch_size) * 4 + 1
    for i in range(batch_size):
        phase_shift_1 = np.random.rand() * np.pi
        omega_1 = omega_1_list[i]
        first_signal_timing = 0
        t = np.arange(0, signal_length * alpha, alpha)

        first_signal = np.sin(omega_1 * t + phase_shift_1) + np.random.normal(0, sigma_in, signal_length)
        signals[i, first_signal_timing: first_signal_timing + signal_length, 0] = first_signal

        signals[i, first_signal_timing + signal_length:, 0] = \
            np.random.normal(
                0,
                # sigma_in,
                0,
                time_length + 1 - signal_length - first_signal_timing,
            )

    return signals, omega_1_list


def main(config_path):
    torch.manual_seed(1)
    device = torch.device('cpu')

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['MODEL']['SIGMA_NEU'] = 0
    model_name = os.path.splitext(os.path.basename(config_path))[0]
    print('model_name: ', model_name)

    model = RecurrentNeuralNetwork(n_in=1, n_out=2, n_hid=cfg['MODEL']['SIZE'], device=device,
                                   alpha_time_scale=0.25, beta_time_scale=cfg['MODEL']['BETA'],
                                   activation=cfg['MODEL']['ACTIVATION'],
                                   sigma_neu=cfg['MODEL']['SIGMA_NEU'],
                                   sigma_syn=cfg['MODEL']['SIGMA_SYN'],
                                   use_bias=cfg['MODEL']['USE_BIAS'],
                                   anti_hebbian=cfg['MODEL']['ANTI_HEBB']).to(device)
    model_path = f'../trained_model/freq/{model_name}/epoch_3000.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    trial_num = 3000
    neural_dynamics = np.zeros((trial_num, 61, model.n_hid))
    outputs_np = np.zeros(trial_num)
    input_signal, omega_1_list = romo_signal(trial_num, signal_length=15, sigma_in=0.05, time_length=60)
    input_signal_split = np.split(input_signal, trial_num // cfg['TRAIN']['BATCHSIZE'])

    for i in range(trial_num // cfg['TRAIN']['BATCHSIZE']):
        hidden = torch.zeros(cfg['TRAIN']['BATCHSIZE'], model.n_hid)
        hidden = hidden.to(device)
        inputs = torch.from_numpy(input_signal_split[i]).float()
        inputs = inputs.to(device)
        hidden_list, outputs, _, _ = model(inputs, hidden)
        hidden_list_np = hidden_list.cpu().detach().numpy()
        outputs_np[i * cfg['TRAIN']['BATCHSIZE']: (i + 1) * cfg['TRAIN']['BATCHSIZE']] = np.argmax(
            outputs.detach().numpy()[:, -1], axis=1)
        neural_dynamics[i * cfg['TRAIN']['BATCHSIZE']: (i + 1) * cfg['TRAIN']['BATCHSIZE']] = hidden_list_np

    time_15_mse = 0
    time_45_mse = 0
    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{model_name}', exist_ok=True)
    for trial in range(50):
        train_data, test_data, train_target, test_target = train_test_split(neural_dynamics, omega_1_list, test_size=0.25)

        clf_coef_norm = []
        timepoint = 45
        """"
        for alpha in [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]:
            clf = Ridge(alpha=alpha)
            clf.fit(train_data[:, timepoint, :], train_target)
        
            clf_coef_norm.append(np.linalg.norm(clf.coef_))
            mse = mean_squared_error(
                clf.predict(test_data[:, timepoint, :]),
                test_target,
            )
            print(alpha, mse)
         """
        clf = Ridge(alpha=0.00001)
        clf.fit(train_data[:, timepoint, :], train_target)

        clf_coef_norm.append(np.linalg.norm(clf.coef_))
        mse = mean_squared_error(
            clf.predict(test_data[:, timepoint, :]),
            test_target,
        )
        print(timepoint, mse)
        time_45_mse += mse

        timepoint = 15
        clf = Ridge(alpha=0.00001)
        clf.fit(train_data[:, timepoint, :], train_target)

        clf_coef_norm.append(np.linalg.norm(clf.coef_))
        mse = mean_squared_error(
            clf.predict(test_data[:, timepoint, :]),
            test_target,
        )
        print(timepoint, mse)
        time_15_mse += mse

    print('average')
    print(f'time: 15, mse: {time_15_mse/50}')
    print(f'time: 45, mse: {time_45_mse/50}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    print(args)
    main(args.config_path)
