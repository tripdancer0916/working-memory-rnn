import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import torch
import yaml

sys.path.append('../')

from model import RecurrentNeuralNetwork

plt.style.use('seaborn-darkgrid')
import seaborn as sns

sns.set_palette('Dark2')


def tangent_angle(u: np.ndarray, v: np.ndarray):
    i = np.inner(u, v)
    n = LA.norm(u) * LA.norm(v)
    c = i / n
    return np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))


def romo_signal(batch_size, signal_length, sigma_in, time_length=400, alpha=0.25):
    signals = np.zeros([batch_size, time_length + 1, 1])
    omega_1_list = [1] * 100
    for omega_1 in np.arange(1.25, 5.01, 0.25):
        omega_1_list += [omega_1] * 100
    # print(len(omega_1_list))
    phase_shift_1 = np.random.rand() * np.pi
    for i in range(batch_size):
        # phase_shift_1 = np.random.rand() * np.pi
        omega_1 = omega_1_list[i]
        first_signal_timing = 0
        t = np.arange(0, signal_length * alpha, alpha)

        first_signal = np.sin(omega_1 * t + phase_shift_1) + np.random.normal(0, sigma_in, signal_length)
        signals[i, first_signal_timing: first_signal_timing + signal_length, 0] = first_signal

        signals[i, first_signal_timing + signal_length:, 0] = \
            np.random.normal(
                0,
                sigma_in,
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

    trial_num = 1700
    neural_dynamics = np.zeros((trial_num, 1001, model.n_hid))
    outputs_np = np.zeros(trial_num)
    input_signal, omega_1_list = romo_signal(trial_num, signal_length=15, sigma_in=0.05, time_length=1000)
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

    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{model_name}', exist_ok=True)

    angle_list = []
    diff_omega_list = []
    for i in range(16):
        for j in range(i+1, 16):
            distance = np.linalg.norm(
                np.mean(neural_dynamics[100 * i:100 * (i + 1), 15], axis=0) -
                np.mean(neural_dynamics[100 * j:100 * (j + 1), 15], axis=0),
            )
            angle_list.append(distance)
            diff_omega_list.append(abs(i-j) * 0.25)

    plt.figure(constrained_layout=True)
    plt.scatter(
        diff_omega_list,
        angle_list,
    )
    plt.xlabel(r'$|\omega_i - \omega_j|$', fontsize=16)
    plt.ylabel(r'$\theta$', fontsize=16)

    plt.savefig(f'results/{model_name}/coding_distance_Ts.png', dpi=200)
    np.save(f'results/{model_name}/active_norm.npy', np.array(angle_list))

    angle_list = []
    diff_omega_list = []
    for i in range(16):
        for j in range(i + 1, 16):
            distance = np.linalg.norm(
                np.mean(neural_dynamics[100 * i:100 * (i + 1), 45], axis=0) -
                np.mean(neural_dynamics[100 * j:100 * (j + 1), 45], axis=0),
            )
            angle_list.append(distance)
            diff_omega_list.append(abs(i - j) * 0.25)

    plt.figure(constrained_layout=True)
    plt.scatter(
        diff_omega_list,
        angle_list,
    )
    plt.xlabel(r'$|\omega_i - \omega_j|$', fontsize=16)
    plt.ylabel(r'$\Delta {\bf x}$', fontsize=16)

    plt.savefig(f'results/{model_name}/coding_distance_Tf.png', dpi=200)
    # np.save(f'results/{model_name}/active_norm.npy', np.array(angle_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    print(args)
    main(args.config_path)
