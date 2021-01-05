import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.append('../')

from model import RecurrentNeuralNetwork

plt.style.use('seaborn-darkgrid')
import seaborn as sns

sns.set_palette('Dark2')


def romo_signal(delta, N, signal_length, sigma_in):
    signals = np.zeros([N, 61, 1])
    freq_range = 4 - abs(delta)
    for i in range(N):
        first_signal_timing = 0
        second_signal_timing = 60 - signal_length
        first_signal_freq = np.random.rand() * freq_range + max(1, 1 - delta)
        second_signal_freq = first_signal_freq + delta
        t = np.arange(0, signal_length / 4, 0.25)
        phase_shift_1 = np.random.rand() * np.pi
        first_signal = np.sin(first_signal_freq * t + phase_shift_1) + np.random.normal(0, sigma_in, signal_length)
        phase_shift_2 = np.random.rand() * np.pi
        second_signal = np.sin(second_signal_freq * t + phase_shift_2) + np.random.normal(0, sigma_in, signal_length)

        signals[i, first_signal_timing: first_signal_timing + signal_length, 0] = first_signal
        signals[i, second_signal_timing: second_signal_timing + signal_length, 0] = second_signal

    return signals


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

    trial_num = 1000
    signal_length = 15
    sigma_in = 0.05
    deltas = np.arange(-1, 1, 0.05)
    score = np.zeros(deltas.shape[0])
    delta_index = 0
    acc_list = np.zeros(deltas.shape[0])
    for delta in deltas:
        output_list = np.zeros(trial_num)
        input_signal = romo_signal(delta, trial_num, signal_length, sigma_in)
        input_signal_split = np.split(input_signal, 4)
        for i in range(4):
            hidden = torch.zeros(250, model.n_hid)
            hidden = hidden.to(device)
            inputs = torch.from_numpy(input_signal_split[i]).float()
            inputs = inputs.to(device)
            _, outputs, _, _ = model(inputs, hidden)
            outputs_np = outputs.cpu().detach().numpy()
            output_list[i * 250: (i + 1) * 250] = np.argmax(outputs_np[:, -1], axis=1)
        score[delta_index] = np.mean(output_list)
        if delta > 0:
            acc_list[delta_index] = np.mean(output_list)
        else:
            acc_list[delta_index] = 1 - np.mean(output_list)
        if delta_index % 5 == 0:
            print(f'{delta:.3f}', np.mean(output_list))
        delta_index += 1
    os.makedirs('results', exist_ok=True)
    os.makedirs(f'results/{model_name}', exist_ok=True)

    print(np.mean(acc_list))

    np.save(f'results/{model_name}/score.npy', np.array(score))
    np.save(f'results/{model_name}/acc_list.npy', np.array(acc_list))
    plt.plot(deltas, score, color='orange')
    plt.xlabel(r'$\omega_2-\omega_1$', fontsize=16)
    plt.savefig(f'results/{model_name}/psychometric_curve.png', dpi=200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    print(args)
    main(args.config_path)
