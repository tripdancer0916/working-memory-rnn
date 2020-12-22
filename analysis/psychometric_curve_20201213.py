import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.append('../')

from model import RecurrentNeuralNetwork


def romo_signal(delta, N, signal_length, sigma_in, delay, time_length=60):
    signals = np.zeros([N, time_length + 1, 1])
    first_signal_timing = 0
    freq_range = 4 - abs(delta)
    second_signal_timing = -signal_length + delay + time_length
    second_signal_length = signal_length - delay
    for i in range(N):
        first_signal_freq = np.random.rand() * freq_range + max(1, 1 - delta)
        second_signal_freq = first_signal_freq + delta
        t = np.arange(0, signal_length / 4, 0.25)
        phase_shift_1 = np.random.rand() * np.pi
        first_signal = np.sin(first_signal_freq * t + phase_shift_1) + np.random.normal(0, sigma_in, signal_length)
        t = np.arange(0, second_signal_length / 4, 0.25)
        phase_shift_2 = np.random.rand() * np.pi
        second_signal = np.sin(second_signal_freq * t + phase_shift_2) + \
            np.random.normal(0, sigma_in, second_signal_length)

        signals[i, first_signal_timing: first_signal_timing + signal_length, 0] = first_signal
        signals[i, second_signal_timing: second_signal_timing + second_signal_length, 0] = second_signal

    return signals


def main(config_path, sigma_in, signal_length, delay, epoch):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = os.path.splitext(os.path.basename(config_path))[0]

    # モデルのロード
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg['MODEL']['SIGMA_NEU'] = 0
    model = RecurrentNeuralNetwork(n_in=1, n_out=2, n_hid=cfg['MODEL']['SIZE'], device=device,
                                   alpha_time_scale=0.25, beta_time_scale=cfg['MODEL']['BETA'],
                                   activation=cfg['MODEL']['ACTIVATION'],
                                   sigma_neu=cfg['MODEL']['SIGMA_NEU'],
                                   sigma_syn=cfg['MODEL']['SIGMA_SYN'],
                                   use_bias=cfg['MODEL']['USE_BIAS'],
                                   anti_hebbian=cfg['MODEL']['ANTI_HEBB']).to(device)

    model_path = f'../trained_model/freq/{model_name}/epoch_{epoch}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    os.makedirs('results/', exist_ok=True)
    save_path = f'results/psychometric_curve/'
    os.makedirs(save_path, exist_ok=True)

    deltas = np.arange(-2, 2, 0.05)
    N = 1000
    score = np.zeros(deltas.shape[0])
    delta_index = 0
    print('delta score')
    for delta in deltas:
        output_list = np.zeros(N)
        input_signal = romo_signal(delta, N, signal_length, sigma_in, delay)
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
        if delta_index % 5 == 0:
            print(f'{delta:.3f}', np.mean(output_list))
        delta_index += 1
    if sigma_in == 0.05:
        np.save(os.path.join(save_path, f'{model_name}_{delay}_{epoch}_2.npy'), score)
    else:
        np.save(os.path.join(save_path, f'{model_name}_{sigma_in}_2.npy'), score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    parser.add_argument('--sigma_in', type=float, default=0.05)
    parser.add_argument('--signal_length', type=int, default=15)
    parser.add_argument('--delay', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=3000)
    args = parser.parse_args()
    print(args)
    main(args.config_path, args.sigma_in, args.signal_length, args.delay, args.epoch)
