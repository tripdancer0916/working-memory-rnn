import argparse
import os

import numpy as np
import torch
import yaml

from model import RecurrentNeuralNetwork


def romo_signal(delta, N, signal_length, sigma_in):
    signals = np.zeros([N, 61, 1])
    freq_range = min(5, 5 - delta) - max(1, 1 + delta)
    for i in range(N):
        first_signal_timing = 0
        second_signal_timing = 60 - signal_length
        first_signal_freq = np.random.rand() * freq_range + max(1, 1 + delta)
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

    model_path = f'trained_model/romo/{model_name}/epoch_500.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    os.makedirs('results/', exist_ok=True)
    save_path = f'results/psychometric_curve/'
    os.makedirs(save_path, exist_ok=True)

    # model 1
    deltas = np.arange(-2, 2, 0.05)
    N = 200
    signal_length = 25
    sigma_in = 0.05
    score = np.zeros(deltas.shape[0])
    delta_index = 0
    print('delta score')
    for delta in deltas:
        output_list = np.zeros(N)
        input_signal = romo_signal(delta, N, signal_length, sigma_in)
        input_signal_split = np.split(input_signal, 4)
        for i in range(4):
            hidden = torch.zeros(50, model.n_hid)
            hidden = hidden.to(device)
            inputs = torch.from_numpy(input_signal_split[i]).float()
            _, outputs, _, _ = model(inputs, hidden)
            outputs_np = outputs.detach().numpy()
            output_list[i * 50: (i + 1) * 50] = np.argmax(outputs_np[:, -1], axis=1)
        score[delta_index] = np.mean(output_list)
        print(f'{delta:.3f}', np.mean(output_list))
        delta_index += 1

    np.save(os.path.join(save_path, f'{model_name}.npy'), score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    print(args)
    main(args.config_path)
