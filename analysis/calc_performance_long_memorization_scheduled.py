import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.append('../')

from model import RecurrentNeuralNetwork


def romo_signal(delta, N, time_length, signal_length, sigma_in, time_scale, second_signal_timing):
    signals = np.zeros([N, time_length + 1, 1])
    freq_range = 4 - abs(delta)
    first_signal_timing = 0
    # second_signal_timing = 60 - signal_length

    for i in range(N):
        first_signal_freq = np.random.rand() * freq_range + max(1, 1 - delta)
        second_signal_freq = first_signal_freq + delta
        second_signal_length = time_length - second_signal_timing

        t = np.arange(0, signal_length * time_scale, time_scale)
        if len(t) != signal_length:
            t = t[:-1]
        phase_shift_1 = np.random.rand() * np.pi
        first_signal = np.sin(first_signal_freq * t + phase_shift_1) + np.random.normal(0, sigma_in, signal_length)

        t = np.arange(0, second_signal_length * time_scale, time_scale)
        if len(t) != second_signal_length:
            t = t[:-1]
        phase_shift_2 = np.random.rand() * np.pi
        second_signal = np.sin(second_signal_freq * t + phase_shift_2) + np.random.normal(0, sigma_in, second_signal_length)

        signals[i, first_signal_timing: first_signal_timing + signal_length, 0] = first_signal
        signals[i, second_signal_timing: second_signal_timing + second_signal_length, 0] = second_signal

    return signals


def main(config_path, sigma_in):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['MODEL']['ALPHA'] = 0.075
    cfg['DATALOADER']['TIME_LENGTH'] = 200
    cfg['DATALOADER']['SIGNAL_LENGTH'] = 50
    cfg['DATALOADER']['VARIABLE_DELAY'] = 15

    model_name = os.path.splitext(os.path.basename(config_path))[0]

    os.makedirs('../results/', exist_ok=True)
    save_path = f'../results/accuracy_long_memorization/'
    os.makedirs(save_path, exist_ok=True)

    print('memorization_duration accuracy')
    results_acc = np.zeros(10)

    # モデルのロード
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg['MODEL']['SIGMA_NEU'] = 0
    model = RecurrentNeuralNetwork(n_in=1, n_out=2, n_hid=cfg['MODEL']['SIZE'], device=device,
                                   alpha_time_scale=cfg['MODEL']['ALPHA'], beta_time_scale=cfg['MODEL']['BETA'],
                                   activation=cfg['MODEL']['ACTIVATION'],
                                   sigma_neu=cfg['MODEL']['SIGMA_NEU'],
                                   sigma_syn=cfg['MODEL']['SIGMA_SYN'],
                                   use_bias=cfg['MODEL']['USE_BIAS'],
                                   anti_hebbian=cfg['MODEL']['ANTI_HEBB']).to(device)

    model_path = f'../trained_model/freq_schedule/{model_name}/epoch_{cfg["TRAIN"]["NUM_EPOCH"]}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    for duration_idx in range(10):
        correct = 0
        num_data = 0
        # print('delta correct_rate')
        for delta_idx in range(30):
            while True:
                delta = np.random.rand() * 8 - 4
                if abs(delta) >= 1:
                    break
            N = 500
            output_list = np.zeros(N)
            input_signal = romo_signal(delta, N, cfg['DATALOADER']['TIME_LENGTH'],
                                       cfg['DATALOADER']['SIGNAL_LENGTH'], sigma_in,
                                       cfg['MODEL']['ALPHA'],
                                       second_signal_timing=150+duration_idx*2)
            input_signal_split = np.split(input_signal, 10)
            for i in range(10):
                hidden = torch.zeros(50, model.n_hid)
                hidden = hidden.to(device)
                inputs = torch.from_numpy(input_signal_split[i]).float()
                inputs = inputs.to(device)
                _, outputs, _, _ = model(inputs, hidden)
                outputs_np = outputs.cpu().detach().numpy()
                output_list[i * 50: (i + 1) * 50] = np.argmax(outputs_np[:, -1], axis=1)
            num_data += 500
            if delta > 0:
                ans = 1
            else:
                ans = 0
            correct += (output_list == ans).sum()
        # print(correct / num_data)
        results_acc[duration_idx] = correct / num_data
        print(duration_idx*2+100, correct / num_data)

    np.save(os.path.join(save_path, f'{model_name}.npy'), results_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    parser.add_argument('--sigma_in', type=float, default=0.05)
    args = parser.parse_args()
    print(args)
    main(args.config_path, args.sigma_in)
