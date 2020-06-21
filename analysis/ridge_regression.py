import argparse
import os

import numpy as np
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from model import RecurrentNeuralNetwork


def romo_signal(batch_size, signal_length, sigma_in):
    signals = np.zeros([batch_size, 61, 1])
    omega_1_list = np.random.rand(batch_size) * 4 + 1
    omega_2_list = np.random.rand(batch_size) * 4 + 1
    for i in range(batch_size):
        omega_1 = omega_1_list[i]
        omega_2 = omega_2_list[i]
        phase_shift_1 = np.random.rand() * np.pi
        phase_shift_2 = np.random.rand() * np.pi
        first_signal_timing = 0
        second_signal_timing = 60 - signal_length
        t = np.arange(0, signal_length / 4, 0.25)
        first_signal = np.sin(omega_1 * t + phase_shift_1) + np.random.normal(0, sigma_in, signal_length)
        second_signal = np.sin(omega_2 * t + phase_shift_2) + np.random.normal(0, sigma_in, signal_length)
        signals[i, first_signal_timing: first_signal_timing + signal_length, 0] = first_signal
        signals[i, second_signal_timing: second_signal_timing + signal_length, 0] = second_signal

    return signals, omega_1_list, omega_2_list


def main(config_path, sigma_in, signal_length, alpha):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = os.path.splitext(os.path.basename(config_path))[0]

    os.makedirs('results/', exist_ok=True)
    save_path = f'results/w_ridge/'
    os.makedirs(save_path, exist_ok=True)

    # モデルのロード
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RecurrentNeuralNetwork(n_in=1, n_out=2, n_hid=cfg['MODEL']['SIZE'], device=device,
                                   alpha_time_scale=0.25, beta_time_scale=cfg['MODEL']['BETA'],
                                   activation=cfg['MODEL']['ACTIVATION'],
                                   sigma_neu=cfg['MODEL']['SIGMA_NEU'],
                                   sigma_syn=cfg['MODEL']['SIGMA_SYN'],
                                   use_bias=cfg['MODEL']['USE_BIAS'],
                                   anti_hebbian=cfg['MODEL']['ANTI_HEBB']).to(device)

    model_path = f'trained_model/romo/{model_name}/epoch_{cfg["TRAIN"]["NUM_EPOCH"]}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    sample_num = 5000
    neural_dynamics = np.zeros((sample_num, 61, model.n_hid))
    input_signal, omega_1_list, omega_2_list = romo_signal(sample_num, signal_length=signal_length,
                                                           sigma_in=sigma_in)
    input_signal_split = np.split(input_signal, sample_num // cfg['TRAIN']['BATCHSIZE'])

    # メモリを圧迫しないために推論はバッチサイズごとに分けて行う。
    for i in range(sample_num // cfg['TRAIN']['BATCHSIZE']):
        hidden = torch.zeros(cfg['TRAIN']['BATCHSIZE'], model.n_hid)
        hidden = hidden.to(device)
        inputs = torch.from_numpy(input_signal_split[i]).float()
        inputs = inputs.to(device)
        hidden_list, outputs, _, _ = model(inputs, hidden)
        hidden_list_np = hidden_list.cpu().detach().numpy()
        neural_dynamics[i * cfg['TRAIN']['BATCHSIZE']: (i + 1) * cfg['TRAIN']['BATCHSIZE']] = hidden_list_np

    sample_X_1 = np.zeros([30 * sample_num, model.n_hid])
    sample_X_2 = np.zeros([sample_num, model.n_hid])
    sample_y_1 = np.zeros([30 * sample_num])
    sample_y_2 = np.zeros(sample_num)

    for i in range(sample_num):
        sample_X_1[i * 30: (i + 1) * 30, :] = neural_dynamics[i, 15:45, :] ** 2
        sample_X_2[i, :] = neural_dynamics[i, 55, :] ** 2
        sample_y_1[i * 30: (i + 1) * 30] = omega_1_list[i]
        sample_y_2[i] = omega_2_list[i]

    # 訓練データとテストデータを分離
    train_X_1, test_X_1, train_y_1, test_y_1 = train_test_split(sample_X_1, sample_y_1, random_state=0)
    train_X_2, test_X_2, train_y_2, test_y_2 = train_test_split(sample_X_2, sample_y_2, random_state=0)

    ridge_1 = Ridge(alpha=alpha)
    ridge_1.fit(train_X_1, train_y_1)

    ridge_2 = Ridge(alpha=alpha)
    ridge_2.fit(train_X_2, train_y_2)

    np.save(os.path.join(save_path, f'{model_name}_omega_1.npy'), ridge_1.coef_)
    np.save(os.path.join(save_path, f'{model_name}_omega_2.npy'), ridge_2.coef_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    parser.add_argument('--sigma_in', type=float, default=0.05)
    parser.add_argument('--signal_length', type=int, default=15)
    parser.add_argument('--alpha', type=float, default=2)
    args = parser.parse_args()
    print(args)
    main(args.config_path, args.sigma_in, args.signal_length, args.alpha)
