import argparse
import os

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr


from model import RecurrentNeuralNetwork


def romo_signal(batch_size, signal_length, sigma_in):
    signals = np.zeros([batch_size, 61, 1])
    omega_1_list = np.zeros(batch_size)
    omega_2_list = np.zeros(batch_size)
    for i in range(batch_size):
        omega_1 = np.random.rand() * 4 + 1
        omega_2 = np.random.rand() * 4 + 1
        first_signal_timing = 0
        second_signal_timing = 60 - signal_length
        t = np.arange(0, signal_length / 4, 0.25)
        phase_shift_1 = np.random.rand() * np.pi
        first_signal = np.sin(omega_1 * t + phase_shift_1) + np.random.normal(0, sigma_in, signal_length)
        phase_shift_2 = np.random.rand() * np.pi
        second_signal = np.sin(omega_2 * t + phase_shift_2) + np.random.normal(0, sigma_in, signal_length)

        signals[i, first_signal_timing: first_signal_timing + signal_length, 0] = first_signal
        signals[i, second_signal_timing: second_signal_timing + signal_length, 0] = second_signal

    return signals, omega_1_list, omega_2_list


def main(config_path, sigma_in, signal_length, trial_num):
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
    save_path = f'results/freq_correlation/'
    os.makedirs(save_path, exist_ok=True)

    neural_dynamics = np.zeros((trial_num, model.n_hid, 61))
    input_signal, omega_1_list, omega_2_list = romo_signal(trial_num, signal_length, sigma_in)
    input_signal_split = np.split(input_signal, trial_num / cfg['TRAIN']['BATCHSIZE'])

    # メモリを圧迫しないために推論はバッチサイズごとに分けて行う。
    for i in range(trial_num / cfg['TRAIN']['BATCHSIZE']):
        hidden = torch.zeros(cfg['TRAIN']['BATCHSIZE'], model.n_hid)
        hidden = hidden.to(device)
        inputs = torch.from_numpy(input_signal_split[i]).float()
        inputs = inputs.to(device)
        hidden_list, outputs, _, _ = model(inputs, hidden)
        hidden_list_np = hidden_list.cpu().detach().numpy()
        neural_dynamics[i * cfg['TRAIN']['BATCHSIZE']: (i + 1) * cfg['TRAIN']['BATCHSIZE']] = hidden_list_np

    # 各neuron index, 時間ごとにcorrelation(Spearmanの順位相関係数）を計算する。
    # 期間はomega_1はt=10~60の50単位時間にかけて、omega_2はt=45~60かなあ
    omega_1_correlation = np.zeros((50, model.n_hid))
    omega_2_correlation = np.zeros((15, model.n_hid))

    # omega_1
    for time in range(10, 60):
        for neuron_idx in range(model.n_hid):
            spearman_r, _ = spearmanr(neural_dynamics[:, neuron_idx, time], omega_1_list)
            omega_1_correlation[time - 10, neuron_idx] = spearman_r
    # omega_2
    for time in range(45, 60):
        for neuron_idx in range(model.n_hid):
            spearman_r, _ = spearmanr(neural_dynamics[:, neuron_idx, time], omega_2_list)
            omega_2_correlation[time - 45, neuron_idx] = spearman_r

    np.save(os.path.join(save_path, f'{model_name}_{sigma_in}_{trial_num}_omega_1.npy'), omega_1_correlation)
    np.save(os.path.join(save_path, f'{model_name}_{sigma_in}_{trial_num}_omega_2.npy'), omega_2_correlation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    parser.add_argument('--sigma_in', type=float, default=0.05)
    parser.add_argument('--signal_length', type=int, default=25)
    parser.add_argument('--trial_num', type=int, default=100)
    args = parser.parse_args()
    print(args)
    main(args.config_path, args.sigma_in, args.signal_length, args.trial_num)
