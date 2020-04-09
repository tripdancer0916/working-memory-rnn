import argparse
import os

import numpy as np
import torch
import yaml

from model import RecurrentNeuralNetwork


def romo_signal_fix_omega_2(batch_size, omega_2, signal_length, sigma_in):
    signals = np.zeros([batch_size, 61, 1])
    omega_1_list = np.linspace(1, 5, batch_size)
    # phase_shiftは全サンプル通して固定
    phase_shift_1 = np.random.rand() * np.pi
    phase_shift_2 = np.random.rand() * np.pi
    for i in range(batch_size):
        omega_1 = omega_1_list[i]
        first_signal_timing = 0
        second_signal_timing = 60 - signal_length
        t = np.arange(0, signal_length / 4, 0.25)
        first_signal = np.sin(omega_1 * t + phase_shift_1) + np.random.normal(0, sigma_in, signal_length)
        second_signal = np.sin(omega_2 * t + phase_shift_2) + np.random.normal(0, sigma_in, signal_length)
        signals[i, first_signal_timing: first_signal_timing + signal_length, 0] = first_signal
        signals[i, second_signal_timing: second_signal_timing + signal_length, 0] = second_signal

    return signals


def main(config_path, sigma_in, signal_length):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = os.path.splitext(os.path.basename(config_path))[0]

    os.makedirs('results/', exist_ok=True)
    save_path = f'results/moving_distance/'
    os.makedirs(save_path, exist_ok=True)

    print('mean_moving_distance std')

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

    trial_num = 100
    neural_dynamics = np.zeros((trial_num, 61, model.n_hid))
    outputs_np = np.zeros(trial_num)
    input_signal = romo_signal_fix_omega_2(trial_num, 3, signal_length=signal_length, sigma_in=sigma_in)
    input_signal_split = np.split(input_signal, trial_num // cfg['TRAIN']['BATCHSIZE'])

    # メモリを圧迫しないために推論はバッチサイズごとに分けて行う。
    for i in range(trial_num // cfg['TRAIN']['BATCHSIZE']):
        hidden = torch.zeros(cfg['TRAIN']['BATCHSIZE'], model.n_hid)
        hidden = hidden.to(device)
        inputs = torch.from_numpy(input_signal_split[i]).float()
        inputs = inputs.to(device)
        hidden_list, outputs, _, _ = model(inputs, hidden)
        hidden_list_np = hidden_list.cpu().detach().numpy()
        outputs_np[i * cfg['TRAIN']['BATCHSIZE']: (i + 1) * cfg['TRAIN']['BATCHSIZE']] = np.argmax(
            outputs.cpu().detach().numpy()[:, -1], axis=1)
        neural_dynamics[i * cfg['TRAIN']['BATCHSIZE']: (i + 1) * cfg['TRAIN']['BATCHSIZE']] = hidden_list_np

    moving_distance = np.zeros(100)
    for i in range(100):
        moving_distance[i] = np.sum(np.linalg.norm(np.diff(neural_dynamics[i, 15:45, :], axis=0), axis=1))

    print(np.mean(moving_distance), np.std(moving_distance))

    np.save(os.path.join(save_path, f'{model_name}.npy'), moving_distance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    parser.add_argument('--sigma_in', type=float, default=0)
    parser.add_argument('--signal_length', type=int, default=15)
    args = parser.parse_args()
    print(args)
    main(args.config_path, args.sigma_in, args.signal_length)
