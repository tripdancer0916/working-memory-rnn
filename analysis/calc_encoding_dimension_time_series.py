import argparse
import os
import sys
import warnings

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

sys.path.append('../')

from model import RecurrentNeuralNetwork

warnings.filterwarnings(action='ignore', module='scipy', message='^internal gelsd')


def romo_signal_fixed_omega_1(omega_1, batch_size, signal_length, sigma_in):
    signals = np.zeros([batch_size, 61, 1])
    omega_2_list = np.random.rand(batch_size) * 4 + 1
    for i in range(batch_size):
        omega_2 = omega_2_list[i]
        # phase_shift_1 = np.random.rand() * np.pi
        # phase_shift_2 = np.random.rand() * np.pi
        phase_shift_1 = 0
        phase_shift_2 = 0
        first_signal_timing = 0
        second_signal_timing = 60 - signal_length
        t = np.arange(0, signal_length / 4, 0.25)
        first_signal = np.sin(omega_1 * t + phase_shift_1) + np.random.normal(0, sigma_in, signal_length)
        second_signal = np.sin(omega_2 * t + phase_shift_2) + np.random.normal(0, sigma_in, signal_length)
        signals[i, first_signal_timing: first_signal_timing + signal_length, 0] = first_signal
        # signals[i, second_signal_timing: second_signal_timing + signal_length, 0] = second_signal

    return signals, omega_2_list


def main(config_path, sigma_in, signal_length, model_epoch):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = os.path.splitext(os.path.basename(config_path))[0]

    os.makedirs('results/', exist_ok=True)
    os.makedirs(f'results/{model_name}', exist_ok=True)
    save_path = f'results/{model_name}/encoding_dimension_time_series/'
    os.makedirs(save_path, exist_ok=True)

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

    model_path = f'../trained_model/freq/{model_name}/epoch_{model_epoch}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    correct_ratio = np.zeros(6)
    acc_list = np.zeros([6, 200])
    division_num = 21
    time_sample = np.linspace(25, 45, division_num)
    omega_idx = 0
    for omega_1 in [1, 1.8, 2.6, 3.4, 4.2, 5]:
        sample_num = 100
        neural_dynamics = np.zeros((sample_num, 61, model.n_hid))
        input_signal, omega_2_list = romo_signal_fixed_omega_1(omega_1, sample_num,
                                                               signal_length=signal_length,
                                                               sigma_in=sigma_in)
        input_signal_split = np.split(input_signal, sample_num // cfg['TRAIN']['BATCHSIZE'])

        base_1 = np.random.randn(256)
        base_2 = np.random.randn(256)
        base_3 = np.random.randn(256)
        base_4 = np.random.randn(256)
        base_5 = np.random.randn(256)
        base_6 = np.random.randn(256)

        # メモリを圧迫しないために推論はバッチサイズごとに分けて行う。
        for i in range(sample_num // cfg['TRAIN']['BATCHSIZE']):
            hidden = torch.zeros(cfg['TRAIN']['BATCHSIZE'], model.n_hid)
            hidden = hidden.to(device)
            inputs = torch.from_numpy(input_signal_split[i]).float()
            inputs = inputs.to(device)
            hidden_list, outputs, _, _ = model(inputs, hidden)
            hidden_list_np = hidden_list.cpu().detach().numpy()
            neural_dynamics[i * cfg['TRAIN']['BATCHSIZE']: (i + 1) * cfg['TRAIN']['BATCHSIZE']] = hidden_list_np
            # neural_dynamics[i * cfg['TRAIN']['BATCHSIZE']: (i + 1) * cfg['TRAIN']['BATCHSIZE']] = np.array([np.random.randn() * base_1 for i in range(61)])

        sample_X = np.zeros([division_num, model.n_hid])
        sample_y = np.zeros(division_num)

        for i, time in enumerate(time_sample):
            # time = np.random.choice(time_sample)
            sample_X[i, :] = neural_dynamics[0, int(time), :]
            sample_y[i] = int(time)

        # print(sample_y)

        correct = 0
        for i in range(200):
            binary_def = np.random.choice([0, 1], division_num)
            binary_dict = {}
            for j in range(division_num):
                binary_dict[time_sample[j]] = binary_def[j]
            label = np.zeros(division_num)
            for j in range(division_num):
                if binary_dict[sample_y[j]] == 1:
                    label[j] = 1
                else:
                    label[j] = 0

            # 訓練データとテストデータを分離
            # train_X, test_X, train_label, test_label = train_test_split(sample_X, label, random_state=0)

            # 線形SVMのインスタンスを生成
            linear_svc = LinearSVC(random_state=None, max_iter=5000)

            # モデルの学習。fit関数で行う。
            linear_svc.fit(sample_X, label)

            # train dataに対する精度
            pred_train = linear_svc.predict(sample_X)
            accuracy_train = accuracy_score(label, pred_train)

            print(f'omega_1: {omega_1}, train acc: {accuracy_train:.2f}')

            # テストデータに対する精度
            # pred_test = linear_svc.predict(test_X)
            # accuracy_test = accuracy_score(test_label, pred_test)
            # print(f'omega_1: {omega_1}, train acc: {accuracy_train:.2f}, test acc: {accuracy_test:.2f}')
            if accuracy_train > 0.85:
                correct += 1

            acc_list[omega_idx, i] = accuracy_train

        correct_ratio[omega_idx] = correct / 200
        omega_idx += 1

    np.save(os.path.join(save_path, f'correct_ratio.npy'), correct_ratio)
    np.save(os.path.join(save_path, 'acc_list.npy'), acc_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    parser.add_argument('--sigma_in', type=float, default=0)
    parser.add_argument('--signal_length', type=int, default=15)
    parser.add_argument('--model_epoch', type=int, default=3000)
    args = parser.parse_args()
    print(args)
    main(args.config_path, args.sigma_in, args.signal_length, args.model_epoch)
