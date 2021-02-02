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


def romo_signal(batch_size, signal_length, sigma_in):
    signals = np.zeros([batch_size, 201, 1])
    omega_1_list = np.random.choice([1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6,
                                     2.8, 3, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8,
                                     5], batch_size)
    omega_2_list = np.random.rand(batch_size) * 4 + 1
    for i in range(batch_size):
        omega_1 = omega_1_list[i]
        omega_2 = omega_2_list[i]
        # phase_shift_1 = np.random.rand() * np.pi
        # phase_shift_2 = np.random.rand() * np.pi
        phase_shift_1 = 0
        phase_shift_2 = 0
        first_signal_timing = 0
        # second_signal_timing = 60 - signal_length
        t = np.arange(0, signal_length / 4, 0.25)
        first_signal = np.sin(omega_1 * t + phase_shift_1) + np.random.normal(0, sigma_in, signal_length)
        # second_signal = np.sin(omega_2 * t + phase_shift_2) + np.random.normal(0, sigma_in, signal_length)
        signals[i, first_signal_timing: first_signal_timing + signal_length, 0] = first_signal
        # signals[i, second_signal_timing: second_signal_timing + signal_length, 0] = second_signal

    return signals, omega_1_list, omega_2_list


def main(config_path, sigma_in, signal_length, time_point, model_epoch):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = os.path.splitext(os.path.basename(config_path))[0]

    os.makedirs('results/', exist_ok=True)
    os.makedirs(f'results/{model_name}', exist_ok=True)
    save_path = f'results/{model_name}/encoding_dimension/'
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

    sample_num = 5000
    neural_dynamics = np.zeros((sample_num, 201, model.n_hid))
    input_signal, omega_1_list, omega_2_list = romo_signal(sample_num, signal_length=signal_length, sigma_in=sigma_in)
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

    sample_X = np.zeros([sample_num, model.n_hid])
    sample_y = np.zeros(sample_num)

    for i in range(sample_num):
        sample_X[i: (i + 1), :] = neural_dynamics[i, time_point, :]
        sample_y[i] = omega_1_list[i]

    correct = 0
    acc_list = np.zeros(300)
    for i in range(300):
        binary_def = np.random.choice([0, 1], 21)
        omega_1_sample = [1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3,
                          3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5]
        binary_dict = {}
        for j in range(21):
            binary_dict[omega_1_sample[j]] = binary_def[j]
        label = np.zeros(sample_num)
        for j in range(sample_num):
            if binary_dict[sample_y[j]] == 1:
                label[j] = 1
            else:
                label[j] = 0

        # 訓練データとテストデータを分離
        train_X, test_X, train_label, test_label = train_test_split(sample_X, label, random_state=0)

        # 線形SVMのインスタンスを生成
        linear_svc = LinearSVC(random_state=None, max_iter=5000)

        # モデルの学習。fit関数で行う。
        linear_svc.fit(train_X, train_label)

        # テストデータに対する精度
        pred_test = linear_svc.predict(test_X)
        accuracy_test = accuracy_score(test_label, pred_test)
        print('テストデータに対する正解率： %.2f' % accuracy_test)
        if accuracy_test > 0.85:
            correct += 1
        acc_list[i] = accuracy_test

    np.save(os.path.join(save_path, f'correct_ratio.npy'), np.array(correct / 300))
    np.save(os.path.join(save_path, f'acc_list.npy'), acc_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    parser.add_argument('--sigma_in', type=float, default=0)
    parser.add_argument('--signal_length', type=int, default=15)
    parser.add_argument('--time_point', type=int, default=45)
    parser.add_argument('--model_epoch', type=int, default=3000)
    args = parser.parse_args()
    print(args)
    main(args.config_path, args.sigma_in, args.signal_length, args.time_point, args.model_epoch)
