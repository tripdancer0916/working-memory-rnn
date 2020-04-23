import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.decomposition import PCA


class RecurrentNeuralNetwork(nn.Module):
    def __init__(
            self,
            n_in,
            n_out,
            n_hid,
            device,
            alpha_time_scale=0.25,
            beta_time_scale=0.1,
            activation='tanh',
            use_bias=True,
            anti_hebbian=True):
        super(RecurrentNeuralNetwork, self).__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.w_in = nn.Linear(n_in, n_hid, bias=False)
        self.w_hh = nn.Linear(n_hid, n_hid, bias=use_bias)
        self.w_out = nn.Linear(n_hid, n_out, bias=False)

        self.activation = activation
        self.device = device

        self.alpha = torch.ones(self.n_hid) * alpha_time_scale
        self.beta = torch.ones(self.n_hid) * beta_time_scale
        self.alpha = self.alpha.to(self.device)
        self.beta = self.beta.to(self.device)
        self.anti_hebbian = anti_hebbian

    def make_neural_noise(self, hidden, alpha, sigma_neu):
        return torch.randn_like(hidden).to(
            self.device) * sigma_neu * torch.sqrt(alpha)

    def forward(self, input_signal, hidden, perturbation_timing, sigma_neu):
        num_batch = input_signal.size(0)
        length = input_signal.size(1)
        hidden_list = torch.zeros(
            length, num_batch, self.n_hid).type_as(
            input_signal.data)
        output_list = torch.zeros(
            length, num_batch, self.n_out).type_as(
            input_signal.data)

        input_signal = input_signal.permute(1, 0, 2)

        for t in range(length):
            activated = torch.tanh(hidden)
            tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(activated)

            if t == perturbation_timing:
                neural_noise = self.make_neural_noise(
                    hidden, self.alpha, sigma_neu)
                hidden = (1 - self.alpha) * hidden + \
                    self.alpha * tmp_hidden + neural_noise
            else:
                hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden

            output = self.w_out(hidden)
            hidden_list[t] = hidden
            output_list[t] = output

        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)

        return hidden_list, output_list, hidden


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
        first_signal = np.sin(omega_1 * t + phase_shift_1) + \
            np.random.normal(0, sigma_in, signal_length)
        second_signal = np.sin(omega_2 * t + phase_shift_2) + \
            np.random.normal(0, sigma_in, signal_length)
        signals[i, first_signal_timing: first_signal_timing +
                signal_length, 0] = first_signal
        signals[i, second_signal_timing: second_signal_timing +
                signal_length, 0] = second_signal

    return signals, omega_1_list, omega_2_list


def main(config_path, signal_length):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = os.path.splitext(os.path.basename(config_path))[0]

    os.makedirs('results/', exist_ok=True)
    save_path = f'results/transient_attractor/'
    os.makedirs(save_path, exist_ok=True)

    # モデルのロード
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg['MODEL']['SIGMA_NEU'] = 0.05
    model = RecurrentNeuralNetwork(
        n_in=1,
        n_out=2,
        n_hid=cfg['MODEL']['SIZE'],
        device=device,
        alpha_time_scale=0.25,
        beta_time_scale=cfg['MODEL']['BETA'],
        activation=cfg['MODEL']['ACTIVATION'],
        use_bias=cfg['MODEL']['USE_BIAS'],
        anti_hebbian=cfg['MODEL']['ANTI_HEBB']).to(device)

    model_path = f'trained_model/romo/{model_name}/epoch_{cfg["TRAIN"]["NUM_EPOCH"]}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    trial_num = 20
    sample_num = 10

    explained_variance_ratio = np.zeros([20, sample_num, 5, 256])

    for trial_id in range(trial_num):
        print('trial_id: ', trial_id)
        perturb_timing = 20 + trial_num
        input_signal, omega_1_list, omega_2_list = romo_signal(
            sample_num, signal_length=signal_length, sigma_in=0.05)

        neural_dynamics = np.zeros((1000, sample_num, 30, model.n_hid))

        hidden = torch.zeros(sample_num, model.n_hid)
        hidden = hidden.to(device)
        inputs = torch.from_numpy(input_signal).float()
        inputs = inputs.to(device)

        hidden_list, outputs, _ = model(inputs, hidden, perturb_timing, 0)
        hidden_list_np = hidden_list.cpu().detach().numpy()
        neural_dynamics[0, :] = hidden_list_np[:, 15:45, :]

        for trial in range(1, 1000):
            hidden = torch.zeros(sample_num, model.n_hid)
            hidden = hidden.to(device)
            inputs = torch.from_numpy(input_signal).float()
            inputs = inputs.to(device)

            hidden_list, outputs, _ = model(
                inputs, hidden, perturb_timing, 0.05)
            hidden_list_np = hidden_list.cpu().detach().numpy()
            neural_dynamics[trial, :] = hidden_list_np[:, 15:45, :]

        for sample_id in range(sample_num):
            for elapsed_time in range(5):
                pca = PCA()
                pca.fit(neural_dynamics[:,
                                        sample_id,
                                        perturb_timing - 15 + elapsed_time,
                                        :] - neural_dynamics[:1,
                                                             sample_id,
                                                             perturb_timing - 15 + elapsed_time,
                                                             :])
                print(pca.explained_variance_ratio_[:10])
                explained_variance_ratio[trial_id * sample_num + sample_id,
                                         elapsed_time, :] = pca.explained_variance_ratio_

    np.save(
        os.path.join(
            save_path,
            f'{model_name}.npy'),
        explained_variance_ratio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    parser.add_argument('--signal_length', type=int, default=15)
    args = parser.parse_args()
    print(args)
    main(args.config_path, args.signal_length)
