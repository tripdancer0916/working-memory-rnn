import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

plt.style.use('seaborn-darkgrid')
import seaborn as sns

sns.set_palette('Dark2')


class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, n_in, n_out, n_hid, device,
                 alpha_time_scale=0.25, beta_time_scale=0.1, activation='tanh', sigma_neu=0.05, sigma_syn=0.002,
                 use_bias=True, anti_hebbian=True):
        super(RecurrentNeuralNetwork, self).__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.w_in = nn.Linear(n_in, n_hid, bias=False)
        self.w_hh = nn.Linear(n_hid, n_hid, bias=use_bias)
        self.w_out = nn.Linear(n_hid, n_out, bias=False)

        self.activation = activation
        self.sigma_neu = sigma_neu
        self.sigma_syn = sigma_syn
        self.device = device

        self.alpha = torch.ones(self.n_hid) * alpha_time_scale
        self.beta = torch.ones(self.n_hid) * beta_time_scale
        self.alpha = self.alpha.to(self.device)
        self.beta = self.beta.to(self.device)
        self.anti_hebbian = anti_hebbian

    def change_alpha(self, new_alpha_time_scale):
        self.alpha = torch.ones(self.n_hid) * new_alpha_time_scale
        self.alpha = self.alpha.to(self.device)

    def make_neural_noise(self, hidden, alpha, sigma_neu):
        return torch.randn_like(hidden).to(self.device) * sigma_neu * torch.sqrt(alpha)

    def anti_hebbian_synaptic_plasticity(self, num_batch, firing_rate, synapse, beta):
        outer_product = torch.zeros([num_batch, self.n_hid, self.n_hid]).to(self.device)
        for i in range(num_batch):
            outer_product[i, :, :] = torch.eye(self.n_hid)
        for i in range(num_batch):
            outer_product[i, :, :] = -torch.ger(firing_rate[i], firing_rate[i])
        return outer_product + torch.randn_like(synapse).to(self.device) * self.sigma_syn * torch.sqrt(beta)

    def hebbian_synaptic_plasticity(self, num_batch, firing_rate, synapse, beta):
        outer_product = torch.zeros([self.num, self.n_hid, self.n_hid]).to(self.device)
        for i in range(num_batch):
            outer_product[i, :, :] = torch.ger(firing_rate[i], firing_rate[i])
        return outer_product + torch.randn_like(synapse).to(self.device) * self.sigma_syn * torch.sqrt(beta)

    def forward_(self, input_signal, hidden, perturbed_time):
        num_batch = input_signal.size(0)
        length = input_signal.size(1)
        hidden_list = torch.zeros(length, num_batch, self.n_hid).type_as(input_signal.data)
        output_list = torch.zeros(length, num_batch, self.n_out).type_as(input_signal.data)

        input_signal = input_signal.permute(1, 0, 2)

        different_j = torch.zeros((num_batch, self.n_hid, self.n_hid)).to(self.device)
        additional_w = torch.zeros((num_batch, self.n_hid, self.n_hid)).to(self.device)
        new_j = self.w_hh.weight
        for t in range(length):
            if self.activation == 'tanh':
                activated = torch.tanh(hidden)
            elif self.activation == 'relu':
                activated = F.relu(hidden)
            elif self.activation == 'identity':
                activated = hidden
            else:
                raise ValueError

            if self.beta[0].item() == 0:  # Short-term synaptic plasticityを考えない場合
                tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(activated)
                if t == perturbed_time:
                    neural_noise = self.make_neural_noise(hidden, self.alpha, 0.2)
                else:
                    neural_noise = self.make_neural_noise(hidden, self.alpha, 0)
                hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden + neural_noise
            else:
                different_j_activity = torch.matmul(activated.unsqueeze(1), different_j).squeeze(1)
                tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(activated) + different_j_activity

                if t == perturbed_time:
                    neural_noise = self.make_neural_noise(hidden, self.alpha, 0.1)
                else:
                    neural_noise = self.make_neural_noise(hidden, self.alpha, 0)
                hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden + neural_noise

                if self.anti_hebbian:
                    additional_w = self.anti_hebbian_synaptic_plasticity(num_batch, activated, additional_w, self.beta)
                else:
                    additional_w = self.hebbian_synaptic_plasticity(num_batch, activated, additional_w, self.beta)
                new_j = new_j + self.beta * additional_w
                different_j = new_j - self.w_hh.weight

            output = self.w_out(hidden)
            hidden_list[t] = hidden
            output_list[t] = output

        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)

        return hidden_list, output_list, hidden, new_j


def romo_signal(batch_size, signal_length, sigma_in, time_length=400, alpha=0.25):
    signals = np.zeros([batch_size, time_length + 1, 1])
    omega_1_list = np.random.rand(batch_size) * 4 + 1
    for i in range(batch_size):
        phase_shift_1 = np.random.rand() * np.pi
        omega_1 = omega_1_list[i]
        first_signal_timing = 0
        t = np.arange(0, signal_length * alpha, alpha)

        first_signal = np.sin(omega_1 * t + phase_shift_1) + np.random.normal(0, sigma_in, signal_length)
        signals[i, first_signal_timing: first_signal_timing + signal_length, 0] = first_signal

    return signals, omega_1_list


def main(config_path):
    torch.manual_seed(1)
    device = torch.device('cpu')

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['MODEL']['SIGMA_NEU'] = 0
    model_name = os.path.splitext(os.path.basename(config_path))[0]
    print('model_name: ', model_name)

    cfg['TRAIN']['BATCHSIZE'] = 2

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

    trial_num = 300
    neural_dynamics = np.zeros((trial_num, 1501, model.n_hid))
    neural_dynamics2 = np.zeros((trial_num, 1501, model.n_hid))
    outputs_np = np.zeros(trial_num)
    input_signal, omega_1_list = romo_signal(trial_num, signal_length=15, sigma_in=0.05, time_length=1500)
    input_signal_split = np.split(input_signal, trial_num // cfg['TRAIN']['BATCHSIZE'])

    for i in range(trial_num // cfg['TRAIN']['BATCHSIZE']):
        hidden = torch.zeros(cfg['TRAIN']['BATCHSIZE'], model.n_hid)
        hidden = hidden.to(device)
        inputs = torch.from_numpy(input_signal_split[i]).float()
        inputs = inputs.to(device)
        hidden_list, outputs, _, _ = model.forward_(inputs, hidden, 10000)
        hidden_list_np = hidden_list.cpu().detach().numpy()
        outputs_np[i * cfg['TRAIN']['BATCHSIZE']: (i + 1) * cfg['TRAIN']['BATCHSIZE']] = np.argmax(
            outputs.detach().numpy()[:, -1], axis=1)
        neural_dynamics[i * cfg['TRAIN']['BATCHSIZE']: (i + 1) * cfg['TRAIN']['BATCHSIZE']] = hidden_list_np

    norm_diff_list = np.zeros((trial_num, 100))

    for i in range(trial_num // cfg['TRAIN']['BATCHSIZE']):
        hidden = torch.zeros(cfg['TRAIN']['BATCHSIZE'], model.n_hid)
        hidden = hidden.to(device)
        inputs = torch.from_numpy(input_signal_split[i]).float()
        inputs = inputs.to(device)
        perturbed_time = np.random.randint(1000, 1200)
        hidden_list, outputs, _, _ = model.forward_(inputs, hidden, perturbed_time)
        hidden_list_np = hidden_list.cpu().detach().numpy()
        outputs_np[i * cfg['TRAIN']['BATCHSIZE']: (i + 1) * cfg['TRAIN']['BATCHSIZE']] = np.argmax(
            outputs.detach().numpy()[:, -1], axis=1)
        neural_dynamics2[i * cfg['TRAIN']['BATCHSIZE']: (i + 1) * cfg['TRAIN']['BATCHSIZE']] = hidden_list_np

        norm_diff_list[i * cfg['TRAIN']['BATCHSIZE']: (i + 1) * cfg['TRAIN']['BATCHSIZE']] = np.linalg.norm(
            neural_dynamics2[i * cfg['TRAIN']['BATCHSIZE']:
                             (i + 1) * cfg['TRAIN']['BATCHSIZE'], perturbed_time:perturbed_time + 100] -
            neural_dynamics[i * cfg['TRAIN']['BATCHSIZE']:
                            (i + 1) * cfg['TRAIN']['BATCHSIZE'], perturbed_time:perturbed_time + 100],
            axis=2)

    np.save(f'results/{model_name}/norm_diff.npy', np.array(norm_diff_list))
    plt.figure(constrained_layout=True)
    plt.plot(np.mean(norm_diff_list, axis=0), color='blue')
    plt.xlabel(r'$|{\bf x}_{\rm per}(t)-{\bf x}(t)|$', fontsize=16)
    # plt.ylim([0, 0.03])
    plt.savefig(f'results/{model_name}/norm_diff.png', dpi=200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    print(args)
    main(args.config_path)
