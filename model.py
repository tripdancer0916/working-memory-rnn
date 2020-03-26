"""define recurrent neural networks"""

import torch
import torch.nn as nn


class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, n_in, n_out, n_hid, device,
                 alpha_time_scale=0.25, beta_time_scale=0.1, activation='tanh', sigma_neu=0.05, sigma_syn=0.002, use_bias=True):
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

    def make_neural_noise(self, hidden, alpha):
        return torch.randn_like(hidden).to(self.device) * self.sigma_neu * torch.sqrt(alpha)

    def make_synaptic_plasticity(self, firing_rate, synapse, beta):
        outer_product = torch.zeros([50, self.n_hid, self.n_hid]).to(self.device)
        for i in range(50):
            outer_product[i, :, :] = torch.eye(self.n_hid)
        for i in range(50):
            outer_product[i, :, :] = -torch.ger(firing_rate[i], firing_rate[i])
        return outer_product + torch.randn_like(synapse).to(self.device) * self.sigma_syn * torch.sqrt(beta)

    def forward(self, input_signal, hidden):
        num_batch = input_signal.size(0)
        length = input_signal.size(1)
        hidden_list = torch.zeros(length, num_batch, self.n_hid).type_as(input_signal.data)
        output_list = torch.zeros(length, num_batch, self.n_out).type_as(input_signal.data)

        input_signal = input_signal.permute(1, 0, 2)

        different_j = torch.zeros((num_batch, self.n_hid, self.n_hid)).to(self.device)
        additional_w = torch.zeros((num_batch, self.n_hid, self.n_hid)).to(self.device)
        new_j = self.w_hh.weight
        for t in range(length):
            activated = torch.tanh(hidden)

            different_j_activity = torch.matmul(activated.unsqueeze(1), different_j).squeeze(1)
            # print(torch.norm(different_j_activity).item())
            tmp_hidden = self.w_in(input_signal[t]) + self.w_hh(activated) + different_j_activity

            neural_noise = self.make_neural_noise(hidden, self.alpha)
            hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden + neural_noise

            additional_w = self.make_synaptic_plasticity(activated, additional_w, self.beta)
            new_j = new_j + self.beta * additional_w
            different_j = new_j - self.w_hh.weight

            output = self.w_out(hidden)
            hidden_list[t] = hidden
            output_list[t] = output

        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)

        return hidden_list, output_list, hidden
