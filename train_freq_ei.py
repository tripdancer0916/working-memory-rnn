"""training models"""

import argparse
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

sys.path.append('../')

from torch.autograd import Variable

from freq_dataset import FreqDataset


class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, n_in, n_out, n_hid, device,
                 alpha_time_scale=0.25, activation='tanh', sigma_neu=0.05,
                 ):
        super(RecurrentNeuralNetwork, self).__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.w_in = nn.Linear(n_in, n_hid, bias=False)
        self.w_hh = nn.Linear(n_hid, n_hid, bias=False)
        self.w_hh.weight.data = torch.rand(n_hid, n_hid) / n_hid
        self.e_i_neuron = torch.eye(n_hid) * torch.from_numpy(np.array([1 if i < 180 else -1 for i in range(256)])).float()
        self.e_i_neuron = self.e_i_neuron.to(device)

        self.w_out = nn.Linear(n_hid, n_out, bias=False)

        self.activation = activation
        self.sigma_neu = sigma_neu
        self.device = device

        self.alpha = torch.ones(self.n_hid) * alpha_time_scale
        self.alpha = self.alpha.to(self.device)

    def make_neural_noise(self, hidden, alpha):
        return torch.randn_like(hidden).to(self.device) * self.sigma_neu * torch.sqrt(alpha)

    def forward(self, input_signal, hidden):
        w_rec = torch.mm(self.w_hh.weight.data, self.e_i_neuron)
        num_batch = input_signal.size(0)
        length = input_signal.size(1)
        hidden_list = torch.zeros(length, num_batch, self.n_hid).type_as(input_signal.data)
        output_list = torch.zeros(length, num_batch, self.n_out).type_as(input_signal.data)

        input_signal = input_signal.permute(1, 0, 2)

        for t in range(length):
            activated = torch.tanh(hidden)

            tmp_hidden = self.w_in(input_signal[t]) + F.linear(activated, w_rec)
            neural_noise = self.make_neural_noise(hidden, self.alpha)
            hidden = (1 - self.alpha) * hidden + self.alpha * tmp_hidden + neural_noise

            output = self.w_out(hidden)
            hidden_list[t] = hidden
            output_list[t] = output

        hidden_list = hidden_list.permute(1, 0, 2)
        output_list = output_list.permute(1, 0, 2)

        return hidden_list, output_list, hidden


def main(config_path):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = os.path.splitext(os.path.basename(config_path))[0]

    # save path
    os.makedirs('trained_model', exist_ok=True)
    os.makedirs('trained_model/freq_EI', exist_ok=True)
    save_path = f'trained_model/freq_EI/{model_name}'
    os.makedirs(save_path, exist_ok=True)

    # copy config file
    shutil.copyfile(config_path, os.path.join(save_path, os.path.basename(config_path)))

    use_cuda = cfg['MACHINE']['CUDA'] and torch.cuda.is_available()
    torch.manual_seed(cfg['MACHINE']['SEED'])
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

    if 'ALPHA' not in cfg['MODEL'].keys():
        cfg['MODEL']['ALPHA'] = 0.25

    model = RecurrentNeuralNetwork(
        n_in=1, n_out=2,
        n_hid=cfg['MODEL']['SIZE'],
        device=device,
        alpha_time_scale=cfg['MODEL']['ALPHA'],
        activation=cfg['MODEL']['ACTIVATION'],
        sigma_neu=cfg['MODEL']['SIGMA_NEU'],
    ).to(device)

    train_dataset = FreqDataset(
        time_length=cfg['DATALOADER']['TIME_LENGTH'],
        time_scale=cfg['MODEL']['ALPHA'],
        freq_min=cfg['DATALOADER']['FREQ_MIN'],
        freq_max=cfg['DATALOADER']['FREQ_MAX'],
        min_interval=cfg['DATALOADER']['MIN_INTERVAL'],
        signal_length=cfg['DATALOADER']['SIGNAL_LENGTH'],
        variable_signal_length=cfg['DATALOADER']['VARIABLE_SIGNAL_LENGTH'],
        sigma_in=cfg['DATALOADER']['SIGMA_IN'],
        delay_variable=cfg['DATALOADER']['VARIABLE_DELAY'],
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
        num_workers=2, shuffle=True,
        worker_init_fn=lambda x: np.random.seed(),
    )

    print(model)
    print(model.state_dict())
    print('Epoch Loss Acc')
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=cfg['TRAIN']['LR'], weight_decay=cfg['TRAIN']['WEIGHT_DECAY'])
    correct = 0
    num_data = 0
    for epoch in range(cfg['TRAIN']['NUM_EPOCH'] + 1):
        model.train()
        for i, data in enumerate(train_dataloader):
            # print(i)
            inputs, target = data
            inputs, target = inputs.float(), target.long()
            inputs, target = Variable(inputs).to(device), Variable(target).to(device)

            hidden = torch.zeros(cfg['TRAIN']['BATCHSIZE'], cfg['MODEL']['SIZE'])
            hidden = hidden.to(device)

            hidden = hidden.detach()
            hidden_list, output, hidden = model(inputs, hidden)

            loss = torch.nn.CrossEntropyLoss()(output[:, -1], target)
            dummy_zero = torch.zeros(
                [cfg['TRAIN']['BATCHSIZE'],
                 cfg['DATALOADER']['TIME_LENGTH'] + 1,
                 cfg['MODEL']['SIZE']],
            ).float().to(device)
            active_norm = torch.nn.MSELoss()(hidden_list, dummy_zero)

            loss += cfg['TRAIN']['ACTIVATION_LAMBDA'] * active_norm
            loss.backward()

            optimizer.step()
            correct += (np.argmax(output[:, -1].cpu().detach().numpy(),
                                  axis=1) == target.cpu().detach().numpy()).sum().item()
            num_data += target.cpu().detach().numpy().shape[0]
            plus_index = model.w_hh.weight.detach().cpu().numpy() > 0
            model.w_hh.weight.data *= torch.from_numpy(plus_index.astype(np.int)).float().to(device)

        if epoch % cfg['TRAIN']['DISPLAY_EPOCH'] == 0:
            # print(model.w_hh.data[:10, :10])
            acc = correct / num_data
            print(f'{epoch}, {loss.item():.6f}, {acc:.6f}')
            print(active_norm)

        if epoch > 0 and epoch % cfg['TRAIN']['NUM_SAVE_EPOCH'] == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pth'))
            # np.save(os.path.join(save_path, f'w_sign_{epoch}.npy'), model.w_sign.detach().cpu().numpy())
            # np.save(os.path.join(save_path, f'tensor_is_con_{epoch}.npy'), model.tensor_is_con_0.detach().cpu().numpy())
            # np.save(os.path.join(save_path, f'abs_w_0_{epoch}.npy'), model.abs_w_0.detach().cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    print(args)
    main(args.config_path)
