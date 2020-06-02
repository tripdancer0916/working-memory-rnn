"""training models"""

import argparse
import os
import shutil
import sys

import numpy as np
import torch
import torch.optim as optim
import yaml

sys.path.append('../')

from torch.autograd import Variable

from freq_dataset import FreqDataset
from model import RecurrentNeuralNetwork


def main(config_path):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = os.path.splitext(os.path.basename(config_path))[0]

    # save path
    os.makedirs('trained_model', exist_ok=True)
    os.makedirs('trained_model/freq', exist_ok=True)
    save_path = f'trained_model/freq/{model_name}'
    os.makedirs(save_path, exist_ok=True)

    # copy config file
    shutil.copyfile(config_path, os.path.join(save_path, os.path.basename(config_path)))

    use_cuda = cfg['MACHINE']['CUDA'] and torch.cuda.is_available()
    torch.manual_seed(cfg['MACHINE']['SEED'])
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

    model = RecurrentNeuralNetwork(n_in=1, n_out=2, n_hid=cfg['MODEL']['SIZE'], device=device,
                                   alpha_time_scale=0.25, beta_time_scale=cfg['MODEL']['BETA'],
                                   activation=cfg['MODEL']['ACTIVATION'],
                                   sigma_neu=cfg['MODEL']['SIGMA_NEU'],
                                   sigma_syn=cfg['MODEL']['SIGMA_SYN'],
                                   use_bias=cfg['MODEL']['USE_BIAS'],
                                   anti_hebbian=cfg['MODEL']['ANTI_HEBB']).to(device)

    train_dataset = FreqDataset(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                                freq_min=cfg['DATALOADER']['FREQ_MIN'],
                                freq_max=cfg['DATALOADER']['FREQ_MAX'],
                                min_interval=cfg['DATALOADER']['MIN_INTERVAL'],
                                signal_length=cfg['DATALOADER']['SIGNAL_LENGTH'],
                                variable_signal_length=cfg['DATALOADER']['VARIABLE_SIGNAL_LENGTH'],
                                sigma_in=cfg['DATALOADER']['SIGMA_IN'])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                                                   num_workers=2, shuffle=True,
                                                   worker_init_fn=lambda x: np.random.seed())

    print(model)
    print('Epoch Loss Acc')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg['TRAIN']['LR'], weight_decay=cfg['TRAIN']['WEIGHT_DECAY'])
    correct = 0
    num_data = 0
    for epoch in range(cfg['TRAIN']['NUM_EPOCH'] + 1):
        model.train()
        for i, data in enumerate(train_dataloader):
            inputs, target = data
            # print(inputs.shape)
            inputs, target = inputs.float(), target.long()
            inputs, target = Variable(inputs).to(device), Variable(target).to(device)

            hidden = torch.zeros(cfg['TRAIN']['BATCHSIZE'], cfg['MODEL']['SIZE'])
            hidden = hidden.to(device)

            optimizer.zero_grad()
            hidden = hidden.detach()
            hidden_list, output, hidden, new_j = model(inputs, hidden)
            # print(output)

            loss = torch.nn.CrossEntropyLoss()(output[:, -1], target)
            """
            dummy_zero = torch.zeros([cfg['TRAIN']['BATCHSIZE'],
                                      cfg['DATALOADER']['TIME_LENGTH'] + 1,
                                      cfg['MODEL']['SIZE']]).float().to(device)
            active_norm = torch.nn.MSELoss()(hidden_list, dummy_zero)

            loss += cfg['TRAIN']['ACTIVATION_LAMBDA'] * active_norm
            """
            loss.backward()
            optimizer.step()
            correct += (np.argmax(output[:, -1].cpu().detach().numpy(),
                                  axis=1) == target.cpu().detach().numpy()).sum().item()
            num_data += target.cpu().detach().numpy().shape[0]

        if epoch % cfg['TRAIN']['DISPLAY_EPOCH'] == 0:
            acc = correct / num_data
            print(f'{epoch}, {loss.item():.6f}, {acc:.6f}')
            # print('w_hh: ', model.w_hh.weight.cpu().detach().numpy()[:4, :4])
            # print('new_j: ', new_j.cpu().detach().numpy()[0, :4, :4])
            correct = 0
            num_data = 0
        if epoch > 0 and epoch % cfg['TRAIN']['NUM_SAVE_EPOCH'] == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    print(args)
    main(args.config_path)
