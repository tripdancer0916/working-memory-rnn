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

from dist_dataset import DistDataset
from model import RecurrentNeuralNetwork


def main(config_path):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if 'ACTIVATION_REG' not in cfg['TRAIN'].keys():
        cfg['TRAIN']['ACTIVATION_REG'] = 0
    model_name = os.path.splitext(os.path.basename(config_path))[0]

    # save path
    os.makedirs('trained_model', exist_ok=True)
    os.makedirs('trained_model/dist', exist_ok=True)
    save_path = f'trained_model/dist/{model_name}'
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

    train_dataset = DistDataset(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                                sigma_min=cfg['DATALOADER']['SIGMA_MIN'],
                                sigma_max=cfg['DATALOADER']['SIGMA_MAX'],
                                min_interval=cfg['DATALOADER']['MIN_INTERVAL'],
                                signal_length=cfg['DATALOADER']['SIGNAL_LENGTH'],
                                freq=cfg['DATALOADER']['FREQ'],
                                delay_variable=cfg['DATALOADER']['VARIABLE_DELAY'],
                                phase_shift=cfg['DATALOADER']['PHASE_SHIFT'])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                                                   num_workers=2, shuffle=True,
                                                   worker_init_fn=lambda x: np.random.seed())

    print(model)
    print('Epoch Loss Acc')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg['TRAIN']['LR'], weight_decay=cfg['TRAIN']['WEIGHT_DECAY'])
    correct = 0
    num_data = 0
    phase2 = False
    phase3 = False
    phase4 = False
    phase5 = False
    phase6 = False
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
            dummy_zero = torch.zeros([cfg['TRAIN']['BATCHSIZE'],
                                      cfg['DATALOADER']['TIME_LENGTH'] + 1,
                                      cfg['MODEL']['SIZE']]).float().to(device)
            active_loss = torch.nn.MSELoss()(hidden_list, dummy_zero)

            loss += cfg['TRAIN']['ACTIVATION_REG'] * active_loss
            loss.backward()
            optimizer.step()
            correct += (np.argmax(output[:, -1].cpu().detach().numpy(),
                                  axis=1) == target.cpu().detach().numpy()).sum().item()
            num_data += target.cpu().detach().numpy().shape[0]

            if not phase2 and float(loss.item()) < 0.6:
                cfg['DATALOADER']['VARIABLE_DELAY'] = 3
                cfg['DATALOADER']['PHASE_SHIFT'] = 0.1
                print("phase2 start! cfg['DATALOADER']['VARIABLE_DELAY'] = 3, cfg['DATALOADER']['PHASE_SHIFT'] = 0.1")
                phase2 = True
                train_dataset = DistDataset(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                                            sigma_min=cfg['DATALOADER']['SIGMA_MIN'],
                                            sigma_max=cfg['DATALOADER']['SIGMA_MAX'],
                                            min_interval=cfg['DATALOADER']['MIN_INTERVAL'],
                                            signal_length=cfg['DATALOADER']['SIGNAL_LENGTH'],
                                            freq=cfg['DATALOADER']['FREQ'],
                                            delay_variable=cfg['DATALOADER']['VARIABLE_DELAY'],
                                            phase_shift=cfg['DATALOADER']['PHASE_SHIFT'])
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                                                               num_workers=2, shuffle=True,
                                                               worker_init_fn=lambda x: np.random.seed())
                break

            if not phase3 and float(loss.item()) < 0.4:
                cfg['DATALOADER']['MIN_INTERVAL'] = 0.1
                cfg['DATALOADER']['VARIABLE_DELAY'] = 5
                print("phase3 start! cfg['DATALOADER']['MIN_INTERVAL'] = 0.1, cfg['DATALOADER']['VARIABLE_DELAY'] = 5")
                phase3 = True
                train_dataset = DistDataset(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                                            sigma_min=cfg['DATALOADER']['SIGMA_MIN'],
                                            sigma_max=cfg['DATALOADER']['SIGMA_MAX'],
                                            min_interval=cfg['DATALOADER']['MIN_INTERVAL'],
                                            signal_length=cfg['DATALOADER']['SIGNAL_LENGTH'],
                                            freq=cfg['DATALOADER']['FREQ'],
                                            delay_variable=cfg['DATALOADER']['VARIABLE_DELAY'],
                                            phase_shift=cfg['DATALOADER']['PHASE_SHIFT'])
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                                                               num_workers=2, shuffle=True,
                                                               worker_init_fn=lambda x: np.random.seed())
                break

            if not phase4 and float(loss.item()) < 0.4:
                cfg['DATALOADER']['PHASE_SHIFT'] = 0.5
                print("phase4 start! cfg['DATALOADER']['PHASE_SHIFT'] = 0.5")
                phase4 = True
                train_dataset = DistDataset(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                                            sigma_min=cfg['DATALOADER']['SIGMA_MIN'],
                                            sigma_max=cfg['DATALOADER']['SIGMA_MAX'],
                                            min_interval=cfg['DATALOADER']['MIN_INTERVAL'],
                                            signal_length=cfg['DATALOADER']['SIGNAL_LENGTH'],
                                            freq=cfg['DATALOADER']['FREQ'],
                                            delay_variable=cfg['DATALOADER']['VARIABLE_DELAY'],
                                            phase_shift=cfg['DATALOADER']['PHASE_SHIFT'])
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                                                               num_workers=2, shuffle=True,
                                                               worker_init_fn=lambda x: np.random.seed())
                break

            if not phase5 and float(loss.item()) < 0.25:
                cfg['DATALOADER']['TIME_LENGTH'] = 60
                cfg['DATALOADER']['SIGNAL_LENGTH'] = 15
                print("phase5 start! cfg['DATALOADER']['TIME_LENGTH'] = 60, cfg['DATALOADER']['SIGNAL_LENGTH'] = 15")
                phase5 = True
                train_dataset = DistDataset(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                                            sigma_min=cfg['DATALOADER']['SIGMA_MIN'],
                                            sigma_max=cfg['DATALOADER']['SIGMA_MAX'],
                                            min_interval=cfg['DATALOADER']['MIN_INTERVAL'],
                                            signal_length=cfg['DATALOADER']['SIGNAL_LENGTH'],
                                            freq=cfg['DATALOADER']['FREQ'],
                                            delay_variable=cfg['DATALOADER']['VARIABLE_DELAY'],
                                            phase_shift=cfg['DATALOADER']['PHASE_SHIFT'])
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                                                               num_workers=2, shuffle=True,
                                                               worker_init_fn=lambda x: np.random.seed())
                break

            if not phase6 and float(loss.item()) < 0.25:
                cfg['DATALOADER']['PHASE_SHIFT'] = 1
                print("phase6 start! cfg['DATALOADER']['PHASE_SHIFT'] = 1")
                phase6 = True
                train_dataset = DistDataset(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                                            sigma_min=cfg['DATALOADER']['SIGMA_MIN'],
                                            sigma_max=cfg['DATALOADER']['SIGMA_MAX'],
                                            min_interval=cfg['DATALOADER']['MIN_INTERVAL'],
                                            signal_length=cfg['DATALOADER']['SIGNAL_LENGTH'],
                                            freq=cfg['DATALOADER']['FREQ'],
                                            delay_variable=cfg['DATALOADER']['VARIABLE_DELAY'],
                                            phase_shift=cfg['DATALOADER']['PHASE_SHIFT'])
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                                                               num_workers=2, shuffle=True,
                                                               worker_init_fn=lambda x: np.random.seed())
                break

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
