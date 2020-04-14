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
    os.makedirs('trained_model/freq_schedule', exist_ok=True)
    save_path = f'trained_model/freq_schedule/{model_name}'
    os.makedirs(save_path, exist_ok=True)

    # copy config file
    shutil.copyfile(config_path, os.path.join(save_path, os.path.basename(config_path)))

    use_cuda = cfg['MACHINE']['CUDA'] and torch.cuda.is_available()
    torch.manual_seed(cfg['MACHINE']['SEED'])
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

    if 'ALPHA' not in cfg['MODEL'].keys():
        cfg['MODEL']['ALPHA'] = 0.25

    model = RecurrentNeuralNetwork(n_in=1, n_out=2, n_hid=cfg['MODEL']['SIZE'], device=device,
                                   alpha_time_scale=cfg['MODEL']['ALPHA'], beta_time_scale=cfg['MODEL']['BETA'],
                                   activation=cfg['MODEL']['ACTIVATION'],
                                   sigma_neu=cfg['MODEL']['SIGMA_NEU'],
                                   sigma_syn=cfg['MODEL']['SIGMA_SYN'],
                                   use_bias=cfg['MODEL']['USE_BIAS'],
                                   anti_hebbian=cfg['MODEL']['ANTI_HEBB']).to(device)

    train_dataset = FreqDataset(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                                time_scale=cfg['MODEL']['ALPHA'],
                                freq_min=cfg['DATALOADER']['FREQ_MIN'],
                                freq_max=cfg['DATALOADER']['FREQ_MAX'],
                                min_interval=cfg['DATALOADER']['MIN_INTERVAL'],
                                signal_length=cfg['DATALOADER']['SIGNAL_LENGTH'],
                                variable_signal_length=cfg['DATALOADER']['VARIABLE_SIGNAL_LENGTH'],
                                sigma_in=cfg['DATALOADER']['SIGMA_IN'],
                                delay_variable=cfg['DATALOADER']['VARIABLE_DELAY'])

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
    if 'PHASE_TRANSIT' in cfg['TRAIN'].keys():
        phase_transition_criteria = cfg['TRAIN']['PHASE_TRANSIT']
    else:
        phase_transition_criteria = [0.5, 0.45, 0.4, 0.3, 0.2]
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
                                      int(cfg['DATALOADER']['TIME_LENGTH'] - 2 * cfg['DATALOADER']['SIGNAL_LENGTH']),
                                      cfg['MODEL']['SIZE']]).float().to(device)
            active_norm = torch.nn.MSELoss()(hidden_list[:,
                                             cfg['DATALOADER']['SIGNAL_LENGTH']:
                                             cfg['DATALOADER']['TIME_LENGTH']-cfg['DATALOADER']['SIGNAL_LENGTH'], :],
                                             dummy_zero)

            loss += cfg['TRAIN']['ACTIVATION_LAMBDA'] * active_norm
            loss.backward()
            optimizer.step()
            correct += (np.argmax(output[:, -1].cpu().detach().numpy(),
                                  axis=1) == target.cpu().detach().numpy()).sum().item()
            num_data += target.cpu().detach().numpy().shape[0]

            if not phase2 and float(loss.item()) < phase_transition_criteria[0]:
                cfg['MODEL']['ALPHA'] = 0.2
                cfg['DATALOADER']['TIME_LENGTH'] = 75
                cfg['DATALOADER']['SIGNAL_LENGTH'] = 20
                cfg['DATALOADER']['VARIABLE_DELAY'] = 6

                print("phase2 start! cfg['MODEL']['ALPHA'] = 0.2")
                phase2 = True
                model.change_alpha(cfg['MODEL']['ALPHA'])
                train_dataset = FreqDataset(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                                            time_scale=cfg['MODEL']['ALPHA'],
                                            freq_min=cfg['DATALOADER']['FREQ_MIN'],
                                            freq_max=cfg['DATALOADER']['FREQ_MAX'],
                                            min_interval=cfg['DATALOADER']['MIN_INTERVAL'],
                                            signal_length=cfg['DATALOADER']['SIGNAL_LENGTH'],
                                            variable_signal_length=cfg['DATALOADER']['VARIABLE_SIGNAL_LENGTH'],
                                            sigma_in=cfg['DATALOADER']['SIGMA_IN'],
                                            delay_variable=cfg['DATALOADER']['VARIABLE_DELAY'])
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                                                               num_workers=2, shuffle=True,
                                                               worker_init_fn=lambda x: np.random.seed())
                break

            if not phase3 and float(loss.item()) < phase_transition_criteria[1]:
                cfg['MODEL']['ALPHA'] = 0.175
                cfg['DATALOADER']['TIME_LENGTH'] = 82
                cfg['DATALOADER']['SIGNAL_LENGTH'] = 22
                cfg['DATALOADER']['VARIABLE_DELAY'] = 7

                print("phase3 start! cfg['MODEL']['ALPHA'] = 0.175")
                phase3 = True
                model.change_alpha(cfg['MODEL']['ALPHA'])
                train_dataset = FreqDataset(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                                            time_scale=cfg['MODEL']['ALPHA'],
                                            freq_min=cfg['DATALOADER']['FREQ_MIN'],
                                            freq_max=cfg['DATALOADER']['FREQ_MAX'],
                                            min_interval=cfg['DATALOADER']['MIN_INTERVAL'],
                                            signal_length=cfg['DATALOADER']['SIGNAL_LENGTH'],
                                            variable_signal_length=cfg['DATALOADER']['VARIABLE_SIGNAL_LENGTH'],
                                            sigma_in=cfg['DATALOADER']['SIGMA_IN'],
                                            delay_variable=cfg['DATALOADER']['VARIABLE_DELAY'])
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                                                               num_workers=2, shuffle=True,
                                                               worker_init_fn=lambda x: np.random.seed())
                break

            if not phase4 and float(loss.item()) < phase_transition_criteria[2]:
                cfg['MODEL']['ALPHA'] = 0.15
                cfg['DATALOADER']['TIME_LENGTH'] = 90
                cfg['DATALOADER']['SIGNAL_LENGTH'] = 25
                cfg['DATALOADER']['VARIABLE_DELAY'] = 8

                print("phase4 start! cfg['MODEL']['ALPHA'] = 0.15")
                phase4 = True
                model.change_alpha(cfg['MODEL']['ALPHA'])
                train_dataset = FreqDataset(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                                            time_scale=cfg['MODEL']['ALPHA'],
                                            freq_min=cfg['DATALOADER']['FREQ_MIN'],
                                            freq_max=cfg['DATALOADER']['FREQ_MAX'],
                                            min_interval=cfg['DATALOADER']['MIN_INTERVAL'],
                                            signal_length=cfg['DATALOADER']['SIGNAL_LENGTH'],
                                            variable_signal_length=cfg['DATALOADER']['VARIABLE_SIGNAL_LENGTH'],
                                            sigma_in=cfg['DATALOADER']['SIGMA_IN'],
                                            delay_variable=cfg['DATALOADER']['VARIABLE_DELAY'])
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                                                               num_workers=2, shuffle=True,
                                                               worker_init_fn=lambda x: np.random.seed())
                break

            if not phase5 and float(loss.item()) < phase_transition_criteria[3]:
                cfg['MODEL']['ALPHA'] = 0.10
                cfg['DATALOADER']['TIME_LENGTH'] = 120
                cfg['DATALOADER']['SIGNAL_LENGTH'] = 30
                cfg['DATALOADER']['VARIABLE_DELAY'] = 10

                print("phase5 start! cfg['MODEL']['ALPHA'] = 0.1")
                phase5 = True
                model.change_alpha(cfg['MODEL']['ALPHA'])
                train_dataset = FreqDataset(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                                            time_scale=cfg['MODEL']['ALPHA'],
                                            freq_min=cfg['DATALOADER']['FREQ_MIN'],
                                            freq_max=cfg['DATALOADER']['FREQ_MAX'],
                                            min_interval=cfg['DATALOADER']['MIN_INTERVAL'],
                                            signal_length=cfg['DATALOADER']['SIGNAL_LENGTH'],
                                            variable_signal_length=cfg['DATALOADER']['VARIABLE_SIGNAL_LENGTH'],
                                            sigma_in=cfg['DATALOADER']['SIGMA_IN'],
                                            delay_variable=cfg['DATALOADER']['VARIABLE_DELAY'])
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                                                               num_workers=2, shuffle=True,
                                                               worker_init_fn=lambda x: np.random.seed())
                break

            if not phase6 and float(loss.item()) < phase_transition_criteria[4]:
                cfg['MODEL']['ALPHA'] = 0.075
                cfg['DATALOADER']['TIME_LENGTH'] = 200
                cfg['DATALOADER']['SIGNAL_LENGTH'] = 50
                cfg['DATALOADER']['VARIABLE_DELAY'] = 15

                print("phase6 start! cfg['MODEL']['ALPHA'] = 0.075")
                phase6 = True
                model.change_alpha(cfg['MODEL']['ALPHA'])
                train_dataset = FreqDataset(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                                            time_scale=cfg['MODEL']['ALPHA'],
                                            freq_min=cfg['DATALOADER']['FREQ_MIN'],
                                            freq_max=cfg['DATALOADER']['FREQ_MAX'],
                                            min_interval=cfg['DATALOADER']['MIN_INTERVAL'],
                                            signal_length=cfg['DATALOADER']['SIGNAL_LENGTH'],
                                            variable_signal_length=cfg['DATALOADER']['VARIABLE_SIGNAL_LENGTH'],
                                            sigma_in=cfg['DATALOADER']['SIGMA_IN'],
                                            delay_variable=cfg['DATALOADER']['VARIABLE_DELAY'])
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                                                               num_workers=2, shuffle=True,
                                                               worker_init_fn=lambda x: np.random.seed())
                break

        if epoch % cfg['TRAIN']['DISPLAY_EPOCH'] == 0:
            acc = correct / num_data
            print(f'{epoch}, {loss.item():.6f}, {acc:.6f}')
            print(f'activation norm: {active_norm.item():.4f}, time scale: , '
                  f'{model.alpha.detach().cpu().numpy()[0]:.3f}')
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
