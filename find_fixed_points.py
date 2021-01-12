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
from fixed_point_analyzer import FixedPoint
from model import RecurrentNeuralNetwork


def main(config_path, model_epoch):
    # hyper-parameter
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_name = os.path.splitext(os.path.basename(config_path))[0]

    # save path
    os.makedirs('fixed_points', exist_ok=True)
    os.makedirs('fixed_points/freq', exist_ok=True)
    save_path = f'fixed_points/freq/{model_name}'
    os.makedirs(save_path, exist_ok=True)

    # copy config file
    shutil.copyfile(config_path, os.path.join(save_path, os.path.basename(config_path)))

    use_cuda = cfg['MACHINE']['CUDA'] and torch.cuda.is_available()
    torch.manual_seed(cfg['MACHINE']['SEED'])
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(device)

    if 'ALPHA' not in cfg['MODEL'].keys():
        cfg['MODEL']['ALPHA'] = 0.25

    # cfg['DATALOADER']['TIME_LENGTH'] = 200
    # cfg['DATALOADER']['SIGNAL_LENGTH'] = 50
    cfg['DATALOADER']['VARIABLE_DELAY'] = 0

    # model load
    model = RecurrentNeuralNetwork(n_in=1, n_out=2, n_hid=cfg['MODEL']['SIZE'], device=device,
                                   alpha_time_scale=cfg['MODEL']['ALPHA'], beta_time_scale=cfg['MODEL']['BETA'],
                                   activation=cfg['MODEL']['ACTIVATION'],
                                   sigma_neu=cfg['MODEL']['SIGMA_NEU'],
                                   sigma_syn=cfg['MODEL']['SIGMA_SYN'],
                                   use_bias=cfg['MODEL']['USE_BIAS'],
                                   anti_hebbian=cfg['MODEL']['ANTI_HEBB']).to(device)

    model_path = f'../trained_model/freq/{model_name}/epoch_{model_epoch}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    eval_dataset = FreqDataset(time_length=cfg['DATALOADER']['TIME_LENGTH'],
                               time_scale=cfg['MODEL']['ALPHA'],
                               freq_min=cfg['DATALOADER']['FREQ_MIN'],
                               freq_max=cfg['DATALOADER']['FREQ_MAX'],
                               min_interval=cfg['DATALOADER']['MIN_INTERVAL'],
                               signal_length=cfg['DATALOADER']['SIGNAL_LENGTH'],
                               variable_signal_length=cfg['DATALOADER']['VARIABLE_SIGNAL_LENGTH'],
                               sigma_in=cfg['DATALOADER']['SIGMA_IN'],
                               delay_variable=cfg['DATALOADER']['VARIABLE_DELAY'])

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=cfg['TRAIN']['BATCHSIZE'],
                                                  num_workers=2, shuffle=True,
                                                  worker_init_fn=lambda x: np.random.seed())

    analyzer = FixedPoint(model=model, device=device, alpha=cfg['MODEL']['ALPHA'],
                          max_epochs=140000)

    for trial in range(50):
        for i, data in enumerate(eval_dataloader):
            inputs, target = data
            # print(inputs.shape)
            inputs, target = inputs.float(), target.long()
            inputs, target = Variable(inputs).to(device), Variable(target).to(device)

            hidden = torch.zeros(cfg['TRAIN']['BATCHSIZE'], cfg['MODEL']['SIZE'])
            hidden = hidden.to(device)

            hidden = hidden.detach()
            hidden_list, output, hidden, _ = model(inputs, hidden)

            const_signal = torch.tensor([0] * 1)
            const_signal = const_signal.float().to(device)

            reference_time_point = np.random.randint(35, 55)
            fixed_point, result_ok = analyzer.find_fixed_point(hidden_list[0, reference_time_point], view=True)

            fixed_point = fixed_point.detach().cpu().numpy()

            print(fixed_point)
            fixed_point_tensor = torch.from_numpy(fixed_point).float()
            jacobian = analyzer.calc_jacobian(fixed_point_tensor, const_signal)

            # print(np.dot(model.w_out.weight.detach().cpu().numpy(), fixed_point))

            w, v = np.linalg.eig(jacobian)
            # print('eigenvalues', w)

            np.savetxt(os.path.join(save_path, f'fixed_point_{trial}_{i}.txt'), fixed_point)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN fixed point analysis')
    parser.add_argument('config_path', type=str)
    parser.add_argument('--model_epoch', type=int, default=3000)
    args = parser.parse_args()
    print(args)
    main(args.config_path, args.model_epoch)
