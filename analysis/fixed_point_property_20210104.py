import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

sys.path.append('../')

from model import RecurrentNeuralNetwork

plt.style.use('seaborn-darkgrid')
import seaborn as sns

sns.set_palette('Dark2')


def calc_speed(hidden_activated, model):
    activated = torch.tanh(hidden_activated)
    tmp_hidden = model.w_hh(activated)
    tmp_hidden = (1 - model.alpha) * hidden_activated + model.alpha * tmp_hidden

    speed = torch.norm(tmp_hidden - hidden_activated)

    return speed.item()


def main(config_path):
    torch.manual_seed(1)
    device = torch.device('cpu')

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['MODEL']['SIGMA_NEU'] = 0
    model_name = os.path.splitext(os.path.basename(config_path))[0]
    print('model_name: ', model_name)

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

    fixed_point_list = np.zeros((200, 256))
    count = 0
    for i in range(50):
        for j in range(4):
            fixed_point_list[count] = np.loadtxt(f'../fixed_points/freq/{model_name}/fixed_point_{i}_{j}.txt')
            count += 1

    speed_list = []
    for i in range(200):
        speed_list.append(calc_speed(torch.from_numpy(fixed_point_list[i]).float(), model))

    print('average speed: ', np.mean(speed_list))

    pca = PCA(n_components=3)
    pca.fit(fixed_point_list)

    pc_fixed_point_list = pca.transform(fixed_point_list)

    fig = plt.figure(figsize=(7, 6))
    ax = Axes3D(fig)
    ax.view_init(elev=25, azim=20)

    # 軸ラベルの設定
    ax.set_xlabel('PC1', fontsize=14)
    ax.set_ylabel('PC2', fontsize=14)
    ax.set_zlabel('PC3', fontsize=14)

    ax.scatter(
        pc_fixed_point_list[:, 0],
        pc_fixed_point_list[:, 1],
        pc_fixed_point_list[:, 2]
    )

    plt.savefig(f'results/{model_name}/fixed_points.png', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN training')
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()
    print(args)
    main(args.config_path)
