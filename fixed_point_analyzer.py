"""class for fixed point analysis"""

import torch
from torch.autograd import Variable
import numpy as np


class FixedPoint(object):
    def __init__(self, model, device, alpha, gamma=0.01, speed_tor=1e-10, max_epochs=1600000,
                 lr_decay_epoch=1000):
        self.model = model
        self.device = device
        self.gamma = gamma
        self.speed_tor = speed_tor
        self.max_epochs = max_epochs
        self.lr_decay_epoch = lr_decay_epoch
        self.alpha = torch.ones(256) * alpha
        self.alpha = self.alpha.to(device)
        self.model.eval()

    def calc_speed(self, hidden_activated):
        activated = torch.tanh(hidden_activated)
        tmp_hidden = self.model.w_hh(activated)
        tmp_hidden = (1 - self.alpha) * hidden_activated + self.alpha * tmp_hidden

        speed = torch.norm(tmp_hidden - hidden_activated)

        return speed

    def find_fixed_point(self, init_hidden, view=False):
        new_hidden = init_hidden.clone()
        gamma = self.gamma
        result_ok = True
        i = 0
        while True:
            hidden_activated = Variable(new_hidden).to(self.device)
            hidden_activated.requires_grad = True
            speed = self.calc_speed(hidden_activated)
            if view and i % 5000 == 0:
                print(f'epoch: {i}, speed={speed.item()}')
            if speed.item() < self.speed_tor:
                print(f'epoch: {i}, speed={speed.item()}')
                break
            speed.backward()
            if i % self.lr_decay_epoch == 0 and 0 < i:
                gamma *= 0.9
            if i == self.max_epochs:
                print(f'forcibly finished. speed={speed.item()}')
                result_ok = False
                break
            i += 1

            new_hidden = hidden_activated - gamma * hidden_activated.grad

        fixed_point = new_hidden
        return fixed_point, result_ok

    def calc_jacobian(self, fixed_point, const_signal):
        # print('calculation!')
        fixed_point = torch.unsqueeze(fixed_point, dim=1)
        fixed_point = Variable(fixed_point).to(self.device)
        fixed_point.requires_grad = True

        w_hh = self.model.w_hh.weight
        w_hh.requires_grad = False
        w_hh = w_hh.to(self.device)
        activated = torch.tanh(fixed_point)
        # activated = torch.unsqueeze(activated, dim=1)
        # print('activated', activated.shape)
        tmp_hidden = torch.unsqueeze(self.model.w_in(const_signal), dim=1) + w_hh @ activated

        activated = tmp_hidden - fixed_point

        jacobian = torch.zeros(self.model.n_hid, self.model.n_hid)
        for i in range(self.model.n_hid):
            # print(i)
            output = torch.zeros(self.model.n_hid, 1).to(self.device)
            output[i] = 1.
            jacobian[:, i:i + 1] = torch.autograd.grad(activated, fixed_point, grad_outputs=output, retain_graph=True)[
                0]

        jacobian = jacobian.cpu().numpy().T

        return jacobian

    def calc_jacobian2(self, fixed_point):
        tanh_dash = 1 - np.tanh(fixed_point) ** 2

        w_hh = self.model.w_hh.weight.data.numpy()
        jacobian = np.zeros((self.model.n_hid, self.model.n_hid))
        for i in range(self.model.n_hid):
            for j in range(self.model.n_hid):
                jacobian[i, j] = tanh_dash[j] * w_hh[i, j]
                if i == j:
                    jacobian[i, j] -= 1

        return jacobian


class FixedPoint2(object):
    def __init__(self, model, device, alpha, gamma=0.01, speed_tor=1e-10, max_epochs=1600000,
                 lr_decay_epoch=1000):
        self.model = model
        self.device = device
        self.gamma = gamma
        self.speed_tor = speed_tor
        self.max_epochs = max_epochs
        self.lr_decay_epoch = lr_decay_epoch
        self.alpha = torch.ones(256) * alpha
        self.alpha = self.alpha.to(device)
        self.model.eval()

    def calc_speed(self, hidden_activated):
        activated = torch.tanh(hidden_activated)
        tmp_hidden = self.model.w_hh(activated)
        tmp_hidden = (1 - self.alpha) * hidden_activated + self.alpha * tmp_hidden

        speed = torch.norm(tmp_hidden - hidden_activated)

        return speed

    def find_fixed_point(self, init_hidden, view=False):
        new_hidden = init_hidden.clone()
        gamma = self.gamma
        result_ok = True
        i = 0
        while True:
            hidden_activated = Variable(new_hidden).to(self.device)
            hidden_activated.requires_grad = True
            speed = self.calc_speed(hidden_activated)
            if view and i % 5000 == 0:
                print(f'epoch: {i}, speed={speed.item()}')
            if speed.item() < self.speed_tor:
                print(f'epoch: {i}, speed={speed.item()}')
                break
            speed.backward()
            if i % self.lr_decay_epoch == 0 and 0 < i:
                gamma *= 0.9
            if i == self.max_epochs:
                print(f'forcibly finished. speed={speed.item()}')
                result_ok = False
                break
            i += 1

            new_hidden = hidden_activated - gamma * (hidden_activated.grad +
                                                     torch.randn_like(new_hidden).to(self.device) * 0.0001)

        fixed_point = new_hidden
        return fixed_point, result_ok

    def calc_jacobian(self, fixed_point, const_signal):
        # print('calculation!')
        fixed_point = torch.unsqueeze(fixed_point, dim=1)
        fixed_point = Variable(fixed_point).to(self.device)
        fixed_point.requires_grad = True

        w_hh = self.model.w_hh.weight
        w_hh.requires_grad = False
        w_hh = w_hh.to(self.device)
        activated = torch.tanh(fixed_point)
        # activated = torch.unsqueeze(activated, dim=1)
        # print('activated', activated.shape)
        tmp_hidden = torch.unsqueeze(self.model.w_in(const_signal), dim=1) + w_hh @ activated

        activated = tmp_hidden - fixed_point

        jacobian = torch.zeros(self.model.n_hid, self.model.n_hid)
        for i in range(self.model.n_hid):
            # print(i)
            output = torch.zeros(self.model.n_hid, 1).to(self.device)
            output[i] = 1.
            jacobian[:, i:i + 1] = torch.autograd.grad(activated, fixed_point, grad_outputs=output, retain_graph=True)[
                0]

        jacobian = jacobian.cpu().numpy().T

        return jacobian

    def calc_jacobian2(self, fixed_point):
        tanh_dash = 1 - np.tanh(fixed_point) ** 2

        w_hh = self.model.w_hh.weight.data.numpy()
        jacobian = np.zeros((self.model.n_hid, self.model.n_hid))
        for i in range(self.model.n_hid):
            for j in range(self.model.n_hid):
                jacobian[i, j] = tanh_dash[j] * w_hh[i, j]
                if i == j:
                    jacobian[i, j] -= 1

        return jacobian
