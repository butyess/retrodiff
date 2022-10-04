from collections import OrderedDict
from socket import if_nameindex

import numpy as np

import torch
import torch.nn as nn

from retrodiff.utils.function import MSELossFun, ReLU
from retrodiff.utils.nn import Sequential, Loss, Linear
from retrodiff.utils.optim import GradientDescent



def loss_optim_torch():
    net = nn.Sequential(OrderedDict([
        ('one', nn.Linear(1, 1)),
        ('two', nn.Linear(1, 1))
    ]))

    net.one.weight = torch.nn.Parameter(torch.tensor([[2.0]], requires_grad=True))
    net.two.weight = torch.nn.Parameter(torch.tensor([[3.0]], requires_grad=True))
    net.one.bias = torch.nn.Parameter(torch.tensor([[0.0]], requires_grad=True))
    net.two.bias = torch.nn.Parameter(torch.tensor([[0.0]], requires_grad=True))

    loss = nn.MSELoss()
    optim = torch.optim.SGD(net.parameters(), 1)

    x = torch.tensor([[1.0]])
    p = torch.tensor([[5.0]])

    y = net(x)
    l = loss(y, p)
    l.backward()

    print('before:')
    print(net.one.weight.item())
    print(net.one.bias.item())
    print(net.two.weight.item())
    print(net.two.bias.item())

    optim.step()

    print('\nafter:')
    print(net.one.weight.item())
    print(net.one.bias.item())
    print(net.two.weight.item())
    print(net.two.bias.item())

    print('\nloss: ', l.item())


def loss_optim_retrodiff():
    mse = MSELossFun()
    relu = ReLU()

    net = Sequential(Linear(relu, np.array(2.0), np.array(0.0)),
                    Linear(relu, np.array(3.0), np.array(0.0)))
    loss = Loss(mse, net)

    l = loss.forward(np.array(1.0), np.array(5.0))
    loss.backward()

    print('before:')
    print(', '.join(map(str, net.weights)))

    optim = GradientDescent(net, 1)
    optim.step()

    print('\nafter:')
    print(', '.join(map(str, net.weights)))

    print('\nloss: ', l)


if __name__ == '__main__':
    print('RETRODIFF\n' + '-'*10)
    loss_optim_retrodiff()
    print('\nTORCH\n' + '-'*10)
    loss_optim_torch()
