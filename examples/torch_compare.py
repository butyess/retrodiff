from functools import reduce

import numpy as np

import torch
from torch.nn.functional import relu, multi_margin_loss
from torch.optim import SGD

from retrodiff.utils import GradientDescent

from classify import Network, SVMBinaryLoss


np.set_printoptions(precision=4)


def just_loss():
    loss_fn = SVMBinaryLoss()
    out = np.array([[1, 1]])
    labl = np.array([0])
    loss = loss_fn.forward(out, labl)
    grad = loss_fn.backward(1, 0, out, labl)

    out_t = torch.tensor(out, dtype=torch.float, requires_grad=True)
    labl_t = torch.tensor(labl)
    loss_t = multi_margin_loss(out_t, labl_t)
    loss_t.backward()
    grad_t = out_t.grad

    print('Loss:')
    print(loss)
    print(loss_t)
    print('\nGrads:')
    print(grad)
    print(grad_t)


def just_net():
    layers = [2, 16, 1]

    net = Network(layers)
    x = np.array([[1, 2]])
    out = net.evaluate(x)
    grads = net._dag.backward(np.array([[1]]))

    x_t = torch.tensor([[1., 2.]], requires_grad=True)
    ws = [torch.tensor(w, dtype=torch.float, requires_grad=True) for w in net.parameters[:len(layers) - 1]]
    bs = [torch.tensor(b, dtype=torch.float, requires_grad=True) for b in net.parameters[len(layers) - 1:]]
    out_t = reduce(lambda acc, x: relu(acc @ x[0] + x[1]), zip(ws[:-1], bs[:-1]), x_t) @ ws[-1] + bs[-1]
    out_t.backward()
    grads_t = [x_t.grad] + [p.grad for p in ws + bs]

    diffs = [np.sum(gnp - gt.detach().numpy()) for gnp, gt in zip(grads, grads_t)]

    print('Out:')
    print(out)
    print(out_t)
    print('\nGrads:')
    print(grads[0])
    print(grads_t[0])
    print('\nDiffs:')
    print(diffs)


def net_and_loss():
    layers = [2, 16, 2]

    net = Network(layers)
    x = np.array([[1, 2]])
    exp = np.array([0])
    out = net.evaluate(x)
    loss = net.loss(out, exp)
    loss_grad = net._loss_dag.backward(1)
    grads = net._dag.backward(loss_grad[0])

    x_t = torch.tensor([[1., 2.]], requires_grad=True)
    exp_t = torch.tensor([0])
    ws = [torch.tensor(w, dtype=torch.float, requires_grad=True) for w in net.parameters[:len(layers) - 1]]
    bs = [torch.tensor(b, dtype=torch.float, requires_grad=True) for b in net.parameters[len(layers) - 1:]]
    out_t = reduce(lambda acc, x: relu(acc @ x[0] + x[1]), zip(ws[:-1], bs[:-1]), x_t) @ ws[-1] + bs[-1]
    loss_t = multi_margin_loss(out_t, exp_t)
    loss_t.backward()
    grads_t = [x_t.grad] + [p.grad for p in ws + bs]

    # print('outputs:\n', 'retro: ', out, '\ntorch: ', out_t, '\n'*2)
    # print('loss:\n', 'retro: ', loss, '\ntorch: ', loss_t, '\n'*2)

    print([g.shape for g in grads])
    print([g.shape for g in grads_t])

    print(grads[-1])
    print(grads_t[-1])

    print(out_t.retain_grad())


def just_optim():
    optim = GradientDescent(lr=0.001)
    param = np.array([1, 1, 1])
    grad = np.array([1, 1, 1])
    new_params = optim.new_param([param], [grad])

    param_t = torch.tensor(param, dtype=torch.float, requires_grad=True)
    param_t.grad = torch.tensor(grad, dtype=torch.float)
    optim_t = SGD([param_t], lr=0.001)
    optim_t.step()

    print(new_params)
    print(param_t)


if __name__ == '__main__':
    just_optim()
