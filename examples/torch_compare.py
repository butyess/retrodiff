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
    out = np.array([[0, 10]])
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

    diffs = [np.sum((gnp - gt.detach().numpy())**2) for gnp, gt in zip(grads, grads_t)]

    print('Out:')
    print(out)
    print(out_t)
    print('\nGrads:')
    print(grads[0])
    print(grads_t[0])
    print('\nDiffs:')
    print(diffs)


def net_and_loss():
    x_init = [[0., 0.]]
    exp_init = [0]
    layers = [2, 8, 8, 2]

    net = Network(layers)
    x = np.array(x_init, dtype=np.int)
    exp = np.array(exp_init)
    out = net.evaluate(x)
    loss = net.loss(out, exp)
    loss_grad = net._loss_dag.backward(1)
    grads = net._dag.backward(loss_grad[0])

    x_t = torch.tensor(x_init, dtype=torch.float, requires_grad=True)
    exp_t = torch.tensor(exp_init)
    ws = [torch.tensor(w, dtype=torch.float, requires_grad=True) for w in net.parameters[:len(layers) - 1]]
    bs = [torch.tensor(b, dtype=torch.float, requires_grad=True) for b in net.parameters[len(layers) - 1:]]
    out_t = reduce(lambda acc, x: relu(acc @ x[0] + x[1]), zip(ws[:-1], bs[:-1]), x_t) @ ws[-1] + bs[-1]
    loss_t = multi_margin_loss(out_t, exp_t)
    loss_t.backward()
    grads_t = [x_t.grad] + [p.grad for p in ws + bs]

    # print('outputs:\n', 'retro: ', out, '\ntorch: ', out_t, '\n')
    # print('loss:\n', 'retro: ', loss, '\ntorch: ', loss_t, '\n')
    # print('grads[-1]:\n', 'retro: ', grads[-1], '\ntorch: ', grads_t[-1], '\n')
    # print('grads shapes:\n', 'retro: ', [g.shape for g in grads], '\ntorch: ', [g.shape for g in grads_t] , '\n')
    # print(out_t.retain_grad())

    print('outputs:')
    print('- retro:', out, '\n- torch:', out_t)

    print('loss:')
    print('- retro:', loss, '\n- torch:', loss_t)

    print('grads:')
    print('- retro[-1]:', grads[-1], '\n- torch[-1]:', grads_t[-1])
    grads_diff = [np.sum((g_r - g_t.detach().numpy())**2) for g_r, g_t in zip(grads, grads_t)]
    print('- diffs:', grads_diff)

    print('grads shapes:')
    print('- retro:', [g.shape for g in grads], '\n- torch:', [g.shape for g in grads_t] )
    print(out_t.retain_grad())

    diffs = [np.sum((params - params_t.detach().numpy())**2) for params, params_t in zip(net.parameters, ws+bs)]
    print('parameters differences:', diffs)
    print()


def net_and_loss_optim():
    x_init = [[0., 0.]]
    exp_init = [0]
    layers = [2, 8, 8, 2]

    net = Network(layers)
    x = np.array(x_init, dtype=np.int)
    exp = np.array(exp_init)
    out = net.evaluate(x)
    loss = net.loss(out, exp)
    loss_grad = net._loss_dag.backward(1)
    grads = net._dag.backward(loss_grad[0])

    x_t = torch.tensor(x_init, dtype=torch.float, requires_grad=True)
    exp_t = torch.tensor(exp_init)
    ws = [torch.tensor(w, dtype=torch.float, requires_grad=True) for w in net.parameters[:len(layers) - 1]]
    bs = [torch.tensor(b, dtype=torch.float, requires_grad=True) for b in net.parameters[len(layers) - 1:]]
    out_t = reduce(lambda acc, x: relu(acc @ x[0] + x[1]), zip(ws[:-1], bs[:-1]), x_t) @ ws[-1] + bs[-1]
    loss_t = multi_margin_loss(out_t, exp_t)
    loss_t.backward()
    grads_t = [x_t.grad] + [p.grad for p in ws + bs]

    print(f'ITERATION 0')
    print('-'*30)

    print('outputs:')
    print('- retro:', out, '\n- torch:', out_t)

    print('loss:')
    print('- retro:', loss, '\n- torch:', loss_t)

    print('grads:')
    print('- retro[-1]:', grads[-1], '\n- torch[-1]:', grads_t[-1])
    # grads_diff = [np.sum((g_r - g_t.detach().numpy())**2) for g_r, g_t in zip(grads, grads_t)]
    grads_isclose = [np.isclose(g_r, g_t.detach().numpy()) for g_r, g_t in zip(grads, grads_t)]
    print('- diffs:', grads_isclose)

    print('grads shapes:')
    print('- retro:', [g.shape for g in grads], '\n- torch:', [g.shape for g in grads_t] )
    print(out_t.retain_grad())

    diffs = [np.sum((params - params_t.detach().numpy())**2) for params, params_t in zip(net.parameters, ws+bs)]
    params_isclose = [np.isclose(params, params_t.detach().numpy()) for params, params_t in zip(net.parameters, ws+bs)]
    print('parameters differences:', params_isclose)
    print()

    optim = GradientDescent(lr=1)
    optim_t = SGD(ws + bs, lr=1)

    for i in range(10):
        out = net.evaluate(x)
        loss = net.loss(out, exp)
        loss_grad = net._loss_dag.backward(1)
        grads = net._dag.backward(loss_grad[0])
        net.parameters = optim.new_param(net.parameters, grads[1:])

        optim_t.zero_grad()
        out_t = reduce(lambda acc, x: relu(acc @ x[0] + x[1]), zip(ws[:-1], bs[:-1]), x_t) @ ws[-1] + bs[-1]
        loss_t = multi_margin_loss(out_t, exp_t)
        loss_t.backward()
        optim_t.step()
        grads_t = [x_t.grad] + [p.grad for p in ws + bs]

        print(f'ITERATION {i+1}')
        print('-'*30)

        print('outputs:')
        print('- retro:', out, '\n- torch:', out_t)

        print('loss:')
        print('- retro:', loss, '\n- torch:', loss_t)

        print('grads:')
        print('- retro[-1]:', grads[-1], '\n- torch[-1]:', grads_t[-1])
        # grads_diff = [np.sum((g_r - g_t.detach().numpy())**2) for g_r, g_t in zip(grads, grads_t)]
        grads_isclose = [np.isclose(g_r, g_t.detach().numpy()) for g_r, g_t in zip(grads, grads_t)]
        print('- diffs:', grads_isclose)

        print('grads shapes:')
        print('- retro:', [g.shape for g in grads], '\n- torch:', [g.shape for g in grads_t] )
        print(out_t.retain_grad())

        # params_diffs = [np.sum((params - params_t.detach().numpy())**2) for params, params_t in zip(net.parameters, ws+bs)]
        params_isclose = [np.isclose(params, params_t.detach().numpy()) for params, params_t in zip(net.parameters, ws+bs)]
        print('parameters differences:', params_isclose)
        print()


def just_optim():
    init_params = [100, 100, 100]
    init_grads = [1, 1, 1]

    optim = GradientDescent(lr=1)
    param = np.array(init_params)
    grad = np.array(init_grads)
    for i in range(100):
        param = optim.new_param([param], [grad])[0]

    param_t = torch.tensor(init_params, dtype=torch.float, requires_grad=True)
    param_t.grad = torch.tensor(init_grads, dtype=torch.float)
    optim_t = SGD([param_t], lr=1)
    for i in range(100):
        optim_t.zero_grad()
        param_t.grad = torch.tensor(grad, dtype=torch.float)
        optim_t.step()

    print(param)
    print(param_t)


if __name__ == '__main__':
    net_and_loss_optim()
