from socket import if_nameindex
from retrodiff.utils.nn import Linear, Sequential, Recurrent
from retrodiff.utils.function import ReLU, Mul, Add


def lin_seq():
    print('Linear and Sequential:')
    relu = ReLU()

    lin1 = Linear(relu)
    lin1.weights = [3, 5]

    lin2 = Linear(relu)
    lin2.weights = [2, 0]

    seq = Sequential(lin1, lin2)

    print(lin1.run(10))
    print(lin2.run(lin1.run(10)))
    print(seq.run(10))


def rnn():
    print('Recurrent:')
    mul, add = Mul(), Add()
    f = lambda x, w, y: add(mul(x, w), y)

    rnnl = Recurrent(f)
    rnnl.weights = [1]

    out = rnnl.run(0, 1, 2, 3)
    e = ((1 + 0) + 2) + 3

    print(out)
    print(out[0] == e)


if __name__ == '__main__':
    lin_seq()
    rnn()
