import sys
sys.path.append('/home/i-chenyunpeng/zhoubin/incubator-mxnet/python')
import mxnet as mx


def ConvFactory(data, kernel, stride, pad, num_filter):
    act = mx.symbol.Activation(data=data, act_type='relu')
    conv = mx.symbol.Convolution(data=act, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    bn = mx.symbol.BatchNorm(data=conv)
    return bn

def ConvFactorySep(data, kernel, stride, pad, num_filter):
    act = mx.symbol.Activation(data=data, act_type='relu')
    conv_dw = mx.symbol.Convolution(data=act, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, num_group=num_filter)
    conv_pw = mx.symbol.Convolution(data=conv_dw, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0))
    bn = mx.symbol.BatchNorm(data=conv_pw)
    return bn


def normal_cell(h1, h2, num_filter):

    # Sep 3x3 + id
    b1 = ConvFactorySep(data=h1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=num_filter)
    b1 = ConvFactorySep(data=b1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=num_filter)
    b1x = mx.sym.identity(data=h1)
    b1 = b1 + b1x

    # sep 5x5 + sep 3x3
    b2 = ConvFactorySep(data=h1, kernel=(5, 5), stride=(1, 1), pad=(2, 2), num_filter=num_filter)
    b2 = ConvFactorySep(data=b2, kernel=(5, 5), stride=(1, 1), pad=(2, 2), num_filter=num_filter)
    b2x = ConvFactorySep(data=h2, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=num_filter)
    b2x = ConvFactorySep(data=b2x, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=num_filter)
    b2 = b2 + b2x

    # avg 3x3 + idx
    b3 = mx.sym.Pooling(data=h1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='avg')
    b3 = b3 + mx.sym.identity(data=h2)

    # avg 3x3 + avg 3x3
    b4 = mx.sym.Pooling(data=h2, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='avg')
    b4x = mx.sym.Pooling(data=h2, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='avg')
    b4 = b4 + b4x

    # sep 5x5 + sep 3x3
    b5 = ConvFactorySep(data=h2, kernel=(5, 5), stride=(1, 1), pad=(2, 2), num_filter=num_filter)
    b5 = ConvFactorySep(data=b5, kernel=(5, 5), stride=(1, 1), pad=(2, 2), num_filter=num_filter)
    b5x = ConvFactorySep(data=h2, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=num_filter)
    b5x = ConvFactorySep(data=b5x, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=num_filter)
    b5 = b5 + b5x

    concat = mx.sym.Concat(*[b1, b2, b3, b4, b5])
    out = mx.sym.Convolution(data=concat, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_filter=num_filter)
    return out

def reduction_cell(h1, h2, num_filter, num_filter_next):

    # sep 5x5 + sep 7x7
    b1 = ConvFactorySep(data=h1, kernel=(5, 5), stride=(1, 1), pad=(2, 2), num_filter=num_filter)
    b1 = ConvFactorySep(data=b1, kernel=(5, 5), stride=(1, 1), pad=(2, 2), num_filter=num_filter)

    b1x = ConvFactorySep(data=h2, kernel=(7, 7), stride=(1, 1), pad=(3, 3), num_filter=num_filter)
    b1x = ConvFactorySep(data=b1x, kernel=(7, 7), stride=(1, 1), pad=(3, 3), num_filter=num_filter)
    b1 = b1 + b1x

    # max 3x3 + sep 7x7
    b2 = ConvFactorySep(data=h2, kernel=(7, 7), stride=(1, 1), pad=(3, 3), num_filter=num_filter)
    b2 = ConvFactorySep(data=b2, kernel=(7, 7), stride=(1, 1), pad=(3, 3), num_filter=num_filter)

    b2x = mx.sym.Pooling(data=h1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='max')
    b2 = b2 + b2x

    # avg 3x3 + sep 5x5
    b3 = ConvFactorySep(data=h2, kernel=(5, 5), stride=(1, 1), pad=(2, 2), num_filter=num_filter)
    b3 = ConvFactorySep(data=b3, kernel=(5, 5), stride=(1, 1), pad=(2, 2), num_filter=num_filter)

    b3x = mx.sym.Pooling(data=h1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='avg')
    b3 = b3 + b3x


    # max 3x3 + sep 3x3
    b4x = ConvFactorySep(data=b1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=num_filter)
    b4x = ConvFactorySep(data=b4x, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=num_filter)

    b4 = mx.sym.Pooling(data=h1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='max')
    b4 = b4 + b4x


    # avg 3x3 + id
    b5 = mx.sym.Pooling(data=b1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type='avg')
    b5 = b5 + mx.sym.identity(data=b2)

    concat = mx.sym.Concat(*[b4, b5, b3])
    out = mx.sym.Convolution(data=concat, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_filter=num_filter_next)
    out = mx.sym.Pooling(data=out, kernel=(2, 2), stride=(2, 2), pool_type='avg')
    return out

def get_symbol(num_classes=10):
    num_filter_list = [16, 32, 64]
    N = 6
    data = mx.symbol.Variable('data')
    conv = mx.sym.Convolution(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=num_filter_list[0])
    h1 = mx.sym.identity(data=conv)
    h2 = mx.sym.identity(data=conv)

    # Stage 1
    for _ in range(N):
        temp = mx.sym.identity(data=h1)
        h1 = normal_cell(h1, h2, num_filter_list[0])
        h2 = mx.sym.identity(data=temp)

    h1 = reduction_cell(h1, h2, num_filter_list[0], num_filter_next=num_filter_list[1])
    h2 = mx.sym.identity(data=h1)

    # Stage 2
    for _ in range(N):
        temp = mx.sym.identity(data=h1)
        h1 = normal_cell(h1, h2, num_filter_list[1])
        h2 = mx.sym.identity(data=temp)

    h1 = reduction_cell(h1, h2, num_filter_list[1], num_filter_next=num_filter_list[2])
    h2 = mx.sym.identity(data=h1)

    # Stage 3
    for _ in range(N):
        temp = mx.sym.identity(data=h1)
        h1 = normal_cell(h1, h2, num_filter_list[2])
        h2 = mx.sym.identity(data=temp)

    pool = mx.sym.Pooling(data=h1, kernel=(8, 8), stride=(1, 1), pool_type='avg')
    flatten = mx.symbol.Flatten(data=pool)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')

    return softmax

if __name__ == '__main__':
    sym = get_symbol()
    sym.save('nasnet.json')
    mx.visualization.print_summary(sym, shape={'data':(1,3,32,32)})

