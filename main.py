import argparse

from tqdm import trange

import torch
from torch import nn, optim
from resnet import resnet50


torch.set_default_tensor_type(torch.cuda.FloatTensor)


def main(flags):
    '''
    fit model to random data
    '''
    net = resnet50(
        weight_bit_width=flags.weight_bit_width,
        act_bit_width=flags.act_bit_width
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=flags.lr)

    pbar = trange(flags.steps, unit='sample', unit_scale=flags.batch_size)
    for i in pbar:
        optimizer.zero_grad()
        X = torch.randn(flags.batch_size, 3, 224, 224)
        y = torch.randint(1000, size=(flags.batch_size,))

        y_hat = net(X)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weight_bit_width',
        type=int,
        default=4,
        help='Bit width for model weights'
    )
    parser.add_argument(
        '--act_bit_width',
        type=int,
        default=4,
        help='Bit width for activations'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='Samples per update'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=100,
        help='Number of updates'
    )
    flags = parser.parse_args()
    main(flags)
