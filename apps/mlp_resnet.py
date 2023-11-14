import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    # main block. Recall that Sequential applies operations onto X sequentially
    # which is what we want (as seen in the image)
    block = nn.Sequential(ndl.nn.Linear(in_features=dim, out_features=hidden_dim),
                  norm(hidden_dim),
                  ndl.nn.ReLU(),
                  ndl.nn.Dropout(drop_prob),
                  ndl.nn.Linear(in_features=hidden_dim, out_features=dim),
                  norm(dim),
                  ndl.nn.ReLU()
                  )
    # residual link
    res = ndl.nn.Residual(block)
    # then, we need to apply RELU and to do this
    # we need to create a sequential block that has res and then Relu
    return nn.Sequential(res, ndl.nn.ReLU())


    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
