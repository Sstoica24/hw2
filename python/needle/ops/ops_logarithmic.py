from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # Z is numpy array. Always want to keep dims true
        # (you find the max over an axis, a sum over an axis...)
        max_z_original = array_api.max(Z, self.axes, keepdims=True) #ex1, (3,1,1)
        max_z_reduce = array_api.max(Z, self.axes) # ex1, (3,)
        # by not keeping dims = True, array_api.log(array_api.sum(array_api.exp(Z - max_z_original), self.axes)).shape
        # will be (3,). Thus, output is (3,)
        return array_api.log(array_api.sum(array_api.exp(Z - max_z_original), self.axes)) + max_z_reduce 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # the gradient is going to be the softmax_function
        # no need to overcomplicate and use ndl class. You can just convery a to numpy and
        # then just use numpy operations
        a = node.inputs[0]
        exp_a = array_api.exp(a.numpy() - array_api.max(a.numpy(), axis=self.axes, keepdims=True))
        softamx = exp_a / array_api.sum(exp_a, axis=self.axes, keepdims=True)
        return broadcast_to(out_grad, input.shape) * Tensor(softamx)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)