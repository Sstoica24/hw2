"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.use_bias = bias
        w = init.kaiming_uniform(in_features, out_features)
        # paramter is specific tesnor class for paramters that was
        # given above
        self.weight = Parameter(w, device=device, dtype=dtype)
        if self.use_bias:
            # fan_in == out_features and since b must ultimatley be 
            # (1, out_features), fan_out == 1. 
            b = ops.reshape(init.kaiming_uniform(out_features, 1), (1, out_features))
            self.bias = Parameter(b, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = ops.matmul(X, self.weight)
        if self.use_bias:
            bias = ops.broadcast_to(self.bias, y.shape)
            y += bias
        return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        shape_list = list(X.shape)
        prod = 1
        for i in range(1, len(shape_list)):
            prod *= shape_list[i]
        return X.reshape((X.shape[0], prod))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = logits
        y_one_hot = init.one_hot(Z.shape[1], y, device=None, dtype="float32", requires_grad=False)
        m = Z.shape[0]
        exp_Z = ops.exp(Z)
        #exp_Z is 2 dim, but summation would return 1D, so we need to make summation 2D
        # based on how the code is written in the backward pass if you don't explicitley call broadcast
        # then it doesn't know that broadcast was used. 
        logSumExp = ops.logsumexp(Z)
        # softmax_probs = exp_Z / ops.summation(exp_Z, axes=(1,)).reshape((exp_Z.shape[0], 1)).broadcast_to((exp_Z.shape) )
        # Compute the loss for each sample in the batch
        log_t = ops.log(logSumExp) * y_one_hot
        loss = -ops.summation(log_t) / m
        # Average the loss over all samples
        return loss
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, requires_grad=True))
        # the following two are not paramters and, of course, do not need grads
        # are they are not learnable. 
        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # module has self.training
        if self.training:
            # summing over batch_size ==> summing over axis = 0
            # note, we do not need to reshape to 2D (x.shape[0], 1) because we are working
            # with the batch size. 
            # you can't set batch_mean = (x.sum((0,)) / x.shape[0]).broadcast_to(x.shape)
            # because the tester expects batch_mean TO NOT BE SIZE BE SIZE OF x.shape (
            # saw this while I was debugging). Same story for batch_var
            batch_mean = x.sum((0,)) / x.shape[0]
            batch_var = ((x - batch_mean.broadcast_to(x.shape))**2).sum((0,)) / x.shape[0]
            norm = (x - batch_mean.broadcast_to(x.shape)) / (batch_var.broadcast_to(x.shape) + self.eps)**0.5
            y = self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
            # what is meant by the equation given is that x_old = old self.running_mean and 
            # self.running_var and x_observed = the current batch_mean.data and batch_var.data.
            # .data since batch_mean and batch_var are both tensors 
            # no need to worry about broadcasting since these have no derivatives!
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.data
        else:
            y = (x - self.running_mean.broadcast_to(x.shape)) / (self.running_var.broadcast_to(x.shape) + self.eps)**0.5
        return y
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, requires_grad=True))
        ### END YOUR SOLUTION 

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean = (x.sum((1,))/x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        var = (((x - mean)**2).sum((1,))/x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        deno = (var + self.eps)**0.5
        return self.weight.broadcast_to(x.shape) * (x - mean)/deno + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            # we want z_(x+1)/(1-p) to occur 1-p times ==> prob in ranb must be 1-p
            bernoulli_mask = init.randb(*x.shape, p=1-self.p, requires_grad=True)
            return x * bernoulli_mask / (1-self.p)
        else:
            # no dropout during testing
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
       return self.fn(x) + x
        ### END YOUR SOLUTION
