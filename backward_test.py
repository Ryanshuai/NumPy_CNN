import torch
from numPytorch.tensor import Tensor
import torch.nn as nn

nn.NLLLoss



if __name__ == '__main__':
    print('--------------------------------------------')

    # x = torch.tensor(1.0, requires_grad=True)
    x = torch.tensor([[5.0, 5.0], [1.0, 1.0], [1.0, 1.0]], requires_grad=True)
    y = torch.tensor([[2.], [1.]], requires_grad=True)
    z = x @ y
    y.retain_grad()
    z.retain_grad()
    z.backward(gradient=torch.tensor([[1], [5], [3]]))

    # y = x + 2
    # z = x - y

    # # z.backward(gradient=torch.tensor(6))
    # loss = z.mean(axis=1)
    # loss.backward(gradient=torch.tensor([3, 5, 2]))
    # # loss.backward(gradient=torch.tensor(7))
    print(z.grad)
    print(z.grad_fn)
    print(y.grad)
    print(y.grad_fn)
    print(x.grad)

    print('--------------------------------------------')

    x = Tensor([[5.0, 5.0], [1.0, 1.0], [1.0, 1.0]], requires_grad=True)
    y = Tensor([[2.], [1.]], requires_grad=True)
    z = x @ y
    z.backward(gradient=Tensor([[1], [5], [3]]))

    # x = tensor(1, requires_grad=True)
    # y = x + 2
    # z = x - y
    # # z.backward(gradient=tensor(6))
    # loss = z.mean(axis=1)
    # loss.backward(gradient=Tensor([3, 5, 2]))
    # loss.backward(gradient=Tensor(7))
    print(z.grad)
    print(z.grad_fn)
    print(y.grad)
    print(y.grad_fn)
    print(x.grad)
