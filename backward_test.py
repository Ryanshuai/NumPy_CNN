


if __name__ == '__main__':
    import torch

    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2
    y.retain_grad()
    z = (x + y)
    z.retain_grad()
    print(y.grad)
    loss = z.mean()
    loss.backward(gradient=torch.tensor(1.0))
    print(z.grad_fn)
    print(z.grad)
    print(y.grad)
    print(x.grad)