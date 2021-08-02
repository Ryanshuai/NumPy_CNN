from ..tensor import Tensor


class Parameter(Tensor):
    # def __new__(cls, data=None, requires_grad=True):
    #     if data is None:
    #         data = torch.tensor([])
    #     return torch.Tensor._make_subclass(cls, data, requires_grad)
    def __init__(self, data=None, requires_grad=True):
        super().__init__(data, requires_grad=requires_grad)
