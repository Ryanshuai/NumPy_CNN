class Layer:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        raise NotImplementedError

    def backward(self, input):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError
