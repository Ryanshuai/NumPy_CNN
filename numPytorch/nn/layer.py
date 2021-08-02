class Layer:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, input):
        return self.forward(input)

    def forward(self, *args):
        raise NotImplementedError
