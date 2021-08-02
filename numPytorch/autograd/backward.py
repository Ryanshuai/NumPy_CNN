class LayerBackward:
    def __init__(self, *args, **kwargs):
        self.next_functions = tuple()
        self.gradients = tuple()
        pass

    def __call__(self, *args, **kwargs):
        for func, grad in zip(self.next_functions, self.gradients):
            func(grad)
