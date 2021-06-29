class AddBackward:
    def __init__(self, func1, func2):
        self.next_functions = (func1, func2)

    def __call__(self, gradient):
        for func in self.next_functions:
            func(gradient)
