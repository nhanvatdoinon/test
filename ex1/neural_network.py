
class Module:
    def __init__(self):
        pass

    def __call__(self, input):
        return self.layers(input)
