class BaseMetric:
    def __init__(self, name=None, split=None, *args, **kwargs):
        self.name = name if name is not None else type(self).__name__
        self.split = split

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
