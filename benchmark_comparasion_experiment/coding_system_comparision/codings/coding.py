import torch

class Coding:
    def __init__(self, *args, **kwargs):
        self.codes = []

    def encode(self, grad, *args, **kwargs):
        raise NotImplementedError()

    def decode(self, code, *args, **kwargs):
        raise NotImplementedError()