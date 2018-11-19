import typing
import torch

tensor = typing.Union[torch.Tensor, torch.FloatTensor]


class Coding:
    def __init__(self, *args, **kwargs):
        self.codes = []

    def encode(self, grad: tensor, *args, **kwargs) -> dict:
        raise NotImplementedError()

    def decode(self, code: dict, *args, **kwargs) -> tensor:
        raise NotImplementedError()