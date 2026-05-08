import abc

import torch.nn as nn


class BaseNet(nn.Module, abc.ABC):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def feature_extractor(self, x):
        pass

    @abc.abstractmethod
    def classifier(self, x):
        pass

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
