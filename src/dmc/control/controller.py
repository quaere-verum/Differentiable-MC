import torch
import abc
from ..feature_extraction.feature_extractor import FeatureExtractorResult

class Controller(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, features: FeatureExtractorResult) -> torch.Tensor:
        raise NotImplementedError