from .controller import Controller
from ..feature_extraction.feature_extractor import FeatureExtractorResult
import torch

class MlpController(Controller):
    def __init__(self, feature_dim: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        self._feature_dim = feature_dim
        self._hidden_sizes = hidden_sizes
        in_channels = self._feature_dim
        model = []
        for hidden_size in hidden_sizes:

            model.extend([
                torch.nn.Linear(in_channels, hidden_size),
                torch.nn.SiLU(),
            ])
            in_channels = hidden_size
        model.append(torch.nn.Linear(in_channels, 1))
        self._model = torch.nn.Sequential(*model)


    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(self._model.forward(features), 1)