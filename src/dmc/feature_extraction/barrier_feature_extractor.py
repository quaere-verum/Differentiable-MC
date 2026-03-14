from .feature_extractor import FeatureExtractor, FeatureExtractorResult
from ..simulation.simulation_state import SimulationState
import torch
from enum import IntEnum

class VarianceFeatureType(IntEnum):
    NONE = 1
    LEARNED_FILTER = 2
    MARKOV_STATE = 3
    LEARNED_GATED_FILTER = 4

class DownAndOutCallFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        maturity: float,
        barrier: float,
        strike: float, 
        *,
        variance_feature_type: VarianceFeatureType = VarianceFeatureType.LEARNED_FILTER
    ):
        super().__init__()
        self._variance_feature_type = variance_feature_type
        self.register_buffer("maturity", torch.tensor(maturity, dtype=torch.float32))
        self.register_buffer("barrier", torch.tensor(barrier, dtype=torch.float32))
        self.register_buffer("strike", torch.tensor(strike, dtype=torch.float32))

        if self._variance_feature_type == VarianceFeatureType.LEARNED_FILTER:
            self.register_parameter("alpha_raw", torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32)))
            self.register_parameter("beta", torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32)))
            self.register_parameter("bias", torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32)))
        if self._variance_feature_type == VarianceFeatureType.LEARNED_GATED_FILTER:
            self.register_parameter("W_g", torch.nn.Parameter(torch.zeros((2, 1), dtype=torch.float32)))
            self.register_parameter("U_g", torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float32)))
            self.register_parameter("b_g", torch.nn.Parameter(torch.zeros((1,), dtype=torch.float32)))

            self.register_parameter("W_h", torch.nn.Parameter(torch.zeros((2, 1), dtype=torch.float32)))
            self.register_parameter("U_h", torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float32)))
            self.register_parameter("b_h", torch.nn.Parameter(torch.zeros((1,), dtype=torch.float32)))
        

    def get_features(self, state: SimulationState):
        S = state.spot
        S_min = state.spot_cumulative_min
        S_prev = state.spot_previous
        t = state.t
        t_prev = state.t_prev
        h_prev = state.hidden_state

        logS = S.log()
        logS_prev = S_prev.log()
        dlogS = logS - logS_prev

        dt = (t - t_prev)

        sq_ret = (dlogS * dlogS) / max(dt, 1e-6) 
        sq_ret = sq_ret.unsqueeze(1)

        if self._variance_feature_type == VarianceFeatureType.LEARNED_FILTER:
            next_hidden = (
                self.get_parameter("alpha_raw").sigmoid() * h_prev
                + self.get_parameter("beta") * sq_ret
                + self.get_parameter("bias")
            )
        elif self._variance_feature_type == VarianceFeatureType.LEARNED_GATED_FILTER:
            z = torch.cat((dlogS.unsqueeze(1), sq_ret), dim=1)
            gate = torch.sigmoid(
                torch.matmul(z, self.get_parameter("W_g"))
                + torch.matmul(h_prev, self.get_parameter("U_g"))
                + self.get_parameter("b_g") 
            )
            h_tilde = torch.tanh(
                torch.matmul(z, self.get_parameter("W_h"))
                + torch.matmul(h_prev, self.get_parameter("U_h"))
                + self.get_parameter("b_h") 
            )
            next_hidden = (1.0 - gate) * h_tilde + gate * h_prev
        else:
            next_hidden = None
           
        n_paths = S.size(0)

        tau = (self.get_buffer("maturity") - t).expand((n_paths, 1))
        log_barrier_dist = (S / self.get_buffer("barrier")).log().unsqueeze(1)
        log_moneyness = (S / self.get_buffer("strike")).log().unsqueeze(1)
        alive = (S_min  > self.get_buffer("barrier")).to(S.dtype).unsqueeze(1)

        features = [tau, log_barrier_dist, log_moneyness, alive]
        if (
            self._variance_feature_type == VarianceFeatureType.LEARNED_FILTER
            or self._variance_feature_type == VarianceFeatureType.LEARNED_GATED_FILTER
        ):
            features.append(next_hidden)
        elif self._variance_feature_type == VarianceFeatureType.MARKOV_STATE:
            features.append(state.variance.unsqueeze(1))
        features = torch.cat(features, dim=1)
        return FeatureExtractorResult(features, next_hidden)
    
    def hidden_state_dim(self):
        return (
            1 
            if self._variance_feature_type == VarianceFeatureType.LEARNED_FILTER 
            else 1
            if self._variance_feature_type == VarianceFeatureType.LEARNED_GATED_FILTER
            else 
            None
        )
    
    def feature_dim(self):
        return 4 if self._variance_feature_type == VarianceFeatureType.NONE else 5
    

class DownAndOutPutFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        maturity: float,
        barrier: float,
        strike: float,
        *,
        variance_feature_type: VarianceFeatureType = VarianceFeatureType.LEARNED_FILTER,
    ):
        super().__init__()
        self._variance_feature_type = variance_feature_type

        self.register_buffer("maturity", torch.tensor(maturity, dtype=torch.float32))
        self.register_buffer("barrier", torch.tensor(barrier, dtype=torch.float32))
        self.register_buffer("strike", torch.tensor(strike, dtype=torch.float32))

        if self._variance_feature_type == VarianceFeatureType.LEARNED_FILTER:
            self.register_parameter("alpha_raw", torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32)))
            self.register_parameter("beta", torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32)))
            self.register_parameter("bias", torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32)))

        if self._variance_feature_type == VarianceFeatureType.LEARNED_GATED_FILTER:
            self.register_parameter("W_g", torch.nn.Parameter(torch.zeros((2, 1), dtype=torch.float32)))
            self.register_parameter("U_g", torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float32)))
            self.register_parameter("b_g", torch.nn.Parameter(torch.zeros((1,), dtype=torch.float32)))

            self.register_parameter("W_h", torch.nn.Parameter(torch.zeros((2, 1), dtype=torch.float32)))
            self.register_parameter("U_h", torch.nn.Parameter(torch.zeros((1, 1), dtype=torch.float32)))
            self.register_parameter("b_h", torch.nn.Parameter(torch.zeros((1,), dtype=torch.float32)))

    def get_features(self, state: SimulationState):
        S = state.spot
        S_min = state.spot_cumulative_min
        S_prev = state.spot_previous
        t = state.t
        t_prev = state.t_prev
        h_prev = state.hidden_state

        logS = S.log()
        logS_prev = S_prev.log()
        dlogS = logS - logS_prev

        dt = t - t_prev
        sq_ret = (dlogS * dlogS) / max(dt, 1e-6)
        sq_ret = sq_ret.unsqueeze(1)

        if self._variance_feature_type == VarianceFeatureType.LEARNED_FILTER:
            next_hidden = (
                self.get_parameter("alpha_raw").sigmoid() * h_prev
                + self.get_parameter("beta") * sq_ret
                + self.get_parameter("bias")
            )

        elif self._variance_feature_type == VarianceFeatureType.LEARNED_GATED_FILTER:
            z = torch.cat((dlogS.unsqueeze(1), sq_ret), dim=1)

            gate = torch.sigmoid(
                torch.matmul(z, self.get_parameter("W_g"))
                + torch.matmul(h_prev, self.get_parameter("U_g"))
                + self.get_parameter("b_g")
            )

            h_tilde = torch.tanh(
                torch.matmul(z, self.get_parameter("W_h"))
                + torch.matmul(h_prev, self.get_parameter("U_h"))
                + self.get_parameter("b_h")
            )

            next_hidden = (1.0 - gate) * h_tilde + gate * h_prev

        else:
            next_hidden = None

        n_paths = S.size(0)

        tau = (self.get_buffer("maturity") - t).expand((n_paths, 1))
        log_barrier_dist = (S / self.get_buffer("barrier")).log().unsqueeze(1)
        log_moneyness = (self.get_buffer("strike") / S).log().unsqueeze(1)

        alive = (S_min > self.get_buffer("barrier")).to(S.dtype).unsqueeze(1)

        features = [tau, log_barrier_dist, log_moneyness, alive]

        if self._variance_feature_type in (
            VarianceFeatureType.LEARNED_FILTER,
            VarianceFeatureType.LEARNED_GATED_FILTER,
        ):
            features.append(next_hidden)
        elif self._variance_feature_type == VarianceFeatureType.MARKOV_STATE:
            features.append(state.variance.unsqueeze(1))

        features = torch.cat(features, dim=1)
        return FeatureExtractorResult(features, next_hidden)

    def hidden_state_dim(self):
        if self._variance_feature_type in (
            VarianceFeatureType.LEARNED_FILTER,
            VarianceFeatureType.LEARNED_GATED_FILTER,
        ):
            return 1
        return None

    def feature_dim(self):
        return 4 if self._variance_feature_type == VarianceFeatureType.NONE else 5
    