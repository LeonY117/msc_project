from torch import nn

class Bayesian_net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout_state=None
        self.stochastic_depth_state=None

    def _toggle_dropout(self) -> None:
        if self.dropout_state == True:
            for m in self.modules():
                if m.__class__.__name__.startswith("Dropout"):
                    m.train()

    def _toggle_stochastic_depth(self) -> None:
        if self.stochastic_depth_state == True:
            for m in self.modules():
                if m.__class__.__name__.startswith("StochasticDepth"):
                    m.train()

    def _set_dropout(self, state: bool) -> None:
        self.dropout_state = state

    def _set_stochastic_depth(self, state: bool) -> None:
        self.stochastic_depth_state = state

    def set_bayes_mode(self, state: bool, mode: str) -> None:
        if mode == "dropout":
            self._set_dropout(state)
        elif mode == "stochastic_depth":
            self._set_stochastic_depth(state)
        elif mode == "all":
            self._set_dropout(state)
            self._set_stochastic_depth(state)
        else:
            raise ValueError(
                f"mode must be dropout, stochastic_depth, or all, got {mode}"
            )
