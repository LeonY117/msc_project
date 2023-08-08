from functools import partial
from typing import Callable, List, Optional

from torch import nn, Tensor
from torchvision.ops import StochasticDepth
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation


class MBConv(nn.Module):
    def __init__(
        self,
        cfg,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Optional[Callable[..., nn.Module]] = SqueezeExcitation,
        conv_layer: Optional[Callable[..., nn.Module]] = Conv2dNormActivation,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.SiLU,
        use_res_connect: Optional[bool] = None,
    ) -> None:
        super().__init__()

        if not (1 <= cfg.stride <= 2):
            raise ValueError("illegal stride value")

        if use_res_connect is None:
            self.use_res_connect = (
                cfg.stride == 1 and cfg.input_channels == cfg.out_channels
            )
        else:
            self.use_res_connect = use_res_connect

        layers: List[nn.Module] = []

        # expand
        expanded_channels = cfg.adjust_channels(cfg.input_channels, cfg.expand_ratio)
        if expanded_channels != cfg.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cfg.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise (can be a deconv block)
        layers.append(
            conv_layer(
                expanded_channels,
                expanded_channels,
                kernel_size=cfg.kernel,
                stride=cfg.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )
        if se_layer is not None:
            # squeeze and excitation
            squeeze_channels = max(1, cfg.input_channels // 4)
            layers.append(
                se_layer(
                    expanded_channels,
                    squeeze_channels,
                    activation=partial(activation_layer, inplace=True),
                )
            )

        # project
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                cfg.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)
        # if stochastic_depth_prob > 0.0:
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cfg.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result