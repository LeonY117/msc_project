from torch import nn
from typing import Callable, Tuple, Optional, Union


class Deconv2dNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        out_pad: Optional[Union[int, Tuple[int, ...]]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        inplace: Optional[bool] = True,
        dilation: Union[int, Tuple[int, ...]] = 1,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., nn.Module] = nn.ConvTranspose2d,
    ) -> None:
        # calculate padding
        if padding is None:
            if dilation > 1:
                padding = dilation * (kernel_size - 1) // 2
            else:
                padding = kernel_size // 2
        # calculate output padding
        if out_pad is None:
            if stride == 2:
                out_pad = 1
            else:
                out_pad = 0

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                out_pad,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))

        super().__init__(*layers)

        self.out_channels = out_channels
