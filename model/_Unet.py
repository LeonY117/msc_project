import math
import copy
import torch

from torch import nn, Tensor
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from functools import partial
from collections import OrderedDict

from .MBConfig import MBConvConfig
from .modules.Deconv2dNormActivation import Deconv2dNormActivation
from .Bayesian_net import Bayesian_net


class _Unet(Bayesian_net):
    def __init__(
        self,
        inverted_residual_setting: Sequence[MBConvConfig],
        input_dim: Union[int, Tuple[int, int]],
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        use_se: Optional[bool] = True,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.SiLU,
        dropout_layer: Optional[Callable[..., nn.Module]] = nn.Dropout2d,
        last_channel: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.inv_res_setting = inverted_residual_setting
        self.stochastic_depth_prob = stochastic_depth_prob
        self.num_classes = num_classes

        self.se_layer = SqueezeExcitation if use_se else None
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        self.dropout_layer = dropout_layer
        self.input_dim = input_dim
        self.last_channel = (
            last_channel
            if last_channel is not None
            else self.inv_res_setting[-1].out_channels * 4
        )
        # self.last_channel = last_channel

        self.decoder_config = self._decoder_config()
        self.dropout_state = True
        self.stochastic_depth_state = True

        # return

        # ========= Build blocks ==========
        encoder: OrderedDict = self._build_encoder()
        decoder: OrderedDict = self._build_decoder()
        self.encoder = nn.Sequential(encoder)
        self.decoder = nn.Sequential(decoder)
        # self.logsoftmax = nn.LogSoftmax(dim=1)

        # ========= weight initialization ========
        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        """
        Returns logits
        """
        self._toggle_stochastic_depth()
        self._toggle_dropout()
        return self._forward_impl(x)

    def _forward_impl(self, x: Tensor) -> Tensor:
        features = []
        # ================= Encoder =================
        # first conv
        x = self.encoder.conv0(x)
        # stages
        for i, cfg in enumerate(self.inv_res_setting):
            x = self.encoder[i + 1](x)
            if cfg.use_skip:
                features.append(x)
        # last conv
        x = self.encoder.lastConv(x)

        # ================= Decoder =================
        # first deconv
        x = self.decoder.deconv0(x)
        # stages
        for i, cfg in enumerate(self.decoder_config):
            x = self.decoder[i + 1](x)
            if i < len(self.decoder_config) - 1 and self.decoder_config[i + 1].use_skip:
                x = torch.cat([x, features.pop()], dim=1)
        # last deconv
        x = self.decoder.lastDeconv(x)
        # x = self.logsoftmax(x)

        return x

    def _decoder_config(self) -> List[MBConvConfig]:
        decoder_cfg: List[MBConvConfig] = []
        oup = self.inv_res_setting[0].input_channels
        skip = 0
        for cfg in self.inv_res_setting:
            skip = cfg.out_channels if cfg.use_skip else 0
            cfg_copy = copy.copy(cfg)
            cfg_copy.input_channels = cfg.out_channels + skip
            cfg_copy.out_channels = oup
            decoder_cfg.append(cfg_copy)

            oup = cfg.out_channels

        return decoder_cfg[::-1]

    def _build_decoder(self) -> OrderedDict:
        decoder: OrderedDict[str, nn.Module] = OrderedDict()

        if self.last_channel == 0:
            decoder["deconv0"] = nn.Identity()
        else:
            out = self.inv_res_setting[-1].out_channels
            decoder["deconv0"] = Conv2dNormActivation(
                self.last_channel, out, 1, 1, bias=False
            )

        def compute_out_pad(stage: int, dim: int) -> int:
            dim = math.ceil(dim / 2)  # account for first encoder
            # reverse the order to calculate output dim for corresponding encoder
            stage = len(self.inv_res_setting) - stage - 1
            for cfg in self.inv_res_setting[:stage]:
                if cfg.stride != 1:
                    # reduce dim if stride is not 1
                    dim = math.ceil(dim / 2)
            return int(dim % 2 == 0)

        # build inverted residual blocks
        total_stage_blocks = sum(cfg.num_layers for cfg in self.inv_res_setting)
        stage_block_id = 0
        for i, cfg in enumerate(self.decoder_config):
            stage: List[nn.Module] = []
            for j in range(cfg.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cfg = copy.copy(cfg)

                conv_layer = Conv2dNormActivation
                if block_cfg.stride != 1 and j == 0:
                    if isinstance(self.input_dim, tuple):
                        dim1, dim2 = self.input_dim
                        oup_pad = compute_out_pad(i, dim1), compute_out_pad(i, dim2)
                    else:
                        oup_pad = compute_out_pad(i, self.input_dim)
                    conv_layer = partial(
                        Deconv2dNormActivation,
                        out_pad=oup_pad,
                    )

                # adjust stochastic depth probability
                sd_prob = self.stochastic_depth_prob * (
                    1 - float(stage_block_id) / total_stage_blocks
                )

                # overwrite cfg if not first conv in the stage
                if j != 0:
                    block_cfg.input_channels = block_cfg.out_channels
                    block_cfg.stride = 1
                    sd_prob = 0

                stage.append(
                    block_cfg.block(
                        block_cfg,
                        sd_prob,
                        self.norm_layer,
                        self.se_layer,
                        conv_layer,
                        self.activation_layer,
                    )
                )
                stage_block_id += 1

            if cfg.dropout_p > 0:
                stage.append(self.dropout_layer(cfg.dropout_p))

            decoder[f"stage{i+1}"] = nn.Sequential(*stage)

        # build last deconv
        decoder["lastDeconv"] = Deconv2dNormActivation(
            self.decoder_config[-1].out_channels,
            self.num_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_layer=None,
            activation_layer=None,
        )
        return decoder

    def _build_encoder(self) -> OrderedDict:
        encoder: OrderedDict[str, nn.Module] = OrderedDict()

        # build first conv layer
        firstconv_output_channels = self.inv_res_setting[0].input_channels
        encoder["conv0"] = Conv2dNormActivation(
            3,
            firstconv_output_channels,
            kernel_size=3,
            stride=2,
            norm_layer=self.norm_layer,
            activation_layer=self.activation_layer,
        )

        # build inverted residual blocks
        total_stage_blocks = sum(cfg.num_layers for cfg in self.inv_res_setting)
        stage_block_id = 0
        for i, cfg in enumerate(self.inv_res_setting):
            stage: List[nn.Module] = []
            for j in range(cfg.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cfg = copy.copy(cfg)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cfg.input_channels = block_cfg.out_channels
                    block_cfg.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    self.stochastic_depth_prob
                    * float(stage_block_id)
                    / total_stage_blocks
                )

                stage.append(
                    block_cfg.block(
                        block_cfg,
                        sd_prob,
                        self.norm_layer,
                        self.se_layer,
                        Conv2dNormActivation,
                        activation_layer=self.activation_layer,
                    )
                )
                stage_block_id += 1

            if cfg.dropout_p > 0:
                stage.append(self.dropout_layer(cfg.dropout_p))

            encoder[f"stage{i+1}"] = nn.Sequential(*stage)

        # build last several layers
        lastconv_input_channels = self.inv_res_setting[-1].out_channels
        if self.last_channel == 0:
            # set last layer to identity
            encoder["lastConv"] = nn.Identity()
        else:
            lastconv_output_channels = self.last_channel
            encoder["lastConv"] = nn.Sequential(
                Conv2dNormActivation(
                    lastconv_input_channels,
                    lastconv_output_channels,
                    kernel_size=1,
                    norm_layer=self.norm_layer,
                    activation_layer=self.activation_layer,
                ),
                nn.Dropout2d(p=self.inv_res_setting[-1].dropout_p),
            )

        return encoder

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
