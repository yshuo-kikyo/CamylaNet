from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from camylanet.network_architecture.custom.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from camylanet.network_architecture.custom.layers.factories import Dropout
from camylanet.network_architecture.custom.layers.utils import get_act_layer, get_norm_layer
from camylanet.network_architecture.custom.utils.enums import UpsampleMode
from camylanet.network_architecture.neural_network import SegmentationNetwork


class DownLayers(nn.Module):
    def __init__(
        self,
        blocks_down_num, 
        spatial_dims, 
        norm,
        layer_in_channels,
        is_pre_conv,
        act
    ):
        super().__init__()
        if is_pre_conv:
            pre_conv = get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
        else:
            pre_conv = nn.Identity()
        
        self.down_layer = nn.Sequential(
            pre_conv,
            *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=act) for _ in range(blocks_down_num)],
        )

    def forward(self, x):
        x = self.down_layer(x)
        return x


class UpLayers(nn.Module):
    def __init__(
        self,
        upsample_mode, 
        blocks_up_num, 
        spatial_dims, 
        filters, 
        norm,
        sample_in_channels,
        act,
    ):
        super().__init__()

        self.output_channels = sample_in_channels // 2
        self.res_block = nn.Sequential(*[ResBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=act)
                        for _ in range(blocks_up_num)
                        ]
                    )

        self.up_layer = nn.Sequential(*[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                        ]
                    )

        self.acitvate = nn.Sequential(get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=self.output_channels),
                                      )
        
    def forward(self, x, skip):
        # print(x.shape, self.up_layer(x).shape, skip.shape)
        x = torch.add(self.up_layer(x), skip)
        # print(x.shape, self.res_block(x).shape, skip.shape)
        x = self.res_block(x)
        x = self.acitvate(x)
        return x


class SegResNet(SegmentationNetwork):
    """
    SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    The module does not include the variational autoencoder (VAE).
    The model supports 2D or 3D inputs.

    Args:
        spatial_dims: spatial dimension of the input data. Defaults to 3.
        init_filters: number of output channels for initial convolution layer. Defaults to 8.
        in_channels: number of input channels for the network. Defaults to 1.
        out_channels: number of output channels for the network. Defaults to 2.
        dropout_prob: probability of an element to be zero-ed. Defaults to ``None``.
        act: activation type and arguments. Defaults to ``RELU``.
        norm: feature normalization type and arguments. Defaults to ``GROUP``.
        norm_name: deprecating option for feature normalization type.
        num_groups: deprecating option for group norm. parameters.
        use_conv_final: if add a final convolution block to output. Defaults to ``True``.
        blocks_down: number of down sample blocks in each layer. Defaults to ``[1,2,2,4]``.
        blocks_up: number of up sample blocks in each layer. Defaults to ``[1,1,1]``.
        upsample_mode: [``"deconv"``, ``"nontrainable"``, ``"pixelshuffle"``]
            The mode of upsampling manipulations.
            Using the ``nontrainable`` modes cannot guarantee the model's reproducibility. Defaults to``nontrainable``.

            - ``deconv``, uses transposed convolution layers.
            - ``nontrainable``, uses non-trainable `linear` interpolation.
            - ``pixelshuffle``, uses :py:class:`blocks.SubpixelUpsample`.

    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 32,
        input_channels: int = 1,
        num_classes: int = 2,
        act: Union[Tuple, str] = ("RELU", {"inplace": True}),
        norm: Union[Tuple, str] = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE,
        deep_supervision: bool = True,
        conv_op=nn.Conv3d,
        num_pool=4,
        seg_output_use_bias=False,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = input_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final

        # nnUNet config
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.conv_op = conv_op
        self.num_classes = num_classes
        self.seg_outputs = []
        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.num_pool = num_pool

        self.conv_blocks_context.append(get_conv_layer(spatial_dims, input_channels, init_filters))
        for i in range(len(blocks_down)):
            layer_in_channels = self.init_filters * 2**i
            is_pre_conv = True if i>0 else False
            # print('layer_in_channels {}: {}'.format(i, layer_in_channels))
            self.conv_blocks_context.append(DownLayers(blocks_down[i], spatial_dims, self.norm, 
                                                       layer_in_channels, is_pre_conv, self.act))
        
        for i in range(len(blocks_up)):
            sample_in_channels = self.init_filters * 2 ** (len(blocks_up) - i)
            # print('sample_in_channels', sample_in_channels)
            self.conv_blocks_localization.append(UpLayers(self.upsample_mode, blocks_up[i],
                                                          self.spatial_dims, self.init_filters,
                                                          self.norm, sample_in_channels, self.act))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))

        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = []
        seg_outputs = []
        # encoder
        for d in range(len(self.conv_blocks_context)):
            x = self.conv_blocks_context[d](x)
            if d>0:
                skip.append(x)

        skip.reverse()

        for u in range(len(self.conv_blocks_localization)):
            x = self.conv_blocks_localization[u](x, skip[u+1])
            seg_outputs.append(self.seg_outputs[u](x))

        if self.do_ds:
            return seg_outputs[::-1]
            # return seg_outputs
        else:
            return seg_outputs[-1]