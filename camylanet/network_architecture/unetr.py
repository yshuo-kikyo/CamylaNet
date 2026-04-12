from typing import Sequence, Tuple, Union
import torch.nn as nn


from camylanet.network_architecture.custom.blocks.dynunet_block import UnetOutBlock
from camylanet.network_architecture.custom.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from camylanet.network_architecture.custom.nets.vit import ViT
from camylanet.network_architecture.custom.utils.misc import ensure_tuple_rep
from camylanet.network_architecture.neural_network import SegmentationNetwork
'''
from blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from nets.vit import ViT
from utils.misc import ensure_tuple_rep'''
import numpy as np


class UNETR(SegmentationNetwork):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        crop_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int] = (1, 16, 16),
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        num_pool: int = 3,
        pos_embed: str = "conv",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        deep_supervision: bool = True,
        conv_op=nn.Conv3d,
        seg_output_use_bias=False,
        encoder_kernel_sizes=None,
        num_layer_sizes=None,
        decoder_kernel_sizes=None
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        Examples::

            # for single channel input 4-channel output with image size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

             # for single channel input 4-channel output with image size of (96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=96, feature_size=32, norm_name='batch', spatial_dims=2)

            # for 4-channel input 3-channel output with image size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        # nnUNet config
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.seg_outputs = []
        if decoder_kernel_sizes is None:
            decoder_kernel_sizes = [(1, 2, 2)] * (num_pool+1)
        if num_layer_sizes is None:
            num_layer_sizes = [2, 1, 0]
        if encoder_kernel_sizes is None:
            encoder_kernel_sizes = [[(1, 2, 2)] * (nls+1) for nls in num_layer_sizes]
        # print('encoder_kernel_sizes', encoder_kernel_sizes)
        
        vit_output_z = crop_size[0] // patch_size[0]
        vit_output_x = crop_size[1] // patch_size[1]
        vit_output_y = crop_size[2] // patch_size[2]
        decoder_kernel_sizes

        decoder_channels = [hidden_size, feature_size*8, feature_size*4, feature_size*2, feature_size]

        for p in range(num_pool+1):
            decoder_kernel_sizes

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.seg_outputs = []
        self.conv_op = conv_op
        self.num_classes = num_classes

        self.num_layers = 12
        self.num_pool = num_pool
        # determine upsample_kernel_size
        # upsample_kernel_size = 
        crop_size = ensure_tuple_rep(crop_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(crop_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=input_channels,
            img_size=crop_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        self.conv_blocks_context.append(UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=input_channels,
                                                        out_channels=feature_size, kernel_size=3,
                                                        stride=1, norm_name=norm_name, res_block=res_block))

        for d in range(num_pool):
            out_channels = 2 ** (d+1) * feature_size
            num_layer = num_layer_sizes[d]
            upsample_kernel_size = encoder_kernel_sizes[d]
            self.conv_blocks_context.append(UnetrPrUpBlock(spatial_dims=spatial_dims, in_channels=hidden_size,
                                                           out_channels=out_channels, num_layer=num_layer, 
                                                           kernel_size=3, stride=1, 
                                                           upsample_kernel_size=upsample_kernel_size,
                                                           norm_name=norm_name, conv_block=conv_block,
                                                           res_block=res_block))

        for d in range(num_pool+1):
            in_channels = decoder_channels[d]
            out_channels = decoder_channels[d+1]
            upsample_kernel_size = decoder_kernel_sizes[d]
            self.conv_blocks_localization.append(UnetrUpBlock(spatial_dims=spatial_dims, in_channels=in_channels,
                                                         out_channels=out_channels, kernel_size=3, 
                                                         upsample_kernel_size=upsample_kernel_size,
                                                         norm_name=norm_name, res_block=res_block))

        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)

        # self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=num_classes)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))
        self.seg_outputs = nn.ModuleList(self.seg_outputs)

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        skip = []
        seg_outputs = []
        x_out, hidden_states_out = self.vit(x_in)
        x3 = self.proj_feat(hidden_states_out[3])
        x6 = self.proj_feat(hidden_states_out[6])
        x9 = self.proj_feat(hidden_states_out[9])
        vit_output = [x_in, x3, x6, x9]

        # encoder
        for d in range(len(self.conv_blocks_context)):
            proj_x = vit_output[d]
            # print('encoder {} proj_x output shape: '.format(d), proj_x.shape)
            skip.append(self.conv_blocks_context[d](proj_x))
            # print('encoder {} output shape: '.format(d), skip[d].shape)

        # decoder
        dec_x = self.proj_feat(x_out)
        for u in range(len(self.conv_blocks_localization)):
            enc_x = skip[-(u + 1)]
            # print('decoder {} output shape: '.format(u), dec_x.shape)
            dec_x = self.conv_blocks_localization[u](dec_x, enc_x)
            seg_outputs.append(self.seg_outputs[u](dec_x))
            
        if self.do_ds:
            return seg_outputs[::-1]
            # return seg_outputs
        else:
            return seg_outputs[-1]