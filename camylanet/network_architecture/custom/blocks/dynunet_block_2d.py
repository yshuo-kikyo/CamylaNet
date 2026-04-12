from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from camylanet.network_architecture.custom.blocks.convolutions import Convolution
from camylanet.network_architecture.custom.layers.factories import Act, Norm
from camylanet.network_architecture.custom.layers.utils import get_act_layer, get_norm_layer

class ChannelShuffle:
    def __init__(self, groups: int = 4):
        """
        Initialize the channel-shuffle module.

        Args:
            groups (int): Number of groups to split the channels into.
        """
        self.groups = groups

    def channel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel shuffle to a 2D feature map (4D tensor).

        Args:
            x (torch.Tensor): Input feature map of shape [B, C, H, W].

        Returns:
            torch.Tensor: Channel-shuffled feature map.
        """
        # 1. Get the input shape; here it is a 4D tensor (batch, channels, height, width)
        B, C, H, W = x.shape

        # Ensure channels are divisible by the group count
        assert C % self.groups == 0, "Number of channels must be divisible by the number of groups"

        channels_per_group = C // self.groups

        # 2. Reshape: split the channel dim into 'groups' and 'channels per group'
        # The original S (depth/slice) dimension has been removed
        x = x.view(B, self.groups, channels_per_group, H, W)

        # 3. Transpose: swap 'groups' and 'channels per group' — the core of channel shuffle
        x = torch.transpose(x, 1, 2).contiguous()

        # 4. Flatten: merge 'groups' and 'channels per group' back into the channel dim
        # The original S (depth/slice) dimension has likewise been removed
        x = x.view(B, -1, H, W)

        return x

class UnetResBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet.
    It is controlled by the `spatial_dims` parameter.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str] = "instance",
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, 
            out_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            dropout=dropout, 
            conv_only=True
        )
        self.conv3 = get_conv_layer(
            spatial_dims, 
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=stride, 
            dropout=dropout, 
            conv_only=True
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample:
            residual = self.conv3(residual)
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


class ContextInjectionModuleV4(nn.Module):
    """
    Context Injection Module used in the Encoder.
    Handles both 2D and 3D cases by checking spatial_dims.
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm_name: Union[Tuple, str] = "instance",
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.pcim1 = PCIMv2(spatial_dims, in_channels*2, mid_channels=in_channels//2, out_channels=in_channels, norm_name=norm_name, act_name=act_name)
        self.pcim2 = PCIMv2(spatial_dims, in_channels*2, mid_channels=in_channels//2, out_channels=in_channels, norm_name=norm_name, act_name=act_name)
        self.pcim3 = PCIMv2(spatial_dims, in_channels*2, mid_channels=in_channels//2, out_channels=in_channels, norm_name=norm_name, act_name=act_name)

    def forward(self, inp, inp1, inp2, inp3):
        B, L, C = inp.shape
        
        # MODIFIED FOR 2D/3D: Adapt tensor reshaping based on spatial_dims
        if self.spatial_dims == 2:
            _, _, H, W = inp1.shape
            assert L == H * W, f"Input token length {L} does not match spatial dimensions {H}x{W}"
            inp_reshaped = inp.permute(0, 2, 1).contiguous().view(B, C, H, W)
        else: # Original 3D path
            _, _, S, H, W = inp1.shape
            assert L == S*H*W, f"Input token length {L} does not match spatial dimensions {S}x{H}x{W}"
            inp_reshaped = inp.permute(0, 2, 1).contiguous().view(B, C, S, H, W)
        
        # The residual connection should be from the reshaped main modality feature
        residual = inp_reshaped
        
        out = self.pcim1(inp_reshaped, inp1, residual)
        out = self.pcim2(out, inp2, residual)
        out = self.pcim3(out, inp3, residual)
        
        # MODIFIED FOR 2D/3D: Reshape back to sequence
        if self.spatial_dims == 2:
            out = out.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        else: # Original 3D path
            out = out.permute(0, 2, 3, 4, 1).contiguous().view(B, S*H*W, C)

        return out


class PCIMv2(nn.Module):
    """
    Progressive Context Injection Module v2.
    It is controlled by the `spatial_dims` parameter.
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        norm_name: Union[Tuple, str] = "instance",
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
    ):
        super().__init__()
        
        self.conv1 = get_conv_layer(spatial_dims, in_channels, mid_channels, kernel_size=3)
        self.conv2 = get_conv_layer(spatial_dims, mid_channels, 1, kernel_size=1)
        self.conv3 = get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=3)
        # This conv is for the residual path
        self.conv4 = get_conv_layer(spatial_dims, in_channels // 2, out_channels, kernel_size=1)
        
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=mid_channels)
        self.norm2 = nn.Sigmoid()
        self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm4 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        
        self.lrelu = get_act_layer(name=act_name)
        
        self.downsample = (in_channels // 2) != out_channels

    def forward(self, major_inp, minor_inp, residual):        
        inp = torch.cat((major_inp, minor_inp), 1)
        
        # Attention branch
        attn_weight = self.conv1(inp)
        attn_weight = self.norm1(attn_weight)
        attn_weight = self.lrelu(attn_weight)
        attn_weight = self.conv2(attn_weight)
        attn_weight = self.norm2(attn_weight)
        
        # Main convolution branch
        out = self.conv3(inp)
        out = self.norm3(out)
        out = self.lrelu(out)
        
        out = out * attn_weight.expand_as(out)
        
        # Residual connection
        if self.downsample:
            residual = self.conv4(residual)
            residual = self.norm4(residual)
            
        out += residual
        out = self.lrelu(out)

        return out

class NeighborAttention(nn.Module):
    """
    Neighbor Attention (NA) module.
    Handles both 2D and 3D cases by checking input tensor dimensions.
    """
    def __init__(self, k_size: int = 9):
        super().__init__()
        self.shuffle = ChannelShuffle(groups=2)
        self.avg_pool_2d = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_3d = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_main: torch.Tensor, x_minor: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x_main, x_minor), dim=1)
        x_shuffled = self.shuffle.channel_shuffle(x)

        # MODIFIED FOR 2D/3D: Adapt pooling and reshaping based on input dimension
        if x_shuffled.dim() == 4: # 2D Case
            y = self.avg_pool_2d(x_shuffled)
            y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        else: # 3D Case
            y = self.avg_pool_3d(x_shuffled)
            y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)

        y = self.sigmoid(y)
        return x_shuffled * y.expand_as(x_shuffled)


class NIM(nn.Module):
    """
    Neighborhood Integrated Module (NIM).
    It is controlled by the `spatial_dims` parameter.
    """
    def __init__(self, spatial_dims: int, channels: int, k_size: int = 9,
                 norm_name: Union[Tuple, str] = "instance",
                 act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
                 kernel_size=3, stride=1):
        super().__init__()
        self.channels = channels
        self.na = NeighborAttention(k_size=k_size)
        self.res_block = UnetResBlock(spatial_dims=spatial_dims,
                                        in_channels=channels * 2,
                                        out_channels=channels,
                                        norm_name=norm_name,
                                        act_name=act_name,
                                        kernel_size=kernel_size, 
                                        stride=stride)

    def forward(self, x_main_acc: torch.Tensor, x_minor: torch.Tensor, x_main_orig: torch.Tensor) -> torch.Tensor:
        u_mi = self.na(x_main_acc, x_minor)
        res_out = self.res_block(u_mi)
        out = res_out + x_main_orig
        return out


class PNIM(nn.Module):
    """
    Progressive Neighborhood Integrated Module (PNIM).
    It is controlled by the `spatial_dims` parameter.
    """
    def __init__(self, spatial_dims: int, channels: int, k_size: int = 9,  kernel_size=3, stride=1,
                 norm_name: Union[Tuple, str] = "instance",
                 act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01})):
        super().__init__()
        self.nim1 = NIM(spatial_dims, channels, k_size, norm_name, act_name, kernel_size, stride)
        self.nim2 = NIM(spatial_dims, channels, k_size, norm_name, act_name, kernel_size, stride)
        self.nim3 = NIM(spatial_dims, channels, k_size, norm_name, act_name, kernel_size, stride)

    def forward(self, x) -> torch.Tensor:
        x_main, x_minor1, x_minor2, x_minor3 = torch.chunk(x, 4, dim=1)
        x_m1 = self.nim1(x_main, x_minor1, x_main)
        x_m2 = self.nim2(x_m1, x_minor2, x_main)
        x_m3 = self.nim3(x_m2, x_minor3, x_main)
        return x_m3


class UnetResSEBlock(nn.Module):
    """
    A skip-connection based module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str] = "instance",
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, 
            out_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            dropout=dropout, 
            conv_only=True
        )
        self.conv3 = get_conv_layer(
            spatial_dims, 
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=stride, 
            dropout=dropout, 
            conv_only=True
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True

        self.se = SELayer(channel=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)
        if self.downsample:
            residual = self.conv3(residual)
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # featurize dataset download c369a55a-2c7c-4273-a48b-0c90e4fef9e3

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = Act.PRELU,
    norm: Union[Tuple, str] = Norm.INSTANCE,
    dropout: Optional[Union[Tuple, str, float]] = None,
    bias: bool = False,
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )


def get_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:

    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    if np.min(padding_np) < 0:
        raise AssertionError("padding value should not be negative, please change the kernel size and/or stride.")
    padding = tuple(int(p) for p in padding_np)

    return padding if len(padding) > 1 else padding[0]


def get_output_padding(
    kernel_size: Union[Sequence[int], int], stride: Union[Sequence[int], int], padding: Union[Sequence[int], int]
) -> Union[Tuple[int, ...], int]:
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)

    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    if np.min(out_padding_np) < 0:
        raise AssertionError("out_padding value should not be negative, please change the kernel size and/or stride.")
    out_padding = tuple(int(p) for p in out_padding_np)

    return out_padding if len(out_padding) > 1 else out_padding[0]