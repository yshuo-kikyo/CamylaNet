from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from camylanet.network_architecture.custom.blocks.convolutions import Convolution
from camylanet.network_architecture.custom.layers.factories import Act, Norm
from camylanet.network_architecture.custom.layers.utils import ChannelShuffle, DepthShuffle, get_act_layer, get_norm_layer


class UnetResBlock(nn.Module):
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
    
    
class ContextInjectionModule(nn.Module):
    """
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
        kernel_size: Union[Sequence[int], int] = 3,
        stride: Union[Sequence[int], int] = 1,
        norm_name: Union[Tuple, str] = "instance",
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels * 2,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, 
            in_channels * 2, 
            in_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            dropout=dropout, 
            conv_only=True
        )
        
        self.conv3 = get_conv_layer(
            spatial_dims, 
            in_channels * 2, 
            in_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            dropout=dropout, 
            conv_only=True
        )
        
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=in_channels)
        self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=in_channels)

    def forward(self, inp, inp1, inp2, inp3):
        # print('----------')
        # print(inp.shape)
        B, L, C = inp.shape
        B, C, S, H, W = inp1.shape
        assert L == S*H*W
        
        inp = inp.permute(0, 2, 1).contiguous().view(B, C, S, H, W)
        residual = inp
        
        inp = torch.cat((inp, inp1), 1)
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        
        out = torch.cat((out, inp2), 1)
        out = self.conv2(out)
        out = self.norm2(out)
        
        out = torch.cat((out, inp3), 1)
        out = self.conv3(out)
        out = self.norm3(out)
        
        out += residual
        out = self.lrelu(out)
        
        out = out.permute(0, 2, 3, 4, 1).contiguous().view(B, S*H*W, C)
        # print(out.shape)
        # print('----------')
        return out
    
    
class ContextInjectionModulev2(nn.Module):
    """
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
        kernel_size: Union[Sequence[int], int] = 3,
        stride: Union[Sequence[int], int] = 1,
        norm_name: Union[Tuple, str] = "instance",
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            in_channels * 2,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout,
            conv_only=True,
        )
        self.conv2 = get_conv_layer(
            spatial_dims, 
            in_channels * 2, 
            in_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            dropout=dropout, 
            conv_only=True
        )
        
        self.conv3 = get_conv_layer(
            spatial_dims, 
            in_channels * 2, 
            in_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            dropout=dropout, 
            conv_only=True
        )
        
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=in_channels)
        self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=in_channels)

    def forward(self, inp, inp1, inp2, inp3):
        # print('----------')
        # print(inp.shape)
        B, L, C = inp.shape
        B, C, S, H, W = inp1.shape
        assert L == S*H*W
        
        inp = inp.permute(0, 2, 1).contiguous().view(B, C, S, H, W)
        residual = inp
        
        inp = torch.cat((inp, inp1), 1)
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        
        out = torch.cat((out, inp2), 1)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out) # v2 add this
        
        out = torch.cat((out, inp3), 1)
        out = self.conv3(out)
        out = self.norm3(out)
        
        out += residual
        out = self.lrelu(out)
        
        out = out.permute(0, 2, 3, 4, 1).contiguous().view(B, S*H*W, C)
        # print(out.shape)
        # print('----------')
        return out


class ContextInjectionModuleV3(nn.Module):
    """
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
        norm_name: Union[Tuple, str] = "instance",
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.pcim1 = PCIM(spatial_dims, in_channels*2, 2, in_channels, norm_name, act_name)
        self.pcim2 = PCIM(spatial_dims, in_channels*2, 2, in_channels, norm_name, act_name)
        self.pcim3 = PCIM(spatial_dims, in_channels*2, 2, in_channels, norm_name, act_name)

    def forward(self, inp, inp1, inp2, inp3):
        # print('----------')
        # print(inp.shape)
        B, L, C = inp.shape
        B, C, S, H, W = inp1.shape
        assert L == S*H*W
        
        inp = inp.permute(0, 2, 1).contiguous().view(B, C, S, H, W)
        
        out = self.pcim1(inp, inp1)
        out = self.pcim2(out, inp2)
        out = self.pcim3(out, inp3)
        
        out = out.permute(0, 2, 3, 4, 1).contiguous().view(B, S*H*W, C)
        # print(out.shape)
        # print('----------')
        return out


class ContextInjectionModuleV3_3Channel(nn.Module):
    """
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
        norm_name: Union[Tuple, str] = "instance",
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.pcim1 = PCIM(spatial_dims, in_channels*2, 2, in_channels, norm_name, act_name)
        self.pcim2 = PCIM(spatial_dims, in_channels*2, 2, in_channels, norm_name, act_name)

    def forward(self, inp, inp1, inp2):
        # print('----------')
        # print(inp.shape)
        B, L, C = inp.shape
        B, C, S, H, W = inp1.shape
        assert L == S*H*W
        
        inp = inp.permute(0, 2, 1).contiguous().view(B, C, S, H, W)
        
        out = self.pcim1(inp, inp1)
        out = self.pcim2(out, inp2)
        
        out = out.permute(0, 2, 3, 4, 1).contiguous().view(B, S*H*W, C)
        # print(out.shape)
        # print('----------')
        return out
    

class ContextInjectionModuleV3_2Channel(nn.Module):
    """
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
        norm_name: Union[Tuple, str] = "instance",
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.pcim1 = PCIM(spatial_dims, in_channels*2, 2, in_channels, norm_name, act_name)

    def forward(self, inp, inp1):
        # print('----------')
        # print(inp.shape)
        B, L, C = inp.shape
        B, C, S, H, W = inp1.shape
        assert L == S*H*W
        
        inp = inp.permute(0, 2, 1).contiguous().view(B, C, S, H, W)
        
        out = self.pcim1(inp, inp1)
        
        out = out.permute(0, 2, 3, 4, 1).contiguous().view(B, S*H*W, C)
        # print(out.shape)
        # print('----------')
        return out


class ContextInjectionModuleV3_1Channel(nn.Module):
    """
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
        norm_name: Union[Tuple, str] = "instance",
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.pcim1 = PCIM(spatial_dims, in_channels*2, 2, in_channels, norm_name, act_name)

    def forward(self, inp, inp1):
        # print('----------')
        # print(inp.shape)
        B, L, C = inp.shape
        B, C, S, H, W = inp1.shape
        assert L == S*H*W
        
        inp = inp.permute(0, 2, 1).contiguous().view(B, C, S, H, W)
        
        out = self.pcim1(inp, inp1)
        
        out = out.permute(0, 2, 3, 4, 1).contiguous().view(B, S*H*W, C)
        # print(out.shape)
        # print('----------')
        return out
    

class ContextInjectionModuleV4(nn.Module):
    """
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
        norm_name: Union[Tuple, str] = "instance",
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        self.pcim1 = PCIMv2(spatial_dims, in_channels*2, 2, in_channels, norm_name, act_name)
        self.pcim2 = PCIMv2(spatial_dims, in_channels*2, 2, in_channels, norm_name, act_name)
        self.pcim3 = PCIMv2(spatial_dims, in_channels*2, 2, in_channels, norm_name, act_name)

    def forward(self, inp, inp1, inp2, inp3):
        # print('----------')
        # print(inp.shape)
        B, L, C = inp.shape
        B, C, S, H, W = inp1.shape
        assert L == S*H*W
        
        inp = inp.permute(0, 2, 1).contiguous().view(B, C, S, H, W)
        
        out = self.pcim1(inp, inp1, inp)
        out = self.pcim2(out, inp2, inp)
        out = self.pcim3(out, inp3, inp)
                
        out = out.permute(0, 2, 3, 4, 1).contiguous().view(B, S*H*W, C)
        # print(out.shape)
        # print('----------')
        return out
    

class PCIM(nn.Module):
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
        
        self.conv1 = get_conv_layer(spatial_dims, in_channels, mid_channels)
        self.conv2 = get_conv_layer(spatial_dims, mid_channels, 1)
        self.conv3 = get_conv_layer(spatial_dims, in_channels, out_channels)
        self.conv4 = get_conv_layer(spatial_dims, in_channels // 2, out_channels)
        
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = nn.Sigmoid()
        self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm4 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        
        self.lrelu = get_act_layer(name=act_name)
        
        self.downsample = int(in_channels // 2) != out_channels

    def forward(self, major_inp, minor_inp):
        residual = major_inp
        
        # attention branch
        inp = torch.cat((major_inp, minor_inp), 1)
        attn_weight = self.conv1(inp)
        attn_weight = self.norm1(attn_weight)
        attn_weight = self.lrelu(attn_weight)
        attn_weight = self.conv2(attn_weight)
        attn_weight = self.norm2(attn_weight)
        
        # conv branch
        out = self.conv3(inp)
        out = self.norm3(out)
        out = self.lrelu(out)
        
        out = out * attn_weight.expand_as(out)
        
        if self.downsample:
            residual = self.conv4(residual)
            residual = self.norm4(residual)
            
        out += residual
        out = self.lrelu(out)

        return out
    
    
class PCIMv2(nn.Module):
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
        
        self.conv1 = get_conv_layer(spatial_dims, in_channels, mid_channels)
        self.conv2 = get_conv_layer(spatial_dims, mid_channels, 1)
        self.conv3 = get_conv_layer(spatial_dims, in_channels, out_channels)
        self.conv4 = get_conv_layer(spatial_dims, in_channels // 2, out_channels)
        
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = nn.Sigmoid()
        self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm4 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        
        self.lrelu = get_act_layer(name=act_name)
        
        self.downsample = int(in_channels // 2) != out_channels

    def forward(self, major_inp, minor_inp, residual):        
        # attention branch
        inp = torch.cat((major_inp, minor_inp), 1)
        attn_weight = self.conv1(inp)
        attn_weight = self.norm1(attn_weight)
        attn_weight = self.lrelu(attn_weight)
        attn_weight = self.conv2(attn_weight)
        attn_weight = self.norm2(attn_weight)
        
        # conv branch
        out = self.conv3(inp)
        out = self.norm3(out)
        out = self.lrelu(out)
        
        out = out * attn_weight.expand_as(out)
        
        if self.downsample:
            residual = self.conv4(residual)
            residual = self.norm4(residual)
            
        out += residual
        out = self.lrelu(out)

        return out
     
    
class GlobalIntegratedModule(nn.Module):
    """
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
        k_size: int = 9,
        modality_num: int = 4,
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

        self.shuffle = ChannelShuffle(groups=modality_num)
        self.attn = ECAttention(k_size=k_size)

    def forward(self, inp):
        residual = inp
        out = self.shuffle.channel_shuffle(inp)
        out = self.attn(out)
        out = self.conv1(out)
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


class GlobalIntegratedModuleV3(nn.Module):
    """
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
        k_size: int = 9,
        modality_num: int = 4,
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

        self.shuffle = ChannelShuffle(groups=modality_num)
        self.attn = ECAttention(k_size=k_size)

    def forward(self, inp):
        B, C, S, H, W = inp.shape
        residual = inp
        out = self.shuffle.channel_shuffle(inp)
        out = self.attn(out)
        out = self.conv1(out)
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



class GlobalIntegratedModuleV2(nn.Module):
    """
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
        upsample_kernel_size: Union[Sequence[int], int],
        k_size: int = 9,
        modality_num: int = 4,
        norm_name: Union[Tuple, str] = "instance",
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        
        self.transp_conv_init = get_conv_layer(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_kernel_size,
            conv_only=True,
            is_transposed=True,
        )
        
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
        self.norm_transp = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=in_channels)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm3 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

        self.downsample = in_channels != out_channels
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True

        self.shuffle = ChannelShuffle(groups=modality_num)
        self.attn = ECAttention(k_size=k_size)

    def forward(self, inp):
        out = self.transp_conv_init(inp)
        out = self.norm_transp(out)
        out = self.shuffle.channel_shuffle(out)
        out = self.attn(out)
        out = self.lrelu(out)
        
        residual = out        
        out = self.conv1(out)
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
    
    
class ECAttention(nn.Module):
    def __init__(self, k_size=9):
        super(ECAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
    

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avgout, maxout], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # featurize dataset download c369a55a-2c7c-4273-a48b-0c90e4fef9e3

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class UnetBasicBlock(nn.Module):
    """
    A CNN module module that can be used for DynUNet, based on:
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
        norm_name: Union[Tuple, str],
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
            spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1, dropout=dropout, conv_only=True
        )
        self.lrelu = get_act_layer(name=act_name)
        self.norm1 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.norm2 = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.lrelu(out)
        return out


class UnetUpBlock(nn.Module):
    """
    An upsampling module that can be used for DynUNet, based on:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.
    `nnU-Net: Self-adapting Framework for U-Net-Based Medical Image Segmentation <https://arxiv.org/abs/1809.10486>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        stride: convolution stride.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        norm_name: feature normalization type and arguments.
        act_name: activation layer type and arguments.
        dropout: dropout probability.
        trans_bias: transposed convolution bias.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        trans_bias: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            conv_only=True,
            is_transposed=True,
        )
        self.conv_block = UnetBasicBlock(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
        )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UnetOutBlock(nn.Module):
    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, dropout: Optional[Union[Tuple, str, float]] = None
    ):
        super().__init__()
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.conv = get_conv_layer(
            spatial_dims, in_channels, out_channels, kernel_size=1, stride=1, dropout=dropout, bias=True, conv_only=True
        )

    def forward(self, inp):
        return self.conv(inp)


class ResBlockSimple(nn.Module):
    """
    Simple residual block implemented following the paper description [125].
    Key assumption: must reduce the channel count from in_channels (typically 2*C) to out_channels (typically C)
                    to match the residual connection in NIM (Eq 17).
    Structure: Conv1(2C->C) -> Norm1 -> Act -> Conv2(C->C) -> Norm2
    """
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int,
                 norm_name: Union[Tuple, str] = "instance",
                 act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01})):
        super().__init__()
        self.conv1 = get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=3, stride=1, conv_only=True)
        self.norm1 = get_norm_layer(norm_name, spatial_dims=spatial_dims, channels=out_channels)
        self.act = get_act_layer(act_name)
        self.conv2 = get_conv_layer(spatial_dims, out_channels, out_channels, kernel_size=3, stride=1, conv_only=True)
        self.norm2 = get_norm_layer(norm_name, spatial_dims=spatial_dims, channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class NeighborAttention(nn.Module):
    """
    Implementation of the Neighbor Attention (NA) module (Paper Fig 4c, Eq 15-16).
    """
    def __init__(self, k_size: int = 9):
        super().__init__()
        # Input is assumed to be the concatenation of the main and minor modalities, hence groups=2
        self.shuffle = ChannelShuffle(groups=2)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_main: torch.Tensor, x_minor: torch.Tensor) -> torch.Tensor:
        # 1. Concatenate along channel dimension
        x = torch.cat((x_main, x_minor), dim=1) # Shape: (B, 2*C, S, H, W)

        # 2. Apply channel shuffle
        x_shuffled = self.shuffle.channel_shuffle(x) # Shape: (B, 2*C, S, H, W)

        # 3. Apply attention (GAP + Conv1D + Sigmoid + Multiply)
        y = self.avg_pool(x_shuffled)
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        # Return the weighted feature u_mi^l
        return x_shuffled * y.expand_as(x)


class NIM(nn.Module):
    """
    Implementation of the Neighborhood Integrated Module (NIM) (Paper Fig 4b, Eq 17).
    """
    def __init__(self, spatial_dims: int, channels: int, k_size: int = 9,
                 norm_name: Union[Tuple, str] = "instance",
                 act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
                 kernel_size=3, stride=1):
        super().__init__()
        self.channels = channels
        self.na = NeighborAttention(k_size=k_size)

        # ResBlock input is the NA output (2*channels); output must be channels to add with x_main_orig
        self.res_block = UnetResBlock(spatial_dims=spatial_dims,
                                        in_channels=channels * 2,
                                        out_channels=channels,
                                        norm_name=norm_name,
                                        act_name=act_name,
                                        kernel_size=kernel_size, 
                                        stride=stride)

    def forward(self, x_main_acc: torch.Tensor, x_minor: torch.Tensor, x_main_orig: torch.Tensor) -> torch.Tensor:
        # 1. Apply Neighbor Attention
        # Inputs: x_main_acc (from the previous NIM or the initial x_main) and x_minor
        u_mi = self.na(x_main_acc, x_minor) # Shape: (B, 2*C, S, H, W)

        # 2. Apply ResBlock (reduce channels from 2C to C)
        res_out = self.res_block(u_mi) # Shape: (B, C, S, H, W)

        # 3. Add the residual from the original main-modality feature (Eq 17)
        # x_main_orig is the initial x_main (x_m0^l)
        out = res_out + x_main_orig # Shape: (B, C, S, H, W)
        return out


class PNIM(nn.Module):
    """
    Implementation of the Progressive Neighborhood Integrated Module (PNIM) (Paper Fig 4a).
    Built from cascaded NIM modules.
    """
    def __init__(self, spatial_dims: int, channels: int, k_size: int = 9,  kernel_size=3, stride=1,
                 norm_name: Union[Tuple, str] = "instance",
                 act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01})):
        super().__init__()
        self.nim1 = NIM(spatial_dims, channels, k_size, norm_name, act_name, kernel_size, stride)
        self.nim2 = NIM(spatial_dims, channels, k_size, norm_name, act_name, kernel_size, stride)
        self.nim3 = NIM(spatial_dims, channels, k_size, norm_name, act_name, kernel_size, stride)

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x_main (torch.Tensor): Initial main-modality feature (x_m0^l) - Shape (B, C, S, H, W)
            x_minor1 (torch.Tensor): First minor-modality feature (x_1^l) - Shape (B, C, S, H, W)
            x_minor2 (torch.Tensor): Second minor-modality feature (x_2^l) - Shape (B, C, S, H, W)
            x_minor3 (torch.Tensor): Third minor-modality feature (x_3^l) - Shape (B, C, S, H, W)

        Returns:
            torch.Tensor: Final PNIM output (x_m3^l) - Shape (B, C, S, H, W)
        """
        x_main, x_minor1, x_minor2, x_minor3 = torch.chunk(x, 4, dim=1)
        # Apply NIM in cascade
        # First injection
        x_m1 = self.nim1(x_main, x_minor1, x_main)
        # Second injection
        x_m2 = self.nim2(x_m1, x_minor2, x_main)
        # Third injection
        x_m3 = self.nim3(x_m2, x_minor3, x_main)

        return x_m3


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
