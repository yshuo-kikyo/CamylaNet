import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional

# thop for FLOPs/Params profiling
try:
    from thop import profile, clever_format
except Exception as _e:
    profile = None
    clever_format = None


class KANLinear(torch.nn.Module):
    """
    KAN Linear Layer: Learnable activation functions using B-spline basis
    
    This layer implements the Kolmogorov-Arnold Networks approach with
    learnable activation functions based on B-spline interpolation.
    """
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute regularization loss for KAN layers
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KANLayer(nn.Module):
    """
    KAN Layer: Composite layer containing multiple KAN linear layers and depthwise convolutions
    """
    def __init__(self, dim, no_kan=False, is_3d=False):
        super().__init__()
        
        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]
        
        if not no_kan:
            self.fc1 = KANLinear(
                dim, dim,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.fc2 = KANLinear(
                dim, dim,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.fc3 = KANLinear(
                dim, dim,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
        else:
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)
            self.fc3 = nn.Linear(dim, dim)
        
        # Depthwise separable convolutions
        if is_3d:
            self.dwconv1 = DWConv3D(dim)
            self.dwconv2 = DWConv3D(dim)
            self.dwconv3 = DWConv3D(dim)
        else:
            self.dwconv1 = DWConv2D(dim)
            self.dwconv2 = DWConv2D(dim)
            self.dwconv3 = DWConv2D(dim)
        
    def forward(self, x, H, W, D=None):
        B, N, C = x.shape
        
        x = self.fc1(x.reshape(B*N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv1(x, H, W, D)
        
        x = self.fc2(x.reshape(B*N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv2(x, H, W, D)
        
        x = self.fc3(x.reshape(B*N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv3(x, H, W, D)
        
        return x


class DWConv2D(nn.Module):
    """2D Depthwise Separable Convolution"""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()
    
    def forward(self, x, H, W, D=None):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DWConv3D(nn.Module):
    """3D Depthwise Separable Convolution"""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm3d(dim)
        self.relu = nn.ReLU()
    
    def forward(self, x, H, W, D):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, D, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchEmbed(nn.Module):
    """Patch Embedding for feature tokenization"""
    def __init__(self, in_channels, embed_dim, patch_size=3, stride=2, is_3d=False):
        super().__init__()
        if is_3d:
            self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, 
                                 stride=stride, padding=patch_size//2)
        else:
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, 
                                 stride=stride, padding=patch_size//2)
        self.norm = nn.LayerNorm(embed_dim)
        self.is_3d = is_3d
    
    def forward(self, x):
        x = self.proj(x)
        if self.is_3d:
            _, _, D, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            return x, D, H, W
        else:
            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            return x, H, W


class ConvLayer(nn.Module):
    """Basic convolutional layer block"""
    def __init__(self, in_ch, out_ch, is_3d=False):
        super().__init__()
        if is_3d:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, input):
        return self.conv(input)


class D_ConvLayer(nn.Module):
    """Decoder convolutional layer block"""
    def __init__(self, in_ch, out_ch, is_3d=False):
        super().__init__()
        if is_3d:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, in_ch, 3, padding=1),
                nn.BatchNorm3d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, input):
        return self.conv(input)


class UKAN(nn.Module):
    """
    UKAN: U-Net with Kolmogorov-Arnold Networks
    
    A hybrid architecture combining traditional U-Net with KAN layers
    for enhanced feature processing and learnable activation functions.
    """
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, 
                 img_size=224, patch_size=16, in_chans=3, embed_dims=[256, 320, 512], 
                 no_kan=False, drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, 
                 depths=[1, 1, 1], **kwargs):
        super().__init__()
        
        # Determine if this is a 3D network based on embed_dims size
        self.is_3d = len(embed_dims) > 3  # If more than 3 dimensions, assume 3D
        
        kan_input_dim = embed_dims[0]
        
        # Encoder layers (traditional CNN)
        self.encoder1 = ConvLayer(input_channels, kan_input_dim//8, self.is_3d)
        self.encoder2 = ConvLayer(kan_input_dim//8, kan_input_dim//4, self.is_3d)
        self.encoder3 = ConvLayer(kan_input_dim//4, kan_input_dim, self.is_3d)
        
        # KAN layers
        self.kan_layer1 = KANLayer(embed_dims[1], no_kan=no_kan, is_3d=self.is_3d)
        self.kan_layer2 = KANLayer(embed_dims[2], no_kan=no_kan, is_3d=self.is_3d)
        self.kan_layer3 = KANLayer(embed_dims[1], no_kan=no_kan, is_3d=self.is_3d)
        self.kan_layer4 = KANLayer(embed_dims[0], no_kan=no_kan, is_3d=self.is_3d)
        
        # Store KAN layers for regularization loss calculation
        self.kan_layers = [self.kan_layer1, self.kan_layer2, self.kan_layer3, self.kan_layer4]
        
        # Patch embedding for tokenization
        self.patch_embed1 = PatchEmbed(kan_input_dim, embed_dims[1], is_3d=self.is_3d)
        self.patch_embed2 = PatchEmbed(embed_dims[1], embed_dims[2], is_3d=self.is_3d)
        
        # Decoder layers
        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1], self.is_3d)
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0], self.is_3d)
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0]//4, self.is_3d)
        self.decoder4 = D_ConvLayer(embed_dims[0]//4, embed_dims[0]//8, self.is_3d)
        self.decoder5 = D_ConvLayer(embed_dims[0]//8, embed_dims[0]//8, self.is_3d)
        
        # Final output
        if self.is_3d:
            self.final = nn.Conv3d(embed_dims[0]//8, num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(embed_dims[0]//8, num_classes, kernel_size=1)
        
        # Normalization layers
        self.norm1 = norm_layer(embed_dims[1])
        self.norm2 = norm_layer(embed_dims[2])
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[0])
        
    def forward(self, x):
        B = x.shape[0]
        
        # Encoder path
        if self.is_3d:
            # 3D operations
            enc1 = F.relu(F.max_pool3d(self.encoder1(x), 2))
            enc2 = F.relu(F.max_pool3d(self.encoder2(enc1), 2))
            enc3 = F.relu(F.max_pool3d(self.encoder3(enc2), 2))
        else:
            # 2D operations
            enc1 = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
            enc2 = F.relu(F.max_pool2d(self.encoder2(enc1), 2, 2))
            enc3 = F.relu(F.max_pool2d(self.encoder3(enc2), 2, 2))
        
        # KAN tokenization stage 1
        if self.is_3d:
            out, D, H, W = self.patch_embed1(enc3)
            out = self.kan_layer1(out, H, W, D)
            out = self.norm1(out)
            out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        else:
            out, H, W = self.patch_embed1(enc3)
            out = self.kan_layer1(out, H, W)
            out = self.norm1(out)
            out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        skip1 = out
        
        # KAN bottleneck
        if self.is_3d:
            out, D, H, W = self.patch_embed2(out)
            out = self.kan_layer2(out, H, W, D)
            out = self.norm2(out)
            out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        else:
            out, H, W = self.patch_embed2(out)
            out = self.kan_layer2(out, H, W)
            out = self.norm2(out)
            out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        # Decoder path with KAN
        if self.is_3d:
            out = F.interpolate(self.decoder1(out), scale_factor=2, mode='trilinear', align_corners=False)
        else:
            out = F.interpolate(self.decoder1(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.add(out, skip1)
        
        # KAN processing in decoder
        if self.is_3d:
            _, _, D, H, W = out.shape
            out_flat = out.flatten(2).transpose(1, 2)
            out_flat = self.kan_layer3(out_flat, H, W, D)
            out = self.norm3(out_flat).reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        else:
            _, _, H, W = out.shape
            out_flat = out.flatten(2).transpose(1, 2)
            out_flat = self.kan_layer3(out_flat, H, W)
            out = self.norm3(out_flat).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        if self.is_3d:
            out = F.interpolate(self.decoder2(out), scale_factor=2, mode='trilinear', align_corners=False)
        else:
            out = F.interpolate(self.decoder2(out), scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.add(out, enc3)
        
        # Final KAN processing
        if self.is_3d:
            _, _, D, H, W = out.shape
            out_flat = out.flatten(2).transpose(1, 2)
            out_flat = self.kan_layer4(out_flat, H, W, D)
            out = self.norm4(out_flat).reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        else:
            _, _, H, W = out.shape
            out_flat = out.flatten(2).transpose(1, 2)
            out_flat = self.kan_layer4(out_flat, H, W)
            out = self.norm4(out_flat).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        # Traditional decoder layers
        if self.is_3d:
            out = F.interpolate(self.decoder3(out), scale_factor=2, mode='trilinear', align_corners=False)
            out = torch.add(out, enc2)
            out = F.interpolate(self.decoder4(out), scale_factor=2, mode='trilinear', align_corners=False)
            out = torch.add(out, enc1)
            out = F.interpolate(self.decoder5(out), scale_factor=2, mode='trilinear', align_corners=False)
        else:
            out = F.interpolate(self.decoder3(out), scale_factor=2, mode='bilinear', align_corners=False)
            out = torch.add(out, enc2)
            out = F.interpolate(self.decoder4(out), scale_factor=2, mode='bilinear', align_corners=False)
            out = torch.add(out, enc1)
            out = F.interpolate(self.decoder5(out), scale_factor=2, mode='bilinear', align_corners=False)
        
        return self.final(out)
    
    def get_kan_regularization_loss(self):
        """
        Get the total regularization loss from all KAN layers
        """
        total_reg_loss = 0.0
        for layer in self.kan_layers:
            if hasattr(layer, 'fc1') and hasattr(layer.fc1, 'regularization_loss'):
                total_reg_loss += layer.fc1.regularization_loss()
            if hasattr(layer, 'fc2') and hasattr(layer.fc2, 'regularization_loss'):
                total_reg_loss += layer.fc2.regularization_loss()
            if hasattr(layer, 'fc3') and hasattr(layer.fc3, 'regularization_loss'):
                total_reg_loss += layer.fc3.regularization_loss()
        return total_reg_loss


def compute_ukan_stats(model=None, input_size=(1, 3, 224, 224), is_3d=False):
    """
    Compute FLOPs and active-parameter statistics for UKAN.

    Args:
        model: Pre-built model; a new one is created if None.
        input_size: Input shape (batch, channels, height, width) or (batch, channels, depth, height, width).
        is_3d: Whether the model is 3D.
    """
    if profile is None:
        print("thop is not installed; cannot compute FLOPs/Params. Install it first: pip install thop")
        return None

    # Create the model if not supplied
    if model is None:
        if is_3d:
            embed_dims = [128, 160, 256]  # 3D configuration
        else:
            embed_dims = [256, 320, 512]  # 2D configuration

        model = UKAN(
            num_classes=2,  # Default 2 classes
            input_channels=input_size[1],
            deep_supervision=False,
            img_size=max(input_size[2:]),
            patch_size=16,
            in_chans=input_size[1],
            embed_dims=embed_dims,
            no_kan=False,  # Enable KAN layers
            drop_rate=0.1,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            depths=[1, 1, 1]
        )

    model.eval()

    # Count active parameters (all parameters are trainable)
    total_params = 0
    kan_params = 0
    cnn_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if 'kan' in name.lower() or 'fc' in name.lower():
            kan_params += param.numel()
        else:
            cnn_params += param.numel()
    
    # Try to compute FLOPs with thop; skip on failure
    try:
        # Create a test input
        x = torch.randn(input_size)

        macs, thop_total_params = profile(model, inputs=(x,), verbose=False)

        # Format the output
        macs_str, thop_total_params_str, total_params_str, kan_params_str, cnn_params_str = clever_format(
            [macs, thop_total_params, total_params, kan_params, cnn_params], '%.3f'
        )

        print("=" * 70)
        print("UKAN statistics:")
        print("=" * 70)
        print(f"Input size:       {input_size}")
        print(f"Model type:       {'3D' if is_3d else '2D'}")
        print(f"FLOPs (MACs):     {macs_str:>10s}")
        print(f"Total params (thop):  {thop_total_params_str:>10s}")
        print(f"Total params (manual):{total_params_str:>10s}")
        print(f"KAN layer params: {kan_params_str:>10s}")
        print(f"CNN layer params: {cnn_params_str:>10s}")
        print("=" * 70)
        
        return {
            'macs': macs,
            'total_params': total_params,
            'kan_params': kan_params,
            'cnn_params': cnn_params
        }
        
    except Exception as e:
        print(f"thop profiling failed: {e}")
        print("Showing parameter statistics only...")

        print("=" * 70)
        print("UKAN parameter statistics:")
        print("=" * 70)
        print(f"Input size:       {input_size}")
        print(f"Model type:       {'3D' if is_3d else '2D'}")
        print(f"Total params:     {total_params:,}")
        print(f"KAN layer params: {kan_params:,}")
        print(f"CNN layer params: {cnn_params:,}")
        print("=" * 70)
        
        return {
            'total_params': total_params,
            'kan_params': kan_params,
            'cnn_params': cnn_params
        }


if __name__ == "__main__":
    # Test the UKAN model
    print("Testing UKAN model...")

    # Test the 2D model
    print("\n" + "="*70)
    print("Testing UKAN 2D model...")
    print("="*70)
    
    model_2d = UKAN(
        num_classes=2,
        input_channels=3,
        deep_supervision=False,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dims=[256, 320, 512],
        no_kan=False,
        drop_rate=0.1,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        depths=[1, 1, 1]
    )
    
    # Test forward pass
    with torch.no_grad():
        x = torch.randn(1, 3, 512, 512)
        out = model_2d(x)
        print(f"2D output shape: {out.shape}")

    # Profile 2D model parameters
    stats_2d = compute_ukan_stats(
        model=model_2d, 
        input_size=(1, 3, 512, 512), 
        is_3d=False
    )