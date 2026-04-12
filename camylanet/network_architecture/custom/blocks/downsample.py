from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from camylanet.network_architecture.custom.layers.factories import Pool
from camylanet.network_architecture.custom.utils.misc import ensure_tuple_rep


class MaxAvgPool(nn.Module):
    """
    Downsample with both maxpooling and avgpooling,
    double the channel size by concatenating the downsampled feature maps.
    """

    def __init__(
        self,
        spatial_dims: int,
        kernel_size: Union[Sequence[int], int],
        stride: Optional[Union[Sequence[int], int]] = None,
        padding: Union[Sequence[int], int] = 0,
        ceil_mode: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            kernel_size: the kernel size of both pooling operations.
            stride: the stride of the window. Default value is `kernel_size`.
            padding: implicit zero padding to be added to both pooling operations.
            ceil_mode: when True, will use ceil instead of floor to compute the output shape.
        """
        super().__init__()
        _params = {
            "kernel_size": ensure_tuple_rep(kernel_size, spatial_dims),
            "stride": None if stride is None else ensure_tuple_rep(stride, spatial_dims),
            "padding": ensure_tuple_rep(padding, spatial_dims),
            "ceil_mode": ceil_mode,
        }
        self.max_pool = Pool[Pool.MAX, spatial_dims](**_params)
        self.avg_pool = Pool[Pool.AVG, spatial_dims](**_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor in shape (batch, channel, spatial_1[, spatial_2, ...]).

        Returns:
            Tensor in shape (batch, 2*channel, spatial_1[, spatial_2, ...]).
        """
        return torch.cat([self.max_pool(x), self.avg_pool(x)], dim=1)
