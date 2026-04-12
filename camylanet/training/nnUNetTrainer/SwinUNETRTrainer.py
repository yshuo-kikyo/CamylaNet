"""
SwinUNETR Trainer for camylanet

Uses the MONAI 1.5.x SwinUNETR implementation, which supports flexible input sizes (no need to specify img_size).

Size constraints:
- Each spatial dimension should be divisible by 32
- Minimum size is roughly 64 (2D) or 32 (per dimension)
- Non-square / non-cubic inputs are supported

Successfully tested sizes:
- 3D: (64, 320, 320), (128, 128, 128), (32, 128, 128), (64, 64, 64), (96, 96, 96)
- 2D: (256, 256), (512, 512), (512, 320), (128, 128), (96, 96), (224, 224), (384, 384)

Sizes that may fail (divisibility problems):
- 3D: Z axis too small (e.g. 16, 8) or not divisible by 32 (e.g. 48, 24)
- 2D: dimensions too small (e.g. 48, 56, 32) or not divisible by 32 (e.g. 100)
"""
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR as SwinUNETR_MONAI
from camylanet.training.nnUNetTrainer.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from camylanet.network_architecture.neural_network import SegmentationNetwork
from camylanet.network_architecture.utils import softmax_helper
from camylanet.utilities.plans_handling.plans_handler import ConfigurationManager
from camylanet.utilities.label_handling.label_handling import determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP


# Wrapper using SegmentationNetwork object for nnUNet compatibility
class SwinUNETR(SwinUNETR_MONAI, SegmentationNetwork):
    """
    SwinUNETR wrapper for camylanet pipeline compatibility.

    Based on MONAI 1.5.x SwinUNETR, supporting dynamic input sizes (no img_size argument required).
    """

    def __init__(self, spatial_dims: int = 3, **kwargs):
        super().__init__(spatial_dims=spatial_dims, **kwargs)
        # Segmentation Network Params. Needed for the nnUNet evaluation pipeline
        self.conv_op = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 32  # SwinUNETR requires dimensions divisible by 32
        self.num_classes = kwargs['out_channels']
        self.do_ds = False


class SwinUNETRTrainer(nnUNetTrainerNoDeepSupervision):
    """
    SwinUNETR Trainer - supports flexible input sizes.

    Features:
    - Uses MONAI 1.5.x SwinUNETR (no img_size; automatically adapts to the input size)
    - Supports both 2D and 3D modes (detected automatically from the configuration)
    - Supports non-square / non-cubic inputs

    Notes:
    - Each spatial dimension should be divisible by 32
    - In 3D mode, the Z axis should not be too small (>= 32 recommended)
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        plans_identifier: str = 'nnUNetPlans',
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, plans_identifier, device)
        self.enable_deep_supervision = False
        self.initial_lr = 5e-4

    def configure_optimizers(self):
        """Use the AdamW optimizer, which is better suited to Transformer architectures."""
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-4
        )
        # Use the PolyLR scheduler
        from camylanet.training.lr_scheduler.polylr import PolyLRScheduler
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )

            self.network = self.build_network_architecture(
                self.configuration_manager,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
            ).to(self.device)

            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()

            # Log network configuration info
            self._log_network_info()

            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")

    def _log_network_info(self):
        """Log network configuration info."""
        patch_size = self.configuration_manager.patch_size
        spatial_dims = len(patch_size)

        self.print_to_log_file(f"SwinUNETR Configuration:")
        self.print_to_log_file(f"  - Spatial dims: {spatial_dims}D")
        self.print_to_log_file(f"  - Patch size: {patch_size}")
        self.print_to_log_file(f"  - Input channels: {self.num_input_channels}")
        self.print_to_log_file(f"  - Output channels: {self.label_manager.num_segmentation_heads}")

        # Check size constraints
        warnings = []
        for i, dim in enumerate(patch_size):
            if dim < 32:
                warnings.append(f"Dimension {i} = {dim} is very small (< 32)")
            if dim % 32 != 0:
                warnings.append(f"Dimension {i} = {dim} is not divisible by 32")

        if warnings:
            self.print_to_log_file("  ⚠️ Size Warnings:")
            for w in warnings:
                self.print_to_log_file(f"    - {w}")
        else:
            self.print_to_log_file("  ✅ All dimensions satisfy SwinUNETR constraints")

    @staticmethod
    def build_network_architecture(
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int
    ) -> nn.Module:
        """
        Build the SwinUNETR network architecture.

        In MONAI 1.5.x, SwinUNETR no longer needs an img_size argument and will adapt automatically
        to any input size that satisfies the divisibility requirements.

        Args:
            configuration_manager: configuration manager
            num_input_channels: number of input channels
            num_output_channels: number of output channels (segmentation classes)

        Returns:
            A SwinUNETR network instance.
        """
        patch_size = configuration_manager.patch_size
        spatial_dims = len(patch_size)

        # Adjust feature_size based on spatial dimensions
        # 2D uses a larger feature_size; 3D uses a smaller one to save GPU memory
        if spatial_dims == 2:
            feature_size = 48
        else:
            # In 3D mode, adjust based on patch size
            total_voxels = 1
            for d in patch_size:
                total_voxels *= d

            if total_voxels > 128 * 128 * 128:
                feature_size = 24  # Small feature_size for large volumes to save GPU memory
            elif total_voxels > 64 * 64 * 64:
                feature_size = 36
            else:
                feature_size = 48

        network = SwinUNETR(
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            feature_size=feature_size,
            spatial_dims=spatial_dims,
            use_checkpoint=False,  # Set to True to save GPU memory
            use_v2=False,  # Set to True to use the v2 version
        )

        return network


class SwinUNETRTrainer_100epochs(SwinUNETRTrainer):
    """100 epoch variant, for quick testing."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, plans_identifier: str = 'nnUNetPlans',
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, plans_identifier, device)
        self.num_epochs = 100


class SwinUNETRTrainer_200epochs(SwinUNETRTrainer):
    """200 epoch variant."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, plans_identifier: str = 'nnUNetPlans',
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, plans_identifier, device)
        self.num_epochs = 200


class SwinUNETRTrainer_500epochs(SwinUNETRTrainer):
    """500 epoch variant."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, plans_identifier: str = 'nnUNetPlans',
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, plans_identifier, device)
        self.num_epochs = 500


class SwinUNETRTrainer_checkpoint(SwinUNETRTrainer):
    """Variant that uses gradient checkpointing to save GPU memory."""

    @staticmethod
    def build_network_architecture(
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int
    ) -> nn.Module:
        patch_size = configuration_manager.patch_size
        spatial_dims = len(patch_size)

        if spatial_dims == 2:
            feature_size = 48
        else:
            feature_size = 24  # With checkpointing we can use a smaller feature_size

        network = SwinUNETR(
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            feature_size=feature_size,
            spatial_dims=spatial_dims,
            use_checkpoint=True,  # Enable gradient checkpointing
            use_v2=False,
        )

        return network
