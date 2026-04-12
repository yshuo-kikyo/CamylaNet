import torch
import torch.nn as nn
from camylanet.training.nnUNetTrainer.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from camylanet.utilities.plans_handling.plans_handler import ConfigurationManager
from camylanet.utilities.label_handling.label_handling import determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP
from batchgenerators.utilities.file_and_folder_operations import *

# Import UKAN network architecture
from camylanet.network_architecture.ukan import UKAN


class UKANTrainer(nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 plans_identifier: str = 'nnUNetPlans', device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, plans_identifier, device)
        self.num_epochs = 200
    """
    UKAN Trainer: Trainer for U-Net with Kolmogorov-Arnold Networks
    
    This trainer implements the training logic for UKAN networks, which combine
    traditional U-Net architecture with KAN layers for enhanced feature processing.
    """
    
    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)
            
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
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")
        

    @staticmethod
    def build_network_architecture(configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int) -> nn.Module:
        """
        Build UKAN network architecture
        
        Args:
            configuration_manager: Configuration manager containing network settings
            num_input_channels: Number of input channels
            num_output_channels: Number of output channels (segmentation classes)
            
        Returns:
            nn.Module: UKAN network instance
        """
        
        # Get configuration parameters
        patch_size = configuration_manager.patch_size
        
        # Determine if we should use 2D or 3D based on patch size
        is_2d = len(patch_size) == 2
        
        # UKAN configuration parameters
        if is_2d:
            # 2D configuration
            embed_dims = [256, 320, 512]  # KAN embedding dimensions for 2D
            img_size = max(patch_size)  # Use the larger dimension as image size
        else:
            # 3D configuration
            embed_dims = [128, 160, 256]  # Smaller dimensions for 3D to save memory
            img_size = max(patch_size)  # Use the larger dimension as image size
        
        # Build UKAN network
        network = UKAN(
            num_classes=num_output_channels,
            input_channels=num_input_channels,
            deep_supervision=False,  # UKAN doesn't support deep supervision by default
            img_size=img_size,
            patch_size=16,  # Default patch size for tokenization
            in_chans=num_input_channels,
            embed_dims=embed_dims,
            no_kan=False,  # Enable KAN layers by default
            drop_rate=0.1,  # Dropout rate
            drop_path_rate=0.1,  # Drop path rate for regularization
            norm_layer=nn.LayerNorm,
            depths=[1, 1, 1]  # Number of KAN blocks at each stage
        )

        return network
    
    def _build_loss(self):
        """
        Build loss function for UKAN training
        
        Returns:
            Loss function instance
        """
        # Use the same loss as the parent class
        return super()._build_loss()
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers for UKAN training
        
        Returns:
            Tuple of (optimizer, lr_scheduler)
        """
        # Use the same optimizer configuration as the parent class
        return super().configure_optimizers()
    
    def on_epoch_end(self):
        """
        Called at the end of each epoch
        Can be used for custom logging or model adjustments
        """
        super().on_epoch_end()
        
        # Add custom logging for UKAN-specific metrics if needed
        if hasattr(self.network, 'kan_layers'):
            # Log KAN layer statistics
            for i, layer in enumerate(self.network.kan_layers):
                if hasattr(layer, 'regularization_loss'):
                    reg_loss = layer.regularization_loss()
                    self.print_to_log_file(f'KAN Layer {i} regularization loss: {reg_loss:.6f}')
    
    def on_batch_end(self):
        """
        Called at the end of each batch
        Can be used for custom batch-level operations
        """
        super().on_batch_end()
        
        # Add any custom batch-level operations here if needed
        pass
