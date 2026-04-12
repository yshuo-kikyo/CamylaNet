import torch
from camylanet.network_architecture.swinunet import nnFormer
from camylanet.training.nnUNetTrainer.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from camylanet.utilities.plans_handling.plans_handler import ConfigurationManager
from camylanet.utilities.label_handling.label_handling import determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP
from batchgenerators.utilities.file_and_folder_operations import *
from torch import nn
from torch._dynamo import OptimizedModule


class nnFormerTrainer(nnUNetTrainerNoDeepSupervision):
    """
    nnFormer launch
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
        network = nnFormer(
            crop_size=configuration_manager.patch_size, 
            # crop_size=[64, 128, 128],
            input_channels=num_input_channels, 
            num_classes=num_output_channels,
            # pool_op_kernel_sizes=configuration_manager.pool_op_kernel_sizes,
            embedding_dim=192
        )

        return network
