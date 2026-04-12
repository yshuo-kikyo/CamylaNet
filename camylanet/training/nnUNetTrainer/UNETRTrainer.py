import torch
import torch.nn as nn
from camylanet.network_architecture.unetr import UNETR
from camylanet.training.nnUNetTrainer.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from camylanet.utilities.plans_handling.plans_handler import ConfigurationManager
from camylanet.utilities.label_handling.label_handling import determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP
from batchgenerators.utilities.file_and_folder_operations import *


class UNETRTrainer(nnUNetTrainerNoDeepSupervision):
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

        encoder_kernel = [[(1, 2, 2), (2, 2, 2), (2, 2, 2)], 
                          [(1, 2, 2), (2, 2, 2)], 
                          [(1, 2, 2)]]

        decoder_kernel =  [(1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]

        network = UNETR(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            crop_size=configuration_manager.patch_size,
            patch_size=(8, 16, 16),
            feature_size=16,
            hidden_size=768,
            encoder_kernel_sizes=encoder_kernel,
            decoder_kernel_sizes=decoder_kernel,
            deep_supervision=False
        )

        return network