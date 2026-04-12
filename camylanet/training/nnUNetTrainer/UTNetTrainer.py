import torch
import torch.nn as nn
from camylanet.network_architecture.custom_modules.custom_networks.UTNet.utnet import UTNet as UTNet_Orig
from camylanet.network_architecture.neural_network import SegmentationNetwork
from camylanet.network_architecture.utils import softmax_helper
from camylanet.training.nnUNetTrainer.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from camylanet.utilities.plans_handling.plans_handler import ConfigurationManager
from camylanet.utilities.label_handling.label_handling import determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP
from batchgenerators.utilities.file_and_folder_operations import *


class UTNetTrainer(nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset, plans_identifier, device): 
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, plans_identifier, device)
        self.initial_lr = 5e-4

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

        network = UTNet(
            in_chan=num_input_channels, 
            base_chan=32, 
            num_classes=num_output_channels,
            block_list='1234', 
            num_blocks=[1,1,1,1], 
            num_heads=[2,4,8,16],
            dummy=False,
        )

        return network
    

class UTNetTrainer_1000epochs(UTNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 plans_identifier: str = 'nnUNetPlans', device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, plans_identifier, device)
        self.num_epochs = 1000
    

# Wrapper using SegmentationNetwork object
class UTNet(UTNet_Orig, SegmentationNetwork):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Segmentation Network Params. Needed for the nnUNet evaluation pipeline
        self.conv_op = nn.Conv2d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 16 # just some random val 2**5
        self.num_classes = kwargs['num_classes']
        self.do_ds = False
