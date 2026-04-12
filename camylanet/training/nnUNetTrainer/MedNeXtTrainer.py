import torch
import os
import torch.nn as nn
from camylanet.network_architecture.mednextv1.MedNextV1 import MedNeXt as MedNeXt_Orig
from camylanet.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from camylanet.network_architecture.neural_network import SegmentationNetwork
from camylanet.network_architecture.utils import softmax_helper

import torch
from camylanet.training.nnUNetTrainer.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from camylanet.utilities.plans_handling.plans_handler import ConfigurationManager
from camylanet.utilities.label_handling.label_handling import determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP
from batchgenerators.utilities.file_and_folder_operations import *
from torch import nn


class MedNeXtTrainer(nnUNetTrainerNoDeepSupervision):
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
        network = MedNeXt(
            in_channels=num_input_channels, 
            n_channels=32,
            n_classes=num_output_channels, 
            exp_r=2,
            kernel_size=5,
            deep_supervision=False,
            do_res=True,
            do_res_up_down=True,
            block_counts=[2,2,2,2,2,2,2,2,2]
        )


        return network
    

class MedNeXt(MedNeXt_Orig, SegmentationNetwork):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Segmentation Network Params. Needed for the nnUNet evaluation pipeline
        self.conv_op = nn.Conv3d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 2**5
        self.num_classes = kwargs['n_classes']
        # self.do_ds = False        Already added this in the main class


class nnUNetTrainerV2_Optim_and_LR(nnUNetTrainer):

    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-3

    def process_plans(self, plans):
        super().process_plans(plans)
        # Please don't do this for nnunet. This is only for MedNeXt for all the DS to be used
        num_of_outputs_in_mednext = 5
        self.net_num_pool_op_kernel_sizes = [[2,2,2] for i in range(num_of_outputs_in_mednext+1)]    
    
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), 
                                            self.initial_lr, 
                                            weight_decay=self.weight_decay,
                                            eps=1e-4        # 1e-8 might cause nans in fp16
                                        )
        self.lr_scheduler = None


class nnUNetTrainerV2_MedNeXt_S_kernel3(nnUNetTrainerV2_Optim_and_LR):   
    
    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=2,         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [2,2,2,2,2,2,2,2,2]
        )

        if torch.cuda.is_available():
            self.network.cuda()


class nnUNetTrainerV2_MedNeXt_B_kernel3(nnUNetTrainerV2_Optim_and_LR):   
        
    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [2,2,2,2,2,2,2,2,2]
        )

        if torch.cuda.is_available():
            self.network.cuda()


class nnUNetTrainerV2_MedNeXt_M_kernel3(nnUNetTrainerV2_Optim_and_LR):   
        
    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [3,4,4,4,4,4,4,4,3],
            checkpoint_style = 'outside_block'
        )

        if torch.cuda.is_available():
            self.network.cuda()


class nnUNetTrainerV2_MedNeXt_L_kernel3(nnUNetTrainerV2_Optim_and_LR):   
        
    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=[3,4,8,8,8,8,8,4,3],         # Expansion ratio as in Swin Transformers
            # exp_r=[3,4,8,8,8,8,8,4,3],         # Expansion ratio as in Swin Transformers
            kernel_size=3,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            # block_counts = [6,6,6,6,4,2,2,2,2],
            block_counts = [3,4,8,8,8,8,8,4,3],
            checkpoint_style = 'outside_block'
        )

        if torch.cuda.is_available():
            self.network.cuda()


# Kernels of size 5
class nnUNetTrainerV2_MedNeXt_S_kernel5(nnUNetTrainerV2_Optim_and_LR):   

    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=2,                           # Expansion ratio as in Swin Transformers
            kernel_size=5,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                       # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [2,2,2,2,2,2,2,2,2]
        )

        if torch.cuda.is_available():
            self.network.cuda()


class nnUNetTrainerV2_MedNeXt_S_kernel5_lr_1e_4(nnUNetTrainerV2_MedNeXt_S_kernel5):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-4


class nnUNetTrainerV2_MedNeXt_S_kernel5_lr_25e_5(nnUNetTrainerV2_MedNeXt_S_kernel5):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 25e-5


class nnUNetTrainerV2_MedNeXt_B_kernel5(nnUNetTrainerV2_Optim_and_LR):   

    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
            kernel_size=5,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [2,2,2,2,2,2,2,2,2]
        )

        if torch.cuda.is_available():
            self.network.cuda()


class nnUNetTrainerV2_MedNeXt_B_kernel5_lr_5e_4(nnUNetTrainerV2_MedNeXt_B_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 5e-4


class nnUNetTrainerV2_MedNeXt_B_kernel5_lr_25e_5(nnUNetTrainerV2_MedNeXt_B_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 25e-5


class nnUNetTrainerV2_MedNeXt_B_kernel5_lr_1e_4(nnUNetTrainerV2_MedNeXt_B_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-4


class nnUNetTrainerV2_MedNeXt_M_kernel5(nnUNetTrainerV2_Optim_and_LR):   

    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],         # Expansion ratio as in Swin Transformers
            kernel_size=5,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            block_counts = [3,4,4,4,4,4,4,4,3],
            checkpoint_style = 'outside_block'
        )

        if torch.cuda.is_available():
            self.network.cuda()


class nnUNetTrainerV2_MedNeXt_M_kernel5_lr_5e_4(nnUNetTrainerV2_MedNeXt_M_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 5e-4


class nnUNetTrainerV2_MedNeXt_M_kernel5_lr_25e_5(nnUNetTrainerV2_MedNeXt_M_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 25e-5


class nnUNetTrainerV2_MedNeXt_M_kernel5_lr_1e_4(nnUNetTrainerV2_MedNeXt_M_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-4


class nnUNetTrainerV2_MedNeXt_L_kernel5(nnUNetTrainerV2_Optim_and_LR):   
    
    def initialize_network(self):
        self.network = MedNeXt(
            in_channels = self.num_input_channels, 
            n_channels = 32,
            n_classes = self.num_classes, 
            exp_r=[3,4,8,8,8,8,8,4,3],         # Expansion ratio as in Swin Transformers
            kernel_size=5,                     # Can test kernel_size
            deep_supervision=True,             # Can be used to test deep supervision
            do_res=True,                      # Can be used to individually test residual connection
            do_res_up_down = True,
            # block_counts = [6,6,6,6,4,2,2,2,2],
            block_counts = [3,4,8,8,8,8,8,4,3],
            checkpoint_style = 'outside_block'
        )

        if torch.cuda.is_available():
            self.network.cuda()


class nnUNetTrainerV2_MedNeXt_L_kernel5_lr_5e_4(nnUNetTrainerV2_MedNeXt_L_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 5e-4


class nnUNetTrainerV2_MedNeXt_L_kernel5_lr_25e_5(nnUNetTrainerV2_MedNeXt_L_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 25e-5


class nnUNetTrainerV2_MedNeXt_L_kernel5_lr_1e_4(nnUNetTrainerV2_MedNeXt_L_kernel5):   
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-4

