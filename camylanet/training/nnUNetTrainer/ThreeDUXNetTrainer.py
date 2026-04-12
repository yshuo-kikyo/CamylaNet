import torch
import torch.nn as nn
from camylanet.network_architecture.custom_modules.custom_networks.UXNet3D. \
    network_backbone import UXNET as UXNET_Orig
from camylanet.training.nnUNetTrainer.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from camylanet.network_architecture.neural_network import SegmentationNetwork
from camylanet.network_architecture.utils import softmax_helper
from typing import Union, Tuple, List
from torch import nn
from torch._dynamo import OptimizedModule


class UXNET(UXNET_Orig, SegmentationNetwork):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Segmentation Network Params. Needed for the nnUNet evaluation pipeline
        self.conv_op = nn.Conv3d
        self.inference_apply_nonlin = softmax_helper
        self.input_shape_must_be_divisible_by = 2**5
        self.num_classes = kwargs['out_chans']
        self.do_ds = False        


class ThreeDUXNetTrainer(nnUNetTrainerNoDeepSupervision):

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset, plans_identifier, device): 
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, plans_identifier, device)
        self.initial_lr = 5e-4

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        network = UXNET(
            in_chans=num_input_channels,
            out_chans=num_output_channels,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            spatial_dims=3,
        )

        return network
    

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
        chances you need to change this as well!
        """
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        # mod.decoder1.deep_supervision = enabled
