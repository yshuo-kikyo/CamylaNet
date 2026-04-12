import os
import torch
import re
import torch.nn as nn
from camylanet.network_architecture.swinumamba.nets.SwinUMamba import SwinUMamba
from camylanet.network_architecture.neural_network import SegmentationNetwork
from camylanet.network_architecture.utils import softmax_helper
from camylanet.training.nnUNetTrainer.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from camylanet.training.nnUNetTrainer.nnUNetTrainer_Xepochs import nnUNetTrainerNoDeepSupervision_1000epochs
from camylanet.utilities.plans_handling.plans_handler import ConfigurationManager
from camylanet.utilities.label_handling.label_handling import determine_num_input_channels
from torch.nn.parallel import DistributedDataParallel as DDP
from batchgenerators.utilities.file_and_folder_operations import *


class SwinUMambaTrainer(nnUNetTrainerNoDeepSupervision):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 plans_identifier: str = 'nnUNetPlans', device: torch.device = torch.device('cuda')): 
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, plans_identifier, device)
        self.initial_lr = 1e-4
        self.num_epochs = 200

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
        Build the SwinUMamba network based on the configuration and the input/output channel counts.
        """
        # Retrieve the deep supervision flag
        deep_supervision = getattr(configuration_manager,"deep_supervision",False)

        # Build the SwinUMamba network
        network = SwinUMamba(
            in_chans=num_input_channels,
            out_chans=num_output_channels,
            feat_size=[48, 96, 192, 384, 768],
            deep_supervision=deep_supervision,
            hidden_size=768,  # Adjust this value as needed
        )

        # Load pretrained weights if required
        # use_pretrain = getattr(configuration_manager,"use_pretrain", False)
        # if use_pretrain:
        #     network = load_pretrained_ckpt(network)

        return network
    
def load_pretrained_ckpt(
    model,
    ckpt_path=None,
):
    # Resolve the checkpoint path from argument → env var → error.
    # Download the Swin-UMamba / VSSM pretrained weights separately and point
    # the CAMYLANET_SWINUMAMBA_CKPT environment variable at the file.
    if ckpt_path is None:
        ckpt_path = os.environ.get("CAMYLANET_SWINUMAMBA_CKPT")
    if not ckpt_path:
        raise ValueError(
            "Pretrained Swin-UMamba checkpoint path not provided. "
            "Pass ckpt_path=... or set the CAMYLANET_SWINUMAMBA_CKPT environment variable."
        )
    print(f"Loading weights from: {ckpt_path}")
    skip_params = ["norm.weight", "norm.bias", "head.weight", "head.bias", 
                   "patch_embed.proj.weight", "patch_embed.proj.bias", 
                   "patch_embed.norm.weight", "patch_embed.norm.weight"]

    ckpt = torch.load(ckpt_path, map_location='cpu')
    print(f"Checkpoint keys: {ckpt.keys()}")  # Inspect checkpoint keys
    model_dict = model.state_dict()

    # Use 'state_dict' here instead of 'model'
    for k, v in ckpt['state_dict'].items():
        if k in skip_params:
            print(f"Skipping weights: {k}")
            continue
        kr = f"vssm_encoder.{k}"

        # Print debug info
        if kr not in model_dict:
            print(f"Key not found in model_dict: {kr}")
            print(f"Model dict keys: {model_dict.keys()}")
        
        if "downsample" in kr:
            i_ds = int(re.findall(r"layers\.(\d+)\.downsample", kr)[0])
            kr = kr.replace(f"layers.{i_ds}.downsample", f"downsamples.{i_ds}")
            assert kr in model_dict.keys()
        
        if kr in model_dict.keys():
            assert v.shape == model_dict[kr].shape, f"Shape mismatch: {v.shape} vs {model_dict[kr].shape}"
            model_dict[kr] = v
        else:
            print(f"Passing weights: {k}")

    model.load_state_dict(model_dict)
    return model






