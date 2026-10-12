import random

import numpy as np
import torch
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter

from camylanet.training.data_augmentation.custom_transforms.limited_length_multithreaded_augmenter import (
    LimitedLenWrapper,
)
from camylanet.training.nnUNetTrainer.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_500epochs,
)
from camylanet.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class nnUNetTrainer_500epochs_Seed47(nnUNetTrainer_500epochs):
    """
    Formal independent 500-epoch run with explicit stochastic seed configuration.

    Controls:
      - Python RNG
      - NumPy RNG
      - Torch CPU RNG
      - Torch CUDA RNG
      - augmentation worker seeds

    Dataset split is unchanged and remains the repository's fold-4 split.
    """

    experiment_seed = 47

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        plans_identifier: str = "nnUNetPlans",
        device: torch.device = torch.device("cuda"),
    ):
        self._set_experiment_seed()
        super().__init__(
            plans,
            configuration,
            fold,
            dataset_json,
            unpack_dataset,
            plans_identifier,
            device,
        )

        # nnUNetTrainer_500epochs must establish this before optimizer/scheduler init.
        assert self.num_epochs == 500, self.num_epochs

    def _set_experiment_seed(self):
        seed = int(self.experiment_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def initialize(self):
        # Guarantees matched network initialization between paired trainers.
        self._set_experiment_seed()
        return super().initialize()

    def get_dataloaders(self):
        # Reset data/augmentation RNG state at the same logical point in both
        # paired trainers.
        self._set_experiment_seed()

        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            order_resampling_data=3,
            order_resampling_seg=1,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )

        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )

        dl_tr, dl_val = self.get_plain_dataloaders(
            initial_patch_size,
            dim,
        )

        nproc = get_allowed_n_proc_DA()

        if nproc == 0:
            mt_gen_train = SingleThreadedAugmenter(
                dl_tr,
                tr_transforms,
            )
            mt_gen_val = SingleThreadedAugmenter(
                dl_val,
                val_transforms,
            )
        else:
            # Explicit, matched worker seeds for paired experiments.
            train_seeds = [
                self.experiment_seed + i
                for i in range(nproc)
            ]

            nproc_val = max(1, nproc // 2)
            val_seeds = [
                self.experiment_seed + 10000 + i
                for i in range(nproc_val)
            ]

            mt_gen_train = LimitedLenWrapper(
                self.num_iterations_per_epoch,
                data_loader=dl_tr,
                transform=tr_transforms,
                num_processes=nproc,
                num_cached=6,
                seeds=train_seeds,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )

            mt_gen_val = LimitedLenWrapper(
                self.num_val_iterations_per_epoch,
                data_loader=dl_val,
                transform=val_transforms,
                num_processes=nproc_val,
                num_cached=3,
                seeds=val_seeds,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )

        self.print_to_log_file(
            f"Formal paired run seed={self.experiment_seed}; "
            f"augmentation worker seeds explicitly fixed."
        )

        return mt_gen_train, mt_gen_val
