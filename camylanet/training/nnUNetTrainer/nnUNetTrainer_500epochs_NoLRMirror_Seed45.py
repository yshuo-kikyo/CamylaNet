from camylanet.training.nnUNetTrainer.nnUNetTrainer_500epochs_Seed45 import (
    nnUNetTrainer_500epochs_Seed45,
)


class nnUNetTrainer_500epochs_NoLRMirror_Seed45(
    nnUNetTrainer_500epochs_Seed45
):
    """
    Paired Seed45 experiment.

    Identical to nnUNetTrainer_500epochs_Seed45 except that
    LR mirroring (axis 2 under the current AMOS22 configuration)
    is disabled for the full 0-500 epoch training trajectory.
    """

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation, dummy, initial_patch, axes = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )

        assert len(self.configuration_manager.patch_size) == 3

        original_axes = tuple(axes)
        axes = tuple(
            axis
            for axis in original_axes
            if axis != 2
        )

        self.inference_allowed_mirroring_axes = axes

        self.print_to_log_file(
            "Full NoLRMirror Seed45: "
            f"original mirror axes={original_axes}; "
            f"training mirror axes={axes}; "
            "LR axis=2 excluded for epochs 0-500."
        )

        return rotation, dummy, initial_patch, axes
