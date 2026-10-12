from camylanet.training.nnUNetTrainer.nnUNetTrainer_500epochs_Seed46 import nnUNetTrainer_500epochs_Seed46


class nnUNetTrainer_500epochs_NoMirrorAxis1_Seed46(nnUNetTrainer_500epochs_Seed46):
    """
    B5 axis-specific mirroring control.

    Same seed/configuration as nnUNetTrainer_500epochs_Seed46.
    Only difference: mirror axis 1 is disabled.
    """

    excluded_mirror_axis = 1

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation, dummy, initial_patch, axes = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )

        assert len(self.configuration_manager.patch_size) == 3

        original_axes = tuple(axes)
        assert self.excluded_mirror_axis in original_axes, (
            self.excluded_mirror_axis,
            original_axes,
        )

        axes = tuple(
            a for a in original_axes
            if a != self.excluded_mirror_axis
        )

        self.inference_allowed_mirroring_axes = axes

        self.print_to_log_file(
            f"B5 Seed46: original mirror axes={original_axes}; "
            f"excluded axis=1; active mirror axes={axes}"
        )

        return rotation, dummy, initial_patch, axes
