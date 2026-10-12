from camylanet.training.nnUNetTrainer.nnUNetTrainer_500epochs_Seed47 import nnUNetTrainer_500epochs_Seed47


class nnUNetTrainer_500epochs_NoMirrorAxis0_Seed47(nnUNetTrainer_500epochs_Seed47):
    excluded_mirror_axis = 0

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation, dummy, initial_patch, axes = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )

        assert len(self.configuration_manager.patch_size) == 3

        original_axes = tuple(axes)
        assert self.excluded_mirror_axis in original_axes

        axes = tuple(
            a for a in original_axes
            if a != self.excluded_mirror_axis
        )

        self.inference_allowed_mirroring_axes = axes

        self.print_to_log_file(
            f"B5 Seed47: original={original_axes}, "
            f"excluded=0, active={axes}"
        )

        return rotation, dummy, initial_patch, axes
