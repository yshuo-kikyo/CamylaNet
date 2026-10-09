from camylanet.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_500epochs_NoLRMirror(nnUNetTrainer):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        self.num_epochs = 500
        rotation, dummy, initial_patch, axes = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        assert len(self.configuration_manager.patch_size) == 3
        axes = tuple(axis for axis in axes if axis != 2)
        self.inference_allowed_mirroring_axes = axes
        self.print_to_log_file(
            f"NoLRMirror: training mirror axes={axes}; LR axis=2 excluded"
        )
        return rotation, dummy, initial_patch, axes
