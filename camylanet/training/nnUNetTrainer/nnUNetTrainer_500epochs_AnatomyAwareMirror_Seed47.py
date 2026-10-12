from camylanet.training.nnUNetTrainer.nnUNetTrainer_500epochs_Seed47 import (
    nnUNetTrainer_500epochs_Seed47,
)
from camylanet.training.nnUNetTrainer.nnUNetTrainer_AnatomyAwareMirror import (
    AnatomyAwareMirrorMixin,
)


class nnUNetTrainer_500epochs_AnatomyAwareMirror_Seed47(
    AnatomyAwareMirrorMixin,
    nnUNetTrainer_500epochs_Seed47,
):
    """
    B6: standard mirroring retained, but LR flips exchange
    left/right kidney and adrenal labels.
    """
    pass
