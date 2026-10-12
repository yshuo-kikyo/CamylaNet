from camylanet.training.nnUNetTrainer.nnUNetTrainer_500epochs_Seed45 import (
    nnUNetTrainer_500epochs_Seed45,
)
from camylanet.training.nnUNetTrainer.nnUNetTrainer_AnatomyAwareMirror import (
    AnatomyAwareMirrorMixin,
)


class nnUNetTrainer_500epochs_AnatomyAwareMirror_Seed45(
    AnatomyAwareMirrorMixin,
    nnUNetTrainer_500epochs_Seed45,
):
    """
    B6: standard mirroring retained, but LR flips exchange
    left/right kidney and adrenal labels.
    """
    pass
