from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import MirrorTransform

from camylanet.training.data_augmentation.custom_transforms.anatomy_aware_mirror import (
    AnatomyAwareMirrorTransform,
)


class AnatomyAwareMirrorMixin:
    """
    Replaces the standard MirrorTransform inside the regular
    training transform pipeline with AnatomyAwareMirrorTransform.
    """

    @classmethod
    def get_training_transforms(cls, *args, **kwargs):
        transforms = super().get_training_transforms(
            *args,
            **kwargs,
        )

        # Standard nnU-Net returns Compose.
        assert isinstance(transforms, Compose)

        replaced = 0
        new_transforms = []

        for t in transforms.transforms:
            if isinstance(t, MirrorTransform):
                new_transforms.append(
                    AnatomyAwareMirrorTransform(
                        axes=t.axes,
                        lr_axis=2,
                        swap_pairs=((2, 3), (11, 12)),
                        data_key=t.data_key,
                        label_key=t.label_key,
                        p_per_sample=t.p_per_sample,
                    )
                )
                replaced += 1
            else:
                new_transforms.append(t)

        assert replaced == 1, (
            f"Expected exactly one MirrorTransform, found {replaced}"
        )

        return Compose(new_transforms)
