import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform


class AnatomyAwareMirrorTransform(AbstractTransform):
    """
    Mirror transform with semantic label exchange for LR flips.

    For axis 0/1:
        identical behavior to the standard MirrorTransform.

    For axis 2:
        spatially flips image and segmentation, then swaps
        left/right organ identities:

            right kidney (2) <-> left kidney (3)
            right adrenal (11) <-> left adrenal (12)

    This preserves anatomical side identity after an LR reflection.
    """

    def __init__(
        self,
        axes=(0, 1, 2),
        lr_axis=2,
        swap_pairs=((2, 3), (11, 12)),
        data_key="data",
        label_key="seg",
        p_per_sample=1,
    ):
        self.axes = tuple(axes)
        self.lr_axis = int(lr_axis)
        self.swap_pairs = tuple(tuple(x) for x in swap_pairs)
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample

        if max(self.axes) > 2:
            raise ValueError(
                "axes must use spatial indices 0, 1, 2"
            )

    @staticmethod
    def _swap_labels(seg, pairs):
        if seg is None:
            return seg

        # Must use masks from the pre-swap segmentation.
        original = seg.copy()

        for a, b in pairs:
            seg[original == a] = b
            seg[original == b] = a

        return seg

    def __call__(self, **data_dict):
        data = data_dict.get(self.data_key)
        seg = data_dict.get(self.label_key)

        for b in range(len(data)):
            if np.random.uniform() >= self.p_per_sample:
                continue

            sample_data = data[b]
            sample_seg = None if seg is None else seg[b]

            # axis 0
            if 0 in self.axes and np.random.uniform() < 0.5:
                sample_data[:, :] = sample_data[:, ::-1]

                if sample_seg is not None:
                    sample_seg[:, :] = sample_seg[:, ::-1]

            # axis 1
            if 1 in self.axes and np.random.uniform() < 0.5:
                sample_data[:, :, :] = sample_data[:, :, ::-1]

                if sample_seg is not None:
                    sample_seg[:, :, :] = sample_seg[:, :, ::-1]

            # axis 2 = LR axis
            if (
                2 in self.axes
                and len(sample_data.shape) == 4
                and np.random.uniform() < 0.5
            ):
                sample_data[:, :, :, :] = (
                    sample_data[:, :, :, ::-1]
                )

                if sample_seg is not None:
                    sample_seg[:, :, :, :] = (
                        sample_seg[:, :, :, ::-1]
                    )

                    if self.lr_axis == 2:
                        sample_seg = self._swap_labels(
                            sample_seg,
                            self.swap_pairs,
                        )

            data[b] = sample_data

            if seg is not None:
                seg[b] = sample_seg

        data_dict[self.data_key] = data

        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict
