from pathlib import Path
import json
import numpy as np
import nibabel as nib

RAW_ROOT = Path("/data/hdd1/yanshuo/CamylaNetData/raw")
DATASET = RAW_ROOT / "Dataset999_SmokeTest"

imagesTr = DATASET / "imagesTr"
labelsTr = DATASET / "labelsTr"

imagesTr.mkdir(parents=True, exist_ok=True)
labelsTr.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

shape = (64, 64, 64)
affine = np.eye(4, dtype=np.float32)

# 10 cases: enough for a 5-fold smoke-test split
for i in range(10):
    case_id = f"smoke_{i:03d}"

    image = rng.normal(0, 0.08, size=shape).astype(np.float32)
    label = np.zeros(shape, dtype=np.uint8)

    # foreground class 1: sphere-like object
    cx = 22 + (i % 4)
    cy = 28 + (i % 3)
    cz = 30 + (i % 5)

    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    sphere = (
        (x - cx) ** 2
        + (y - cy) ** 2
        + (z - cz) ** 2
        <= 10 ** 2
    )

    label[sphere] = 1
    image[sphere] += 1.0

    # foreground class 2: smaller cuboid
    x0 = 38 + (i % 3)
    y0 = 15 + (i % 4)
    z0 = 20 + (i % 3)

    label[x0:x0+10, y0:y0+12, z0:z0+8] = 2
    image[x0:x0+10, y0:y0+12, z0:z0+8] += 1.5

    image_path = imagesTr / f"{case_id}_0000.nii.gz"
    label_path = labelsTr / f"{case_id}.nii.gz"

    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(label, affine), label_path)

dataset_json = {
    "channel_names": {
        "0": "synthetic"
    },
    "labels": {
        "background": 0,
        "object_1": 1,
        "object_2": 2
    },
    "numTraining": 10,
    "file_ending": ".nii.gz"
}

with open(DATASET / "dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset_json, f, indent=4)

print(f"Created: {DATASET}")
print("Training cases: 10")
print("Shape:", shape)
