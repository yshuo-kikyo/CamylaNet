# CamylaNet-ours v1 Smoke Test Report

## 1. Purpose

This document records the first end-to-end smoke test of **CamylaNet-ours** on a synthetic 3D medical image segmentation dataset.

The purpose of this smoke test is **not** to evaluate segmentation accuracy. Instead, it verifies that the core training pipeline is operational for several representative 3D segmentation architectures under the current CamylaNet-ours framework.

A model is considered to pass the smoke test only if it successfully completes the following end-to-end pipeline:

```text
dataset discovery
→ preprocessing
→ trainer discovery
→ network construction
→ dataloader initialization
→ forward pass
→ loss computation
→ backward pass
→ optimizer step
→ validation
→ checkpoint saving
→ validation prediction export
→ validation summary generation
```

The smoke test therefore validates both framework-level compatibility and model-specific training integration.

---

## 2. Repository and Branch

Repository:

```text
CamylaNet
```

Development branch:

```text
framework-v1
```

Main development directory:

```text
/data/hdd1/yanshuo/2026code/CamylaNet
```

The smoke test was performed after introducing the following CamylaNet-ours framework changes:

1. A unified baseline protocol document.
2. Separate base and Mamba environments.
3. Explicit dataset path setup.
4. More robust trainer discovery when optional model dependencies are unavailable.
5. Dedicated 2-epoch smoke-test trainers.
6. A synthetic 3D dataset generator.

---

## 3. Test Date

```text
2026-10-05
```

---

## 4. Hardware

Server:

```text
SYS-4029GP-TRTC-ZY001
```

GPU hardware:

```text
4 × NVIDIA GeForce RTX 3090
24 GB VRAM per GPU
```

Driver:

```text
NVIDIA Driver 535.171.04
```

Driver-reported CUDA capability:

```text
CUDA 12.2
```

Important note:

The CUDA version shown by `nvidia-smi` reflects the maximum runtime version supported by the installed NVIDIA driver. It does not necessarily equal the CUDA version bundled with PyTorch.

---

## 5. Data Directories

CamylaNet paths are configured through:

```text
scripts/setup_paths.sh
```

Current paths:

```text
camylanet_raw
/data/hdd1/yanshuo/CamylaNetData/raw

camylanet_preprocessed
/data/hdd1/yanshuo/CamylaNetData/preprocessed

camylanet_results
/data/hdd1/yanshuo/CamylaNetData/results
```

These directories intentionally live outside the Git repository.

This prevents accidental dataset commits, checkpoint commits, repository bloat, and mixing code with experimental outputs.

---

## 6. Synthetic Smoke-Test Dataset

Dataset name:

```text
Dataset999_SmokeTest
```

Dataset ID:

```text
999
```

Generator:

```text
scripts/create_smoketest_dataset.py
```

Raw dataset directory:

```text
/data/hdd1/yanshuo/CamylaNetData/raw/Dataset999_SmokeTest
```

Dataset layout:

```text
Dataset999_SmokeTest/
├── dataset.json
├── imagesTr/
│   ├── smoke_000_0000.nii.gz
│   ├── smoke_001_0000.nii.gz
│   ├── ...
│   └── smoke_009_0000.nii.gz
└── labelsTr/
    ├── smoke_000.nii.gz
    ├── smoke_001.nii.gz
    ├── ...
    └── smoke_009.nii.gz
```

Number of training cases:

```text
10
```

Image shape:

```text
64 × 64 × 64
```

Number of image channels:

```text
1
```

Segmentation labels:

```text
0 = background
1 = synthetic sphere-like foreground
2 = synthetic cuboid foreground
```

The dataset was deliberately designed to be small enough for rapid debugging while still exercising the full 3D segmentation pipeline.

---

## 7. Preprocessing Verification

The dataset was processed with:

```bash
nnUNetv2_plan_and_preprocess   -d 999   --verify_dataset_integrity
```

The following preprocessing outputs were successfully generated:

```text
dataset_fingerprint.json
dataset.json
nnUNetPlans.json
gt_segmentations/
nnUNetPlans_2d/
nnUNetPlans_3d_fullres/
```

Preprocessing status:

```text
PASS
```

---

## 8. Smoke-Test Training Protocol

Smoke tests use dedicated 2-epoch trainers.

This is intentionally separate from the formal baseline protocol.

Formal baseline training budget:

```text
100 epochs
```

Smoke-test training budget:

```text
2 epochs
```

The smoke trainers exist only to validate implementation and runtime compatibility and must never be used for final experimental comparisons.

Smoke-test configuration:

```text
Configuration:
    3d_fullres

Fold:
    0

Epochs:
    2

Pretraining:
    disabled

Dataset:
    Dataset999_SmokeTest
```

---

## 9. Models Tested

The initial smoke-test suite contains five representative 3D architectures:

```text
1. nnU-Net
2. SegResNet
3. SwinUNETR
4. MedNeXt
5. U-Mamba Bot
```

These models cover several major medical segmentation architecture families:

```text
nnU-Net
→ strong self-configuring CNN baseline

SegResNet
→ residual CNN baseline

SwinUNETR
→ Transformer-based baseline

MedNeXt
→ modern convolutional architecture

U-Mamba Bot
→ state-space / Mamba architecture
```

---

## 10. Smoke-Test Result Summary

| Model | Trainer | Environment | Forward/Backward | Validation | Final Checkpoint | Status |
|---|---|---|---:|---:|---:|---:|
| nnU-Net | `nnUNetTrainerSmoke2Epochs` | camylanet-base | PASS | PASS | PASS | **PASS** |
| SegResNet | `SegResNetSmoke2Epochs` | camylanet-base | PASS | PASS | PASS | **PASS** |
| SwinUNETR | `SwinUNETRSmoke2Epochs` | camylanet-base | PASS | PASS | PASS | **PASS** |
| MedNeXt | `MedNeXtSmoke2Epochs` | camylanet-base | PASS | PASS | PASS | **PASS** |
| U-Mamba Bot | `UMambaSmoke2Epochs` | camylanet-mamba | PASS | PASS | PASS | **PASS** |

Overall result:

```text
5 / 5 core 3D baselines passed
```

---

## 11. nnU-Net Smoke Test

Trainer:

```text
nnUNetTrainerSmoke2Epochs
```

Command:

```bash
nnUNetv2_train 999 3d_fullres 0 -tr nnUNetTrainerSmoke2Epochs
```

The following outputs were successfully generated:

```text
checkpoint_best.pth
checkpoint_final.pth
debug.json
progress.png
training_log_*.txt
validation/*.nii.gz
validation/summary.json
plans.json
```

Status:

```text
PASS
```

This confirms that the core CamylaNet training pipeline is operational.

---

## 12. SegResNet Smoke Test

Trainer:

```text
SegResNetSmoke2Epochs
```

Command:

```bash
nnUNetv2_train 999 3d_fullres 0 -tr SegResNetSmoke2Epochs
```

Outputs successfully generated:

```text
checkpoint_best.pth
checkpoint_final.pth
debug.json
progress.png
training_log_*.txt
validation/*.nii.gz
validation/summary.json
plans.json
```

Status:

```text
PASS
```

This confirms that SegResNet is fully integrated into the shared CamylaNet training and validation pipeline.

---

## 13. SwinUNETR Smoke Test

Trainer:

```text
SwinUNETRSmoke2Epochs
```

Command:

```bash
nnUNetv2_train 999 3d_fullres 0 -tr SwinUNETRSmoke2Epochs
```

Actual configuration recorded in `debug.json`:

```text
initial_lr = 0.0005
num_epochs = 2
optimizer = AdamW
weight_decay = 3e-5
eps = 1e-4
```

Observed optimizer configuration:

```text
AdamW
initial_lr: 0.0005
lr: 0.0005
weight_decay: 3e-05
eps: 0.0001
betas: (0.9, 0.999)
```

Status:

```text
PASS
```

Important finding:

The CLI trainer preserves the architecture-specific SwinUNETR learning rate:

```text
5e-4
```

The formal CamylaNet-ours protocol therefore keeps:

```text
SwinUNETR
Optimizer = AdamW
Initial LR = 5e-4
```

---

## 14. MedNeXt Smoke Test

Trainer:

```text
MedNeXtSmoke2Epochs
```

Command:

```bash
nnUNetv2_train 999 3d_fullres 0 -tr MedNeXtSmoke2Epochs
```

Actual configuration recorded in `debug.json`:

```text
initial_lr = 0.01
num_epochs = 2
optimizer = SGD
momentum = 0.99
nesterov = True
weight_decay = 3e-5
```

Observed optimizer:

```text
SGD
lr: 0.01
momentum: 0.99
nesterov: True
weight_decay: 3e-05
```

Status:

```text
PASS
```

The MedNeXt configuration used in CamylaNet-ours remains the CamylaNet-specific variant:

```text
channels = 32
kernel_size = 5
expansion_ratio = 2
blocks = [2,2,2,2,2,2,2,2,2]
deep_supervision = OFF
```

Internal recommended identifier:

```text
MedNeXt-Camyla-k5
```

---

## 15. U-Mamba Bot Smoke Test

Trainer:

```text
UMambaSmoke2Epochs
```

Architecture:

```text
U-Mamba Bot
```

Command:

```bash
nnUNetv2_train 999 3d_fullres 0 -tr UMambaSmoke2Epochs
```

Actual configuration recorded in `debug.json`:

```text
initial_lr = 0.01
num_epochs = 2
optimizer = SGD
momentum = 0.99
nesterov = True
weight_decay = 3e-5
```

Outputs successfully generated:

```text
checkpoint_best.pth
checkpoint_final.pth
debug.json
progress.png
training_log_*.txt
validation/*.nii.gz
validation/summary.json
plans.json
```

Status:

```text
PASS
```

---

## 16. Base Environment

Environment name:

```text
camylanet-base
```

Primary validated models:

```text
nnU-Net
SegResNet
SwinUNETR
UNETR
nnFormer
MedNeXt
STU-Net
```

Key validated packages:

```text
Python 3.10
PyTorch 2.4.1+cu121
MONAI 1.5.0
```

Environment snapshot:

```text
docs/environment_camylanet-base_v1.txt
```

---

## 17. Mamba Environment

Environment location:

```text
/data/hdd1/yanshuo/conda_envs/camylanet-mamba
```

The environment was intentionally placed outside `/home` because the system partition had insufficient free space.

Final validated Mamba stack:

```text
Python          3.10
NumPy           1.26.4
PyTorch         2.1.1+cu118
Torch CUDA      11.8
Transformers    4.38.2
causal-conv1d   1.2.0.post2
mamba-ssm       1.2.0.post1
```

GPU:

```text
NVIDIA GeForce RTX 3090
```

CUDA availability:

```text
True
```

This exact dependency combination successfully completed the U-Mamba Bot end-to-end smoke test.

---

## 18. Mamba Compatibility Investigation

Several compatibility problems were encountered while establishing a stable Mamba environment.

### 18.1 Build isolation issue

Initial installation of `causal-conv1d` failed because the isolated build environment could not import PyTorch:

```text
ModuleNotFoundError: No module named 'torch'
```

Using:

```text
--no-build-isolation
```

allowed the build process to access the installed PyTorch environment.

### 18.2 Binary mismatch

A prebuilt `causal-conv1d 1.1.1` wheel installed successfully but failed during import with an undefined-symbol error.

This demonstrated that:

```text
successful pip installation
≠
valid CUDA extension compatibility
```

### 18.3 Stable PyTorch baseline

The Mamba environment was rebuilt around:

```text
torch 2.1.1+cu118
```

instead of:

```text
torch 2.2.2+cu121
```

### 18.4 NumPy compatibility

`numpy 2.2.6` caused compatibility warnings with PyTorch 2.1.1 and compiled extensions.

The environment was pinned to:

```text
numpy 1.26.4
```

### 18.5 Transformers compatibility

`transformers 5.18.0` was incompatible with `mamba-ssm 1.2.0.post1` and caused:

```text
ImportError:
cannot import name 'GreedySearchDecoderOnlyOutput'
from transformers.generation
```

The environment was pinned to:

```text
transformers 4.38.2
```

### 18.6 causal-conv1d API mismatch

The combination:

```text
causal-conv1d 1.1.1
mamba-ssm 1.2.0.post1
```

produced an interface mismatch during actual U-Mamba execution.

The environment was cleaned completely and rebuilt with:

```text
causal-conv1d 1.2.0.post2
```

This version successfully passed a real CUDA forward test and the subsequent U-Mamba smoke test.

---

## 19. CUDA Extension Verification

Before rerunning U-Mamba, `causal-conv1d` was tested independently.

Test:

```python
import torch
from causal_conv1d import causal_conv1d_fn

x = torch.randn(2, 8, 32, device="cuda")
w = torch.randn(8, 4, device="cuda")
b = torch.randn(8, device="cuda")

y = causal_conv1d_fn(
    x,
    w,
    b,
    activation="silu"
)

print(y.shape)
```

Observed output:

```text
torch.Size([2, 8, 32])
```

Status:

```text
PASS
```

---

## 20. Framework Issue Discovered: Trainer Discovery

An important CamylaNet framework issue was discovered during the smoke test.

Original behavior:

```text
recursive_find_python_class()
```

attempted to import every module inside:

```text
camylanet/training/nnUNetTrainer/
```

When using the base environment, trainer discovery failed because `SwinUMambaTrainer.py` required `mamba_ssm`, even when the requested trainer was a non-Mamba model.

Example failure:

```text
ModuleNotFoundError: No module named 'mamba_ssm'
```

CamylaNet-ours modified trainer discovery so that trainer modules whose optional dependencies are unavailable are skipped instead of blocking unrelated trainers.

This allows:

```text
camylanet-base
```

to run non-Mamba models without installing Mamba.

This is an important framework-level improvement.

---

## 21. Smoke-Test Trainers

The following dedicated smoke trainers are retained for future regression testing:

```text
nnUNetTrainerSmoke2Epochs.py
SegResNetSmoke2Epochs.py
SwinUNETRSmoke2Epochs.py
MedNeXtSmoke2Epochs.py
UMambaSmoke2Epochs.py
```

Location:

```text
camylanet/training/nnUNetTrainer/
```

These trainers differ from the formal baseline trainers only by reducing:

```text
num_epochs = 2
```

They are intended for:

```text
installation validation
dependency validation
CUDA compatibility testing
framework regression testing
new-server deployment checks
major dependency upgrade checks
```

They are not intended for publication experiments.

---

## 22. Pass Criteria

A smoke test is considered **PASS** only when all of the following are true:

```text
[1] trainer can be discovered
[2] trainer initializes successfully
[3] network builds successfully
[4] dataloader starts successfully
[5] GPU forward succeeds
[6] loss computation succeeds
[7] backward succeeds
[8] optimizer step succeeds
[9] validation runs
[10] validation predictions are exported
[11] checkpoint_best.pth is generated
[12] checkpoint_final.pth is generated
[13] validation/summary.json is generated
```

An import-only test is not sufficient.

---

## 23. Result Files Used as Evidence

For each passing trainer, the following output artifacts were used as evidence:

```text
checkpoint_best.pth
checkpoint_final.pth
debug.json
progress.png
training_log_*.txt
validation/*.nii.gz
validation/summary.json
plans.json
```

Example U-Mamba result directory:

```text
/data/hdd1/yanshuo/CamylaNetData/results/
Dataset999_SmokeTest/
UMambaSmoke2Epochs__nnUNetPlans__3d_fullres/
fold_0/
```

---

## 24. Known Warnings

The following warning appeared during training:

```text
FutureWarning:
torch.cuda.amp.GradScaler(args...) is deprecated.
Please use torch.amp.GradScaler('cuda', args...) instead.
```

This does not currently affect training correctness.

It should be addressed in a later framework cleanup release.

---

## 25. Reproducibility Rules

The following rules should be followed for future smoke tests.

### Rule 1

Do not modify `Dataset999_SmokeTest` silently.

If the synthetic dataset changes, create a new smoke-test dataset version or document the change.

### Rule 2

Do not use smoke trainers for formal benchmark experiments.

### Rule 3

Do not interpret smoke-test segmentation metrics as scientific benchmark results.

### Rule 4

After changing any of the following, rerun the relevant smoke tests:

```text
PyTorch
CUDA wheel
MONAI
NumPy
Transformers
mamba-ssm
causal-conv1d
trainer discovery
base Trainer
network implementation
preprocessing pipeline
augmentation pipeline
loss implementation
```

### Rule 5

Any change to Mamba CUDA dependencies requires rerunning:

```text
causal_conv1d CUDA forward test
+
U-Mamba 2-epoch smoke test
```

---

## 26. Recommended Regression Suite

Before a formal release of CamylaNet-ours, run:

```text
nnU-Net
SegResNet
SwinUNETR
MedNeXt
U-Mamba Bot
```

If all five pass, the current core framework can be considered operational.

As more models are integrated, the smoke-test suite should eventually expand to include:

```text
UNETR
nnFormer
STU-Net
3D UX-Net
SegMamba
nnMamba
MetaUNETR
EfficientMedNeXt
```

---

## 27. Current Framework Readiness

After this smoke test:

```text
Core preprocessing pipeline: PASS
Trainer discovery: PASS after CamylaNet-ours fix
Base CNN training: PASS
Transformer training: PASS
Modern CNN training: PASS
Mamba training: PASS
Validation pipeline: PASS
Checkpoint pipeline: PASS
```

Current status:

> **CamylaNet-ours v1 is ready for the next stage: real-dataset baseline validation.**

---

## 28. Recommended Next Stage

Recommended progression:

```text
Stage 1
Synthetic smoke test
DONE

Stage 2
Single real dataset
single fold
short validation run

Stage 3
Single real dataset
formal 100-epoch protocol

Stage 4
5-fold cross-validation

Stage 5
multi-model baseline table

Stage 6
integration of new research methods
```

Recommended first real dataset candidates:

```text
AMOS
BraTS
BTCV
```

---

## 29. Final Conclusion

The first CamylaNet-ours v1 end-to-end smoke-test suite was completed successfully.

All five core 3D baselines:

```text
nnU-Net
SegResNet
SwinUNETR
MedNeXt
U-Mamba Bot
```

successfully completed real training and validation on `Dataset999_SmokeTest`.

The test additionally identified and resolved several important framework and dependency issues, including:

```text
optional trainer dependency blocking
Mamba CUDA extension compatibility
NumPy ABI compatibility
Transformers API compatibility
causal-conv1d / mamba-ssm interface compatibility
system-disk environment placement
```

The validated Mamba configuration is:

```text
Python          3.10
NumPy           1.26.4
PyTorch         2.1.1+cu118
Transformers    4.38.2
causal-conv1d   1.2.0.post2
mamba-ssm       1.2.0.post1
```

Final result:

```text
CORE 3D SMOKE TEST
5 / 5 PASS
```

This establishes a stable foundation for subsequent AMOS, BraTS, BTCV, and future 3D medical image segmentation experiments.
