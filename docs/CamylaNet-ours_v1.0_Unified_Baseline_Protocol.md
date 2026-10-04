# CamylaNet-ours v1.0 Unified Baseline Protocol

## 1. Purpose

This document freezes the unified baseline protocol for future medical image segmentation experiments based on **CamylaNet-ours v1.0**.

The goal is to ensure that experiments on datasets such as **AMOS, BraTS, BTCV**, and future 3D medical image segmentation tasks use a consistent and reproducible baseline configuration.

The protocol follows the principle:

> **Unify data processing, training budget, augmentation, and evaluation, while retaining necessary architecture-specific optimization settings.**

---

## 2. Locked Baseline Models

| Model | Locked Variant | Optimizer | Initial LR | Scheduler | Epochs | Deep Supervision | Pretraining |
|---|---|---|---:|---|---:|---:|---:|
| **nnU-Net** | nnU-Net v2, 3D fullres | SGD, momentum=0.99, Nesterov, wd=3e-5 | **1e-2** | PolyLR | **100** | **ON** | **No** |
| **SegResNet** | MONAI SegResNet, init_filters=32 | SGD, momentum=0.99, Nesterov, wd=3e-5 | **1e-2** | PolyLR | **100** | OFF | **No** |
| **SwinUNETR** | MONAI SwinUNETR, CamylaNet adaptive feature size | **AdamW** | **5e-4** | PolyLR | **100** | OFF | **No** |
| **UNETR** | CamylaNet UNETR, hidden=768, feature=16 | SGD, momentum=0.99, Nesterov, wd=3e-5 | **1e-2** | PolyLR | **100** | OFF | **No** |
| **nnFormer** | CamylaNet nnFormer, embedding_dim=192 | SGD, momentum=0.99, Nesterov, wd=3e-5 | **1e-2** | PolyLR | **100** | OFF | **No** |
| **MedNeXt** | CamylaNet config: ch=32, k=5, exp=2, blocks=[2]×9 | SGD, momentum=0.99, Nesterov, wd=3e-5 | **1e-2** | PolyLR | **100** | OFF | **No** |
| **U-Mamba** | **U-Mamba Bot** | SGD, momentum=0.99, Nesterov, wd=3e-5 | **1e-2** | PolyLR | **100** | **ON** | **No** |
| **STU-Net** | **STU-Net Base**, dims=[32,64,128,256,512,512] | SGD, momentum=0.99, Nesterov, wd=3e-5 | **1e-2** | PolyLR | **100** | **ON** | **No** |

---

## 3. Unified Settings

The following settings are frozen and should not vary across models within the same dataset.

```text
Dataset split:
    Same split / same folds

Preprocessing:
    Same nnU-Net-style preprocessing

Spacing:
    Same dataset-specific planned spacing

Patch size:
    Same nnU-Net plan within each dataset

Batch size:
    Same planned batch size within each dataset
    unless a model OOMs

Training budget:
    100 epochs
    50 training iterations / epoch
    10 validation iterations / epoch

Augmentation:
    Same nnU-Net augmentation pipeline

Loss:
    Same Dice + Cross Entropy
    Region-based tasks: Dice + BCE

Initialization:
    From scratch

Pretraining:
    Disabled for all eight baselines

Evaluation:
    Same prediction pipeline
    Same post-processing policy
    Same metrics

Cross-validation:
    Same folds for every model

Random seed:
    42
```

---

## 4. Architecture-Specific Settings

The following settings are intentionally allowed to differ because they are part of the architecture-specific design.

### 4.1 Optimizer and Learning Rate

```text
SwinUNETR:
    Optimizer = AdamW
    Initial LR = 5e-4

All other models:
    Optimizer = SGD
    Initial LR = 1e-2
    Momentum = 0.99
    Nesterov = True
    Weight decay = 3e-5
```

The SwinUNETR learning rate must **not** be overwritten by the CamylaNet API default value of `0.01`.

Recommended API behavior:

```python
if initial_lr is not None:
    trainer.initial_lr = initial_lr
```

---

### 4.2 Deep Supervision

| Model | Deep Supervision |
|---|---:|
| nnU-Net | ON |
| SegResNet | OFF |
| SwinUNETR | OFF |
| UNETR | OFF |
| nnFormer | OFF |
| MedNeXt | OFF |
| U-Mamba Bot | ON |
| STU-Net Base | ON |

Deep supervision is treated as an architecture-specific mechanism rather than a globally forced training option.

---

## 5. Locked Model Naming

Use the following names consistently in experiment logs and internal records:

```text
nnU-Net
SegResNet
SwinUNETR
UNETR
nnFormer
MedNeXt-Camyla-k5
U-Mamba Bot
STU-Net Base
```

For papers, `MedNeXt` may be used in tables, but the exact configuration must be stated in the experimental settings.

---

## 6. OOM Handling Policy

If a model runs out of memory, use the following order:

### Priority 1: Gradient Checkpointing

Enable gradient checkpointing when the architecture supports it.

### Priority 2: Reduce Batch Size

Reduce batch size while keeping the planned patch size unchanged.

### Priority 3: Reduce Patch Size

Only reduce patch size if the model still cannot run after reasonable memory-saving measures.

Any deviation in patch size must be:

1. Recorded in the experiment log.
2. Reported in the experimental settings when relevant.
3. Not presented as a fully identical-input comparison without disclosure.

Example of an undesirable hidden mismatch:

```text
nnU-Net:      128×128×128
MedNeXt:      128×128×128
SwinUNETR:     96×96×96
```

---

## 7. Patch Size and Batch Planning

Patch size is determined using the nnU-Net planning procedure:

```text
Dataset
  ↓
Fingerprint extraction
  ↓
Experiment planning
  ↓
Patch size / batch size
  ↓
Shared by all architectures
```

For the same dataset, all architectures should use the same planned patch size whenever technically feasible.

SwinUNETR additionally requires attention to spatial divisibility constraints.

---

## 8. Augmentation

All models use the same nnU-Net-style augmentation pipeline, including, where applicable:

```text
Spatial transforms
Rotation
Scaling
Gaussian noise
Gaussian blur
Brightness augmentation
Contrast augmentation
Low-resolution simulation
Gamma augmentation
Mirroring
Foreground oversampling
```

For standard 3D non-highly-anisotropic cases, the default rotation range is approximately:

```text
x: ±30°
y: ±30°
z: ±30°
```

Typical spatial scaling:

```text
0.85 – 1.25
```

Mirroring axes:

```text
(0, 1, 2)
```

---

## 9. Loss

The main loss family is fixed as:

```text
Multiclass segmentation:
    Dice + Cross Entropy

Region-based / overlapping-label segmentation:
    Dice + BCE
```

For models using deep supervision:

```text
Total loss = Σ wi × Li
```

where each `Li` is the same base Dice+CE or Dice+BCE loss computed at a different supervision scale.

---

## 10. Training Budget

All eight baseline architectures use:

```text
Epochs:
    100

Training iterations per epoch:
    50

Validation iterations per epoch:
    10
```

All experiments should be launched through the same unified CamylaNet-ours API to avoid inconsistencies between direct Trainer invocation and API-level overrides.

---

## 11. Pretraining Policy

All baseline models are trained:

```text
From scratch
```

Default:

```text
pretrained_weights = None
```

No pretrained weights are permitted in the unified baseline table unless a separate experiment is explicitly marked as a pretrained setting.

---

## 12. Cross-Validation and Reproducibility

Default random seed:

```text
seed = 42
```

For datasets such as AMOS and BraTS:

```text
5-fold cross-validation
same fold definitions
same preprocessing
same evaluation protocol
```

Cross-validation consistency takes priority over repeating multiple random seeds for the main comparison.

---

## 13. Fairness Principle

The protocol does **not** enforce the extreme rule that every model must use exactly the same optimizer and every architecture-specific mechanism must be disabled.

Instead, the protocol follows:

### Unified

```text
Dataset
Split
Preprocessing
Spacing
Patch planning
Batch planning
Augmentation
Training budget
Loss family
Pretraining policy
Evaluation
Cross-validation
```

### Architecture-specific

```text
Network topology
Optimizer when required by architecture
Initial LR paired with that optimizer
Deep supervision when intrinsic to architecture
Architecture-specific internal design
```

This avoids artificially weakening individual architectures merely for superficial parameter uniformity.

---

## 14. Known CamylaNet Issues to Fix Before Formal Experiments

### 14.1 SwinUNETR Learning Rate Override

Current CamylaNet API may overwrite the SwinUNETR trainer LR:

```text
Trainer intended LR:
    5e-4

API default LR:
    1e-2
```

This must be fixed before formal comparison experiments.

Locked value:

```text
SwinUNETR:
    AdamW
    LR = 5e-4
```

### 14.2 MedNeXt Variant Naming

The current CamylaNet `MedNeXtTrainer` corresponds to a specific configuration:

```text
channels = 32
kernel size = 5
expansion ratio = 2
blocks = [2,2,2,2,2,2,2,2,2]
deep supervision = OFF
```

Internal name:

```text
MedNeXt-Camyla-k5
```

### 14.3 U-Mamba Variant

Use:

```text
U-Mamba Bot
```

not a generic unspecified U-Mamba label.

### 14.4 STU-Net Variant

Use:

```text
STU-Net Base
```

with:

```text
dims = [32,64,128,256,512,512]
```

---

## 15. Versioning Policy

This document defines:

```text
CamylaNet-ours v1.0
```

Once experiments are launched under v1.0, settings must not be silently modified.

If a protocol change becomes necessary:

```text
v1.0 → v1.1 → v1.2 ...
```

Each revision must record:

```text
What changed
Why it changed
Which experiments are affected
Whether previous results need to be rerun
```

This avoids configuration drift across AMOS, BraTS, and future segmentation studies.

---

## 16. Final Frozen Principle

> **CamylaNet-ours v1.0 standardizes the experimental environment and training budget while preserving necessary architecture-specific optimization mechanisms.**

The framework is intended to provide a stable long-term baseline platform for:

```text
AMOS
BraTS
BTCV
other 3D MRI/CT segmentation datasets
future proposed segmentation methods
```

All future baseline experiments should reference this protocol unless a formally versioned update is introduced.
