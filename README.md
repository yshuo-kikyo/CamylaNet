# CamylaNet

Camylanet is a wrapper framework built on [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) for medical image segmentation. It provides a simplified Python API for data preprocessing, model training, and evaluation, and ships with a curated set of convolutional, Transformer, and state-space backbones ready to use as drop-in trainers.

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/yifangao112/camylanet.git
cd camylanet

# 2. Install (Python >= 3.9)
pip install -e .

# 3. Set Camylanet data-path environment variables.
#    (These mirror nnU-Net v2's layout but use a `camylanet_*` prefix so
#    Camylanet and nnU-Net can coexist in the same environment.)
export camylanet_raw="/path/to/camylanet_raw"
export camylanet_preprocessed="/path/to/camylanet_preprocessed"
export camylanet_results="/path/to/camylanet_results"
```

## Built-in architectures

All architectures below are available as ready-to-use trainers under
`camylanet.training.nnUNetTrainer`:

| Architecture | 2D | 3D | Family        | Trainer class              |
|--------------|----|----|---------------|----------------------------|
| nnU-Net      | ✓  | ✓  | Convolutional | `nnUNetTrainer`            |
| SwinUNETR    | ✓  | ✓  | Transformer   | `SwinUNETRTrainer`         |
| SegResNet    | ✓  | ✓  | Convolutional | `SegResNetTrainer`         |
| U-Net++      | ✓  | ✓  | Convolutional | `UNetPlusPlusTrainer`      |
| U-Mamba      | ✓  | ✓  | State-space   | `UMambaTrainer`            |
| MedNeXt      |    | ✓  | Convolutional | `MedNeXtTrainer`           |
| 3D UX-Net    |    | ✓  | Transformer   | `ThreeDUXNetTrainer`       |
| UNETR        |    | ✓  | Transformer   | `UNETRTrainer`             |
| nnFormer     |    | ✓  | Transformer   | `nnFormerTrainer`          |
| STU-Net      |    | ✓  | Convolutional | `STUNetTrainer`            |
| TransUNet    | ✓  |    | Transformer   | `TransUNetTrainer`         |
| UTNet        | ✓  |    | Transformer   | `UTNetTrainer`             |
| SwinUMamba   | ✓  |    | State-space   | `SwinUMambaTrainer`        |
| U-KAN        | ✓  |    | Convolutional | `UKANTrainer`              |

All shipped architectures preserve their original authors' citations;
see [NOTICE](NOTICE) for the full attribution list.

## Basic Usage

> **💡 Tip**: On first use, run `training_network_1epoch()` for a quick smoke test (trains only 1 epoch) to verify the data and model configuration. See [Quick Testing Example](#quick-testing-with-1-epoch-recommended-for-initial-testing).

```python
import camylanet

# Dataset ID (following nnUNet dataset format)
dataset_id = xxx
configuration = '2d'

# Step 1: Data preprocessing
plans_identifier = camylanet.plan_and_preprocess(
    dataset_id=dataset_id,
    configurations=[configuration]  # Options: ['2d', '3d_fullres']
)

# Step 2: Model training
result_folder, training_log = camylanet.training_network(
    dataset_id=dataset_id,
    configuration=configuration,
    plans_identifier=plans_identifier
)

# Step 3: Result evaluation
results = camylanet.evaluate(
    dataset_id=dataset_id,
    result_folder=result_folder,
)

# View training log (contains only loss values, no Dice/HD95 per epoch)
print(f"Training Log:")
print(f"Number of epochs: {len(training_log['epochs'])}")
if training_log['train_losses']:
    print(f"Final training loss: {training_log['train_losses'][-1]:.4f}")
    print(f"Final validation loss: {training_log['val_losses'][-1]:.4f}")

# View evaluation results (for Dice/HD95 scores)
print(f"Evaluation Results:")
print(f"Mean Dice Score: {results['foreground_mean']['Dice']:.4f}")
print(f"Mean IoU Score: {results['foreground_mean']['IoU']:.4f}")
```

## Organizing Experiments with exp_name

You can organize your experiments using the `exp_name` parameter to create a separate folder for each experiment:

```python
# Run experiment with a custom name
result_folder, training_log = camylanet.training_network(
    dataset_id=dataset_id,
    configuration=configuration,
    plans_identifier=plans_identifier,
    exp_name="experiment_v1"  # Creates a separate folder for this experiment
)

# Evaluate with matching exp_name
results = camylanet.evaluate(
    dataset_id=dataset_id,
    result_folder=result_folder,
    exp_name="experiment_v1"  # Must match the exp_name used in training
)
```

**Result folder structure with exp_name:**
```
$camylanet_results/
└── Datasetxxx_xxxxx/
    └── experiment_v1/
        └── nnUNetTrainer__nnUNetPlans__2d/
            └── fold_0/
                ├── checkpoint_best.pth
                ├── checkpoint_final.pth
                ├── checkpoint_latest.pth
                └── validation/
```

**Without exp_name (default):**
```
$camylanet_results/
└── Datasetxxx_xxxxx/
    └── nnUNetTrainer__nnUNetPlans__2d/
        └── fold_0/
            ├── checkpoint_best.pth
            ├── checkpoint_final.pth
            └── validation/
```

## Network Configuration

The `plan_and_preprocess` function performs data preprocessing and creates plans files. Network configuration details are stored in the plans files and can be accessed through the nnUNet configuration management system:

```python
plans_identifier = camylanet.plan_and_preprocess(
    dataset_id=dataset_id,
    configurations=['2d', '3d_fullres']
)

```
## Custom Network Architecture

> **🚨 CRITICAL REQUIREMENT**: All custom trainers MUST inherit from `nnUNetTrainerNoDeepSupervision`.
> **DO NOT use the default `nnUNetTrainer` as base class.**
>
> **Correct import:**
> ```python
> from camylanet.training.nnUNetTrainer.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
> ```

### 1. Create Custom Network Class

```python
import torch
from torch import nn
from typing import List

class PlainConvUNet(nn.Module):
    """Simple PlainConvUNet implementation"""
    def __init__(self, input_channels: int, num_classes: int,
                 features: List[int] = [32, 64, 128, 256], is_3d: bool = False):
        super().__init__()
        self.features = features
        self.is_3d = is_3d

        # Choose convolution operations based on 2D/3D
        if is_3d:
            self.conv_op = nn.Conv3d
            self.norm_op = nn.InstanceNorm3d
            self.pool_op = nn.MaxPool3d
            self.upsample_op = nn.ConvTranspose3d
        else:
            self.conv_op = nn.Conv2d
            self.norm_op = nn.InstanceNorm2d
            self.pool_op = nn.MaxPool2d
            self.upsample_op = nn.ConvTranspose2d

        # Build encoder, bottleneck, and decoder
        self.encoder = nn.ModuleList()
        self.pools = nn.ModuleList()
        # ... (encoder implementation)

        self.bottleneck = self._make_conv_block(features[-1], features[-1] * 2)

        self.decoder = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        # ... (decoder implementation)

        self.final_conv = self.conv_op(features[0], num_classes, 1)

    def _make_conv_block(self, in_channels: int, out_channels: int):
        """Create convolution block with normalization and activation"""
        return nn.Sequential(
            self.conv_op(in_channels, out_channels, 3, padding=1),
            self.norm_op(out_channels),
            nn.LeakyReLU(inplace=True),
            self.conv_op(out_channels, out_channels, 3, padding=1),
            self.norm_op(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder with skip connections
        skip_connections = []
        for encoder, pool in zip(self.encoder, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        # ... (decoder forward pass)

        return self.final_conv(x)
```

### 2. Create Custom Trainer

> **🚨 CRITICAL REQUIREMENTS:**
> 1. **MUST inherit from `nnUNetTrainerNoDeepSupervision`** (NOT `nnUNetTrainer`)
> 2. **MUST use correct import:** `from camylanet.training.nnUNetTrainer.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision`
> 3. **Only override `build_network_architecture` method** - DO NOT override `__init__`
>
> **❌ WRONG (will cause errors):**
> ```python
> from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
> class MyTrainer(nnUNetTrainer):  # ❌ WRONG BASE CLASS
>     ...
> ```
>
> **✅ CORRECT:**
> ```python
> from camylanet.training.nnUNetTrainer.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
> class MyTrainer(nnUNetTrainerNoDeepSupervision):  # ✅ CORRECT BASE CLASS
>     ...
> ```

```python
# 🚨 CRITICAL: Must inherit from nnUNetTrainerNoDeepSupervision (NOT nnUNetTrainer)
from camylanet.training.nnUNetTrainer.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from typing import Union, List, Tuple
import torch.nn as nn

# ✅ CORRECT: Inheriting from nnUNetTrainerNoDeepSupervision
class PlainConvUNetTrainer(nnUNetTrainerNoDeepSupervision):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                  arch_init_kwargs: dict,
                                  arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                  num_input_channels: int,
                                  num_output_channels: int,
                                  enable_deep_supervision: bool = True) -> nn.Module:

        # Detect 2D/3D from architecture parameters
        is_3d = 'Conv3d' in str(arch_init_kwargs.get('conv_op', ''))

        network = PlainConvUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            features=[32, 64, 128, 256],
            is_3d=is_3d
        )
        return network
```



### 3. Use Custom Trainer

```python
# Preprocess data
plans_identifier = camylanet.plan_and_preprocess(
    dataset_id=dataset_id,
    configurations=[configuration]
)

# Train with custom network
result_folder, training_log = camylanet.training_network(
    dataset_id=dataset_id,
    configuration=configuration,
    trainer_class=PlainConvUNetTrainer,
    plans_identifier=plans_identifier
)
```



## Quick Testing Workflow (Recommended)

Before launching a full training run, it is strongly recommended to use `training_network_1epoch()` for a quick test:

```python
# 1. Preprocess the data
plans_identifier = camylanet.plan_and_preprocess(
    dataset_id=dataset_id,
    configurations=['2d']
)

# 2. Quick test (1 epoch only) - verify that everything works
result_folder, training_log = camylanet.training_network_1epoch(
    dataset_id=dataset_id,
    configuration='2d',
    plans_identifier=plans_identifier,
    exp_name='quick_test'
)

# 3. Inspect the test result
if training_log['train_losses']:
    print(f"✅ Test passed! Train loss: {training_log['train_losses'][0]:.4f}")
    print(f"Val loss: {training_log['val_losses'][0]:.4f}")

    # 4. If the test passes, run full training
    result_folder, training_log = camylanet.training_network(
        dataset_id=dataset_id,
        configuration='2d',
        plans_identifier=plans_identifier,
        num_epochs=100,  # Full training
        exp_name='full_training'
    )
```

**Advantages of quick testing:**
- ⚡ **Saves time**: 1 epoch usually finishes in minutes, not hours
- 🐛 **Catch issues early**: surface data or configuration problems before long runs
- 💾 **Saves resources**: avoid wasting GPU time on broken configurations
- ✅ **End-to-end validation**: exercises the full pipeline (data loading, forward, backward, validation)

## Training Log

Training returns a log with loss values:

```python
training_log = {
    'epochs': [0, 1, 2, ...],
    'train_losses': [1.2, 1.1, 0.9, ...],
    'val_losses': [1.3, 1.2, 1.0, ...]
}
```



## API Reference

### `plan_and_preprocess`
```python
plans_identifier = camylanet.plan_and_preprocess(
    dataset_id: Union[int, List[int]],
    preprocessor_name: str = 'DefaultPreprocessor',
    configurations: List[str] = ['2d', '3d_fullres']
)
```

**Returns:** `str` - The plans identifier for the created plans file.

### `training_network`
```python
result_folder, training_log = camylanet.training_network(
    dataset_id: Union[int, str],
    configuration: str,
    trainer_class: Union[Type[nnUNetTrainer], str] = 'nnUNetTrainer',
    plans_identifier: str = 'nnUNetPlans',
    exp_name: Optional[str] = None,  # Experiment name for organizing results
    num_epochs: int = 100  # Number of training epochs
)
```

**Returns:** `Tuple[str, dict]` - (output_folder, training_log) containing the path to training results and training metrics.

### `training_network_1epoch`

**Quick-test helper** - automatically sets the number of epochs to 1, for fast verification of the model and data pipeline.

```python
result_folder, training_log = camylanet.training_network_1epoch(
    dataset_id: Union[int, str],
    configuration: str,
    trainer_class: Union[Type[nnUNetTrainer], str] = 'nnUNetTrainer',
    plans_identifier: str = 'nnUNetPlans',
    exp_name: Optional[str] = None  # Experiment name for organizing results
)
```

**Use cases:**
- ✅ Verify that data preprocessing is correct
- ✅ Test that a custom network architecture works
- ✅ Quickly check for errors in the training pipeline
- ✅ Validate that the GPU / memory configuration is reasonable

**Returns:** `Tuple[str, dict]` - (output_folder, training_log) containing the path to training results and training metrics from 1 epoch.

### `evaluate`
```python
results = camylanet.evaluate(
    dataset_id: Union[int, str],
    result_folder: str,
    output_file: Optional[str] = None,
    exp_name: Optional[str] = None  # Must match exp_name used in training
)
```

**Returns:** `dict` - Evaluation metrics including Dice coefficients, HD95 distances, and other segmentation metrics.

## Examples

### Quick testing with 1 epoch (recommended for initial testing)
```python
# Quick test to verify data pipeline and model setup
plans_identifier = camylanet.plan_and_preprocess(
    dataset_id=4, configurations=['2d']
)

# Run 1 epoch for quick testing
result_folder, training_log = camylanet.training_network_1epoch(
    dataset_id=4,
    configuration='2d',
    plans_identifier=plans_identifier,
    exp_name='quick_test'
)

# Check if training ran successfully
print(f"Training loss: {training_log['train_losses'][0]:.4f}")
print(f"Validation loss: {training_log['val_losses'][0]:.4f}")
```

### Basic usage with default network
```python
# Standard workflow with 3d_fullres configuration
plans_identifier = camylanet.plan_and_preprocess(
    dataset_id=4, configurations=['3d_fullres']
)
result_folder, training_log = camylanet.training_network(
    dataset_id=4,
    configuration='3d_fullres',
    plans_identifier=plans_identifier
)
results = camylanet.evaluate(dataset_id=4, result_folder=result_folder)
```

### Custom network implementation
```python
# Custom PlainConvUNet implementation
class PlainConvUNet(nn.Module):
    def __init__(self, input_channels: int, num_classes: int,
                 features: List[int] = [32, 64, 128, 256], is_3d: bool = False):
        # Simple U-Net with plain convolutions
        # Encoder-decoder architecture with skip connections

# Custom trainer - 🚨 MUST inherit from nnUNetTrainerNoDeepSupervision
class PlainConvUNetTrainer(nnUNetTrainerNoDeepSupervision):  # ✅ CORRECT BASE CLASS
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        # Detect 2D/3D from architecture parameters
        is_3d = 'Conv3d' in str(arch_init_kwargs.get('conv_op', ''))

        return PlainConvUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            features=[32, 64, 128, 256],
            is_3d=is_3d
        )

# Preprocess data
plans_identifier = camylanet.plan_and_preprocess(
    dataset_id=4, configurations=['2d']
)

# 💡 Recommended: First test with 1 epoch to verify custom network works
result_folder, training_log = camylanet.training_network_1epoch(
    dataset_id=4,
    configuration='2d',
    trainer_class=PlainConvUNetTrainer,
    plans_identifier=plans_identifier,
    exp_name='custom_net_test'
)

# If test passes, run full training
result_folder, training_log = camylanet.training_network(
    dataset_id=4,
    configuration='2d',
    trainer_class=PlainConvUNetTrainer,
    plans_identifier=plans_identifier,
    exp_name='custom_net_full'
)
```

## Configuration Types

- `2d`: 2D U-Net for 2D images or slice-by-slice processing
- `3d_fullres`: 3D U-Net for full resolution 3D processing

## Evaluation Metrics

- **Dice Coefficient**: Segmentation overlap [0,1], higher is better
- **HD95**: 95th percentile Hausdorff distance (mm), lower is better

## License

Released under the **Apache License, Version 2.0** — see [LICENSE](LICENSE).
Third-party upstream projects and bundled architecture implementations are
attributed in [NOTICE](NOTICE).

## Related repositories

- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) — upstream framework Camylanet builds on.
- [nnprep](https://github.com/yifangao112/nnprep) — companion agent that converts arbitrary medical segmentation datasets into the nnU-Net v2 format consumed by Camylanet.