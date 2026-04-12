from typing import Union, List, Optional, Type, Tuple, Dict
import multiprocessing
import os

import torch
from camylanet.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess
from camylanet.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from camylanet.run.run_training import run_training, get_trainer_from_args
from camylanet.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from camylanet.evaluation.evaluate_predictions import compute_metrics_on_folder2
from camylanet.paths import camylanet_preprocessed
from camylanet.utilities.plans_handling.plans_handler import PlansManager
from batchgenerators.utilities.file_and_folder_operations import *


def _extract_training_log(logger):
    """
    Extract the training log from the trainer's logger.

    Args:
        logger: nnUNetLogger instance.

    Returns:
        dict: Dictionary containing the training log.
    """
    if logger is None or not hasattr(logger, 'my_fantastic_logging'):
        return {
            'epochs': [],
            'train_losses': [],
            'val_losses': []
        }

    log_data = logger.my_fantastic_logging
    num_epochs = len(log_data.get('train_losses', []))

    return {
        'epochs': list(range(num_epochs)),
        'train_losses': log_data.get('train_losses', []),
        'val_losses': log_data.get('val_losses', [])
    }



def _load_training_log_from_folder(output_folder):
    """
    Load the training log from an output folder.

    Args:
        output_folder: Path to the training output folder.

    Returns:
        dict: Dictionary containing the training log.
    """
    # Try to load logger info from a checkpoint
    checkpoint_files = ['checkpoint_final.pth', 'checkpoint_best.pth', 'checkpoint_latest.pth']

    for checkpoint_file in checkpoint_files:
        checkpoint_path = join(output_folder, checkpoint_file)
        if isfile(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'logger' in checkpoint:
                    logger_data = checkpoint['logger']
                    num_epochs = len(logger_data.get('train_losses', []))
                    return {
                        'epochs': list(range(num_epochs)),
                        'train_losses': logger_data.get('train_losses', []),
                        'val_losses': logger_data.get('val_losses', [])
                    }
            except Exception as e:
                print(f"Warning: Could not load training log from {checkpoint_file}: {e}")
                continue

    # If loading from a checkpoint fails, return an empty log
    print("Warning: Could not load training log from any checkpoint file")
    return {
        'epochs': [],
        'train_losses': [],
        'val_losses': []
    }


def _run_extract_fingerprints(dataset_id, fingerprint_extractor_class, num_processes,
                             verify_dataset_integrity, clean, verbose):
    """
    Run extract_fingerprints in a dedicated function so it can be executed under the if __name__ == '__main__' guard.
    """
    # Convert single dataset_id to list if necessary
    if isinstance(dataset_id, int):
        dataset_id = [dataset_id]
    
    # Extract fingerprints
    print("Running fingerprint extraction...")
    extract_fingerprints(dataset_id, fingerprint_extractor_class, num_processes, 
                        verify_dataset_integrity, clean, verbose)


def _run_plan_experiments(dataset_id, experiment_planner_class, gpu_memory_target, 
                         preprocessor_name, overwrite_target_spacing, overwrite_plans_name, force_target_shape, max_batch_size, force_n_stages):
    """
    Run plan_experiments in a dedicated function so it can be executed under the if __name__ == '__main__' guard.
    """
    # Convert single dataset_id to list if necessary
    if isinstance(dataset_id, int):
        dataset_id = [dataset_id]
    
    # Plan experiments
    print("Running experiment planning...")
    return plan_experiments(dataset_id, experiment_planner_class, 
                           gpu_memory_target, preprocessor_name,
                           overwrite_target_spacing, overwrite_plans_name, force_target_shape, max_batch_size, force_n_stages)


def _run_preprocess(dataset_id, plans_identifier, configurations, num_processes, verbose):
    """
    Run preprocess in a dedicated function so it can be executed under the if __name__ == '__main__' guard.
    """
    # Convert single dataset_id to list if necessary
    if isinstance(dataset_id, int):
        dataset_id = [dataset_id]
    
    # Run preprocessing
    print("Running preprocessing...")
    preprocess(dataset_id, plans_identifier, configurations, num_processes, verbose)


def _check_preprocessing_completed(dataset_id: Union[int, List[int]],
                                 plans_identifier: str,
                                 configurations: List[str]) -> bool:
    """
    Check whether preprocessing for the given dataset and configurations has already been completed.

    Args:
        dataset_id: Dataset ID or list of IDs.
        plans_identifier: Plans file identifier.
        configurations: List of configurations to check.

    Returns:
        bool: True if all configurations are already preprocessed, False otherwise.
    """
    from camylanet.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    # Ensure dataset_id is a list
    if isinstance(dataset_id, int):
        dataset_ids = [dataset_id]
    else:
        dataset_ids = dataset_id

    for did in dataset_ids:
        dataset_name = maybe_convert_to_dataset_name(did)
        preprocessed_folder = join(camylanet_preprocessed, dataset_name)

        # Check whether the fingerprint file exists
        fingerprint_file = join(preprocessed_folder, 'dataset_fingerprint.json')
        if not isfile(fingerprint_file):
            return False

        # Check whether the plans file exists
        plans_file = join(preprocessed_folder, f'{plans_identifier}.json')
        if not isfile(plans_file):
            return False

        # Check whether preprocessed data exists for each configuration
        try:
            plans = load_json(plans_file)
            plans_manager = PlansManager(plans)

            for config in configurations:
                if config not in plans_manager.available_configurations:
                    continue  # Skip unavailable configurations

                configuration_manager = plans_manager.get_configuration(config)
                config_folder = join(preprocessed_folder, configuration_manager.data_identifier)

                # Check whether the configuration folder exists and is not empty
                if not isdir(config_folder):
                    return False

                # Check for the presence of preprocessed data files
                npz_files = [f for f in listdir(config_folder) if f.endswith('.npz')]
                if len(npz_files) == 0:
                    return False

        except Exception:
            # Any error while loading the plans file or during the checks is treated as "not preprocessed"
            return False

    return True


def plan_and_preprocess(
    dataset_id: Union[int, List[int]],
    verify_dataset_integrity: bool = False,
    gpu_memory_target: float = 8,
    preprocessor_name: str = 'DefaultPreprocessor',
    overwrite_plans_name: Optional[str] = None,
    overwrite_target_spacing: Optional[List[float]] = None,
    force_target_shape: Optional[List[int]] = None,  # New arg: force the image size after preprocessing
    max_batch_size: int = 32,  # New arg: maximum batch size cap
    force_n_stages: Optional[int] = None,  # New arg: force the number of network stages
    clean: bool = False,
    configurations: List[str] = ['2d', '3d_fullres', '3d_lowres'],
    num_processes: Optional[List[int]] = None,
    verbose: bool = False,
    force_rerun: bool = False,
) -> str:
    """
    Run fingerprint extraction, experiment planning and preprocessing for the specified dataset(s).

    Args:
        dataset_id: Dataset ID or list of dataset IDs
        verify_dataset_integrity: Set to True to check dataset integrity
        gpu_memory_target: GPU memory target in GB
        preprocessor_name: Name of the preprocessor class to use
        overwrite_plans_name: Custom plans identifier
        overwrite_target_spacing: Custom target spacing for 3d_fullres and 3d_cascade_fullres
        force_target_shape: Force the target shape for preprocessing.
                           For 2D configurations: can be [x, y] or [z, x, y] (z will be ignored)
                           For 3D configurations: must be [z, x, y]
                           If None, uses automatic spacing calculation
        max_batch_size: Maximum batch size limit to prevent unreasonably large batch sizes
                       Default is 32, can be adjusted based on GPU memory and requirements
        clean: Set to True to overwrite existing fingerprints
        configurations: Configurations for which preprocessing should be run
        num_processes: Number of processes to use for preprocessing
        verbose: Set to True for verbose output
        force_rerun: Set to True to force rerun even if preprocessing is already completed

    Returns:
        str: plans_identifier - The identifier of the created plans
    """
    # Determine plans_identifier
    plans_identifier = 'nnUNetPlans' if overwrite_plans_name is None else overwrite_plans_name

    # Check whether preprocessing has already been completed
    if not force_rerun and _check_preprocessing_completed(dataset_id, plans_identifier, configurations):
        print("Preprocessing already completed for the specified dataset and configurations. Skipping...")
        if verbose:
            from camylanet.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
            if isinstance(dataset_id, int):
                dataset_name = maybe_convert_to_dataset_name(dataset_id)
                print(f"Dataset: {dataset_name}")
            else:
                for did in dataset_id:
                    dataset_name = maybe_convert_to_dataset_name(did)
                    print(f"Dataset: {dataset_name}")
            print(f"Plans identifier: {plans_identifier}")
            print(f"Configurations: {configurations}")

        # Skip preprocessing and return plans_identifier directly
        return plans_identifier

    if verbose:
        print("Starting plan and preprocess pipeline...")

    # Ensure multiprocessing safety
    ctx = multiprocessing.get_context('spawn')

    # Extract fingerprints
    print("Running fingerprint extraction...")
    p = ctx.Process(target=_run_extract_fingerprints,
                   args=(dataset_id, 'DatasetFingerprintExtractor', 8,
                         verify_dataset_integrity, clean, verbose))
    p.start()
    p.join()

    # Plan experiments
    print("Running experiment planning...")
    p = ctx.Process(target=_run_plan_experiments,
                   args=(dataset_id, 'ExperimentPlanner', gpu_memory_target,
                         preprocessor_name, overwrite_target_spacing, overwrite_plans_name, force_target_shape, max_batch_size, force_n_stages))
    p.start()
    p.join()

    # Set default num_processes
    if num_processes is None:
        default_np = {"2d": 8, "3d_fullres": 4, "3d_lowres": 8}
        num_processes = [default_np[c] if c in default_np.keys() else 4 for c in configurations]

    # Preprocess
    print("Running preprocessing...")
    p = ctx.Process(target=_run_preprocess,
                   args=(dataset_id, plans_identifier, configurations, num_processes, verbose))
    p.start()
    p.join()

    return plans_identifier


def training_network(
    dataset_id: Union[int, str],
    configuration: str,
    fold: Union[int, str] = 0,
    trainer_class: Union[Type[nnUNetTrainer], str] = 'nnUNetTrainer',
    plans_identifier: str = 'nnUNetPlans',
    pretrained_weights: Optional[str] = None,
    num_gpus: int = 1,
    use_compressed_data: bool = False,
    export_validation_probabilities: bool = False,
    continue_training: bool = False,
    only_run_validation: bool = False,
    disable_checkpointing: bool = False,
    val_with_best: bool = False,
    device: Union[torch.device, str] = 'cuda',
    initial_lr: float = 0.01,
    num_epochs: int = 100,
    batch_size: int = None,
    exp_name: Optional[str] = None
) -> Tuple[str, dict]:
    """
    Run training for the specified dataset, configuration and fold.

    Args:
        dataset_id: Dataset ID or name
        configuration: Configuration to use (e.g. '2d', '3d_fullres')
        fold: Fold to use or 'all' for all folds
        trainer_class: Trainer class to use (name or actual class)
        plans_identifier: Plans identifier
        pretrained_weights: Path to pretrained weights
        num_gpus: Number of GPUs to use
        use_compressed_data: Set to True to use compressed data
        export_validation_probabilities: Set to True to export validation probabilities
        continue_training: Set to True to continue training from checkpoint
        only_run_validation: Set to True to only run validation
        disable_checkpointing: Set to True to disable checkpointing
        val_with_best: Set to True to validate with best checkpoint
        device: Device to use for training
        batch_size: Batch size for training (if None, uses default from trainer)
        exp_name: Experiment name to organize results (if None, no additional folder is created)

    Returns:
        tuple: (output_folder, training_log)
            - output_folder (str): Path to the output folder containing training results
            - training_log (dict): Training log containing epoch-wise metrics with keys:
                - 'epochs': List of epoch numbers
                - 'train_losses': List of training losses per epoch
                - 'val_losses': List of validation losses per epoch
    """
    # Make sure the join function is imported

    # Ensure dataset_id is a string (convert if integer)
    if isinstance(dataset_id, int):
        dataset_id = str(dataset_id)

    # Convert device to torch.device if it's a string
    if isinstance(device, str):
        device = torch.device(device)

    # Read the default number of epochs from the environment variable
    if num_epochs == 100:  # Only read from env when the default is in use
        env_num_epochs = os.environ.get('default_num_epochs')
        if env_num_epochs is not None:
            num_epochs = int(env_num_epochs)

    # If custom parameters are provided, instantiate the trainer first so we can modify it
    if initial_lr is not None or num_epochs is not None or batch_size is not None:
        # Check whether trainer_class is an actual class object (rather than a string)
        if not isinstance(trainer_class, str):
            # If trainer_class is a class object, use it directly
            # Obtain the preprocessed_dataset_folder
            from camylanet.paths import camylanet_preprocessed
            dataset_name = maybe_convert_to_dataset_name(dataset_id)
            preprocessed_folder = join(camylanet_preprocessed, dataset_name)

            # Load plans and dataset_json
            plans_file = join(preprocessed_folder, f"{plans_identifier}.json")
            plans = load_json(plans_file)
            dataset_json_file = join(preprocessed_folder, "dataset.json")
            dataset_json = load_json(dataset_json_file)

            # Instantiate the trainer
            trainer = trainer_class(plans=plans, configuration=configuration, fold=fold,
                                    dataset_json=dataset_json, device=device)
        else:
            # If trainer_class is a string, fall back to the original get_trainer_from_args helper
            trainer = get_trainer_from_args(
                dataset_name_or_id=dataset_id,
                configuration=configuration,
                fold=fold,
                trainer_name=trainer_class,
                plans_identifier=plans_identifier,
                use_compressed=use_compressed_data,
                device=device
            )
        
        # If exp_name is provided, modify the output folder path
        if exp_name is not None:
            from camylanet.paths import camylanet_results
            dataset_folder = maybe_convert_to_dataset_name(dataset_id)
            trainer_name_str = trainer_class if isinstance(trainer_class, str) else trainer_class.__name__
            trainer_config_folder = f"{trainer_name_str}__{plans_identifier}__{configuration}"
            
            new_output_folder = join(camylanet_results,
                                    dataset_folder,
                                    exp_name,
                                    trainer_config_folder,
                                    f"fold_{fold}")
            trainer.output_folder = new_output_folder
            print(f"Using custom experiment name. Output folder: {new_output_folder}")
        
        # Set the custom learning rate
        if initial_lr is not None:
            trainer.initial_lr = initial_lr
            print(f"Using custom initial learning rate: {initial_lr}")
        
        # Set the custom number of epochs
        if num_epochs is not None:
            trainer.num_epochs = num_epochs
            print(f"Using custom number of epochs: {num_epochs}")
        
        # Set the custom batch size
        if batch_size is not None:
            # Must be set before initialize(), since initialize() derives other params from batch_size
            if hasattr(trainer, 'batch_size'):
                trainer.batch_size = batch_size
                print(f"Using custom batch size: {batch_size}")
            else:
                print("Warning: Trainer does not have a batch_size attribute. Custom batch size will be ignored.")
        
        # Run training
        if pretrained_weights is not None:
            if not trainer.was_initialized:
                trainer.initialize()
            from camylanet.run.load_pretrained_weights import load_pretrained_weights
            load_pretrained_weights(trainer.network, pretrained_weights, verbose=True)
        
        # Disable checkpointing if requested
        if disable_checkpointing:
            trainer.disable_checkpointing = disable_checkpointing

        # Run training
        if not only_run_validation:
            trainer.run_training()
        else:
            # When only running validation, ensure the trainer is initialized and the checkpoint is loaded
            if not trainer.was_initialized:
                trainer.initialize()
            # Load the final checkpoint for validation
            expected_checkpoint_file = join(trainer.output_folder, 'checkpoint_final.pth')
            if not isfile(expected_checkpoint_file):
                raise RuntimeError(f"Cannot run validation because the training is not finished yet! Expected checkpoint file: {expected_checkpoint_file}")
            trainer.load_checkpoint(expected_checkpoint_file)
        
        # Validate
        if val_with_best:
            trainer.load_checkpoint(join(trainer.output_folder, 'checkpoint_best.pth'))
        trainer.perform_actual_validation(export_validation_probabilities)
        
        # Retrieve output folder and training log
        output_folder = trainer.output_folder
        training_log = _extract_training_log(trainer.logger)
    else:
        # Check whether trainer_class is an actual class object (rather than a string)
        if not isinstance(trainer_class, str):
            # If it is a class object, create a temporary module to hold it
            import sys
            import types

            # Create a temporary module
            mod_name = f"camylanet.training.nnUNetTrainer.{trainer_class.__name__}"
            if mod_name not in sys.modules:
                mod = types.ModuleType(mod_name)
                setattr(mod, trainer_class.__name__, trainer_class)
                sys.modules[mod_name] = mod

            # Use the class name as trainer_class_name
            trainer_class_name = trainer_class.__name__
        else:
            trainer_class_name = trainer_class

        # Use the original run_training function
        run_training(
            dataset_name_or_id=dataset_id,
            configuration=configuration,
            fold=fold,
            trainer_class_name=trainer_class_name,
            plans_identifier=plans_identifier,
            pretrained_weights=pretrained_weights,
            num_gpus=num_gpus,
            use_compressed_data=use_compressed_data,
            export_validation_probabilities=export_validation_probabilities,
            continue_training=continue_training,
            only_run_validation=only_run_validation,
            disable_checkpointing=disable_checkpointing,
            val_with_best=val_with_best,
            device=device
        )
        
        # Determine the output folder
        from camylanet.paths import camylanet_results
        dataset_folder = f"Dataset{dataset_id}" if isinstance(dataset_id, int) else dataset_id
        trainer_config_folder = f"{trainer_class_name}__{plans_identifier}__{configuration}"
        
        if exp_name is not None:
            output_folder = join(camylanet_results,
                                dataset_folder,
                                exp_name,
                                trainer_config_folder,
                                f"fold_{fold}")
        else:
            output_folder = join(camylanet_results,
                                dataset_folder,
                                trainer_config_folder,
                                f"fold_{fold}")

        # Load the training log from the output folder
        training_log = _load_training_log_from_folder(output_folder)

    return output_folder, training_log


def evaluate(
    dataset_id: Union[int, str],
    result_folder: str,
    fold: Optional[Union[int, str]] = 0,
    output_file: Optional[str] = None,
    num_processes: int = 8,
    chill: bool = True,
    exp_name: Optional[str] = None
) -> dict:
    """
    Evaluate predictions in the result folder.
    
    Args:
        dataset_id: Dataset ID or name
        result_folder: Path to the result folder
        fold: Fold to evaluate (if None, uses the fold from result_folder)
        output_file: Path to output file
        num_processes: Number of processes to use
        chill: Set to True to not crash if folder_pred doesn't have all files
        exp_name: Experiment name (should match the exp_name used in training)
        
    Returns:
        results: Evaluation results
    """
    # Ensure dataset_id is a string (convert if integer)
    if isinstance(dataset_id, int):
        dataset_id = str(dataset_id)

    # Convert dataset_id to dataset name if it's an integer
    dataset_name = maybe_convert_to_dataset_name(dataset_id)
    
    # Determine fold from result_folder if not provided
    if fold is None and "fold_" in result_folder:
        fold = result_folder.split("fold_")[-1].split("/")[0]
    
    # Get preprocessed dataset folder
    preprocessed_folder = join(camylanet_preprocessed, dataset_name)
    
    # Load dataset.json
    dataset_json_file = join(preprocessed_folder, "dataset.json")
    
    # Determine plans file - parse the trainer_class_name__plans_identifier__configuration format
    # The path may be: Dataset/exp_name/trainer__plans__config/fold or Dataset/trainer__plans__config/fold
    path_parts = result_folder.replace('\\', '/').split('/')
    trainer_config_part = None
    for part in path_parts:
        if '__' in part and part.count('__') >= 2:
            trainer_config_part = part
            break
    
    if trainer_config_part is None:
        raise ValueError(f"Cannot parse trainer configuration from result_folder: {result_folder}")
    
    plans_identifier = trainer_config_part.split("__")[1]
    plans_file = join(preprocessed_folder, f"{plans_identifier}.json")
    
    # Determine gt and pred folders
    if fold is not None:
        gt_folder = join(preprocessed_folder, "gt_segmentations")
        pred_folder = join(result_folder, "validation")
    else:
        # If no fold is specified, assume we're evaluating test predictions
        gt_folder = join(preprocessed_folder, "gt_segmentations")
        pred_folder = join(result_folder, "test_predictions")
    
    # Run evaluation
    if output_file is None:
        output_file = join(pred_folder, "summary.json")
    
    # Run evaluation in a separate process to avoid multiprocessing issues
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(target=compute_metrics_on_folder2,
                   args=(gt_folder, pred_folder, dataset_json_file, plans_file, 
                         output_file, num_processes, chill))
    p.start()
    p.join()
    
    # Load and return results
    from camylanet.evaluation.evaluate_predictions import load_summary_json
    return load_summary_json(output_file)


def get_patch_size_from_plans(
    dataset_id: Union[int, str],
    plans_identifier: str = 'nnUNetPlans',
    configurations: Optional[List[str]] = None
) -> Dict[str, List[int]]:
    """
    Retrieve patch size info for each configuration from the plans file of a preprocessed dataset.

    Args:
        dataset_id: Dataset ID or name.
        plans_identifier: Plans file identifier, default 'nnUNetPlans'.
        configurations: List of configurations to query. If None, returns the patch size of every available
                        configuration, e.g. ['2d', '3d_fullres', '3d_lowres'].

    Returns:
        dict: Mapping from configuration name to patch size, for example:
              {
                  '2d': [512, 512],
                  '3d_fullres': [128, 128, 128],
                  '3d_lowres': [64, 64, 64]
              }

    Example:
        >>> # Get patch size for all configurations
        >>> patch_sizes = get_patch_size_from_plans(dataset_id=4)
        >>> print(patch_sizes)
        {'2d': [512, 512], '3d_fullres': [40, 224, 192]}

        >>> # Get patch size for specific configurations
        >>> patch_sizes = get_patch_size_from_plans(
        ...     dataset_id=4,
        ...     configurations=['2d', '3d_fullres']
        ... )
        >>> print(patch_sizes['2d'])
        [512, 512]
    """
    # Ensure dataset_id is a string
    if isinstance(dataset_id, int):
        dataset_id = str(dataset_id)

    # Convert to dataset name
    dataset_name = maybe_convert_to_dataset_name(dataset_id)

    # Path to the preprocessed dataset folder
    preprocessed_folder = join(camylanet_preprocessed, dataset_name)

    # Check that the preprocessed folder exists
    if not isdir(preprocessed_folder):
        raise RuntimeError(
            f"Preprocessed folder does not exist: {preprocessed_folder}\n"
            f"Please run plan_and_preprocess() on dataset {dataset_name} first"
        )

    # Plans file path
    plans_file = join(preprocessed_folder, f'{plans_identifier}.json')

    # Check that the plans file exists
    if not isfile(plans_file):
        raise RuntimeError(
            f"Plans file does not exist: {plans_file}\n"
            f"Please run plan_and_preprocess() on dataset {dataset_name} first"
        )

    # Load the plans file
    plans_manager = PlansManager(plans_file)

    # All available configurations
    available_configs = plans_manager.available_configurations

    # If no configurations were specified, use all available ones
    if configurations is None:
        configurations = available_configs
    else:
        # Verify the requested configurations exist
        for config in configurations:
            if config not in available_configs:
                print(f"Warning: configuration '{config}' is not in the plans file. Available: {available_configs}")

    # Gather patch size info
    patch_sizes = {}

    for config_name in configurations:
        if config_name in available_configs:
            try:
                config_manager = plans_manager.get_configuration(config_name)
                patch_size = config_manager.patch_size
                patch_sizes[config_name] = patch_size
            except Exception as e:
                print(f"Warning: could not retrieve patch size for configuration '{config_name}': {str(e)}")

    return patch_sizes


def print_patch_size_info(
    dataset_id: Union[int, str],
    plans_identifier: str = 'nnUNetPlans',
    configurations: Optional[List[str]] = None
) -> None:
    """
    Print detailed patch size info for each configuration of a dataset, plus additional related parameters.

    Args:
        dataset_id: Dataset ID or name.
        plans_identifier: Plans file identifier, default 'nnUNetPlans'.
        configurations: List of configurations to query; if None, prints all available configurations.

    Example:
        >>> print_patch_size_info(dataset_id=4)
        ================================================================================
        Dataset: Dataset004_Hippocampus
        Plans: nnUNetPlans
        ================================================================================

        Configuration: 2d
        --------------------------------------------------------------------------------
          Patch Size:              [512, 512]
          Spacing:                 [1.0, 1.0]
          Batch Size:              12
          Median Image Size:       [35, 512, 512]

        Configuration: 3d_fullres
        --------------------------------------------------------------------------------
          Patch Size:              [40, 224, 192]
          Spacing:                 [3.0, 1.0, 1.0]
          Batch Size:              2
          Median Image Size:       [35, 512, 512]
    """
    # Ensure dataset_id is a string
    if isinstance(dataset_id, int):
        dataset_id = str(dataset_id)

    # Convert to dataset name
    dataset_name = maybe_convert_to_dataset_name(dataset_id)

    # Path to the preprocessed dataset folder
    preprocessed_folder = join(camylanet_preprocessed, dataset_name)

    # Plans file path
    plans_file = join(preprocessed_folder, f'{plans_identifier}.json')

    # Check that the plans file exists
    if not isfile(plans_file):
        raise RuntimeError(
            f"Plans file does not exist: {plans_file}\n"
            f"Please run plan_and_preprocess() on dataset {dataset_name} first"
        )

    # Load the plans file
    plans_manager = PlansManager(plans_file)

    # All available configurations
    available_configs = plans_manager.available_configurations

    # If no configurations were specified, use all available ones
    if configurations is None:
        configurations = available_configs

    # Print header info
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Plans: {plans_identifier}")
    print("=" * 80)
    print()

    # Print info for each configuration
    for config_name in configurations:
        if config_name in available_configs:
            try:
                config_manager = plans_manager.get_configuration(config_name)

                print(f"Configuration: {config_name}")
                print("-" * 80)
                print(f"  Patch Size:              {config_manager.patch_size}")
                print(f"  Spacing:                 {config_manager.spacing}")
                print(f"  Batch Size:              {config_manager.batch_size}")
                print(f"  Median Image Size:       {config_manager.median_image_size_in_voxels}")

                # Retrieve network architecture info
                try:
                    arch_kwargs = config_manager.network_arch_init_kwargs
                    print(f"  Network Stages:          {arch_kwargs.get('n_stages', 'N/A')}")
                    print(f"  Features per Stage:      {arch_kwargs.get('features_per_stage', 'N/A')}")
                    print(f"  Kernel Sizes:            {arch_kwargs.get('kernel_sizes', 'N/A')}")
                    print(f"  Strides:                 {arch_kwargs.get('strides', 'N/A')}")
                except:
                    pass

                print()
            except Exception as e:
                print(f"Warning: could not retrieve info for configuration '{config_name}': {str(e)}")
                print()
        else:
            print(f"Warning: configuration '{config_name}' does not exist. Available: {available_configs}")
            print()


def training_network_1epoch(
    dataset_id: Union[int, str],
    configuration: str,
    fold: Union[int, str] = 0,
    trainer_class: Union[Type[nnUNetTrainer], str] = 'nnUNetTrainer',
    plans_identifier: str = 'nnUNetPlans',
    pretrained_weights: Optional[str] = None,
    num_gpus: int = 1,
    use_compressed_data: bool = False,
    export_validation_probabilities: bool = False,
    continue_training: bool = False,
    disable_checkpointing: bool = False,
    val_with_best: bool = False,
    device: Union[torch.device, str] = 'cuda',
    initial_lr: float = 0.01,
    batch_size: int = None,
    exp_name: Optional[str] = None
) -> Tuple[str, dict]:
    """
    Run training for 1 epoch, for quick testing of the model, data pipeline, and training flow.

    This is a convenient test variant of training_network that automatically sets num_epochs=1.
    Useful for:
    - Verifying that data preprocessing is correct
    - Testing that a custom network architecture works
    - Quickly checking for errors in the training pipeline
    - Validating that the GPU / memory configuration is reasonable

    Args:
        dataset_id: Dataset ID or name.
        configuration: Configuration (e.g. '2d', '3d_fullres').
        fold: Fold number (default 0).
        trainer_class: Trainer class (default 'nnUNetTrainer').
        plans_identifier: Plans identifier (default 'nnUNetPlans').
        pretrained_weights: Path to pretrained weights.
        num_gpus: Number of GPUs (default 1).
        use_compressed_data: Whether to use compressed data (default False).
        export_validation_probabilities: Whether to export validation probabilities (default False).
        continue_training: Whether to continue training (default False).
        disable_checkpointing: Disable checkpointing (default False).
        val_with_best: Validate with the best checkpoint (default False).
        device: Device (default 'cuda').
        initial_lr: Initial learning rate (default 0.01).
        batch_size: Batch size.
        exp_name: Experiment name (used to organize results).

    Returns:
        tuple: (output_folder, training_log)
            - output_folder (str): Path to the training-output folder.
            - training_log (dict): Training log with keys:
                - 'epochs': list of epoch indices
                - 'train_losses': list of training losses
                - 'val_losses': list of validation losses

    Example:
        >>> # Quick test of 2D configuration
        >>> result_folder, log = camylanet.training_network_1epoch(
        ...     dataset_id=4,
        ...     configuration='2d',
        ...     plans_identifier='nnUNetPlans',
        ...     exp_name='quick_test'
        ... )
        >>> print(f"Training loss: {log['train_losses'][0]:.4f}")
        >>> print(f"Validation loss: {log['val_losses'][0]:.4f}")

        >>> # Test a custom network
        >>> result_folder, log = camylanet.training_network_1epoch(
        ...     dataset_id=4,
        ...     configuration='2d',
        ...     trainer_class=MyCustomTrainer,
        ...     exp_name='custom_net_test'
        ... )
    """
    print("=" * 80)
    print("Running 1-epoch training test")
    print("=" * 80)
    print(f"Dataset: {dataset_id}")
    print(f"Configuration: {configuration}")
    print(f"Trainer: {trainer_class if isinstance(trainer_class, str) else trainer_class.__name__}")
    print(f"Experiment name: {exp_name if exp_name else '(default)'}")
    print("=" * 80)
    print()

    # Call the original training_network function with num_epochs fixed to 1
    return training_network(
        dataset_id=dataset_id,
        configuration=configuration,
        fold=fold,
        trainer_class=trainer_class,
        plans_identifier=plans_identifier,
        pretrained_weights=pretrained_weights,
        num_gpus=num_gpus,
        use_compressed_data=use_compressed_data,
        export_validation_probabilities=export_validation_probabilities,
        continue_training=continue_training,
        only_run_validation=False,  # Should not only run validation during a test
        disable_checkpointing=disable_checkpointing,
        val_with_best=val_with_best,
        device=device,
        initial_lr=initial_lr,
        num_epochs=1,  # Fixed to 1 epoch
        batch_size=batch_size,
        exp_name=exp_name
    )


def get_available_configurations(
    dataset_id: Union[int, str],
    plans_identifier: str = 'nnUNetPlans'
) -> List[str]:
    """
    Return the list of available configurations for the given dataset (excluding 3d_lowres).

    Args:
        dataset_id: Dataset ID or name.
        plans_identifier: Plans file identifier (default 'nnUNetPlans').

    Returns:
        list: Available configurations, e.g. ['2d', '3d_fullres']. Does not include '3d_lowres'.
    """
    # Ensure dataset_id is a string
    if isinstance(dataset_id, int):
        dataset_id = str(dataset_id)

    # Convert to dataset name
    dataset_name = maybe_convert_to_dataset_name(dataset_id)

    # Path to the preprocessed dataset folder
    preprocessed_folder = join(camylanet_preprocessed, dataset_name)

    # Check that the preprocessed folder exists
    if not isdir(preprocessed_folder):
        raise RuntimeError(
            f"Preprocessed folder does not exist: {preprocessed_folder}\n"
            f"Please run plan_and_preprocess() on dataset {dataset_name} first"
        )

    # Plans file path
    plans_file = join(preprocessed_folder, f'{plans_identifier}.json')

    # Check that the plans file exists
    if not isfile(plans_file):
        raise RuntimeError(
            f"Plans file does not exist: {plans_file}\n"
            f"Please run plan_and_preprocess() on dataset {dataset_name} first"
        )

    # Load the plans file
    plans_manager = PlansManager(plans_file)

    # All available configurations
    available_configs = plans_manager.available_configurations

    # Filter out 3d_lowres
    filtered_configs = [c for c in available_configs if c != '3d_lowres']
    
    return filtered_configs


def dataset_exists(dataset_id: Union[int, str]) -> bool:
    """
    Check if a dataset exists with the given ID or name.
    
    Args:
        dataset_id: Dataset ID (int) or name (str)
        
    Returns:
        bool: True if the dataset exists and is unique, False otherwise.
    """
    try:
        maybe_convert_to_dataset_name(dataset_id)
        return True
    except (RuntimeError, ValueError):
        return False 