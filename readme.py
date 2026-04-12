import camylanet

# Dataset ID
dataset_id = 4
configuration = '3d_fullres'
exp_name = "experiment_v1"

# Step 1: preprocess the data
plans_identifier = camylanet.plan_and_preprocess(
    dataset_id=dataset_id,
    configurations=[configuration]
)

# Step 2: train using the default trainer
result_folder, training_log = camylanet.training_network(
    dataset_id=dataset_id,
    configuration=configuration,
    plans_identifier=plans_identifier,
    exp_name=exp_name  # Add experiment name
)

print(training_log)

# Step 3: evaluate the results
results = camylanet.evaluate(
    dataset_id=dataset_id,
    result_folder=result_folder,
    exp_name=exp_name  # Must match the experiment name used during training
)

# Print training log information
print(f"\nTraining log info:")
print(f"Training epochs: {len(training_log['epochs'])}")
if training_log['train_losses']:
    print(f"Final training loss: {training_log['train_losses'][-1]:.4f}")
    print(f"Final validation loss: {training_log['val_losses'][-1]:.4f}")

# Print evaluation results
print(f"\nEvaluation results:")
print(f"Mean Dice score: {results['foreground_mean']['Dice']:.4f}")
print(f"Mean IoU score: {results['foreground_mean']['IoU']:.4f}")
print(f"Mean HD95 score: {results['foreground_mean']['HD95']:.4f}")
