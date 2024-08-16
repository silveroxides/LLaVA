import torch

def inspect_saved_weights(output_dir, component_name, checkpoint_folder=None):
    """Load and inspect the structure of the saved weights."""
    if checkpoint_folder:
        filepath = os.path.join(output_dir, 'components', f'{component_name}_{checkpoint_folder}.bin')
    else:
        filepath = os.path.join(output_dir, f'{component_name}.bin')

    component_weights = torch.load(filepath)
    return component_weights

# Load and inspect the structure for a specific component, e.g., 'mm_projector'
mm_projector_weights = inspect_saved_weights('/path/to/output_dir', 'mm_projector', 'checkpoint-12345')

# Print the keys (which represent the structure)
print("Keys in mm_projector weights:")
for key in mm_projector_weights.keys():
    print(f"{key}: {mm_projector_weights[key].shape}")

# Do the same for other components, e.g., 'cross_attention'
cross_attention_weights = inspect_saved_weights('/path/to/output_dir', 'cross_attention', 'checkpoint-12345')

print("\nKeys in cross_attention weights:")
for key in cross_attention_weights.keys():
    print(f"{key}: {cross_attention_weights[key].shape}")
