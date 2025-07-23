import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

# Update to load batch data instead of trajectory data
output_folder = 'best_batches'
best_batches_file = os.path.join(output_folder, 'best_batches_data.pkl')
output_folder_vis = 'best_batches_baseline_vis'
os.makedirs(output_folder_vis, exist_ok=True)

def plot_batch_trajectories(batch, batch_rank, output_folder):
    """
    Plot trajectories for a single batch with proper styling and save to file.
    
    Args:
        batch: Dictionary containing batch data with keys 'observed', 'predicted', 'ground_truth', etc.
        batch_rank: Integer rank/index of the batch
        output_folder: String path to output folder for saving plots
    """
    # Define colors for different pedestrians
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Extract batch data
    batch_observed = batch['observed']
    batch_predicted = batch['predicted'] 
    batch_ground_truth = batch['ground_truth']
    batch_ades = batch['pedestrian_ades']
    batch_fdes = batch['pedestrian_fdes']
    num_peds = len(batch_ades)
    
    # Plot each pedestrian
    for ped_idx in range(num_peds):
        color = colors[ped_idx % len(colors)]
        
        observed = batch_observed[ped_idx]
        predicted = batch_predicted[ped_idx] 
        ground_truth = batch_ground_truth[ped_idx]
        ped_ade = batch_ades[ped_idx]
        
        # Plot trajectories
        ax.plot(observed[:, 0], observed[:, 1], 
                color=color, linewidth=2, alpha=0.8,
                label=f'Ped {ped_idx+1} Hist (ADE: {ped_ade:.3f})', 
                marker='o', markersize=3)
        
        ax.plot(predicted[:, 0], predicted[:, 1], 
                color=color, linewidth=2, alpha=0.8, linestyle='--',
                marker='s', markersize=3)
        
        ax.plot(ground_truth[:, 0], ground_truth[:, 1], 
                color=color, linewidth=1.5, alpha=0.6, linestyle=':',
                marker='^', markersize=3)
        
        # Mark start and end points
        ax.scatter(observed[0, 0], observed[0, 1], color=color, s=60, marker='o', 
                  alpha=1.0, edgecolors='black', linewidth=1)
        ax.scatter(predicted[-1, 0], predicted[-1, 1], color=color, s=60, marker='s', 
                  alpha=1.0, edgecolors='black', linewidth=1)
        ax.scatter(ground_truth[-1, 0], ground_truth[-1, 1], color=color, s=60, marker='^', 
                  alpha=1.0, edgecolors='black', linewidth=1)
    
    # Add legend explanation
    ax.plot([], [], color='gray', linewidth=2, alpha=0.8, label='Historical')
    ax.plot([], [], color='gray', linewidth=2, alpha=0.8, linestyle='--', label='Predicted')
    ax.plot([], [], color='gray', linewidth=1.5, alpha=0.6, linestyle=':', label='Ground Truth')
    
    # Set labels and title
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    ax.set_title(f'Batch Rank {batch_rank} - Dataset: {batch["dataset"]} - Step: {batch["step"]}\n'
                f'Average ADE: {batch["avg_ade"]:.4f}, Average FDE: {batch["avg_fde"]:.4f}\n'
                f'Number of Pedestrians: {num_peds}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Create compact legend
    handles, labels = ax.get_legend_handles_labels()
    max_ped_labels = min(6, num_peds)
    ped_handles = handles[:max_ped_labels]
    ped_labels = labels[:max_ped_labels]
    type_handles = handles[-3:]
    type_labels = labels[-3:]
    
    if num_peds > max_ped_labels:
        ped_labels[-1] = f'... and {num_peds - max_ped_labels + 1} more pedestrians'
    
    all_handles = ped_handles + type_handles
    all_labels = ped_labels + type_labels
    ax.legend(all_handles, all_labels, fontsize=9, loc='upper right', 
             bbox_to_anchor=(1.0, 1.0))
    
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    # Save the plot
    filename = f'batch_rank_{batch_rank:02d}_dataset_{batch["dataset"]}_step_{batch["step"]}_avg_ade_{batch["avg_ade"]:.4f}.png'
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()

def print_batch_info():
    """
    Load and print detailed information about the best batches data.
    """

    with open(best_batches_file, 'rb') as f:
        best_batches = pickle.load(f)
    print(f"Successfully loaded {len(best_batches)} best batches from saved data")
    print("="*80)
    
    for i, batch in enumerate(best_batches):
        print(f"\nBATCH RANK {i+1}:")
        print("-" * 50) 
        # Use the new plotting function
        plot_batch_trajectories(batch, i+1, output_folder_vis)
    
    return True


if __name__ == "__main__":
    print("LOADING AND ANALYZING BEST BATCHES DATA")
    print("="*80)
    
    
    # Load and print batch information
    success = print_batch_info()
    