import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import glob
from utils import *
from metrics import *
from model import STCRF 
from CFG import CFG
import os
import random
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Static distance thresholds for trajectory filtering
static_dist_dict = {
    'eth': 0.419,
    'hotel': 0.353,
    'univ': 0.227,
    'zara1': 0.338,
    'zara2': 0.350
}

def plot_best_trajectories(trajectory_data, num_trajectories=15):
    """
    Plot the best trajectories based on ADE scores.
    Each trajectory is saved as a separate image in a folder.
    
    Args:
        trajectory_data: List of trajectory dictionaries or None (if should load from saved)
        num_trajectories: Number of best trajectories to plot
    """
    # Create output folder
    output_folder = 'best_trajectories'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Path for saving/loading best trajectories data
    best_trajectories_file = os.path.join(output_folder, 'best_trajectories_data.pkl')
    
    # Check if we should load existing data or use provided data
    if trajectory_data is None:
        # Try to load existing best trajectories data
        if os.path.exists(best_trajectories_file):
            print(f"Loading existing best trajectories data from {best_trajectories_file}")
            with open(best_trajectories_file, 'rb') as f:
                best_trajectories = pickle.load(f)
            print(f"Loaded {len(best_trajectories)} best trajectories from saved data")
        else:
            print("No saved best trajectories data found. Please run the full computation first.")
            return
    else:
        # Sort trajectories by ADE (ascending - lower is better)
        sorted_trajectories = sorted(trajectory_data, key=lambda x: x['ade'])
        
        # Select the best trajectories
        best_trajectories = sorted_trajectories[:num_trajectories]
        
        # Save the best trajectories data for future use
        with open(best_trajectories_file, 'wb') as f:
            pickle.dump(best_trajectories, f)
        print(f"Saved {len(best_trajectories)} best trajectories to {best_trajectories_file}")
    
    # Ensure we don't exceed the available trajectories
    if len(best_trajectories) > num_trajectories:
        best_trajectories = best_trajectories[:num_trajectories]
    
    print(f"\nPlotting {len(best_trajectories)} most accurate trajectories...")
    print(f"ADE range: {best_trajectories[0]['ade']:.4f} to {best_trajectories[-1]['ade']:.4f}")
    
    # Plot each trajectory separately
    for i, traj in enumerate(best_trajectories):
        # Create figure for this trajectory
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract trajectory data
        observed = traj['observed']  # Historical path
        predicted = traj['predicted']  # Predicted future path
        ground_truth = traj['ground_truth']  # Ground truth future path
        
        # Use different colors for each trajectory type
        hist_color = '#2E86AB'  # Blue for historical
        pred_color = '#A23B72'  # Red for predicted
        gt_color = '#F18F01'    # Orange for ground truth
        
        # Plot historical trajectory
        ax.plot(observed[:, 0], observed[:, 1], 
                color=hist_color, linewidth=3, alpha=0.9,
                label='Historical', marker='o', markersize=4)
        
        # Plot predicted trajectory
        ax.plot(predicted[:, 0], predicted[:, 1], 
                color=pred_color, linewidth=3, alpha=0.9, linestyle='--',
                label='Predicted', marker='s', markersize=4)
        
        # Plot ground truth
        ax.plot(ground_truth[:, 0], ground_truth[:, 1], 
                color=gt_color, linewidth=2, alpha=0.8, linestyle=':',
                label='Ground Truth', marker='^', markersize=4)
        
        # Mark start and end points
        ax.scatter(observed[0, 0], observed[0, 1], color=hist_color, s=100, marker='o', 
                  alpha=1.0, edgecolors='black', linewidth=2, label='Start')
        ax.scatter(predicted[-1, 0], predicted[-1, 1], color=pred_color, s=100, marker='s', 
                  alpha=1.0, edgecolors='black', linewidth=2, label='Pred End')
        ax.scatter(ground_truth[-1, 0], ground_truth[-1, 1], color=gt_color, s=100, marker='^', 
                  alpha=1.0, edgecolors='black', linewidth=2, label='GT End')
        
        # Set labels and title
        ax.set_xlabel('X coordinate', fontsize=12)
        ax.set_ylabel('Y coordinate', fontsize=12)
        ax.set_title(f'Trajectory Rank {i+1} - Dataset: {traj["dataset"]}\n'
                    f'ADE: {traj["ade"]:.4f}, FDE: {traj["fde"]:.4f}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Make axes equal for better visualization
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # Save each trajectory as a separate image
        filename = f'trajectory_rank_{i+1:02d}_dataset_{traj["dataset"]}_ade_{traj["ade"]:.4f}.png'
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()  # Close the figure to free memory
    
    print(f"\nAll {len(best_trajectories)} trajectory plots saved in folder: {output_folder}")
    
    # Print summary statistics
    print("\nSummary of best trajectories:")
    for i, traj in enumerate(best_trajectories[:5]):  # Show top 5
        print(f"Rank {i+1}: ADE={traj['ade']:.4f}, FDE={traj['fde']:.4f}, "
              f"Dataset={traj['dataset']}, Step={traj['step']}, Ped={traj['pedestrian_id']}")


def load_and_plot_saved_trajectories(num_trajectories=15):
    """
    Load and plot saved best trajectories without recomputing.
    
    Args:
        num_trajectories: Number of best trajectories to plot
    """
    output_folder = 'best_trajectories'
    best_trajectories_file = os.path.join(output_folder, 'best_trajectories_data.pkl')
    
    if os.path.exists(best_trajectories_file):
        print("Found saved best trajectories data. Loading and plotting...")
        plot_best_trajectories(trajectory_data=None, num_trajectories=num_trajectories)
        return True
    else:
        print("No saved best trajectories data found.")
        return False

def plot_best_batches(batch_data, num_batches=10):
    """
    Plot the best batches based on average ADE scores.
    Each batch is saved as a separate image showing all pedestrians in that batch.
    
    Args:
        batch_data: List of batch dictionaries or None (if should load from saved)
        num_batches: Number of best batches to plot
    """
    # Create output folder
    output_folder = 'best_batches'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Path for saving/loading best batches data
    best_batches_file = os.path.join(output_folder, 'best_batches_data.pkl')
    
    # Check if we should load existing data or use provided data
    if batch_data is None:
        # Try to load existing best batches data
        if os.path.exists(best_batches_file):
            print(f"Loading existing best batches data from {best_batches_file}")
            with open(best_batches_file, 'rb') as f:
                best_batches = pickle.load(f)
            print(f"Loaded {len(best_batches)} best batches from saved data")
        else:
            print("No saved best batches data found. Please run the full computation first.")
            return
    else:
        # Sort batches by average ADE (ascending - lower is better)
        sorted_batches = sorted(batch_data, key=lambda x: x['avg_ade'])
        
        # Select the best batches
        best_batches = sorted_batches[:num_batches]
        
        # Save the best batches data for future use
        with open(best_batches_file, 'wb') as f:
            pickle.dump(best_batches, f)
        print(f"Saved {len(best_batches)} best batches to {best_batches_file}")
    
    # Ensure we don't exceed the available batches
    if len(best_batches) > num_batches:
        best_batches = best_batches[:num_batches]
    
    print(f"\nPlotting {len(best_batches)} most accurate batches...")
    print(f"Average ADE range: {best_batches[0]['avg_ade']:.4f} to {best_batches[-1]['avg_ade']:.4f}")
    
    # Define colors for different pedestrians
    colors = plt.cm.Set3(np.linspace(0, 1, 12))  # Use Set3 colormap for distinct colors
    
    # Plot each batch separately
    for i, batch in enumerate(best_batches):
        # Create figure for this batch
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Extract batch data
        batch_observed = batch['observed']  # [num_ped, obs_len, 2]
        batch_predicted = batch['predicted']  # [num_ped, pred_len, 2]
        batch_ground_truth = batch['ground_truth']  # [num_ped, pred_len, 2]
        batch_ades = batch['pedestrian_ades']  # [num_ped]
        batch_fdes = batch['pedestrian_fdes']  # [num_ped]
        num_peds = len(batch_ades)
        
        # Plot each pedestrian in the batch
        for ped_idx in range(num_peds):
            color = colors[ped_idx % len(colors)]
            
            observed = batch_observed[ped_idx]  # [obs_len, 2]
            predicted = batch_predicted[ped_idx]  # [pred_len, 2]
            ground_truth = batch_ground_truth[ped_idx]  # [pred_len, 2]
            ped_ade = batch_ades[ped_idx]
            ped_fde = batch_fdes[ped_idx]
            
            # Plot historical trajectory
            ax.plot(observed[:, 0], observed[:, 1], 
                    color=color, linewidth=2, alpha=0.8,
                    label=f'Ped {ped_idx+1} Hist (ADE: {ped_ade:.3f})', 
                    marker='o', markersize=3)
            
            # Plot predicted trajectory
            ax.plot(predicted[:, 0], predicted[:, 1], 
                    color=color, linewidth=2, alpha=0.8, linestyle='--',
                    marker='s', markersize=3)
            
            # Plot ground truth
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
        ax.set_title(f'Batch Rank {i+1} - Dataset: {batch["dataset"]} - Step: {batch["step"]}\n'
                    f'Average ADE: {batch["avg_ade"]:.4f}, Average FDE: {batch["avg_fde"]:.4f}\n'
                    f'Number of Pedestrians: {num_peds}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Create a compact legend
        handles, labels = ax.get_legend_handles_labels()
        # Show only the first few pedestrian labels to avoid clutter
        max_ped_labels = min(6, num_peds)
        ped_handles = handles[:max_ped_labels]
        ped_labels = labels[:max_ped_labels]
        type_handles = handles[-3:]  # Historical, Predicted, Ground Truth
        type_labels = labels[-3:]
        
        if num_peds > max_ped_labels:
            ped_labels[-1] = f'... and {num_peds - max_ped_labels + 1} more pedestrians'
        
        all_handles = ped_handles + type_handles
        all_labels = ped_labels + type_labels
        ax.legend(all_handles, all_labels, fontsize=9, loc='upper right', 
                 bbox_to_anchor=(1.0, 1.0))
        
        # Make axes equal for better visualization
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # Save each batch as a separate image
        filename = f'batch_rank_{i+1:02d}_dataset_{batch["dataset"]}_step_{batch["step"]}_avg_ade_{batch["avg_ade"]:.4f}.png'
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()  # Close the figure to free memory
    
    print(f"\nAll {len(best_batches)} batch plots saved in folder: {output_folder}")
    
    # Print summary statistics
    print("\nSummary of best batches:")
    for i, batch in enumerate(best_batches[:5]):  # Show top 5
        print(f"Rank {i+1}: Avg ADE={batch['avg_ade']:.4f}, Avg FDE={batch['avg_fde']:.4f}, "
              f"Dataset={batch['dataset']}, Step={batch['step']}, Num Peds={len(batch['pedestrian_ades'])}")


def load_and_plot_saved_batches(num_batches=10):
    """
    Load and plot saved best batches without recomputing.
    
    Args:
        num_batches: Number of best batches to plot
    """
    output_folder = 'best_batches'
    best_batches_file = os.path.join(output_folder, 'best_batches_data.pkl')
    
    if os.path.exists(best_batches_file):
        print("Found saved best batches data. Loading and plotting...")
        plot_best_batches(batch_data=None, num_batches=num_batches)
        return True
    else:
        print("No saved best batches data found.")
        return False

def export_best_batch_data(batch_data=None):
    """
    Export the best batch's observed+ground_truth to best.txt and predicted to pred.txt.
    
    Args:
        batch_data: List of batch dictionaries or None (if should load from saved)
    """
    # Create output folder
    output_folder = 'best_batches'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Path for saving/loading best batches data
    best_batches_file = os.path.join(output_folder, 'best_batches_data.pkl')
    
    # Check if we should load existing data or use provided data
    if batch_data is None:
        # Try to load existing best batches data
        if os.path.exists(best_batches_file):
            print(f"Loading existing best batches data from {best_batches_file}")
            with open(best_batches_file, 'rb') as f:
                best_batches = pickle.load(f)
            print(f"Loaded {len(best_batches)} best batches from saved data")
        else:
            print("No saved best batches data found. Please run the full computation first.")
            return False
    else:
        # Sort batches by average ADE (ascending - lower is better)
        sorted_batches = sorted(batch_data, key=lambda x: x['avg_ade'])
        best_batches = sorted_batches
    
    if not best_batches:
        print("No batch data available for export.")
        return False
    
    # Get the best batch (rank 1 - lowest ADE)
    best_batch = best_batches[0]
    
    print(f"\nExporting best batch data...")
    print(f"Best batch info: Dataset={best_batch['dataset']}, Step={best_batch['step']}")
    print(f"Average ADE: {best_batch['avg_ade']:.6f}, Average FDE: {best_batch['avg_fde']:.6f}")
    print(f"Number of pedestrians: {best_batch['num_pedestrians']}")
    
    # Extract trajectory data
    observed_list = best_batch['observed']  # List of [obs_len, 2] arrays
    ground_truth_list = best_batch['ground_truth']  # List of [pred_len, 2] arrays  
    predicted_list = best_batch['predicted']  # List of [pred_len, 2] arrays
    
    # Convert to numpy arrays for easier processing
    observed = np.stack(observed_list)  # [num_ped, obs_len, 2]
    ground_truth = np.stack(ground_truth_list)  # [num_ped, pred_len, 2]
    predicted = np.stack(predicted_list)  # [num_ped, pred_len, 2]
    
    num_peds, obs_len, _ = observed.shape
    pred_len = ground_truth.shape[1]
    total_len = obs_len + pred_len
    
    print(f"Data shapes: observed={observed.shape}, ground_truth={ground_truth.shape}, predicted={predicted.shape}")
    
    # Combine observed + ground_truth for the complete sequence (20 frames)
    complete_sequence = np.concatenate([observed, ground_truth], axis=1)  # [num_ped, total_len, 2]
    
    # Save paths
    best_txt_path = os.path.join(output_folder, 'best.txt')
    pred_txt_path = os.path.join(output_folder, 'pred.txt')
    
    # Write best.txt (observed + ground_truth sequence)
    print(f"\nSaving complete sequence (observed + ground_truth) to {best_txt_path}")
    with open(best_txt_path, 'w') as f:
        # f.write("# Complete trajectory sequence (observed + ground_truth)\n")
        # f.write("# Format: frame_idx\tped_idx\tx\ty\n")
        for ped_idx in range(num_peds):
            for frame_idx in range(total_len):
                x, y = complete_sequence[ped_idx, frame_idx]
                f.write(f"{frame_idx:02d}\t{ped_idx:02d}\t{x:.6f}\t{y:.6f}\n")
    
    # Write pred.txt (predicted sequence)
    print(f"Saving predicted sequence to {pred_txt_path}")
    with open(pred_txt_path, 'w') as f:
        # f.write("# Predicted trajectory sequence\n")
        # f.write("# Format: frame_idx\tped_idx\tx\ty\n")
        for ped_idx in range(num_peds):
            for frame_idx in range(pred_len):
                x, y = predicted[ped_idx, frame_idx]
                # Frame index starts from obs_len (8) to align with ground truth frames
                actual_frame_idx = frame_idx + obs_len
                f.write(f"{actual_frame_idx:02d}\t{ped_idx:02d}\t{x:.6f}\t{y:.6f}\n")
    
    print(f"\nExport completed successfully!")
    print(f"Files saved:")
    print(f"  - {best_txt_path}: Complete sequence ({total_len} frames per pedestrian)")
    print(f"  - {pred_txt_path}: Predicted sequence ({pred_len} frames per pedestrian)")
    print(f"  - Coordinate format: frame_idx, ped_idx, x, y")
    print(f"  - Frame indices: best.txt (0-{total_len-1}), pred.txt ({obs_len}-{obs_len+pred_len-1})")
    
    # Print summary statistics
    print(f"\nSummary statistics:")
    print(f"  - Number of pedestrians: {num_peds}")
    print(f"  - Observation length: {obs_len} frames")
    print(f"  - Prediction length: {pred_len} frames")
    print(f"  - Total sequence length: {total_len} frames")
    print(f"  - Best batch ADE: {best_batch['avg_ade']:.6f}")
    print(f"  - Best batch FDE: {best_batch['avg_fde']:.6f}")
    
    return True

def load_and_export_saved_batch_data():
    """
    Load and export saved best batch data without recomputing or plotting.
    """
    output_folder = 'best_batches'
    best_batches_file = os.path.join(output_folder, 'best_batches_data.pkl')
    
    if os.path.exists(best_batches_file):
        print("Found saved best batches data. Loading and exporting...")
        return export_best_batch_data(batch_data=None)
    else:
        print("No saved best batches data found.")
        return False

def test(KSTEPS=1, dataset='eth'):

    global loader_test, model, ROBUSTNESS
    threshold = static_dist_dict.get(dataset, 0.3)  # Default threshold if dataset not found
    model.eval()
    ade_bigls = []
    fde_bigls = []
    batch_data = []  # Store batch data with average ADE for plotting
    step = 0
    for batch in loader_test:
        step += 1
        #Get data
        batch = [tensor.cuda().double() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,loss_mask,V_obs,A_obs,V_tr,A_tr,_,_ = batch
        # obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,loss_mask,V_obs,A_obs,V_tr,A_tr,_,_ = batch
        ####
        # Filter trajectories based on movement distance between time steps 7 and 5 (indices 6 and 4)
        # obs_traj shape: [batch, num_ped, 2, obs_len]
        # Compute movement distance for each pedestrian
        movement_distances = torch.norm(obs_traj[:, :, :, 6] - obs_traj[:, :, :, 4], dim=2)  # [batch, num_ped]
        valid_peds = movement_distances > threshold  # [batch, num_ped]
        
        # Skip this batch if no pedestrians meet the movement criteria
        if not torch.any(valid_peds):
            continue
            
        seq = torch.cat((obs_traj, pred_traj_gt), dim=3)##[batch,num_ped,2,obs_len+pred_len]=[1, 5, 2, 20]
        seq = seq.permute(0,3,1,2)##[batch,obs_len+pred_len,num_ped,2]
        seq_ma = torch.zeros_like(seq)#[batch,obs_len+pred_len,num_ped,2]
        # seq_ma = Compute_ma.forward(seq)##
        # seq_ma = seq_ma[...,2:]
        # seq_ma = seq_ma.to(V_obs.device)
        ####
        seq_vis = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=3)

        num_of_objs = obs_traj_rel.shape[1]
        # print("num_of_objs:", num_of_objs)

        V_tr = V_tr.squeeze()

        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(
            V_obs.data.cpu().numpy().squeeze(), V_x[0, :, :].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(
            V_tr.data.cpu().numpy().squeeze(), V_x[-1, :, :].copy())

        for n in range(num_of_objs):
            ade_ls[n] = []
            fde_ls[n] = []
        # obs_ma = torch.zeros_like(V_obs)
        # obs_ma = obs_ma.to(V_obs.device)
        seq_ma = seq_ma.to(V_obs.device)
        # print("V_obs.shape:", V_obs.shape)##(1, 8, 3, 2)#[batch,obs_len,num_ped,2]
        # print("seq_ma.shape:", seq_ma.shape)##(1, 20, 3, 2)#[batch,obs_len+pred_len,num_ped,2]
        # print("obs_traj.shape:", obs_traj.shape)##(1, 3, 2, 8)#(batch,num_ped,2,obs_len)
        # print('A_obs.shape:', A_obs.shape)##(1, 8, 3, 2)#[batch,obs_len,num_ped,num_ped]

        V_predx,_,_ = model([V_obs.permute(0, 3, 1, 2),seq_ma.permute(0, 3, 1, 2),obs_traj], A_obs.squeeze(),KSTEPS=KSTEPS)
        # print("V_predx.shape:", V_predx.shape)##(KSTEPS, 2,T_pred, 2)
        # V_predx,_ = model(V_obs_tmp, obs_traj, KSTEPS=KSTEPS)

        for k in range(KSTEPS):
            V_pred = V_predx[k:k + 1, ...]

            V_pred = V_pred.permute(0, 2, 3, 1)

            V_pred = V_pred.squeeze()

            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(
                V_pred.data.cpu().numpy().squeeze(), V_x[-1, :, :].copy())
            #Sensitivity
            V_pred_rel_to_abs += ROBUSTNESS

            for n in range(num_of_objs):
                # Only process pedestrians that meet the movement criteria
                if not valid_peds[0, n].item():  # Skip if this pedestrian doesn't meet criteria
                    continue
                    
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n:n + 1, :])
                target.append(V_y_rel_to_abs[:, n:n + 1, :])
                obsrvs.append(V_x_rel_to_abs[:, n:n + 1, :])
                number_of.append(1)

                ade_ls[n].append(ade(pred, target, number_of))
                fde_ls[n].append(fde(pred, target, number_of))

        # Store best predictions for each pedestrian that meets criteria
        best_predictions = {}
        for n in range(num_of_objs):
            # Only process pedestrians that meet the movement criteria
            if not valid_peds[0, n].item() or len(ade_ls[n]) == 0:
                continue
                
            best_k = np.argmin(ade_ls[n])
            # Get the best prediction from the stored predictions
            V_pred = V_predx[best_k:best_k + 1, ...]
            V_pred = V_pred.permute(0, 2, 3, 1).squeeze()
            V_pred_rel_to_abs_best = nodes_rel_to_nodes_abs(
                V_pred.data.cpu().numpy().squeeze(), V_x[-1, :, :].copy())
            V_pred_rel_to_abs_best += ROBUSTNESS
            best_predictions[n] = V_pred_rel_to_abs_best
        
        # Collect batch-level data
        batch_ades = []
        batch_fdes = []
        batch_observed = []
        batch_predicted = []
        batch_ground_truth = []
        
        for n in range(num_of_objs):
            # Only process pedestrians that meet the movement criteria
            if not valid_peds[0, n].item() or len(ade_ls[n]) == 0 or n not in best_predictions:
                continue
                
            min_ade = min(ade_ls[n])
            min_fde = min(fde_ls[n])
            ade_bigls.append(min_ade)
            fde_bigls.append(min_fde)
            
            # Collect data for this pedestrian
            batch_ades.append(min_ade)
            batch_fdes.append(min_fde)
            batch_observed.append(V_x_rel_to_abs[:, n, :])  # [obs_len, 2]
            batch_ground_truth.append(V_y_rel_to_abs[:, n, :])  # [pred_len, 2]
            batch_predicted.append(best_predictions[n][:, n, :])  # [pred_len, 2]
        
        # Only store batch data if there are valid pedestrians
        if len(batch_ades) > 0:
            # Calculate average ADE and FDE for this batch
            avg_ade = np.mean(batch_ades)
            avg_fde = np.mean(batch_fdes)
            
            # Store batch data for visualization
            batch_info = {
                'avg_ade': avg_ade,
                'avg_fde': avg_fde,
                'pedestrian_ades': batch_ades,
                'pedestrian_fdes': batch_fdes,
                'observed': batch_observed,  # List of [obs_len, 2] arrays
                'ground_truth': batch_ground_truth,  # List of [pred_len, 2] arrays
                'predicted': batch_predicted,  # List of [pred_len, 2] arrays
                'step': step,
                'num_pedestrians': len(batch_ades)
            }
            batch_data.append(batch_info)

    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)
    return ade_, fde_, batch_data


for ROBUSTNESS in [0]:  #[-0.1, -0.01, 0, +0.01, +0.1]:
    print("*" * 30)
    print("*" * 30)
    print("ROBUSTNESS:", ROBUSTNESS)
    print("*" * 30)
    print("*" * 30)
    def seed_torch(seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    seed_torch()

    paths = [
        './checkpoint_deter/stcrf_eth',
    ]

    KSTEPS = 1## use this metric to evaluate the model

    EASY_RESULTS = []
    ALL_BATCH_DATA = []  # Collect all batch data for visualization

    print("*" * 50)
    print('Number of samples:', KSTEPS)
    print("*" * 50)

    for feta in range(len(paths)):

        ade_ls = []
        fde_ls = []
        exp_ls = []
        path = paths[feta]
        exps = glob.glob(path)
        exps.sort()
        print('Models being tested are:', exps)

        for exp_path in exps:
            print('exp_path:',exp_path)

            # try:

            print("*" * 50)
            print("Evaluating model:", exp_path)

            model_path = exp_path + '/val_best.pth'
            args_path = exp_path + '/args.pkl'
            with open(args_path, 'rb') as f:
                args = pickle.load(f)

            stats = exp_path + '/constant_metrics.pkl'
            with open(stats, 'rb') as f:
                cm = pickle.load(f)
            print("Stats:", cm)

            #Data prep
            obs_seq_len = args.obs_seq_len
            pred_seq_len = args.pred_seq_len
            # data_set = './datasets/' + args.dataset + '/'
            data_set = './datasets/' + 'univ' + '/'

            dset_test = TrajectoryDataset(data_set + 'test/',
                                          obs_len=obs_seq_len,
                                          pred_len=pred_seq_len,
                                          skip=1,
                                          norm_lap_matr=True)

            loader_test = DataLoader(
                dset_test,
                batch_size=
                1,  #This is irrelative to the args batch size parameter
                shuffle=False,
                num_workers=1)

            #Defining the model


            if args.dataset == 'eth':
                stgcn_layer = 0
            else: 
                stgcn_layer = 1
            model = STCRF(spatial_input=CFG["spatial_input"],
                       spatial_output=CFG["spatial_output"],
                       temporal_input=CFG["temporal_input"],
                       temporal_output=CFG["temporal_output"],
                       stgcn_layer=stgcn_layer).cuda().double()

            model.load_state_dict(torch.load(model_path))

            ################################################
            
            model.cuda().double()
            model.eval()
            ##############################
            ade_ = 999999
            fde_ = 999999
            print("Testing ....")
            ad, fd, batch_data_list = test(KSTEPS=KSTEPS, dataset=args.dataset)
            ade_ = min(ade_, ad)
            fde_ = min(fde_, fd)
            ade_ls.append(ade_)
            fde_ls.append(fde_)
            exp_ls.append(exp_path)
            
            # Add dataset info to batch data and collect
            for batch_info in batch_data_list:
                batch_info['dataset'] = args.dataset
                batch_info['model_path'] = exp_path
            ALL_BATCH_DATA.extend(batch_data_list)
            
            print("ADE:", ade_, " FDE:", fde_)
        # except Exception as e:
        #     print(e)
        print("*" * 50)
        if len(ade_ls) > 0:  # Check if we have results
            ade_ls = np.asarray(ade_ls)
            fde_ls = np.asarray(fde_ls)
            min_ade_indx = np.argmin(ade_ls)
            min_fde_indx = np.argmin(fde_ls)
            avg_ade_fde = (ade_ls + fde_ls) / 2.0
            min_avg_ade_fde = np.argmin(avg_ade_fde)

            EASY_RESULTS.append([
                exp_ls[min_avg_ade_fde],
                round(ade_ls[min_avg_ade_fde], 4),
                round(fde_ls[min_avg_ade_fde], 4)
            ])
        else:
            print(f"No valid models found for dataset {feta}")
    print(EASY_RESULTS)
    if len(EASY_RESULTS) > 0:
        ade_mean = np.mean([x[1] for x in EASY_RESULTS])
        fde_mean = np.mean([x[2] for x in EASY_RESULTS])
        print("Mean ADE:", ade_mean)
        print("Mean FDE:", fde_mean)
    else:
        print("No results to compute mean values")
    
    # Check if saved best batches data exists
    output_folder = 'best_batches'
    best_batches_file = os.path.join(output_folder, 'best_batches_data.pkl')
    
    # Check for command line arguments
    force_recompute = '--force-recompute' in sys.argv
    use_saved = '--use-saved' in sys.argv
    export_data = '--export-data' in sys.argv
    
    if os.path.exists(best_batches_file) and not force_recompute:
        print("\n" + "="*60)
        print("FOUND SAVED BEST BATCHES DATA!")
        print("="*60)
        
        if use_saved:
            print("Using saved best batches data (--use-saved flag)...")
            load_and_plot_saved_batches(num_batches=10)
            if export_data:
                print("\nExporting best batch data (--export-data flag)...")
                export_best_batch_data(batch_data=None)
        else:
            # Interactive mode
            print("Usage: python test_vis.py [--use-saved] [--force-recompute] [--export-data]")
            print("  --use-saved: automatically use saved data")
            print("  --force-recompute: force recomputation even if saved data exists")
            print("  --export-data: export best batch data to txt files")
            response = input("Do you want to use saved data (y) or recompute (n)? [y/n]: ").lower().strip()
            
            if response == 'y' or response == 'yes' or response == '':
                print("Using saved best batches data...")
                load_and_plot_saved_batches(num_batches=10)
                
                # Ask if user wants to export data
                export_response = input("Do you want to export best batch data to txt files (y/n)? [y/n]: ").lower().strip()
                if export_response == 'y' or export_response == 'yes':
                    print("Exporting best batch data...")
                    export_best_batch_data(batch_data=None)
            else:
                print("Recomputing and plotting best batches...")
                if ALL_BATCH_DATA:
                    print(f"\nTotal batches collected: {len(ALL_BATCH_DATA)}")
                    plot_best_batches(ALL_BATCH_DATA, num_batches=10)
                    
                    # Export data from newly computed results
                    print("Exporting best batch data from computed results...")
                    export_best_batch_data(ALL_BATCH_DATA)
                else:
                    print("No batch data collected for visualization.")
    else:
        if force_recompute:
            print("\n" + "="*60)
            print("FORCE RECOMPUTING BEST BATCHES (--force-recompute flag)")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("NO SAVED DATA FOUND - COMPUTING BEST BATCHES")
            print("="*60)
        
        # Plot the 10 most accurate batches
        if ALL_BATCH_DATA:
            print(f"\nTotal batches collected: {len(ALL_BATCH_DATA)}")
            plot_best_batches(ALL_BATCH_DATA, num_batches=10)
            
            # Export data from newly computed results
            print("Exporting best batch data from computed results...")
            export_best_batch_data(ALL_BATCH_DATA)
        else:
            print("No batch data collected for visualization.")

# Standalone function to quickly plot saved batches
def plot_saved_batches_only():
    """
    Quick function to plot only saved batches without any model evaluation.
    """
    print("="*60)
    print("PLOTTING SAVED BATCHES ONLY")
    print("="*60)
    
    success = load_and_plot_saved_batches(num_batches=10)
    if not success:
        print("No saved batches found. Please run the full evaluation first.")
        print("Usage: python test_vis.py  # to run full evaluation")
        print("       python test_vis.py --use-saved  # to use saved data")
        print("       python test_vis.py --force-recompute  # to force recomputation")
        print("       python test_vis.py --export-data  # to export best batch data")

def export_saved_batches_only():
    """
    Quick function to export only saved batch data without any model evaluation or plotting.
    """
    print("="*60)
    print("EXPORTING SAVED BATCH DATA ONLY")
    print("="*60)
    
    success = load_and_export_saved_batch_data()
    if not success:
        print("No saved batch data found. Please run the full evaluation first.")
        print("Usage: python test_vis.py  # to run full evaluation")
        print("       python test_vis.py --use-saved  # to use saved data")
        print("       python test_vis.py --export-only  # to export data only")

if __name__ == "__main__":
    # Check if user wants to plot saved batches only
    print(sys.argv)
    if len(sys.argv) > 1 and sys.argv[1] == '--plot-only':
        plot_saved_batches_only()
        sys.exit(0)
    
    # Check if user wants to export saved batch data only
    if len(sys.argv) > 1 and sys.argv[1] == '--export-only':
        export_saved_batches_only()
        sys.exit(0)
    
    # Print usage information
    if '--help' in sys.argv or '-h' in sys.argv:
        print("Usage: python test_vis.py [options]")
        print("Options:")
        print("  --use-saved         Use saved best batches data automatically")
        print("  --force-recompute   Force recomputation even if saved data exists")
        print("  --export-data       Export best batch data to txt files")
        print("  --plot-only         Only plot saved batches (no model evaluation)")
        print("  --export-only       Only export saved batch data (no evaluation/plotting)")
        print("  --help, -h          Show this help message")
        print("")
        print("File outputs:")
        print("  best_batches/best.txt      Complete trajectory (observed + ground truth)")
        print("  best_batches/pred.txt      Predicted trajectory")
        print("  best_batches/*.png         Batch visualization plots")
        sys.exit(0)
    
    # Run the main execution logic
    print("Starting batch evaluation and visualization...")
    print("Note: Use --help to see available options")
    print("="*60)

    # Main execution code moved inside if __name__ == "__main__" block

"""
MODIFICATIONS SUMMARY:
======================

This script has been enhanced with the following features:

1. **Batch-level analysis**: Computes average ADE for all pedestrians within each batch 
   and selects the 10 best batches (lowest average ADE). Data is saved to 
   'best_batches/best_batches_data.pkl'

2. **Multi-pedestrian visualization**: Each batch is visualized as a single plot showing 
   all pedestrians in that batch with different colors. Different batches are shown 
   in separate plots.

3. **Smart loading**: The script checks for existing saved data and offers options:
   - Interactive mode: Prompts user to choose between saved data or recomputation
   - Command-line options for automation

4. **Command-line options**:
   - `python test_vis.py` : Normal run with interactive prompts
   - `python test_vis.py --use-saved` : Automatically use saved data
   - `python test_vis.py --force-recompute` : Force recomputation even if saved data exists
   - `python test_vis.py --export-data` : Export best batch data to txt files
   - `python test_vis.py --plot-only` : Only plot saved data (no model evaluation)
   - `python test_vis.py --export-only` : Only export saved data (no evaluation/plotting)
   - `python test_vis.py --help` : Show help message

5. **Functions added**:
   - `plot_best_batches()`: Plot best batches with all pedestrians per batch
   - `load_and_plot_saved_batches()`: Load and plot saved batch data
   - `plot_saved_batches_only()`: Quick plotting without evaluation
   - `export_best_batch_data()`: Export best batch data to txt files
   - `load_and_export_saved_batch_data()`: Load and export saved batch data
   - `export_saved_batches_only()`: Quick export without evaluation/plotting

6. **Data structure**: Each batch contains:
   - Average ADE/FDE for the batch
   - Individual ADE/FDE for each pedestrian
   - Observed, predicted, and ground truth trajectories for all pedestrians
   - Batch metadata (step, dataset, number of pedestrians)

7. **Performance benefits**:
   - Subsequent runs are much faster when using saved data
   - No need to re-evaluate models if you just want to regenerate plots
   - Persistent storage of best batch data for analysis
   - Export trajectory data for use with other models

8. **Data export features**:
   - Exports best batch's observed+ground_truth to `best_batches/best.txt`
   - Exports corresponding predictions to `best_batches/pred.txt`
   - Format: frame_idx, ped_idx, x, y (tab-separated)
   - Ready for use with other trajectory prediction models

Usage examples:
- First run: `python test_vis.py` (computes, saves, and exports best batches)
- Later plotting: `python test_vis.py --use-saved` (fast plotting from saved data)
- Later export: `python test_vis.py --export-only` (fast export from saved data)
- Force new computation: `python test_vis.py --force-recompute`
- Export with plotting: `python test_vis.py --use-saved --export-data`
"""