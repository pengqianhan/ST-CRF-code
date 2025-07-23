#!/usr/bin/env python3
"""
Simple script to visualize exported trajectory data from test_vis.py

This script reads the exported best.txt and pred.txt files and creates
a visualization showing the observed, ground truth, and predicted trajectories.

Usage:
    python plot_exported_data.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_exported_trajectories(best_txt_path='best_batches/best.txt', 
                              pred_txt_path='best_batches/pred.txt',
                              output_path='best_batches/exported_trajectories_plot.png'):
    """
    Plot trajectories from exported txt files.
    
    Args:
        best_txt_path: Path to best.txt (observed + ground truth)
        pred_txt_path: Path to pred.txt (predicted trajectories)
        output_path: Path to save the plot
    """
    
    # Check if files exist
    if not os.path.exists(best_txt_path):
        print(f"Error: {best_txt_path} not found. Please run test_vis.py with export first.")
        return False
        
    if not os.path.exists(pred_txt_path):
        print(f"Error: {pred_txt_path} not found. Please run test_vis.py with export first.")
        return False
    
    print(f"Loading trajectory data from:")
    print(f"  - {best_txt_path}")
    print(f"  - {pred_txt_path}")
    
    # Load data (skip comment lines starting with #)
    best_data = np.loadtxt(best_txt_path, skiprows=2)  # frame_idx, ped_idx, x, y
    pred_data = np.loadtxt(pred_txt_path, skiprows=2)  # frame_idx, ped_idx, x, y
    
    # Extract unique pedestrians
    pedestrians = np.unique(best_data[:, 1]).astype(int)
    num_peds = len(pedestrians)
    
    print(f"Found {num_peds} pedestrians in the data")
    
    # Determine observation and prediction lengths
    all_frames = np.unique(best_data[:, 0]).astype(int)
    pred_frames = np.unique(pred_data[:, 0]).astype(int)
    obs_len = len(all_frames) - len(pred_frames)
    pred_len = len(pred_frames)
    total_len = len(all_frames)
    
    print(f"Sequence info: {obs_len} observed + {pred_len} predicted = {total_len} total frames")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define colors for different pedestrians
    colors = plt.cm.Set3(np.linspace(0, 1, max(12, num_peds)))
    
    # Plot each pedestrian
    for i, ped_id in enumerate(pedestrians):
        color = colors[i % len(colors)]
        
        # Get complete trajectory (observed + ground truth)
        ped_mask = best_data[:, 1] == ped_id
        ped_traj = best_data[ped_mask]
        ped_traj = ped_traj[np.argsort(ped_traj[:, 0])]  # Sort by frame
        
        # Split into observed and ground truth parts
        obs_frames = ped_traj[:obs_len, :]  # First obs_len frames
        gt_frames = ped_traj[obs_len:, :]   # Remaining frames
        
        # Get predicted trajectory
        pred_mask = pred_data[:, 1] == ped_id
        pred_traj = pred_data[pred_mask]
        pred_traj = pred_traj[np.argsort(pred_traj[:, 0])]  # Sort by frame
        
        # Plot observed trajectory (historical)
        if len(obs_frames) > 0:
            ax.plot(obs_frames[:, 2], obs_frames[:, 3], 
                   color=color, linewidth=3, alpha=0.9,
                   label=f'Ped {int(ped_id):02d} - Observed', 
                   marker='o', markersize=4)
        
        # Plot ground truth trajectory
        if len(gt_frames) > 0:
            ax.plot(gt_frames[:, 2], gt_frames[:, 3], 
                   color=color, linewidth=2, alpha=0.8, linestyle=':',
                   label=f'Ped {int(ped_id):02d} - Ground Truth', 
                   marker='^', markersize=4)
        
        # Plot predicted trajectory
        if len(pred_traj) > 0:
            ax.plot(pred_traj[:, 2], pred_traj[:, 3], 
                   color=color, linewidth=3, alpha=0.9, linestyle='--',
                   label=f'Ped {int(ped_id):02d} - Predicted', 
                   marker='s', markersize=4)
        
        # Mark start and end points
        if len(obs_frames) > 0:
            ax.scatter(obs_frames[0, 2], obs_frames[0, 3], 
                      color=color, s=100, marker='o', 
                      alpha=1.0, edgecolors='black', linewidth=2)
        
        if len(gt_frames) > 0:
            ax.scatter(gt_frames[-1, 2], gt_frames[-1, 3], 
                      color=color, s=100, marker='^', 
                      alpha=1.0, edgecolors='black', linewidth=2)
        
        if len(pred_traj) > 0:
            ax.scatter(pred_traj[-1, 2], pred_traj[-1, 3], 
                      color=color, s=100, marker='s', 
                      alpha=1.0, edgecolors='black', linewidth=2)
    
    # Add legend explanation for trajectory types
    ax.plot([], [], color='gray', linewidth=3, alpha=0.9, 
           label='Observed (Historical)', marker='o', markersize=4)
    ax.plot([], [], color='gray', linewidth=2, alpha=0.8, linestyle=':', 
           label='Ground Truth', marker='^', markersize=4)
    ax.plot([], [], color='gray', linewidth=3, alpha=0.9, linestyle='--', 
           label='Predicted', marker='s', markersize=4)
    
    # Set labels and title
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    ax.set_title(f'Exported Trajectory Data Visualization\n'
                f'Pedestrians: {num_peds}, Frames: {obs_len} obs + {pred_len} pred = {total_len} total', 
                fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Create a compact legend
    handles, labels = ax.get_legend_handles_labels()
    # Show only pedestrian-specific labels for a few pedestrians to avoid clutter
    max_ped_labels = min(3, num_peds)
    ped_handles = handles[:max_ped_labels*3]  # 3 lines per pedestrian
    ped_labels = labels[:max_ped_labels*3]
    type_handles = handles[-3:]  # Observed, Ground Truth, Predicted
    type_labels = labels[-3:]
    
    if num_peds > max_ped_labels:
        ped_labels = ped_labels[:3] + [f'... and {(num_peds - max_ped_labels)*3} more trajectories']
        ped_handles = ped_handles[:3] + [ped_handles[3]]
    
    all_handles = ped_handles + type_handles
    all_labels = ped_labels + type_labels
    ax.legend(all_handles, all_labels, fontsize=9, loc='upper right', 
             bbox_to_anchor=(1.0, 1.0))
    
    # Make axes equal for better visualization
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.show()
    
    return True

def print_data_summary(best_txt_path='best_batches/best.txt', 
                      pred_txt_path='best_batches/pred.txt'):
    """Print summary statistics of the exported data."""
    
    if not os.path.exists(best_txt_path) or not os.path.exists(pred_txt_path):
        print("Data files not found. Please export data first.")
        return
    
    # Load data
    best_data = np.loadtxt(best_txt_path, skiprows=2)
    pred_data = np.loadtxt(pred_txt_path, skiprows=2)
    
    # Extract information
    pedestrians = np.unique(best_data[:, 1])
    frames_best = np.unique(best_data[:, 0])
    frames_pred = np.unique(pred_data[:, 0])
    
    print("\n" + "="*50)
    print("EXPORTED DATA SUMMARY")
    print("="*50)
    print(f"Files:")
    print(f"  - {best_txt_path}")
    print(f"  - {pred_txt_path}")
    print(f"\nData structure:")
    print(f"  - Number of pedestrians: {len(pedestrians)}")
    print(f"  - Best.txt frames: {len(frames_best)} (frames {int(frames_best.min())}-{int(frames_best.max())})")
    print(f"  - Pred.txt frames: {len(frames_pred)} (frames {int(frames_pred.min())}-{int(frames_pred.max())})")
    print(f"  - Observation length: {len(frames_best) - len(frames_pred)} frames")
    print(f"  - Prediction length: {len(frames_pred)} frames")
    print(f"\nPedestrian IDs: {[int(p) for p in pedestrians]}")
    
    # Show sample data
    print(f"\nSample data from {best_txt_path}:")
    print("frame_idx\tped_idx\tx\ty")
    for i in range(min(5, len(best_data))):
        frame, ped, x, y = best_data[i]
        print(f"{int(frame):02d}\t{int(ped):02d}\t{x:.6f}\t{y:.6f}")
    
    if len(best_data) > 5:
        print("...")
    
    print(f"\nSample data from {pred_txt_path}:")
    print("frame_idx\tped_idx\tx\ty")  
    for i in range(min(5, len(pred_data))):
        frame, ped, x, y = pred_data[i]
        print(f"{int(frame):02d}\t{int(ped):02d}\t{x:.6f}\t{y:.6f}")
    
    if len(pred_data) > 5:
        print("...")

if __name__ == "__main__":
    print("Trajectory Data Visualization Script")
    print("====================================")
    
    # Print data summary
    print_data_summary()
    
    # Create visualization
    success = plot_exported_trajectories()
    
    if success:
        print("\nVisualization completed successfully!")
        print("You can now use the exported txt files with other models.")
    else:
        print("\nVisualization failed. Please check that the data files exist.")
        print("Run 'python test_vis.py --export-only' to generate the data files first.") 