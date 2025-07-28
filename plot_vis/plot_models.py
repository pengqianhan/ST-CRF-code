#!/usr/bin/env python3
"""
Plot trajectory predictions for STCRF, Social-Implicit, and Social-STGCNN models
on ETH/UCY datasets for paper visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

def parse_trajectory_file(file_path):
    """Parse trajectory file and return data organized by pedestrian ID."""
    trajectories = defaultdict(list)
    
    if not os.path.exists(file_path):
        return trajectories
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 4:
                # Format: step->frame, ped_id, x, y
                step_frame = parts[0].strip()
                ped_id = parts[1].strip()
                x = float(parts[2])
                y = float(parts[3])
                
                # Extract frame number
                if '->' in step_frame:
                    frame = int(step_frame.split('->')[1])
                else:
                    frame = int(step_frame)
                
                trajectories[ped_id].append((frame, x, y))
    
    # Sort by frame number for each pedestrian
    for ped_id in trajectories:
        trajectories[ped_id].sort(key=lambda x: x[0])
    
    return trajectories

def plot_dataset_comparison(dataset_name):
    """Plot comparison for a single dataset."""
    
    # File paths
    stcrf_best = f"stcrf/{dataset_name}/best.txt"
    stcrf_pred = f"stcrf/{dataset_name}/pred.txt"
    social_implicit_pred = f"social_implicit/{dataset_name}/pred.txt"
    social_stgcnn_pred = f"social_stgcnn/{dataset_name}/pred.txt"
    
    # Parse trajectories
    gt_trajectories = parse_trajectory_file(stcrf_best)
    stcrf_trajectories = parse_trajectory_file(stcrf_pred)
    social_implicit_trajectories = parse_trajectory_file(social_implicit_pred)
    social_stgcnn_trajectories = parse_trajectory_file(social_stgcnn_pred)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Get all pedestrian IDs
    all_peds = set(gt_trajectories.keys())
    all_peds.update(stcrf_trajectories.keys())
    all_peds.update(social_implicit_trajectories.keys())
    all_peds.update(social_stgcnn_trajectories.keys())
    
    # Filter pedestrians for hotel and zara2 - only show upper pedestrians
    if dataset_name in ['cross_hotel', 'cross_zara2']:
        # Find pedestrians with higher y coordinates (upper pedestrians)
        ped_y_coords = {}
        for ped_id in all_peds:
            if ped_id in gt_trajectories and gt_trajectories[ped_id]:
                # Use the first coordinate to determine position
                ped_y_coords[ped_id] = gt_trajectories[ped_id][0][2]  # y coordinate
        
        if ped_y_coords:
            # Find the median y coordinate
            y_values = list(ped_y_coords.values())
            median_y = sorted(y_values)[len(y_values) // 2]
            # Keep only pedestrians above median (higher y values)
            all_peds = {ped for ped in all_peds if ped_y_coords.get(ped, float('-inf')) >= median_y}
    
    # Define colors for trajectories
    observed_color = '#8B008B'      # Purple for observed
    future_color = '#FF0000'        # Red for future ground truth
    
    # Define colors for different models
    model_colors = {
        'stcrf': '#00FF00',         # Green
        'social_implicit': '#FFFF00',  # Yellow
        'social_stgcnn': '#0000FF'     # Blue
    }
    
    for ped_id in sorted(all_peds):
        
        # Plot ground truth (observed + future)
        if ped_id in gt_trajectories:
            gt_traj = gt_trajectories[ped_id]
            frames = [t[0] for t in gt_traj]
            x_coords = [t[1] for t in gt_traj]
            y_coords = [t[2] for t in gt_traj]
            
            # Split observed (frame 0-7) and future (frame 8-19) trajectories
            observed_indices = [i for i, f in enumerate(frames) if f <= 7]
            future_indices = [i for i, f in enumerate(frames) if f >= 8]
            
            # Plot observed trajectory (solid line)
            if observed_indices:
                obs_x = [x_coords[i] for i in observed_indices]
                obs_y = [y_coords[i] for i in observed_indices]
                plt.plot(obs_x, obs_y, 'o-', color=observed_color, linewidth=2, 
                        markersize=4, label=f'Observed' if ped_id == sorted(all_peds)[0] else "")
            
            # Plot future trajectory (dashed line)
            if future_indices:
                fut_x = [x_coords[i] for i in future_indices]
                fut_y = [y_coords[i] for i in future_indices]
                # Connect last observed with first future
                if observed_indices and future_indices:
                    plt.plot([obs_x[-1], fut_x[0]], [obs_y[-1], fut_y[0]], 
                            '--', color=future_color, linewidth=2, alpha=0.8)
                plt.plot(fut_x, fut_y, 's--', color=future_color, linewidth=2, 
                        markersize=4, alpha=0.8, 
                        label=f'Ground Truth Future' if ped_id == sorted(all_peds)[0] else "")
        
        # Plot STCRF predictions
        if ped_id in stcrf_trajectories:
            traj = stcrf_trajectories[ped_id]
            x_coords = [t[1] for t in traj]
            y_coords = [t[2] for t in traj]
            plt.plot(x_coords, y_coords, '^-', color=model_colors['stcrf'], linewidth=2, 
                    markersize=4, alpha=0.8,
                    label=f'ST-CRF' if ped_id == sorted(all_peds)[0] else "")
        
        # Plot Social-Implicit predictions
        if ped_id in social_implicit_trajectories:
            traj = social_implicit_trajectories[ped_id]
            x_coords = [t[1] for t in traj]
            y_coords = [t[2] for t in traj]
            plt.plot(x_coords, y_coords, 'v-', color=model_colors['social_implicit'], linewidth=2, 
                    markersize=4, alpha=0.8,
                    label=f'Social-Implicit' if ped_id == sorted(all_peds)[0] else "")
        
        # Plot Social-STGCNN predictions
        if ped_id in social_stgcnn_trajectories:
            traj = social_stgcnn_trajectories[ped_id]
            x_coords = [t[1] for t in traj]
            y_coords = [t[2] for t in traj]
            plt.plot(x_coords, y_coords, 'd-', color=model_colors['social_stgcnn'], linewidth=2, 
                    markersize=4, alpha=0.8,
                    label=f'Social-STGCNN' if ped_id == sorted(all_peds)[0] else "")
    
    # plt.xlabel('X Position (m)', fontsize=12)
    # plt.ylabel('Y Position (m)', fontsize=12)
    plt.title(f'Trajectory Prediction Comparison - {dataset_name.replace("cross_", "").title()}', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.axis('equal')
    
    # Save figure
    output_file = f"{dataset_name}_comparison.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.show()

def plot_all_datasets_combined():
    """Plot all four datasets in a single figure with subplots."""
    datasets = ['cross_hotel', 'cross_univ', 'cross_zara1', 'cross_zara2']
    
    # Create figure with subplots in a single row
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    
    for idx, dataset_name in enumerate(datasets):
        ax = axes[idx]
        plt.sca(ax)  # Set current axis
        
        # File paths
        stcrf_best = f"stcrf/{dataset_name}/best.txt"
        stcrf_pred = f"stcrf/{dataset_name}/pred.txt"
        social_implicit_pred = f"social_implicit/{dataset_name}/pred.txt"
        social_stgcnn_pred = f"social_stgcnn/{dataset_name}/pred.txt"
        
        # Parse trajectories
        gt_trajectories = parse_trajectory_file(stcrf_best)
        stcrf_trajectories = parse_trajectory_file(stcrf_pred)
        social_implicit_trajectories = parse_trajectory_file(social_implicit_pred)
        social_stgcnn_trajectories = parse_trajectory_file(social_stgcnn_pred)
        
        # Get all pedestrian IDs
        all_peds = set(gt_trajectories.keys())
        all_peds.update(stcrf_trajectories.keys())
        all_peds.update(social_implicit_trajectories.keys())
        all_peds.update(social_stgcnn_trajectories.keys())
        
        # Filter pedestrians for hotel and zara2 - only show upper pedestrians
        if dataset_name in ['cross_hotel', 'cross_zara2']:
            # Find pedestrians with higher y coordinates (upper pedestrians)
            ped_y_coords = {}
            for ped_id in all_peds:
                if ped_id in gt_trajectories and gt_trajectories[ped_id]:
                    # Use the first coordinate to determine position
                    ped_y_coords[ped_id] = gt_trajectories[ped_id][0][2]  # y coordinate
            
            if ped_y_coords:
                # Find the median y coordinate
                y_values = list(ped_y_coords.values())
                median_y = sorted(y_values)[len(y_values) // 2]
                # Keep only pedestrians above median (higher y values)
                all_peds = {ped for ped in all_peds if ped_y_coords.get(ped, float('-inf')) >= median_y}
        
        # Define colors for trajectories
        observed_color = '#8B008B'      # Purple for observed
        future_color = '#FF0000'        # Red for future ground truth
        
        # Define colors for different models
        model_colors = {
            'stcrf': '#00FF00',         # Green
            'social_implicit': '#FFFF00',  # Yellow
            'social_stgcnn': '#0000FF'     # Blue
        }
        
        # Track if we've added labels for this subplot
        labels_added = {'observed': False, 'future': False, 'stcrf': False, 'social_implicit': False, 'social_stgcnn': False}
        
        for ped_id in sorted(all_peds):
            
            # Plot ground truth (observed + future)
            if ped_id in gt_trajectories:
                gt_traj = gt_trajectories[ped_id]
                frames = [t[0] for t in gt_traj]
                x_coords = [t[1] for t in gt_traj]
                y_coords = [t[2] for t in gt_traj]
                
                # Split observed (frame 0-7) and future (frame 8-19) trajectories
                observed_indices = [i for i, f in enumerate(frames) if f <= 7]
                future_indices = [i for i, f in enumerate(frames) if f >= 8]
                
                # Plot observed trajectory (solid line)
                if observed_indices:
                    obs_x = [x_coords[i] for i in observed_indices]
                    obs_y = [y_coords[i] for i in observed_indices]
                    ax.plot(obs_x, obs_y, 'o-', color=observed_color, linewidth=3, 
                            markersize=6, label='Observed' if not labels_added['observed'] else "")
                    labels_added['observed'] = True
                
                # Plot future trajectory (dashed line)
                if future_indices:
                    fut_x = [x_coords[i] for i in future_indices]
                    fut_y = [y_coords[i] for i in future_indices]
                    # Connect last observed with first future
                    if observed_indices and future_indices:
                        ax.plot([obs_x[-1], fut_x[0]], [obs_y[-1], fut_y[0]], 
                                '--', color=future_color, linewidth=3, alpha=0.8)
                    ax.plot(fut_x, fut_y, 's--', color=future_color, linewidth=3, 
                            markersize=6, alpha=0.8, 
                            label='Ground Truth Future' if not labels_added['future'] else "")
                    labels_added['future'] = True
            
            # Plot STCRF predictions
            if ped_id in stcrf_trajectories:
                traj = stcrf_trajectories[ped_id]
                x_coords = [t[1] for t in traj]
                y_coords = [t[2] for t in traj]
                ax.plot(x_coords, y_coords, '^-', color=model_colors['stcrf'], linewidth=3, 
                        markersize=6, alpha=0.8,
                        label='ST-CRF' if not labels_added['stcrf'] else "")
                labels_added['stcrf'] = True
            
            # Plot Social-Implicit predictions
            if ped_id in social_implicit_trajectories:
                traj = social_implicit_trajectories[ped_id]
                x_coords = [t[1] for t in traj]
                y_coords = [t[2] for t in traj]
                ax.plot(x_coords, y_coords, 'v-', color=model_colors['social_implicit'], linewidth=3, 
                        markersize=6, alpha=0.8,
                        label='Social-Implicit' if not labels_added['social_implicit'] else "")
                labels_added['social_implicit'] = True
            
            # Plot Social-STGCNN predictions
            if ped_id in social_stgcnn_trajectories:
                traj = social_stgcnn_trajectories[ped_id]
                x_coords = [t[1] for t in traj]
                y_coords = [t[2] for t in traj]
                ax.plot(x_coords, y_coords, 'd-', color=model_colors['social_stgcnn'], linewidth=3, 
                        markersize=6, alpha=0.8,
                        label='Social-STGCNN' if not labels_added['social_stgcnn'] else "")
                labels_added['social_stgcnn'] = True
        
        # ax.set_xlabel('X Position (m)', fontsize=10)
        # ax.set_ylabel('Y Position (m)', fontsize=10)
        ax.set_title(f'{dataset_name.replace("cross_", "").title()}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Let matplotlib auto-scale the axes for each dataset with tight margins
        ax.relim()
        ax.autoscale_view(tight=True)
        
        # Add small margins around the data
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_margin = (xlim[1] - xlim[0]) * 0.05
        y_margin = (ylim[1] - ylim[0]) * 0.05
        ax.set_xlim(xlim[0] - x_margin, xlim[1] + x_margin)
        ax.set_ylim(ylim[0] - y_margin, ylim[1] + y_margin)
        
        ax.set_aspect('equal')
    
    # Create a single legend outside the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Make room for legend
    plt.savefig('all_datasets_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: all_datasets_comparison.png")
    plt.show()

def main():
    """Main function to generate all plots."""
    datasets = ['cross_hotel', 'cross_univ', 'cross_zara1', 'cross_zara2']
    
    print("Generating trajectory comparison plots for paper...")
    
    # Generate individual plots
    for dataset in datasets:
        print(f"\nProcessing {dataset}...")
        plot_dataset_comparison(dataset)
    
    # Generate combined plot
    print(f"\nGenerating combined plot...")
    plot_all_datasets_combined()
    
    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    main()