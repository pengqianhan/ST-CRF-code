#!/usr/bin/env python3
"""
Plot trajectory predictions for STCRF, Social-Implicit, and Social-STGCNN models
on ETH/UCY datasets for paper visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_trajectory_file(file_path):
    """Parse trajectory file and return data organized by pedestrian ID."""
    trajectories = defaultdict(list)
    
    # Convert to absolute path relative to script directory
    abs_path = os.path.join(SCRIPT_DIR, file_path)
    if not os.path.exists(abs_path):
        print(f"Warning: File not found: {abs_path}")
        return trajectories
    
    file_path = abs_path
    
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
    # plt.axis('equal')  # Remove equal aspect to prevent distortion
    
    # Save figure in the script directory
    output_file = os.path.join(SCRIPT_DIR, f"{dataset_name}_comparison.pdf")
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.show()

def plot_all_datasets_combined():
    """Plot all four datasets in a single figure with subplots."""
    datasets = ['cross_hotel', 'cross_univ', 'cross_zara1', 'cross_zara2']
    
    # Create figure with subplots in a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(3.36, 3.3))  # IEEE Access column width, so fonts print at true size
    axes = axes.flatten()
    
    
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
            'stcrf': '#1B9E3A',         # Green
            'social_implicit': '#E69F00',  # Orange (yellow is unreadable on white)
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
                    ax.plot(obs_x, obs_y, 'o-', color=observed_color, linewidth=1.2, 
                            markersize=2.5, label='Observed' if not labels_added['observed'] else "")
                    labels_added['observed'] = True
                
                # Plot future trajectory (dashed line)
                if future_indices:
                    fut_x = [x_coords[i] for i in future_indices]
                    fut_y = [y_coords[i] for i in future_indices]
                    # Connect last observed with first future
                    if observed_indices and future_indices:
                        ax.plot([obs_x[-1], fut_x[0]], [obs_y[-1], fut_y[0]], 
                                '--', color=future_color, linewidth=1.2, alpha=0.8)
                    ax.plot(fut_x, fut_y, 's--', color=future_color, linewidth=1.2, 
                            markersize=2.5, alpha=0.8, 
                            label='Ground Truth Future' if not labels_added['future'] else "")
                    labels_added['future'] = True
            
            # Plot STCRF predictions
            if ped_id in stcrf_trajectories:
                traj = stcrf_trajectories[ped_id]
                x_coords = [t[1] for t in traj]
                y_coords = [t[2] for t in traj]
                ax.plot(x_coords, y_coords, '^-', color=model_colors['stcrf'], linewidth=1.2, 
                        markersize=2.5, alpha=0.8,
                        label='ST-CRF' if not labels_added['stcrf'] else "")
                labels_added['stcrf'] = True
            
            # Plot Social-Implicit predictions
            if ped_id in social_implicit_trajectories:
                traj = social_implicit_trajectories[ped_id]
                x_coords = [t[1] for t in traj]
                y_coords = [t[2] for t in traj]
                ax.plot(x_coords, y_coords, 'v-', color=model_colors['social_implicit'], linewidth=1.2, 
                        markersize=2.5, alpha=0.8,
                        label='Social-Implicit' if not labels_added['social_implicit'] else "")
                labels_added['social_implicit'] = True
            
            # Plot Social-STGCNN predictions
            if ped_id in social_stgcnn_trajectories:
                traj = social_stgcnn_trajectories[ped_id]
                x_coords = [t[1] for t in traj]
                y_coords = [t[2] for t in traj]
                ax.plot(x_coords, y_coords, 'd-', color=model_colors['social_stgcnn'], linewidth=1.2, 
                        markersize=2.5, alpha=0.8,
                        label='Social-STGCNN' if not labels_added['social_stgcnn'] else "")
                labels_added['social_stgcnn'] = True
        
        # ax.set_xlabel('X Position (m)', fontsize=10)
        # ax.set_ylabel('Y Position (m)', fontsize=10)
        ax.set_title(f'{dataset_name.replace("cross_", "").title()}', fontsize=8, pad=2)
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.tick_params(labelsize=6, length=2, width=0.5, pad=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        
        # Let matplotlib auto-scale the axes for each dataset with tight margins
        ax.relim()
        ax.autoscale_view(tight=True)
        
        # Add small margins around the data
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_margin = (xlim[1] - xlim[0]) * 0.1
        y_margin = (ylim[1] - ylim[0]) * 0.1
        ax.set_xlim(xlim[0] - x_margin, xlim[1] + x_margin)
        ax.set_ylim(ylim[0] - y_margin, ylim[1] + y_margin)
        
        # Set a reasonable aspect ratio that doesn't distort the visualization
        # Allow some flexibility rather than enforcing equal aspect
        ax.set_aspect('auto')
    
    # Create a single legend outside the subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=7,
               frameon=False, handlelength=1.8, handletextpad=0.4, columnspacing=0.8)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Make room for legend
    output_file = os.path.join(SCRIPT_DIR, 'all_datasets_comparison.pdf')
    plt.savefig(output_file, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.show()

def world_to_image(xy, H):
    """Map world coordinates (m) to pixel coordinates via the inverse of an image->world homography."""
    xy = np.asarray(xy, dtype=float)
    pts = np.column_stack([xy, np.ones(len(xy))]) @ np.linalg.inv(H).T
    return pts[:, :2] / pts[:, 2:3]

def plot_zara1_on_scene(background_frame=340, output_name='zara1_scene_comparison.pdf'):
    """Plot the ETH-trained models' predictions on the Zara1 test scene, over the real camera frame.

    This sample is annotated pedestrians 12 and 13 in video frames 340-530 (observed 340-410, future 420-530);
    frame 340 shows them at the start of their observed trajectories, frame 410 at the end.
    """
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D

    dataset_name = 'cross_zara1'
    frame_file = os.path.join(SCRIPT_DIR, 'scene', f'crowds_zara01_frame{background_frame:04d}.png')
    if not os.path.exists(frame_file):
        import cv2
        cap = cv2.VideoCapture(os.path.join(SCRIPT_DIR, 'scene', 'crowds_zara01.avi'))
        cap.set(cv2.CAP_PROP_POS_FRAMES, background_frame)
        ok, frame = cap.read()
        assert ok, f"Cannot read frame {background_frame}"
        cv2.imwrite(frame_file, frame)
    image = plt.imread(frame_file)
    H = np.loadtxt(os.path.join(SCRIPT_DIR, 'scene', 'crowds_zara01_H.txt'))

    gt_trajectories = parse_trajectory_file(f"stcrf/{dataset_name}/best.txt")
    predictions = [
        # (trajectories, colour, marker, label)
        (parse_trajectory_file(f"social_stgcnn/{dataset_name}/pred.txt"), '#1F5BFF', 'D', 'Social-STGCNN'),
        (parse_trajectory_file(f"social_implicit/{dataset_name}/pred.txt"), '#FFB000', 'v', 'Social-Implicit'),
        (parse_trajectory_file(f"stcrf/{dataset_name}/pred.txt"), '#00D84A', '^', 'ST-CRF (ours)'),
    ]
    observed_color = '#E040FB'      # Magenta for observed
    future_color = '#FF2D2D'        # Red for future ground truth

    # Crop to the walkway around the trajectories (pixels); the figure keeps the crop's aspect ratio
    x_min, x_max, y_min, y_max = 140, 720, 230, 490
    fig_w = 3.36  # IEEE Access column width, so fonts print at true size
    img_h = fig_w * (y_max - y_min) / (x_max - x_min)
    legend_h = 0.34
    fig = plt.figure(figsize=(fig_w, img_h + legend_h))
    ax = fig.add_axes([0, legend_h / (img_h + legend_h), 1, img_h / (img_h + legend_h)])
    ax.imshow(image, zorder=0)
    ax.imshow(np.ones_like(image), alpha=0.18, zorder=1)  # light veil so the trajectories stand out

    # White outline keeps each line readable over bright and dark parts of the frame
    outline = [pe.Stroke(linewidth=2.6, foreground='white'), pe.Normal()]
    line_kw = dict(linewidth=1.4, markersize=3.2, markeredgecolor='white', markeredgewidth=0.4,
                   path_effects=outline, zorder=3)

    for ped_id in sorted(gt_trajectories):
        traj = gt_trajectories[ped_id]
        obs = world_to_image([(t[1], t[2]) for t in traj if t[0] <= 7], H)
        fut = world_to_image([(t[1], t[2]) for t in traj if t[0] >= 8], H)
        fut = np.vstack([obs[-1:], fut])  # connect last observed point with the future

        for trajectories, color, marker, _ in predictions:
            if ped_id in trajectories:
                pred = world_to_image([(t[1], t[2]) for t in trajectories[ped_id]], H)
                pred = np.vstack([obs[-1:], pred])
                # Mark only the predicted endpoint to keep the overlapping lines readable
                ax.plot(pred[:, 0], pred[:, 1], '-', color=color, marker=marker, markevery=[-1],
                        alpha=0.95, **{**line_kw, 'markersize': 4.5, 'zorder': 4})
        ax.plot(fut[:, 0], fut[:, 1], '--', color=future_color, marker='s', markevery=range(1, len(fut)),
                **{**line_kw, 'linewidth': 1.1, 'markersize': 2.6, 'markeredgewidth': 0,
                   'path_effects': [pe.Stroke(linewidth=1.9, foreground='white'), pe.Normal()], 'zorder': 5})
        ax.plot(obs[:, 0], obs[:, 1], '-', color=observed_color, marker='o', **{**line_kw, 'zorder': 6})
        ax.plot(*obs[0], 'o', color='white', markersize=4.5, markeredgecolor=observed_color,
                markeredgewidth=1.2, zorder=7)  # start of the observation
        ax.annotate('', xy=fut[-1], xytext=fut[-2], zorder=5,
                    arrowprops=dict(arrowstyle='-|>', color=future_color, lw=1.2, mutation_scale=9))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_axis_off()

    ax.text(0.015, 0.975, r'Train: ETH  $\rightarrow$  Test: Zara1 (zero-shot)', transform=ax.transAxes,
            fontsize=7, color='white', va='top', ha='left', zorder=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6, edgecolor='none'))

    handles = [
        Line2D([], [], color=observed_color, marker='o', markersize=3.2, linewidth=1.4, label='Observed'),
        Line2D([], [], color=future_color, marker='s', markersize=3.2, linewidth=1.4, linestyle='--',
               label='Ground Truth Future'),
    ] + [Line2D([], [], color=c, marker=m, markersize=3.2, linewidth=1.4, label=l) for _, c, m, l in predictions[::-1]]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, fontsize=7,
               frameon=False, handlelength=1.8, handletextpad=0.4, columnspacing=0.8, borderaxespad=0.1)

    output_file = os.path.join(SCRIPT_DIR, output_name)
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.01, dpi=300)
    print(f"Saved: {output_file}")
    plt.show()

def main():
    """Main function to generate all plots."""
    datasets = ['cross_hotel', 'cross_univ', 'cross_zara1', 'cross_zara2']
    
    print("Generating trajectory comparison plots for paper...")
    
    # Generate individual plots
    # for dataset in datasets:
    #     print(f"\nProcessing {dataset}...")
    #     plot_dataset_comparison(dataset)
    
    # Generate combined plot
    print(f"\nGenerating combined plot...")
    plot_all_datasets_combined()

    # Generate Zara1 plot over the real scene
    print(f"\nGenerating Zara1 scene plot...")
    plot_zara1_on_scene()

    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    main()