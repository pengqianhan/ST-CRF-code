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
        trajectory_data: List of trajectory dictionaries
        num_trajectories: Number of best trajectories to plot
    """
    # Sort trajectories by ADE (ascending - lower is better)
    sorted_trajectories = sorted(trajectory_data, key=lambda x: x['ade'])
    
    # Select the best trajectories
    best_trajectories = sorted_trajectories[:num_trajectories]
    
    print(f"\nPlotting {len(best_trajectories)} most accurate trajectories...")
    print(f"ADE range: {best_trajectories[0]['ade']:.4f} to {best_trajectories[-1]['ade']:.4f}")
    
    # Create output folder
    output_folder = 'best_trajectories'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
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

def test(KSTEPS=1, dataset='eth'):

    global loader_test, model, ROBUSTNESS
    threshold = static_dist_dict.get(dataset, 0.3)  # Default threshold if dataset not found
    model.eval()
    ade_bigls = []
    fde_bigls = []
    trajectory_data = []  # Store trajectory data with ADE for plotting
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
        # print("V_obs.shape:", V_obs.shape)##(1, 8, 2, 2)#[batch,obs_len,num_ped,2]
        # print("seq_ma.shape:", seq_ma.shape)##(1, 20, 2, 2)#[batch,obs_len+pred_len,num_ped,2]
        # print("obs_traj.shape:", obs_traj.shape)##(1, 2, 2, 8)#(batch,num_ped,2,obs_len)
        # print('A_obs.shape:', A_obs.shape)##(1, 8, 2, 2)#[batch,obs_len,num_ped,num_ped]

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
        
        for n in range(num_of_objs):
            # Only process pedestrians that meet the movement criteria
            if not valid_peds[0, n].item() or len(ade_ls[n]) == 0 or n not in best_predictions:
                continue
                
            min_ade = min(ade_ls[n])
            min_fde = min(fde_ls[n])
            ade_bigls.append(min_ade)
            fde_bigls.append(min_fde)
            
            # Store trajectory data for visualization
            traj_data = {
                'ade': min_ade,
                'fde': min_fde,
                'observed': V_x_rel_to_abs[:, n, :],  # Historical trajectory
                'ground_truth': V_y_rel_to_abs[:, n, :],  # Ground truth future
                'predicted': best_predictions[n][:, n, :],  # Best prediction
                'step': step,
                'pedestrian_id': n
            }
            trajectory_data.append(traj_data)

    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)
    return ade_, fde_, trajectory_data


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
        './checkpoint_deter/stcrf_hotel',
        './checkpoint_deter/stcrf_univ',
        './checkpoint_deter/stcrf_zara1',
        './checkpoint_deter/stcrf_zara2',
    ]

    KSTEPS = 1## use this metric to evaluate the model

    EASY_RESULTS = []
    ALL_TRAJECTORY_DATA = []  # Collect all trajectory data for visualization

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
            data_set = './datasets/' + args.dataset + '/'

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
            ad, fd, traj_data = test(KSTEPS=KSTEPS, dataset=args.dataset)
            ade_ = min(ade_, ad)
            fde_ = min(fde_, fd)
            ade_ls.append(ade_)
            fde_ls.append(fde_)
            exp_ls.append(exp_path)
            
            # Add dataset info to trajectory data and collect
            for traj in traj_data:
                traj['dataset'] = args.dataset
                traj['model_path'] = exp_path
            ALL_TRAJECTORY_DATA.extend(traj_data)
            
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
    
    # Plot the 15 most accurate trajectories
    if ALL_TRAJECTORY_DATA:
        print(f"\nTotal trajectories collected: {len(ALL_TRAJECTORY_DATA)}")
        plot_best_trajectories(ALL_TRAJECTORY_DATA, num_trajectories=15)
    else:
        print("No trajectory data collected for visualization.")