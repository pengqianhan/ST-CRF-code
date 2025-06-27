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
from datetime import datetime

def test(KSTEPS=1):

    global loader_test, model, ROBUSTNESS
    model.eval()
    ade_bigls = []
    fde_bigls = []
    step = 0
    for batch in loader_test:
        step += 1
        #Get data
        batch = [tensor.cuda().double() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,loss_mask,V_obs,A_obs,V_tr,A_tr,_,_ = batch
        # obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,loss_mask,V_obs,A_obs,V_tr,A_tr,_,_ = batch
        ####
        seq = torch.cat((obs_traj, pred_traj_gt), dim=3)##[batch,num_ped,2,obs_len+pred_len]=[1, 5, 2, 20]
        seq = seq.permute(0,3,1,2)##[batch,obs_len+pred_len,num_ped,2]
        seq_ma = torch.zeros_like(seq)#[batch,obs_len+pred_len,num_ped,2]
        # seq_ma = Compute_ma.forward(seq)##
        # seq_ma = seq_ma[...,2:]
        # seq_ma = seq_ma.to(V_obs.device)
        ####

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

        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)
    return ade_, fde_


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Set random seed for reproducibility
seed_torch()

# Define all available datasets and their corresponding checkpoint paths
available_datasets = ['eth', 'hotel', 'univ', 'zara1', 'zara2', 'sdd']
checkpoint_datasets = ['eth', 'hotel', 'univ', 'zara1', 'zara2']  # Datasets with trained models

# Generate all zero-shot combinations (train != test)
zero_shot_combinations = []
for train_dataset in checkpoint_datasets:
    for test_dataset in available_datasets:
        if train_dataset != test_dataset:
            zero_shot_combinations.append((train_dataset, test_dataset))

print(f"Total zero-shot combinations to evaluate: {len(zero_shot_combinations)}")
print("Combinations:", zero_shot_combinations)

ROBUSTNESS = 0  # Set robustness to 0 for standard evaluation
KSTEPS = 1  # Number of prediction samples

# Store all results
all_results = []

print("*" * 50)
print('Number of samples:', KSTEPS)
print("*" * 50)

# Evaluate each zero-shot combination
for i, (train_dataset, test_dataset) in enumerate(zero_shot_combinations):
    print(f"\n{'='*60}")
    print(f"Evaluation {i+1}/{len(zero_shot_combinations)}")
    print(f"Training model: {train_dataset} → Testing on: {test_dataset}")
    print(f"{'='*60}")
    
    try:
        # Set up paths for current combination
        checkpoint_path = f'./checkpoint_deter/stcrf_{train_dataset}'
        
        print("*" * 50)
        print("Evaluating model:", checkpoint_path)

        model_path = checkpoint_path + '/val_best.pth'
        args_path = checkpoint_path + '/args.pkl'
        
        # Load model arguments
        with open(args_path, 'rb') as f:
            args = pickle.load(f)

        # Load statistics
        stats = checkpoint_path + '/constant_metrics.pkl'
        with open(stats, 'rb') as f:
            cm = pickle.load(f)
        print("Stats:", cm)

        # Data preparation
        obs_seq_len = args.obs_seq_len
        pred_seq_len = args.pred_seq_len
        
        data_set = f'./datasets/{test_dataset}/'

        dset_test = TrajectoryDataset(data_set + 'test/',
                                      obs_len=obs_seq_len,
                                      pred_len=pred_seq_len,
                                      skip=1,
                                      norm_lap_matr=True)

        loader_test = DataLoader(
            dset_test,
            batch_size=1,
            shuffle=False,
            num_workers=1)

        # Define the model
        if args.dataset == 'eth':
            stgcn_layer = 0
        else: 
            stgcn_layer = 1
            
        model = STCRF(spatial_input=CFG["spatial_input"],
                   spatial_output=CFG["spatial_output"],
                   temporal_input=CFG["temporal_input"],
                   temporal_output=CFG["temporal_output"],
                   stgcn_layer=stgcn_layer).cuda().double()

        # Load model weights
        model.load_state_dict(torch.load(model_path))
        model.cuda().double()
        model.eval()
        
        # Test the model
        print("Testing ....")
        ade_, fde_ = test(KSTEPS=KSTEPS)
        
        # Store results
        result = {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'ade': round(ade_, 4),
            'fde': round(fde_, 4),
            'model_path': checkpoint_path
        }
        all_results.append(result)
        
        print(f"ADE: {ade_:.4f}, FDE: {fde_:.4f}")
        
    except Exception as e:
        print(f"Error evaluating {train_dataset} → {test_dataset}: {str(e)}")
        # Store error result
        result = {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'ade': 'ERROR',
            'fde': 'ERROR',
            'model_path': checkpoint_path,
            'error': str(e)
        }
        all_results.append(result)

# Generate markdown report
def generate_markdown_report(results):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown_content = f"""# Zero-Shot Trajectory Prediction Results

**Generated on:** {timestamp}  
**Total Combinations Evaluated:** {len(results)}  
**KSTEPS:** {KSTEPS}  
**ROBUSTNESS:** {ROBUSTNESS}  

## Summary Table

| Training Dataset | Testing Dataset | ADE | FDE | Status |
|------------------|-----------------|-----|-----|--------|
"""
    
    for result in results:
        status = "✅ Success" if result['ade'] != 'ERROR' else "❌ Failed"
        markdown_content += f"| {result['train_dataset']} | {result['test_dataset']} | {result['ade']} | {result['fde']} | {status} |\n"
    
    # Add detailed results section
    markdown_content += f"""
## Detailed Results

"""
    
    # Group results by training dataset
    train_datasets = list(set([r['train_dataset'] for r in results]))
    for train_ds in sorted(train_datasets):
        markdown_content += f"### Model trained on {train_ds.upper()}\n\n"
        train_results = [r for r in results if r['train_dataset'] == train_ds]
        
        for result in train_results:
            if result['ade'] != 'ERROR':
                markdown_content += f"- **{result['test_dataset'].upper()}**: ADE={result['ade']}, FDE={result['fde']}\n"
            else:
                markdown_content += f"- **{result['test_dataset'].upper()}**: ERROR - {result.get('error', 'Unknown error')}\n"
        markdown_content += "\n"
    
    # Add statistics
    successful_results = [r for r in results if r['ade'] != 'ERROR']
    if successful_results:
        ade_values = [r['ade'] for r in successful_results]
        fde_values = [r['fde'] for r in successful_results]
        
        markdown_content += f"""## Statistics

- **Successful Evaluations:** {len(successful_results)}/{len(results)}
- **Mean ADE:** {np.mean(ade_values):.4f}
- **Mean FDE:** {np.mean(fde_values):.4f}
- **Min ADE:** {np.min(ade_values):.4f}
- **Max ADE:** {np.max(ade_values):.4f}
- **Min FDE:** {np.min(fde_values):.4f}
- **Max FDE:** {np.max(fde_values):.4f}

## Best Performing Combinations

### Best ADE
"""
        best_ade = min(successful_results, key=lambda x: x['ade'])
        markdown_content += f"- **{best_ade['train_dataset']} → {best_ade['test_dataset']}**: ADE={best_ade['ade']}, FDE={best_ade['fde']}\n\n"
        
        markdown_content += "### Best FDE\n"
        best_fde = min(successful_results, key=lambda x: x['fde'])
        markdown_content += f"- **{best_fde['train_dataset']} → {best_fde['test_dataset']}**: ADE={best_fde['ade']}, FDE={best_fde['fde']}\n\n"
    
    return markdown_content

# Generate and save the markdown report
markdown_report = generate_markdown_report(all_results)
output_filename = f"zero_shot_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

with open(output_filename, 'w', encoding='utf-8') as f:
    f.write(markdown_report)

print(f"\n{'='*60}")
print("EVALUATION COMPLETE!")
print(f"Results saved to: {output_filename}")
print(f"{'='*60}")

# Print summary to console
successful_results = [r for r in all_results if r['ade'] != 'ERROR']
print(f"\nSummary:")
print(f"- Total combinations: {len(all_results)}")
print(f"- Successful: {len(successful_results)}")
print(f"- Failed: {len(all_results) - len(successful_results)}")

if successful_results:
    ade_values = [r['ade'] for r in successful_results]
    fde_values = [r['fde'] for r in successful_results]
    print(f"- Mean ADE: {np.mean(ade_values):.4f}")
    print(f"- Mean FDE: {np.mean(fde_values):.4f}")