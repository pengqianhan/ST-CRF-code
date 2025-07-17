import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Update to load batch data instead of trajectory data
output_folder = 'best_batches'
best_batches_file = os.path.join(output_folder, 'best_batches_data.pkl')

def print_batch_info():
    """
    Load and print detailed information about the best batches data.
    """
    try:
        with open(best_batches_file, 'rb') as f:
            best_batches = pickle.load(f)
        print(f"Successfully loaded {len(best_batches)} best batches from saved data")
        print("="*80)
        
        # Print overall statistics
        print("OVERALL STATISTICS:")
        print("-" * 40)
        avg_ades = [batch['avg_ade'] for batch in best_batches]
        avg_fdes = [batch['avg_fde'] for batch in best_batches]
        total_pedestrians = sum(batch['num_pedestrians'] for batch in best_batches)
        
        print(f"Total batches: {len(best_batches)}")
        print(f"Total pedestrians across all batches: {total_pedestrians}")
        print(f"Average ADE range: {min(avg_ades):.4f} to {max(avg_ades):.4f}")
        print(f"Average FDE range: {min(avg_fdes):.4f} to {max(avg_fdes):.4f}")
        print(f"Mean of batch averages - ADE: {np.mean(avg_ades):.4f}, FDE: {np.mean(avg_fdes):.4f}")
        print()
        
        # Print detailed information for each batch
        print("DETAILED BATCH INFORMATION:")
        print("="*80)
        
        for i, batch in enumerate(best_batches):
            print(f"\nBATCH RANK {i+1}:")
            print("-" * 50)
            print(batch.keys())
            print(len(batch['observed']))
            print(batch['observed'][0].shape)
            
            # Basic batch info
            print(f"Dataset: {batch.get('dataset', 'Unknown')}")
            print(f"Step: {batch.get('step', 'Unknown')}")
            print(f"Number of pedestrians: {batch['num_pedestrians']}")
            print(f"Average ADE: {batch['avg_ade']:.4f}")
            print(f"Average FDE: {batch['avg_fde']:.4f}")
            
            # Pedestrian-level statistics
            ped_ades = batch['pedestrian_ades']
            ped_fdes = batch['pedestrian_fdes']
            print(f"Individual pedestrian ADEs: {[f'{ade:.4f}' for ade in ped_ades]}")
            print(f"Individual pedestrian FDEs: {[f'{fde:.4f}' for fde in ped_fdes]}")
            print(f"Best pedestrian ADE: {min(ped_ades):.4f}")
            print(f"Worst pedestrian ADE: {max(ped_ades):.4f}")
            print(f"ADE std deviation: {np.std(ped_ades):.4f}")
            
            # Trajectory shape information
            observed = batch['observed']
            predicted = batch['predicted']
            ground_truth = batch['ground_truth']
            
            print(f"Number of observed trajectories: {len(observed)}")
            print(f"Number of predicted trajectories: {len(predicted)}")
            print(f"Number of ground truth trajectories: {len(ground_truth)}")
            
            if len(observed) > 0:
                print(f"Observed trajectory shape (first ped): {observed[0].shape}")
                print(f"Predicted trajectory shape (first ped): {predicted[0].shape}")
                print(f"Ground truth trajectory shape (first ped): {ground_truth[0].shape}")
            
            # Optional: Print first few points of first pedestrian for debugging
            if len(observed) > 0 and i == 0:  # Only for first batch to avoid too much output
                print(f"\nSample data for first pedestrian:")
                print(f"First 3 observed points: {observed[0][:3]}")
                print(f"First 3 predicted points: {predicted[0][:3]}")
                print(f"First 3 ground truth points: {ground_truth[0][:3]}")
            
            # Stop after first 5 batches to avoid too much output
            if i >= 4:
                remaining = len(best_batches) - (i + 1)
                if remaining > 0:
                    print(f"\n... and {remaining} more batches")
                break
        
        print("\n" + "="*80)
        print("BATCH DATA LOADING COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except FileNotFoundError:
        print(f"Error: File {best_batches_file} not found!")
        print("Please run test_vis.py first to generate the batch data.")
        return False
    except Exception as e:
        print(f"Error loading batch data: {e}")
        return False
    
    return True

def print_data_structure():
    """
    Print the expected data structure for reference.
    """
    print("\nEXPECTED BATCH DATA STRUCTURE:")
    print("-" * 40)
    print("Each batch contains:")
    print("  - avg_ade: float (average ADE for all pedestrians in batch)")
    print("  - avg_fde: float (average FDE for all pedestrians in batch)")
    print("  - pedestrian_ades: list of floats (ADE for each pedestrian)")
    print("  - pedestrian_fdes: list of floats (FDE for each pedestrian)")
    print("  - observed: list of arrays (historical trajectories, shape [obs_len, 2])")
    print("  - predicted: list of arrays (predicted trajectories, shape [pred_len, 2])")
    print("  - ground_truth: list of arrays (ground truth trajectories, shape [pred_len, 2])")
    print("  - step: int (batch step number)")
    print("  - num_pedestrians: int (number of pedestrians in batch)")
    print("  - dataset: str (dataset name)")
    print("  - model_path: str (path to model used)")

if __name__ == "__main__":
    print("LOADING AND ANALYZING BEST BATCHES DATA")
    print("="*80)
    
    # Print expected data structure
    print_data_structure()
    
    # Load and print batch information
    success = print_batch_info()
    
    if success:
        print(f"\nData file location: {best_batches_file}")
        print("Use this data for further analysis or visualization.")
    else:
        print("\nFailed to load batch data. Please check if the file exists and was generated correctly.")