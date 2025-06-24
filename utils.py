import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time
import pickle


def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    if NORM == 0:
        return 0
    return 1 / (NORM)


def seq_to_graph(seq_, seq_rel, norm_lap_matr=True):
    
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    A = np.zeros((seq_len, max_nodes, max_nodes))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]
            A[s, h, h] = 1
            for k in range(h + 1, len(step_)):
                l2_norm = anorm(step_rel[h], step_rel[k])
                A[s, h, k] = l2_norm
                A[s, k, h] = l2_norm
        if norm_lap_matr:
            G = nx.from_numpy_matrix(A[s, :, :])
            A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()

    # Create Local graphs
    # print('A.shape:-------------',A.shape)
    return torch.from_numpy(V).type(torch.double),\
        torch.from_numpy(A).type(torch.double)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)



class ManeuverCalculator:
    def __init__(self):
        pass
    def calculate_distance(self, point_a, point_b):
        return np.sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)

    def calculate_velocity(self, point_a, point_b, time_delta):
        distance = self.calculate_distance(point_a, point_b)
        time_delta_seconds = (time_delta / 10.0)*0.4 ## 0.4s per frame
        velocity = distance / time_delta_seconds
        return velocity

    def calculate_tangent_angle(self, prev_point, next_point):
        delta_x = next_point[0] - prev_point[0]
        delta_y = next_point[1] - prev_point[1]
        angle = np.arctan2(delta_y, delta_x)
        return angle

    def calculate_maneuvers(self, data):
        trajectories = {}
        for row in data:
            frame_id, ped_id, x, y = row
            if ped_id not in trajectories:
                trajectories[ped_id] = []
            trajectories[ped_id].append((frame_id, x, y))

        maneuvers_data = np.zeros((data.shape[0], data.shape[1] + 2))
        maneuvers_data[:, :-2] = data

        for ped_id, traj in trajectories.items():
            traj = sorted(traj, key=lambda x: x[0])
            for i in range(1, len(traj)-1):
                frame_id, x, y = traj[i]

                ind = np.where((maneuvers_data[:, 0] == frame_id) & (maneuvers_data[:, 1] == ped_id))[0][0]

                prev_frame_id, prev_x, prev_y = traj[i - 1]
                next_frame_id, next_x, next_y = traj[i + 1]

                # print('frame_id:',frame_id)
                # print('prev_frame_id:',prev_frame_id)
                time_delta_prev = frame_id - prev_frame_id
                time_delta_next = next_frame_id - frame_id
                if time_delta_prev > 0 and time_delta_next > 0:
                    velocity_prev = self.calculate_velocity((prev_x, prev_y), (x, y), time_delta_prev)
                    velocity_next = self.calculate_velocity((x, y), (next_x, next_y), time_delta_next)
                    velocity = (velocity_prev + velocity_next) / 2
                    angle_threshold = np.clip(0.1 * velocity, 0.05, 0.2)

                    current_angle = self.calculate_tangent_angle((prev_x, prev_y), (next_x, next_y))
                    prev_angle = self.calculate_tangent_angle((prev_x, prev_y), (x, y))
                    angle_diff = abs(current_angle - prev_angle)
                ##Get lateral maneuver:  2: Turn Right, 1: Turn Left, 0: Keep Lane
                    if angle_diff > angle_threshold:
                        if current_angle > prev_angle:
                            lateral_maneuver = 1.0## Turn Left==>1
                        else:
                            lateral_maneuver = 2.0## Turn Left==>2
                    else:
                        lateral_maneuver = 0.0 ## Keep Lane==>0
                else:
                    lateral_maneuver = 1.0

                maneuvers_data[ind, -2] = lateral_maneuver
            
                if time_delta_prev > 0 and time_delta_next > 0:
                    velocity_prev = self.calculate_velocity((prev_x, prev_y), (x, y), time_delta_prev)
                    velocity_next = self.calculate_velocity((x, y), (next_x, next_y), time_delta_next)
                ## Get longitudinal maneuver: Keep Speed->0, Brake ->1, Accelerate ->2  for pedestrian
                    if velocity_next > velocity_prev * 1.1:
                        longitudinal_maneuver = 2.0## Accelerate==>2
                    elif velocity_next < velocity_prev * 0.9:
                        longitudinal_maneuver = 1.0## Decelerate==>1
                    else:
                        longitudinal_maneuver = 0.0## Keep Speed==>0
                else:
                    longitudinal_maneuver = 0.0 ## Keep Speed==>0

                maneuvers_data[ind, -1] = longitudinal_maneuver

        return maneuvers_data

class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(self,
                 data_dir,
                 obs_len=8,
                 pred_len=8,
                 skip=1,
                 threshold=0.002,
                 min_ped=1,
                 delim='\t',
                 norm_lap_matr=True):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr
        self.caculator_maneuvers = ManeuverCalculator()

        args_str = data_dir + str(obs_len) + str(pred_len) + str(skip) + \
            str(threshold) + str(min_ped) + str(norm_lap_matr)
        pkl_path = './pkls/' + args_str.replace("/", "_") + '.pkl'

        if os.path.exists(pkl_path):
            print("Dataset found, Loading dataset from:", pkl_path)
            with open(pkl_path, 'rb') as f:
                __data = pickle.load(f)

                self.obs_traj = __data["obs_traj"]
                self.pred_traj = __data["pred_traj"]
                self.obs_traj_m = __data["self.obs_traj_m"]
                self.pred_traj_m = __data["self.pred_traj_m"]
                self.obs_traj_rel = __data["obs_traj_rel"]
                self.pred_traj_rel = __data["pred_traj_rel"]
                self.non_linear_ped = __data["non_linear_ped"]
                self.loss_mask = __data["loss_mask"]
                self.v_obs = __data["v_obs"]
                self.A_obs = __data["A_obs"]
                self.v_pred = __data["v_pred"]
                self.A_pred = __data["A_pred"]
                self.num_seq = __data["num_seq"]
                self.seq_start_end = __data["seq_start_end"]

        else:
            all_files = os.listdir(self.data_dir)
            all_files = [
                os.path.join(self.data_dir, _path) for _path in all_files
            ]
            num_peds_in_seq = []
            seq_list = []
            seq_list_m = []  # Added code
            seq_list_rel = []
            loss_mask_list = []
            non_linear_ped = []
            # Initialize lists to store maneuvers for obs and pred sequences
            # obs_maneuvers_list = []  # Added code
            # pred_maneuvers_list = []  # Added code
            for path in all_files:
                data = read_file(path, delim)
                # print('data.shape',data.shape)
                # data_with_maneuvers = calculate_maneuvers(data)### Added code
                data_with_maneuvers = self.caculator_maneuvers.calculate_maneuvers(data)### Added code
                # print('data_with_maneuvers.shape',data_with_maneuvers.shape)
                data_with_maneuvers = data_with_maneuvers[:, [0, 1, 4,5 ]]  # Added code
                # print(data[:,:2]==data_with_maneuvers[:,:2])###true
                ## 只取 data_with_maneuvers 的前两列和最后两列
                # data_with_maneuvers = data_with_maneuvers[:, [0, 1, -2, -1]]  # Added code
                # data_with_m = data_with_maneuvers[:, -2:]  # Added code
                frames = np.unique(data[:, 0]).tolist()
                frame_data = []
                frame_data_m = []  # Added code
                for frame in frames:
                    frame_data.append(data[frame == data[:, 0], :])
                    frame_data_m.append(data_with_maneuvers[frame == data[:, 0], :])
                num_sequences = int(
                    math.ceil((len(frames) - self.seq_len + 1) / skip))

                for idx in range(0, num_sequences * self.skip + 1, skip):
                    curr_seq_data = np.concatenate(frame_data[idx:idx +
                                                              self.seq_len],
                                                   axis=0)
                    curr_seq_data_m = np.concatenate(frame_data_m[idx:idx + self.seq_len], axis=0)
                    # Extract maneuvers for the current sequence
                    # curr_seq_maneuvers = data_with_maneuvers[idx:idx + self.seq_len, :]  # Added code
                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                    self.max_peds_in_frame = max(self.max_peds_in_frame,
                                                 len(peds_in_curr_seq))
                    curr_seq_rel = np.zeros(
                        (len(peds_in_curr_seq), 2, self.seq_len))
                    curr_seq = np.zeros(
                        (len(peds_in_curr_seq), 2, self.seq_len))
                    curr_seq_m = np.zeros(
                        (len(peds_in_curr_seq), 2, self.seq_len))
                    curr_loss_mask = np.zeros(
                        (len(peds_in_curr_seq), self.seq_len))
                    num_peds_considered = 0
                    _non_linear_ped = []
                    for _, ped_id in enumerate(peds_in_curr_seq):
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                     ped_id, :]
                        curr_ped_seq_m = curr_seq_data_m[curr_seq_data_m[:, 1] == ped_id, :]  # Added code
                        # curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                        pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                        if pad_end - pad_front != self.seq_len:
                            continue
                        curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                        curr_ped_seq_m = np.transpose(curr_ped_seq_m[:, 2:])  # Added code
                        curr_ped_seq = curr_ped_seq
                        # Make coordinates relative
                        rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                        rel_curr_ped_seq[:, 1:] = \
                            curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                        _idx = num_peds_considered
                        # print('curr_ped_seq.shape:',curr_ped_seq.shape)##(2,20)
                        # print('curr_ped_seq_m.shape:',curr_ped_seq_m.shape)#(2,20)
                        curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                        curr_seq_m[_idx, :, pad_front:pad_end] = curr_ped_seq_m
                        curr_seq_rel[_idx, :,
                                     pad_front:pad_end] = rel_curr_ped_seq
                        # Linear vs Non-Linear Trajectory
                        _non_linear_ped.append(
                            poly_fit(curr_ped_seq, pred_len, threshold))
                        curr_loss_mask[_idx, pad_front:pad_end] = 1
                        num_peds_considered += 1

                    if num_peds_considered > min_ped:
                        non_linear_ped += _non_linear_ped
                        num_peds_in_seq.append(num_peds_considered)
                        loss_mask_list.append(
                            curr_loss_mask[:num_peds_considered])
                        seq_list.append(curr_seq[:num_peds_considered])
                        seq_list_m.append(curr_seq_m[:num_peds_considered])  # Added code
                        seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                        # Store maneuvers for observation and prediction sequences
                        seq_list_m.append(curr_seq_m[:num_peds_considered])  # Added code

            self.num_seq = len(seq_list)
            seq_list = np.concatenate(seq_list, axis=0)
            seq_list_m = np.concatenate(seq_list_m, axis=0)
            seq_list_rel = np.concatenate(seq_list_rel, axis=0)
            loss_mask_list = np.concatenate(loss_mask_list, axis=0)
            non_linear_ped = np.asarray(non_linear_ped)

            # Convert numpy -> Torch Tensor
            self.obs_traj = torch.from_numpy(
                seq_list[:, :, :self.obs_len]).type(torch.double)
            self.pred_traj = torch.from_numpy(
                seq_list[:, :, self.obs_len:]).type(torch.double)
            self.obs_traj_m = torch.from_numpy(
                seq_list_m[:, :, :self.obs_len]).type(torch.double)
            self.pred_traj_m = torch.from_numpy(
                seq_list_m[:, :, self.obs_len:]).type(torch.double)   
            self.obs_traj_rel = torch.from_numpy(
                seq_list_rel[:, :, :self.obs_len]).type(torch.double)
            self.pred_traj_rel = torch.from_numpy(
                seq_list_rel[:, :, self.obs_len:]).type(torch.double)
            self.loss_mask = torch.from_numpy(loss_mask_list).type(
                torch.double)
            self.non_linear_ped = torch.from_numpy(non_linear_ped).type(
                torch.double)
            cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
            self.seq_start_end = [
                (start, end)
                for start, end in zip(cum_start_idx, cum_start_idx[1:])
            ]
            # Convert maneuvers lists to Torch Tensors and store them
            # Convert to Graphs
            self.v_obs = []
            self.A_obs = []
            self.v_pred = []
            self.A_pred = []
            print("Processing Data .....")
            pbar = tqdm(total=len(self.seq_start_end))
            for ss in range(len(self.seq_start_end)):
                pbar.update(1)

                start, end = self.seq_start_end[ss]
                
                v_, a_ = seq_to_graph(self.obs_traj[start:end, :],
                                      self.obs_traj_rel[start:end, :],
                                      self.norm_lap_matr)
                self.v_obs.append(v_.clone())
                self.A_obs.append(a_.clone())
                v_, a_ = seq_to_graph(self.pred_traj[start:end, :],
                                      self.pred_traj_rel[start:end, :],
                                      self.norm_lap_matr)
                self.v_pred.append(v_.clone())
                self.A_pred.append(a_.clone())
            pbar.close()

            __data = {}
            __data["obs_traj"] = self.obs_traj
            __data["pred_traj"] = self.pred_traj
            __data["self.obs_traj_m"] = self.obs_traj_m
            __data["self.pred_traj_m"] = self.pred_traj_m
            __data["obs_traj_rel"] = self.obs_traj_rel
            __data["pred_traj_rel"] = self.pred_traj_rel
            __data["non_linear_ped"] = self.non_linear_ped
            __data["loss_mask"] = self.loss_mask
            __data["v_obs"] = self.v_obs
            __data["A_obs"] = self.A_obs
            __data["v_pred"] = self.v_pred
            __data["A_pred"] = self.A_pred
            __data["num_seq"] = self.num_seq
            __data["seq_start_end"] = self.seq_start_end
            print("Saving dataset to:", pkl_path)
            with open(pkl_path, "wb") as output_file:
                pickle.dump(__data, output_file)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index], self.v_pred[index],
            self.A_pred[index],self.obs_traj_m[start:end, :],self.pred_traj_m[start:end, :]
        ]
        return out

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    #Data specifc paremeters
    parser.add_argument('--obs_seq_len', type=int, default=8)
    parser.add_argument('--pred_seq_len', type=int, default=12)
    parser.add_argument('--dataset',
                        default='hotel_s',
                        help='eth,hotel,univ,zara1,zara2,sdd')

    #Training specifc parameters
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='minibatch size')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=2, 
                        help='number of epochs')
    parser.add_argument('--clip_grad',
                        type=float,
                        default=None,
                        help='gadient clipping')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_sh_rate',
                        type=int,
                        default=40,
                        help='number of steps to drop the lr')

    parser.add_argument('--tag', default='tag', help='personal tag for the model ')
    args = parser.parse_args()
    obs_seq_len = args.obs_seq_len
    pred_seq_len = args.pred_seq_len
    data_set = './datasets/' + args.dataset + '/'

    dset_train = TrajectoryDataset(data_set + 'train_val/',
                                obs_len=obs_seq_len,
                                pred_len=pred_seq_len,
                                skip=1,
                                norm_lap_matr=True)
    loader_train = DataLoader(
    dset_train,
    batch_size=1,  #This is irrelative to the args batch size parameter
    shuffle=True,
    num_workers=0)
    for cnt, batch in enumerate(loader_train):

            #Get data
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr,obs_ma,pred_ma = batch
            # print('V_obs.shape before aug:',V_obs.shape)###[batch,obs_len,num_ped,2]
            print('obs_traj.shape:',obs_traj.shape)###[batch,num_ped,obs_len,2]
            print('pred_traj_gt.shape:',pred_traj_gt.shape)
            print('obs_ma.shape:',obs_ma.shape)
            print('pred_ma.shape:',pred_ma.shape)
            print('-----------------------------------------')
            # print(obs_traj[:,:,:,:])
            print('obs_traj',obs_traj)
            print('pred_traj_gt',pred_traj_gt)
            print('obs_ma',obs_ma)
            print('pred_ma',pred_ma)
            break

