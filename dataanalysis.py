import os
import torch
from torch.utils.data import DataLoader
from utils import *
from metrics import *
import pickle
from utils_crf import BatchManeuverCalculator
import os  


#Data prep
obs_seq_len = 8
pred_seq_len = 12
data_name_list = ['eth','hotel','univ','zara1','zara2']
for data_name in data_name_list:

    # data_name = 'hotel'
    print('data_name:',data_name)
    data_set = './datasets/' + data_name + '/'

    print('data_set:',data_set)
    dset_train = TrajectoryDataset(data_set + 'train_val/',
                                obs_len=obs_seq_len,
                                pred_len=pred_seq_len,
                                skip=1,
                                norm_lap_matr=True)## skip = obs_len + pred_len

    loader_train = DataLoader(
        dset_train,
        batch_size=1, 
        shuffle=False,
        num_workers=0)

    #################maneuver label compute###########################
    Compute_ma = BatchManeuverCalculator()
    ###################################################################

    lat_num = {'straight_0':0,'left_1':0,'right_2':0}
    lon_num = {'norm_0':0,'de_1':0,'ac_2':0}
    seq_num = 0
    norm_action_num = 0
    for cnt, batch in enumerate(loader_train):
        #Get data
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr,obs_traj_m,pred_traj_m = batch

        seq_m = torch.cat((obs_traj_m, pred_traj_m), dim=3)##[batch,num_ped,2,obs_len+pred_len]=[1, 4, 2, 20]
        
        ## compute maneuver label on obs_traj and pred_traj_gt, which is the real trajectory not relative position
        seq = torch.cat((obs_traj, pred_traj_gt), dim=3)
        seq = seq.permute(0,3,1,2)
        seq_ma = Compute_ma.calculate_maneuvers(seq)
        # print('seq_ma.shape:',seq_ma.shape)### [1,20,3,2]
        # lat = seq_ma[:,:,:,0]
        # lon = seq_ma[:,:,:,1]
        ##just consider the obs_traj
        lat = seq_ma[:,:obs_seq_len,:,0]
        lon = seq_ma[:,:obs_seq_len,:,1]
        lat = lat.view(-1)
        lon = lon.view(-1)
        # compute the number of lat and lon both are 0 
        seq_num += 1
        if torch.sum(lat) == 0 and torch.sum(lon) == 0:
            norm_action_num += 1

        # compute the number of each maneuver
        for i in lat:
            if i == 0:
                lat_num['straight_0'] += 1
            elif i == 1:
                lat_num['left_1'] += 1
            elif i == 2:
                lat_num['right_2'] += 1
        for i in lon:
            if i == 0:
                lon_num['norm_0'] += 1
            elif i == 1:
                lon_num['de_1'] += 1
            elif i == 2:
                lon_num['ac_2'] += 1

    print(f'the result of data analysis is:{data_name}-----------------')
    print('lat_num:',lat_num)
    #compute percentage of straight_0 
    total_lat = lat_num['straight_0'] + lat_num['left_1'] + lat_num['right_2']
    print('total lat:',total_lat)
    print('straight_0: {:.2f}%'.format((lat_num['straight_0'] / total_lat) * 100))
    print('left_1: {:.2f}%'.format((lat_num['left_1'] / total_lat) * 100))
    print('right_2: {:.2f}%'.format((lat_num['right_2'] / total_lat) * 100))

    print('lon_num:',lon_num)
    total_lon = lon_num['norm_0'] + lon_num['de_1'] + lon_num['ac_2']
    print('total lon:',total_lon)
    print('norm_0: {:.2f}%'.format((lon_num['norm_0'] / total_lon) * 100))
    print('de_1: {:.2f}%'.format((lon_num['de_1'] / total_lon) * 100))
    print('ac_2: {:.2f}%'.format((lon_num['ac_2'] / total_lon) * 100))

    #print the number of norm action in all the sequence
    print('norm_action_num:',norm_action_num)
    print('seq_num:',seq_num)
    # print the norm action percentage in all the sequence
    print('norm_action_percentage {:.2f}%:'.format((norm_action_num / seq_num) * 100))






   

