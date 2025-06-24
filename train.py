import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
from metrics import *
import pickle
import argparse
from model import STCRF
from trajectory_augmenter import TrajectoryAugmenter
from utils_crf import BatchManeuverCalculator
from CFG import CFG
from utils import seq_to_graph
import random
from param_parser import parameter_parser
from torch.optim.lr_scheduler import LambdaLR
import os  

args = parameter_parser()
print('*' * 30)
print("Training initiating....")
print(args)

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



# _l1_mean = nn.L1Loss()
_l1_mean = nn.MSELoss()


def cdist_cosine_sim(a, b, eps=1e-08):
    a_norm = a / torch.clamp(a.norm(dim=1)[:, None], min=eps)
    b_norm = b / torch.clamp(b.norm(dim=1)[:, None], min=eps)
    return torch.acos(
        torch.clamp(torch.mm(a_norm, b_norm.transpose(0, 1)),
                    min=-1.0 + eps,
                    max=1.0 - eps))

def implicit_likelihood_estimation_fast_with_trip_geo(V_pred, V_target):
    # print('V_pred.shape:',V_pred.shape)###[KSTEPS,obs_len,num_ped,2]
    # print('V_target.shape:',V_target.shape)###[1,obs_len,num_ped,2]

    V_pred = V_pred.contiguous()

    diff = torch.abs(V_pred - V_target)

    diff_sum = torch.sum(diff, dim=(1, 2, 3))
    _, indices = torch.sort(diff_sum)
    min_indx = indices[0]
    V_pred_min = V_pred[min_indx]
    V_target = V_target.squeeze()
    # print('V_pred_min.shape:',V_pred_min.shape)###[obs_len,num_ped,2]
    # print('V_target.shape:',V_target.shape)###[obs_len,num_ped,2]
    error = _l1_mean(V_pred_min, V_target)
    return error


def graph_loss(V_pred, V_target, V_obs):
    return implicit_likelihood_estimation_fast_with_trip_geo(V_pred, V_target)

#Data prep
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

dset_val = TrajectoryDataset(data_set + 'test/',
                             obs_len=obs_seq_len,
                             pred_len=pred_seq_len,
                             skip=1,
                             norm_lap_matr=True)

loader_val = DataLoader(
    dset_val,
    batch_size=1,  #This is irrelative to the args batch size parameter
    shuffle=False,
    num_workers=1)
##
if args.dataset == 'eth':
    stgcn_layer = 0
else: 
    stgcn_layer = 1
model = STCRF(spatial_input=CFG["spatial_input"],
                       spatial_output=CFG["spatial_output"],
                       temporal_input=CFG["temporal_input"],
                       temporal_output=CFG["temporal_output"],
                       stgcn_layer=stgcn_layer).cuda().double()
#Optimizer and Schedule
optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
# scheduler = optim.lr_scheduler.StepLR(optimizer,
#                                       step_size=50,
#                                       gamma=0.2)
## lr_lambda for crf loss and mse loss interchangeably
def lr_lambda(epoch):
    if 0 <= epoch < 51:
        return 0.01 / 0.01
    elif 51 <= epoch < args.crf_epoch +1:
        return 0.002 / 0.01
    elif args.crf_epoch +1 <= epoch < 151:
        return 0.01 / 0.01
    elif 151<= epoch <= 200:
        return 0.002 / 0.01
    elif 200 < epoch <= 250:
        return 0.002 / 0.01
    elif 250 < epoch <= 300:
        return 0.001 / 0.01
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)



#Check pointing
checkpoint_dir = './checkpoint/' + args.tag + '/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)

print('Data and model loaded')
print('Checkpoint dir:', checkpoint_dir)

#Training
metrics = {'train_loss': [], 'val_loss': [],'crf_loss':[],'obs_loss':[]}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999}
trajaugmenter = TrajectoryAugmenter(data_loader=loader_train)
#################maneuver label compute###########################
Compute_ma = BatchManeuverCalculator()
###################################################################
def train(epoch):
    global metrics, loader_train
    model.train()
    batch_loss = 0
    batch_crf_loss = 0
    batch_obs_loss = 0
    fusion_loss = 0
    total_loss = 0
    total_obs_loss = 0
    total_crf_loss = 0
    for cnt, batch in enumerate(loader_train):

        #Get data
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr,obs_traj_m,pred_traj_m = batch
        # print('V_obs.shape before aug:',V_obs.shape)###[batch,obs_len,num_ped,2]
        # print('obs_traj_rel.shape:',obs_traj_rel.shape)###[batch,num_ped,2,obs_len]
        # print('obs_ma.shape:',obs_traj_m.shape)###[batch,num_ped,2,obs_len]
        # print('pred_ma.shape:',pred_traj_m.shape)###[batch,num_ped,2,pred_len]
        seq_m = torch.cat((obs_traj_m, pred_traj_m), dim=3)##[batch,num_ped,2,obs_len+pred_len]=[1, 4, 2, 20]
        assert obs_traj_rel.shape == obs_traj_m.shape, "obs_traj_rel.shape!=obs_ma.shape"
        assert A_obs.shape[-1] == V_obs.shape[-2], "A_obs.shape[-1]!=V_obs.shape[-2]"
        assert torch.equal(V_obs,obs_traj_rel.permute(0,3,1,2)), "V_obs!=obs_traj_rel.permute(0,3,1,2)"
        

        V_obs, V_tr, obs_traj, pred_traj_gt = trajaugmenter.augment(
            V_obs, V_tr, obs_traj, pred_traj_gt)#when input seq_m, remove the augment
        # print('V_obs.shape after aug:',V_obs.shape)##[batch,obs_len,num_ped,2]
        # print('V_tr.shape after aug:',V_tr.shape)##[batch,pred_len,num_ped,2]
        # print('obs_traj.shape after aug:',obs_traj.shape)##[batch,num_ped,2,obs_len]
        # print('pred_traj_gt.shape after aug:',pred_traj_gt.shape)##[batch,num_ped,2,pred_len]
        # print('A_obs.shape after aug:',A_obs.shape)
        assert obs_traj.shape == V_obs.permute(0,2,3,1).shape, "obs_traj.shape!=V_obs.permute(0,2,3,1).shape"
        if A_obs.shape[-1] != V_obs.shape[-2]:
            _,A_obs = seq_to_graph(obs_traj,V_obs.permute(0,2,3,1))
        assert A_obs.shape[-1] == V_obs.shape[-2], "A_obs.shape[-1]!=V_obs.shape[-2]"

        ## compute maneuver label on obs_traj and pred_traj_gt, which is the real trajectory not relative position
        seq = torch.cat((obs_traj, pred_traj_gt), dim=3)##[batch,num_ped,2,obs_len+pred_len]=[1, 5, 2, 20]
        seq = seq.permute(0,3,1,2)##[batch,obs_len+pred_len,num_ped,2]
        seq_ma = Compute_ma.calculate_maneuvers(seq)##[batch,obs_len+pred_len,num_ped,4]
        # seq_ma = seq_ma[...,2:]##[batch,obs_len+pred_len,num_ped,2]
        ## save seq_ma to npy
        # seq_ma = seq_ma.cpu().numpy()
        # np.save('seq_ma.npy',seq_ma)##

        V_obs, V_tr, A_obs, obs_traj = V_obs.cuda().double(), V_tr.cuda(
        ).double(), A_obs.cuda().double(), obs_traj.cuda().double()
        ################# CRF loss Compute ###########################
        optimizer.zero_grad()
        seq_ma = seq_ma.to(V_obs.device)###在每一个seq 中计算的maneuver label
        # print('seq_ma.shape before model~~~~~~~~~~~:', seq_ma.shape)###[batch,num_ped,2,obs_len+pred_len]
        seq_m = seq_m.to(V_obs.device)##[batch,num_ped,2,obs_len+pred_len]=[1, 4, 2, 20]，在原始数据集中计算的maneuver label
        seq_m = seq_m.permute(0,3,1,2)###[batch,obs_len+pred_len,num_ped,2]
        # print('seq_m.shape before model~~~~~~~~~~~:', seq_m.shape)###[batch,num_ped,2,obs_len+pred_len]
        ###
        # print('obs_traj.shape:',obs_traj.shape)###[batch,num_ped,2,obs_len]
        # print('V_obs.shape:',V_obs.shape)
        # print('model',model)
        
        V_pred,crf_loss,obs_pred = model([V_obs.permute(0, 3, 1, 2),seq_ma.permute(0, 3, 1, 2),obs_traj], A_obs.squeeze(),train=True)
        # V_pred,crf_loss = model([V_obs.permute(0, 3, 1, 2),seq_m.permute(0,2,3,1)], A_obs.squeeze())
        # print('obs_pred.shape:',obs_pred.shape)###[batch,obs_len,num_ped,2]
        V_pred = V_pred.permute(0, 2, 3, 1)
        # print('V_pred.shape:',V_pred.shape)###[KSTEPS,obs_len,num_ped,2]

        #Loss
        crf_loss = crf_loss.cpu()
        # batch_loss += graph_loss(V_pred, V_tr, V_obs)
        batch_crf_loss += crf_loss
        total_crf_loss += batch_crf_loss.item()
        batch_loss += graph_loss(V_pred, V_tr, V_obs)
        total_loss += batch_loss.item()
        ## obs loss train on the obs_traj, use MSE loss
        batch_obs_loss += graph_loss(obs_pred, obs_traj.permute(0,3,1,2),V_obs)
        total_obs_loss += batch_obs_loss.item() 
        
        # print(f'batch_loss:{batch_loss.item()},crf_loss:{crf_loss.item()}')

        #Learn
        # print('----------cnt:',cnt)
        if cnt % args.batch_size == 0 and cnt != 0:
            # print('loss backward')
            fusion_loss = fusion_loss / args.batch_size
            batch_loss = batch_loss / args.batch_size
            batch_crf_loss = batch_crf_loss / args.batch_size
            batch_obs_loss = batch_obs_loss / args.batch_size
            # fusion_loss.backward()
            # batch_loss.backward()
            if epoch <args.crf_epoch:##=100
                batch_crf_loss.backward()
                # batch_obs_loss.backward(retain_graph=True)
            elif epoch < 201:
                batch_loss.backward()
            elif epoch < 251:
                batch_crf_loss.backward()
                # batch_obs_loss.backward(retain_graph=True)
            elif epoch < 301:
                batch_loss.backward()

            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.clip_grad)
            optimizer.step()
            #Log
            print(args.tag, ' |TRAIN:', '\t Epoch:', epoch, '\t Batch loss:',
                  batch_loss.item(),'\t CRF loss:',crf_loss.item())


            #Reset
            batch_loss = 0
            batch_crf_loss = 0
            batch_obs_loss = 0
    metrics['train_loss'].append(total_loss / (cnt + 1))
    metrics['crf_loss'].append(total_crf_loss/ (cnt + 1))
    metrics['obs_loss'].append(total_obs_loss/ (cnt + 1))


iteration = 0

def vald():
    global metrics, loader_val, constant_metrics, iteration
    model.eval()
    total_loss = 0
    batch_loss = 0

    with torch.no_grad():
        for cnt, batch in enumerate(loader_val):

            #Get data
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr,_,_ = batch

            # Compute maneuver label on obs_traj and pred_traj_gt, which is the real trajectory not relative position
            seq = torch.cat((obs_traj, pred_traj_gt), dim=3)##[batch,num_ped,2,obs_len+pred_len]=[1, 5, 2, 20]
            seq = seq.permute(0,3,1,2)##[batch,obs_len+pred_len,num_ped,2]
            # seq_ma = Compute_ma.forward(seq)##
            seq_ma = torch.zeros_like(seq)#[batch,obs_len+pred_len,num_ped,2]

            #Forward
            V_obs, V_tr, A_obs, obs_traj = V_obs.cuda().double(), V_tr.cuda(
            ).double(), A_obs.cuda().double(), obs_traj.cuda().double()

            seq_ma = seq_ma.to(V_obs.device)
            V_pred,_,_ = model([V_obs.permute(0, 3, 1, 2),seq_ma.permute(0, 3, 1, 2),obs_traj], A_obs.squeeze())
            # V_pred,_ = model([V_obs.permute(0, 3, 1, 2),obs_ma.permute(0, 3, 1, 2)], A_obs.squeeze())
            V_pred = V_pred.permute(0, 2, 3, 1)

            #Loss
            total_loss += graph_loss(V_pred, V_tr, V_obs).item()
            # batch_loss += graph_loss(V_pred, V_tr, V_obs)
            # total_loss += batch_loss.item()
            # batch_loss = 0

        print(args.tag, ' |VALD:', '\t Iteration:', iteration, '\t Loss:',
              total_loss / (cnt + 1))
        metrics['val_loss'].append(total_loss / (cnt + 1))
        store_per = 0.05 * constant_metrics['min_val_loss']

        # if (constant_metrics['min_val_loss'] -
        #         metrics['val_loss'][-1]) > store_per:
        ##this is means that metrics['val_loss'][-1]<0.95*constant_metrics['min_val_loss']
        if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:## when the dataset is train
            constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
            constant_metrics['min_val_epoch'] = iteration
            torch.save(model.state_dict(),
                       checkpoint_dir + 'val_best.pth')  # OK
    iteration += 1


print('Training started ...')
print('train start time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
for epoch in tqdm(range(args.num_epochs),desc='Epochs',colour = 'blue'):
    start_time = time.time()
    
    train(epoch)
    vald()

    scheduler.step()

    print('*' * 30)
    print(args.tag, ' |Epoch:', args.tag, ":", epoch)
    for k, v in metrics.items():
        if len(v) > 0:
            print(k, v[-1])

    print(constant_metrics)
    print('*' * 30)
    for g in optimizer.param_groups:
        print("------------->LR = ", g['lr'])
    with open(checkpoint_dir + 'metrics.pkl', 'wb') as fp:
        pickle.dump(metrics, fp)

    with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
        pickle.dump(constant_metrics, fp)

print('train end time:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

print('Training completed and the total training time length is:',time.strftime('%H:%M:%S',time.gmtime(time.time()-start_time))) 
