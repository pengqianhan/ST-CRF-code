import torch
import numpy as np
from utils import *
from metrics import *
from model import STCRF 
from CFG import CFG
import os
from torchinfo import summary

model = STCRF(spatial_input=CFG["spatial_input"],
            spatial_output=CFG["spatial_output"],
            temporal_input=CFG["temporal_input"],
            temporal_output=CFG["temporal_output"],
            stgcn_layer=2).cuda().double()

# Set parameters
batch_size = KSTEPS = 1
num_peds = 2
obs_len = 8
pred_len = 12

# Define shapes
V_obs_shape = (batch_size, obs_len, num_peds, 2)
seq_ma_shape = (batch_size, obs_len+pred_len, num_peds, 2)
obs_traj_shape = (batch_size, num_peds, 2, obs_len)
A_obs_shape = (batch_size, obs_len, num_peds, num_peds)
# batch,obs_len,num_ped,num_ped]

# Create dummy inputs and move them to CUDA
V_obs = torch.randn(V_obs_shape).cuda().double()
seq_ma = torch.randn(seq_ma_shape).cuda().double()
obs_traj = torch.randn(obs_traj_shape).cuda().double()
A_obs = torch.randn(A_obs_shape).cuda().double()

input1 = [V_obs.permute(0, 3, 1, 2), seq_ma.permute(0, 3, 1, 2), obs_traj]
input2 = A_obs.squeeze()
output,_,_ = model(input1, input2)
print(output.shape)
# KSTEPS value
# KSTEPS = 1  # Replace with your actual KSTEPS value

summary(model, 
        input_data=(input1, input2),
        depth=0,
        verbose=1)
'''
====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
STCRF                                              [1, 2, 12, 2]             6,107
====================================================================================================
Total params: 6,107
Trainable params: 6,107
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 0.01
====================================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.02
Params size (MB): 0.01
Estimated Total Size (MB): 0.04
====================================================================================================

'''