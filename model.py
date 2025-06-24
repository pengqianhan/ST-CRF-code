import torch
import torch.nn as nn
import torch.distributions as tdist
from utils_stgcn import ConvTemporalGraphical, st_gcn
# from utils_crf import *
import math
from crf import CRF
from param_parser import parameter_parser
args = parameter_parser()

class Itention_encoder_lstm(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_lstm_layers: int =1,drop_lstm:float=0.0):##drop_lstm:float=0.1
        super().__init__()
        # self.label_size = label_size
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim//2, num_layers=num_lstm_layers, batch_first=True, bidirectional=True)
        self.drop_lstm = nn.Dropout(drop_lstm)
        # self.hidden2tag = nn.Linear(hidden_dim, self.label_size)
    def init_hidden(self):
        return (torch.randn(self.num_lstm_layers*2,1, self.hidden_dim // 2).double(),
                torch.randn(self.num_lstm_layers*2,1, self.hidden_dim // 2).double())
    def forward(self,x):
        ## x shape [batch=1,2,obs_len,num_ped]
        h0,c0 = self.init_hidden()
        h0 = h0.to(x.device)
        c0 = c0.to(x.device)
        x = x.reshape(-1,x.shape[2],x.shape[1])##[batch=KSTEPS*num_ped,obs_len,2]
        h0 = h0.repeat(1,x.shape[0],1)##[num_layers*2,batch=KSTEPS*num_ped,hidden_dim//2]
        c0 = c0.repeat(1,x.shape[0],1)
        # print("x.shape in Itention_encoder_lstm forward: ",x.shape)##[batch=KSTEPS*num_ped,obs_len,2]
        lstm_out, (h0,c0) = self.lstm(x,(h0,c0))
        lstm_out = self.drop_lstm(lstm_out)
        return lstm_out

class MLP(nn.Module):
    def __init__(self, obs_len, hidden_dim, pred_len, num_labels):
        super(MLP, self).__init__()
        self.obs_len = obs_len
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.num_labels = num_labels
    
        self.fc1 = nn.Linear(obs_len * hidden_dim, obs_len * hidden_dim*4)
        self.fc2 = nn.Linear(obs_len * hidden_dim*4, obs_len * hidden_dim*2)
        self.fc3 = nn.Linear(obs_len * hidden_dim*2, pred_len * num_labels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
				
        # print('x.shape in mlp:',x.shape)####x shape: [KSTEPS*num_ped,obs_len,hidden_dim]
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # 将输入数据展平
        # print('x.shape in mlp:',x.shape)###[KSTEPS*num_ped,obs_len*hidden_dim]assert x.shape[1] == self.obs_len * self.hidden_dim, f"Expected second dimension to be {self.obs_len * self.hidden_dim}, but got {x.shape[1]} instead."
        # assert x.shape[1] == self.obs_len * self.hidden_dim, f"Expected second dimension to be {self.obs_len * self.hidden_dim}, but got {x.shape[1]} instead."
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = x.view(batch_size, self.pred_len, self.num_labels)  # 重塑输出维度
        return x
    
class crf_e(nn.Module):
    def __init__(self,hidden_dim = 2,num_labels=3,obs_len = 8,pred_len = 12) -> None:
      super().__init__()
      self.mlp = MLP(obs_len, hidden_dim, pred_len, num_labels)

    def forward(self,e_list,obs_seq_ma,pred_seq_ma):
      e_lat = e_list[0]##lateral emission shape is [KSTEPS*num_ped,obs_len,hidden_dim]
      e_lon = e_list[1]##longitudinal emission shape is [KSTEPS*num_ped,obs_len,hidden_dim]
    #   print("e_lat.shape in crf_e forward: ",e_lat.shape)##[KSTEPS*num_ped,obs_len,hidden_dim]
    #   print("e_lon.shape in crf_e forward: ",e_lon.shape)
      e_lat = self.mlp(e_lat)
      e_lon = self.mlp(e_lon)
      # print('e_lat:',e_lat.shape)###[KSTEPS*num_ped,pred_len=12,num_labels=3]
      # print('e_lon:',e_lon.shape)###[KSTEPS*num_ped,pred_len=12,num_labels=3]

      # print('obs_seq_ma.shape:',obs_seq_ma.shape)##[KSTEPS=1,2,obs_len,num_ped]
      # print('pred_seq_ma.shape:',pred_seq_ma.shape)##[KSTEPS=1,2,pred_len,num_ped]
      labels_lat = pred_seq_ma[:,0,:,:]## shape is [KSTEPS,pred_len,num_ped]
      labels_lat = labels_lat.reshape(-1,labels_lat.shape[1])###[KSTEPS*num_ped,pred_len]
      labels_lon = pred_seq_ma[:,1,:,:]
      labels_lon = labels_lon.reshape(-1,labels_lon.shape[1])###[KSTEPS*num_ped,pred_len]
      labels_lat = labels_lat.long()
      labels_lon = labels_lon.long()
    #   print('labels_lat:',labels_lat.shape)###[KSTEPS*num_ped,pred_len]
    #   print('labels_lon:',labels_lon.shape)###[KSTEPS*num_ped,pred_len]
    #   print('e_lat:',e_lat.shape)###[KSTEPS*num_ped,pred_len,num_labels]
    #   print('e_lon:',e_lon.shape)###[KSTEPS*num_ped,pred_len,num_labels]
      return [e_lat,labels_lat],[e_lon,labels_lon]
class crf_e_obs(nn.Module):
    def __init__(self,hidden_dim = 4,num_labels=3,obs_len = 8,pred_len = 12) -> None:
      super().__init__()
      self.mlp = nn.Linear(hidden_dim,num_labels)

    def forward(self,e_list,obs_seq_ma,pred_seq_ma):
      e_lat = e_list[0]##lateral emission shape is [KSTEPS*num_ped,obs_len,hidden_dim]
      e_lon = e_list[1]##longitudinal emission shape is [KSTEPS*num_ped,obs_len,hidden_dim]
    #   print("e_lat.shape in crf_e forward: ",e_lat.shape)##[KSTEPS*num_ped,obs_len,hidden_dim]
    #   print("e_lon.shape in crf_e forward: ",e_lon.shape)
      e_lat = self.mlp(e_lat)##
      e_lon = self.mlp(e_lon)

      labels_lat = obs_seq_ma[:,0,:,:]## shape is [KSTEPS,pred_len,num_ped]
      labels_lat = labels_lat.reshape(-1,labels_lat.shape[1])###[KSTEPS*num_ped,obs_len]
      labels_lon = obs_seq_ma[:,1,:,:]
      labels_lon = labels_lon.reshape(-1,labels_lon.shape[1])###[KSTEPS*num_ped,obs_len]
      labels_lat = labels_lat.long()
      labels_lon = labels_lon.long()
    #   print('labels_lat:',labels_lat.shape)###[KSTEPS*num_ped,obs_len]
    #   print('labels_lon:',labels_lon.shape)###[KSTEPS*num_ped,obs_len]
    #   print('e_lat:',e_lat.shape)###[KSTEPS*num_ped,obs_len,num_labels]
    #   print('e_lon:',e_lon.shape)###[KSTEPS*num_ped,obs_len,num_labels]
      return [e_lat,labels_lat],[e_lon,labels_lon]
class crossentropy_loss(nn.Module):
    def __init__(self) -> None:
      super().__init__()
      self.loss = nn.CrossEntropyLoss()
    def forward(self,lat_list,lon_list):
        lat_embeding = lat_list[0]
        lat_embeding = lat_embeding.reshape(-1,lat_embeding.shape[-1])
        lat_label = lat_list[1]
        lat_label = lat_label.reshape(-1)
        lon_embeding = lon_list[0]
        lon_embeding = lon_embeding.reshape(-1,lon_embeding.shape[-1])
        lon_label = lon_list[1]
        lon_label = lon_label.reshape(-1)
        loss_lat = self.loss(lat_embeding,lat_label)
        loss_lon = self.loss(lon_embeding,lon_label)
        return loss_lat+loss_lon

class crf_lat_lon(nn.Module):
    def __init__(self,num_labels_lat=3,num_labels_lon=3) -> None:
      super().__init__()
      self.crf_lat = CRF(num_labels_lat,batch_first = True)
      self.crf_lon = CRF(num_labels_lon,batch_first = True)
    def forward(self,lat_list,lon_list):
      loss_lat = self.crf_lat(lat_list[0],lat_list[1])
      loss_lon = self.crf_lon(lon_list[0],lon_list[1])
      return -loss_lat-loss_lon
    def decode_crf(self,lat_list,lon_list):
        decode_lat = self.crf_lat.decode(lat_list[0])
        decode_lon = self.crf_lon.decode(lon_list[0])
        return [decode_lat,decode_lon]
class stgcn_model(nn.Module):
    def __init__(self,output_feat=2,
                 spatial_input=2,
                 temporal_input=8,
                 num_layers=1):
        super().__init__()
        self.st_gcns = nn.ModuleList()
        kernel_size=3
        self.st_gcns.append(st_gcn(spatial_input,output_feat,(kernel_size,temporal_input)))
        for j in range(1,num_layers):
            self.st_gcns.append(st_gcn(output_feat,output_feat,(kernel_size,temporal_input)))
    def forward(self,v,a):
        for k in range(len(self.st_gcns)):
            v,a = self.st_gcns[k](v,a)
        return v,a
class decoder(nn.Module):
    def __init__(self,hidden_dim = 4,out_feat=2,obs_len = 8,pred_len = 12) -> None:
      super().__init__()
      self.out_feat = out_feat
      self.pred_len = pred_len
      self.fc1 = nn.Linear(obs_len*(hidden_dim*2+2),obs_len*(hidden_dim*2+2)*4)
      self.fc2 = nn.Linear(obs_len*(hidden_dim*2+2)*4,obs_len*(hidden_dim*2+2)*2)
      self.fc3 = nn.Linear(obs_len*(hidden_dim*2+2)*2,pred_len*out_feat)
      self.relu = nn.ReLU()

    def forward(self,x):
        ##x shape is [KSTEPS,obs_len,hiddem_dim*2+2,num_ped]
        ksteps,_,_,num_ped = x.shape
        new_x = x.reshape(ksteps*num_ped,-1)
        x = self.fc1(new_x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = x.reshape(ksteps, self.pred_len,self.out_feat,num_ped)##
        return x

class STCRF(nn.Module):
    def __init__(self,
                 spatial_input=2,
                 spatial_output=5,
                 temporal_input=8,
                 temporal_output=12,
                 stgcn_layer=1):##stgcn_layer
        super(STCRF, self).__init__()

        hidden_dim = 2
        self.It_encoder_lstm = Itention_encoder_lstm(input_dim=hidden_dim,hidden_dim=hidden_dim,num_lstm_layers=3)
        self.It_encoder_lstm2 = Itention_encoder_lstm(input_dim=hidden_dim,hidden_dim=hidden_dim,num_lstm_layers=3)
###############################STGCN############################################
        self.stgcn_model = stgcn_model(output_feat=spatial_output,spatial_input=spatial_input,temporal_input=temporal_input,num_layers=stgcn_layer)
        self.encoder_cnn = nn.Conv2d(spatial_output,
                              spatial_output,
                              3,
                              padding=1,
                              padding_mode='zeros')
        # self.act = nn.GELU()
        # self.act = nn.ReLU()
        self.act = nn.LeakyReLU(0.2)## best performance No.1
        # self.act = nn.PReLU()## not good
        self.res_obs = nn.Conv2d(spatial_output,spatial_output,1,padding=0)
        self.crf_e = crf_e(hidden_dim=hidden_dim)
        self.crf_e_obs = crf_e_obs(hidden_dim=hidden_dim)
        self.crf_lat_lon = crf_lat_lon()
        self.crossentropy_loss = crossentropy_loss()
        ##################################################################################################
        # self.decoder = decoder(hidden_dim=hidden_dim,out_feat=spatial_output,obs_len=temporal_input,pred_len=temporal_output)
        self.decoder = nn.Conv2d(in_channels=temporal_input, out_channels=temporal_output, kernel_size=(hidden_dim*2+spatial_output-1, 1))
        # self.decoder_cnn = nn.Conv2d(in_channels=temporal_output, out_channels=temporal_output, kernel_size=3,padding=1)
        self.decoder_linear = nn.Linear(temporal_input*(hidden_dim*2+spatial_output-1), temporal_output*2)
        self.res_pred = nn.Conv2d(temporal_output,temporal_output,1,padding=0)
        
        self.noise = tdist.multivariate_normal.MultivariateNormal(
            torch.zeros(2), torch.Tensor([[1, 0], [0, 1]]))
        self.noise_w = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.noise_w_traj = nn.Parameter(torch.zeros(1), requires_grad=True)

    # def forward(self, v_seq_ma, a, KSTEPS=20):
    def forward(self, v_seq_ma, a,train = True, KSTEPS=args.KSTEPS):
        # print('a.shape:',a.shape)##[KSTEPS,2,obs_len,num_ped]
        v = v_seq_ma[0].contiguous()
        # print("v.shape in forward: ",v.shape)##[1,2,obs_len,num_ped]
        _,_,obs_len,num_ped=v.shape
        seq_ma = v_seq_ma[1]
        obs_traj = v_seq_ma[2]##[batch,num_ped,2,obs_len]
        obs_traj = obs_traj.reshape(v.shape)
        # print("obs_traj.shape in forward: ",obs_traj.shape)##[KSTEPS,2,obs_len,num_ped]
        noise = self.noise.sample((KSTEPS, )).unsqueeze(-1).unsqueeze(-1).to(
            v.device).double().contiguous()
        obs_ma = seq_ma[:,:,:obs_len,:]
        obs_ma = obs_ma.repeat(KSTEPS,1,1,1)### 让模型可以有多个输出
        pred_ma = seq_ma[:,:,obs_len:,:]
        pred_ma = pred_ma.repeat(KSTEPS,1,1,1)
        # seq_ma = seq_ma.permute(0,2,3,1)
        #Combine Vectorized Noise
        # args = parameter_parser()
        if args.dataset == 'eth':
            weight = 1.1
        else:
            weight =1

        v = v + self.noise_w  * noise* weight
        # print('v.shape:',v.shape)
        obs_traj = obs_traj + self.noise_w_traj * noise
        # print('self.noise_w----------------------',self.noise_w)
        
        # it_e_lat = self.It_encoder_lstm(v)## 
        # it_e_lon = self.It_encoder_lstm2(v)##
        it_e_lat = self.It_encoder_lstm(obs_traj)## 
        it_e_lon = self.It_encoder_lstm2(obs_traj)##
        # print("it_e_lat.shape in forward: ",it_e_lat.shape)##[KSTEPS*num_ped,obs_len,hiddem_dim]
        # print("it_e_lon.shape in forward: ",it_e_lon.shape)##[KSTEPS*num_ped,obs_len,hiddem_dim]
        it_e_lat_v = it_e_lat.reshape(KSTEPS,-1,obs_len,num_ped)##[KSTEPS,hiddem_dim,obs_len,num_ped]
        it_e_lon_v = it_e_lon.reshape(KSTEPS,-1,obs_len,num_ped)##[KSTEPS,hiddem_dim,obs_len,num_ped]
        # print("it_e_lat_v.shape in forward: ",it_e_lat_v.shape)##[KSTEPS,hiddem_dim,obs_len,num_ped]
        # print("it_e_lon_v.shape in forward: ",it_e_lon_v.shape)##[KSTEPS,hiddem_dim,obs_len,num_ped]
        # print('v.shape:',v.shape)##[KSTEPS,2,obs_len,num_ped]
        # print('a.shape:',a.shape)##[KSTEPS,2,obs_len,num_ped]
        v,a = self.stgcn_model(v,a)
        # print("v.shape in forward after st_gcns: ",v.shape)##[KSTEPS,5,obs_len,num_ped]
        ### add CNN after stgcn###
        v = self.act(self.encoder_cnn(v)) + self.res_obs(v)
        # print("v.shape in forward after encoder_cnn: ",v.shape)##[KSTEPS,5,obs_len,num_ped]

        # print("v.shape in forward after encoder_lstm: ",v.shape)##[KSTEPS,5,obs_len,num_ped]
        fusion_e_new = torch.cat((v,it_e_lat_v,it_e_lon_v),dim=1) ##[KSTEPS,hiddem_dim*2+spatial_output,obs_len,num_ped]
        # print("fusion_e_new.shape in forward: ",fusion_e_new.shape)
        fusion_e_new = fusion_e_new.permute(0,2,1,3)##[KSTEPS=20,obs_len,hiddem_dim*2+2,num_ped]
        emission_list = [it_e_lat,it_e_lon]
        use_obs_labels = True
        if use_obs_labels:
            # emission_list[0].shape =[KSTEPS*num_ped,obs_len,hiddem_dim]
            # emission_list[1].shape =[KSTEPS*num_ped,obs_len,hiddem_dim]
            obs_pred = emission_list[0] + emission_list[1]
            obs_pred = obs_pred.reshape(KSTEPS,obs_len,num_ped,-1)
            # print('obs_pred.shape:',obs_pred.shape)##[KSTEPS,obs_len,num_ped,hiddem_dim=2]
            lat_list,lon_list = self.crf_e_obs(emission_list,obs_ma,pred_ma)
            # print('lat_list[0] shape:',lat_list[0].shape)
            # print('lat_list[1] shape:',lat_list[1].shape)
            # print('lon_list[0] shape:',lon_list[0].shape)
            # print('lon_list[1] shape:',lon_list[1].shape)
        else:
            lat_list,lon_list = self.crf_e(emission_list,obs_ma,pred_ma)
        if train:
            # print("In train mode, model output CRF loss")
            crf_loss = self.crf_lat_lon(lat_list,lon_list)
            # crf_loss = self.crossentropy_loss(lat_list,lon_list)
            decodeIn = None
        else:
            crf_loss = 0.0
            decodeIn = self.crf_lat_lon.decode_crf(lat_list,lon_list)
        # print("fusion_e_new.shape before: ",fusion_e_new.shape)##[KSTEPS=20,obs_len,hiddem_dim*2+spatial_output,num_ped]
        fut_tr = self.decoder(fusion_e_new) 
        # print("fut_tr.shape in forward: ",fut_tr.shape)##[KSTEPS=20,pred_len,2,num_ped]
        fut_tr = fut_tr.permute(0, 2, 1, 3)
        return fut_tr.contiguous(),crf_loss,obs_pred