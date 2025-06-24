#################################################
import torch
# from TorchCRF import CRF
from torchcrf import CRF
from torch import nn  
import numpy as np


class MLP(nn.Module):
    def __init__(self, obs_len, hidden_dim, pred_len, num_labels):
        super(MLP, self).__init__()
        self.obs_len = obs_len
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.num_labels = num_labels
        
        self.fc1 = nn.Linear(obs_len * hidden_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, pred_len * num_labels)
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
class crf_loss_c(nn.Module):
    def __init__(self,hidden_dim = 4,num_labels=3,obs_len = 8,pred_len = 12) -> None:
      super().__init__()

      self.crf_lat = CRF(num_labels,batch_first = True)
      self.crf_lon = CRF(num_labels,batch_first = True)
      self.fc_lat = nn.LazyLinear(num_labels)
      self.fc_lon = nn.LazyLinear(num_labels)
      self.mlp = MLP(obs_len, hidden_dim, pred_len, num_labels)


    def forward(self,e_list,obs_seq_ma,pred_seq_ma):
      e_lat = e_list[0]##lateral emission shape is [KSTEPS*num_ped,obs_len,hidden_dim]
      e_lon = e_list[1]##longitudinal emission shape is [KSTEPS*num_ped,obs_len,hidden_dim]
      e_lat = self.mlp(e_lat)
      e_lon = self.mlp(e_lon)
      # print('e_lat:',e_lat.shape)###[KSTEPS*num_ped,pred_len=12,num_labels=3]
      # print('e_lon:',e_lon.shape)###[KSTEPS*num_ped,pred_len=12,num_labels=3]
      # emission = emission.reshape
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
      loss_lat = self.crf_lat(e_lat,labels_lat)## loss.shape is [batch]
      loss_lon = self.crf_lon(e_lon,labels_lon)## loss.shape is [batch]
      loss = loss_lat + loss_lon

      return -loss

    def crf_process(self, e_list, obs_seq_ma, pred_seq_ma):
        e_lat = e_list[0]  ##lateral emission shape is [KSTEPS*num_ped,obs_len,hidden_dim]
        e_lon = e_list[1]  ##longitudinal emission shape is [KSTEPS*num_ped,obs_len,hidden_dim]
        e_lat = self.mlp(e_lat)
        e_lon = self.mlp(e_lon)
        labels_lat = pred_seq_ma[:, 0, :, :]  ## shape is [KSTEPS,pred_len,num_ped]
        labels_lat = labels_lat.reshape(-1, labels_lat.shape[1])  ###[KSTEPS*num_ped,pred_len]
        labels_lon = pred_seq_ma[:, 1, :, :]
        labels_lon = labels_lon.reshape(-1, labels_lon.shape[1])  ###[KSTEPS*num_ped,pred_len]
        labels_lat = labels_lat.long()
        labels_lon = labels_lon.long()
        return e_lat, e_lon, labels_lat, labels_lon
    def decode_crf(self,e_list,obs_seq_ma,pred_seq_ma):
        e_lat, e_lon, labels_lat, labels_lon = self.crf_process(e_list, obs_seq_ma, pred_seq_ma)
        pred_lat = self.crf_lat.decode(e_lat)
        pred_lon = self.crf_lon.decode(e_lon)
        return [pred_lat, pred_lon]
    
class BatchManeuverCalculator(nn.Module):
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def calculate_distance(self, point_a, point_b):
        return torch.sqrt((point_a[..., 0] - point_b[..., 0])**2 + (point_a[..., 1] - point_b[..., 1])**2)

    def calculate_velocity(self, point_a, point_b, time_delta):
        distance = self.calculate_distance(point_a, point_b)
        time_delta_seconds = time_delta * 0.4  # Assuming each frame is 0.4s apart
        velocity = distance / time_delta_seconds
        return velocity

    def calculate_tangent_angle(self, prev_point, next_point):
        delta_x = next_point[..., 0] - prev_point[..., 0]
        delta_y = next_point[..., 1] - prev_point[..., 1]
        angle = torch.atan2(delta_y, delta_x)
        return angle

    def calculate_maneuvers(self, data):
        # Convert input data to a PyTorch tensor and send it to the specified device (GPU)
        data_tensor = torch.tensor(data, dtype=torch.float32).to(self.device)

        batch_size, sequence_length, num_ped, _ = data_tensor.shape
        maneuvers_data = torch.zeros((batch_size, sequence_length, num_ped, 2), device=self.device)

        # Extract previous, current, and next points for all pedestrians and time steps in the batch
        prev_points = data_tensor[:, :-2]
        current_points = data_tensor[:, 1:-1]
        next_points = data_tensor[:, 2:]

        # Calculate velocities
        velocity_prev = self.calculate_velocity(prev_points, current_points, 1)
        velocity_next = self.calculate_velocity(current_points, next_points, 1)
        velocity = (velocity_prev + velocity_next) / 2

        # Compute angle differences
        current_angle = self.calculate_tangent_angle(prev_points, next_points)
        prev_angle = self.calculate_tangent_angle(prev_points, current_points)
        angle_diff = torch.abs(current_angle - prev_angle)

        # Lateral maneuvers
        angle_threshold = torch.clamp(0.1 * velocity, 0.05, 0.2)
        lateral_maneuver = torch.zeros_like(current_angle)
        lateral_maneuver[angle_diff > angle_threshold] = torch.where(
            current_angle[angle_diff > angle_threshold] > prev_angle[angle_diff > angle_threshold],
            1.0,  # Turn Left
            2.0   # Turn Right
        )

        # Longitudinal maneuvers
        longitudinal_maneuver = torch.zeros_like(velocity_next)
        longitudinal_maneuver[velocity_next > velocity_prev * 1.1] = 2.0  # Accelerate
        longitudinal_maneuver[velocity_next < velocity_prev * 0.9] = 1.0  # Decelerate

        # Update maneuvers data
        maneuvers_data[:, 1:-1] = torch.stack((lateral_maneuver, longitudinal_maneuver), dim=-1)

        return maneuvers_data