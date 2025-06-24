import argparse
import torch
if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')

def parameter_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--w_crfloss',
                      type=float,
                      default=0.01,
                      help='crf loss weight')

  #Data specifc paremeters
  parser.add_argument('--obs_seq_len', type=int, default=8)
  parser.add_argument('--pred_seq_len', type=int, default=12)
  parser.add_argument('--dataset',
                      default='hotel_small',
                      help='eth,hotel,univ,zara1,zara2,sdd')

  #Training specifc parameters
  parser.add_argument('--device',
                        type=str,
                        default=device,
                        help='')
  parser.add_argument('--batch_size',
                      type=int,
                      default=32,
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
  parser.add_argument('--crf_epoch',
                      type=int,
                      default=100,
                      help='number of epochs to train crf loss')
  parser.add_argument('--lr_sh_rate',
                      type=int,
                      default=50,
                      help='number of steps to drop the lr')
  parser.add_argument('--weight_decay',type=float,default=1e-4,help='weight decay')
  

  parser.add_argument('--tag', default='tag', help='personal tag for the model ')
  parser.add_argument('--KSTEPS', type= int,default=1, help= 'number of results to predict')
  args = parser.parse_args()
  return args