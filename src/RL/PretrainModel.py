from torch.utils.data.dataset import Dataset, random_split
from gzip import GzipFile
import numpy as np 
import torch as th
import gym
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import one_hot
import json
import matplotlib.pyplot as plt
import os
from gym import spaces
from math import exp, floor, ceil
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from gym.spaces import Box
from src.Utils.DataUtils import load_cell_prediction_npz

class CellDataSet:
    def __init__(self, data):
        self.input = data[0]
        self.target = data[1]
        
    def __getitem__(self, idx):
        return {'theta': th.tensor(self.input['theta'][idx],), 'lidar': th.tensor(self.input['lidar'][idx]).unsqueeze(1), 'cameraRight': th.tensor(self.input['cameraRight'][idx]), 'cameraLeft': th.tensor(self.input['cameraLeft'][idx]), 'cameraFront': th.tensor(self.input['cameraFront'][idx]), 'cameraRear': th.tensor(self.input['cameraRear'][idx])}, th.tensor(self.target[idx])

    def __len__(self): # Returns Num of Sequences
        return self.target.shape[0]
      
def get_expert_data():    
  FILE_PATH = 'data/ExpertDataset/'
  expert_observations = []
  
  return ExpertDataSet(expert_observations)

  # Only the 3 fingers


# train_size = int(0.8 * len(expert_dataset))

# test_size = len(expert_dataset) - train_size

# train_expert_dataset, test_val_expert_dataset = random_split(
#     expert_dataset, [train_size, test_size]
# )

# test_val_split = len(test_val_expert_dataset)/2
# val_expert_dataset, test_expert_dataset = random_split(
#     test_val_expert_dataset, [floor(test_val_split), ceil(test_val_split)]
# )

# print("test_expert_dataset: ", len(test_expert_dataset))
# print("train_expert_dataset: ", len(train_expert_dataset))
# print('val_expert_dataset: ', len(val_expert_dataset))

def get_accuracy(model, data, dtype):
    count = 0
    total = 0
    for i in data:
      x = model.predict(i[0], deterministic = True)[0]
      if all(x == i[1]):
        count += 1
      total += 1
    print(dtype, count, total, count/total)

def pretrain_agent(
    student, 
    batch_size = 16, 
    epochs = 1000, 
    learning_rate = 1e-3, 
    log_interval = 5, 
    no_cuda = False, 
    seed = 1, 
    patience = 10):

  use_cuda = not no_cuda and th.cuda.is_available()
#   th.manual_seed(seed)
  device = th.device("cuda" if use_cuda else "cpu")
  print(device)
#   kwargs = {"num_workers": 1, "pin_memory" : True} if use_cuda else {}
  kwargs = {}
  criterion = nn.BCELoss()
  

  def train(model, device, train_loader, optimizer):
    
    model.train()
    model.to('cpu')
    
    for data, target in train_loader:
        target = target.to(device)
        for k, v in data.items():
         data[k] = v.float().to('cpu')
        data['theta'] = data['theta'].unsqueeze(1)
        data['lidar'] = data['lidar'].squeeze()

        optimizer.zero_grad()
        output_target = th.nn.functional.one_hot(target.long(), num_classes=16).float()
        output_predition = model(data)
        loss = th.nn.functional.cross_entropy(output_predition, output_target)
        loss.backward()
        optimizer.step()
        print('Loss: ', loss.item())
    

    

  def validation(model, device, val_loader):
    model.eval()
    loss_total = 0
    with th.no_grad():
      for data, target in val_loader:
        target = target.to(device)
        for k, v in data.items():
          data[k] = v.float().to(device)

        dist = model.get_distribution(data)
        action_prediction = [i.logits for i in dist.distribution]
        #   action_prediction = [i.probs for i in dist.distribution]
        target = target.long()

        loss1 = criterion(action_prediction[0], target[:, 0])
        loss2 = criterion(action_prediction[1], target[:, 1])
        loss3 = criterion(action_prediction[2], target[:, 2])
        val_loss = loss1 + loss2 + loss3
        loss_total += val_loss.item()

    val_loss = loss_total / len(val_loader.dataset)
    print('Validation_loss:', val_loss)
    return val_loss


  def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with th.no_grad():
      for data, target in test_loader:
        target = target.to(device)
        for k, v in data.items():
          data[k] = v.to(device)

        dist = model.get_distribution(data)
        action_prediction = [i.logits for i in dist.distribution]
        #   action_prediction = [i.probs for i in dist.distribution]
        target = target.long()

        loss1 = criterion(action_prediction[0], target[:, 0])
        loss2 = criterion(action_prediction[1], target[:, 1])
        loss3 = criterion(action_prediction[2], target[:, 2])
        loss = loss1 + loss2 + loss3

        test_loss += loss.item()

    test_loss = test_loss / len(test_loader.dataset)
    # print('Validation_loss:', val_loss)
        # test_loss = criterion(action_prediction, target)
    # test_loss /= len(test_loader.dataset)
    print(f'test set: average loss: {test_loss:.4f}')

  train_loader = th.utils.data.DataLoader(dataset = CellDataSet(load_cell_prediction_npz(os.path.join(os.getcwd() , 'data', 'CellPrediction',))), batch_size = batch_size, shuffle = True,  **kwargs)
  # test_loader = th.utils.data.DataLoader(dataset = test_expert_dataset, batch_size = batch_size, shuffle = True, **kwargs)
  # val_loader = th.utils.data.DataLoader(dataset = val_expert_dataset, batch_size = batch_size, shuffle = True, **kwargs)

  student.to('cpu')

  optimizer = optim.Adam(student.parameters(), lr = learning_rate,
        eps=1e-5,
        weight_decay=0,)
  
  print('Training...')
  for epoch in range(1, epochs+1):
    train(student, device, train_loader, optimizer)
    if epoch % 50 == 0:
        student.save_checkpoint(os.path.join(os.getcwd(), 'src', 'Networks', 'SavedModels', f'CellPredictionNetwork_Epoch_{epoch}.zip'))
