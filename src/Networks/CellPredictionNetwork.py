from torch import nn
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os

class CellPredictionNetwork(nn.Module):
    def __init__(self, observation_space):
        super(CellPredictionNetwork, self).__init__()
        extractors = {}
        total_concat_size = 0
        self.checkpoint_file = os.path.join(os.getcwd(), 'src', 'Networks', 'SavedModels', 'CellPredictionNetwork.zip')
        
        ### FIRST BLOCK ###
        
        self.__block_0_size__ = {
                'sensor_positioning': 0,
                'image': 0,
            }
        for key, subspace in observation_space.spaces.items():
            # if key == 'delta_t':
            #     extractors[key] = nn.Sequential(
            #         ## Network
            #     )
            #     total_concat_size+=1
            # elif key == 'left_distance':
            #     extractors[key] = nn.Sequential(
            #         ## Network
            #     ) 
            #     total_concat_size+=1
            # elif  key == 'right_distance':
            #     extractors[key] = nn.Sequential(
            #          ## Network
            #     )
            #     total_concat_size+=1
            # elif key == 'front_distance':
            #     extractors[key] = nn.Sequential(
            #         ## Network
            #     )
            #     total_concat_size+=1
            # elif key == 'rear_distance':
            #     extractors[key] = nn.Sequential(
            #         ## Network
            #     )
            #     total_concat_size+=1
            # elif key == 'previous_pos':
            #     extractors[key] = nn.Sequential(
            #         ## Network
            #     )
            #     total_concat_size+=1
            # elif key == 'current_pos':
            #     extractors[key] = nn.Sequential(
            #        ## Network
            #     ) 
            #     total_concat_size+=1
            # elif key == 'omega':
            #     extractors[key] = nn.Sequential(
            #        ## Network
            #     )
            #   total_concat_size+=1
            if key == 'theta':
                extractors[key] = nn.Sequential(
                   nn.Linear(1, 720),
                   nn.LeakyReLU(),
                )
                self.__block_0_size__['sensor_positioning']+=720
            
                total_concat_size+=10
            elif key == 'lidar':
                extractors[key] = nn.Sequential(
                    # Network
                    nn.Linear(360, 720),
                    nn.LeakyReLU(),
                )
                self.__block_0_size__['sensor_positioning']+=720
            elif 'camera' in key:
                extractors[key] = nn.Sequential(
                    # we have a 3x80x80 image
                    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Flatten(),
                )
                self.__block_0_size__['image']+=12800
            
        ### SECOND BLOCK ###  
        self.__encoded_positioning_size__ = 1440
        self.__encoded_positioning__ = nn.Sequential(
            nn.Linear(720, 1440),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1440, 1440),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),     
        )
        
        self.__decoded_imaging_size__ = 3200
        self.__decoded_imaging__ = nn.Sequential(
            nn.Linear(12800 * 4, 12800),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(12800, 6400),
            nn.LeakyReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(6400, 3200),
            nn.LeakyReLU(),
            nn.Dropout(p=0.05),
        )
        
        ### THIRD BLOCK ###
        self.__block_3_size__ = 1200
        self.__block_3__ = nn.Sequential(
            nn.Linear(self.__encoded_positioning_size__ + self.__decoded_imaging_size__, self.__block_3_size__),
            nn.LeakyReLU(),
        )
        
        ### OUTPUT BLOCK ###
        self.__output_block_size__ = 16
        self.__output_block__ = nn.Sequential(
            nn.Linear(self.__block_3_size__, self.__output_block_size__),
            nn.Softmax(dim=1),     
        )
        
        self.extractors = nn.ModuleDict(extractors)
    
    def forward(self, observations):
        ### BLOCK 1 ###
        encoded_positioning_tensors = {}
        encoded_imaging_tensors = []
        for key, extractor in self.extractors.items():
            if 'camera' in key:
                encoded_imaging_tensors.append(extractor(observations[key]))
                continue
            else:
                encoded_positioning_tensors[key] = extractor(observations[key])
                continue

        encoded_positioning = (encoded_positioning_tensors['theta'] + encoded_positioning_tensors['lidar']) / 2
        encoded_imaging_tensor = th.cat(encoded_imaging_tensors, dim=1)

        ### BLOCK 2 ###      
        encoded_positioning_output = self.__encoded_positioning__(encoded_positioning)
        for i in range(0, encoded_positioning_output.shape[1], int(self.__encoded_positioning_size__/2)):
            encoded_positioning_output[:, i:i+int(self.__encoded_positioning_size__/2)] += encoded_positioning
            
        decoded_imaging_tensor = self.__decoded_imaging__(encoded_imaging_tensor)
        for i in range(0, decoded_imaging_tensor.shape[1], self.__decoded_imaging_size__):
            decoded_imaging_tensor += encoded_imaging_tensor[:, i:i+self.__decoded_imaging_size__]
        decoded_imaging_tensor /= (encoded_imaging_tensor.shape[1] /self.__decoded_imaging_size__)
        
        ### BLOCK 3 ###
        block_3_input = th.cat([encoded_positioning_output, decoded_imaging_tensor], dim=1)
        block_3_output = self.__block_3__(block_3_input)
        
        ### OUTPUT BLOCK ###
        return self.__output_block__(block_3_output)
    
    def save_checkpoint(self, file=None):
        if file != None:
            th.save(self.state_dict(), file)
        elif self.checkpoint_file != None:
            th.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(th.load(self.checkpoint_file)) 
    
    