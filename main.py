from src.Env.EpuckEnviornment import EpuckEnv
from src.Networks.CellPredictionNetwork import CellPredictionNetwork
from src.RL.PretrainModel import pretrain_agent
from sb3_contrib import ppo_recurrent
from src.Utils.Vector2 import Vector2
from src.Utils.DataUtils import save_cell_prediction_npz
import os

SAVE_DIR = os.path.join(os.getcwd(), 'data', 'CellPrediction')

def main():
    ### DATA COLLECTION ###
    env = EpuckEnv(connection=False)
    # obs = []
    # done = False
    # obs.append(env.reset())
    # while True:
    #     action = env.action_space.sample()
    #     state, reward, done, info = env.step(action)
    #     obs.append(state)
    #     if done:
    #         index = len(os.listdir(SAVE_DIR))
    #         save_cell_prediction_npz(obs, os.path.join(SAVE_DIR, f'{index}.npz'))
    #         obs = [env.reset()]
    model = CellPredictionNetwork(env.observation_space)
    pretrain_agent(model, no_cuda=True)
    model.save_checkpoint()
if __name__ == '__main__':
    exit(main())


    
    