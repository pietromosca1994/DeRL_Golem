from utils import plotTensorboard, evalAgent
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

tensorflow_path='/home/pietro/repos/DeRL_Golem/src/training/2022-03-30T14_55/logs'
model_path='/home/pietro/repos/DeRL_Golem/src/training/2022-03-30T14_55/models/federated_model_episode_8'

#plotTensorboard(tensorflow_path, -1)
evalAgent(model_path)
'''
model=PPO.load(model_path)
env = make_vec_env("LunarLander-v2")
obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    '''

