from utils import plotTensorboard
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

tensorflow_path='/home/pietro/repos/DeRL_Golem/src/training/2022-03-29T10_55/logs'
model_path='/home/pietro/repos/DeRL_Golem/src/training/2022-03-29T10_55/models/model_episode_1_worker_1'

plotTensorboard(tensorflow_path, 1)

model=PPO.load(model_path)
env = make_vec_env("LunarLander-v2")
obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

