import gym
from stable_baselines3 import PPO
import torch as th
import json
import os
from stable_baselines3.common.env_util import make_vec_env
import shutil

env=make_vec_env('LunarLander-v2', n_envs=1)
env.reset()
policy='MlpPolicy'
# Custom actor (pi) and value function (vf) networks
# of two layers of size 32 each with Relu activation function
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[32, 32], vf=[32, 32])])

total_timesteps=1000000

INPUT_VOLUME='/golem/input/'
OUTPUT_VOLUME='/golem/output/'

'''
Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
'''
def init_model(policy, env, *args, **kwargs):    
    model=PPO(  policy=                 policy,           
                env=                    env,                                
                learning_rate=          3e-4,          
                n_steps=                2048,                                   
                batch_size=             32,                                   
                n_epochs=               32,                                    
                gamma=                  0.99,                                    
                gae_lambda=             0.95,                              
                clip_range=             0.2,               
                clip_range_vf=          None,     
                ent_coef=               0.0,                                  
                vf_coef=                0.5,                                   
                max_grad_norm=          0.5,                            
                use_sde=                False,                                 
                sde_sample_freq=        -1,                              
                target_kl=              None,                      
                tensorboard_log=        os.path.join(OUTPUT_VOLUME, 'tensorboard/'),               
                create_eval_env=        False,                         
                policy_kwargs=          policy_kwargs,         
                verbose=                1,                                       
                seed=                   None,                            
                device=                 "auto",              
                _init_setup_model=      True)
    
    return model

def init_model1(kwargs):
     model=PPO(**kwargs)
     return model

def load_model():
    model.load()
    return model

def train_model(model, total_timesteps):
    model.learn(total_timesteps=        total_timesteps,
                callback=               None,
                log_interval=           1,
                eval_env=               None,
                eval_freq=              -1,
                n_eval_episodes=        1,
                tb_log_name=            "PPO",
                eval_log_path=          None,
                reset_num_timesteps=    True)
    return model 

def conf_parser(path):
    with open(path, 'r') as f:
        conf = json.load(f) 
    return conf

def save_model(model, path):
    try:
        model.save(path)
    except:
        print('[ERROR] Failed to save model')
    return None

def remFolderContent(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

if __name__ == "__main__":
    # remove output folder content
    remFolderContent(OUTPUT_VOLUME)

    # get model configurations 
    #conf_model=conf_parser('./conf_model.json')

    model=init_model(policy, env)
    #model=init_model(kwargs)
    model=train_model(model, total_timesteps)
    model=save_model(model, os.path.join(OUTPUT_VOLUME, 'model_out.zip'))
    shutil.make_archive(os.path.join(OUTPUT_VOLUME, 'tensorboard'), 'zip', os.path.join(OUTPUT_VOLUME, 'tensorboard/'))

    # remove input volume content
    remFolderContent(INPUT_VOLUME)


