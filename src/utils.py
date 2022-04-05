from stable_baselines3 import PPO
import torch as th
from client_stable_baselines3 import init_model
import os 
import glob
from zipfile import ZipFile
from stable_baselines3.common.env_util import make_vec_env

#reference https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html

# PATH_1='/home/pietro/repos/DeRL_Golem/src/training/2022-03-25T15_22/models/model_episode_1_worker_1.zip' 
# PATH_2='/home/pietro/repos/DeRL_Golem/src/training/2022-03-25T15_22/models/model_episode_1_worker_2.zip'
# PATH_3='/home/pietro/repos/DeRL_Golem/src/training/2022-03-25T15_22/models/model_episode_1_worker_3.zip'

# model_list=[]
# model_list.append(PPO.load(PATH_1))
# model_list.append(PPO.load(PATH_2))
# model_list.append(PPO.load(PATH_3))


# federated_model=FederatedLearning(model_list)

# # load model 
# model_1=PPO.load(PATH_1)
# parameters_1=model_1.get_parameters()
# net_arch_1=model_1. policy.net_arch
# policy_1=parameters_1['policy']
# state_dict_1=model_1.policy.state_dict

# model_2=PPO.load(PATH_2)
# parameters_2=model_2.get_parameters()
# net_arch_2=model_2.policy.net_arch
# policy_2=parameters_2['policy']
# state_dict_1=model_2.policy.state_dict()

# policy_3=OrderedDict()
# for key in policy_2.keys():
#     policy_3[key]=th.add(policy_1[key], policy_2[key])

# model_3=PPO.load(PATH_1)
# parameters_3=parameters_1
# parameters_3['policy']=policy_3
# model_3.set_parameters(parameters_3, exact_match=False)

# parameters_test=model_3.get_parameters()
# print('done')


def FederatedLearning(model_list):
    federated_policy=model_list[0].get_parameters()['policy']

    # get the federated model policy as 
    for i in range(1, len(model_list)):
        for key in federated_policy.keys():
            federated_policy[key]=th.add(th.div(federated_policy[key], 2), 
                                        th.div(model_list[i].get_parameters()['policy'][key], 2))
    
    # model initialization
    federated_model=init_model(policy='MlpPolicy', env=make_vec_env('LunarLander-v2', n_envs=1))
    federated_parameters=federated_model.get_parameters()

    # substitute federated parameters
    federated_parameters['policy']=federated_policy
    federated_model.set_parameters(federated_parameters)

    return federated_model


def plotTensorboard(path, episode):
    if episode > 0:
        dir_list=glob.glob(os.path.join(path, f'*episode_{episode}_*.zip'))
    else:
        dir_list=glob.glob(os.path.join(path, '*episode_*.zip'))

    cmd='tensorboard --logdir ' + path
    if len(dir_list)==0:
        print('[INFO] No Tensorboard logs found')
    else:
        for zip_file in dir_list:
            with ZipFile(zip_file, 'r') as f:
                f.extractall(zip_file.strip('.zip'))
                print(f'[INFO] Extracted {zip_file}')
        os.system(cmd)
    return None

def getFederatedModel(path, episode):
    dir_list=glob.glob(os.path.join(path, f'*episode_{episode}_*.zip'))
    model_list=[]
    for model_zip in dir_list:
        model_list.append(PPO.load(model_zip))
    
    federated_model=FederatedLearning(model_list)
    federated_model.save(os.path.join(path, f'federated_model_episode_{episode}'), include=['env'])

    return federated_model
    
def evalAgent(path):
    model=PPO.load(path)
    env = make_vec_env("LunarLander-v2")
    obs = env.reset()

    dones=False
    while dones==False:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == "__main__":

    PATH_1='/home/pietro/repos/DeRL_Golem/src/training/2022-03-29T16_58/models/model_episode_1_worker_1.zip' 
    PATH_2='/home/pietro/repos/DeRL_Golem/src/training/2022-03-29T16_58/models/model_episode_1_worker_2.zip'
    PATH_3='/home/pietro/repos/DeRL_Golem/src/training/2022-03-29T16_58/models/model_episode_1_worker_3.zip'

    model_list=[]
    model_list.append(PPO.load(PATH_1))
    model_list.append(PPO.load(PATH_2))
    model_list.append(PPO.load(PATH_3))

    federated_model=FederatedLearning(model_list)
    #path='/home/pietro/repos/DeRL_Golem/src/training/2022-03-29T10_55/models'
    #getFederatedModel(path, 1)


