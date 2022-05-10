from turtle import end_fill
from stable_baselines3 import PPO
import torch as th
from client_stable_baselines3 import init_model
import os 
import glob
from zipfile import ZipFile
from stable_baselines3.common.env_util import make_vec_env
from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd
import re
import numpy as np
import seaborn as sns

#reference https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html

def AverageFederatedLearning(model_list):
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

def WeightedAverageFederatedLearning(model_list, report):
    return None

def EvolutionFederatedLearning(model_list, report):
    return None


def extractZip(path):
    zip_list=glob.glob(os.path.join(path, '*.zip'))
    if len(zip_list)==0:
        print('[INFO] No zip files found')
    else:
        for zip_file in zip_list:
            if not(os.path.isdir(zip_file.strip('.zip'))):
                with ZipFile(zip_file, 'r') as f:
                    f.extractall(zip_file.strip('.zip'))
                    # print(f'[INFO] Extracted {zip_file}')
    return None

def plotTensorboard(path, episode=0):
    extractZip(path)
    cmd='tensorboard --logdir ' + path
    os.system(cmd)
    return None

def getFederatedModel(folder_tree, episode, method):
    dir_list=glob.glob(os.path.join(folder_tree['req_models_dir'], f'*episode_{episode}_*.zip'))
    model_list=[]
    for model_zip in dir_list:
        model_list.append(PPO.load(model_zip))
    
    # get the report
    report=getTrainingReport(folder_tree['req_logs_dir'], episode)

    # compute the federated model
    if method=='average':
        federated_model=AverageFederatedLearning(model_list)
    elif method=='weighted-average':
        federated_model=WeightedAverageFederatedLearning(model_list, report)
    else:
        federated_model=EvolutionFederatedLearning(model_list, report)
        
    # save the federated model
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
    return None

def tb2df(root_dir, sort_by=None):
    """Convert local TensorBoard data into Pandas DataFrame.
    
    Function takes the root directory path and recursively parses
    all events data.    
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.
    
    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.
    
    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.
    
    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.
    
    """
    def convert_tfevent(filepath):
        return pd.DataFrame([
            parse_tfevent(e) for e in summary_iterator(filepath) if len(e.summary.value)
        ])

    def parse_tfevent(tfevent):
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=float(tfevent.summary.value[0].simple_value),
        )
    
    columns_order = ['wall_time', 'name', 'step', 'value']
    
    out = []
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path))

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    
    # make name in columns
    report=pd.DataFrame()
    report['step']=all_df['step'].unique()
    for name in all_df['name'].unique():
        try:
            report[name]=all_df[all_df['name']==name]['value'].values
        except:
            report[name]=np.insert(all_df[all_df['name']==name]['value'].values, 0, 'NaN')


    if sort_by is not None:
        report = report.sort_values(sort_by)
        
    return report.reset_index(drop=True)

def getTrainingReport(path, episode=None):
    extractZip(path)
    dir_list=[f.path for f in os.scandir(path) if f.is_dir()]
    
    report=pd.DataFrame()
    for dir in dir_list:
        exp_report=tb2df(dir)
        exp_name=os.path.basename(os.path.normpath(dir))
        exp_report['episode']=int(re.findall(r'\d+', exp_name)[0])
        exp_report['worker']=int(re.findall(r'\d+', exp_name)[1])
        report=pd.concat([report, exp_report])  

    report.reset_index(inplace=True)   
    
    if episode is None:
        report=report[report['episode']==episode]

    return report

def plotTrainingReport(path):
    report=getTrainingReport(path)
    report['episode'] = report['episode'].astype('category')
    sns.lineplot(data=report.sort_values('episode'), x="step", y="rollout/ep_rew_mean", hue='episode', palette="husl", markers=True)  
    return None

if __name__ == "__main__":
    PATH='/home/pietro/repos/DeRL_Golem/src/training/2022-03-30T14_55/logs/'
    plotTrainingReport(PATH)   




