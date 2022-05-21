#!/usr/bin/env python3
import argparse
import asyncio
from datetime import datetime, timedelta
from tempfile import gettempdir
from typing import AsyncIterable
from datetime import datetime
import os 
from datetime import datetime
from utils import evalAgent, getFederatedModel

# import yapapi modules
from yapapi import Golem, Task, WorkContext, Executor
from yapapi.events import CommandExecuted
from yapapi.log import enable_default_logger
from yapapi.payload import vm
from yapapi.rest.activity import CommandExecutionError


# import provider modules 
from provider import load_conf, makeOutFolder

# bash parameters 
TEXT_COLOR_RED =        "\033[31;1m"
TEXT_COLOR_GREEN =      "\033[32;1m"
TEXT_COLOR_YELLOW =     "\033[33;1m"
TEXT_COLOR_BLUE =       "\033[34;1m"
TEXT_COLOR_MAGENTA =    "\033[35;1m"
TEXT_COLOR_CYAN =       "\033[36;1m"
TEXT_COLOR_WHITE =      "\033[37;1m"
TEXT_COLOR_DEFAULT =    "\033[0m"

# yapapi parameters
PROJECT_PATH=os.path.join('/home/pietro/repos/DeRL_Golem/src/')

async def main(conf_yapapi, conf_RL, folder_tree): # conf_RL is prepared but not used 
    package = await vm.repo(
        image_hash=conf_yapapi['image_hash'],
        min_mem_gib=conf_yapapi['min_mem_gib'],
        min_storage_gib=conf_yapapi['min_storage_gib'],
    )
    '''
    Main function
    '''

    async with Golem(
        budget=conf_yapapi ['budget'],
        subnet_tag=conf_yapapi['subnet_tag'],
        app_key=conf_yapapi['app_key']
    ) as golem:
        
        print(f"{TEXT_COLOR_RED}" f"[INFO] {datetime.now()} Starting computation" f"{TEXT_COLOR_DEFAULT}")
        
        for episode in range(1, conf_yapapi['episodes']+1): # loop through episodes 
            print(f"{TEXT_COLOR_YELLOW}" f"[INFO] {datetime.now()} Starting training espisode n: {episode}" f"{TEXT_COLOR_DEFAULT}")

            data = [Task(data={ 'folder_tree': folder_tree,
                                'episode': episode,
                                'worker_id': c}) for c in range(1, conf_yapapi['max_workers']+1)]
            
            async for task in golem.execute_tasks(
                                    worker=worker,
                                    data=data,
                                    payload=package,
                                    max_workers=conf_yapapi['max_workers'],
                                    timeout=timedelta(minutes=conf_yapapi['timeout'])):
                                    done=1
            
            print(f"{TEXT_COLOR_CYAN}" f"[INFO] Generating Federated Agent for episode n: {episode}" f"{TEXT_COLOR_DEFAULT}")
            # compute the federated model 
            getFederatedModel(folder_tree, episode, conf_RL['federatedRL']['method']) # make a configuration in json to read the method
            
            # evaluate agent 
            #evalAgent(os.path.join(folder_tree['req_models_dir'], f'federated_model_episode_{episode}.zip'))
            print(f"{TEXT_COLOR_YELLOW}" f"[INFO] Completed training for episode n: {episode}" f"{TEXT_COLOR_DEFAULT}")

        print(f"{TEXT_COLOR_RED}" f"[INFO] {datetime.now()} Ending computation" f"{TEXT_COLOR_DEFAULT}")


async def worker(context: WorkContext, tasks: AsyncIterable[Task]):
    '''
    Worker function
    '''
    async for task in tasks:
        print(  f"{TEXT_COLOR_GREEN}" f"[INFO] {datetime.now()} Start task on {context.provider_name}" f"{TEXT_COLOR_DEFAULT}")
       
        # computation
        worker_id=task.data['worker_id']
        episode=task.data['episode']
        folder_tree=task.data['folder_tree']

        s = context.new_script(timeout=timedelta(minutes=conf_yapapi['timeout']))
        
        # upload inputs
        s.upload_file(os.path.join(PROJECT_PATH, "client_stable_baselines3.py"), os.path.join(folder_tree['prov_src_dir'], "client_stable_baselines3.py"))              # upload client script
        
        if os.path.isfile(os.path.join(folder_tree['req_models_dir'], f'federated_model_episode_{episode-1}.zip')): # in case the federated model is available from the step before load it 
            s.upload_file(os.path.join(folder_tree['req_models_dir'], f'federated_model_episode_{episode-1}.zip'), os.path.join(folder_tree['prov_in_dir'], 'federated_model.zip'))
        
        # start training
        s.run("/bin/sh", "-c", "python3 " + os.path.join(folder_tree['prov_src_dir'], "client_stable_baselines3.py"))               # run client computation        
        
        # download results 
        s.download_file(os.path.join(folder_tree['prov_out_dir'], "model_out.zip"), os.path.join(folder_tree['req_models_dir'], f'model_episode_{episode}_worker_{worker_id}.zip')) # dowload output
        s.download_file(os.path.join(folder_tree['prov_out_dir'], "tensorboard.zip"), os.path.join(folder_tree['req_logs_dir'], f'tensorboard_episode_{episode}_worker_{worker_id}.zip')) # dowload output
        yield s
        
        print(f"{TEXT_COLOR_RED}" f"[INFO] {datetime.now()} End task on {context.provider_name}" f"{TEXT_COLOR_DEFAULT}")
        
        # task completed
        task.accept_result()
       

if __name__ == "__main__":
    # initialize environment
    # create folder structure
    folder_tree=makeOutFolder(PROJECT_PATH)
    
    # parse configurations
    conf_yapapi=load_conf(os.path.join(folder_tree['req_conf_dir'], 'conf_yapapi.json'))
    conf_RL=load_conf(os.path.join(folder_tree['req_conf_dir'], 'conf_RL.json'))

    # initialize yapapi 
    loop = asyncio.get_event_loop()
    task = loop.create_task(main(conf_yapapi, conf_RL, folder_tree))

    # yapapi debug logging to a file
    enable_default_logger(log_file=os.path.join(folder_tree['req_logs_dir'], "yapapi.log"))

    # run yapapi
    try:
        loop.run_until_complete(task)
    except KeyboardInterrupt:
        print(f"{TEXT_COLOR_YELLOW}" "Shutting down gracefully, please wait a short while or press Ctrl+C to exit immediately..."f"{TEXT_COLOR_DEFAULT}")
        task.cancel()
        try:
            loop.run_until_complete(task)
            print(
                f"{TEXT_COLOR_YELLOW}" "Shutdown completed, thank you for waiting!" f"{TEXT_COLOR_DEFAULT}")
        except KeyboardInterrupt:
            pass
