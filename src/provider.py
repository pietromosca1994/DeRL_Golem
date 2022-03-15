#!/usr/bin/python3
import json
import os 
import datetime 
from distutils.dir_util import copy_tree

def load_conf(path):
    with open(path) as json_file:
        conf = json.load(json_file)
    return conf

def makeOutFolder(path):
    timestamp=datetime.datetime.now().strftime("%Y-%m-%dT%H_%M")
    folder_tree={   "req_logs_dir":     os.path.join(path, 'training', str(timestamp), 'logs'),
                    "req_models_dir":   os.path.join(path, 'training', str(timestamp), 'models'),
                    "req_conf_dir":     os.path.join(path, 'training', str(timestamp), 'conf'),
                    "prov_in_dir":      os.path.join("/golem/input/"),
                    "prov_out_dir":     os.path.join("/golem/output/"),
                    "prov_src_dir":     os.path.join("/golem/src/")}                      

    # create folder 
    os.makedirs(folder_tree["req_logs_dir"], exist_ok=True)
    os.makedirs(folder_tree["req_models_dir"], exist_ok=True)
    os.makedirs(folder_tree["req_conf_dir"], exist_ok=True)

    copy_tree(os.path.join(path, 'conf'), folder_tree['req_conf_dir'])     
    
    return folder_tree


