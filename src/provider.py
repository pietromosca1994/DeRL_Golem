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
    folder_tree={   "req_logs_dir":     os.path.join(path, 'training', str(timestamp), 'logs'),         # requestor logs dir 
                    "req_models_dir":   os.path.join(path, 'training', str(timestamp), 'models'),       # requestor models directory
                    "req_conf_dir":     os.path.join(path, 'training', str(timestamp), 'conf'),         # requestor cofiguration directory
                    "prov_in_dir":      os.path.join("/golem/input/"),                                  # provider input directory
                    "prov_out_dir":     os.path.join("/golem/output/"),                                 # provider output directory
                    "prov_src_dir":     os.path.join("/golem/src/")}                                    # provider source directory   

    # create folder
    try:
        os.makedirs(folder_tree["req_logs_dir"], exist_ok=False)
        os.makedirs(folder_tree["req_models_dir"], exist_ok=False)
        os.makedirs(folder_tree["req_conf_dir"], exist_ok=False)
    except:
        None

    copy_tree(os.path.join(path, 'conf'), folder_tree['req_conf_dir'])     
    
    return folder_tree


