ENV_NAME=yapapi

conda create --name $ENV_NAME --clone base
conda activate $ENV_NAME
 ~/miniconda3/envs/$ENV_NAME/bin/pip install -U pip
 ~/miniconda3/envs/$ENV_NAME/bin/pip install yapapi
