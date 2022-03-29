task_package=./docker-test-latest-7cb8d16ec3.gvmi
runtime_env=~/.local/lib/yagna/plugins/ya-runtime-vm/ya-runtime-vm

mkdir -p ./tmp/workdir
#ya-runtime-dbg --runtime $runtime_env --task-package $task_package --workdir /tmp/workdir
ya-runtime-dbg --runtime ~/.local/lib/yagna/plugins/ya-runtime-vm/ya-runtime-vm --task-package ./docker-baselines3_vm-latest-0f215d23a2.gvmi  --workdir /tmp/workdir