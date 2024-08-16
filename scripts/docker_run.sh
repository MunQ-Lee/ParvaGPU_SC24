GPU=$1
MODEL=$2
BATCH=$3

docker run -i --cap-add SYS_ADMIN \
--gpus '"device='${GPU}'"' \
--rm --ipc=host --pid=host \
-v /tmp:/tmp \
-v $PWD:/workspace/gpu_workload \
\
-w /workspace/gpu_workload \
-e "CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-${GPU}" \
\
nvcr.io/nvidia/pytorch:21.11-py3 \
\
bash -c \
"python -u /workspace/gpu_workload/dl_infer_runtime.py \
--gpus=${GPU} --steps=1000 \
--model=${MODEL} --batch-size=${BATCH}"