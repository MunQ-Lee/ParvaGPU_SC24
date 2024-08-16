echo "MPS start in $1:$2 ($3)"

CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$1:$2 CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps-$1:$2 CUDA_VISIBLE_DEVICES=$3 nvidia-cuda-mps-control -d
echo start_server -uid 0 | CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$1:$2 CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps-$1:$2 CUDA_VISIBLE_DEVICES=$3 nvidia-cuda-mps-control