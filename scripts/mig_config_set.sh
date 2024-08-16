# total 11 cases
# 7
# 4,3
# 4,2,1
# 4,1,1,1
# 2,2,3
# 2,1,1,3
# 1,1,1,1,3
# 2,2,2,1
# 2,2,1,1,1
# 2,1,1,1,1,1
# 1,1,1,1,1,1,1

# nvidia-smi -L

mig_config=("0", "5,9", "5,14,19", "5,19,19,19", "9,14,14", "9,14,19,19", "9,19,19,19,19", "14,14,14,19", "14,14,19,19,19", "14,19,19,19,19,19", "19,19,19,19,19,19,19")

pgrep -f mps | xargs kill;
nvidia-smi mig -dci; nvidia-smi mig -dgi;

if [[ $1 == -1 ]]; then
    exit 0
fi

nvidia-smi mig -cgi ${mig_config[$1]} -C

GPU_UUID=()
MIG_UUID=()
gpu_id=-1
mig_id=0
NB_GPUS=$(nvidia-smi -L | grep "UUID:" | wc -l)
if [[ "$NB_GPUS" == 0 ]]; then
    ALL_GPUS=$(nvidia-smi -L | grep "UUID: GPU" | cut -d"(" -f2 | cut -d' ' -f2 | cut -d')' -f1)
    echo "No MIG GPU available, using the full GPUs ($ALL_GPUS)."

else
    ALL_GPUS=$(nvidia-smi -L | grep "UUID: " | cut -d"(" -f2 | cut -d' ' -f2 | cut -d')' -f1)
    for gpu in $(echo "$ALL_GPUS"); do
        gpu_mig=$(echo $gpu | cut -d"-" -f1)
        if [[ $gpu_mig == "GPU" ]]; then
            gpu_id=$(($gpu_id + 1))
            GPU_UUID+=("$gpu")
            mig_id=0
        elif [[ $gpu_mig == "MIG" ]]; then
            MIG_UUID+=("$gpu")
            ./mps_set.sh $gpu_id $mig_id $gpu
            mig_id=$(($mig_id + 1))
        fi
    done
fi
