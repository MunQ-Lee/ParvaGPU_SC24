#!/bin/bash
SCENARIO_NUM=$1
INSTANCE_ID=$2

results="../deployment/SLO${SCENARIO_NUM}/SLO${SCENARIO_NUM}_ParvaGPU_deploy.csv"

declare -A exec_commands
declare -A mig_commands

final_gpu=0
skip_first_line=1
previous_gpu=0
mig_position=0

while IFS= read -r line; do
    if [ ${skip_first_line} -eq 1 ]; then
        skip_first_line=0
        continue
    fi
    
    instance_num=$(echo "$line" | awk -F ',' '{print $1}')
    current_gpu=$(echo "$line" | awk -F ',' '{print $2}')
    model_name=$(echo "$line" | awk -F ',' '{print $3}')
    placement=$(echo "$line" | awk -F ',' '{print $4}')
    instance_size=$(echo "$line" | awk -F ',' '{print $5}')
    batch=$(echo "$line" | awk -F ',' '{print $6}')
    num_proc=$(echo "$line" | awk -F ',' '{print $7}')

    array_key="${instance_num}:${current_gpu}"

    if [[ $INSTANCE_ID -ne $instance_num ]]; then
        mig_commands[$array_key]+=""
        exec_commands[$array_key]+=""
        continue
    fi

    if [[ "$current_gpu" != "$previous_gpu" ]]; then
        mig_position=0
        previous_gpu=$current_gpu
    fi


    if [[ $instance_size == "1" ]]; then
        mig_commands[$array_key]+="nvidia-smi mig -i ${current_gpu} -cgi 1g.10gb -C;"
    elif [[ $instance_size == "2" ]]; then
        mig_commands[$array_key]+="nvidia-smi mig -i ${current_gpu} -cgi 2g.20gb -C;"
    elif [[ $instance_size == "3" ]]; then
        mig_commands[$array_key]+="nvidia-smi mig -i ${current_gpu} -cgi 3g.40gb -C;"
    elif [[ $instance_size == "4" ]]; then
        mig_commands[$array_key]+="nvidia-smi mig -i ${current_gpu} -cgi 4g.40gb -C;"
    else
        mig_commands[$array_key]+="nvidia-smi mig -i ${current_gpu} -cgi 7g.80gb -C;"
    fi
    
    for ((i = 0; i < $num_proc; i++))
    do
        exec_commands[$array_key]+="./docker_run_parva.sh ${current_gpu}:${mig_position} $model_name $batch > /dev/null 2>&1 & "
    done

    mig_position=$((mig_position + 1))
    final_gpu=$current_gpu
done < "$results"

pgrep -f mps | xargs kill;
nvidia-smi mig -dci; nvidia-smi mig -dgi;

for ((j = 0; j <= final_gpu; j++)); 
do
    array_key="${INSTANCE_ID}:${j}"
    eval "${mig_commands[$array_key]}"
done

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

for ((j = 0; j <= final_gpu; j++)); 
do
    array_key="${INSTANCE_ID}:${j}"
    eval "${exec_commands[$array_key]}"
done

unset mig_commands
unset exec_commands
