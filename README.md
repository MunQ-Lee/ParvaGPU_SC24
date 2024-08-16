# ParvaGPU

ParvaGPU represents an efficient GPU space-sharing technology that optimizes GPU usage while facilitating large-scale DNN inference in cloud environments, improving cost-effectiveness. This technology utilizes NVIDIA's Multi-Instance GPU (MIG) and Multi-Process Service (MPS) to enhance GPU performance. ParvaGPU assigns partitioned MIG instances to individual inference tasks, ensuring no interference between different workloads. It also activates MPS within each MIG instance, increasing the process count for the same workload to fully utilize the resources within each instance. We refer to an MPS-activated MIG instance as a GPU segment. ParvaGPU significantly reduces GPU usage while meeting the Service Level Objectives (SLOs) of each workload by employing two main algorithms. The Segment Configurator determines the optimal GPU segment configuration for each workload to minimize underutilization, based on profiling data. The Segment Allocator then assigns these segments across multiple GPUs to reduce external fragmentation, ensuring efficient use of resources.

## Dependencies
### Hardware
One or more GPUs that support NVIDIA's Multi-Instance GPU (MIG) feature (e.g., NVIDIA A100 or H100)

### Software
- Ubuntu 20.04.6
- NVIDIA driver version >= 525.125.06  
- gcc 9.4.0
- g++ 9.4.0
- Docker engine 24.0.5
- NVIDIA docker 2.13.0

## Profiling
Users need to conduct a single profiling on the target GPU to use ParvaGPU. Through this profiling, users need to measure the throughput and latency of each DNN model as the MIG instance size, batch size, and the number of MPS processes within the MIG partition change. There are five available MIG instance sizes (1, 2, 3, 4, and 7 Graphics Processing Clusters (GPCs)). To avoid an exhaustive search, we suggest using eight batch sizes that increase in powers of two from 1 to 128, which are commonly used. Additionally, the number of processes is limited to three within each MIG instance, to consider out-of-memory issues. To understand the format for storing measurement data, refer to the example CSV files stored in `prof_data`. The example CSV files contain the profiling results for 11 DNN models on an NVIDIA A100 80GB GPU.

## Service Level Objective (SLO) Specification

Users can specify an service level objective (SLO) that includes the request rate and latency for each model when using ParvaGPU. Furthermore, users are able to select different combinations of SLOs for each model at runtime. The `SLO_req_rate.csv` and `SLO_latency.csv` files in the `SLO` folder document the request rate and latency for each model, respectively. In each CSV file, one row contains a single SLO combination for the models executed in ParvaGPU. Each column represents the request rate or latency for each model, and the order of the columns is determined by the alphabetical order of the profiling result CSV files for each model located in the `prof_data` folder. The current `SLO_req_rate.csv` and `SLO_latency.csv` files contain a total of six SLO combinations as examples, based on the profiling CSV files located in the `prof_data` folder.


## Installation
```
# Clone the repository
git clone https://gitfront.io/r/sslab6943/GPwisjzMNg8H/ParvaGPU.git
```

```
# Pull the required Docker image
docker pull nvcr.io/nvidia/pytorch:21.11-py3
```

```
# Configure the number of GPUs per GPU node
ParvaGPU allows the specification of the number of GPUs per GPU node (or cloud instance) and the total number of GPUs across nodes within inc/parva_sched.h. 
- NUM_GPU_PER_NODE: Number of GPUs per GPU node (or cloud instance)
- TOTAL_GPU: Total number of GPUs across all nodes
```

```
# Build the project
make
```

## Running ParvaGPU
```
./parva_sched SLO_INDEX
```

`SLO_INDEX` refers to the row number in the `SLO_req_rate.csv` and `SLO_latency.csv` files that describe the SLO, with the assumption that it starts at 1.The excutable produces a GPU deployment map that reduces GPU usage throughout the entire GPU cluster while satisfying the SLO requirements for each workload. Within the deployment directory, a subdirectory named after the SLO_INDEX is created, and the relevant deployment map is saved as a CSV file in this subdirectory. The GPU deployment map is structured into groups of GPU segments based on node number and the specific GPU within that node. Each segment details the MIG instance size, its location, the batch size for the model, and the number of MPS processes.

## Deploying DNN Inference
```
cd scripts
./scripts/parvagpu_run.sh SLO_INDEX NODE_NUM
```
`SLO_INDEX` refers to the row number in the `SLO_req_rate.csv` and `SLO_latency.csv` files where the SLO is described, with the assumption that it starts at 1. `NODE_INDEX` represents the node number within the GPU cluster where this script is currently running. This script reconfigures the MIG instances of GPUs within the node according to the contents of the deployment CSV files located in subfolders for each `SLO_INDEX` within the `deployment` folder, and then deploys each DNN inference task.
