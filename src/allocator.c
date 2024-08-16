#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stddef.h>
#include <math.h>

#include "parva_sched.h"
#include "queue.h"

typedef struct GPC_INFO {
    int batch_size;
    int num_proc;
    int gpc_size;
    int placement;
    int model_idx;
    char* model_name;
} gpc_info;

Queue *size_7_queue;
Queue *size_4_queue;
Queue *size_3_queue;
Queue *size_2_queue;
Queue *size_1_queue;

int empty_gpcs;
int total_instance_usage;
int total_gpu_usage;

void init_deploy()
{
    size_7_queue = initQueue();
    size_4_queue = initQueue();
    size_3_queue = initQueue();
    size_2_queue = initQueue();
    size_1_queue = initQueue();

    gpus = malloc(TOTAL_GPU * sizeof(struct gpu_device));  
    for (int i = 0; i < TOTAL_GPU; i++)
    {
        gpus[i].idx = i;
        gpus[i].usage = 0;
        gpus[i].config = 0;
        gpus[i].avail_req_rate = 0;
        for (int j = 0; j < 7; j++)
            gpus[i].mig_instances[j] = NULL;
    }
}


struct gpu_device *find_fit_gpu(int instance_size)
{
    struct gpu_device *min_gpu = NULL;
    int min_val = INT_MAX;

    for (int i = 0; i < TOTAL_GPU; i++)
    {
        struct gpu_device *gpu = &gpus[i];
        int empty = MAX_MIG_SIZE - gpu->usage;
        int diff = empty - instance_size;

        if ((empty >= instance_size) && (min_val > diff))
        {
            if (instance_size == 3 && gpu->mig_instances[4])
            {
                if (gpu->mig_instances[4]->size == 3)
                    continue;
            }

            if (instance_size == 1 && gpu->mig_instances[0] && gpu->mig_instances[4])
            {
                if ((gpu->mig_instances[0]->size == 3) && (gpu->mig_instances[4]->size == 3))
                    continue;
            }
            min_gpu = gpu;
            min_val = diff;
        }
    }
    return min_gpu;
}

int find_fit_placement(struct gpu_device *gpu, int instance_size)
{
    int placement = -1;
    if (instance_size == 7)
    {
        if (!gpu->mig_instances[0])
            placement = 0;
    }
    else if (instance_size == 4)
    {
        if (!gpu->mig_instances[0])
            placement = 0;
    }
    else if (instance_size == 3)
    {
        if (!gpu->mig_instances[4])
            placement = 4;
    }
    else if (instance_size == 2)
    {
        if (!gpu->mig_instances[0])
            placement = 0;
        else if (!gpu->mig_instances[4])
            placement = 4;
        else if (!gpu->mig_instances[2])
            placement = 2;
    }
    else if (instance_size == 1)
    {
        if (!gpu->mig_instances[0])
            placement = 0;
        else if (!gpu->mig_instances[6])
            placement = 6;
        else if (!gpu->mig_instances[4])
            placement = 4;
        else if (!gpu->mig_instances[1])
            placement = 1;
        else if (!gpu->mig_instances[2])
            placement = 2;
        else if (!gpu->mig_instances[3])
            placement = 3;
        else if (!gpu->mig_instances[5])
            placement = 5;
    }
    return placement;
}


struct mig_instance *alloc_mig_instance(struct svc_level_obj *service, int placement, int size)
{
    struct mig_instance *mig_instance = malloc(sizeof(struct mig_instance));
    mig_instance->placement = placement;
    mig_instance->size = size;
    mig_instance->service = service;
    return mig_instance;
}


void dequeueing(int queueing_service)
{
    for (int i = 0; i < queueing_service; i++)
    {
        int size = -1;
        int placement = -1;
        struct svc_level_obj *service = NULL;
        struct gpu_device *target_gpu = NULL;

        if (!is_empty(size_7_queue))
        {
            size = 7;
            service = (struct svc_level_obj *)dequeue(size_7_queue);
            target_gpu = find_fit_gpu(size);
            if (target_gpu == NULL)
                break;
            placement = find_fit_placement(target_gpu, size);
            target_gpu->mig_instances[placement] = alloc_mig_instance(service, placement, size);
            target_gpu->usage += size;
        }
        else if (!is_empty(size_4_queue))
        {
            size = 4;
            service = (struct svc_level_obj *)dequeue(size_4_queue);
            target_gpu = find_fit_gpu(size);
            if (target_gpu == NULL)
                break;
            placement = find_fit_placement(target_gpu, size);
            target_gpu->mig_instances[placement] = alloc_mig_instance(service, placement, size);
            target_gpu->usage += size;
        }
        else if (!is_empty(size_3_queue))
        {
            
            size = 3;
            service = (struct svc_level_obj *)dequeue(size_3_queue);
            target_gpu = find_fit_gpu(size);
            if (target_gpu == NULL)
                break;
            placement = find_fit_placement(target_gpu, size);
            target_gpu->mig_instances[placement] = alloc_mig_instance(service, placement, size);
            target_gpu->usage += size;
        }
        else if (!is_empty(size_2_queue))
        {
            size = 2;
            service = (struct svc_level_obj *)dequeue(size_2_queue);
            target_gpu = find_fit_gpu(size);
            if (target_gpu == NULL)
                break;
            placement = find_fit_placement(target_gpu, size);
            target_gpu->mig_instances[placement] = alloc_mig_instance(service, placement, size);
            target_gpu->usage += size;
        }
        else if (!is_empty(size_1_queue))
        {
            size = 1;
            service = (struct svc_level_obj *)dequeue(size_1_queue);
            target_gpu = find_fit_gpu(size);
            if (target_gpu == NULL)
                break;
            placement = find_fit_placement(target_gpu, size);
            target_gpu->mig_instances[placement] = alloc_mig_instance(service, placement, size);
            target_gpu->usage += size;
        }
        dp("%s model select %d gpu, %d size, %d placement\n", service->model, target_gpu->idx, size, placement);
    }
}


void optimization()
{
    dp("start optimization\n");
    float opti_req_rate[num_models];
    memset(opti_req_rate, 0, num_models*sizeof(float));

    for (int i = TOTAL_GPU; i >= 0; --i)
    {
        if (gpus[i].usage <= 4)
        {
            for (int j = 0; j < 7; j++)
            {
                if (gpus[i].mig_instances[j])
                {
                    struct svc_level_obj *service = gpus[i].mig_instances[j]->service;
                    int size = gpus[i].mig_instances[j]->size;
                    float seg_req_rate = 0;
                    int num_size_1 = 0;
                    int num_size_2 = 0;
                    int reconfig_instance_count = 0;
                    dp("test\n");

                    if (size == service->optimal_point->inst_size)
                        seg_req_rate = service->optimal_point->trp;
                    else
                        seg_req_rate = service->last_instance_point->trp;

                    opti_req_rate[service->model_idx] += seg_req_rate;
                    gpus[i].usage -= size;
                    free(gpus[i].mig_instances[j]);
                    gpus[i].mig_instances[j] = NULL;

                    if (!service->size_2_point && !service->size_1_point)
                    {
                        continue;
                    }
                    else if (!service->size_2_point && service->size_1_point)
                    {
                        num_size_1 = (int)ceil(opti_req_rate[service->model_idx]/service->size_1_point->trp);
                        for (int k = 0; k < num_size_1; k++)
                        {
                            enqueue(size_1_queue, service);
                            reconfig_instance_count++;
                            opti_req_rate[service->model_idx] -= service->size_1_point->trp;
                        }
                    }
                    else if (service->size_2_point && !service->size_1_point)
                    {
                        num_size_2 = (int)ceil(opti_req_rate[service->model_idx]/service->size_2_point->trp);
                        for (int k = 0; k < num_size_2; k++)
                        {
                            enqueue(size_2_queue, service);
                            reconfig_instance_count++;
                            opti_req_rate[service->model_idx] -= service->size_2_point->trp;
                        }
                    }
                    else
                    {
                        if (service->size_2_point->trp/2 > service->size_1_point->trp)
                        {
                            num_size_2 = (int)floor(opti_req_rate[service->model_idx]/service->size_2_point->trp);
                            for (int k = 0; k < num_size_2; k++)
                            {
                                enqueue(size_2_queue, service);
                                reconfig_instance_count++;
                                opti_req_rate[service->model_idx] -= service->size_2_point->trp;
                            }
                            num_size_1 = (int)ceil(opti_req_rate[service->model_idx]/service->size_1_point->trp);
                            for (int k = 0; k < num_size_1; k++)
                            {
                                enqueue(size_1_queue, service);
                                reconfig_instance_count++;
                                opti_req_rate[service->model_idx] -= service->size_1_point->trp;
                            }
                        }
                        else
                        {
                            num_size_1 = (int)ceil(opti_req_rate[service->model_idx]/service->size_1_point->trp);
                            for (int k = 0; k < num_size_1; k++)
                            {
                                enqueue(size_1_queue, service);
                                reconfig_instance_count++;
                                opti_req_rate[service->model_idx] -= service->size_1_point->trp;
                            }
                        }
                    }
                    dp("%d model (%s) reconfig %d -> %d (%d, %d)\n", service->model_idx, service->model, size, num_size_2*2+num_size_1, num_size_1, num_size_2);
                    dequeueing(reconfig_instance_count);
                }
            }
        }
    }
}


int segment_relocation()
{
    dp("\nsegment_relocation start\n");

    total_instance_usage = 0; 
    total_gpu_usage = 0;
    int queueing_service = 0;

    for (int model_idx = 0; model_idx < num_models; model_idx++)
    {
        struct svc_level_obj *service = &svc_level_obj_arr[model_idx];
        if (service->optimal_point)
        {
            int instance_size = service->optimal_point->inst_size;
            int num_instance = service->num_optimal_points;

            dp("%s model %d (%d) size\n", service->model, instance_size, num_instance);

            for (int i = 0; i < num_instance; i++)
            {
                total_instance_usage += instance_size;
                queueing_service++;

                if (instance_size == MAX_MIG_SIZE)
                {
                    enqueue(size_7_queue, service);
                }
                else if (instance_size == 4)
                {
                    enqueue(size_4_queue, service);
                }
                else if (instance_size == 3)
                {
                    enqueue(size_3_queue, service);
                }
                else if (instance_size == 2)
                {
                    enqueue(size_2_queue, service);
                }
                else if (instance_size == 1)
                {
                    enqueue(size_1_queue, service);
                }
            }
            if (service->last_instance_point)
            {
                int last_instance_size = service->last_instance_point->inst_size;
                total_instance_usage += last_instance_size;
                queueing_service++;

                if (last_instance_size == MAX_MIG_SIZE)
                {
                    enqueue(size_7_queue, service);
                }
                else if (last_instance_size == 4)
                {
                    enqueue(size_4_queue, service);
                }
                else if (last_instance_size == 3)
                {
                    enqueue(size_3_queue, service);
                }
                else if (last_instance_size == 2)
                {
                    enqueue(size_2_queue, service);
                }
                else if (last_instance_size == 1)
                {
                    enqueue(size_1_queue, service);
                }
            }
        }
    }

    dp("\nqueueing service: %d\n\n", queueing_service);

    dequeueing(queueing_service);
    
    dp("\n");

    total_gpu_usage = 0;
    for (int i = 0; i < TOTAL_GPU; i++)
    {
        
        if (gpus[i].usage)
        {
            dp(" %d gpu usage: %d\n", i, gpus[i].usage);
            total_gpu_usage++;
        }
    }
    dp("before optimization total instance/gpu usage: %d/%d\n\n", total_instance_usage, total_gpu_usage);
    
#ifndef NO_OPTIM
    optimization();

    rp("\n\nParvaGPU result\n");
    total_gpu_usage = 0;
    total_instance_usage = 0;

#ifdef REQ_RATE_CHECK
    float total_trp[num_models];
    memset(total_trp, 0, num_models*sizeof(float));
    for (int i = 0; i < TOTAL_GPU; i++)
    {
        if (gpus[i].usage)
        {
            for (int j = 0; j < 7; j++)
            {
                if (gpus[i].mig_instances[j])
                {
                    if (gpus[i].mig_instances[j]->size == gpus[i].mig_instances[j]->service->optimal_point->inst_size)
                        total_trp[gpus[i].mig_instances[j]->service->model_idx] += gpus[i].mig_instances[j]->service->optimal_point->trp;
                    else if (gpus[i].mig_instances[j]->size == gpus[i].mig_instances[j]->service->last_instance_point->inst_size)
                        total_trp[gpus[i].mig_instances[j]->service->model_idx] += gpus[i].mig_instances[j]->service->last_instance_point->trp;
                    else if (gpus[i].mig_instances[j]->size == 2)
                        total_trp[gpus[i].mig_instances[j]->service->model_idx] += gpus[i].mig_instances[j]->service->size_2_point->trp;
                    else if (gpus[i].mig_instances[j]->size == 1)
                        total_trp[gpus[i].mig_instances[j]->service->model_idx] += gpus[i].mig_instances[j]->service->size_1_point->trp;
                }
            }
        }
    }
    for (int i = 0; i < num_models; i++)
    {
        printf("model %d (%s): reach throughtput/request rate: %f/%f\n", i, svc_level_obj_arr[i].model, total_trp[i], svc_level_obj_arr[i].req_rate);
    }
#endif

    FILE *deploy;
    char line[MAX_LINE_LENGTH];
    char deploy_dir[MAX_FILENAME_LEN];
    sprintf(deploy_dir, "./deployment/SLO%d/SLO%d_ParvaGPU_deploy.csv", SCENARIO+1, SCENARIO+1);
    deploy = fopen(deploy_dir, "w");
    fprintf(deploy, "Num instance,Num GPU,Model,Placement,Instance size,Batch size,Num proc\n");

    int node_idx = 0;
    int gpu_idx = 0;

    for (int i = 0; i < TOTAL_GPU; i++)
    {
        if (i % NUM_GPU_PER_NODE == 0 && i)
        {
            node_idx++;
        }
        gpu_idx = i % NUM_GPU_PER_NODE;

        if (gpus[i].usage)
        {
            gpc_info gpc_info_array[7];

            int gpc_info_idx = 0;

            rp(" %d gpu usage: %d\n", i, gpus[i].usage);
            for (int j = 0; j < 7; j++)
            {
                if (gpus[i].mig_instances[j])
                {
                    total_instance_usage += gpus[i].mig_instances[j]->size;

                    gpc_info_array[gpc_info_idx].gpc_size = gpus[i].mig_instances[j]->size;
                    gpc_info_array[gpc_info_idx].batch_size = 0;
                    gpc_info_array[gpc_info_idx].num_proc = 0;
                    gpc_info_array[gpc_info_idx].model_idx = gpus[i].mig_instances[j]->service->model_idx;
                    gpc_info_array[gpc_info_idx].model_name = gpus[i].mig_instances[j]->service->model;
                    gpc_info_array[gpc_info_idx].placement = gpus[i].mig_instances[j]->placement;
                    
                    if (gpus[i].mig_instances[j]->size == gpus[i].mig_instances[j]->service->optimal_point->inst_size)
                    {
                        gpc_info_array[gpc_info_idx].batch_size = gpus[i].mig_instances[j]->service->optimal_point->batch_size;
                        gpc_info_array[gpc_info_idx].num_proc = gpus[i].mig_instances[j]->service->optimal_point->num_proc;
                    }
                    else if (gpus[i].mig_instances[j]->size == gpus[i].mig_instances[j]->service->last_instance_point->inst_size)
                    {
                        gpc_info_array[gpc_info_idx].batch_size = gpus[i].mig_instances[j]->service->last_instance_point->batch_size;
                        gpc_info_array[gpc_info_idx].num_proc = gpus[i].mig_instances[j]->service->last_instance_point->num_proc;
                    }
                    else if (gpus[i].mig_instances[j]->size == 2)
                    {
                        gpc_info_array[gpc_info_idx].batch_size = gpus[i].mig_instances[j]->service->size_2_point->batch_size;
                        gpc_info_array[gpc_info_idx].num_proc = gpus[i].mig_instances[j]->service->size_2_point->num_proc;
                    }
                    else if (gpus[i].mig_instances[j]->size == 1)
                    {
                        gpc_info_array[gpc_info_idx].batch_size = gpus[i].mig_instances[j]->service->size_1_point->batch_size;
                        gpc_info_array[gpc_info_idx].num_proc = gpus[i].mig_instances[j]->service->size_1_point->num_proc;
                    }
                    gpc_info_idx++;
                }
            }
            
            for (int j = 0; j < gpc_info_idx - 1; j++) 
            {
                int largest_gpc = j;

                for (int k = j + 1; k < gpc_info_idx; k++)
                {
                    if (gpc_info_array[k].gpc_size > gpc_info_array[largest_gpc].gpc_size) 
                    {
                        largest_gpc = k;
                    }
                }

                if (largest_gpc != j) 
                {
                    gpc_info temp = gpc_info_array[j];
                    gpc_info_array[j] = gpc_info_array[largest_gpc];
                    gpc_info_array[largest_gpc] = temp;
                }
            }

            for (int j = 0; j < gpc_info_idx; j++)
            {
                rp("  %d model (%s) is in %d placement, %d size ", gpc_info_array[j].model_idx, gpc_info_array[j].model_name, gpc_info_array[j].placement, gpc_info_array[j].gpc_size);
                rp("| batch_size: %d, num_proc: %d\n", gpc_info_array[j].batch_size, gpc_info_array[j].num_proc);
                fprintf(deploy, "%d,%d,%s,%d,%d,%d,%d\n", node_idx, gpu_idx, gpc_info_array[j].model_name, gpc_info_array[j].placement, gpc_info_array[j].gpc_size, gpc_info_array[j].batch_size, gpc_info_array[j].num_proc);
            }
            total_gpu_usage++;
        }
    }
    rp("total instance/gpu usage: %d (%f)/%d\n", total_instance_usage, (float)((float)total_instance_usage/7), total_gpu_usage);
    fclose(deploy);
#endif

    return total_gpu_usage;
}