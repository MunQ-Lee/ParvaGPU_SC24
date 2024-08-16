#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include "parva_sched.h"

struct prof_data ****prof_data_arr;
struct svc_level_obj *svc_level_obj_arr;
struct gpu_device *gpus;
int REQ_RATE_RATIO = 0;
int SCENARIO = -1;
int num_models;

int main(int argc, char **argv)
{
    SCENARIO = atoi(argv[1]) - 1;
    printf("Scenario %d selected\n", SCENARIO + 1);

    struct timespec start, end;
	int ParvaGPU_num_gpu = 0;
    int ParvaGPU_num_serive = 0;
    double ParvaGPU_time = 0;

    if (get_data_from_csv())
        return 1;
    dp("data load complete\n");

    init_deploy();

    clock_gettime(CLOCK_MONOTONIC, &start);
    ParvaGPU_num_serive = optimal_triplet_decision();
    ParvaGPU_num_gpu = segment_relocation();
    clock_gettime(CLOCK_MONOTONIC, &end);
    ParvaGPU_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    
    printf("\n\nParvaGPU | num of gpu: %d\n", ParvaGPU_num_gpu);
    printf("ParvaGPU | scheduling time: %f sec\n", ParvaGPU_time);
    printf("ParvaGPU | number of services: %d\n\n", ParvaGPU_num_serive);

    return 0;
}
