#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <time.h>

#include "parva_sched.h"
#include "queue.h"

int demand_matching(int model_idx)
{
    int num_optimal_points = svc_level_obj_arr[model_idx].num_optimal_points;
    int optimal_instance_size = svc_level_obj_arr[model_idx].optimal_point->inst_size;
    int optimal_batch_size = svc_level_obj_arr[model_idx].optimal_point->batch_size;
    int optimal_num_proc = svc_level_obj_arr[model_idx].optimal_point->num_proc;
    float optimal_trp = svc_level_obj_arr[model_idx].optimal_point->trp*(float)num_optimal_points;
    float optimal_ltc = svc_level_obj_arr[model_idx].optimal_point->ltc;
    float optimal_trp_per_instance = optimal_trp/optimal_instance_size;
    float req_rate = svc_level_obj_arr[model_idx].req_rate;
    float slo_ltc = svc_level_obj_arr[model_idx].slo_ltc;
    int total_instance_usage = optimal_instance_size*num_optimal_points;
    float last_instance_trp = req_rate - optimal_trp;

    for (int i = 0; i < MAX_INSTANCE; i++)
    {
        if (i == 0)
        {
            if (last_instance_trp <= svc_level_obj_arr[model_idx].size_1_point->trp)
            {
                svc_level_obj_arr[model_idx].last_instance_point = svc_level_obj_arr[model_idx].size_1_point;
                optimal_trp += svc_level_obj_arr[model_idx].size_1_point->trp;
                break;
            }
        }
        else if (i == 1)
        {
            if (last_instance_trp <= svc_level_obj_arr[model_idx].size_2_point->trp)
            {
                svc_level_obj_arr[model_idx].last_instance_point = svc_level_obj_arr[model_idx].size_2_point;
                optimal_trp += svc_level_obj_arr[model_idx].size_2_point->trp;
                break;
            }
        }
        else if (i == 2)
        {
            if (last_instance_trp <= svc_level_obj_arr[model_idx].size_3_point->trp)
            {
                svc_level_obj_arr[model_idx].last_instance_point = svc_level_obj_arr[model_idx].size_3_point;
                optimal_trp += svc_level_obj_arr[model_idx].size_3_point->trp;
                break;
            }
        }
        else if (i == 3)
        {
            if (last_instance_trp <= svc_level_obj_arr[model_idx].size_4_point->trp)
            {
                svc_level_obj_arr[model_idx].last_instance_point = svc_level_obj_arr[model_idx].size_4_point;
                optimal_trp += svc_level_obj_arr[model_idx].size_4_point->trp;
                break;
            }
        }
        else if (i == 4)
        {
            if (last_instance_trp <= svc_level_obj_arr[model_idx].size_7_point->trp)
            {
                svc_level_obj_arr[model_idx].last_instance_point = svc_level_obj_arr[model_idx].size_7_point;
                optimal_trp += svc_level_obj_arr[model_idx].size_7_point->trp;
                break;
            }
        }
    }
    
    total_instance_usage += svc_level_obj_arr[model_idx].last_instance_point->inst_size;

    dp("%d model (%s) select | instance size: %d (%d), batch size: %d, # of proc: %d\n", model_idx, svc_level_obj_arr[model_idx].model, optimal_instance_size, num_optimal_points, optimal_batch_size, optimal_num_proc);
    dp("last instance: (%d,%d,%d), trp: %f | rest req rate: %f\n", svc_level_obj_arr[model_idx].last_instance_point->inst_size, svc_level_obj_arr[model_idx].last_instance_point->batch_size, svc_level_obj_arr[model_idx].last_instance_point->num_proc, svc_level_obj_arr[model_idx].last_instance_point->trp, last_instance_trp);
    dp("trp: %f/%f, ltc: %f/%f, trp_per_instance: %f\n", optimal_trp, req_rate, optimal_ltc, slo_ltc, optimal_trp_per_instance);

    return total_instance_usage;
}

int optimal_triplet_decision() 
{
    dp("ParvaGPU scheduling start\n");
    int total_instance_usage = 0;
    int total_available_services = 0;

    struct timespec start, end;
    double demand_matching_time = 0;

    for (int model_idx = 0; model_idx < num_models; model_idx++)
    {
        float max_trp_per_instance = FLT_MIN;
        int min_instance_size = INT_MAX;
        char *model = svc_level_obj_arr[model_idx].model;
        float req_rate = svc_level_obj_arr[model_idx].req_rate;
        float slo_ltc = svc_level_obj_arr[model_idx].slo_ltc;

        float size_1_max_trp_per_instance = FLT_MIN;
        float size_2_max_trp_per_instance = FLT_MIN;
        float size_3_max_trp_per_instance = FLT_MIN;
        float size_4_max_trp_per_instance = FLT_MIN;
        float size_7_max_trp_per_instance = FLT_MIN;

        dp("start model %d (%s) | %f %f\n", model_idx, model, req_rate, slo_ltc);
        for (int instance_size = 0; instance_size < MAX_INSTANCE; instance_size++)
        {
            for (int batch_size = 0; batch_size < MAX_BATCH; batch_size++)
            {
                for (int num_proc = 0; num_proc < MAX_NUM_PROC; num_proc++)
                {
                    float target_trp = prof_data_arr[model_idx][instance_size][batch_size][num_proc].trp;
                    float target_ltc = prof_data_arr[model_idx][instance_size][batch_size][num_proc].ltc;
                    float target_trp_per_instance = 0;
                    int target_instance_size;

                    if (instance_size == 4)
                    {
                        target_trp_per_instance = (target_trp/(float)(7));    
                        target_instance_size = 7;
                    }
                    else
                    {
                        target_trp_per_instance = (target_trp/(float)(instance_size+1));
                        target_instance_size = instance_size+1;
                    }

                    float batch_time = (float)(((batch_size*(num_proc+1))/req_rate))*1000;
                    
                    if (target_ltc < (float)(slo_ltc/2 * 0.9))
                    {
                        if ((target_trp_per_instance > max_trp_per_instance))
                        {
                            dp("optimal | %d, %f, %f | %f + %f, %f\n", target_instance_size, target_trp_per_instance, max_trp_per_instance, target_ltc, batch_time, svc_level_obj_arr[model_idx].slo_ltc);
                            max_trp_per_instance = target_trp_per_instance;
                            min_instance_size = target_instance_size;
                            svc_level_obj_arr[model_idx].optimal_point = &(prof_data_arr[model_idx][instance_size][batch_size][num_proc]);
                            svc_level_obj_arr[model_idx].num_optimal_points = (int)floor(req_rate/target_trp);
                            if (fmod(req_rate,target_trp) == 0)
                                svc_level_obj_arr[model_idx].num_optimal_points = svc_level_obj_arr[model_idx].num_optimal_points - 1;
                            dp("optimal | %d number of instance\n", svc_level_obj_arr[model_idx].num_optimal_points);
                        }

                        if ((target_instance_size == 1) && (target_trp_per_instance > size_1_max_trp_per_instance))
                        {
                            size_1_max_trp_per_instance = target_trp_per_instance;
                            svc_level_obj_arr[model_idx].size_1_point = &(prof_data_arr[model_idx][instance_size][batch_size][num_proc]);
                        }
                        if ((target_instance_size == 2) && (target_trp_per_instance > size_2_max_trp_per_instance))
                        {
                            size_2_max_trp_per_instance = target_trp_per_instance;
                            svc_level_obj_arr[model_idx].size_2_point = &(prof_data_arr[model_idx][instance_size][batch_size][num_proc]);
                        }
                        if ((target_instance_size == 3) && (target_trp_per_instance > size_3_max_trp_per_instance))
                        {
                            size_3_max_trp_per_instance = target_trp_per_instance;
                            svc_level_obj_arr[model_idx].size_3_point = &(prof_data_arr[model_idx][instance_size][batch_size][num_proc]);
                        }
                        if ((target_instance_size == 4) && (target_trp_per_instance > size_4_max_trp_per_instance))
                        {
                            size_4_max_trp_per_instance = target_trp_per_instance;
                            svc_level_obj_arr[model_idx].size_4_point = &(prof_data_arr[model_idx][instance_size][batch_size][num_proc]);
                        }
                        if ((target_instance_size == 7) && (target_trp_per_instance > size_7_max_trp_per_instance))
                        {
                            size_7_max_trp_per_instance = target_trp_per_instance;
                            svc_level_obj_arr[model_idx].size_7_point = &(prof_data_arr[model_idx][instance_size][batch_size][num_proc]);
                        }
                    }
                }
            }
        }

        dp("finish loop\n");

        if (!svc_level_obj_arr[model_idx].optimal_point)
        {
            dp("%d model (%s) select | there is no optimal partition | req rate: %f, slo latency: %f\n", model_idx, model, req_rate, slo_ltc);
        }
        else
        {
            clock_gettime(CLOCK_MONOTONIC, &start);
            total_instance_usage += demand_matching(model_idx);
            clock_gettime(CLOCK_MONOTONIC, &end);
            demand_matching_time += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000000000.0;

            total_available_services++;
        }

        dp("\n");
    }
    dp("Total instance/gpu usage: %d/%f | time: %f\n", total_instance_usage, (float)total_instance_usage/7, demand_matching_time);

    return total_available_services;
}