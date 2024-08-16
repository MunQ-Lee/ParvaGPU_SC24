#ifndef _SCHED_H_
#define _SCHED_H_
#include <limits.h>

// Debug print enable
// #define DEBUG
#ifndef DEBUG
#define dp(fmt,args...) 
#define dlp(fmt,args...) 
#else
#define dp(fmt,args...) printf( fmt, ## args )
#define dlp(fmt,args...) printf( "[%s: line%d] " fmt, __func__,__LINE__, ## args )
#endif

// Result print enable
#define RESULT_PRINT
// #define REQ_RATE_CHECK
#ifndef RESULT_PRINT
#define rp(fmt,args...) 
#define rlp(fmt,args...) 
#else
#define rp(fmt,args...) printf( fmt, ## args )
#define rlp(fmt,args...) printf( "[%s: line%d] " fmt, __func__,__LINE__, ## args )
#endif

#ifndef __cplusplus
#define max(x, y) (x) > (y) ? (x) : (y)
#define min(x, y) (x) < (y) ? (x) : (y)
typedef enum {false, true} bool;

int num_files;
#endif

extern int num_models;
extern int SCENARIO;

#define TOTAL_GPU 2000
#define NUM_GPU_PER_NODE 8
#define NUM_MODEL 11
#define MAX_INSTANCE 5
#define MAX_MIG_SIZE 7
#define MAX_BATCH 1024
#define MAX_NUM_PROC 3
#define NUM_SCEN 6
#define MAX_FILENAME_LEN 256
#define MAX_LINE_LENGTH MAX_INSTANCE*MAX_BATCH*MAX_NUM_PROC

#define DATA_GEN
// #define NO_MPS
// #define NO_OPTIM

typedef struct prof_data
{
    int model_idx;
    int inst_size;
    int batch_size;
    int num_proc;
    float trp;
    float ltc;

    float sm_active;
    float sm_occu;
    float dram_util;
    float tensor_active;
} prof_data;

typedef struct svc_level_obj
{
    char *model;
    int model_idx;

    float req_rate;
    float slo_ltc;

    float max_trp;
    float min_trp;
    float max_ltc;
    float min_ltc;

    struct mig_instance *instance;

    int num_optimal_points;
    struct prof_data *optimal_point;
    struct prof_data *last_instance_point;

    struct prof_data *size_1_point;
    struct prof_data *size_2_point;
    struct prof_data *size_3_point;
    struct prof_data *size_4_point;
    struct prof_data *size_7_point;    
} svc_level_obj;

typedef struct mig_instance
{
    int placement;
    int size;

    struct svc_level_obj *service;
} mig_instance;

typedef struct gpu_device
{
    int idx;
    int usage;
    int config;

    float avail_req_rate;
    struct mig_instance *mig_instances[7];
} gpu_device;

extern struct prof_data ****prof_data_arr;
extern struct svc_level_obj *svc_level_obj_arr;
extern struct gpu_device *gpus;

int get_data_from_csv();
int optimal_triplet_decision();
void init_deploy();
int segment_relocation();

#endif