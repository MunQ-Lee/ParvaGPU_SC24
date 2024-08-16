#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <float.h>
#include <time.h>

#include "parva_sched.h"

int MAX_FILES = 11;
int num_files = 0;
char **fileList;

int compare(const void *a, const void *b) {
    return strcmp(*(const char **)a, *(const char **)b);
}

int dir_list()
{
    DIR *dir;
    struct dirent *entry;
    char *directory = "./prof_data"; 

    dir = opendir(directory);
    if (dir == NULL) {
        perror("opendir");
        return 1;
    }

    fileList = malloc(MAX_FILES * sizeof(char *));
    if (fileList == NULL) {
        perror("malloc");
        closedir(dir);
        return 1;
    }

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) { 
            fileList[num_files] = malloc(MAX_FILENAME_LEN);
            if (fileList[num_files] == NULL) {
                perror("malloc");
                closedir(dir);
                return 1;
            }
            strncpy(fileList[num_files], entry->d_name, MAX_FILENAME_LEN);
            num_files++;
            if (num_files >= MAX_FILES) break; 
        }
    }
    qsort(fileList, num_files, sizeof(char *), compare);

    for (int i = 0; i < num_files; i++) {
        dp("%s\n", fileList[i]);
    }
#ifdef DATA_GEN
    num_models = NUM_MODEL;
#else
    num_models = num_files;
#endif
    closedir(dir);
    return 0;
}

int init_prof_data_arr()
{
    int i, j, k, l = 0;
    prof_data_arr = malloc(num_models * sizeof(struct prof_data ***));
    svc_level_obj_arr = malloc(num_models * sizeof(struct svc_level_obj));

    if (prof_data_arr == NULL || svc_level_obj_arr == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    for (i = 0; i < num_models; i++) {
        prof_data_arr[i] = malloc(MAX_INSTANCE * sizeof(struct prof_data **));
        if (prof_data_arr[i] == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }

        for (j = 0; j < MAX_INSTANCE; j++) {
            prof_data_arr[i][j] = malloc(MAX_BATCH * sizeof(struct prof_data *));
            if (prof_data_arr[i][j] == NULL) {
                fprintf(stderr, "Memory allocation failed\n");
                return 1;
            }

            for (k = 0; k < MAX_BATCH; k++) {
                prof_data_arr[i][j][k] = malloc(MAX_NUM_PROC * sizeof(struct prof_data));
                if (prof_data_arr[i][j][k] == NULL) {
                    fprintf(stderr, "Memory allocation failed\n");
                    return 1;
                }
                for (l = 0; l < MAX_NUM_PROC; l++)
                {
                    // memset(&prof_data_arr[i][j][k][l], 0, sizeof(prof_data_arr[i][j][k][l]));
                    prof_data_arr[i][j][k][l].trp = 0;
                    prof_data_arr[i][j][k][l].ltc = 0;

                    prof_data_arr[i][j][k][l].sm_active = 0;
                    prof_data_arr[i][j][k][l].sm_occu = 0;
                    prof_data_arr[i][j][k][l].dram_util = 0;
                    prof_data_arr[i][j][k][l].tensor_active = 0;
                }
            }
        }
        svc_level_obj_arr[i].model = (char *)malloc(20);
        svc_level_obj_arr[i].model_idx = i;
        svc_level_obj_arr[i].req_rate = 0;
        svc_level_obj_arr[i].slo_ltc = 0;

        svc_level_obj_arr[i].max_trp = FLT_MIN;
        svc_level_obj_arr[i].min_trp = FLT_MAX;
        svc_level_obj_arr[i].max_ltc = FLT_MIN;
        svc_level_obj_arr[i].min_ltc = FLT_MAX;

        svc_level_obj_arr[i].optimal_point = NULL;
        svc_level_obj_arr[i].size_1_point = NULL;
        svc_level_obj_arr[i].size_2_point = NULL;
    }
    return 0;
}

int get_data_from_csv()
{
    if (NUM_MODEL >= 11)
        MAX_FILES = 11;
    else
        MAX_FILES = NUM_MODEL;

    if (dir_list())
        return 1;
    dp("%d model files listing\n", num_files);

    if(init_prof_data_arr())
        return 1;
    dp("init data array\n");

    srand(time(NULL) ^ (getpid() << 16));

    FILE *req_rate;
    FILE *latency;
    char req_line[MAX_LINE_LENGTH];
    char ltc_line[MAX_LINE_LENGTH];
    char req_rate_dir[MAX_FILENAME_LEN] = "./SLO/SLO_req_rate.csv";
    char latency_dir[MAX_FILENAME_LEN] = "./SLO/SLO_latency.csv";
    float req_rate_slo[NUM_SCEN][num_files];
    float latency_slo[NUM_SCEN][num_files];

    req_rate = fopen(req_rate_dir, "r");
    latency = fopen(latency_dir, "r");
    if (req_rate == NULL || latency == NULL) {
        fprintf(stderr, "File open error\n");
        return 1;
    }

    for (int scenario = 0; scenario < NUM_SCEN; scenario++)
    {
        fgets(req_line, MAX_LINE_LENGTH, req_rate);
        req_line[strcspn(req_line, "\n")] = 0;
        fgets(ltc_line, MAX_LINE_LENGTH, latency);
        ltc_line[strcspn(ltc_line, "\n")] = 0;
        req_rate_slo[scenario][0] = atoi(strtok(req_line, ","));

        for (int model = 1; model < num_models; model++)
        {
            req_rate_slo[scenario][model] = atoi(strtok(NULL, ","));
        }
        latency_slo[scenario][0] = atoi(strtok(ltc_line, ","));
        for (int model = 1; model < num_models; model++)
        {
            latency_slo[scenario][model] = atoi(strtok(NULL, ","));
        }
    }

    for (int model = 0; model < num_models; model++)
    {
        FILE *file;
        char line[MAX_LINE_LENGTH];
        char file_dir[MAX_FILENAME_LEN] = "./prof_data/";

#ifdef DATA_GEN
        strcat(file_dir, fileList[model%num_files]);
        dp("file: %s \n", file_dir);
        strcpy(svc_level_obj_arr[model].model, fileList[model%num_files]);
        size_t length = strlen(svc_level_obj_arr[model].model);
        if (length > 4) {
            svc_level_obj_arr[model].model[length - 4] = '\0';
        }
#else
        strcat(file_dir, fileList[model]);
        dp("file: %s \n", file_dir);
        strcpy(svc_level_obj_arr[model].model, fileList[model]);
        size_t length = strlen(svc_level_obj_arr[model].model);
        if (length > 4) {
            svc_level_obj_arr[model].model[length - 4] = '\0';
        }
#endif

        file = fopen(file_dir, "r");
        if (file == NULL) {
            fprintf(stderr, "File open error\n");
            return 1;
        }

        fgets(line, MAX_LINE_LENGTH, file);

        while (fgets(line, MAX_LINE_LENGTH, file) != NULL) {
            line[strcspn(line, "\n")] = 0;

            int orig_inst_size = atoi(strtok(line, ","));
            int orig_batch_size = atoi(strtok(NULL, ","));
            int orig_num_proc = atoi(strtok(NULL, ","));
            float trp = atof(strtok(NULL, ","));
            float ltc = atof(strtok(NULL, ","));
            
            int inst_size = 0;
            int batch_size = 0;
            int num_proc = 0;

            if (orig_inst_size == 7)
                inst_size = 4;
            else
                inst_size = orig_inst_size - 1;

            trp = trp*orig_num_proc;
            ltc = ltc*1000;
            num_proc = orig_num_proc - 1;
            batch_size = orig_batch_size;
                        
            prof_data_arr[model][inst_size][batch_size][num_proc].model_idx = model;
            prof_data_arr[model][inst_size][batch_size][num_proc].inst_size = orig_inst_size;
            prof_data_arr[model][inst_size][batch_size][num_proc].batch_size = orig_batch_size;
            prof_data_arr[model][inst_size][batch_size][num_proc].num_proc = orig_num_proc;
            prof_data_arr[model][inst_size][batch_size][num_proc].trp = trp;
            prof_data_arr[model][inst_size][batch_size][num_proc].ltc = ltc;
            
            if ((svc_level_obj_arr[model].max_trp < trp) && trp)
                svc_level_obj_arr[model].max_trp = trp;
            
            if ((svc_level_obj_arr[model].min_trp > trp) && trp)
                svc_level_obj_arr[model].min_trp = trp;
            
            if ((svc_level_obj_arr[model].max_ltc < ltc) && ltc)
                svc_level_obj_arr[model].max_ltc = ltc;
            
            if ((svc_level_obj_arr[model].min_ltc > ltc) && ltc)
                svc_level_obj_arr[model].min_ltc = ltc;
        }

        if (req_rate_slo[SCENARIO][model] && latency_slo[SCENARIO][model])
        {
            svc_level_obj_arr[model].req_rate = req_rate_slo[SCENARIO][model];
            svc_level_obj_arr[model].slo_ltc = latency_slo[SCENARIO][model];
            printf("%d model (%s) | request rate: %f, slo ltc: %f\n", model, svc_level_obj_arr[model].model, svc_level_obj_arr[model].req_rate, svc_level_obj_arr[model].slo_ltc);
        }

        fclose(file);    
    }
    fclose(req_rate);
    fclose(latency);
    return 0;
}
