#ifndef MS_DATA_H
#define MS_DATA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>

typedef struct {
    char id[256];
    int m, n;
    int **A;  // m x n matrix
    int *d;   // m-dimensional vector
} MSInstance;

typedef struct {
    char id[256];
    int n;
    int *x;   // n-dimensional solution vector
} MSSolution;

typedef struct {
    int m, n;
    int count;
    MSInstance **instances;
} SizeGroup;

typedef struct {
    char data_path[512];
    char sol_path[512];
    
    MSInstance *instances;
    int num_instances;
    int capacity_instances;
    
    MSSolution *solutions;
    int num_solutions;
    int capacity_solutions;
    
    SizeGroup *size_groups;
    int num_size_groups;
    int capacity_size_groups;
} MSData;

// Function declarations
MSData* ms_data_create(const char* data_path, const char* sol_path);
void ms_data_destroy(MSData* data);

int ms_data_load(MSData* data);
int ms_data_load_solutions(MSData* data);

MSInstance* ms_data_get_by_id(MSData* data, const char* id);
MSInstance** ms_data_get_by_size(MSData* data, int m, int n, int* count);
MSInstance** ms_data_get_by_m(MSData* data, int m, int* count);
MSInstance** ms_data_get_by_n(MSData* data, int n, int* count);

MSSolution* ms_data_get_solution(MSData* data, const char* id);

int** ms_data_column_filter(MSData* data, const char* instance_id, int additional_cols, int* filtered_n);

void ms_data_print_stats(MSData* data);

// Utility functions
int** allocate_2d_int(int rows, int cols);
void free_2d_int(int** matrix, int rows);
int* allocate_1d_int(int size);
void free_1d_int(int* array);

int is_dat_file(const char* filename);
int is_sol_file(const char* filename);
char* extract_id_from_filename(const char* filename);

#endif // MS_DATA_H