#include "ms_data.h"

#define INITIAL_CAPACITY 100
#define GROWTH_FACTOR 2

// Utility functions
int** allocate_2d_int(int rows, int cols) {
    int** matrix = (int**)malloc(rows * sizeof(int*));
    if (!matrix) return NULL;
    
    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*)malloc(cols * sizeof(int));
        if (!matrix[i]) {
            // Clean up on failure
            for (int j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            return NULL;
        }
    }
    return matrix;
}

void free_2d_int(int** matrix, int rows) {
    if (!matrix) return;
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int* allocate_1d_int(int size) {
    return (int*)malloc(size * sizeof(int));
}

void free_1d_int(int* array) {
    free(array);
}

int is_dat_file(const char* filename) {
    const char* ext = strrchr(filename, '.');
    return ext && strcmp(ext, ".dat") == 0;
}

int is_sol_file(const char* filename) {
    const char* ext = strstr(filename, ".opt.sol");
    return ext && strcmp(ext, ".opt.sol") == 0;
}

char* extract_id_from_filename(const char* filename) {
    char* id = (char*)malloc(256 * sizeof(char));
    if (!id) return NULL;
    
    strcpy(id, filename);
    
    // Remove .dat or .opt.sol extension
    char* ext = strstr(id, ".opt.sol");
    if (ext) {
        *ext = '\0';
    } else {
        ext = strrchr(id, '.');
        if (ext && strcmp(ext, ".dat") == 0) {
            *ext = '\0';
        }
    }
    
    return id;
}

MSData* ms_data_create(const char* data_path, const char* sol_path) {
    MSData* data = (MSData*)malloc(sizeof(MSData));
    if (!data) return NULL;
    
    strcpy(data->data_path, data_path);
    strcpy(data->sol_path, sol_path ? sol_path : data_path);
    
    data->instances = (MSInstance*)malloc(INITIAL_CAPACITY * sizeof(MSInstance));
    data->num_instances = 0;
    data->capacity_instances = INITIAL_CAPACITY;
    
    data->solutions = (MSSolution*)malloc(INITIAL_CAPACITY * sizeof(MSSolution));
    data->num_solutions = 0;
    data->capacity_solutions = INITIAL_CAPACITY;
    
    data->size_groups = (SizeGroup*)malloc(INITIAL_CAPACITY * sizeof(SizeGroup));
    data->num_size_groups = 0;
    data->capacity_size_groups = INITIAL_CAPACITY;
    
    if (!data->instances || !data->solutions || !data->size_groups) {
        ms_data_destroy(data);
        return NULL;
    }
    
    // Load data
    if (ms_data_load(data) != 0 || ms_data_load_solutions(data) != 0) {
        ms_data_destroy(data);
        return NULL;
    }
    
    return data;
}

void ms_data_destroy(MSData* data) {
    if (!data) return;
    
    // Free instances
    for (int i = 0; i < data->num_instances; i++) {
        free_2d_int(data->instances[i].A, data->instances[i].m);
        free_1d_int(data->instances[i].d);
    }
    free(data->instances);
    
    // Free solutions
    for (int i = 0; i < data->num_solutions; i++) {
        free_1d_int(data->solutions[i].x);
    }
    free(data->solutions);
    
    // Free size groups (instances pointers are owned by instances array)
    for (int i = 0; i < data->num_size_groups; i++) {
        free(data->size_groups[i].instances);
    }
    free(data->size_groups);
    
    free(data);
}

int ms_data_load(MSData* data) {
    DIR* dir = opendir(data->data_path);
    if (!dir) {
        fprintf(stderr, "Cannot open directory: %s\n", data->data_path);
        return -1;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (!is_dat_file(entry->d_name)) continue;
        
        char filepath[1024];
        snprintf(filepath, sizeof(filepath), "%s/%s", data->data_path, entry->d_name);
        
        FILE* file = fopen(filepath, "r");
        if (!file) continue;
        
        // Read first line for dimensions
        int m, n;
        if (fscanf(file, "%d %d", &m, &n) != 2) {
            fclose(file);
            continue;
        }
        
        // Expand capacity if needed
        if (data->num_instances >= data->capacity_instances) {
            data->capacity_instances *= GROWTH_FACTOR;
            data->instances = (MSInstance*)realloc(data->instances, 
                data->capacity_instances * sizeof(MSInstance));
            if (!data->instances) {
                fclose(file);
                closedir(dir);
                return -1;
            }
        }
        
        MSInstance* inst = &data->instances[data->num_instances];
        
        // Set ID
        char* id = extract_id_from_filename(entry->d_name);
        if (!id) {
            fclose(file);
            continue;
        }
        strcpy(inst->id, id);
        free(id);
        
        inst->m = m;
        inst->n = n;
        inst->A = allocate_2d_int(m, n);
        inst->d = allocate_1d_int(m);
        
        if (!inst->A || !inst->d) {
            fclose(file);
            continue;
        }
        
        // Read matrix A and vector d
        int success = 1;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (fscanf(file, "%d", &inst->A[i][j]) != 1) {
                    success = 0;
                    break;
                }
            }
            if (!success) break;
            
            if (fscanf(file, "%d", &inst->d[i]) != 1) {
                success = 0;
                break;
            }
        }
        
        fclose(file);
        
        if (success) {
            data->num_instances++;
            
            // Add to size groups
            SizeGroup* group = NULL;
            for (int i = 0; i < data->num_size_groups; i++) {
                if (data->size_groups[i].m == m && data->size_groups[i].n == n) {
                    group = &data->size_groups[i];
                    break;
                }
            }
            
            if (!group) {
                // Create new size group
                if (data->num_size_groups >= data->capacity_size_groups) {
                    data->capacity_size_groups *= GROWTH_FACTOR;
                    data->size_groups = (SizeGroup*)realloc(data->size_groups,
                        data->capacity_size_groups * sizeof(SizeGroup));
                }
                
                group = &data->size_groups[data->num_size_groups];
                group->m = m;
                group->n = n;
                group->count = 0;
                group->instances = (MSInstance**)malloc(INITIAL_CAPACITY * sizeof(MSInstance*));
                data->num_size_groups++;
            }
            
            // Add instance to group
            if (group->count < INITIAL_CAPACITY) {  // Simplified capacity management
                group->instances[group->count] = inst;
                group->count++;
            }
        } else {
            free_2d_int(inst->A, m);
            free_1d_int(inst->d);
        }
    }
    
    closedir(dir);
    return 0;
}

int ms_data_load_solutions(MSData* data) {
    DIR* dir = opendir(data->sol_path);
    if (!dir) {
        fprintf(stderr, "Cannot open solution directory: %s\n", data->sol_path);
        return -1;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (!is_sol_file(entry->d_name)) continue;
        
        char filepath[1024];
        snprintf(filepath, sizeof(filepath), "%s/%s", data->sol_path, entry->d_name);
        
        FILE* file = fopen(filepath, "r");
        if (!file) continue;
        
        // Expand capacity if needed
        if (data->num_solutions >= data->capacity_solutions) {
            data->capacity_solutions *= GROWTH_FACTOR;
            data->solutions = (MSSolution*)realloc(data->solutions,
                data->capacity_solutions * sizeof(MSSolution));
            if (!data->solutions) {
                fclose(file);
                closedir(dir);
                return -1;
            }
        }
        
        MSSolution* sol = &data->solutions[data->num_solutions];
        
        // Set ID
        char* id = extract_id_from_filename(entry->d_name);
        if (!id) {
            fclose(file);
            continue;
        }
        strcpy(sol->id, id);
        free(id);
        
        // Parse solution file
        char line[1024];
        int x_vars[10000] = {0}; // Assuming max 10000 variables
        int max_var = 0;
        
        while (fgets(line, sizeof(line), file)) {
            // Skip comments and empty lines
            if (line[0] == '#' || line[0] == '\n') continue;
            
            // Parse x# variables
            if (strncmp(line, "x#", 2) == 0) {
                int var_num, var_val;
                if (sscanf(line, "x#%d %d", &var_num, &var_val) == 2) {
                    if (var_num > 0 && var_num < 10000) {
                        x_vars[var_num] = var_val;
                        if (var_num > max_var) max_var = var_num;
                    }
                }
            }
        }
        
        fclose(file);
        
        if (max_var > 0) {
            sol->n = max_var;
            sol->x = allocate_1d_int(max_var);
            if (sol->x) {
                for (int i = 0; i < max_var; i++) {
                    sol->x[i] = x_vars[i + 1]; // x_vars is 1-indexed
                }
                data->num_solutions++;
            }
        }
    }
    
    closedir(dir);
    return 0;
}

MSInstance* ms_data_get_by_id(MSData* data, const char* id) {
    for (int i = 0; i < data->num_instances; i++) {
        if (strcmp(data->instances[i].id, id) == 0) {
            return &data->instances[i];
        }
    }
    return NULL;
}

MSInstance** ms_data_get_by_size(MSData* data, int m, int n, int* count) {
    for (int i = 0; i < data->num_size_groups; i++) {
        if (data->size_groups[i].m == m && data->size_groups[i].n == n) {
            *count = data->size_groups[i].count;
            return data->size_groups[i].instances;
        }
    }
    *count = 0;
    return NULL;
}

MSInstance** ms_data_get_by_m(MSData* data, int m, int* count) {
    static MSInstance* results[10000]; // Static buffer for results
    *count = 0;
    
    for (int i = 0; i < data->num_instances && *count < 10000; i++) {
        if (data->instances[i].m == m) {
            results[*count] = &data->instances[i];
            (*count)++;
        }
    }
    
    return *count > 0 ? results : NULL;
}

MSInstance** ms_data_get_by_n(MSData* data, int n, int* count) {
    static MSInstance* results[10000]; // Static buffer for results
    *count = 0;
    
    for (int i = 0; i < data->num_instances && *count < 10000; i++) {
        if (data->instances[i].n == n) {
            results[*count] = &data->instances[i];
            (*count)++;
        }
    }
    
    return *count > 0 ? results : NULL;
}

MSSolution* ms_data_get_solution(MSData* data, const char* id) {
    for (int i = 0; i < data->num_solutions; i++) {
        if (strcmp(data->solutions[i].id, id) == 0) {
            return &data->solutions[i];
        }
    }
    return NULL;
}

int** ms_data_column_filter(MSData* data, const char* instance_id, int additional_cols, int* filtered_n) {
    MSInstance* inst = ms_data_get_by_id(data, instance_id);
    MSSolution* sol = ms_data_get_solution(data, instance_id);
    
    if (!inst || !sol) {
        *filtered_n = 0;
        return NULL;
    }
    
    // Count columns to keep (where solution == 1)
    int keep_count = 0;
    for (int j = 0; j < sol->n && j < inst->n; j++) {
        if (sol->x[j] == 1) keep_count++;
    }
    
    // Count filtered columns (where solution == 0)
    int filtered_count = 0;
    int filtered_indices[10000];
    for (int j = 0; j < sol->n && j < inst->n; j++) {
        if (sol->x[j] == 0) {
            filtered_indices[filtered_count] = j;
            filtered_count++;
        }
    }
    
    // Determine final column count
    int final_cols = keep_count + (additional_cols < filtered_count ? additional_cols : filtered_count);
    *filtered_n = final_cols;
    
    // Allocate filtered matrix
    int** filtered_A = allocate_2d_int(inst->m, final_cols);
    if (!filtered_A) return NULL;
    
    int col_idx = 0;
    
    // Add kept columns first
    for (int j = 0; j < sol->n && j < inst->n; j++) {
        if (sol->x[j] == 1) {
            for (int i = 0; i < inst->m; i++) {
                filtered_A[i][col_idx] = inst->A[i][j];
            }
            col_idx++;
        }
    }
    
    // Add additional filtered columns
    for (int k = 0; k < additional_cols && k < filtered_count; k++) {
        int j = filtered_indices[k];
        for (int i = 0; i < inst->m; i++) {
            filtered_A[i][col_idx] = inst->A[i][j];
        }
        col_idx++;
    }
    
    return filtered_A;
}

void ms_data_print_stats(MSData* data) {
    if (data->num_instances == 0) {
        printf("No instances loaded.\n");
        return;
    }
    
    // Calculate ranges
    int min_m = data->instances[0].m, max_m = data->instances[0].m;
    int min_n = data->instances[0].n, max_n = data->instances[0].n;
    
    for (int i = 1; i < data->num_instances; i++) {
        if (data->instances[i].m < min_m) min_m = data->instances[i].m;
        if (data->instances[i].m > max_m) max_m = data->instances[i].m;
        if (data->instances[i].n < min_n) min_n = data->instances[i].n;
        if (data->instances[i].n > max_n) max_n = data->instances[i].n;
    }
    
    printf("Total instances: %d\n", data->num_instances);
    printf("M range: %d-%d\n", min_m, max_m);
    printf("N range: %d-%d\n", min_n, max_n);
    printf("Solutions loaded: %d\n", data->num_solutions);
    printf("Instances per size:\n");
    
    for (int i = 0; i < data->num_size_groups; i++) {
        printf("  %2d x %3d: %d\n", 
               data->size_groups[i].m, 
               data->size_groups[i].n, 
               data->size_groups[i].count);
    }
}