#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <getopt.h>
#include "ms_data.h"
#include "ms_solve.h"

void print_usage(const char* program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("Options:\n");
    printf("  --data_path PATH    Path to instance data (default: ms_instance/01-marketsplit/instances)\n");
    printf("  --sol_path PATH     Path to solution data (default: ms_instance/01-marketsplit/solutions)\n");
    printf("  --max_sols N        Maximum number of solutions to find (-1 for all, default: -1)\n");
    printf("  --debug             Enable debug output\n");
    printf("  -h, --help          Show this help message\n");
}

void print_and_log(const char* text, FILE* log_file) {
    printf("%s\n", text);
    if (log_file) {
        fprintf(log_file, "%s\n", text);
        fflush(log_file);
    }
}

void free_ms_result(MSResult* result) {
    if (!result) return;
    
    if (result->solutions) {
        for (int i = 0; i < result->solutions_count; i++) {
            free(result->solutions[i].solution);
        }
        free(result->solutions);
    }
    free(result);
}

int main(int argc, char* argv[]) {
    // Default parameters
    char data_path[512] = "ms_instance/01-marketsplit/instances";
    char sol_path[512] = "ms_instance/01-marketsplit/solutions";
    int max_sols = -1;
    int debug = 0;
    
    // Parse command line arguments
    static struct option long_options[] = {
        {"data_path", required_argument, 0, 'd'},
        {"sol_path", required_argument, 0, 's'},
        {"max_sols", required_argument, 0, 'm'},
        {"debug", no_argument, 0, 'g'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };
    
    int option_index = 0;
    int c;
    
    while ((c = getopt_long(argc, argv, "d:s:m:gh", long_options, &option_index)) != -1) {
        switch (c) {
            case 'd':
                strncpy(data_path, optarg, sizeof(data_path) - 1);
                break;
            case 's':
                strncpy(sol_path, optarg, sizeof(sol_path) - 1);
                break;
            case 'm':
                max_sols = atoi(optarg);
                break;
            case 'g':
                debug = 1;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            case '?':
                print_usage(argv[0]);
                return 1;
            default:
                abort();
        }
    }
    
    // Load data
    printf("Loading data from %s and %s...\n", data_path, sol_path);
    MSData* ms_data = ms_data_create(data_path, sol_path);
    if (!ms_data) {
        fprintf(stderr, "Failed to load data\n");
        return 1;
    }
    
    printf("Loaded %d instances\n", ms_data->num_instances);
    ms_data_print_stats(ms_data);
    printf("\n");
    
    // Test configuration
    int test_m_values[] = {3, 4, 5, 6, 7};
    int num_test_m = sizeof(test_m_values) / sizeof(test_m_values[0]);
    
    // Collect all results
    MSResult** all_results = NULL;
    int total_results = 0;
    int results_capacity = 100;
    
    all_results = (MSResult**)malloc(results_capacity * sizeof(MSResult*));
    if (!all_results) {
        fprintf(stderr, "Memory allocation failed\n");
        ms_data_destroy(ms_data);
        return 1;
    }
    
    // Process each m value
    for (int m_idx = 0; m_idx < num_test_m; m_idx++) {
        int m = test_m_values[m_idx];
        int count;
        MSInstance** instances = ms_data_get_by_m(ms_data, m, &count);
        
        printf("Testing %d instances with m = %d\n", count, m);
        
        if (!instances) continue;
        
        for (int i = 0; i < count; i++) {
            MSInstance* inst = instances[i];
            MSSolution* opt_sol = ms_data_get_solution(ms_data, inst->id);
            
            // Expand results array if needed
            if (total_results >= results_capacity) {
                results_capacity *= 2;
                all_results = (MSResult**)realloc(all_results, results_capacity * sizeof(MSResult*));
                if (!all_results) {
                    fprintf(stderr, "Memory reallocation failed\n");
                    return 1;
                }
            }
            
            // Run solver
            MSResult* result = ms_run(inst->A, inst->d, inst->m, inst->n, inst->id,
                                    opt_sol ? opt_sol->x : NULL,
                                    opt_sol ? opt_sol->n : 0,
                                    max_sols, debug);
            
            all_results[total_results] = result;
            total_results++;
            
            // Print status
            const char* status = result->success ? "✓" : "✗";
            const char* opt_status = result->optimal_found ? "✓" : "✗";
            
            printf("%s %s: %d solutions, optimal: %s, bt_loops: %ld, "
                   "dive_loops: %ld, 1st_prune: %ld, 2nd_prune: %ld, 3rd_prune: %ld, "
                   "time: %.4fs, 1st_sol: %.4fs, 1st_bt: %ld, init: %.4fs\n",
                   status, result->id, result->solutions_count, opt_status,
                   result->backtrack_loops, result->dive_loops,
                   result->first_pruning_effect_count, result->second_pruning_effect_count,
                   result->third_pruning_effect_count, result->solve_time,
                   result->first_solution_time, result->first_sol_bt_loops,
                   result->init_time);
        }
        printf("\n");
    }
    
    // Generate log filename
    time_t now;
    time(&now);
    struct tm* tm_info = localtime(&now);
    
    char m_choices[64] = "";
    for (int i = 0; i < num_test_m; i++) {
        char temp[16];
        sprintf(temp, "%s%d", (i > 0) ? "_" : "", test_m_values[i]);
        strcat(m_choices, temp);
    }
    
    char log_filename[256];
    snprintf(log_filename, sizeof(log_filename), "res_%s_%dd%dh%dm%ds.log",
             m_choices, tm_info->tm_mday, tm_info->tm_hour, tm_info->tm_min, tm_info->tm_sec);
    
    // Write results table
    FILE* log_file = fopen(log_filename, "w");
    if (!log_file) {
        fprintf(stderr, "Warning: Could not create log file %s\n", log_filename);
    }
    
    char line[256];
    
    sprintf(line, "================================================================================");
    print_and_log(line, log_file);
    print_and_log("RESULTS", log_file);
    print_and_log(line, log_file);
    print_and_log("", log_file);
    
    sprintf(line, "%-15s %-8s %-8s %-8s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-8s %-12s %-12s",
            "ID", "Size", "Status", "Optimal", "Time(s)", "1st_Sol(s)", "1st_Prune", "2nd_Prune", 
            "3rd_Prune", "Init(s)", "Solutions", "1st_BT", "BT_Loops", "Dive_Loops");
    print_and_log(line, log_file);
    
    sprintf(line, "--------------------------------------------------------------------------------");
    print_and_log(line, log_file);
    
    for (int i = 0; i < total_results; i++) {
        MSResult* result = all_results[i];
        MSInstance* inst = ms_data_get_by_id(ms_data, result->id);
        
        char size_str[16];
        snprintf(size_str, sizeof(size_str), "(%d,%d)", inst->m, inst->n);
        
        const char* status = result->success ? "SUCCESS" : "FAILED";
        const char* optimal = result->optimal_found ? "✓" : "✗";
        
        sprintf(line, "%-15s %-8s %-8s %-8s %-10.4f %-10.4f %-10ld %-10ld %-10ld %-10.4f %-10d %-8ld %-12ld %-12ld",
                result->id, size_str, status, optimal, result->solve_time,
                result->first_solution_time, result->first_pruning_effect_count,
                result->second_pruning_effect_count, result->third_pruning_effect_count,
                result->init_time, result->solutions_count, result->first_sol_bt_loops,
                result->backtrack_loops, result->dive_loops);
        print_and_log(line, log_file);
    }
    
    print_and_log("", log_file);
    sprintf(line, "================================================================================");
    print_and_log(line, log_file);
    
    if (log_file) {
        fclose(log_file);
        printf("Results saved to %s\n", log_filename);
    }
    
    // Write solutions to file
    char sol_filename[256];
    strcpy(sol_filename, log_filename);
    char* ext = strstr(sol_filename, ".log");
    if (ext) {
        strcpy(ext, ".sol");
        
        FILE* sol_file = fopen(sol_filename, "w");
        if (sol_file) {
            for (int i = 0; i < total_results; i++) {
                MSResult* result = all_results[i];
                if (result->success && result->solutions_count > 0) {
                    fprintf(sol_file, "=======%s=============\n", result->id);
                    for (int j = 0; j < result->solutions_count; j++) {
                        fprintf(sol_file, "-----------%d-th solution------------\n", j + 1);
                        for (int k = 0; k < result->solutions[j].n; k++) {
                            fprintf(sol_file, "%d%s", result->solutions[j].solution[k],
                                   (k < result->solutions[j].n - 1) ? " " : "\n");
                        }
                    }
                    fprintf(sol_file, "\n");
                }
            }
            fclose(sol_file);
            printf("Solutions saved to %s\n", sol_filename);
        }
    }
    
    // Cleanup
    for (int i = 0; i < total_results; i++) {
        free_ms_result(all_results[i]);
    }
    free(all_results);
    ms_data_destroy(ms_data);
    
    return 0;
}