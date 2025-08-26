#ifndef MS_SOLVE_H
#define MS_SOLVE_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <fplll/fplll.h>

typedef struct {
    int* solution;
    int n;
} Solution;

typedef struct {
    Solution* solutions;
    int count;
    int capacity;
} SolutionSet;

typedef struct {
    // Input data
    int** A;
    int* d;
    int m, n;
    int n_basis;
    int* r;
    int rmax;
    int* c;
    
    // Extended matrix
    int** L;
    int L_rows, L_cols;
    
    // Lattice basis (stored as row vectors)
    int** basis;      // n_basis x (n+1) matrix
    
    // Gram-Schmidt orthogonalization
    double** b_hat;   // n_basis x (n+1) matrix
    double* b_hat_norms_sq;
    double** mu;      // n_basis x n_basis matrix
    
    // Dual basis
    double** b_bar;   // n_basis x (n+1) matrix
    double* b_bar_norms_l1;
    double* b_bar_norms_l2;
    
    // Coordinates
    double** coords;  // n_basis x (n+1) matrix
    
    // Statistics
    long backtrack_loops;
    long first_sol_bt_loops;
    long dive_loops;
    long first_pruning_effect_count;
    long second_pruning_effect_count;
    long third_pruning_effect_count;
    
    // Timing
    double first_solution_time;
    struct timeval start_time;
    
    // Configuration
    int max_sols;
    int debug;
} MarketSplit;

typedef struct {
    char id[256];
    int solutions_count;
    Solution* solutions;
    int optimal_found;
    long backtrack_loops;
    long first_sol_bt_loops;
    long dive_loops;
    long first_pruning_effect_count;
    long second_pruning_effect_count;
    long third_pruning_effect_count;
    double solve_time;
    double first_solution_time;
    double init_time;
    int success;
    char error[512];
} MSResult;

// Function declarations
MarketSplit* ms_create(int** A, int* d, int m, int n, int* r, int max_sols, int debug);
void ms_destroy(MarketSplit* ms);

int ms_get_extended_matrix(MarketSplit* ms, int N);
int ms_get_reduced_basis(MarketSplit* ms);
int ms_get_gso(MarketSplit* ms);
int ms_compute_dual_norms(MarketSplit* ms);
int ms_get_coordinates(MarketSplit* ms);

int ms_verify_gso(MarketSplit* ms, double tol);
int ms_verify_dual(MarketSplit* ms, double tol);

SolutionSet* ms_enumerate(MarketSplit* ms);
MSResult* ms_run(int** A, int* d, int m, int n, const char* instance_id, int* opt_sol, int opt_n, int max_sols, int debug);

// Solution set management
SolutionSet* solution_set_create();
void solution_set_destroy(SolutionSet* set);
int solution_set_add(SolutionSet* set, int* solution, int n);
int solutions_equal(int* sol1, int* sol2, int n);

// Utility functions
int compute_lcm_array(int* nums, int count);
int gcd(int a, int b);
int lcm(int a, int b);
double get_time_diff(struct timeval start, struct timeval end);
double get_current_time();

// Matrix utilities for lattice operations
double** allocate_2d_double(int rows, int cols);
void free_2d_double(double** matrix, int rows);
double vector_dot_product(double* a, double* b, int n);
double vector_norm_l1(double* v, int n);
double vector_norm_l2(double* v, int n);
int vectors_equal_approx(double* a, double* b, int n, double tol);

#endif // MS_SOLVE_H