#include "ms_solve.h"

// Utility functions
int gcd(int a, int b) {
    a = abs(a);
    b = abs(b);
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

int lcm(int a, int b) {
    return abs(a * b) / gcd(a, b);
}

int compute_lcm_array(int* nums, int count) {
    if (count == 0) return 1;
    int result = nums[0];
    for (int i = 1; i < count; i++) {
        result = lcm(result, nums[i]);
    }
    return result;
}

double get_time_diff(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
}

double get_current_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

double** allocate_2d_double(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    if (!matrix) return NULL;
    
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
        if (!matrix[i]) {
            for (int j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            return NULL;
        }
    }
    return matrix;
}

void free_2d_double(double** matrix, int rows) {
    if (!matrix) return;
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

double vector_dot_product(double* a, double* b, int n) {
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

double vector_norm_l1(double* v, int n) {
    double norm = 0.0;
    for (int i = 0; i < n; i++) {
        norm += fabs(v[i]);
    }
    return norm;
}

double vector_norm_l2(double* v, int n) {
    double norm = 0.0;
    for (int i = 0; i < n; i++) {
        norm += v[i] * v[i];
    }
    return sqrt(norm);
}

int vectors_equal_approx(double* a, double* b, int n, double tol) {
    for (int i = 0; i < n; i++) {
        if (fabs(a[i] - b[i]) > tol) return 0;
    }
    return 1;
}

// Solution set management
SolutionSet* solution_set_create() {
    SolutionSet* set = (SolutionSet*)malloc(sizeof(SolutionSet));
    if (!set) return NULL;
    
    set->capacity = 100;
    set->count = 0;
    set->solutions = (Solution*)malloc(set->capacity * sizeof(Solution));
    if (!set->solutions) {
        free(set);
        return NULL;
    }
    
    return set;
}

void solution_set_destroy(SolutionSet* set) {
    if (!set) return;
    
    for (int i = 0; i < set->count; i++) {
        free(set->solutions[i].solution);
    }
    free(set->solutions);
    free(set);
}

int solutions_equal(int* sol1, int* sol2, int n) {
    for (int i = 0; i < n; i++) {
        if (sol1[i] != sol2[i]) return 0;
    }
    return 1;
}

int solution_set_add(SolutionSet* set, int* solution, int n) {
    if (set->count >= set->capacity) {
        set->capacity *= 2;
        set->solutions = (Solution*)realloc(set->solutions, set->capacity * sizeof(Solution));
        if (!set->solutions) return -1;
    }
    
    Solution* sol = &set->solutions[set->count];
    sol->n = n;
    sol->solution = (int*)malloc(n * sizeof(int));
    if (!sol->solution) return -1;
    
    memcpy(sol->solution, solution, n * sizeof(int));
    set->count++;
    return 0;
}

MarketSplit* ms_create(int** A, int* d, int m, int n, int* r, int max_sols, int debug) {
    MarketSplit* ms = (MarketSplit*)malloc(sizeof(MarketSplit));
    if (!ms) return NULL;
    
    memset(ms, 0, sizeof(MarketSplit));
    
    ms->A = A;
    ms->d = d;
    ms->m = m;
    ms->n = n;
    ms->n_basis = n - m + 1;
    ms->max_sols = max_sols;
    ms->debug = debug;
    
    // Set r vector (default to all ones)
    ms->r = (int*)malloc(n * sizeof(int));
    if (!ms->r) {
        free(ms);
        return NULL;
    }
    
    if (r) {
        memcpy(ms->r, r, n * sizeof(int));
    } else {
        for (int i = 0; i < n; i++) {
            ms->r[i] = 1;
        }
    }
    
    // Compute rmax and c
    ms->rmax = compute_lcm_array(ms->r, n);
    ms->c = (int*)malloc(n * sizeof(int));
    if (!ms->c) {
        free(ms->r);
        free(ms);
        return NULL;
    }
    
    for (int i = 0; i < n; i++) {
        ms->c[i] = ms->rmax / ms->r[i];
    }
    
    gettimeofday(&ms->start_time, NULL);
    
    // Run preprocessing steps
    if (ms_get_extended_matrix(ms, 0) != 0 ||
        ms_get_reduced_basis(ms) != 0 ||
        ms_get_gso(ms) != 0 ||
        ms_compute_dual_norms(ms) != 0 ||
        ms_get_coordinates(ms) != 0) {
        ms_destroy(ms);
        return NULL;
    }
    
    // Verify operations
    if (!ms_verify_gso(ms, 1e-10) || !ms_verify_dual(ms, 1e-10)) {
        if (debug) {
            fprintf(stderr, "Verification failed for instance\n");
        }
    }
    
    return ms;
}

void ms_destroy(MarketSplit* ms) {
    if (!ms) return;
    
    if (ms->r) free(ms->r);
    if (ms->c) free(ms->c);
    
    if (ms->L) free_2d_int(ms->L, ms->L_rows);
    if (ms->basis) free_2d_int(ms->basis, ms->n_basis);
    
    if (ms->b_hat) free_2d_double(ms->b_hat, ms->n_basis);
    if (ms->b_hat_norms_sq) free(ms->b_hat_norms_sq);
    if (ms->mu) free_2d_double(ms->mu, ms->n_basis);
    
    if (ms->b_bar) free_2d_double(ms->b_bar, ms->n_basis);
    if (ms->b_bar_norms_l1) free(ms->b_bar_norms_l1);
    if (ms->b_bar_norms_l2) free(ms->b_bar_norms_l2);
    
    if (ms->coords) free_2d_double(ms->coords, ms->n_basis);
    
    free(ms);
}

int ms_get_extended_matrix(MarketSplit* ms, int N) {
    if (N == 0) {
        // Calculate N based on matrix values
        int max_A = 0, max_d = 0;
        for (int i = 0; i < ms->m; i++) {
            for (int j = 0; j < ms->n; j++) {
                if (abs(ms->A[i][j]) > max_A) max_A = abs(ms->A[i][j]);
            }
            if (abs(ms->d[i]) > max_d) max_d = abs(ms->d[i]);
        }
        
        int digits = 0;
        int temp = max_A;
        while (temp > 0) { digits++; temp /= 10; }
        temp = max_d;
        while (temp > 0) { digits++; temp /= 10; }
        digits += 2;
        
        N = 1;
        for (int i = 0; i < digits; i++) N *= 10;
    }
    
    ms->L_rows = ms->m + ms->n + 1;
    ms->L_cols = ms->n + 1;
    ms->L = allocate_2d_int(ms->L_rows, ms->L_cols);
    if (!ms->L) return -1;
    
    // Initialize to zero
    for (int i = 0; i < ms->L_rows; i++) {
        for (int j = 0; j < ms->L_cols; j++) {
            ms->L[i][j] = 0;
        }
    }
    
    // First column
    for (int i = 0; i < ms->m; i++) {
        ms->L[i][0] = -N * ms->d[i];
    }
    for (int i = 0; i < ms->n; i++) {
        ms->L[ms->m + i][0] = -ms->rmax;
    }
    ms->L[ms->m + ms->n][0] = ms->rmax;
    
    // Top-right block (A matrix)
    for (int i = 0; i < ms->m; i++) {
        for (int j = 0; j < ms->n; j++) {
            ms->L[i][j + 1] = N * ms->A[i][j];
        }
    }
    
    // Middle diagonal block
    for (int i = 0; i < ms->n; i++) {
        ms->L[ms->m + i][i + 1] = 2 * ms->c[i];
    }
    
    return 0;
}

int ms_get_reduced_basis(MarketSplit* ms) {
    // Convert to fplll format and perform reduction
    fplll::ZZ_mat<long> L_fplll(ms->L_cols, ms->L_rows);
    
    // Copy matrix (transpose for fplll)
    for (int i = 0; i < ms->L_rows; i++) {
        for (int j = 0; j < ms->L_cols; j++) {
            L_fplll[j][i] = ms->L[i][j];
        }
    }
    
    // Perform BKZ reduction
    int block_size = (ms->L_cols < 60) ? ms->L_cols / 2 : 30;
    if (block_size < 2) block_size = 2;
    
    fplll::BKZParam param(block_size);
    fplll::bkz_reduction(L_fplll, param);
    
    // Extract null space basis
    ms->basis = allocate_2d_int(ms->n_basis, ms->n + 1);
    if (!ms->basis) return -1;
    
    int basis_count = 0;
    for (int j = 0; j < ms->L_cols && basis_count < ms->n_basis; j++) {
        // Check if this column is in the null space (first m entries are zero)
        int is_null = 1;
        for (int i = 0; i < ms->m; i++) {
            if (L_fplll[j][i] != 0) {
                is_null = 0;
                break;
            }
        }
        
        if (is_null) {
            // Copy the relevant part (from position m onwards)
            for (int i = 0; i < ms->n + 1; i++) {
                ms->basis[basis_count][i] = L_fplll[j][ms->m + i].get_si();
            }
            basis_count++;
        }
    }
    
    if (basis_count != ms->n_basis) {
        if (ms->debug) {
            fprintf(stderr, "Warning: Expected %d basis vectors, got %d\n", ms->n_basis, basis_count);
        }
    }
    
    return 0;
}

int ms_get_gso(MarketSplit* ms) {
    ms->b_hat = allocate_2d_double(ms->n_basis, ms->n + 1);
    ms->mu = allocate_2d_double(ms->n_basis, ms->n_basis);
    ms->b_hat_norms_sq = (double*)malloc(ms->n_basis * sizeof(double));
    
    if (!ms->b_hat || !ms->mu || !ms->b_hat_norms_sq) return -1;
    
    // Initialize mu matrix
    for (int i = 0; i < ms->n_basis; i++) {
        for (int j = 0; j < ms->n_basis; j++) {
            ms->mu[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    
    // Manual Gram-Schmidt orthogonalization
    for (int i = 0; i < ms->n_basis; i++) {
        // Start with original basis vector
        for (int k = 0; k < ms->n + 1; k++) {
            ms->b_hat[i][k] = (double)ms->basis[i][k];
        }
        
        // Subtract projections onto previous orthogonal vectors
        for (int j = 0; j < i; j++) {
            // mu[i][j] = <basis[i], b_hat[j]> / ||b_hat[j]||²
            double dot_prod = 0.0;
            for (int k = 0; k < ms->n + 1; k++) {
                dot_prod += (double)ms->basis[i][k] * ms->b_hat[j][k];
            }
            ms->mu[i][j] = dot_prod / ms->b_hat_norms_sq[j];
            
            // Subtract projection
            for (int k = 0; k < ms->n + 1; k++) {
                ms->b_hat[i][k] -= ms->mu[i][j] * ms->b_hat[j][k];
            }
        }
        
        // Compute squared norm
        ms->b_hat_norms_sq[i] = vector_dot_product(ms->b_hat[i], ms->b_hat[i], ms->n + 1);
    }
    
    return 0;
}

int ms_compute_dual_norms(MarketSplit* ms) {
    // Compute dual basis: b_bar = B * (B^T * B)^(-1)
    // where B is the basis matrix (transposed)
    
    ms->b_bar = allocate_2d_double(ms->n_basis, ms->n + 1);
    ms->b_bar_norms_l1 = (double*)malloc(ms->n_basis * sizeof(double));
    ms->b_bar_norms_l2 = (double*)malloc(ms->n_basis * sizeof(double));
    
    if (!ms->b_bar || !ms->b_bar_norms_l1 || !ms->b_bar_norms_l2) return -1;
    
    // Compute Gram matrix: G = B^T * B
    double** gram = allocate_2d_double(ms->n_basis, ms->n_basis);
    if (!gram) return -1;
    
    for (int i = 0; i < ms->n_basis; i++) {
        for (int j = 0; j < ms->n_basis; j++) {
            gram[i][j] = 0.0;
            for (int k = 0; k < ms->n + 1; k++) {
                gram[i][j] += (double)ms->basis[i][k] * (double)ms->basis[j][k];
            }
        }
    }
    
    // Compute inverse of Gram matrix using Gaussian elimination
    double** gram_inv = allocate_2d_double(ms->n_basis, ms->n_basis);
    double** identity = allocate_2d_double(ms->n_basis, ms->n_basis);
    if (!gram_inv || !identity) {
        free_2d_double(gram, ms->n_basis);
        return -1;
    }
    
    // Initialize identity matrix
    for (int i = 0; i < ms->n_basis; i++) {
        for (int j = 0; j < ms->n_basis; j++) {
            identity[i][j] = (i == j) ? 1.0 : 0.0;
            gram_inv[i][j] = gram[i][j]; // Copy gram to gram_inv
        }
    }
    
    // Gaussian elimination with pivoting
    for (int i = 0; i < ms->n_basis; i++) {
        // Find pivot
        int pivot = i;
        for (int j = i + 1; j < ms->n_basis; j++) {
            if (fabs(gram_inv[j][i]) > fabs(gram_inv[pivot][i])) {
                pivot = j;
            }
        }
        
        // Swap rows
        if (pivot != i) {
            for (int j = 0; j < ms->n_basis; j++) {
                double temp = gram_inv[i][j];
                gram_inv[i][j] = gram_inv[pivot][j];
                gram_inv[pivot][j] = temp;
                
                temp = identity[i][j];
                identity[i][j] = identity[pivot][j];
                identity[pivot][j] = temp;
            }
        }
        
        // Make diagonal element 1
        double pivot_val = gram_inv[i][i];
        if (fabs(pivot_val) < 1e-12) {
            free_2d_double(gram, ms->n_basis);
            free_2d_double(gram_inv, ms->n_basis);
            free_2d_double(identity, ms->n_basis);
            return -1; // Matrix is singular
        }
        
        for (int j = 0; j < ms->n_basis; j++) {
            gram_inv[i][j] /= pivot_val;
            identity[i][j] /= pivot_val;
        }
        
        // Eliminate column
        for (int j = 0; j < ms->n_basis; j++) {
            if (j != i) {
                double factor = gram_inv[j][i];
                for (int k = 0; k < ms->n_basis; k++) {
                    gram_inv[j][k] -= factor * gram_inv[i][k];
                    identity[j][k] -= factor * identity[i][k];
                }
            }
        }
    }
    
    // Compute b_bar = B * gram_inv
    for (int i = 0; i < ms->n_basis; i++) {
        for (int j = 0; j < ms->n + 1; j++) {
            ms->b_bar[i][j] = 0.0;
            for (int k = 0; k < ms->n_basis; k++) {
                ms->b_bar[i][j] += identity[i][k] * (double)ms->basis[k][j];
            }
        }
        
        // Compute norms
        ms->b_bar_norms_l1[i] = vector_norm_l1(ms->b_bar[i], ms->n + 1);
        ms->b_bar_norms_l2[i] = vector_norm_l2(ms->b_bar[i], ms->n + 1);
    }
    
    free_2d_double(gram, ms->n_basis);
    free_2d_double(gram_inv, ms->n_basis);
    free_2d_double(identity, ms->n_basis);
    
    return 0;
}

int ms_get_coordinates(MarketSplit* ms) {
    ms->coords = allocate_2d_double(ms->n_basis, ms->n + 1);
    if (!ms->coords) return -1;
    
    // For now, we'll set coords to basis (simplified)
    // In the Python version, this solves L_bottom @ coords[i] = basis[i]
    for (int i = 0; i < ms->n_basis; i++) {
        for (int j = 0; j < ms->n + 1; j++) {
            ms->coords[i][j] = (double)ms->basis[i][j];
        }
    }
    
    return 0;
}

int ms_verify_gso(MarketSplit* ms, double tol) {
    // Check orthogonality
    for (int i = 0; i < ms->n_basis; i++) {
        for (int j = i + 1; j < ms->n_basis; j++) {
            double dot_prod = vector_dot_product(ms->b_hat[i], ms->b_hat[j], ms->n + 1);
            if (fabs(dot_prod) > tol) {
                if (ms->debug) {
                    printf("Orthogonality failed: b_hat[%d] · b_hat[%d] = %f\n", i, j, dot_prod);
                }
                return 0;
            }
        }
    }
    
    // Check GSO formula
    for (int i = 0; i < ms->n_basis; i++) {
        double* reconstructed = (double*)malloc((ms->n + 1) * sizeof(double));
        if (!reconstructed) return 0;
        
        // Start with b_hat[i]
        memcpy(reconstructed, ms->b_hat[i], (ms->n + 1) * sizeof(double));
        
        // Add projections
        for (int j = 0; j < i; j++) {
            for (int k = 0; k < ms->n + 1; k++) {
                reconstructed[k] += ms->mu[i][j] * ms->b_hat[j][k];
            }
        }
        
        // Compare with original basis vector
        int matches = 1;
        for (int k = 0; k < ms->n + 1; k++) {
            if (fabs(reconstructed[k] - (double)ms->basis[i][k]) > tol) {
                matches = 0;
                break;
            }
        }
        
        free(reconstructed);
        if (!matches) {
            if (ms->debug) {
                printf("GSO formula failed for basis[%d]\n", i);
            }
            return 0;
        }
    }
    
    if (ms->debug) {
        printf("GSO verification passed\n");
    }
    return 1;
}

int ms_verify_dual(MarketSplit* ms, double tol) {
    // Check dual property: <b_bar[i], basis[j]> = δ_ij
    for (int i = 0; i < ms->n_basis; i++) {
        for (int j = 0; j < ms->n_basis; j++) {
            double dot_prod = 0.0;
            for (int k = 0; k < ms->n + 1; k++) {
                dot_prod += ms->b_bar[i][k] * (double)ms->basis[j][k];
            }
            
            double expected = (i == j) ? 1.0 : 0.0;
            if (fabs(dot_prod - expected) > tol) {
                if (ms->debug) {
                    printf("Dual property failed: <b_bar[%d], basis[%d]> = %f, expected %f\n", 
                           i, j, dot_prod, expected);
                }
                return 0;
            }
        }
    }
    
    if (ms->debug) {
        printf("Dual verification passed\n");
    }
    return 1;
}

// Backtracking enumeration function
int backtrack_enumerate(MarketSplit* ms, SolutionSet* sols, int idx, int* u_values, 
                       double* prev_w, double prev_w_norm_sq, double c, 
                       double* u_global_bounds, struct timeval start_time) {
    ms->backtrack_loops++;
    
    if (idx == -1) {
        ms->dive_loops++;
        
        // Compute v = sum(u_i * basis_i)
        double* v = (double*)calloc(ms->n + 1, sizeof(double));
        if (!v) return 0;
        
        for (int i = 0; i < ms->n_basis; i++) {
            for (int j = 0; j < ms->n + 1; j++) {
                v[j] += u_values[i] * (double)ms->basis[i][j];
            }
        }
        
        // Check constraints: v[-1] = rmax and -rmax <= v[i] <= rmax
        int valid = (fabs(v[ms->n] - ms->rmax) < 1e-10);
        if (valid) {
            for (int i = 0; i < ms->n; i++) {
                if (v[i] < -ms->rmax - 1e-10 || v[i] > ms->rmax + 1e-10) {
                    valid = 0;
                    break;
                }
            }
        }
        
        if (valid) {
            // Recover solution: x_i = (v_{i-1} + rmax) / (2 * c_{i-1})
            int* x = (int*)malloc(ms->n * sizeof(int));
            if (x) {
                int solution_valid = 1;
                for (int i = 0; i < ms->n; i++) {
                    x[i] = (int)round((v[i] + ms->rmax) / (2.0 * ms->c[i]));
                }
                
                // Verify solution
                for (int i = 0; i < ms->m; i++) {
                    int sum = 0;
                    for (int j = 0; j < ms->n; j++) {
                        sum += ms->A[i][j] * x[j];
                    }
                    if (sum != ms->d[i]) {
                        solution_valid = 0;
                        break;
                    }
                }
                
                if (solution_valid) {
                    // Track first solution time
                    if (ms->first_solution_time == 0.0) {
                        struct timeval current_time;
                        gettimeofday(&current_time, NULL);
                        ms->first_solution_time = get_time_diff(start_time, current_time);
                        ms->first_sol_bt_loops = ms->backtrack_loops;
                    }
                    
                    solution_set_add(sols, x, ms->n);
                    
                    if (ms->max_sols > 0 && sols->count >= ms->max_sols) {
                        free(x);
                        free(v);
                        return 1; // Stop search
                    }
                }
                free(x);
            }
        }
        free(v);
        return 0;
    }
    
    // First pruning: check ||w^(idx+1)||_2^2
    if (prev_w_norm_sq > c + 1e-10) {
        ms->dive_loops++;
        ms->first_pruning_effect_count++;
        return 0;
    }
    
    // Compute sum_{i=idx+1}^{n_basis-1} u_i * mu_{i,idx}
    double mu_sum = 0.0;
    for (int j = idx + 1; j < ms->n_basis; j++) {
        mu_sum += u_values[j] * ms->mu[j][idx];
    }
    
    // First pruning bounds
    double bound_sq = (c - prev_w_norm_sq) / ms->b_hat_norms_sq[idx];
    double bound = sqrt(fmax(0, bound_sq));
    
    int u_min_pruning1 = (int)floor(-bound - mu_sum);
    int u_max_pruning1 = (int)ceil(bound - mu_sum);
    
    // Second pruning: global bounds
    int u_min_pruning2 = (int)floor(-u_global_bounds[idx]);
    int u_max_pruning2 = (int)ceil(u_global_bounds[idx]);
    
    // Take intersection
    int u_min = (u_min_pruning1 > u_min_pruning2) ? u_min_pruning1 : u_min_pruning2;
    int u_max = (u_max_pruning1 < u_max_pruning2) ? u_max_pruning1 : u_max_pruning2;
    
    // Track second pruning effectiveness
    int original_range = u_max_pruning1 - u_min_pruning1 + 1;
    int final_range = (u_max >= u_min) ? u_max - u_min + 1 : 0;
    if (final_range < original_range) {
        ms->second_pruning_effect_count++;
    }
    
    // Third pruning: iterate through range
    for (int u_val = u_min; u_val <= u_max; u_val++) {
        u_values[idx] = u_val;
        
        // Compute w^(idx) = (u_val + mu_sum) * b_hat[idx] + prev_w
        double coeff = u_val + mu_sum;
        double* curr_w = (double*)malloc((ms->n + 1) * sizeof(double));
        if (!curr_w) continue;
        
        for (int k = 0; k < ms->n + 1; k++) {
            curr_w[k] = coeff * ms->b_hat[idx][k] + prev_w[k];
        }
        
        // Third pruning: check ||w||_2^2 <= rmax * ||w||_1
        double w_norm_sq = coeff * coeff * ms->b_hat_norms_sq[idx] + prev_w_norm_sq;
        double w_norm_l1 = vector_norm_l1(curr_w, ms->n + 1);
        
        if (w_norm_sq > ms->rmax * w_norm_l1 + 1e-10) {
            if (coeff > 0) {
                // Skip remaining iterations
                ms->third_pruning_effect_count += (u_max - u_val);
                free(curr_w);
                break;
            }
            ms->third_pruning_effect_count++;
            free(curr_w);
            continue;
        }
        
        if (backtrack_enumerate(ms, sols, idx - 1, u_values, curr_w, w_norm_sq, 
                               c, u_global_bounds, start_time)) {
            free(curr_w);
            return 1;
        }
        
        free(curr_w);
    }
    
    return 0;
}

SolutionSet* ms_enumerate(MarketSplit* ms) {
    struct timeval start_time;
    gettimeofday(&start_time, NULL);
    
    SolutionSet* sols = solution_set_create();
    if (!sols) return NULL;
    
    double c = (ms->n + 1) * ms->rmax * ms->rmax;
    
    // Second pruning: compute global bounds
    double sqrt_c = sqrt(c);
    double* u_global_bounds = (double*)malloc(ms->n_basis * sizeof(double));
    if (!u_global_bounds) {
        solution_set_destroy(sols);
        return NULL;
    }
    
    for (int i = 0; i < ms->n_basis; i++) {
        double u_bound_l2 = ms->b_bar_norms_l2[i] * sqrt_c;
        double u_bound_l1 = ms->b_bar_norms_l1[i] * ms->rmax;
        u_global_bounds[i] = (u_bound_l2 < u_bound_l1) ? u_bound_l2 : u_bound_l1;
    }
    
    // Start backtracking
    int* u_values = (int*)calloc(ms->n_basis, sizeof(int));
    double* initial_w = (double*)calloc(ms->n + 1, sizeof(double));
    
    if (u_values && initial_w) {
        backtrack_enumerate(ms, sols, ms->n_basis - 1, u_values, initial_w, 0.0, 
                           c, u_global_bounds, start_time);
    }
    
    free(u_global_bounds);
    free(u_values);
    free(initial_w);
    
    return sols;
}

MSResult* ms_run(int** A, int* d, int m, int n, const char* instance_id, 
                 int* opt_sol, int opt_n, int max_sols, int debug) {
    MSResult* result = (MSResult*)malloc(sizeof(MSResult));
    if (!result) return NULL;
    
    memset(result, 0, sizeof(MSResult));
    strcpy(result->id, instance_id);
    
    struct timeval start_time, init_end;
    gettimeofday(&start_time, NULL);
    
    MarketSplit* ms = ms_create(A, d, m, n, NULL, max_sols, debug);
    gettimeofday(&init_end, NULL);
    
    if (!ms) {
        strcpy(result->error, "Failed to create MarketSplit instance");
        return result;
    }
    
    result->init_time = get_time_diff(start_time, init_end);
    
    SolutionSet* sols = ms_enumerate(ms);
    
    struct timeval end_time;
    gettimeofday(&end_time, NULL);
    
    if (sols) {
        result->solutions_count = sols->count;
        result->solutions = (Solution*)malloc(sols->count * sizeof(Solution));
        
        if (result->solutions) {
            for (int i = 0; i < sols->count; i++) {
                result->solutions[i].n = sols->solutions[i].n;
                result->solutions[i].solution = (int*)malloc(sols->solutions[i].n * sizeof(int));
                if (result->solutions[i].solution) {
                    memcpy(result->solutions[i].solution, sols->solutions[i].solution,
                           sols->solutions[i].n * sizeof(int));
                }
            }
        }
        
        // Check if optimal solution was found
        if (opt_sol && opt_n > 0) {
            for (int i = 0; i < sols->count; i++) {
                if (sols->solutions[i].n == opt_n && 
                    solutions_equal(sols->solutions[i].solution, opt_sol, opt_n)) {
                    result->optimal_found = 1;
                    break;
                }
            }
        }
        
        solution_set_destroy(sols);
    }
    
    // Copy statistics
    result->backtrack_loops = ms->backtrack_loops;
    result->first_sol_bt_loops = ms->first_sol_bt_loops;
    result->dive_loops = ms->dive_loops;
    result->first_pruning_effect_count = ms->first_pruning_effect_count;
    result->second_pruning_effect_count = ms->second_pruning_effect_count;
    result->third_pruning_effect_count = ms->third_pruning_effect_count;
    result->first_solution_time = ms->first_solution_time;
    result->solve_time = get_time_diff(start_time, end_time);
    result->success = 1;
    
    ms_destroy(ms);
    
    return result;
}