#ifndef MS_SOLVE_H
#define MS_SOLVE_H

#include <vector>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>
#include <chrono>
#include <fplll/fplll.h>

using namespace Eigen;
using namespace std;
using namespace fplll;
using namespace std::chrono;

struct Instance {
    string id;
    int m, n;
    MatrixXi A;
    VectorXi d;
};

class MSData {
private:
    vector<Instance> data;
    unordered_map<string, int> by_id;
    unordered_map<string, VectorXi> solutions;
    
    void load_instances(const string& path);
    void load_solutions(const string& path);

public:
    MSData(const string& data_path, const string& sol_path);
    const Instance* get(const string& id) const;
    vector<const Instance*> get_by_size(int m, int n) const;
    vector<const Instance*> get_by_m(int m) const;
    const VectorXi* get_solution(const string& id) const;
    size_t size() const { return data.size(); }
};

class MarketSplit {
private:
    MatrixXi A;
    VectorXi d, r, c;
    int m, n, n_basis, rmax;
    
    // Lattice data
    MatrixXi L;
    MatrixXi basis;
    MatrixXd b_hat, b_bar, mu;
    VectorXd b_hat_norms_sq, b_bar_norms_l2, b_bar_norms_l1;
    MatrixXd coords;
    
    // Statistics
    long long backtrack_loops, dive_loops, first_sol_bt_loops;
    long long first_pruning_effect_count, second_pruning_effect_count, third_pruning_effect_count;
    double first_solution_time;
    high_resolution_clock::time_point start_time;
    
    int max_sols;
    bool debug;
    
    // Helper methods
    int compute_lcm(const vector<int>& nums);
    void get_extended_matrix();
    void get_reduced_basis();
    void get_gso();
    void compute_dual_norms();
    void get_coordinates();
    
public:
    MarketSplit(const MatrixXi& A, const VectorXi& d, const VectorXi& r = VectorXi(), int max_sols = -1, bool debug = false);
    vector<VectorXi> enumerate();
    
    bool verify_gso(double tol = 1e-10) const;
    bool verify_dual(double tol = 1e-10) const;
    // Getters for statistics
    long long get_backtrack_loops() const { return backtrack_loops; }
    long long get_dive_loops() const { return dive_loops; }
    long long get_first_sol_bt_loops() const { return first_sol_bt_loops; }
    long long get_first_pruning_effect_count() const { return first_pruning_effect_count; }
    long long get_second_pruning_effect_count() const { return second_pruning_effect_count; }
    long long get_third_pruning_effect_count() const { return third_pruning_effect_count; }
    double get_first_solution_time() const { return first_solution_time; }
};

struct SolveResult {
    string id;
    int solutions_count;
    vector<VectorXi> solutions;
    bool optimal_found;
    long long backtrack_loops, dive_loops, first_sol_bt_loops;
    long long first_pruning_effect_count, second_pruning_effect_count, third_pruning_effect_count;
    double solve_time, first_solution_time, init_time;
    bool success;
    string error;
};

struct StackFrame {
    int idx;
    int u_min, u_max, u_current;
    double mu_sum, prev_w_norm_sq;
    VectorXd w;  // Store current w vector
    
    StackFrame(int i, int min_u, int max_u, int curr_u, double ms, double w_norm, const VectorXd& w_vec)
        : idx(i), u_min(min_u), u_max(max_u), u_current(curr_u), mu_sum(ms), prev_w_norm_sq(w_norm), w(w_vec) {}
};


SolveResult ms_run(const MatrixXi& A, const VectorXi& d, const string& instance_id, const VectorXi* opt_sol = nullptr, int max_sols = -1, bool debug = false);

#endif