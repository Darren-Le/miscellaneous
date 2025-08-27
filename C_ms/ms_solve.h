#ifndef MS_SOLVE_H
#define MS_SOLVE_H

#include <vector>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>
#include <fplll.h>

using namespace Eigen;
using namespace std;
using namespace fplll;

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
    
    // Statistics
    long long backtrack_loops, dive_loops, first_sol_bt_loops;
    long long first_pruning_count, second_pruning_count, third_pruning_count;
    double first_solution_time;
    
    int max_sols;
    bool debug;
    
    // Helper methods
    int compute_lcm(const vector<int>& nums);
    void get_extended_matrix();
    void get_reduced_basis();
    void get_gso();
    void compute_dual_norms();
    bool backtrack(int idx, vector<int>& u_values, const VectorXd& prev_w, double prev_w_norm_sq, vector<VectorXi>& solutions, double c, const VectorXd& u_global_bounds);

public:
    MarketSplit(const MatrixXi& A, const VectorXi& d, const VectorXi& r = VectorXi(), int max_sols = -1, bool debug = false);
    vector<VectorXi> enumerate();
    
    // Getters for statistics
    long long get_backtrack_loops() const { return backtrack_loops; }
    long long get_dive_loops() const { return dive_loops; }
    long long get_first_sol_bt_loops() const { return first_sol_bt_loops; }
    long long get_first_pruning_count() const { return first_pruning_count; }
    long long get_second_pruning_count() const { return second_pruning_count; }
    long long get_third_pruning_count() const { return third_pruning_count; }
    double get_first_solution_time() const { return first_solution_time; }
};

struct SolveResult {
    string id;
    int solutions_count;
    vector<VectorXi> solutions;
    bool optimal_found;
    long long backtrack_loops, dive_loops, first_sol_bt_loops;
    long long first_pruning_count, second_pruning_count, third_pruning_count;
    double solve_time, first_solution_time, init_time;
    bool success;
    string error;
};

SolveResult ms_run(const MatrixXi& A, const VectorXi& d, const string& instance_id, const VectorXi* opt_sol = nullptr, int max_sols = -1, bool debug = false);

#endif