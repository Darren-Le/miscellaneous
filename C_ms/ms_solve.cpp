#include "ms_solve.h"
#include <fstream>
#include <sstream>
#include <experimental/filesystem>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <iostream>

using namespace std::chrono;

namespace fs = std::experimental::filesystem;

// MSData Implementation
MSData::MSData(const string& data_path, const string& sol_path) {
    load_instances(data_path);
    load_solutions(sol_path);
}

void MSData::load_instances(const string& path) {
    for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.path().extension() == ".dat") {
            ifstream file(entry.path());
            if (!file.is_open()) continue;
            
            string line;
            vector<string> lines;
            
            while (getline(file, line)) {
                if (!line.empty() && line[0] != '#') {
                    lines.push_back(line);
                }
            }
            
            if (lines.empty()) continue;
            
            istringstream iss(lines[0]);
            int m, n;
            iss >> m >> n;
            
            MatrixXi A(m, n);
            VectorXi d(m);
            
            for (int i = 0; i < m; i++) {
                istringstream row_iss(lines[i + 1]);
                for (int j = 0; j < n; j++) {
                    row_iss >> A(i, j);
                }
                row_iss >> d(i);
            }
            
            string id = entry.path().stem().string();
            data.push_back({id, m, n, A, d});
            by_id[id] = data.size() - 1;
        }
    }
}

void MSData::load_solutions(const string& path) {
    for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.path().filename().string().find(".opt.sol") != string::npos) {
            ifstream file(entry.path());
            if (!file.is_open()) continue;
            
            unordered_map<int, int> x_vars;
            string line;
            
            while (getline(file, line)) {
                if (!line.empty() && line[0] != '#' && line.substr(0, 2) == "x#") {
                    istringstream iss(line);
                    string var_name;
                    int var_val;
                    iss >> var_name >> var_val;
                    
                    int var_num = stoi(var_name.substr(2));
                    x_vars[var_num] = var_val;
                }
            }
            
            if (!x_vars.empty()) {
                int n = 0;
                for (const auto& pair : x_vars) {
                    n = max(n, pair.first);
                }
                VectorXi x = VectorXi::Zero(n);
                
                for (const auto& pair : x_vars) {
                    if (pair.first >= 1 && pair.first <= n) {
                        x(pair.first - 1) = pair.second;
                    }
                }
                
                string id = entry.path().stem().string();
                size_t pos = id.find(".opt");
                if (pos != string::npos) {
                    id = id.substr(0, pos);
                }
                solutions[id] = x;
            }
        }
    }
}

const Instance* MSData::get(const string& id) const {
    auto it = by_id.find(id);
    return (it != by_id.end()) ? &data[it->second] : nullptr;
}

vector<const Instance*> MSData::get_by_size(int m, int n) const {
    vector<const Instance*> result;
    for (const auto& inst : data) {
        if (inst.m == m && inst.n == n) {
            result.push_back(&inst);
        }
    }
    return result;
}

vector<const Instance*> MSData::get_by_m(int m) const {
    vector<const Instance*> result;
    for (const auto& inst : data) {
        if (inst.m == m) {
            result.push_back(&inst);
        }
    }
    return result;
}

const VectorXi* MSData::get_solution(const string& id) const {
    auto it = solutions.find(id);
    return (it != solutions.end()) ? &it->second : nullptr;
}

// MarketSplit Implementation
MarketSplit::MarketSplit(const MatrixXi& A, const VectorXi& d, const VectorXi& r, int max_sols, bool debug)
    : A(A), d(d), m(A.rows()), n(A.cols()), n_basis(n - m + 1), max_sols(max_sols), debug(debug),
      backtrack_loops(0), dive_loops(0), first_sol_bt_loops(0),
      first_pruning_count(0), second_pruning_count(0), third_pruning_count(0),
      first_solution_time(0.0) {
    
    start_time = high_resolution_clock::now();

    if (r.size() == 0) {
        this->r = VectorXi::Ones(n);
    } else {
        this->r = r;
    }
    
    get_extended_matrix();
    get_reduced_basis();
    get_gso();
    compute_dual_norms();

    verify_gso();
    verify_dual();
}

int MarketSplit::compute_lcm(const vector<int>& nums) {
    if (nums.empty()) return 1;
    
    int result = nums[0];
    for (size_t i = 1; i < nums.size(); i++) {
        result = (result / std::gcd(result, nums[i])) * nums[i];
    }
    return abs(result);
}

void MarketSplit::get_extended_matrix() {
    // Compute rmax and c
    vector<int> r_vec(r.data(), r.data() + r.size());
    rmax = compute_lcm(r_vec);
    
    c.resize(n);
    for (int i = 0; i < n; i++) {
        c(i) = rmax / r(i);
    }
    
    // Create extended matrix L
    int pows = to_string(A.cwiseAbs().maxCoeff()).length() + to_string(d.cwiseAbs().maxCoeff()).length() + 2;
    int N = static_cast<int>(pow(10, pows));
    
    L.resize(m + n + 1, n + 1);
    L.setZero();  // Explicit zero initialization
    
    // First column
    L.block(0, 0, m, 1) = -N * d;
    L.block(m, 0, n, 1) = VectorXi::Constant(n, -rmax);
    L(m + n, 0) = rmax;
    
    // Top-right block
    L.block(0, 1, m, n) = N * A;
    
    // Middle diagonal block
    for (int i = 0; i < n; i++) {
        L(m + i, 1 + i) = 2 * c(i);
    }

    std::cout << "L.rows(): " << L.rows() << std::endl;
    std::cout << "L.cols(): " << L.cols() << std::endl;
    std::cout << "m: " << m << std::endl; 
    std::cout << "n: " << n << std::endl;
    std::cout << "rmax: " << rmax << std::endl;
    std::cout << "L matrix (first 10 rows, first 10 cols):" << std::endl;
    for (int i = 0; i < min(10, (int)L.rows()); i++) {
        for (int j = 0; j < min(10, (int)L.cols()); j++) {
            std::cout << L(i,j) << " ";
        }
        std::cout << std::endl;
}
}

void MarketSplit::get_reduced_basis() {
    // Convert to fplll format
    int ext_m = L.rows();
    int ext_n = L.cols();
    
    ZZ_mat<mpz_t> fplll_mat(ext_n, ext_m);
    for (int i = 0; i < ext_n; i++) {
        for (int j = 0; j < ext_m; j++) {
            fplll_mat[i][j] = L(j, i);
        }
    }
    // Debug: Print original matrix
    std::cout << "Before BKZ reduction:" << std::endl;
    for (int i = 0; i < min(5, ext_n); i++) {
        for (int j = 0; j < ext_m; j++) {
            std::cout << fplll_mat[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "..." << std::endl;
    
    // Apply BKZ reduction
    int block_size = min(30, ext_n / 2);
    int status = bkz_reduction(fplll_mat, block_size, BKZ_VERBOSE, FT_DEFAULT, 0);

    // Debug: Print the matrix after BKZ
    std::cout << "After BKZ reduction (" << ext_n << "x" << ext_m << "):" << std::endl;
    for (int i = 0; i < min(10, ext_n); i++) {  // Print first 10 rows only
        for (int j = 0; j < ext_m; j++) {
            std::cout << fplll_mat[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "..." << std::endl;

    // Extract null space basis
    vector<VectorXi> basis_vectors;
    
    for (int j = 0; j < ext_n; j++) {
        bool is_null = true;
        for (int i = 0; i < m; i++) {
            if (fplll_mat[j][i] != 0) {
                is_null = false;
                break;
            }
        }
        
        if (is_null) {
            VectorXi col(n + 1);
            for (int i = 0; i < n + 1; i++) {
                col(i) = fplll_mat[j][m + i].get_si();
            }
            basis_vectors.push_back(col);
        }
    }
    
    // Convert to matrix
    basis.resize(n_basis, n + 1);
    std::cout << "n_basis: " << n_basis << std::endl;
    std::cout << "basis_vectors.size(): " << basis_vectors.size() << std::endl;
    std::cout << "basis.rows(): " << basis.rows() << std::endl;
    std::cout << "basis.cols(): " << basis.cols() << std::endl;
    for (int i = 0; i < n_basis; i++) {
        basis.row(i) = basis_vectors[i];
    }
}

void MarketSplit::get_gso() {
    b_hat.resize(n_basis, n + 1);
    mu.resize(n_basis, n_basis);
    b_hat_norms_sq.resize(n_basis);
    
    mu.setZero();
    
    for (int i = 0; i < n_basis; i++) {
        b_hat.row(i) = basis.row(i).cast<double>();
        
        for (int j = 0; j < i; j++) {
            mu(i, j) = basis.row(i).cast<double>().dot(b_hat.row(j)) / b_hat_norms_sq(j);
            b_hat.row(i) -= mu(i, j) * b_hat.row(j);
        }
        
        mu(i, i) = 1.0;
        b_hat_norms_sq(i) = b_hat.row(i).squaredNorm();
    }
}

void MarketSplit::compute_dual_norms() {
    MatrixXd B = basis.cast<double>().transpose();
    MatrixXd B_T = basis.cast<double>();
    MatrixXd gram = B_T * B;
    MatrixXd gram_inv = gram.inverse();
    b_bar = (B * gram_inv).transpose();
    
    b_bar_norms_l2.resize(n_basis);
    b_bar_norms_l1.resize(n_basis);
    
    for (int i = 0; i < n_basis; i++) {
        b_bar_norms_l2(i) = b_bar.row(i).norm();
        b_bar_norms_l1(i) = b_bar.row(i).lpNorm<1>();
    }
}

bool MarketSplit::backtrack(int idx, vector<int>& u_values, const VectorXd& prev_w, double prev_w_norm_sq, 
                           vector<VectorXi>& solutions, double c, const VectorXd& u_global_bounds) {
    backtrack_loops++;
    
    if (idx == -1) {
        dive_loops++;
        
        VectorXd v = VectorXd::Zero(n + 1);
        for (int i = 0; i < n_basis; i++) {
            v += u_values[i] * basis.row(i).cast<double>();
        }
        
        if (abs(v(n) - rmax) < 1e-10) {
            bool valid = true;
            for (int i = 0; i < n; i++) {
                if (v(i) < -rmax - 1e-10 || v(i) > rmax + 1e-10) {
                    valid = false;
                    break;
                }
            }
            
            if (valid) {
                VectorXi x(n);
                for (int i = 0; i < n; i++) {
                    x(i) = static_cast<int>(round((v(i) + rmax) / (2 * this->c(i))));
                }
                
                if ((A * x - d).norm() < 1e-10) {
                    if (first_solution_time == 0.0) {
                        auto current_time = high_resolution_clock::now();
                        first_solution_time = duration<double>(current_time - start_time).count();
                        first_sol_bt_loops = backtrack_loops;
                    }
                    solutions.push_back(x);
                    if (max_sols > 0 && static_cast<int>(solutions.size()) >= max_sols) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    
    // First pruning condition
    if (prev_w_norm_sq > c + 1e-10) {
        dive_loops++;
        first_pruning_count++;
        return false;
    }
    
    // Compute mu_sum
    double mu_sum = 0.0;
    for (int j = idx + 1; j < n_basis; j++) {
        mu_sum += u_values[j] * mu(j, idx);
    }
    
    // First pruning bounds
    double bound_sq = (c - prev_w_norm_sq) / b_hat_norms_sq(idx);
    double bound = sqrt(max(0.0, bound_sq));
    
    int u_min_pruning1 = static_cast<int>(floor(-bound - mu_sum));
    int u_max_pruning1 = static_cast<int>(ceil(bound - mu_sum));
    
    // Second pruning bounds
    int u_min_pruning2 = static_cast<int>(floor(-u_global_bounds(idx)));
    int u_max_pruning2 = static_cast<int>(ceil(u_global_bounds(idx)));
    
    int u_min = max(u_min_pruning1, u_min_pruning2);
    int u_max = min(u_max_pruning1, u_max_pruning2);
    
    int original_range = u_max_pruning1 - u_min_pruning1 + 1;
    int final_range = max(0, u_max - u_min + 1);
    
    if (final_range < original_range) {
        second_pruning_count++;
    }
    
    // Third pruning strategy
    for (int u_val = u_min; u_val <= u_max; u_val++) {
        u_values[idx] = u_val;
        
        double coeff = u_val + mu_sum;
        VectorXd curr_w = coeff * b_hat.row(idx).transpose() + prev_w;
        
        double w_norm_sq = coeff * coeff * b_hat_norms_sq(idx) + prev_w_norm_sq;
        double w_norm_l1 = curr_w.lpNorm<1>();
        
        if (w_norm_sq > rmax * w_norm_l1 + 1e-10) {
            if (coeff > 0) {
                third_pruning_count += (u_max - u_val);
                break;
            }
            third_pruning_count++;
            continue;
        }
        
        if (backtrack(idx - 1, u_values, curr_w, w_norm_sq, solutions, c, u_global_bounds)) {
            return true;
        }
    }
    
    return false;
}

bool MarketSplit::verify_gso(double tol) const {
    // Check orthogonality
    for (int i = 0; i < n_basis; i++) {
        for (int j = i + 1; j < n_basis; j++) {
            double dot_product = b_hat.row(i).dot(b_hat.row(j));
            if (abs(dot_product) > tol) {
                if (debug) cout << "Orthogonality failed: b_hat[" << i << "] · b_hat[" << j << "] = " << dot_product << endl;
                return false;
            }
        }
    }
    for (int i = 0; i < n_basis; i++) {
        VectorXd reconstructed = b_hat.row(i);
        for (int j = 0; j < i; j++) {
            reconstructed += mu(i, j) * b_hat.row(j);
        }
        
        if (!basis.row(i).cast<double>().isApprox(reconstructed, tol)) {
            if (debug) cout << "GSO formula failed for basis[" << i << "]" << endl;
            return false;
        }
    }
    return true;
}

bool MarketSplit::verify_dual(double tol) const {
    for (int i = 0; i < n_basis; i++) {
        for (int j = 0; j < n_basis; j++) {
            double dot_product = b_bar.row(i).dot(basis.row(j).cast<double>());
            double expected = (i == j) ? 1.0 : 0.0;
            if (abs(dot_product - expected) > tol) {
                if (debug) cout << "Dual property failed" << endl;
                return false;
            }
        }
    }
    return true;
}


vector<VectorXi> MarketSplit::enumerate() {
    vector<VectorXi> solutions;
    double c = (n + 1) * rmax * rmax;
    
    // Global bounds
    double sqrt_c = sqrt(c);
    VectorXd u_global_bounds(n_basis);
    for (int i = 0; i < n_basis; i++) {
        u_global_bounds(i) = min(b_bar_norms_l2(i) * sqrt_c, b_bar_norms_l1(i) * rmax);
    }
    
    vector<int> u_values(n_basis, 0);
    VectorXd initial_w = VectorXd::Zero(n + 1);
    
    backtrack(n_basis - 1, u_values, initial_w, 0.0, solutions, c, u_global_bounds);
    
    return solutions;
}

SolveResult ms_run(const MatrixXi& A, const VectorXi& d, const string& instance_id, const VectorXi* opt_sol, int max_sols, bool debug) {
    auto start_time = high_resolution_clock::now();
    
    try {
        auto init_start = high_resolution_clock::now();
        MarketSplit ms(A, d, VectorXi(), max_sols, debug);
        auto init_end = high_resolution_clock::now();
        double init_time = duration<double>(init_end - init_start).count();
        
        vector<VectorXi> solutions = ms.enumerate();
        auto end_time = high_resolution_clock::now();
        double solve_time = duration<double>(end_time - start_time).count();
        
        bool found_opt = false;
        if (opt_sol != nullptr) {
            for (const auto& sol : solutions) {
                if (sol == *opt_sol) {
                    found_opt = true;
                    break;
                }
            }
        }
        
        return {
            instance_id,
            static_cast<int>(solutions.size()),
            solutions,
            found_opt,
            ms.get_backtrack_loops(),
            ms.get_dive_loops(),
            ms.get_first_sol_bt_loops(),
            ms.get_first_pruning_count(),
            ms.get_second_pruning_count(),
            ms.get_third_pruning_count(),
            solve_time,
            ms.get_first_solution_time(),
            init_time,
            true,
            ""
        };
    } catch (const exception& e) {
        return {
            instance_id, 0, {}, false, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, false, e.what()
        };
    }
}