#include "ms_solve.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <chrono>

using namespace std;
using namespace std::chrono;

void print_and_log(const string& text, ofstream& file) {
    cout << text << endl;
    file << text << endl;
}

int main(int argc, char* argv[]) {
    // Default parameters
    string data_path = "ms_instance/01-marketsplit/instances";
    string sol_path = "ms_instance/01-marketsplit/solutions";
    int max_sols = -1;
    bool debug = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--data_path" && i + 1 < argc) {
            data_path = argv[++i];
        } else if (arg == "--sol_path" && i + 1 < argc) {
            sol_path = argv[++i];
        } else if (arg == "--max_sols" && i + 1 < argc) {
            max_sols = stoi(argv[++i]);
        } else if (arg == "--debug") {
            debug = true;
        } else if (arg == "--help") {
            cout << "Usage: " << argv[0] << " [options]\n"
                 << "Options:\n"
                 << "  --data_path PATH   Path to instance data (default: ms_instance/01-marketsplit/instances)\n"
                 << "  --sol_path PATH    Path to solution data (default: ms_instance/01-marketsplit/solutions)\n" 
                 << "  --max_sols N       Maximum solutions to find (-1 for all, default: -1)\n"
                 << "  --debug            Enable debug output\n"
                 << "  --help             Show this help message\n";
            return 0;
        }
    }
    
    try {
        MSData ms_data(data_path, sol_path);
        cout << "Loaded " << ms_data.size() << " instances" << endl;
        
        vector<int> test_m_values = {3, 4, 5, 6, 7};
        vector<SolveResult> all_results;
        
        for (int m : test_m_values) {
            auto m_instances = ms_data.get_by_m(m);
            cout << "Testing " << m_instances.size() << " instances with m = " << m << endl;
            
            for (const auto* inst : m_instances) {
                const VectorXi* opt_sol = ms_data.get_solution(inst->id);
                auto result = ms_run(inst->A, inst->d, inst->id, opt_sol, max_sols, debug);
                all_results.push_back(result);
                
                string status = result.success ? "✓" : "✗";
                string opt_status = result.optimal_found ? "✓" : "✗";
                
                cout << fixed << setprecision(4)
                     << status << " " << result.id << ": " << result.solutions_count << " solutions, "
                     << "optimal: " << opt_status << ", bt_loops: " << result.backtrack_loops << ", "
                     << "dive_loops: " << result.dive_loops << ", 1st_prune: " << result.first_pruning_count << ", "
                     << "2nd_prune: " << result.second_pruning_count << ", 3rd_prune: " << result.third_pruning_count << ", "
                     << "time: " << result.solve_time << "s, "
                     << "1st_sol: " << result.first_solution_time << "s, 1st_bt: " << result.first_sol_bt_loops << ", "
                     << "init: " << result.init_time << "s" << endl;
            }
            cout << endl;
        }
        
        // Generate log file
        auto now = system_clock::now();
        auto time_t_now = system_clock::to_time_t(now);
        auto tm_now = *localtime(&time_t_now);
        
        stringstream ss;
        ss << "res_";
        for (size_t i = 0; i < test_m_values.size(); i++) {
            if (i > 0) ss << "_";
            ss << test_m_values[i];
        }
        ss << "_" << tm_now.tm_mday << "d" << tm_now.tm_hour << "h" 
           << tm_now.tm_min << "m" << tm_now.tm_sec << "s.log";
        
        string log_filename = ss.str();
        ofstream log_file(log_filename);
        
        print_and_log(string(136, '='), log_file);
        print_and_log("RESULTS", log_file);
        print_and_log(string(136, '='), log_file);
        print_and_log("", log_file);
        
        // Table header
        stringstream header;
        header << left << setw(15) << "ID" 
               << setw(8) << "Size"
               << setw(8) << "Status"
               << setw(8) << "Optimal"
               << setw(10) << "Time(s)"
               << setw(10) << "1st_Sol(s)"
               << setw(10) << "1st_Prune"
               << setw(10) << "2nd_Prune"
               << setw(10) << "3rd_Prune"
               << setw(10) << "Init(s)"
               << setw(10) << "Solutions"
               << setw(8) << "1st_BT"
               << setw(12) << "BT_Loops"
               << setw(12) << "Dive_Loops";
        
        print_and_log(header.str(), log_file);
        print_and_log(string(136, '-'), log_file);
        
        for (const auto& result : all_results) {
            stringstream row;
            // Find the instance to get actual size
            const Instance* inst = ms_data.get(result.id);
            string size = inst ? ("(" + to_string(inst->m) + "," + to_string(inst->n) + ")") : "(?,?)";
            string status = result.success ? "SUCCESS" : "FAILED";
            string optimal = result.optimal_found ? "✓" : "✗";
            
            row << left << setw(15) << result.id
                << setw(8) << size
                << setw(8) << status
                << setw(8) << optimal
                << fixed << setprecision(4)
                << setw(10) << result.solve_time
                << setw(10) << result.first_solution_time
                << setw(10) << result.first_pruning_count
                << setw(10) << result.second_pruning_count
                << setw(10) << result.third_pruning_count
                << setw(10) << result.init_time
                << setw(10) << result.solutions_count
                << setw(8) << result.first_sol_bt_loops
                << setw(12) << result.backtrack_loops
                << setw(12) << result.dive_loops;
            
            print_and_log(row.str(), log_file);
        }
        
        print_and_log("", log_file);
        print_and_log(string(136, '='), log_file);
        log_file.close();
        
        cout << "Results saved to " << log_filename << endl;
        
        // Save solutions
        string sol_filename = log_filename;
        sol_filename.replace(sol_filename.find(".log"), 4, ".sol");
        
        ofstream sol_file(sol_filename);
        for (const auto& result : all_results) {
            if (result.success && !result.solutions.empty()) {
                sol_file << "=======" << result.id << "=============" << endl;
                for (size_t i = 0; i < result.solutions.size(); i++) {
                    sol_file << "-----------" << (i + 1) << "-th solution------------" << endl;
                    for (int j = 0; j < result.solutions[i].size(); j++) {
                        if (j > 0) sol_file << " ";
                        sol_file << result.solutions[i](j);
                    }
                    sol_file << endl;
                }
                sol_file << endl;
            }
        }
        sol_file.close();
        
        cout << "Solutions saved to " << sol_filename << endl;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}