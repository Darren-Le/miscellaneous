#ifdef HIGHS_HAVE_LATTICE_ENUM
#include "ms_solve.h"
#endif

// Inside main() function, add this test:
#ifdef HIGHS_HAVE_LATTICE_ENUM
    std::cout << "=== Testing cpp_ms integration ===" << std::endl;
    
    // Create a simple 2x2 test matrix A and vector d
    MatrixXi A(2, 2);
    A << 1, 0,
         0, 1;
    
    VectorXi d(2);
    d << 1, 1;
    
    std::cout << "Calling ms_run with test data..." << std::endl;
    SolveResult result = ms_run(A, d, "test_instance");
    
    std::cout << "Result: " << result.solutions_count << " solutions found" << std::endl;
    std::cout << "Success: " << (result.success ? "Yes" : "No") << std::endl;
    std::cout << "=== cpp_ms test complete ===" << std::endl;
#endif