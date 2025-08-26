#!/bin/bash

# Test script for Market Split C implementation

set -e  # Exit on any error

echo "==================================="
echo "Market Split C Implementation Test"
echo "==================================="
echo

# Check if the executable exists
if [ ! -f "./ms_solve" ]; then
    echo "Error: ms_solve executable not found."
    echo "Please run 'make' to build the project first."
    exit 1
fi

# Check if data directories exist
DATA_DIR="ms_instance/01-marketsplit/instances"
SOL_DIR="ms_instance/01-marketsplit/solutions"

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory $DATA_DIR not found."
    echo "Please make sure the instance data is available."
    exit 1
fi

if [ ! -d "$SOL_DIR" ]; then
    echo "Warning: Solution directory $SOL_DIR not found."
    echo "Optimal solution checking will be disabled."
    SOL_DIR=""
fi

# Test 1: Quick test with limited solutions
echo "Test 1: Quick test with max 5 solutions per instance"
echo "----------------------------------------------------"
if [ -n "$SOL_DIR" ]; then
    ./ms_solve --data_path "$DATA_DIR" --sol_path "$SOL_DIR" --max_sols 5
else
    ./ms_solve --data_path "$DATA_DIR" --max_sols 5
fi
echo

# Test 2: Debug mode test on small instances
echo "Test 2: Debug mode test (first 3 instances only)"
echo "------------------------------------------------"
echo "Note: This test will show detailed debug information"
echo "Press Enter to continue or Ctrl+C to skip..."
read -r

# Create a temporary directory with just a few instances for debug test
TEMP_DIR=$(mktemp -d)
echo "Creating temporary test directory: $TEMP_DIR"

# Copy first few .dat files for testing
count=0
for file in "$DATA_DIR"/*.dat; do
    if [ $count -ge 3 ]; then break; fi
    cp "$file" "$TEMP_DIR/"
    count=$((count + 1))
done

# Copy corresponding solution files if they exist
if [ -n "$SOL_DIR" ]; then
    for file in "$TEMP_DIR"/*.dat; do
        basename=$(basename "$file" .dat)
        if [ -f "$SOL_DIR/${basename}.opt.sol" ]; then
            cp "$SOL_DIR/${basename}.opt.sol" "$TEMP_DIR/"
        fi
    done
    ./ms_solve --data_path "$TEMP_DIR" --sol_path "$TEMP_DIR" --max_sols 10 --debug
else
    ./ms_solve --data_path "$TEMP_DIR" --max_sols 10 --debug
fi

# Clean up temporary directory
rm -rf "$TEMP_DIR"
echo

# Test 3: Performance comparison setup
echo "Test 3: Performance test (all solutions)"
echo "----------------------------------------"
echo "Warning: This may take a significant amount of time for large instances."
echo "Press Enter to continue or Ctrl+C to skip..."
read -r

echo "Running performance test..."
time_start=$(date +%s)

if [ -n "$SOL_DIR" ]; then
    ./ms_solve --data_path "$DATA_DIR" --sol_path "$SOL_DIR" --max_sols -1
else
    ./ms_solve --data_path "$DATA_DIR" --max_sols -1
fi

time_end=$(date +%s)
duration=$((time_end - time_start))

echo "Performance test completed in $duration seconds."
echo

# Test 4: Compare with Python version (if available)
echo "Test 4: Python comparison (optional)"
echo "-----------------------------------"
if [ -f "ms_solve.py" ]; then
    echo "Python version found. Running comparison test..."
    echo "C version results are in the latest .log file"
    echo "Running Python version with same parameters..."
    
    if command -v python3 &> /dev/null; then
        if [ -n "$SOL_DIR" ]; then
            python3 ms_solve.py --data_path "$DATA_DIR" --sol_path "$SOL_DIR" --max_sols 1
        else
            python3 ms_solve.py --data_path "$DATA_DIR" --max_sols 1
        fi
        echo "Python version completed. Compare the .log files for differences."
    else
        echo "Python3 not found. Skipping Python comparison."
    fi
else
    echo "Python version (ms_solve.py) not found. Skipping comparison."
fi

echo
echo "==================================="
echo "All tests completed!"
echo "==================================="
echo "Check the generated .log and .sol files for detailed results."
echo

# Display recent log files
echo "Recent result files:"
ls -lt *.log *.sol 2>/dev/null | head -5 || echo "No result files found."