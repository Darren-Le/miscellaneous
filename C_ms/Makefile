# Makefile for Market Split C Implementation

# Compiler and flags
CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -O3 -std=c11 -g
CXXFLAGS = -Wall -Wextra -O3 -std=c++11 -g

# Directories
FPLLL_DIR = /home/szg/local
INCLUDE_DIR = -I$(FPLLL_DIR)/include -I.
LIB_DIR = -L$(FPLLL_DIR)/lib

# Libraries
LIBS = -lfplll -lmpfr -lgmp -lm

# Source files
C_SOURCES = ms_data.c
CXX_SOURCES = ms_solve.c main.cpp
HEADERS = ms_data.h ms_solve.h

# Object files  
C_OBJECTS = $(C_SOURCES:.c=.o)
CXX_OBJECTS = ms_solve.o main.o
OBJECTS = $(C_OBJECTS) $(CXX_OBJECTS)

# Target executable
TARGET = ms_solve

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(LIB_DIR) $(LIBS)

# Compile C source files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDE_DIR) -c $< -o $@

# For ms_solve.c, compile as C++ due to fplll dependency
ms_solve.o: ms_solve.c ms_solve.h ms_data.h
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIR) -c ms_solve.c -o ms_solve.o

main.o: main.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIR) -c main.cpp -o main.o

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Clean and rebuild
rebuild: clean all

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install -y build-essential libgmp-dev libmpfr-dev

# Install fplll (if not already installed)
install-fplll:
	@echo "Installing fplll to $(FPLLL_DIR)..."
	@echo "Make sure you have downloaded and built fplll in $(FPLLL_DIR)"
	@echo "If not installed, run:"
	@echo "  wget https://github.com/fplll/fplll/releases/download/5.4.2/fplll-5.4.2.tar.gz"
	@echo "  tar xzf fplll-5.4.2.tar.gz"
	@echo "  cd fplll-5.4.2"
	@echo "  ./configure --prefix=$(FPLLL_DIR)"
	@echo "  make && make install"

# Test with sample data (if available)
test: $(TARGET)
	@echo "Running test with default parameters..."
	./$(TARGET) --debug

# Test specific m values
test-small: $(TARGET)
	@echo "Testing with small instances (m=3,4)..."
	./$(TARGET) --max_sols 10 --debug

# Help target
help:
	@echo "Available targets:"
	@echo "  all         - Build the executable (default)"
	@echo "  clean       - Remove build artifacts"
	@echo "  rebuild     - Clean and rebuild"
	@echo "  test        - Run test with default parameters"
	@echo "  test-small  - Run test with small instances"
	@echo "  install-deps - Install system dependencies"
	@echo "  install-fplll - Show fplll installation instructions"
	@echo "  help        - Show this help message"

# Phony targets
.PHONY: all clean rebuild test test-small install-deps install-fplll help

# Debug information
debug-info:
	@echo "Compiler: $(CC) / $(CXX)"
	@echo "C Flags: $(CFLAGS)"
	@echo "C++ Flags: $(CXXFLAGS)"
	@echo "Include dirs: $(INCLUDE_DIR)"
	@echo "Library dirs: $(LIB_DIR)"
	@echo "Libraries: $(LIBS)"
	@echo "FPLLL directory: $(FPLLL_DIR)"
	@echo "Sources: $(C_SOURCES) $(CXX_SOURCES)"
	@echo "Objects: $(OBJECTS)"