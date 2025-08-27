CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -DNDEBUG -fopenmp
INCLUDES = -I/usr/include/eigen3 -I/usr/local/include/fplll
LIBS = -lfplll -lgmp -lmpfr
LDFLAGS = -fopenmp

# Source files
SOURCES = ms_solve.cpp main.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = ms_solver

# Default target
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Dependencies
ms_solve.o: ms_solve.cpp ms_solve.h
main.o: main.cpp ms_solve.h

# Clean target
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install -y libeigen3-dev libfplll-dev libgmp-dev libmpfr-dev build-essential

# Alternative install for other systems
install-deps-brew:
	brew install eigen fplll gmp mpfr

# Debug build
debug: CXXFLAGS = -std=c++17 -O0 -g -Wall -Wextra -fopenmp
debug: $(TARGET)

# Profile build  
profile: CXXFLAGS = -std=c++17 -O3 -march=native -pg -fopenmp
profile: LDFLAGS = -fopenmp -pg
profile: $(TARGET)

# Test target
test: $(TARGET)
	./$(TARGET) --data_path ms_instance/01-marketsplit/instances --sol_path ms_instance/01-marketsplit/solutions

.PHONY: all clean install-deps install-deps-brew debug profile test