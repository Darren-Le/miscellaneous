CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -flto -funroll-loops -DNDEBUG
INCLUDES = -I/home/szg/local/include/eigen3 -I/home/szg/local/include
LIBS = -L/home/szg/local/lib -lfplll -lgmp -lmpfr -lstdc++fs

# Source files
SOURCES = ms_solve.cpp main.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = ms_solver

# Debug configuration
CXXFLAGS_DEBUG = -std=c++17 -g -O0 -DDEBUG
TARGET_DEBUG = ms_solver_debug
OBJECTS_DEBUG = $(SOURCES:.cpp=.debug.o)

# Default target
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Debug target
debug: $(TARGET_DEBUG)

$(TARGET_DEBUG): $(OBJECTS_DEBUG)
	$(CXX) -o $@ $^ $(LIBS)

%.debug.o: %.cpp
	$(CXX) $(CXXFLAGS_DEBUG) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(OBJECTS_DEBUG) $(TARGET) $(TARGET_DEBUG)