CXX = g++
CXXFLAGS = -std=c++17 -O3
INCLUDES = -I/home/szg/local/include/eigen3 -I/home/szg/local/include
LIBS = -L/home/szg/local/lib -lfplll -lgmp -lmpfr -lstdc++fs

# Source files
SOURCES = ms_solve.cpp main.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = ms_solver

# Default target
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)
