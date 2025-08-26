CC = g++
CFLAGS = -Wall -O2 -std=c++11
LIBS = -lfplll -lm

TARGET = ms_solver
SOURCES = main.cpp ms_data.c ms_solve.c

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: clean