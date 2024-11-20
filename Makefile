SRC = $(wildcard src/*.cpp)
OBJS = $(SRC:src/%.cpp=bin/%.o)
DEBUG_FLAGS = -g -WALL
FLAGS = -std=c++20 -O2

all: build run

build: $(OBJS)
	g++ $(FLAGS) $^ -o bin/my_program

run:
	bin/my_program

# Clean target to remove object files and binaries
clean:
	rm -f bin/*.o bin/my_program

# Object file rule
bin/%.o: src/%.cpp
	g++ $(FLAGS) -c $< -o $@

# Debug target for building with debug flags
debug: $(OBJS)
	g++ $(DEBUG_FLAGS) $^ -o bin/my_program
