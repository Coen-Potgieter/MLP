SRC = $(wildcard src/*.cpp)
OBJS = $(SRC:src/%.cpp=bin/%.o)
DEBUG_OBJS = $(SRC:src/%.cpp=bin/debug_%.o)
FLAGS = -std=c++20 -O2
DEBUG_FLAGS = $(FLAGS) -DDEBUG -g -Wall

all: build run

build: $(OBJS)
	g++ $(FLAGS) $^ -o bin/my_program

run:
	bin/my_program

# Object file rule
bin/%.o: src/%.cpp
	g++ $(FLAGS) -c $< -o $@

# Clean target to remove object files and binaries
clean:
	rm -f bin/*.o bin/my_program

# Debug build
debug: $(DEBUG_OBJS)
	g++ $(DEBUG_FLAGS) $^ -o bin/my_program


# Object files for debug (built with debug flags)
bin/debug_%.o: src/%.cpp
	g++ $(DEBUG_FLAGS) -c $< -o $@

