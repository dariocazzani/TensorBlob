CXX = g++ -std=c++11
DEBUG = -g
CFLAGS = -Wall -Wextra -Werror -pedantic $(DEBUG)

SRCDIR := ../src/
BUILDDIR := ../build/
TARGET := ../bin/

all: main
	
main.o: first main.cc
	$(CXX) $(CFLAGS) -c main.cc -o $(BUILDDIR)main.o

main: main.o
	$(CXX) $(CFLAGS) -o $(TARGET)main $(BUILDDIR)main.o

first:
	@echo " Building main..."
