CXX = g++ -std=c++11
DEBUG = -g
CFLAGS = -Wall -Wextra -Werror -pedantic $(DEBUG)

BUILDDIR := ../build/
TARGET := ../bin/

all: build
	@echo " Running tests..."
	$(TARGET)node_test
	$(TARGET)graph_test

first:
	@echo " Building tests..."

build: node_test graph_test

node_test : node_test.o
	$(CXX) -o $(TARGET)node_test $(BUILDDIR)node_test.o

node_test.o: first node_test.cc
	$(CXX) $(CFLAGS) -c node_test.cc -o $(BUILDDIR)node_test.o

graph_test : graph_test.o
	$(CXX) -o $(TARGET)graph_test $(BUILDDIR)graph_test.o

graph_test.o: graph_test.cc
	$(CXX) $(CFLAGS) -c graph_test.cc -o $(BUILDDIR)graph_test.o

clean:
	rm -rf $(BUILDDIR)*.o $(TARGET)*

.PHONY: all run
