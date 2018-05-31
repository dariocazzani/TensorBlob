#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "../include/catch.hpp"

#include "../src/nodes/Node.h"
#include "../src/nodes/Input.h"
#include "../src/nodes/Add.h"

#include "../src/graph/graph_utils.h"

TEST_CASE("Build Directed Acyclic Graph", "[DAG]" ) {
  Input n5;
  Input n4;
  vector<Node *> inputs = {&n5};
  Add n2(inputs);

  inputs = {&n5, &n4};
  Add n0(inputs);

  inputs = {&n2};
  Add n3(inputs);

  inputs = {&n3, &n4};
  Add n1(inputs);

  vector<Node *> graph = {&n0, &n1, &n2, &n3, &n4, &n5};
  CHECK_NOTHROW(buildGraph(graph));
}

TEST_CASE("Build NOT Direceted Acyclic Graph", "[DAG]" ) {
  Input n5;
  Input n4;
  vector<Node *> inputs = {&n5};
  Add n2(inputs);

  inputs = {&n5, &n4};
  Add n0(inputs);

  inputs = {&n2};
  Add n3(inputs);

  inputs = {&n3, &n4};
  Add n1(inputs);

  n0.addInput(&n3);
  n5.addInput(&n0);

  vector<Node *> graph = {&n0, &n1, &n2, &n3, &n4, &n5};
  CHECK_THROWS(buildGraph(graph));
}
