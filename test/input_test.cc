#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "../include/catch.hpp"
#include "../include/Eigen/Dense"

#include "../src/nodes/Sigmoid.h"
#include "../src/nodes/Input.h"
#include "../src/nodes/MSE.h"
#include "../src/graph/graph_utils.h"

TEST_CASE("Backward propagation for Input with no output", "[INPUT]" ) {
  Input in;
  Eigen::MatrixXd inputs = Eigen::MatrixXd::Random(2,3);
  in.setValues(inputs);

  Eigen::MatrixXd refValue = Eigen::MatrixXd::Zero(2,3);

  in.forward();
  in.backward();

  Eigen::MatrixXd g;
  in.getGradients(&in, g);
  double epsilon = 1.0e-7;
  REQUIRE((refValue - g).norm() < epsilon);
}

TEST_CASE("Backward propagation for Input with 1 output", "[INPUT]" ) {
  Input x;
  Sigmoid s(&x);

  Eigen::MatrixXd x_(5, 1);
  x_ << -1.0, 0.5, -0.6, 0.9, -1.5;

  vector<Node *> graph = {&x, &s};

  // FEED_DICT
  map<Node*, Eigen::MatrixXd> inputMap;
  inputMap[&x] = x_;

  Eigen::MatrixXd refValue(5,1);
  refValue << 0.19661193, 0.235003716, 0.2287842, 0.20550031, 0.14914645;

  buildGraph(graph);
  feedValues(inputMap);
  vector<Eigen::MatrixXd> results = forwardBackward(graph);

  Eigen::MatrixXd g;
  x.getGradients(&x, g);
  double epsilon = 1.0e-7;
  REQUIRE((refValue - g).norm() < epsilon);
}


TEST_CASE("Backward propagation for Input with 2 outputs", "[INPUT]" ) {
  Input x;
  Sigmoid s(&x);
  MSE cost(&s, &x);

  Eigen::MatrixXd x_(5, 1);
  x_ << -1.0, 0.5, -0.6, 0.9, -1.5;

  vector<Node *> graph = {&x, &s, &cost};

  // FEED_DICT
  map<Node*, Eigen::MatrixXd> inputMap;
  inputMap[&x] = x_;

  Eigen::MatrixXd refValue(5,1);
  refValue << -0.40778096, -0.03747237, -0.29440196, 0.06008022, -0.57259909;

  buildGraph(graph);
  feedValues(inputMap);
  vector<Eigen::MatrixXd> results = forwardBackward(graph);

  Eigen::MatrixXd g;
  x.getGradients(&x, g);
  double epsilon = 1.0e-7;
  REQUIRE((refValue - g).norm() < epsilon);
}
