#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "../include/catch.hpp"
#include "../include/Eigen/Dense"

#include "../src/nodes/Sigmoid.h"
#include "../src/nodes/Linear.h"
#include "../src/nodes/Variable.h"
#include "../src/nodes/MSE.h"
#include "../src/graph/graph_utils.h"

TEST_CASE("Backward propagation for Variable with no output", "[VARIABLE]" ) {
  Variable W;
  Eigen::MatrixXd weights = Eigen::MatrixXd::Random(2,3);
  W.setValues(weights);

  Eigen::MatrixXd refValue = Eigen::MatrixXd::Zero(2,3);

  W.forward();
  W.backward();

  Eigen::MatrixXd g;
  W.getGradients(&W, g);
  double epsilon = 1.0e-7;
  REQUIRE((refValue - g).norm() < epsilon);
}

TEST_CASE("Backward propagation for Variable with 1 output - Sigmoid Node", "[VARIABLE]" ) {
  Variable x;
  Sigmoid s(&x);

  Eigen::MatrixXd x_(5, 1);
  x_ << -1.0, 0.5, -0.6, 0.9, -1.5;

  vector<Node *> graph = {&x, &s};

  x.setValues(x_);

  Eigen::MatrixXd refValue(5,1);
  refValue << 0.19661193, 0.235003716, 0.2287842, 0.20550031, 0.14914645;

  buildGraph(graph);
  forwardBackward(graph);

  Eigen::MatrixXd g;
  x.getGradients(&x, g);
  double epsilon = 1.0e-7;
  REQUIRE((refValue - g).norm() < epsilon);
}


TEST_CASE("Backward propagation for Input with 2 outputs", "[INPVARIABLEUT]" ) {
  Variable x;
  Sigmoid s(&x);
  MSE cost(&s, &x);

  Eigen::MatrixXd x_(5, 1);
  x_ << -1.0, 0.5, -0.6, 0.9, -1.5;

  vector<Node *> graph = {&x, &s, &cost};

  x.setValues(x_);

  Eigen::MatrixXd refValue(5,1);
  refValue << -0.40778096, -0.03747237, -0.29440196, 0.06008022, -0.57259909;

  buildGraph(graph);
  forwardBackward(graph);

  Eigen::MatrixXd g;
  x.getGradients(&x, g);
  double epsilon = 1.0e-7;
  REQUIRE((refValue - g).norm() < epsilon);
}
