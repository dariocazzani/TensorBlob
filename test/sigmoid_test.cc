#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "../include/catch.hpp"
#include "../include/Eigen/Dense"

#include "../src/nodes/Sigmoid.h"
#include "../src/nodes/MSE.h"
#include "../src/graph/graph_utils.h"

TEST_CASE("Forward propagation for Sigmoid", "[SIGMOID]" ) {
  Node n;
  Eigen::MatrixXd inputs(3,3);
  inputs << 0.0, 0.1, 0.2,
            0.3, 0.4, 0.5,
            0.6, 0.7, 0.8;

  Eigen::MatrixXd refValue(3,3);
  refValue << 0.5, 0.52497919, 0.549834,
              0.57444252, 0.59868766, 0.62245933,
              0.6456563, 0.66818777, 0.68997448;

  n.setValues(inputs);
  Sigmoid s(&n);
  s.forward();

  Eigen::MatrixXd result;
  s.getValues(result);

  double epsilon = 1.0e-7;
  REQUIRE((refValue - result).norm() < epsilon);
}

TEST_CASE("Backward propagation for Sigmoid with no output", "[SIGMOID]" ) {
  Node n;
  Eigen::MatrixXd inputs(5, 2);
  inputs << -1.0, 0.5, -0.6, 0.9, -1.5,
            -1.0, 0.5, -0.6, 0.9, -1.5;

  Eigen::MatrixXd refValue(5,2);
  refValue << 0.19661193, 0.23500371, 0.22878424, 0.20550031, 0.149146456,
              0.19661193, 0.23500371, 0.22878424, 0.20550031, 0.149146456;

  n.setValues(inputs);
  Sigmoid s(&n);
  s.forward();
  s.backward();

  Eigen::MatrixXd g;
  s.getGradients(&n, g);
  double epsilon = 1.0e-7;
  REQUIRE((refValue - g).norm() < epsilon);
}

TEST_CASE("Backward propagation for Sigmoid with 1 output", "[SIGMOID]" ) {
  Input x;
  Input y;
  Sigmoid s(&x);
  MSE cost(&s, &y);

  Eigen::MatrixXd x_(5, 1);
  Eigen::MatrixXd y_(5, 1);
  x_ << -1.0, 0.5, -0.6, 0.9, -1.5;
  y_ << -1.0, 0.5, -0.6, 0.9, -1.5;

  vector<Node *> graph = {&x, &y, &s, &cost};

  // FEED_DICT
  map<Node*, Eigen::MatrixXd> inputMap;
  inputMap[&x] = x_;
  inputMap[&y] = y_;

  Eigen::MatrixXd refValue(5,1);
  refValue << 0.09979561, 0.01151136, 0.08733552, -0.01553997, 0.10037112;

  buildGraph(graph);
  feedValues(inputMap);
  vector<Eigen::MatrixXd> results = forwardBackward(graph);

  Eigen::MatrixXd g;
  s.getGradients(&x, g);
  double epsilon = 1.0e-7;
  REQUIRE((refValue - g).norm() < epsilon);
}
