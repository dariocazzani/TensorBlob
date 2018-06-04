#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "../include/catch.hpp"
#include "../include/Eigen/Dense"

#include "../src/nodes/Sigmoid.h"
#include "../src/nodes/MSE.h"
#include "../src/nodes/Linear.h"
#include "../src/nodes/Input.h"
#include "../src/graph/graph_utils.h"

TEST_CASE("Forward propagation for Linear", "[LINEAR]" ) {
  Input W;
  Eigen::MatrixXd weights(3,3);
  weights << 0.0, 0.1, 0.2,
             0.3, 0.4, 0.5,
             0.6, 0.7, 0.8;

  Input X;
  Eigen::MatrixXd x_(5,3);
  x_ << 0.3, 0.4, 0.5,
        0.0, 0.1, 0.2,
        0.3, 0.4, 0.5,
        0.6, 0.7, 0.8,
        0.0, 0.1, 0.2;

  Input b;
  Eigen::MatrixXd bias(1,3);
  bias << 0.3, 0.4, 0.5;

  Linear f(&X, &W, &b);

  Eigen::MatrixXd refValue(5,3);
  refValue << 0.72,  0.94,  1.16,
              0.45,  0.58,  0.71,
              0.72,  0.94,  1.16,
              0.99,  1.3 ,  1.61,
              0.45,  0.58,  0.71;

  W.setValues(weights);
  X.setValues(x_);
  b.setValues(bias);

  W.forward();
  X.forward();
  b.forward();
  f.forward();

  Eigen::MatrixXd result;
  f.getValues(result);

  double epsilon = 1.0e-7;
  REQUIRE((refValue - result).norm() < epsilon);
}

TEST_CASE("Backward propagation for Linear - 1 hidden layer", "[LINEAR]" ) {
  Input X;
  Eigen::MatrixXd x_(2,2);
  x_ << -1.0, -2.0,
        -1.0, -2.0;

  Input W;
  Eigen::MatrixXd weights(2,1);
  weights << 2.0, 3.0;

  Input b;
  Eigen::MatrixXd bias(1,1);
  bias << -3.0;

  Input y;
  Eigen::MatrixXd y_(2,1);
  y_ << 1.0, 2.0;

  Eigen::MatrixXd xGrads(2,2);
  xGrads << -3.34017280e-05,  -5.01025919e-05,
            -6.68040138e-05,  -1.00206021e-04;

  Eigen::MatrixXd weightsGrads(2,1);
  weightsGrads << 5.01028709e-05, 1.00205742e-04;

  Eigen::MatrixXd biasGrads(1,1);
  biasGrads << -5.01028709e-05;

  // GRAPH
  Linear f(&X, &W, &b);
  Sigmoid a(&f);
  MSE cost(&a, &y);

  vector<Node *> graph = {&y, &a, &f, &cost, &X, &W, &b};

  // FEED_DICT
  map<Node*, Eigen::MatrixXd> inputMap;
  inputMap[&W] = weights;
  inputMap[&X] = x_;
  inputMap[&b] = bias;
  inputMap[&y] = y_;

  buildGraph(graph, inputMap);
  vector<Eigen::MatrixXd> results = forwardBackward(graph);

  Eigen::MatrixXd gW;
  W.getGradients(&W, gW);

  Eigen::MatrixXd gb;
  b.getGradients(&b, gb);

  Eigen::MatrixXd gX;
  X.getGradients(&X, gX);

  double epsilon = 1.0e-7;
  REQUIRE( ((weightsGrads - gW).norm()) +
           ((biasGrads - gb).norm()) +
           ((xGrads - gX).norm())
                  < epsilon);
}
