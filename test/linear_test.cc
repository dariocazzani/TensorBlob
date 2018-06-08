#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "../include/catch.hpp"
#include "../include/Eigen/Dense"

#include "../src/nodes/Sigmoid.h"
#include "../src/nodes/MSE.h"
#include "../src/nodes/Linear.h"
#include "../src/nodes/Input.h"
#include "../src/nodes/Variable.h"
#include "../src/graph/graph_utils.h"

TEST_CASE("Forward propagation for Linear", "[LINEAR]" ) {
  Variable W;
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

  Variable b;
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

  Variable W;
  Eigen::MatrixXd weights(2,1);
  weights << 2.0, 3.0;

  Variable b;
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

  W.setValues(weights);
  b.setValues(bias);

  // GRAPH
  Linear f(&X, &W, &b);
  Sigmoid a(&f);
  MSE cost(&a, &y);

  vector<Node *> graph = {&y, &a, &f, &cost, &X, &W, &b};

  // FEED_DICT
  map<Node*, Eigen::MatrixXd> inputMap;
  inputMap[&X] = x_;
  inputMap[&y] = y_;

  buildGraph(graph);
  feedValues(inputMap);
  forwardBackward(graph);

  Eigen::MatrixXd gW;
  f.getGradients(&W, gW);

  Eigen::MatrixXd gb;
  f.getGradients(&b, gb);

  Eigen::MatrixXd gX;
  f.getGradients(&X, gX);

  double epsilon = 1.0e-7;
  REQUIRE( ((weightsGrads - gW).norm()) +
           ((biasGrads - gb).norm()) +
           ((xGrads - gX).norm())
                  < epsilon);
}

TEST_CASE("Backward propagation for Linear - 2 hidden layers", "[LINEAR]" ) {
  Input X;
  Eigen::MatrixXd x_(4,2);
  x_ << -1.0, -2.0,
        -1.0, -2.0,
        -0.5, 1.3,
        1.2, -0.1;

  Input W1;
  Eigen::MatrixXd weights1(2,4);
  weights1 << 2.0, -0.5, 0.1, 0.6,
              3.0, -1.0, 0.01, -0.8;

  Input b1;
  Eigen::MatrixXd bias1(1,4);
  bias1 << -1.0, -0.5, 0.5, 1.1;

  Input W2;
  Eigen::MatrixXd weights2(4,1);
  weights2 << 2.0, -0.5, 0.1, 3.0;

  Input b2;
  Eigen::MatrixXd bias2(1,1);
  bias2 << -1.0;

  Input y;
  Eigen::MatrixXd y_(4,1);
  y_ << 1.0, 2.0, 0.0, -1.0;

  Eigen::MatrixXd xGrads(4,2);
  xGrads << 0.02979362, -0.02624426,
            -0.07225851,  0.06365025,
            0.95089284,  0.16285119,
            2.0297199 ,  1.92160657;

  Eigen::MatrixXd weightsGrads1(2,4);
  weightsGrads1 << 0.79345211, -0.21252842,  0.04811693,  0.50783704,
                   0.22373232, -0.09748136,  0.03678951,  1.03039037;

  Eigen::MatrixXd biasGrads1(1,4);
  biasGrads1 << 0.98732049, -0.26115161, 0.06506099, 1.37737493;

  Eigen::MatrixXd sigmoidGrad(4,4);
  sigmoidGrad << 0.2919451 , -0.07298627,  0.01459725,  0.43791765,
                 -0.7080549 ,  0.17701373, -0.03540275, -1.06208235,
                 2.03447153, -0.50861788,  0.10172358,  3.0517073,
                 4.04072319, -1.0101808 ,  0.20203616,  6.06108478;

  Eigen::MatrixXd weightsGrads2(4,1);
  weightsGrads2 << 2.4006558,
                   0.5382087,
                   1.81396245,
                   2.02001368;

  Eigen::MatrixXd biasGrads2(1,1);
  biasGrads2 << 2.82954246;

  // GRAPH
  Linear f1(&X, &W1, &b1);
  Sigmoid a1(&f1);
  Linear f2(&a1, &W2, &b2);
  MSE cost(&f2, &y);

  vector<Node *> graph = {&X, &W1, &b1, &a1, &f1, &f2, &W2, &b2, &y, &cost};

  // FEED_DICT
  map<Node*, Eigen::MatrixXd> inputMap;
  inputMap[&X] = x_;
  inputMap[&W1] = weights1;
  inputMap[&b1] = bias1;
  inputMap[&W2] = weights2;
  inputMap[&b2] = bias2;
  inputMap[&y] = y_;

  buildGraph(graph);
  feedValues(inputMap);
  forwardBackward(graph);

  Eigen::MatrixXd gW1;
  f1.getGradients(&W1, gW1);

  Eigen::MatrixXd gb1;
  f1.getGradients(&b1, gb1);

  Eigen::MatrixXd gX;
  f1.getGradients(&X, gX);

  Eigen::MatrixXd gW2;
  f2.getGradients(&W2, gW2);

  Eigen::MatrixXd gb2;
  f2.getGradients(&b2, gb2);

  Eigen::MatrixXd ga1;
  f2.getGradients(&a1, ga1);

  double epsilon = 1.0e-6;
  REQUIRE( ((weightsGrads1 - gW1).norm()) +
           ((biasGrads1 - gb1).norm()) +
           ((xGrads - gX).norm()) +
           ((weightsGrads2 - gW2).norm()) +
           ((biasGrads2 - gb2).norm()) +
           ((sigmoidGrad - ga1).norm())
                   < epsilon);
}
