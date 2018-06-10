#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "../include/catch.hpp"
#include "../include/Eigen/Dense"

#include "../src/nodes/MSE.h"

TEST_CASE("Forward propagation for MSE", "[MSE]" ) {
  Node act, l;
  Eigen::MatrixXd activations(3,1);
  activations << 0.0, 0.1, 0.2;

  Eigen::MatrixXd labels(3,1);
  labels << 0.3, 0.5, 0.7;

  Eigen::MatrixXd refMat(1,1);
  refMat << 0.23570226039551581f,

  act.setValues(activations);
  l.setValues(labels);
  MSE mse(&act, &l);
  mse.forward();

  Eigen::MatrixXd result;
  mse.getValues(result);

  double epsilon = 1.0e-7;
  REQUIRE((refMat - result).norm() < epsilon);
}

TEST_CASE("Correct vector shape", "[MSE]" ) {
  Node act, l;
  Eigen::MatrixXd activations(3,2);
  activations << 0.0, 0.1, 0.2,
                 0.3, 0.5, 0.7;

  Eigen::MatrixXd labels(3,2);
  labels << 0.3, 0.5, 0.7,
            0.0, 0.1, 0.2;

  act.setValues(activations);
  l.setValues(labels);
  MSE mse(&act, &l);
  CHECK_THROWS(mse.forward());
}

TEST_CASE("Same shape for input vectors", "[MSE]" ) {
  Node act, l;
  Eigen::MatrixXd activations(3,2);
  activations << 0.0, 0.1, 0.2,
                 0.3, 0.5, 0.7;

  Eigen::MatrixXd labels(3,1);
  labels << 0.3, 0.5, 0.7;

  act.setValues(activations);
  l.setValues(labels);
  MSE mse(&act, &l);
  CHECK_THROWS(mse.forward());
}

TEST_CASE("Backward propagation for MSE", "[MSE]" ) {
  Node act, l;
  Eigen::MatrixXd activations(5,1);
  activations << -1.0, 0.5, -0.6, 0.9, -1.5;

  Eigen::MatrixXd labels(5,1);
  labels << -1.8, 1.5, 1.6, -0.4, 0.9;

  Eigen::MatrixXd gradAct(5,1);
  Eigen::MatrixXd gradLab(5,1);
  gradLab << -0.32, 0.4, 0.88, -0.52, 0.96;
  gradAct << 0.32, -0.4, -0.88, 0.52, -0.96;

  act.setValues(activations);
  l.setValues(labels);
  MSE mse(&act, &l);
  mse.forward();
  mse.backward();

  double epsilon = 1.0e-7;
  Eigen::MatrixXd ga, gl;
  mse.getGradients(&act, ga);
  mse.getGradients(&l, gl);
  REQUIRE((ga - gradAct).norm() < epsilon);
  REQUIRE((gl - gradLab).norm() < epsilon);
}
