#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "../include/catch.hpp"
#include "../include/Eigen/Dense"

#include "../src/nodes/SoftXent.h"

TEST_CASE("Forward propagation for Softmax Cross-Entropy", "[SOFTMAX_XENT]" ) {
  Node logits,labels;
  Eigen::MatrixXd labels_(5,3);
  labels_ << 0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 1.0;

  Eigen::MatrixXd logits_(5,3);
  logits_ << 0.3, 0.5, 0.7,
            -0.9, 4.0, -98.0,
            599.0, 9.0, 3.0,
            -34.0, -67.0, 0.001,
            0.1, 0.2, 0.999999;

  Eigen::MatrixXd refMat(1,1);
  refMat << 14.0540094602f,

  labels.setValues(labels_);
  logits.setValues(logits_);
  SoftXent xe(&logits, &labels);
  xe.forward();

  Eigen::MatrixXd result;
  xe.getValues(result);

  double epsilon = 1.0e-7;
  REQUIRE((refMat - result).norm() < epsilon);
}

TEST_CASE("Backward propagation for Softmax Cross-Entropy", "[SOFTMAX_XENT]" ) {
  Node logits,labels;
  Eigen::MatrixXd labels_(5,3);
  labels_ << 0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 1.0;

  Eigen::MatrixXd logits_(5,3);
  logits_ << 0.3, 0.5, 0.7,
            -0.9, 4.0, -98.0,
            599.0, 9.0, 3.0,
            -34.0, -67.0, 0.001,
            0.1, 0.2, 0.999999;

  Eigen::MatrixXd refMat(5,3);
  refMat << 5.38614998e-002, -1.34213416e-001, 8.03519157e-002,
            1.47830827e-003, 1.98521692e-001, -2.00000000e-001,
            0.00000000e+000, 1.16757738e-257, 2.89413497e-260,
            -2.00000000e-001, 1.59538467e-030, 2.00000000e-001,
            4.38138025e-002, 4.84217403e-002, -9.22355428e-002;


  labels.setValues(labels_);
  logits.setValues(logits_);
  SoftXent xe(&logits, &labels);
  xe.forward();
  xe.backward();

  double epsilon = 1.0e-7;
  Eigen::MatrixXd gl;
  xe.getGradients(&logits, gl);
  // REQUIRE(true);
  REQUIRE((gl - refMat).norm() < epsilon);
}
