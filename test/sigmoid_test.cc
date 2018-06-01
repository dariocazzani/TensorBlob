#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "../include/catch.hpp"
#include "../include/Eigen/Dense"

#include "../src/nodes/Sigmoid.h"

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
