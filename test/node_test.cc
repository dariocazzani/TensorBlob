#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "../include/catch.hpp"
#include "../include/Eigen/Dense"

#include "../src/nodes/Node.h"

TEST_CASE("Set Value", "[SET]" ) {
    Node n;
    double refValue {3.0};
    n.setValues(refValue);
    Eigen::VectorXd value;
    n.getValues(value);
    REQUIRE(refValue == value(0));
}
