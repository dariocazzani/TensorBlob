#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "../include/catch.hpp"

#include "../src/nodes/Node.h"

TEST_CASE("Set Value", "[SET]" ) {
    Node n;
    double value {3.0};
    n.setValue(value);
    REQUIRE(n.getValue() == value);
}

TEST_CASE("Add input", "[ADD]" ) {
    Node n1;
    double value1 {1.0};
    n1.setValue(value1);
    Node n2;
    double value2 {2.0};
    n2.setValue(value2);

    // Make n1 input node to n2
    n2.addInput(&n1);

    // Test that n2 is output of n1
    vector<Node *> temp = n1.getOutputNodes();
    REQUIRE(temp[0]->getValue() == value2);
}
