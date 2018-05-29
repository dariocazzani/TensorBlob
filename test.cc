#include "Node.h"
#include "Input.h"
#include "Add.h"
#include "inputs.h"

int main()
{
  // Input i1(0.5);
  // Input i2(0.3);
  // vector<Node> inputs = {i1, i2};

  vector<Node> inputs_1;
  vector<double> inputValues_1 = {0.4, 0.1, 0.9};
  initialize_inputs(&inputValues_1, &inputs_1);

  vector<Node> inputs_2;
  vector<double> inputValues_2 = {0.2, 0.9, 3.9};
  initialize_inputs(&inputValues_2, &inputs_2);

  Add a1(inputs_1);
  Add a2(inputs_2);
  a1.forward();
  a1.printValue();
  a2.forward();
  a2.printValue();

  vector<Node> a3_inputs = {a1, a2};
  Add a3(a3_inputs);
  a3.forward();
  a3.printValue();
  return 0;
}
