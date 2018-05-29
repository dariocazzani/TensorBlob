#include "Node.h"
#include "Input.h"
#include "Add.h"
#include "Multiply.h"
#include "inputs.h"

int main()
{
  vector<Node> inputs_1;
  vector<double> inputValues_1 = {0.4, 0.1, 0.9};
  initialize_inputs(&inputValues_1, &inputs_1);

  vector<Node> inputs_2;
  vector<double> inputValues_2 = {3.0, 1.0, 0.5};
  initialize_inputs(&inputValues_2, &inputs_2);

  Add a(inputs_1);
  Multiply m(inputs_2);
  a.forward();
  m.forward();

  vector<Node> out_inputs = {a, m};
  Add out(out_inputs);
  out.forward();
  out.printValue();
  return 0;
}
