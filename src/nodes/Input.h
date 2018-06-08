#ifndef INPUT_H
#define INPUT_H

#include "Node.h"

class Input : public Node
{
public:
  Input();
  Input(double value);
  Input(const Eigen::MatrixXd &values);
  void forward();
  void backward();
  void getGradients(const Node *n, Eigen::MatrixXd &grad);
};

Input::Input() {}
Input::Input(double value)
{
  setValues(value);
}

Input::Input(const Eigen::MatrixXd &values)
{
  setValues(values);
}

void Input::forward() {}
// Input nodes can't be updated
void Input::backward() {}

void Input::getGradients(const Node *n, Eigen::MatrixXd &grad)
{
  (void) n;
  (void) grad;
  throw domain_error("Gradients not defined for Nodes of type Input.");
}
#endif
