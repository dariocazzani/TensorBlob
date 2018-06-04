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
void Input::backward()
{
  vector<Node *> outputs = getOutputNodes();
  // # Initialize the gradients to 0.
  Eigen::MatrixXd tempGrad = Eigen::MatrixXd::Zero(this->getValuesRows(), this->getValuesCols());
  Eigen::MatrixXd gradCost;
  for(auto n : outputs)
  {
    // Get gradient of outBound Node w.r.t. current node
    n->getGradients(this, gradCost);
    tempGrad += gradCost;
  }
  setGradients(this, tempGrad);
}

#endif
