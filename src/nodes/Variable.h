#ifndef VARIABLE_H
#define VARIABLE_H

#include "Node.h"

class Variable : public Node
{
public:
  Variable();
  Variable(double value);
  Variable(const Eigen::MatrixXd &values);
  void forward();
  void backward();
};

Variable::Variable() {}
Variable::Variable(double value)
{
  setValues(value);
}

Variable::Variable(const Eigen::MatrixXd &values)
{
  setValues(values);
}


void Variable::forward() {}
void Variable::backward()
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
