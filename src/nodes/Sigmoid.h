#ifndef SIGMOID_H
#define SIGMOID_H

#include "Node.h"

// NB only 1 Node can be inbound for Sigmoid
class Sigmoid : public Node
{
private:
  Eigen::MatrixXd sigmoid(const Eigen::MatrixXd &values);
public:
  Sigmoid(Node *inputs);
  void forward();
};

Eigen::MatrixXd Sigmoid::sigmoid(const Eigen::MatrixXd &values) {
  return (1.0 + (-values).array().exp()).inverse().matrix();
}

Sigmoid::Sigmoid(Node *inputs)
{
  addInput(inputs);
}
void Sigmoid::forward()
{
  Eigen::MatrixXd inputs;
  inputs = sigmoid(getInputValues()[0]);
  setValues(inputs);
}

#endif
