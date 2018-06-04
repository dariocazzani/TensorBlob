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
  void backward();
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

void Sigmoid::backward()
{
  vector<Node *> inputs = getInputNodes();
  vector<Node *> outputs = getOutputNodes();

  Eigen::MatrixXd currValues;
  if(outputs.size() == 0)
  {
    getValues(currValues);
    // currValues = currValues * (1 - currValues)
    currValues = currValues.cwiseProduct(Eigen::MatrixXd::Ones(currValues.rows(), currValues.cols()) - currValues);
    setGradients(inputs[0], currValues);
  }
  else
  {
    // # Initialize the gradients to 0.
    Eigen::MatrixXd tempGrad = Eigen::MatrixXd::Zero(inputs[0]->getValuesRows(), inputs[0]->getValuesCols());
    Eigen::MatrixXd gradCost;
    for(auto n : getOutputNodes())
    {
      // Get gradient of outBound Node w.r.t. current node
      n->getGradients(this, gradCost);
      getValues(currValues);
      currValues = currValues.cwiseProduct(Eigen::MatrixXd::Ones(currValues.rows(), currValues.cols()) - currValues);
      tempGrad += currValues.cwiseProduct(gradCost);
    }
    setGradients(inputs[0], tempGrad);
  }
}

#endif
