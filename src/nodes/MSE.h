#ifndef MSE_H
#define MSE_H

#include "Node.h"
#include <map>
// NB MSE Has exactly 2 inputs Nodes
class MSE : public Node
{
private:
  double batchSize = {0.0f};
  Eigen::MatrixXd diff;
  map<Node *, Eigen::MatrixXd> gradients;

public:
  MSE(Node *activations, Node *reference);
  void forward();
  void backward();
  void getGradient(Node *n, Eigen::MatrixXd &grad);
};

MSE::MSE(Node *activations, Node *reference)
{
  this->addInput(activations);
  this->addInput(reference);
}
void MSE::forward()
{
  vector<Eigen::MatrixXd> inputs;
  inputs = getInputValues();

  Eigen::MatrixXd activations = inputs[0];
  Eigen::MatrixXd reference = inputs[1];

  if(!(activations.rows()==reference.rows() &&
       activations.cols()==reference.cols())) {
    throw invalid_argument("activation tensor and reference tensor must have the same shape");
  }
  if(!(activations.cols()== 1)) {
    throw invalid_argument("MSE operations can be done on vectors of shape (batchSize x 1)");
  }

  batchSize = activations.rows();

  double mse;
  diff = reference - activations;
  mse = diff.norm() / batchSize;

  setValues(mse);
}

void MSE::backward()
{
  Eigen::MatrixXd gradReference;
  Eigen::MatrixXd gradActivations;
  gradReference = (-2.0f / batchSize) * diff;
  gradActivations = (2.0f / batchSize) * diff;

  vector<Node *> inputs = getInputNodes();
  gradients[inputs[0]] = gradActivations;
  gradients[inputs[1]] = gradReference;
}

void MSE::getGradient(Node *n, Eigen::MatrixXd &grad)
{
  if(n) {
    grad = gradients[n];
  }
}

#endif
