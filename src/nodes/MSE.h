#ifndef MSE_H
#define MSE_H

#include "Node.h"
// NB MSE Has exactly 2 inputs Nodes and 0 output Nodes
class MSE : public Node
{
private:
  double batchSize = {0.0f};
  Eigen::MatrixXd diff;

public:
  MSE(Node *activations, Node *reference);
  void forward();
  void backward();
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
  gradReference = (2.0f / batchSize) * diff;
  gradActivations = (-2.0f / batchSize) * diff;

  vector<Node *> inputs = getInputNodes();
  setGradients(inputs[0], gradActivations);
  setGradients(inputs[1], gradReference);
}

#endif
