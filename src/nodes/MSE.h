#ifndef MSE_H
#define MSE_H

#include "Node.h"

// NB MSE Has exactly 2 inputs Nodes
class MSE : public Node
{
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

  double mse;
  mse = (activations - reference).norm() / activations.size();

  setValues(mse);
}

void MSE::backward() {}

#endif
