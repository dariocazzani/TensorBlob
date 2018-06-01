#ifndef LINEAR_H
#define LINEAR_H

#include "Node.h"
#include <string>
#include <map>

/*
 * NB: I am assuiming that Linear Node has only ONE input Node from the graph.
 * weight and bias are added as input but are implicit of the Linear Node
 */

class Linear : public Node
{
private:
  // Store address of different inputs for easier use
  map<string, Node *> inputsMap;

public:
  Linear(Node *inputs,
         Node *weights,
         Node *bias);
  void getLinearInputs(Eigen::MatrixXd &values);
  void getLinearWeights(Eigen::MatrixXd &values);
  void getLinearBias(Eigen::MatrixXd &values);
  void forward();
};

Linear::Linear(Node *inputs,
               Node *weights,
               Node *bias){

  inputsMap["inputs"] = inputs;
  inputsMap["weights"] = weights;
  inputsMap["bias"] = bias;
  this->addInput(inputs);
  this->addInput(weights);
  this->addInput(bias);
}

void Linear::getLinearInputs(Eigen::MatrixXd &values) {
  inputsMap["inputs"]->getValues(values);
}

void Linear::getLinearWeights(Eigen::MatrixXd &values) {
  inputsMap["weights"]->getValues(values);
}

void Linear::getLinearBias(Eigen::MatrixXd &values) {
  inputsMap["bias"]->getValues(values);
}

void Linear::forward()
{
  Eigen::MatrixXd inputs;
  Eigen::MatrixXd weights;
  Eigen::MatrixXd bias;
  getLinearInputs(inputs);
  getLinearWeights(weights);
  getLinearBias(bias);

  Eigen::MatrixXd value = inputs * weights + bias;
  setValues(value);
}
#endif
