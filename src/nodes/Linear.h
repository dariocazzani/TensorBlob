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
  map<string, vector<Node *>> inputsMap;

public:
  Linear(Node *inputs,
         Node *weights,
         Node *bias);
  void getLinearInputs(Eigen::VectorXd &values);
  void getLinearWeights(Eigen::VectorXd &values);
  void getLinearBias(Eigen::VectorXd &values);
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

void Linear::getLinearInputs(Eigen::VectorXd &values) {
  inputsMap["inputs"]->getValues(values);
}

void Linear::getLinearWeights(Eigen::VectorXd &values) {
  inputsMap["weights"]->getValues(values);
}

void Linear::getLinearBias(Eigen::VectorXd &values) {
  inputsMap["bias"]->getValues(values);
}

void Linear::forward()
{
  Eigen::VectorXd = inputs;
  Eigen::VectorXd = weights;
  Eigen::VectorXd = bias;
  getLinearInputs(inputs);
  getLinearWeights(weights);
  getLinearBias(bias);

  VectorXd value = inputs.dot(weights) + bias;
  setValue(value);
}
#endif
