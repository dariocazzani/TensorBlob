#ifndef LINEAR_H
#define LINEAR_H

#include "Node.h"
#include <string>
#include <map>
#include <stdexcept>

/*
 * NB: I am assuming that Linear Node has only ONE input Node from the graph.
 * weight and bias are added as input but are implicit of the Linear Node
 */

class Linear : public Node
{
private:
  // Store address of different inputs for easier use
  map<string, Node *> inputsMap;
  void validateInputs(const Eigen::MatrixXd &inputs,
                      const Eigen::MatrixXd &weights,
                      const Eigen::MatrixXd &bias);

public:
  Linear(Node *inputs,
         Node *weights,
         Node *bias);
  void getLinearInputs(Eigen::MatrixXd &values);
  void getLinearWeights(Eigen::MatrixXd &values);
  void getLinearBias(Eigen::MatrixXd &values);
  void forward();
  void backward();
};

void Linear::validateInputs(const Eigen::MatrixXd &inputs,
                            const Eigen::MatrixXd &weights,
                            const Eigen::MatrixXd &bias)
{
  if(!(bias.rows()==1)){
    throw invalid_argument("Bias should have 1 rows");
  }
  if(weights.rows() != inputs.cols()){
    throw invalid_argument("Mismatch between inputs and weights");
  }
  if(weights.cols() != bias.cols()){
    throw invalid_argument("Mismatch between bias and weights");
  }
}

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

  validateInputs(inputs, weights, bias);

  // Map bias to Vector for dynamic broadcasting
  Eigen::Map<Eigen::VectorXd> biasFlat(bias.data(), bias.size());

  // NB Equivalent to np.dot(inputs,W)+bias
  Eigen::MatrixXd value = (inputs * weights).transpose().colwise() + biasFlat;
  setValues(value.transpose());
}

void Linear::backward()
{
  vector<Node *> inputs = getInputNodes();
  vector<Node *> outputs = getOutputNodes();

  // # Initialize the gradients to 0.
  Eigen::MatrixXd tempGrad_0 = Eigen::MatrixXd::Zero(inputs[0]->getValuesRows(), inputs[0]->getValuesCols());
  Eigen::MatrixXd tempGrad_1 = Eigen::MatrixXd::Zero(inputs[1]->getValuesRows(), inputs[1]->getValuesCols());
  Eigen::MatrixXd tempGrad_2 = Eigen::MatrixXd::Zero(inputs[2]->getValuesRows(), inputs[2]->getValuesCols());

  Eigen::MatrixXd gradCost;
  Eigen::MatrixXd nodeValueTemp;
  for(auto n : outputs)
  {
    // Get gradient of outBound Node w.r.t. current node
    n->getGradients(this, gradCost);

    inputs[1]->getValues(nodeValueTemp);
    tempGrad_0 += gradCost * nodeValueTemp.transpose();

    inputs[0]->getValues(nodeValueTemp);
    tempGrad_1 += nodeValueTemp.transpose() * gradCost;

    tempGrad_2 += gradCost.colwise().sum();

  }
  setGradients(inputs[0], tempGrad_0);
  setGradients(inputs[1], tempGrad_1);
  setGradients(inputs[2], tempGrad_2);
}

#endif
