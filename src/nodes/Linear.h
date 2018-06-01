#ifndef LINEAR_H
#define LINEAR_H

#include "Node.h"
#include <map>
#include <string>


class Linear : public Node
{
private:
  map<string, vector<Node *>> matrixMap;

public:
  Linear(vector<Node *> &inputs,
         vector<Node *> &weights,
         vector<Node *> &bias);
  vector<double> getLinearInputs();
  vector<double> getLinearWeights();
  vector<double> getLinearBias();
  void forward();
};

Linear::Linear(vector<Node *> &inputs,
               vector<Node *> &weights,
               vector<Node *> &bias){
  matrixMap["inputs"] = inputs;
  matrixMap["weights"] = weights;
  matrixMap["bias"] = bias;

  for (auto i: inputs){
    this->addInput(i);
  }
  for (auto i: weights){
    this->addInput(i);
  }
  for (auto i: bias){
    this->addInput(i);
  }

}

vector<double> Linear::getLinearInputs() {
  vector<double> values;
  for(auto i : matrixMap["inputs"]) {
    values.push_back(i->getValue());
  }
  return values;
}

vector<double> Linear::getLinearWeights() {
  vector<double> values;
  for(auto i : matrixMap["weights"]) {
    values.push_back(i->getValue());
  }
  return values;
}

vector<double> Linear::getLinearBias() {
  vector<double> values;
  for(auto i : matrixMap["bias"]) {
    values.push_back(i->getValue());
  }
  return values;
}


void Linear::forward()
{
  vector<double> i = getLinearInputs();
  double* ptr = &i[0];
  Map<VectorXd> eigenI(ptr, i.size());

  vector<double> w = getLinearWeights();
  ptr = &w[0];
  Map<VectorXd> eigenW(ptr, w.size());

  vector<double> b = getLinearBias();
  ptr = &b[0];
  Map<VectorXd> eigenB(ptr, b.size());

  VectorXd value = eigenI.dot(eigenW) + eigenB;
  setValue(value);
  // setValue(eigenI.dot(eigenW) + eigenB);

  //
  // double* ptr_data = &i[0];
  // Eigen::VectorXf eigenI = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(i.data(), i.size());
  //
  // vector<double> in = this->getInputValues();
  // double sum = 0.0f;
  // for (auto i : in){
  //   sum += i;
  // }
  // setValue(sum);
}
#endif
