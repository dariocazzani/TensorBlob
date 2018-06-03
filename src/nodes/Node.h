#ifndef NODE_H
#define NODE_H

#include "../../include/Eigen/Dense"

#include <iostream>
#include <vector>
#include <map>
using namespace std;

class Node
{
private:
  Eigen::MatrixXd values;
  map<const Node *, Eigen::MatrixXd> gradients;
  vector<Node *> inNodes;
  vector<Node *> outNodes;
public:
  Node();
  Node(vector<Node *> &inNodes);
  void setValues(double value);
  void setValues(const Eigen::MatrixXd &values);
  void addInput(Node *input);
  void addOutput(Node *input);
  void setGradients(Node *n, const Eigen::MatrixXd &grad);
  void getGradients(const Node *n, Eigen::MatrixXd &grad);

  void getValues(Eigen::MatrixXd &values);
  vector<Eigen::MatrixXd> getInputValues();
  vector<Eigen::MatrixXd> getOutputValues();
  vector<Node *> getOutputNodes();
  vector<Node *> getInputNodes();
  virtual void forward();
  virtual void backward();
  void printValue();
};

Node::Node()
{
  this->values.resize(1,1);
  this->values << 0.0f;
}

Node::Node(vector<Node *> &inNodes)
{
  for (auto i: inNodes){
    this->addInput(i);
  }
}

void Node::setValues(double value)
{
  this->values.resize(1,1);
  this->values << value;
}

void Node::setValues(const Eigen::MatrixXd &values)
{
  this->values = values;
}

void Node::setGradients(Node *n, const Eigen::MatrixXd &grad)
{
  if(n) {
    gradients[n] = grad;
  }
}

void Node::getGradients(const Node *n, Eigen::MatrixXd &grad)
{
  if(n) {
    grad = gradients[n];
  }
}

void Node::getValues(Eigen::MatrixXd &values)
{
  values = this->values;
}

vector<Eigen::MatrixXd> Node::getInputValues()
{
  vector<Eigen::MatrixXd> vector_values;
  Eigen::MatrixXd values;

  for(auto n : inNodes) {
    n->getValues(values);
    vector_values.push_back(values);
  }
  return vector_values;
}

vector<Eigen::MatrixXd> Node::getOutputValues()
{
  vector<Eigen::MatrixXd> vector_values;
  Eigen::MatrixXd values;

  for(auto n : outNodes) {
    n->getValues(values);
    vector_values.push_back(values);
  }
return vector_values;
}

vector<Node *> Node::getOutputNodes()
{
  return outNodes;
}

vector<Node *> Node::getInputNodes()
{
  return inNodes;
}

void Node::addInput(Node *input)
{
  this->inNodes.push_back(input);
  input->outNodes.push_back(this);
}

void Node::addOutput(Node *output)
{
  this->outNodes.push_back(output);
  output->outNodes.push_back(this);
}

void Node::forward() {}

void Node::backward() {}

void Node::printValue()
{
  Eigen::MatrixXd values;
  getValues(values);
  cout<<"Value: " << values <<endl;
}

#endif
