#ifndef NODE_H
#define NODE_H

#include "../../include/Eigen/Dense"

#include<iostream>
#include<vector>

using namespace std;

class Node
{
private:
  Eigen::VectorXd values;
  vector<Node *> inNodes;
  vector<Node *> outNodes;
public:
  Node();
  Node(vector<Node *> &inNodes);
  void setValues(double value);
  void setValues(const Eigen::VectorXd &values);
  void addInput(Node *input);
  void addOutput(Node *input);

  void getValues(Eigen::VectorXd &values);
  vector<Eigen::VectorXd> getInputValues();
  vector<Eigen::VectorXd> getOutputValues();
  vector<Node *> getOutputNodes();
  virtual void forward();
  void printValue();
};

Node::Node()
{
  this->values.resize(1);
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
  this->values.resize(1);
  this->values << value;
}

void Node::setValues(const Eigen::VectorXd &values)
{
  this->values = values;
}

void Node::getValues(Eigen::VectorXd &values)
{
  values = this->values;
}

vector<Eigen::VectorXd> Node::getInputValues()
{
  vector<Eigen::VectorXd> vector_values;
  Eigen::VectorXd values;

  for(auto n : inNodes) {
    n->getValues(values);
    vector_values.push_back(values);
  }
  return vector_values;
}

vector<Eigen::VectorXd> Node::getOutputValues()
{
  vector<Eigen::VectorXd> vector_values;
  Eigen::VectorXd values;

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

void Node::printValue()
{
  Eigen::VectorXd values;
  getValues(values);
  cout<<"Value: " << values <<endl;
}

#endif
