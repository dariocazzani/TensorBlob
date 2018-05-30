#ifndef NODE_H
#define NODE_H

#include<iostream>
#include<vector>

using namespace std;

class Node
{
private:
  double value;
  vector<Node *> inNodes;
  vector<Node *> outNodes;
public:
  Node();
  void setValue(double value);
  void addInput(Node *input);
  void addOutput(Node *input);

  double getValue();
  vector<double> getInputValues();
  vector<double> getOutputValues();
  void forward();
  void printValue();
};

Node::Node()
{
  this->value = 0.0f;
}
void Node::setValue(double value)
{
  this->value = value;
}

double Node::getValue()
{
  return value;
}

vector<double> Node::getInputValues()
{
  vector<double> values;
  for(auto n : inNodes)
    values.push_back(n->getValue());
  return values;
}

vector<double> Node::getOutputValues()
{
  vector<double> values;
  for(auto n : outNodes)
    values.push_back(n->getValue());
  return values;
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
  cout<<"Value: " << getValue() <<endl;
}

#endif
