#ifndef NODE_H
#define NODE_H

#include<iostream>
#include<vector>

using namespace std;

class Node
{
private:
  double value;
  vector<Node> inbound_nodes;
  vector<Node> outbound_nodes;
public:
  Node();
  void setValue(double value);
  double getValue();
  void setInbounds(vector<Node> input_nodes);
  vector<Node> * getInbounds();
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
  return this->value;
}

void Node::setInbounds(vector<Node> input_nodes)
{
  for(auto it = input_nodes.begin(); it != input_nodes.end(); ++it){
    inbound_nodes.push_back(*it);
  }
}
vector<Node> * Node::getInbounds()
{
  return &inbound_nodes;
}
void Node::forward() {}

void Node::printValue()
{
  cout<<"Value: " << getValue() <<endl;
}

#endif
