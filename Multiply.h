#ifndef MULTIPLY_H
#define MULTIPLY_H

#include "Node.h"

class Multiply : public Node
{
public:
  Multiply();
  Multiply(vector<Node *> &inNodes);
  void forward();
};

Multiply::Multiply()
{
  Node();
}
Multiply::Multiply(vector<Node *> &inNodes)
{
  for (auto i: inNodes){
    this->addInput(i);
  }
}

void Multiply::forward()
{
  vector<double> in = this->getInputValues();
  double mul = 1.0f;
  for (auto i : in){
    mul *= i;
  }
  setValue(mul);
}

#endif
