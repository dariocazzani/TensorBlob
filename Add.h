#ifndef ADD_H
#define ADD_H

#include "Node.h"

class Add : public Node
{
public:
  Add();
  Add(vector<Node *> &inNodes);
  void forward();
};

Add::Add()
{
  Node();
}
Add::Add(vector<Node *> &inNodes)
{
  for (auto i: inNodes){
    this->addInput(i);
  }
}

void Add::forward()
{
  vector<double> in = this->getInputValues();
  double sum = 0.0f;
  for (auto i : in){
    sum += i;
  }
  setValue(sum);
}

#endif
