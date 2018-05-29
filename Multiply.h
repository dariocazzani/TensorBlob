#ifndef MULTIPLY_H
#define MULTIPLY_H

#include "Node.h"

class Multiply : public Node
{
public:
  Multiply(vector<Node> inputs);
  void forward();
};

Multiply::Multiply(vector<Node> inputs)
{
  setInbounds(inputs);
}

void Multiply::forward()
{
  vector<Node> * inbounds = getInbounds();
  double mul = 1.0f;
  for(auto it = inbounds->begin(); it != inbounds->end(); ++it) {
    mul *= it->getValue();
  }
  setValue(mul);
}

#endif
