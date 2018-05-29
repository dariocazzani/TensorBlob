#ifndef ADD_H
#define ADD_H

#include "Node.h"

class Add : public Node
{
public:
  Add(vector<Node> inputs);
  void forward();
};

Add::Add(vector<Node> inputs)
{
  setInbounds(inputs);
}

void Add::forward()
{
  vector<Node> * inbounds = getInbounds();
  double sum = 0.0f;
  for(auto it = inbounds->begin(); it != inbounds->end(); ++it) {
    sum += it->getValue();
  }
  setValue(sum);
}

#endif
