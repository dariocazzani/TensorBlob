#ifndef INPUT_H
#define INPUT_H

#include "Node.h"

class Input : public Node
{
public:
  Input(double value);
  void forward(double value);
};

Input::Input(double value)
{
  setValue(value);
}
void Input::forward(double value)
{
  setValue(value);
}

#endif
