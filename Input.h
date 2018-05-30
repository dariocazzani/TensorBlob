#ifndef INPUT_H
#define INPUT_H

#include "Node.h"

class Input : public Node
{
public:
  Input();
  Input(double value);
  void forward();
};

Input::Input() {}
Input::Input(double value)
{
  setValue(value);
}

void Input::forward() {}

#endif
