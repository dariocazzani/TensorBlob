#ifndef INPUT_H
#define INPUT_H

#include "Node.h"

class Input : public Node
{
public:
  Input();
  Input(double value);
  Input(const Eigen::VectorXd &values);
  void forward();
};

Input::Input() {}
Input::Input(double value)
{
  setValues(value);
}

Input::Input(const Eigen::VectorXd &values)
{
  setValues(values);
}


void Input::forward() {}

#endif
