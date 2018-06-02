#ifndef INPUT_H
#define INPUT_H

#include "Node.h"

class Input : public Node
{
public:
  Input();
  Input(double value);
  Input(const Eigen::MatrixXd &values);
  void forward();
  void backward();
};

Input::Input() {}
Input::Input(double value)
{
  setValues(value);
}

Input::Input(const Eigen::MatrixXd &values)
{
  setValues(values);
}


void Input::forward() {}
void Input::backward() {}

#endif
