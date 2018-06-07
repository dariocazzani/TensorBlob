#ifndef SGD_H
#define SGD_H

#include "../../include/Eigen/Dense"
#include "../nodes/Node.h"

#include <iostream>
#include <vector>
using namespace std;

void SGD(vector<Node *> &trainables, const double &learningRate)
{
  Eigen::MatrixXd gradTemp;
  Eigen::MatrixXd tempValues;
  for(auto n : trainables)
  {
    n->getGradients(n, gradTemp);
    n->getValues(tempValues);
    tempValues.array() -= (gradTemp.array() * learningRate);
    n->setValues(tempValues);
  }
}

#endif
