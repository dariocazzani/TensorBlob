#ifndef SGD_H
#define SGD_H

#include "../../include/Eigen/Dense"
#include "../nodes/Node.h"

#include <iostream>
#include <vector>
using namespace std;

void SGD(vector<Node *> &trainables, const double &learningRate)
{
  if(learningRate <= 0.0f)
  {
    throw invalid_argument("Learning rate should be a positive number.");
  }
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
