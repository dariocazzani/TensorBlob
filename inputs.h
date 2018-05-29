#ifndef INPUTS_H
#define INPUTS_H

#include "Input.h"

void initialize_inputs(vector<double> * values, vector<Node> * inputs);

void initialize_inputs(vector<double> * values, vector<Node> * inputs)
{
  for(auto it = values->begin(); it != values->end(); ++it) {
    Input temp(*it);
    inputs->push_back(temp);
  }
}

#endif
