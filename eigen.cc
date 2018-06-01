#include "include/Eigen/Dense"

#include <iostream>

void setValues(Eigen::VectorXd &values);
void copy(const Eigen::VectorXd &values);

using namespace std;
int main()
{
  Eigen::MatrixXd weights;
  // 2 features, 3 hidden neurons
  weights.resize(3, 2);
  weights << 1, 4,
             2, 5,
             3, 6;

  Eigen::MatrixXd inputs;
  // 1 sample, 2 features
  inputs.resize(1, 2);
  inputs << 1, 2;

  Eigen::MatrixXd bias;
  bias.resize(3, 1);
  bias << 1, 2, 3;

  /*
   * Equivalent to:
   * np.dot(weights, inputs) + bias
   */
   
  cout<<"Result: \n"<<weights * inputs.transpose() + bias<<endl;
  return 0;
}

void setValues(Eigen::VectorXd &values) {
  values.resize(2);
  values << 3.0, 4.0;
}

void copy(const Eigen::VectorXd &values) {
  Eigen::VectorXd values2;
  values2 = values;
  std::cout<<values2<<std::endl;
}
