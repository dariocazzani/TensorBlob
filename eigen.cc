#include "include/Eigen/Dense"

#include <iostream>

using namespace std;
int main()
{
  Eigen::MatrixXd mat(2,4);
  Eigen::MatrixXd v(1,4);
  mat << 1, 2, 6, 9,
         3, 1, 7, 2;
  v << 0,
       1,
       2,
       3;
  assert(v.rows() == 1);


  Eigen::Map<Eigen::VectorXd> v_vec(v.data(),v.size());


  //add v to each column of m
  mat.transpose().colwise() += v_vec;

  std::cout << "Broadcasting result: " << std::endl;
  std::cout << mat << std::endl;
}
