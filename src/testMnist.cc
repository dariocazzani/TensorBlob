#include "mnist/mnistUtils.h"
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
  Eigen::MatrixXd trainData(TRAIN_SIZE, IMG_SIZE*IMG_SIZE);
  Eigen::MatrixXd validData(VALID_SIZE, IMG_SIZE*IMG_SIZE);
  Eigen::MatrixXd trainLabels(TRAIN_SIZE, NUM_CLASSES);
  Eigen::MatrixXd validLabels(VALID_SIZE, NUM_CLASSES);

	getData(trainData, validData, trainLabels, validLabels);
  cout<<validLabels<<endl;
  return 0;
}
