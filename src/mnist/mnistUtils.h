#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnistLoader.h"
#include "../../include/Eigen/Dense"

#include <iostream>
#include <vector>

using namespace std;

const int TRAIN_SIZE = 50000;
const int VALID_SIZE = 10000;
const int NUM_CLASSES = 10;
const int IMG_SIZE = 28;


void toOneHot(const unsigned int &label, Eigen::VectorXd &oneHot);

void toOneHot(const unsigned int &label, Eigen::VectorXd &oneHot)
{
  Eigen::VectorXd tmp = Eigen::VectorXd::Zero(NUM_CLASSES);
  oneHot = tmp;
  oneHot(label) = 1.0f;
}

void getData(Eigen::MatrixXd &trainData, Eigen::MatrixXd &validData,
  Eigen::MatrixXd &trainLabels, Eigen::MatrixXd &validLabels);

void getData(Eigen::MatrixXd &trainData, Eigen::MatrixXd &validData,
  Eigen::MatrixXd &trainLabels, Eigen::MatrixXd &validLabels)
{
  mnist_data *data;
  unsigned int cnt;
  int ret;

  const char *trainImagesPath = "../data/train-images-idx3-ubyte";
  const char *trainLabelsPath = "../data/train-labels-idx1-ubyte";

  if ((ret = mnist_load(trainImagesPath, trainLabelsPath, &data, &cnt)))
  {
    cout<<"An error occured: "<<ret<<endl;
  switch (ret) {
    	case -1:
      throw runtime_error("Could not open file");
      break;
  case -2:
  throw runtime_error("Not a valid image file");
  break;
  case -3:
  throw runtime_error("Not a valid label file");
  break;
  default:
  throw runtime_error("WTF: Something very bad happened");
  }
  }

  Eigen::VectorXd oneHot(NUM_CLASSES);
  for (int k = 0; k < (TRAIN_SIZE + VALID_SIZE); k++)
  {
  for (int i = 0; i < IMG_SIZE; i++)
  {
  for (int j = 0; j < IMG_SIZE; j++)
  {
  if (k < TRAIN_SIZE)
  {
  trainData(k, i * IMG_SIZE + j) = data[k].data[i][j];
  }
  else
  {
  validData(k-TRAIN_SIZE, i * IMG_SIZE + j) = data[k].data[i][j];
  }
  }
  }

  if (k < TRAIN_SIZE)
  {
  toOneHot(data[k].label, oneHot);
  trainLabels.row(k) = oneHot;
  }
  else
  {
  toOneHot(data[k].label, oneHot);
  validLabels.row(k - TRAIN_SIZE) = oneHot;
  }
  }
  free(data);
}
