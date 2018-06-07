#ifndef SoftXent_H
#define SoftXent_H

#include "Node.h"
// NB Softmax cross entropy with logits - 2 inputs and 0 outputs
class SoftXent : public Node
{
private:
  double batchSize = {0.0f};
  int numClasses = {0};
  Eigen::MatrixXd probabilities;

  void softmaxProbs(Eigen::MatrixXd &probabilities, const Eigen::MatrixXd &logits);
  double xentLoss(const Eigen::MatrixXd &probabilities, const Eigen::MatrixXd &labels);

public:
  SoftXent(Node *logits, Node *labels);

  void forward();
  void backward();
};

SoftXent::SoftXent(Node *logits, Node *labels)
{
  this->addInput(logits);
  this->addInput(labels);
}

void SoftXent::softmaxProbs(Eigen::MatrixXd &probabilities, const Eigen::MatrixXd &logits)
{
  /*
   * Stable computation fot softmax for a 2D matrix
   * Refer to: https://www.deeplearningbook.org/contents/numerical.html
   * Computed rowwise - each row is a sample
  */

  if(!(logits.rows() == batchSize) ||
     !(logits.cols() == numClasses))
  {
    throw invalid_argument("Logits matrix does not match batch size or number of classes");
  }

  Eigen::MatrixXd norm;
  Eigen::MatrixXd exps;
  Eigen::VectorXd sums;

  norm = logits.colwise() - logits.rowwise().maxCoeff();
  exps = norm.array().exp();
  sums = exps.rowwise().sum();
  probabilities = exps.array().colwise() / sums.array();
}

double SoftXent::xentLoss(const Eigen::MatrixXd &probabilities, const Eigen::MatrixXd &labels)
{
  //Log loss is undefined for p=0 or p=1, so probabilities are
  // clipped to max(epsilon, min(1 - epsilon, p)).

  double epsilon = {1e-15};
  Eigen::MatrixXd clippedPropbs = (probabilities.cwiseMin(1.0 - epsilon)).cwiseMax(epsilon);
  Eigen::MatrixXd logLikelihood = (clippedPropbs.array().log()) * labels.array() * (-1);
  double xent = logLikelihood.sum() / batchSize;
  return xent;
}

void SoftXent::forward()
{
  vector<Eigen::MatrixXd> inputs;
  inputs = getInputValues();

  Eigen::MatrixXd logits = inputs[0];
  Eigen::MatrixXd labels = inputs[1];

  if(!(logits.rows()==labels.rows() &&
       logits.cols()==labels.cols()))
  {
    throw invalid_argument("activation tensor and labels tensor must have the same shape");
  }

  batchSize = logits.rows();
  numClasses = logits.cols();

  softmaxProbs(probabilities, logits);
  double loss = xentLoss(probabilities, labels);

  setValues(loss);
}

void SoftXent::backward()
{
  vector<Eigen::MatrixXd> inputs = getInputValues();
  Eigen::MatrixXd labels = inputs[1];
  Eigen::MatrixXd gradientLogits;

  gradientLogits = (probabilities.array() - labels.array()) / batchSize;
  setGradients(getInputNodes()[0], gradientLogits);
  // Gradients for labels are useless, but included for compatibility with
  // the fact that labels are of type Input (which can perform backward prop)
  // In reality these gradients will be ignored
  setGradients(getInputNodes()[1], gradientLogits);
}

#endif
