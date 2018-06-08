#include "nodes/Node.h"
#include "nodes/Linear.h"
#include "nodes/Input.h"
#include "nodes/Variable.h"
#include "nodes/Sigmoid.h"
#include "nodes/MSE.h"
#include "nodes/SoftXent.h"
#include "graph/graph_utils.h"
#include "mnist/mnistUtils.h"
#include "optimizers/SGD.h"

#include <iomanip>

const int NUM_HIDDEN = 32;
const int BATCH_SIZE = 256;
const int NUM_EPOCHS = 4;
constexpr int NUM_ITERATIONS = (TRAIN_SIZE / BATCH_SIZE + 1) * NUM_EPOCHS;
const double LEARNING_RATE = 1e-1;

void getBatch(const Eigen::MatrixXd &data, const Eigen::MatrixXd &labels,
             Eigen::MatrixXd &dataBatch, Eigen::MatrixXd &datalabels);

// Return row-wise argmax
Eigen::VectorXd argMax(const Eigen::MatrixXd &data);

double getAccuracy(const Eigen::VectorXd &truth, const Eigen::VectorXd &pred);

int main()
{
  srand(42);

  //INPUTS and LABELS
  Eigen::MatrixXd trainData(TRAIN_SIZE, IMG_SIZE*IMG_SIZE);
  Eigen::MatrixXd validData(VALID_SIZE, IMG_SIZE*IMG_SIZE);
  Eigen::MatrixXd trainLabels(TRAIN_SIZE, NUM_CLASSES);
  Eigen::MatrixXd validLabels(VALID_SIZE, NUM_CLASSES);

  getMnistData(trainData, validData, trainLabels, validLabels);

  // VARIABLES
  Eigen::MatrixXd weights1(IMG_SIZE*IMG_SIZE, NUM_HIDDEN);
  Eigen::MatrixXd bias1(1, NUM_HIDDEN);
  Eigen::MatrixXd weights2(NUM_HIDDEN, NUM_CLASSES);
  Eigen::MatrixXd bias2(1, NUM_CLASSES);

  // DEFINE NODES
  Variable W1(weights1);
  Variable b1(bias1);
  Variable W2(weights2);
  Variable b2(bias2);
  Input X;
  Input Y;

  // TRAINABLES AND INIT
  vector<Node *> trainables = {&W1, &b1, &W2, &b2};
  initTrainables(trainables);

  // BUILD GRAPH
  Linear hidden1(&X, &W1, &b1);
  Sigmoid outHidden1(&hidden1);
  Linear out(&outHidden1, &W2, &b2);
  SoftXent cost(&out, &Y);

  // Connect all nodes to graph and define vector of trainable variables
  vector<Node *> graph = {&hidden1, &W1, &b1, &W2, &b2, &X, &outHidden1, &out, &Y, &cost};
  buildGraph(graph);

  // Train Batch
  Eigen::MatrixXd trainDataBatch;
  Eigen::MatrixXd trainLabelBatch;
  // Validation Batch
  Eigen::MatrixXd validDataBatch;
  Eigen::MatrixXd validLabelBatch;

  // costs
  Eigen::MatrixXd trainCost;
  Eigen::MatrixXd validCost;
  Eigen::VectorXd trueLabels;
  Eigen::VectorXd predictedLabels;

  // Accuracy
  double trainAccuracy = 0.0f;
  double validAccuracy = 0.0f;

  // FEED_DICT
  map<Node*, Eigen::MatrixXd> inputMap;
  for(size_t i=0; i<NUM_ITERATIONS; ++i)
  {
    //train
    getBatch(trainData, trainLabels, trainDataBatch, trainLabelBatch);
    inputMap[&X] = trainDataBatch;
    inputMap[&Y] = trainLabelBatch;
    feedValues(inputMap);
    forwardBackward(graph);
    SGD(trainables, LEARNING_RATE);
    trainCost = cost.getValues();
    trainAccuracy = getAccuracy(argMax(trainLabelBatch), argMax(cost.getProbabilities()));

    //validate
    getBatch(validData, validLabels, validDataBatch, validLabelBatch);
    inputMap[&X] = validDataBatch;
    inputMap[&Y] = validLabelBatch;
    feedValues(inputMap);
    forward(graph);
    validCost = cost.getValues();
    validAccuracy = getAccuracy(argMax(validLabelBatch), argMax(cost.getProbabilities()));

    if(i % 100 == 0)
    {
      cout<<"*********\nIteration: "<<i+1<<endl;
      cout<<setprecision(5)<<"Training cost:   "<<trainCost;
      cout<<setprecision(5)<<" - Training accuracy:   "<<trainAccuracy*100<<"% "<<endl;
      cout<<setprecision(5)<<"Validation cost: "<<validCost;
      cout<<setprecision(5)<<" - Validation accuracy: "<<validAccuracy*100<<"% "<<endl<<endl;
    }
  }

  cout<<"*********\nFinal Results: "<<endl;
  cout<<setprecision(5)<<"Training cost:   "<<trainCost;
  cout<<setprecision(5)<<" - Training accuracy:   "<<trainAccuracy*100<<"% "<<endl;
  cout<<setprecision(5)<<"Validation cost: "<<validCost;
  cout<<setprecision(5)<<" - Validation accuracy: "<<validAccuracy*100<<"% "<<endl<<endl;
  return 0;
}

void getBatch(const Eigen::MatrixXd &data, const Eigen::MatrixXd &labels,
             Eigen::MatrixXd &dataBatch, Eigen::MatrixXd &labelsBatch)
{
  const int dataSize = data.rows();
  int idx = rand()% (dataSize - BATCH_SIZE+1);
  dataBatch = data.block<BATCH_SIZE, IMG_SIZE*IMG_SIZE>(idx,0);
  labelsBatch = labels.block<BATCH_SIZE, NUM_CLASSES>(idx,0);
}

Eigen::VectorXd argMax(const Eigen::MatrixXd &data)
{
  unsigned int rows = data.rows();
  unsigned int cols = data.cols();
  Eigen::VectorXd results(rows);

  for(size_t i=0; i<rows; ++i)
  {
    double maxTmp = 0.0f;
    for(size_t j=0; j<cols; ++j)
    {
      if(data(i, j) > maxTmp)
      {
        maxTmp = data(i, j);
        results(i) = j;
      }
    }
  }
  return results;
}


double getAccuracy(const Eigen::VectorXd &truth, const Eigen::VectorXd &pred)
{
  unsigned int numElems = truth.size();
  double correct = 0.0f;
  for(size_t i=0; i<numElems; ++i)
  {
    if(truth(i) == pred(i))
    {
      correct += 1.0f;
    }
  }
  return correct/numElems;
}
