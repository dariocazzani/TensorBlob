#include "nodes/Node.h"
#include "nodes/Linear.h"
#include "nodes/Variable.h"
#include "nodes/Input.h"
#include "nodes/Sigmoid.h"
#include "nodes/MSE.h"
#include "nodes/SoftXent.h"
#include "graph/graph_utils.h"
#include "mnist/mnistUtils.h"
#include "optimizers/SGD.h"

const int NUM_HIDDEN = 128;
const int BATCH_SIZE = 256;
const int NUM_EPOCHS = 100;
const double LEARNING_RATE = 1e-2;

void getBatch(const Eigen::MatrixXd &data, const Eigen::MatrixXd &labels,
             Eigen::MatrixXd &dataBatch, Eigen::MatrixXd &datalabels);

int main()
{
  srand(42);

  //INPUTS and LABELS
  Eigen::MatrixXd trainData(TRAIN_SIZE, IMG_SIZE*IMG_SIZE);
  Eigen::MatrixXd validData(VALID_SIZE, IMG_SIZE*IMG_SIZE);
  Eigen::MatrixXd trainLabels(TRAIN_SIZE, NUM_CLASSES);
  Eigen::MatrixXd validLabels(VALID_SIZE, NUM_CLASSES);

  getMnistData(trainData, validData, trainLabels, validLabels);

  Eigen::MatrixXd inputs;
  Eigen::MatrixXd labels;


  Eigen::MatrixXd weights1 = Eigen::MatrixXd::Random(IMG_SIZE*IMG_SIZE, NUM_HIDDEN);

  Eigen::MatrixXd bias1 = Eigen::MatrixXd::Random(1, NUM_HIDDEN);

  Eigen::MatrixXd weights2 = Eigen::MatrixXd::Random(NUM_HIDDEN, NUM_CLASSES);

  Eigen::MatrixXd bias2 = Eigen::MatrixXd::Random(1, NUM_CLASSES);

  // DEFINE NODES AND CONNECT THEM
  Variable W1;
  Variable b1;
  Variable W2;
  Variable b2;
  Input X;
  Input Y;

  // BUILD GRAPH
  Linear hidden1(&X, &W1, &b1);
  Sigmoid outHidden1(&hidden1);
  Linear out(&outHidden1, &W2, &b2);
  SoftXent cost(&out, &Y);

  vector<Node *> graph = {&hidden1, &W1, &b1, &W2, &b2, &X, &outHidden1, &out, &Y, &cost};
  vector<Node *> trainables = {&W1, &b1, &W2, &b2};

  // FEED_DICT
  map<Node*, Eigen::MatrixXd> inputMap;
  W1.setValues(weights1);
  b1.setValues(bias1);
  W2.setValues(weights2);
  b2.setValues(bias2);

  buildGraph(graph);

  for(size_t i=0; i<NUM_EPOCHS; ++i)
  {
    getBatch(trainData, trainLabels, inputs, labels);
    inputMap[&X] = inputs;
    inputMap[&Y] = labels;

    feedValues(inputMap);

    vector<Eigen::MatrixXd> results = forwardBackward(graph);
    SGD(trainables, LEARNING_RATE);

    cout<<"Cost: "<<cost.getValues()<<endl;
  }
  return 0;
}

void getBatch(const Eigen::MatrixXd &data, const Eigen::MatrixXd &labels,
             Eigen::MatrixXd &dataBatch, Eigen::MatrixXd &labelsBatch)
{
  int idx = rand()% (TRAIN_SIZE - BATCH_SIZE+1);
  dataBatch = data.block<BATCH_SIZE, IMG_SIZE*IMG_SIZE>(idx,0);
  labelsBatch = labels.block<BATCH_SIZE, NUM_CLASSES>(idx,0);
}
