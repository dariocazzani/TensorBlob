#include "nodes/Node.h"
#include "nodes/Linear.h"
#include "nodes/Input.h"
#include "nodes/Sigmoid.h"
#include "nodes/MSE.h"
#include "graph/graph_utils.h"

int main()
{
  srand(42);

  //INPUTS
  // 4 sample, 2 features
  Eigen::MatrixXd inputs(4, 2);
  inputs << -0.4, -0.3,
            -0.2, -0.1,
             0.1,  0.2,
             0.3,  0.4;
  // Labels
  Eigen::MatrixXd labels = Eigen::MatrixXd::Random(4,1);

  // 2 features, 3 hidden neurons
  Eigen::MatrixXd weights1 = Eigen::MatrixXd::Random(2,3);

  Eigen::MatrixXd bias1 = Eigen::MatrixXd::Random(1,3);

  // 1 output
  Eigen::MatrixXd weights2 = Eigen::MatrixXd::Random(3,1);

  Eigen::MatrixXd bias2 = Eigen::MatrixXd::Random(1,1);



  // DEFINE NODES AND CONNECT THEM
  Input W1;
  Input b1;
  Input W2;
  Input b2;
  Input X;
  Input Y;

  // BUILD GRAPH
  Linear hidden1(&X, &W1, &b1);
  Sigmoid outHidden1(&hidden1);
  Linear out(&outHidden1, &W2, &b2);
  MSE cost(&out, &Y);

  vector<Node *> graph = {&hidden1, &W1, &b1, &W2, &b2, &X, &outHidden1, &out, &Y, &cost};

  // FEED_DICT
  map<Node*, Eigen::MatrixXd> inputMap;
  inputMap[&W1] = weights1;
  inputMap[&b1] = bias1;
  inputMap[&W2] = weights2;
  inputMap[&b2] = bias2;
  inputMap[&X] = inputs;
  inputMap[&Y] = labels;

  buildGraph(graph, inputMap);

  vector<Eigen::MatrixXd> results = forwardBackward(graph);
  for(auto r : results){
    cout<<r<<endl;
  }
  return 0;
}
