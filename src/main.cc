#include "nodes/Node.h"
#include "nodes/Linear.h"
#include "nodes/Input.h"
#include "graph/graph_utils.h"

int main()
{
  //INPUTS

  // 2 features, 3 hidden neurons
  Eigen::MatrixXd weights(2, 3);
  // weights.resize(2, 3);
  weights << 1, 2, 3,
             4, 5, 6;

  // 4 sample, 2 features
  Eigen::MatrixXd inputs(4, 2);
  inputs << 1, 2,
            3, 4,
            5, 6,
            7, 8;

  Eigen::MatrixXd bias(1,3);
  bias << 1, 2, 3;


  // DEFINE NODES AND CONNECT THEM
  Input W;
  Input X;
  Input b;

  // BUILD GRAPH
  Linear hidden1(&X, &W, &b);
  vector<Node *> graph = {&hidden1, &W, &b, &X};
  buildGraph(graph);

  // FEED_DICT
  map<Node*, Eigen::MatrixXd> inputMap;
  inputMap[&W] = weights;
  inputMap[&b] = bias;
  inputMap[&X] = inputs;


  vector<Eigen::MatrixXd> results = forwardProp(&graph, inputMap);
  for(auto r : results){
    cout<<r<<endl;
  }
  return 0;
}
