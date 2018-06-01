#include "nodes/Node.h"
#include "nodes/Linear.h"
#include "nodes/Input.h"
#include "graph/graph_utils.h"

int main()
{
  Eigen::MatrixXd weights;
  // 2 features, 3 hidden neurons
  weights.resize(2, 3);
  weights << 1, 2, 3,
             4, 5, 6;
  Input W;

  Eigen::MatrixXd inputs;
  // 4 sample, 2 features
  inputs.resize(4, 2);
  inputs << 1, 2,
            3, 4,
            5, 6,
            7, 8;

  Input X;

  Eigen::MatrixXd bias;
  bias.resize(1, 3);
  bias << 1, 2, 3;

  Input b;

  Linear hidden1(&X, &W, &b);

  vector<Node *> graph = {&hidden1, &W, &b, &X};
  buildGraph(graph);

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
