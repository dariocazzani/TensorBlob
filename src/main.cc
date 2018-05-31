#include "nodes/Node.h"
#include "nodes/Add.h"
#include "nodes/Multiply.h"
#include "nodes/Input.h"

#include "graph/graph_utils.h"


int main()
{
  Input i1;
  Input i2;
  Add a;
  a.addInput(&i1);
  a.addInput(&i2);
  Multiply m;
  m.addInput(&i1);
  m.addInput(&i2);
  vector<Node *> graph = {&i1, &i2, &a, &m};

  buildGraph(graph);

  map<Node*, double> inputMap;
  cout<<"Input 1? : ";
  cin>>inputMap[&i1];
  cout<<"Input 2? : ";
  cin>>inputMap[&i2];

  vector<double> results = forwardProp(graph, inputMap);

  cout<<"Output of computation graph: \n";
  for(auto r : results){
    cout<<r<<" - ";
  }
  cout<<endl;
}
