#include "Node.h"
#include "Input.h"
#include "Add.h"
#include "Multiply.h"

#include "graph_utils.h"
// void runGraph(vector<Node *> & g);//, const vector<double> &inputs);

int main()
{
  Input n1;
  Input n2;
  Input n3;
  vector<Node *> inputs = {&n1, &n2, &n3};
  Add a(inputs);
  vector<Node *> v = {&a};
  Multiply m(v);

  vector<Node *> graph = {&m, &n1, &n2, &a, &n3};
  buildGraph(graph);
  cout<<graph.size()<<endl;

  return 0;
}



// void runGraph(vector<Node *> & g)//, const vector<double> &inputs)
// {
//   for(auto n : g){
//     n->forward();
//   }
//   g.back()->printValue();
// }
