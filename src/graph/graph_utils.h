#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include <map>
#include <queue>
#include <stdexcept>
#include "DAGException.h"

/*
 * Use Kahn's algorithm to sort nodes
 * Sort Graph so that we can run forward (and backward) propagation
 */
void buildGraph(vector<Node *> & g);

/*
 * inputMap is loosely inspired by TensorFlow feed_dict
 * Run Forward propagation given a computation graph and the values to assign to
 * the inputs
 */
vector<double> forwardProp(vector<Node *> graph, map<Node*, double> inputMap);




void buildGraph(vector<Node *> & g)
/*
https://www.geeksforgeeks.org/topological-sorting-indegree-based-solution/
*/
{
  int visitedNodes = 0;
  /*
   * Compute in-degree (number of incoming edges) for each
   * of the vertex present in the DAG
   */
  map <Node*, int> inNodesCount;
  vector<double> inputs;
  for(auto n : g){
    inputs = n->getInputValues();
    inNodesCount.insert(pair <Node*, int> (n, inputs.size()));
  }

  /*
   * Pick all the vertices with in-degree as 0 and add
   * them into a queue (Enqueue operation)
   */
  queue<Node *> q;
  map<Node*, int>::iterator it = inNodesCount.begin();
  while(it != inNodesCount.end()){
    if(it->second == 0){
      q.push(it->first);
    }
    ++it;
  }
  /* Remove a vertex from the queue (Dequeue operation) and then.
   *
   * Increment count of visited nodes by 1.
   * Decrease in-degree by 1 for all its neighboring nodes.
   * If in-degree of a neighboring nodes is reduced to zero, then add it to the queue.

   */
  Node *current;
  vector<Node *> neighbours;

  vector<Node *> temp;
  while(!q.empty()){
    current = q.front();
    temp.push_back(current);
    q.pop();

    //Increment count of visited nodes by 1.
    ++visitedNodes;

    //Decrease in-degree by 1 for all its neighboring nodes.
    neighbours = current->getOutputNodes();
    int inDegree = 0;
    for(auto n : neighbours){
      --inNodesCount.find(n)->second;
      //If in-degree of a neighboring nodes is reduced to zero, then add it to the queue.
      inDegree = inNodesCount.find(n)->second;
      if(inDegree==0){
        q.push(n);
      }
    }
  }
  if(visitedNodes <static_cast<int>(g.size())){
    throw DAGException();
  }
  temp.swap(g);
}


vector<double> forwardProp(vector<Node *> graph, map<Node*, double> inputMap)
{
  vector<double> results;

  // Assign the desired values to the inputs
  map<Node*, double>::iterator it = inputMap.begin();
  while(it != inputMap.end()){
    // Verify that we were given only Input nodes to assign values to
    if(Input* b1 = dynamic_cast<Input*> (it->first)){
      it->first->setValue(it->second);
    }
    else{
      throw("Invalid Input type.");
    }
    ++it;
  }
  for(auto n : graph){
    n->forward();
  }

  // Find output nodes
  for(auto n : graph){
    if(n->getOutputNodes().size() == 0){
      results.push_back(n->getValue());
    }
  }
  return results;
}

#endif
