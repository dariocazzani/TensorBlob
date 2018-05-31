#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include <map>
#include <queue>
#include <stdexcept>
#include "DAGException.h"

// Use Kahn's algorithm to sort nodes
void buildGraph(vector<Node *> & g);

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

#endif
