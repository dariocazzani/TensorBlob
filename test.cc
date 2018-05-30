#include "Node.h"
#include "Input.h"
#include "Add.h"
#include "Multiply.h"
// #include "inputs.h"

int main()
{

  Node n1;
  n1.setValue(1.0);
  n1.printValue();

  Node n2;
  n2.setValue(2.0);
  n2.printValue();

  Node n3;
  n3.setValue(3.0);
  n3.printValue();

  vector<Node *> inputs = {&n1, &n2, &n3};
  Add a(inputs);
  Multiply m(inputs);
  cout<<"Addition: ";
  a.forward();
  a.printValue();
  cout<<endl;

  cout<<"Multiplication: ";
  m.forward();
  m.printValue();
  cout<<endl;

  cout<<"Changing value of input 3 to 10"<<endl;
  n3.setValue(10.0);

  cout<<"Addition: ";
  a.forward();
  a.printValue();
  cout<<endl;

  cout<<"Multiplication: ";
  m.forward();
  m.printValue();
  cout<<endl;

  return 0;
}
