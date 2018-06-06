#include "mnist/mnistUtils.h"

#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
  vector<pair<vector<double>, unsigned int> > train_data;
	vector<pair<vector<double>, unsigned int> > valid_data;
	getData(train_data, valid_data);

  for(auto sample : train_data)
  {
    cout<<sample.first.size()<<endl;
  }
  return 0;
}
