#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnistLoader.h"

#include <iostream>
#include <vector>

using namespace std;

void getData(vector<pair<vector<double>, unsigned int> > &train_data, vector<pair<vector<double>, unsigned int> > &valid_data)
{
	mnist_data *data;
  unsigned int cnt;
	int ret;

	if ((ret = mnist_load("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte", &data, &cnt)))
  {
    cout<<"An error occured: "<<ret<<endl;
  }

	train_data.clear();
	pair<vector<double>, unsigned int> tData = make_pair(vector<double>(28 * 28), 0);
	train_data = vector<pair<vector<double>, unsigned int> >(50000, tData);
	valid_data = vector<pair<vector<double>, unsigned int> >(100, tData);
	for (int k = 0; k < 50100; k++)
	{
		for (int i = 0; i < 28; i++)
		{
			for (int j = 0; j < 28; j++)
			{
				if (k < 50000)
					train_data[k].first[i * 28 + j] = data[k].data[i][j];
				else
					valid_data[k-50000].first[i * 28 + j] = data[k].data[i][j];
			}
		}
		if (k < 50000)
			train_data[k].second = data[k].label;
		else
			valid_data[k - 50000].second = data[k].label;
	}
	free(data);

}
