#pragma once
#include "matrix.cuh"
#include "dataLoader.cuh"
#include "layer.cuh"
#include <vector>
#include <initializer_list>
using namespace std;
class network{
public:
	vector<int> sizes;
	vector<matrix> weights;
	vector<matrix> biases;
	int num_layers = -1;

	network();
	network(initializer_list<int> sizes);
	network(const network &source);
	friend ostream& operator<<(ostream& os, network const &net);
	//SGD(data& dataSet, data& testSet, int epochs, int batchsize, float eta, float regularization);

//batchUpdate();
private:
};
