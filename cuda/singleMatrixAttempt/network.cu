#include "network.cuh"
using namespace std;

network::network(std::initializer_list<int> sizes){
	this->sizes = sizes;
	this->num_layers = sizes.size();
	for(int i = 0; i < num_layers - 1; i++){
		weights.push_back(matrix(this->sizes[i+1], this->sizes[i]));
		biases.push_back(matrix(this->sizes[i+1], 1));
	}
}
network::network(const network &source){
	sizes = source.sizes;
	num_layers = source.num_layers;
}


ostream& operator<<(ostream& os, network const &net){
	for(int i = 0; i < net.num_layers - 1; i++){
		os << "Weight: " << i << ", (" << net.weights[i].m << ", " << net.weights[i].n << ")" << endl;
		os << net.weights[i]  << endl;
	}
	return os;
}
