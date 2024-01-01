#include "main.cuh"
using namespace std;

int main(){
	cout  << "Hellow world" << endl;
	data trainingImages("trainingData/train-images.idx3-ubyte", 6);
	

	network net{784,30,10};
	cout << net;
	return 0;
}
