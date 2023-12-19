#include "main.h"

int main(){
	
	setlocale(LC_CTYPE, "");
	network net;

	initMat();
	networkAllocate(&net,3, 784, 30, 10);
	networkWeightsInit(&net);
	
	data* trainingData;
	data* testingData;
	int trainingLength = 1000;
	int testingLength = 1000;
	dataLoader(&trainingData, "trainingData/train-images.idx3-ubyte", "trainingData/train-labels.idx1-ubyte", trainingLength);
	dataLoader(&testingData, "trainingData/t10k-images.idx3-ubyte", "trainingData/t10k-labels.idx1-ubyte", testingLength);
	
	networkSGD(&net, trainingData, trainingLength, testingData, testingLength, 1,1, 100, 5);
	
	evaluateSetManual(&net, testingData, testingLength);
	evaluateSetFailures(&net, testingData, testingLength);

	networkFree(&net);
	dataFree(&trainingData, trainingLength);
	dataFree(&testingData, testingLength);

}
