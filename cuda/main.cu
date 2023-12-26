#include "main.cuh"
#define print 1
#define DontPrint 0
int main(){
	
	setlocale(LC_CTYPE, "");
	initMat();
		
	network net;
	networkAllocate(&net,3, 784, 30, 10); //Initilizes a 3 layer net with sizes 784, 30, 100
	networkWeightsInit(&net); //Sets the weights and biases to gaussian distributed around 0

	
//	matrix test1;
//	matrix test2;
//	matrix test3;
//	matrixAllocate(&test1,14,30);
//	matrixAllocate(&test2,30,10);
//	matrixAllocate(&test3,14,10);
//	matrixRandFill(&test2);
//	matrixRandFill(&test1);
//	matrixMult(&test1,&test2,&test3);
//	matrixPrint(&test3);

	data* trainingData;
	data* testingData;
	int trainingLength = 60000;
	int testingLength = 10000;
	dataLoader(&trainingData, "trainingData/train-images.idx3-ubyte", "trainingData/train-labels.idx1-ubyte", trainingLength);
	dataLoader(&testingData, "trainingData/t10k-images.idx3-ubyte", "trainingData/t10k-labels.idx1-ubyte", testingLength);
	//The above loades data in from the trainingData folder,
	
	networkSGD(&net, trainingData, trainingLength, testingData, testingLength, print, 10, 10, 5);
	//The above does SGD on the training data, with it printing every epoch, and 10 epochs, and a batchsize of 10, with a learning rate of 5.
	
	evaluateSet(&net, testingData, testingLength);
	//evaluateSetManual(&net, testingData, testingLength); //This prints the data, alongside the returned value and the truth value
	//evaluateSetFailures(&net, testingData, testingLength); //Like above but it shows you the failures

	networkFree(&net);//These commands free the networks and arrays
	dataFree(&trainingData, trainingLength);
	dataFree(&testingData, testingLength);

}
