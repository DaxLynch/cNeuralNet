#include "main.h"

int main(){
	
	setlocale(LC_CTYPE, "");
	network net;

	networkAllocate(&net,3, 784, 30, 10);
	networkWeightsInit(&net);
	
	data* dati;
	data* trainingData;
	int dataLength = 10000;
	dataLoader(&dati, "trainingData/train-images.idx3-ubyte", "trainingData/train-labels.idx1-ubyte", dataLength*6);
	dataLoader(&trainingData, "trainingData/t10k-images.idx3-ubyte", "trainingData/t10k-labels.idx1-ubyte", dataLength);
//	structDataViewer(datayums+1 );
//	structDataViewer(datayums+2 );
//	structDataViewer(datayums+3 );
//	fileDataViewer("trainingData/train-images.idx3-ubyte");	
	
	networkSGD(&net, dati, dataLength*6, trainingData, dataLength, 1, 1, 100, 5);

	evaluate(&net, dati, 1);

	networkFree(&net);
	dataFree(&dati);


	//	update_mini_batch(&net, 10, datayums);	dataFree(&datayums);
}

