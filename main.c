#include "main.h"

int main(){
	
	setlocale(LC_CTYPE, "");
	network net;

	networkAllocate(&net,3, 784, 30, 10);

	network nabla;
       	networkSizeAllocate(&nabla, &net);
	data* datayums;
	dataLoader(&datayums, "trainingData/train-images.idx3-ubyte", "trainingData/train-labels.idx1-ubyte", 60000);
	structDataViewer(datayums);
//	structDataViewer(datayums+1 );
//	structDataViewer(datayums+2 );
//	structDataViewer(datayums+3 );
//	fileDataViewer("trainingData/train-images.idx3-ubyte");	
	
	networkSGD(&net, datayums, 1000, 3, 100, .05);

	evaluate(&net, datayums);

	networkFree(&net);
	networkFree(&nabla);
	dataFree(&datayums);


	//	update_mini_batch(&net, 10, datayums);	dataFree(&datayums);
}

