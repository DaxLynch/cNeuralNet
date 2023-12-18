#include "main.h"

int main(){
	
	setlocale(LC_CTYPE, "");
	network net;

	networkAllocate(&net,3, 784, 30, 10);

	network nabla;
       	networkSizeAllocate(&nabla, &net);
	data* datayums;
	dataLoader(&datayums);
	structDataViewer(datayums);
//	structDataViewer(datayums+1 );
//	structDataViewer(datayums+2 );
//	structDataViewer(datayums+3 );
//	fileDataViewer("trainingData/train-images.idx3-ubyte");	
	
	update_mini_batch(&net, .05, 20, datayums);
	networkFree(&net);
	networkFree(&nabla);
	dataFree(&datayums);


	//	update_mini_batch(&net, 10, datayums);	dataFree(&datayums);
}

