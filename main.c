#include "main.h"

int main(){
	
	setlocale(LC_CTYPE, "");
	network net;
	networkInit(&net,3, 728, 30, 10);
	data* datayums;
	dataLoader(&datayums);
	fileDataViewer("trainingData/train-images.idx3-ubyte");
}

