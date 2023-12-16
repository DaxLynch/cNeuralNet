#include "main.h"

int main(){
	
	setlocale(LC_CTYPE, "");
	network net;
	networkAllocate(&net,3, 728, 30, 10);

	data* datayums;
	dataLoader(&datayums);
	dataFree(&datayums);
	
	network nabla; networkSizeAllocate(&nabla, &net);
	backprop(&nabla, &net, 1, datayums);

	//	update_mini_batch(&net, 10, datayums);	dataFree(&datayums);
	networkFree(&net);
}

