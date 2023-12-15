#include "network.h"

void networkInit(network* net, int len, ...){
	net->num_layers = len;
	net->sizes = malloc(sizeof(int)*net->num_layers);

	va_list args;
	va_start(args, len);
	for(int i = 0; i < len; i++){
		net->sizes[i] = va_arg(args, int);
		wprintf(L"%d \n", net->sizes[i]);
	}
	va_end(args);
}
	
