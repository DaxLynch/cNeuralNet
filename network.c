#include "network.h"

void backprop(network* nabla, network* net, int x, data* batch_list);

void networkAllocate(network* net, int len, ...){
	net->num_layers = len;
	net->sizes = malloc(sizeof(int)*len);
	net->weights = malloc(sizeof(matrix)*(len-1));
	net->biases = malloc(sizeof(matrix)*(len-1));

	va_list args;
	va_start(args, len);
	for(int i = 0; i < len; i++){
		net->sizes[i] = va_arg(args, int);
		wprintf(L"%d \n", net->sizes[i]);
	}
	va_end(args);
	
	for(int i = 0; i < len - 1; i++){
		matrixAllocate(&net->weights[i], net->sizes[i+1], net->sizes[i]);
		matrixAllocate(&net->biases[i], net->sizes[i+1], 1);
	}
}

void networkSizeAllocate(network* net, network* src){
	net->num_layers = src->num_layers;
	int len = net->num_layers;
	net->sizes = malloc(sizeof(int)*len);
	net->weights = malloc(sizeof(matrix)*(len-1));
	net->biases = malloc(sizeof(matrix)*(len-1));

	for(int i = 0; i < len; i++){
		net->sizes[i] = src->sizes[i];
	}
	
	for(int i = 0; i < len - 1; i++){
		matrixAllocate(&net->weights[i], src->weights[i].m, src->weights[i].n);
		matrixAllocate(&net->biases[i], src->biases[i].m, 1);
	}	
}

void networkFree(network* net){	
	int len = net->num_layers;
	for(int i = 0; i < len - 1; i++){
		matrixFree(&net->weights[i]);
		matrixFree(&net->biases[i]);
	}
	free(net->sizes);
	free(net->weights);
	free(net->biases);
}




void update_mini_batch(network* net, double eta, int batch_size, data* batch_list){
	network nabla;
	int len = net->num_layers;
	networkSizeAllocate(&nabla, net); //allocate	
	for(int i = 0; i < batch_size; i++){
		network delta_nabla;
		networkSizeAllocate(&delta_nabla, net); //Allocate
	       	backprop(&delta_nabla, net, i, batch_list);
		for(int j = 0; j < len - 1; j++){
			matrixAdd(&(nabla.weights[j]),&(delta_nabla.weights[j]));
			matrixAdd(&(nabla.biases[j]),&(delta_nabla.biases[j]));
		}
		networkFree(&delta_nabla);
	}
	for(int i = 0; i < len - 1; i++){
		matrixScalar(&nabla.weights[i], eta/batch_size);
		matrixScalar(&nabla.biases[i], eta/batch_size);
		matrixAdd(&net->weights[i],&nabla.weights[i]);
		matrixAdd(&net->biases[i],&nabla.biases[i]);
	}
	
	networkFree(&nabla); //deallocate
} 

#include "backprop.c"
