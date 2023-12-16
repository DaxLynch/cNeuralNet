#include "network.h"

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
		matrixAllocate(&net->weights[i], net->sizes[i],net->sizes[i+1]);
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
		matrixAllocate(&net->weights[i], net->sizes[i],net->sizes[i+1]);
		matrixAllocate(&net->biases[i], net->sizes[i+1],1);
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




void update_mini_batch(network* net, int batch_size, data* batch_list){
	network nabla;
	int len = net->num_layers;
	networkSizeAllocate(&nabla, net); //allocate	
	for(int i = 0; i < batch_size; i++){
		network delta_nabla;
		networkSizeAllocate(&delta_nabla, net); //Allocate
	       	//backprop(&delta_nabla, &net, &batchlist, i);
		for(int j = 0; j < len - 1; j++){
			matrixAdd(&(nabla.weights[j]),&(delta_nabla.weights[j]));
			matrixAdd(&(nabla.biases[j]),&(delta_nabla.biases[j]));
		}
		networkFree(&delta_nabla);
	}
	for(int i = 0; i < len - 1; i++){
		matrixAdd(&net->weights[i],&nabla.weights[i]);
		matrixAdd(&net->biases[i],&nabla.biases[i]);
	}
	
	networkFree(&nabla); //deallocate
} 

void backprop(network* nabla, network* net, int x, data* batch_list){
	len = net->num_layers;
	matrix* activations = malloc(sizeof(vector)*len); //free me!
	matrix* zactivations = malloc(sizeof(vector)*(len - 1)); //free me!
	
	matrixAllocate(&activations[0], net->size[0],1); //free me
	matrixCopy(&activations[0], &batch_list[x].matrix);
	for(int i = 1; i < len; i++){
		matrixAllocate(zactivations[i-1], net->sizes[i]); //free me
		matrixMult(&net->weight[i-1],activations[i-1], zactivations[i-1]);
		matrixAdd(zactivations[i-1],net->biases[i-1]);

		matrixAllocate(&activations[i], net->sizes[i], 1); //free me
		matrixCopy(&activations[i], vectorSigmoid(z));
	}
	int y = batch_list[i].truth; //only relevant to this data SHould be changed to a truth vector
	
	matrix delta;
	matrixAllocate(&delta, net->sizes[3], 1);
	matrixCopy(&delta, &activations[len]);
	delta.array[y][0] - 1

	matrixSigmoid(&zactivations[len-1]);
	matrixHamProd(&delta, &zactivations[len - 1]);

	matrixCopy(&nabla->biases[len-1], &delta); //add size check for mat copy
	
	matrixTranspose(&activations[len - 1]); //make this
	matrixMult(&delta, &activations[len - 1], &nabla->weights[len-1]);
	
	for(int i = len - 1; i > 1; i--){
		matrix weightsT;
		matrixAllocate(&weightsT, net->weights[i].m, net->weights[i].n);
		matrixCopy(&weightsT, &net->weights[i]);
		matrixTranspose(&weightsT);

		matrixSigmoid(&zactivations[i-1]);

		matrix delta0; matrixAllocate(&delta0, zactivations[i-1].m, 1);
		matrixMult(&weightsT,&delta, &delta0);
	        matrixFree(&delta);
		delta = delta0;	
		matrixHamProd(&delta, zactivations[i-1]);

		matrixCopy(&nabla->biases[i], &delta);
		matrixTranspose(activations[i-1]);
		matrixMult(&delta, &activations[i-1], nabla->weights[i-1]);
	}
	return;
}
