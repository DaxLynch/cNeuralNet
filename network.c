void networkAllocate(network* net, int len, ...){
	net->num_layers = len;
	net->sizes = malloc(sizeof(int)*len);
	net->weights = malloc(sizeof(matrix)*(len-1));
	net->biases = malloc(sizeof(matrix)*(len-1));

	va_list args;
	va_start(args, len);
	for(int i = 0; i < len; i++){
		net->sizes[i] = va_arg(args, int);
	}
	va_end(args);
	
	for(int i = 0; i < len - 1; i++){
		matrixAllocate(&net->weights[i], net->sizes[i+1], net->sizes[i]);
		matrixAllocate(&net->biases[i], net->sizes[i+1], 1);
	}
}

void networkWeightsInit(network* net){
	for(int i = 0; i < net->num_layers - 1; i++){
		matrixRandFill(&net->weights[i], -1, 1);
		matrixRandFill(&net->biases[i], -1, 1);
	}
}

void networkPrint(network* net){
	for(int i = 0; i < net->num_layers - 1; i++){
		wprintf(L"net->weights[%d]\n", i);
		matrixPrint(&net->weights[i]);
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




void update_mini_batch(network* net, int batch_size, int* batch_list, data* data_set, double eta){
	network nabla;
	int len = net->num_layers;
	networkSizeAllocate(&nabla, net); //allocate	
	for(int i = 0; i < batch_size; i++){
		network delta_nabla;
		networkSizeAllocate(&delta_nabla, net); //Allocate
	       	backprop(&delta_nabla, net, batch_list[i], data_set);
		for(int j = 0; j < len - 1; j++){
			matrixAdd(&(nabla.weights[j]),&(delta_nabla.weights[j]));
			matrixAdd(&(nabla.biases[j]),&(delta_nabla.biases[j]));
		}
		networkFree(&delta_nabla);
	}
	for(int i = 0; i < len - 1; i++){
		matrixScalar(&nabla.weights[i], -eta/batch_size);
		matrixScalar(&nabla.biases[i], -eta/batch_size);
		matrixAdd(&net->weights[i],&nabla.weights[i]);
		matrixAdd(&net->biases[i],&nabla.biases[i]);
	}
	networkFree(&nabla); //deallocate
} 

void shuffle(int *array, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

void networkSGD(network* net, data* dataset, int dataLength, data* testSet, int testLength, int test, int epochs, int batch_size, double eta){
	for (int j = 0; j < epochs; j++){
		wprintf(L"Starting epoch:%d \n", j);
		int* shuffled = malloc(sizeof(int) * dataLength);
		for(int i = 0; i < dataLength; i++){
			shuffled[i] = i;
		}
		shuffle(shuffled, dataLength);
		for( int i = 0; i < dataLength/batch_size; i++){
			update_mini_batch(net, batch_size,  shuffled + (batch_size*i), dataset, eta);
		}
		if (test){
			evaluateSet(net, testSet, testLength);
		}
		free(shuffled);
	}
	
}

