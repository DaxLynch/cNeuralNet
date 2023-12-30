__global__ void matsubone(float* array, int y){
	array[y] -= 1;
}

void backprop(network* nabla, network* net, int x, data* batch_list){
	int len = net->num_layers;

	static int allocated = 1;
	static matrix *activations;
        static matrix *zactivations;
	static matrix *delta;

	if (allocated){
		activations = (matrix*)malloc(sizeof(matrix)*len); 
		zactivations = (matrix*)malloc(sizeof(matrix)*(len - 1)); 
		delta = (matrix*)malloc(sizeof(matrix)*(len - 1)); 
		matrixAllocate(&activations[0], net->sizes[0],1);
		for (int i = 1; i < len; i++){
			matrixAllocate(&zactivations[i-1], net->sizes[i], 1);
			matrixAllocate(&delta[i-1], net->sizes[i], 1);
			matrixAllocate(&activations[i], net->sizes[i], 1); 
		}
		allocated = 0;
	}
	matrixCopy(&activations[0], &batch_list[x].matrix); 

	for(int i = 1; i < len; i++){
		matrixMult(&net->weights[i-1], &activations[i-1], &zactivations[i-1]);	
		matrixAdd(&zactivations[i-1], &net->biases[i-1]);
		matrixCopy(&activations[i], &zactivations[i-1]);
		matrixSigmoid(&activations[i]);
	}
	int y = batch_list[x].truth; //only relevant to this data SHould be changed to a truth vector
	matrixCopy(&delta[len -2], &activations[len-1]);

	matsubone<<<1,1>>>(delta[len -2].array, y); //these ops can be combined

	matrixSigmoidPrime(&zactivations[len-2]);
	matrixHamProd(&delta[len - 2], &zactivations[len-2]);
	matrixAdd(&nabla->biases[len-2], &delta[len -2]); //add size check for mat copy
	
	matrixMultTransSecondNoDelete(&delta[len -2], &activations[len-2], &nabla->weights[len-2]);

	for(int i = len - 2; i > 0; i--){

		matrixSigmoidPrime(&zactivations[i-1]);
		matrixMultTransFirst(&net->weights[i],&delta[i], &delta[i-1]);
		
		matrixHamProd(&delta[i-1], &zactivations[i-1]);

		matrixAdd(&nabla->biases[i-1], &delta[i-1]);
		matrixMultTransSecondNoDelete(&delta[i-1], &activations[i-1], &nabla->weights[i-1]);
	}
	//for(int i = 0; i < 2; i++){
	//	cudaMemset(zactivations[i].array, 0, net->sizes[i+1] * sizeof(float)* 1);
	//	cudaMemset(activations[i].array, 0, net->sizes[i] * sizeof(float)* 1);
	//	cudaMemset(delta[i].array, 0, net->sizes[i+1] * sizeof(float)* 1);
	//
	//cudaMemset(activations[2].array, 0, net->sizes[2] * sizeof(float)* 1);
	cudaDeviceSynchronize();
	return;
}
