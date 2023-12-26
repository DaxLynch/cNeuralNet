__global__ void matsubone(float* array, int y){
	array[y] -= 1;
}

void backprop(network* nabla, network* net, int x, data* batch_list){
	int len = net->num_layers;
	static int allocated = 1;
	static matrix *activations;
        static matrix *zactivations;
	static matrix *z;
	static matrix *delta;

	if (allocated){
		activations = (matrix*)malloc(sizeof(matrix)*len); 
		zactivations = (matrix*)malloc(sizeof(matrix)*(len - 1)); 
		z = (matrix*)malloc(sizeof(matrix)*(len - 1)); 
		delta = (matrix*)malloc(sizeof(matrix)*(len - 1)); 
		matrixAllocate(&activations[0], net->sizes[0],1);
		for (int i = 1; i < len; i++){
			matrixAllocate(&zactivations[i-1], net->sizes[i], 1);
			matrixAllocate(&z[i-1], net->sizes[i], 1);
			matrixAllocate(&delta[i-1], net->sizes[i], 1);
			matrixAllocate(&activations[i], net->sizes[i], 1); 
		}

		allocated = 0;
	}
	matrixCopy(&activations[0], &batch_list[x].matrix);


	for(int i = 1; i < len; i++){

		matrixMultAndDelete(&net->weights[i-1], &activations[i-1], &z[i-1]);
		
		matrixAdd(&z[i-1], &net->biases[i-1]);

		matrixCopy(&zactivations[i-1], &z[i-1]);

		matrixSigmoid(&z[i-1]);
		matrixCopy(&activations[i], &z[i-1]);
	}
	int y = batch_list[x].truth; //only relevant to this data SHould be changed to a truth vector

	//wprintf(L"Y: %d",y);	

	//matrixAllocate(&delta, net->sizes[len-1], 1);
//	matrixPrint(&delta);
	matrixCopy(&delta[1], &activations[len-1]);

	matsubone<<<1,1>>>(delta[1].array, y);

	matrixSigmoidPrime(&zactivations[len-2]);
	matrixHamProd(&delta[1], &zactivations[len-2]);

	matrixAdd(&nabla->biases[len-2], &delta[1]); //add size check for mat copy
	
	matrixMultTransSecondAndDelete(&delta[1], &activations[len-2], &nabla->weights[len-2]);

	for(int i = len - 2; i > 0; i--){
	//	matrix weightsT;
	
	//	matrixAllocate(&weightsT, net->weights[i].m, net->weights[i].n); //free me
	//	matrixCopy(&weightsT, &net->weights[i]);
	//	matrixTranspose(&weightsT);

		matrixSigmoid(&zactivations[i-1]);


		matrixMultTransFirstAndDelete(&net->weights[i],&delta[i], &delta[i-1]);
		
		matrixHamProd(&delta[i-1], &zactivations[i-1]);

		matrixAdd(&nabla->biases[i-1], &delta[i-1]);
		matrixMultTransSecondAndDelete(&delta[i-1], &activations[i-1], &nabla->weights[i-1]);
	}


	return;
}
