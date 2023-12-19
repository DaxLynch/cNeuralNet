void backprop(network* nabla, network* net, int x, data* batch_list){
	int len = net->num_layers;
	matrix* activations = malloc(sizeof(matrix)*len); 
	matrix* zactivations = malloc(sizeof(matrix)*(len - 1)); 
	matrixAllocate(&activations[0], net->sizes[0],1);
	matrixCopy(&activations[0], &batch_list[x].matrix);
	
	for(int i = 1; i < len; i++){
		matrix z; 
		matrixAllocate(&z, net->sizes[i], 1);
		matrixAllocate(&zactivations[i-1], net->sizes[i], 1);
		matrixMult(&net->weights[i-1], &activations[i-1], &z);
		matrixAdd(&z, &net->biases[i-1]);

		matrixCopy(&zactivations[i-1], &z);

		matrixAllocate(&activations[i], net->sizes[i], 1); 
		matrixSigmoid(&z);
		matrixCopy(&activations[i], &z);
		
		matrixFree(&z);
	}
	int y = batch_list[x].truth; //only relevant to this data SHould be changed to a truth vector
//	wprintf(L"Y: %d",y);	
	matrix delta;
	matrixAllocate(&delta, net->sizes[len-1], 1);
//	matrixPrint(&delta);
	matrixCopy(&delta, &activations[len-1]);
	delta.array[y] -= 1;
//	matrixPrint(&delta);

	matrixSigmoidPrime(&zactivations[len-2]);
	matrixHamProd(&delta, &zactivations[len-2]);
	matrixCopy(&nabla->biases[len-2], &delta); //add size check for mat copy
	
	matrixTranspose(&activations[len-2]); //make this
	matrixMult(&delta, &activations[len-2], &nabla->weights[len-2]);
	
	for(int i = len - 2; i > 0; i--){
		matrix weightsT;
		matrixAllocate(&weightsT, net->weights[i].m, net->weights[i].n); //free me
		matrixCopy(&weightsT, &net->weights[i]);
		matrixTranspose(&weightsT);

		matrixSigmoid(&zactivations[i-1]);

		matrix delta0; 
		matrixAllocate(&delta0, zactivations[i-1].m, 1);
		
		matrixMult(&weightsT,&delta, &delta0);
	        matrixFree(&delta);
	        matrixFree(&weightsT);
		delta = delta0;
		matrixHamProd(&delta, &zactivations[i-1]);

		matrixCopy(&nabla->biases[i-1], &delta);
		matrixTranspose(&activations[i-1]);
		matrixMult(&delta, &activations[i-1], &nabla->weights[i-1]);
	}
	matrixFree(&delta);

	matrixFree(&activations[0]);
	for(int i = 1; i < len; i++){
		matrixFree(&zactivations[i-1]);
		matrixFree(&activations[i]);
	}
	
	free(activations);
	free(zactivations);
	

	return;
}
