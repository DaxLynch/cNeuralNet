int evaluate(network* net, data* datum, int print){
	int len = net->num_layers;
	matrix* activations = malloc(sizeof(matrix)*len); 
	matrixAllocate(&activations[0], net->sizes[0],1);
	matrixCopy(&activations[0], &datum->matrix);
	
	for(int i = 1; i < len; i++){
		matrixAllocate(&activations[i], net->sizes[i], 1);
		matrixMult(&net->weights[i-1], &activations[i-1], &activations[i]);
		matrixAdd(&activations[i], &net->biases[i-1]);

		matrixSigmoid(&activations[i]);
	}
	double max = 0;
	int maxArg = 0;
	for(int i = 0; i < 10; i++){
		if (activations[len - 1].array[i] > max){
			max = activations[len -1].array[i];
			maxArg = i;
		}
	}
	if(print){
		structDataViewer(datum);
		wprintf(L"returned %d with value of %.2lf, true value is %d\n", maxArg, max, datum->truth);
	}
	for(int i = 0; i < len; i++){
		matrixFree(&activations[i]);
	}
	free(activations);
	return maxArg;
}
int evaluateSet(network* net, data* testData, int dataLength){
	int correct = 0;
	for(int i = 0; i < dataLength; i++){
		if (evaluate(net, &testData[i], 0)  == testData[i].truth){
			correct++;
		}
	}
	wprintf(L"%.2f %% correct, %d/%d \n", ((double)correct)/((double)dataLength)*100.0, correct, dataLength);
	return 0;
}
int evaluateSetManual(network* net, data* datum, int dataLength){
	wprintf(L"Press q to exit, press any character to see the next evaluation\n");
	int i = 0;
	while(getchar() != 'q' || i == dataLength ){ 
		evaluate(net, datum + i, 1);
		i++;
	}
	return 0;
}
int evaluateSetFailures(network* net, data* datum, int dataLength){
	wprintf(L"Press q to exit, press any character to see the next failed evaluation\n");
	int i = 0;
	while(getchar() != 'q' || i == dataLength ){ 
		for(;evaluate(net, datum+i,0)==datum[i].truth  && i < 60000; i++){
		}
		evaluate(net, datum + i,1);
		i++;
	}
	return 0;
}


