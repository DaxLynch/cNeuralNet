void displayChar(unsigned char input){	
	char ret[] = {' ', '-','*','#','%','@'};
	printf("%c", ret[input/43]);
}

void dataLoader(data** dataPointer, char* images, char* labels, int dataLength){
	FILE* training_images = fopen(images, "rb");
	FILE* training_labels = fopen(labels, "rb");
	if(training_images == NULL){
		printf("fuck");
		exit(EXIT_FAILURE);
	}
	unsigned char bs[16];
	*dataPointer = (data*)malloc(sizeof(data) * dataLength);
	fread(bs, 16, 1, training_images);
	fread(bs, 8, 1, training_labels);
	unsigned char buff[28*28];
	float buff2[28*28];
	for(int m = 0; m < dataLength; m++){
		matrixAllocate(&((*dataPointer)[m].matrix), 784, 1);	
		fread(buff, 1, 28*28, training_images);
		for(int i = 0; i < (28 * 28); i ++){
			buff2[i] = (((float)buff[i] - 33.0f)/255.0f);
		}
		cudaMemcpy((*dataPointer)[m].matrix.array, buff2, 28*28*sizeof(float), cudaMemcpyHostToDevice);
		unsigned char temp;
		fread(&temp, sizeof(unsigned char), 1, training_labels);
		(*dataPointer)[m].truth = (int)temp;
	}
	fclose(training_images);
	fclose(training_labels);
}
void fileDataViewer(char* inputFile){ //"trainingData/train-images.idx3-ubyte",
	FILE* training_images = fopen(inputFile, "rb");
	if (training_images == NULL){
		printf("stupid bs\n");
		return;
	}
	unsigned char bs[16];
	fread(bs, 16, 1, training_images);
	unsigned char array[28 * 28];	
	while(getchar()){
		fread(array, 28*28, 1, training_images);
		for(int i = 0; i < 28; i++){
			for(int j = 0; j < 28; j++){
				displayChar(array[i*28 + j]);
			}
			printf("\n");
		}
	}
	printf("\n");
	fclose(training_images);
}

void structDataViewer(data* dataPointer){
	float temp[28*28];
	cudaMemcpy(temp, dataPointer->matrix.array, 28*28*sizeof(float), cudaMemcpyDeviceToHost);
	for(int i = 0; i < 28; i++){
		for(int j = 0; j < 28; j++){
			displayChar((unsigned char)(temp[i*28 + j]*255.0f  + 33.0f));
		}
		printf("\n");
	}
	printf("\n");
}


void dataFree(data** dataPointer, int dataLength){
	for(int m = 0; m < dataLength; m++){
		matrixFree(&((*dataPointer)[m].matrix));
	}
	free(*dataPointer);
}
