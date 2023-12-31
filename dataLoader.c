void displayChar(unsigned char input){	
	char ret[] = {' ', '-','*','#','%','@'};
	printf("%c", ret[input/43]);
}

void dataLoader(data** dataPointer, char* images, char* labels, int dataLength){
	FILE* training_images = fopen(images, "rb");
	FILE* training_labels = fopen(labels, "rb");
	unsigned char bs[16];
	*dataPointer = malloc(sizeof(data) * dataLength);
	fread(bs, 16, 1, training_images);
	fread(bs, 8, 1, training_labels);
	unsigned char buff[28*28];
	for(int m = 0; m < dataLength; m++){
		matrixAllocate(&((*dataPointer)[m].matrix), 784, 1);	
		fread(buff, 1, 28*28, training_images);
		for(int i = 0; i < 28 * 28; i ++){
			(*dataPointer)[m].matrix.array[i] = ((double)buff[i] - 33.0f)/(double)255.0;
		}
		unsigned char temp;
		fread(&temp, 1, 1, training_labels);
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
	for(int i = 0; i < 28; i++){
		for(int j = 0; j < 28; j++){
			displayChar(((unsigned char)((dataPointer->matrix.array[i*28 + j] * 255) + 33.0f) ));
			//displayChar((unsigned char)128);
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
