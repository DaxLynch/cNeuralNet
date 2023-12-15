#include "dataLoader.h"

void displayChar(unsigned char input){	
	wchar_t ret[] = {' ', 0x2593, 0x2592, 0x2591, 0x25A9, 0x2588};
	wprintf(L"%lc", ret[input/43]);
}

void* dataLoader(data** dataPointer){
	FILE* training_images = fopen("trainingData/train-images.idx3-ubyte", "rb");
	unsigned char bs[16];
	*dataPointer = malloc(sizeof(data) * 1000);
	fread(bs, 16, 1, training_images);
	for(int m = 0; m < 1000; m++){	
		fread((*dataPointer)[m].array, 28*28, 1, training_images);
	}
	fclose(training_images);
}
void fileDataViewer(char* inputFile){ //"trainingData/train-images.idx3-ubyte",
	FILE* training_images = fopen(inputFile, "rb");
	if (training_images == NULL){
		wprintf(L"stupid bs\n");
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
			wprintf(L"\n");
		}
	}
	wprintf(L"\n");
	fclose(training_images);
}

void structDataViewer(data* dataPointer){
	for(int i = 0; i < 28; i++){
		for(int j = 0; j < 28; j++){
			displayChar(dataPointer->array[i*28 + j]);
		}
		wprintf(L"\n");
	}
	printf("\n");
}
