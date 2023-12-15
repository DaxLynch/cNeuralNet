#include "dataLoader.h"

void displayChar(unsigned char input){
	wchar_t ret[] = {' ', 0x2593, 0x2592, 0x2591, 0x2599, 0x2588};
	wprintf(L"%lc", ret[input/52]);
}

void* dataLoader(data** dataPointer){
	FILE* training_images = fopen("trainingData/train-images-idx3-ubyte", "rb");
	unsigned char bs[16];
	*dataPointer = malloc(sizeof(data) * 60000);
	fread(bs, 16, 1, training_images);
	for(int m = 0; m < 60000; m++){	
		fread((*dataPointer)[m].array, 28*28, 1, training_images);
	}
	fclose(training_images);
}
void fileDataViewer(char* inputFile){ //"trainingData/train-images-idx3-ubyte",
	FILE* training_images = fopen(inputFile, "rb");
	unsigned char bs[16];
	fread(bs, 16, 1, training_images);
	unsigned char array[28 * 28];
	setlocale(LC_CTYPE, "");
	char f;
	while(f = getchar()){
	fread(array, 28*28, 1, training_images);
	for(int i = 0; i < 28; i++){
		for(int j = 0; j < 28; j++){
			displayChar(array[i*28 + j]);
		}
		wprintf(L"\n");
	}
	}
	printf("\n");
	fclose(training_images);
}

void structDataViewer(data* dataPointer){
	setlocale(LC_CTYPE, "");
	for(int i = 0; i < 28; i++){
		for(int j = 0; j < 28; j++){
			displayChar(dataPointer->array[i*28 + j]);
		}
		wprintf(L"\n");
	}
	printf("\n");
}

//int main(){
//	data* datayums;
//	dataLoader(&datayums);
//	fileDataViewer("trainingData/train-images-idx3-ubyte");
//
//}
