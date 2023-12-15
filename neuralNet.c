#include "matrix.h"
#include "sigmoid.h"

int sizes[784,30,10];

double *w0, *w1, *w2;
double *b1, *b2; 

int epochs = 30;

data* loadedData;

void* batch_process(int* batch)
int main()
	for (int j = 0; j < epochs; j++){
#do a random shuffle of the training data;
		assign mini batches;
		for each mini batch, do an update
		every epoch, print out the test evaluations









