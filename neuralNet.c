#include "matrix.h"
#include "sigmoid.h"

int sizes[784,30,10];

double *w0, *w1, *w2;
double *b1, *b2; 

int epochs = 30;

data* loadedData;
	
int main(){
	network net;
	networkInit(net, 728,30,10);

}






