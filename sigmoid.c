#include <math.h>

double sigmoid(double z){
	return (double)1.0/((double)1.0 + exp(-z))
}

matrix* matSig()
