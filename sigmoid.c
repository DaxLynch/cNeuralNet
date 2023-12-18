double sigmoid(double z){
	return (double)1.0/((double)1.0 + exp(-z));
}
double sigmoidPrime(double z){
	return sigmoid(z) * ((double)1.0 - sigmoid(z));
}
