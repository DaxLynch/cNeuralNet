double sigmoid(double z){
	return (double)1.0/((double)1.0 + exp(-z));
}
double sigmoidPrime(double z){
	return (double)sigmoid(z) * ((double)1.0 - (double)sigmoid(z));
}
double uniformlyRandD(){
	return (double)rand()/((double)(((double)RAND_MAX)/((double)1.0)));
}
double gaussian(){
	double u1; double u2;

	u1 = uniformlyRandD();
	u2 = uniformlyRandD();
	double z1 = sqrt(-2 * log(u1))*cos(2*M_PI * u2);
	return z1;
}
