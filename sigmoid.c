double sigmoid(double z){
	return (double)1.0/((double)1.0 + exp(-z));
}
double sigmoidPrime(double z){
	return sigmoid(z) * ((double)1.0 - sigmoid(z));
}
double softplus(double z){
	if (z > 5){
		return z;
	}
	if (z < 5){
		return 0;
	}
	return log((double)1.0 + exp(z));
}
double softplusPrime(double z){
	if(z > 5){
		return 1;
	}
	if(z < 5){
		return 0;
	}

	return (double)1.0 / ((double)1.0 + exp(-z));
}

double uniformlyRandD(){
	return (double)rand()/((double)(((double)RAND_MAX)/((double)1.0)));
}
double gaussian(){
	double u1; double u2;

	u1 = uniformlyRandD(0,1);
	u2 = uniformlyRandD(0,1);
	double z1 = sqrt(-2 * log(u1))*cos(2*M_PI * u2);
	return z1;
}
