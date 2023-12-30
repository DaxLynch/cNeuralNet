double uniformlyRandD(){ //Returns uniformly random on the interval [0,1]
	return (double)rand()/((double)(((double)RAND_MAX)/((double)1.0)));
}
double gaussian(){
	double u1; double u2;

	u1 = uniformlyRandD();
	u2 = uniformlyRandD();
	double z1 = sqrt(-2 * log(u1))*cos(2*M_PI * u2);
	return z1;
}
