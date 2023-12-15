#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"
int main() {
	matrix mat1;
	matrix mat2;
	matrix matOut;

	srand(time(NULL)); // Seed the random number generator

	allocateArray(&mat1, 3, 4);
	allocateArray(&mat2, 4, 7);
	allocateArray(&matOut, 3, 7);

	randFill(&mat1, 100); // Assuming you want values between 0 and 100
	randFill(&mat2, 2);



	matMult(&mat1,&mat2,&matOut);

	printMatrix(&mat2);	
	printMatrix(&mat1);	
	printMatrix(&matOut);	
	freeArray(&mat1);freeArray(&mat2); // Free the allocated memory

	return 0;
}
