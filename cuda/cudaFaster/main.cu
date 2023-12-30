#include "main.cuh"
#define Print 1
#define DontPrint 0
int main(){
	
	setlocale(LC_CTYPE, "");
	initMat();
	int m = 4091; int q = 4096; int n = 4091;
	matrix test1;
	matrix test2;
	matrix test3;
	matrix test4;
	matrixAllocate(&test1,m,q);
	matrixAllocate(&test2,q,n);
	matrixAllocate(&test3,m,n);
	matrixAllocate(&test4,m,n);
	matrixRandFill(&test2);
	matrixRandFill(&test1);
	clock_t start = clock();
	matrixMult(&test1,&test2,&test3);
	clock_t end = clock();
	printf("Time taken: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

	start = clock();
	matrixMultNoDelete(&test1,&test2,&test4);
	end = clock();
	printf("Time taken: %f seconds\n", ((double)(end - start)) / CLOCKS_PER_SEC);

	printf("Here lies dax \n");
	matrixEqual(&test3, &test4);
}
