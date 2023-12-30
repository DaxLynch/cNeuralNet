#include "main.cuh"
#define Print 1
#define DontPrint 0
int main(){
	
	setlocale(LC_CTYPE, "");
	initMat();
	int size = 512;	
	matrix test1;
	matrix test2;
	matrix test3;
	matrix test4;
	matrixAllocate(&test1,size,size);
	matrixAllocate(&test2,size,size);
	matrixAllocate(&test3,size,size);
	matrixAllocate(&test4,size,size);
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
	//matrixPrint(&test3);
	//printf("Here lies dax \n");
	//matrixPrint(&test4);

	printf("Here lies dax \n");
	matrixEqual(&test3, &test4);
}
