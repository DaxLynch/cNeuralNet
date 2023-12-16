#include "matrix.h"

void initMat(){
	srand(time(NULL));
};

void matrixAllocate(matrix* mat, int m, int n){
	mat->m = m;
	mat->n = n;
	mat->array = (double**)calloc(m, sizeof(double*));
	for(int i = 0; i < m; i++){
		mat->array[i] = (double*)calloc(n, sizeof(double));
	}
}

int matrixCopy(matrix* dst, matrix* src){
	if ((dst->n != src->n) || (dst->m != src->m)){
		perror("matrixCopy wrong dimension");
		return -1;
	}
	int m = dst->m; int n = dst->n;
	double** dPointer = dst->array;
	double** sPointer = src->array;
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			dPointer[i][j] = sPointer[i][j];
		}	
	}
	return 0;
}

void matrixRandFill(matrix* mat, int max){
	for(int i = 0; i < mat->m; i++){
		for(int j = 0; j < mat->n; j++){
			mat->array[i][j] = (double)rand()/((double)(((double)RAND_MAX)/((double)max)));
		}
	}
}

int matrixTranspose(matrix* mat){
	int temp = mat->n;
	mat->n = mat->m;
	mat->m = temp;
	double** newPointer = malloc(sizeof(double*)*mat->m);
	for(int i = 0; i < mat->m; i++){
		newPointer[i] = malloc(sizeof(double)*mat->n);
		for(int j = 0; j < mat->n; j++){
			newPointer[i][j] = mat->array[j][i];
		};
	}
	for(int i = 0; i < mat->n; i++){
		free(mat->array[i]);
	}
	free(mat->array);
	mat->array = newPointer;
	return 1;
}

int matrixSigmoid(matrix* A){
	if (A->n != 1){
		perror("Cannot do sigmoid");
	};
	for(int i = 0; i < A->m; i++){
		A->array[i][0] = sigmoid(A->array[i][0]);
	}
}
int matrixSigmoidPrime(matrix* A){
	if (A->n != 1){
		perror("Cannot do sigmoid prime");
	};
	for(int i = 0; i < A->m; i++){
		A->array[i][0] = sigmoidPrime(A->array[i][0]);
	}
}



int matrixMult(matrix* A, matrix* B, matrix* out){
	if (A->n != B->m){
		perror("matrixMult error: sizes incorrect");
		return -1;
	}
	for(int i = 0; i < A->m; i++){
	       for(int j = 0; j < B->n; j++){
			for(int k = 0; k < A->m; k++){
				out->array[i][j] = A->array[i][k] * B->array[k][j];
			}
	       }
	}
	return 0;
}

int matrixAdd(matrix* dst, matrix* src){ //in place,a s opposed to matrix multiply???
	if ((dst->n != src->n) || (dst->m != src->m)){
		perror("matrixAdd wrong dimension");
		return -1;
	}
	int m = dst->m; int n = dst->n;
	double** dpointer = dst->array;
	double** spointer = src->array;
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			dpointer[i][j] += spointer[i][j];
		}	
	}
	return 0;
}

int matrixHamProd(matrix* dst, matrix* src){ //in place,a s opposed to matrix multiply???
	if ((dst->n != src->n) || (dst->m != src->m)){
		perror("matrixAdd wrong dimension");
		return -1;
	}
	int m = dst->m; int n = dst->n;
	double** dpointer = dst->array;
	double** spointer = src->array;
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			dpointer[i][j] *= spointer[i][j];
		}	
	}
	return 0;
}

void matrixPrint(matrix* mat) {
    for (int i = 0; i < mat->m; i++) {
        for (int j = 0; j < mat->n; j++) {
            wprintf(L" %lf ", mat->array[i][j]);
        }
        wprintf(L"\n");
    }
}

void matrixFree(matrix* mat) {
    for (int i = 0; i < mat->m; i++) {
        free(mat->array[i]); // Free each row
    }
    
    free(mat->array); // Free the array of rows
}


