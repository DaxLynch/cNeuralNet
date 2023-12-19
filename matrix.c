void initMat(){
	srand(time(NULL));
}

void matrixAllocate(matrix* mat, int m, int n){
	mat->m = m;
	mat->n = n;
	mat->array = (double*)calloc(m*n, sizeof(double));
}

int matrixCopy(matrix* dst, matrix* src){
	if ((dst->n != src->n) || (dst->m != src->m)){
		wprintf(L"(%d, %d) != (%d, %d)\n",dst->m,dst->n,src->m,src->n);
		wprintf(L"matrixCopy wrong dimension\n");
		return -1;
	}
	int m = dst->m; int n = dst->n;
	double* dPointer = dst->array;
	double* sPointer = src->array;
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			dPointer[i*n + j] = sPointer[i*n + j];
		}	
	}
	return 0;
}

void matrixRandFill(matrix* mat, int min, int max){
	int m = mat->m; int n = mat->n;
 	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			mat->array[i*n + j] = gaussian();
		}
	}
}

void matrixScalar(matrix* mat, double scalar){
	int m = mat->m; int n = mat->n;
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			mat->array[i*n + j] *= scalar;
		}
	}
}

int matrixTranspose(matrix* mat){
	double* matrix = mat->array;
	int m = mat->m; int n = mat->n;
	double* newMat = malloc(sizeof(double) * m*n);

	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			newMat[j*m + i] = matrix[i*n + j];
		}
	}
	mat->n = mat->m;
	mat->m = n;
	free(mat->array);
	mat->array = newMat;
	return 0;
}

int matrixSigmoid(matrix* A){
	if (A->n != 1){
		perror("Cannot do sigmoid");
		return -1;
	};
	for(int i = 0; i < A->m; i++){
		A->array[i] = sigmoid(A->array[i]);
	}
	return 0;
}

int matrixSigmoidPrime(matrix* A){
	if (A->n != 1){
		perror("Cannot do sigmoid prime");
		return -1;
	};
	for(int i = 0; i < A->m; i++){
		A->array[i] = sigmoidPrime(A->array[i]);
	}
	return 0;
}





int matrixMult(matrix* A, matrix* B, matrix* out){
	if ((A->n != B->m)||(A->m != out->m)||(B->n != out->n)){
		wprintf(L"(%d, %d) x (%d, %d) != (%d, %d)",A->m,A->n,B->n,B->m,out->m,out->n);
		perror("matrixMult error: sizes incorrect");
		return -1;
	}
	int am = A->m; int an = A->n; int bn = B->n; 

	for(int i = 0; i < am; i++){
	       for(int j = 0; j < bn; j++){
			for(int k = 0; k < an; k++){
				out->array[i*bn + j] += A->array[i*an + k] * B->array[k*bn + j];
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
	double* dpointer = dst->array;
	double* spointer = src->array;
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			dpointer[i*n + j] += spointer[i*n + j];
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
	double* dpointer = dst->array;
	double* spointer = src->array;
	for (int i = 0; i < m; i++){
		for (int j = 0; j < n; j++){
			dpointer[i * n + j] *= spointer[i*n + j];
		}	
	}
	return 0;
}

void matrixPrint(matrix* mat) {
	int m = mat->m; int n = mat->n;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            wprintf(L" %lf ", mat->array[i*n + j]);
        }
        wprintf(L"\n");
    }
}

void matrixFree(matrix* mat) {
    free(mat->array); // Free the array of rows
}


