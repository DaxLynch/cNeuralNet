typedef struct{
	float** array;
	int m; int n;
} matrix;

void initMat(){
	srand(time(NULL));
};

void allocateArray(matrix* mat, int m, int n){
	mat->m = m;
	mat->n = n;
	mat->array = (float**)malloc(sizeof(float*) * m);
	for( int i = 0; i < m; i++){
		mat->array[i] = (float*)malloc(sizeof(float) * n);
	}	
}

void randFill(matrix* mat, int max){
	for(int i = 0; i < mat->m; i++){
		for(int j = 0; j < mat->n; j++){
			mat->array[i][j] = (float)rand()/(float)(RAND_MAX/max);
		}
	}
}

int matMult(matrix* A, matrix* B, matrix* out){
	if (A->n != B->m){
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

void printMatrix(matrix* mat) {
    for (int i = 0; i < mat->m; i++) {
        for (int j = 0; j < mat->n; j++) {
            printf("%f\t", mat->array[i][j]);
        }
        printf("\n");
    }
}

void freeArray(matrix* mat) {
    for (int i = 0; i < mat->m; i++) {
        free(mat->array[i]); // Free each row
    }
    
    free(mat->array); // Free the array of rows
}
	



