typedef struct{
	double* array;
	int m; int n;
} matrix;

void initMat();
void matrixAllocate(matrix* mat, int m, int n);
int matrixCopy(matrix* dst, matrix* src);
void matrixRandFill(matrix* mat, int min, int max);
void matrixScalar(matrix* mat, double scalar);
int matrixTranspose(matrix* mat);
int matrixSigmoid(matrix* A);
int matrixSigmoidPrime(matrix* A);
int matrixMult(matrix* A, matrix* B, matrix* out);
int matrixParaMult(matrix* A, matrix* B, matrix* out);
int matrixAdd(matrix* dst, matrix* src);
int matrixHamProd(matrix* dst, matrix* src);
void matrixPrint(matrix* mat);
void matrixFree(matrix* mat);



