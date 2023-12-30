typedef struct{
	float* array;
	int m; int n;
	int inHost;
} matrix;

void initMat();
void matrixAllocate(matrix* mat, int m, int n);
int matrixCopy(matrix* dst, matrix* src);
void matrixRandFill(matrix* mat);
void matrixScalar(matrix* mat, float scalar);
__global__ void matscalar(float* A, float scalar, int m, int n);
void matrixSgnScalar(matrix* mat, float scalar);
__global__ void matsgnscalar(float* A, float scalar, int m, int n);
int matrixTranspose(matrix* mat);
__global__ void mattrans(float* dst, float* src, int m, int n);
int matrixSigmoid(matrix* A);
__global__ void matsig(float* dst, int m, int n);
int matrixSigmoidPrime(matrix* A);
__global__ void matsigp(float* dst, int m, int n);
int matrixMultNoDelete(matrix* A, matrix* B, matrix* out);
__global__ void matmultD(float *A, float* B, float* C, int m, int q, int n);
int matrixMultTransFirstNoDelete(matrix* A, matrix* B, matrix* out);
__global__ void matmulttrans1D(float* A, float* B, float* C, int m, int q, int n);
int matrixMultTransSecondNoDelete(matrix* A, matrix* B, matrix* out);
__global__ void matmulttrans2D(float* A, float* B, float* C, int m, int q, int n);
int matrixMult(matrix* A, matrix* B, matrix* out);
__global__ void matmult(float *A, float* B, float* C, int m, int q, int n);
int matrixMultTransFirst(matrix* A, matrix* B, matrix* out);
__global__ void matmulttrans1(float* A, float* B, float* C, int m, int q, int n);
int matrixMultTransSecond(matrix* A, matrix* B, matrix* out);
__global__ void matmulttrans2(float* A, float* B, float* C, int m, int q, int n);
int matrixAdd(matrix* dst, matrix* src);
__global__ void matadd(float *A, float* B, int m, int n);
int matrixHamProd(matrix* dst, matrix* src);
__global__ void matham(float* A, float* B, int m, int n);
void matrixPrint(matrix* mat);
int matrixIsZero(matrix* mat);
void matrixFree(matrix* mat);



