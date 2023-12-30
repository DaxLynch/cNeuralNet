void initMat(){
	srand(time(NULL));
}

void matrixAllocate(matrix* mat, int m, int n){
	
	mat->m = m;
	mat->n = n;
	cudaMalloc(&mat->array, sizeof(float)*m*n);
	
}

int matrixCopy(matrix* dst, matrix* src){
	
	if ((dst->n != src->n) || (dst->m != src->m)){
		printf("(%d, %d) != (%d, %d)\n",dst->m,dst->n,src->m,src->n);
		printf("matrixCopy wrong dimension\n");
		return -1;
	}
	int m = dst->m; int n = dst->n;
	
	cudaMemcpy(dst->array, src->array, m*n * sizeof(float), cudaMemcpyDeviceToDevice);
	
	return 0;
}

void matrixRandFill(matrix* mat){ //Do not call if anything has been initilized ther budrick
	
	int m = mat->m; int n = mat->n;
	float* temp = (float*)malloc(sizeof(float)*m*n);
 	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			temp[i*n + j] = gaussian();
		}
	}
	cudaMemcpy(mat->array, temp, sizeof(float)*m*n,cudaMemcpyHostToDevice);
	free(temp);
	
	return;
}

void matrixScalar(matrix* mat, float scalar){
	
	int m = mat->m; int n = mat->n;
	dim3 threadsPerBlock(32,32);
	dim3 numBlocks(ceil(float(n) / 32.0f), ceil(float(m) / 32.0f));
	matscalar<<<numBlocks, threadsPerBlock>>>(mat->array, scalar, m, n);
	
}
__global__ void matscalar(float* A, float scalar, int m, int n){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if((x < n) && (y < m)){
		A[y*n + x] *= scalar;
	}
}

int matrixTranspose(matrix* mat){
	int m = mat->m; int n = mat->n;
	float* newArray = NULL;
	cudaMalloc(&newArray, sizeof(float) * m*n);
	dim3 threadsPerBlock(32,32);
	dim3 numBlocks(ceil(float(n)/32.0f), ceil(float(m)/32.0f));

	mattrans<<<numBlocks, threadsPerBlock>>>(newArray, mat->array, m,n);
	mat->n = mat->m;
	mat->m = n;
	cudaFree(mat->array);
	mat->array = newArray;
	
	return 0;
}
__global__ void mattrans(float* dst, float* src, int m, int n){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if((x < n) && (y < m)){
		dst[x*m + y] = src[y*n + x];
	}
}

int matrixSigmoid(matrix* A){
	
	if (A->n != 1){
		perror("Cannot do sigmoid");
		return -1;
	};
	dim3 threadsPerBlock(32,32);
	dim3 numBlocks(ceil(float(A->m)/32.0f), ceil(float(A->n)/32.0f));
	
	matsig<<<numBlocks, threadsPerBlock>>>(A->array, A->m,A->n);
	
	return 0;
}
__global__ void matsig(float* dst, int m, int n){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if((x < n) && (y < m)){
		dst[y*n + x] = 1.0f/(1.0f + exp(-dst[y*n + x]));
	}
}

int matrixSigmoidPrime(matrix* A){
	
	if (A->n != 1){
		perror("Cannot do sigmoid prime");
		return -1;
	};
	dim3 threadsPerBlock(32,32);
	dim3 numBlocks(ceil(float(A->m)/32.0f), ceil(float(A->n)/32.0f));

	matsigp<<<numBlocks, threadsPerBlock>>>(A->array, A->m,A->n);
	return 0;
}
__global__ void matsigp(float* dst, int m, int n){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if((x < n) && (y < m)){
		float temp = dst[y*n + x];
		dst[y*n + x] = 1.0f/(2.0f + exp(-temp) + exp(temp));
	}
}

int matrixMultTransFirst(matrix* A, matrix* B, matrix* out){
	if ((A->m != B->m)||(A->n != out->m)||(B->n != out->n)){
		printf("(%d, %d)T x (%d, %d) != (%d, %d)",A->m,A->n,B->n,B->m,out->m,out->n);
		perror("matrixMult error: sizes incorrect");
		return -1;
	}
	int am = A->n;
       	int an = A->m;
       	int bn = B->n; 
	
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(ceil(float(bn) / 32.0f), ceil(float(am) / 32.0f));
	
	matmulttrans1<<<numBlocks, threadsPerBlock>>>(A->array,B->array,out->array,am,an,bn);
	
	return 0;
}

__global__ void matmulttrans1(float* A, float* B, float* C, int m, int q, int n){ //basic implementation
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float temp = 0;
	if((x < n) && (y < m)){
		for(int i = 0; i < q; i++){
			temp += A[i*m + y] * B[i*n + x];	
		}
	C[y*n + x] = temp;
	}
}

int matrixMult(matrix* A, matrix* B, matrix* out){
	
	if ((A->n != B->m)||(A->m != out->m)||(B->n != out->n)){
		printf("(%d, %d) x (%d, %d) != (%d, %d)",A->m,A->n,B->n,B->m,out->m,out->n);
		perror("matrixMult error: sizes incorrect");
		return -1;
	}
	
	int am = A->m;
       	int an = A->n;
       	int bn = B->n; 
	
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(ceil(float(bn) / 32.0f), ceil(float(am) / 32.0f));
	
	matmult<<<numBlocks, threadsPerBlock>>>(A->array,B->array,out->array,am,an,bn);
	
	return 0;
}

__global__ void matmult(float* A, float* B, float* C, int m, int q, int n){ //basic implementation
	int x = blockIdx.x * blockDim.x + threadIdx.x ;
	int y = blockIdx.y * blockDim.y + threadIdx.y ;

	A += (blockIdx.y * 32  + threadIdx.y) * q  + threadIdx.x;
	B += (blockIdx.y * 32 + threadIdx.y) * n + (blockIdx.x * 32) + threadIdx.x ;
	C += (blockIdx.y * 32 * n) * (blockIdx.x * 32);
	
	float temp = 0;
		__shared__ float As[32*32];
		__shared__ float Bs[32*32];
		int i = 0;
		while( i < q - 31 ){
			As[threadIdx.y * 32 + threadIdx.x] = A[0];
			Bs[threadIdx.x * 32 + threadIdx.y] = B[0];
			__syncthreads();
			A+= 32;
			B += n;	

			for(int j = 0; j < 32; j++){
				temp += As[threadIdx.y * 32 + j] * Bs[threadIdx.x * 32 + j];	
			}
			i+= 32;
			__syncthreads();
		}

		if (i != q){
			As[threadIdx.y * 32 + threadIdx.x] = A[0];
			Bs[threadIdx.x * 32 + threadIdx.y] = B[0];
			__syncthreads();

			for(int j = 0; j < (q - i); j++){	
				temp += As[threadIdx.y * 32 + j] * Bs[ threadIdx.x * 32 + j];	
			}
		}				

			
		if((x < n) && (y < m)){
			C[y*n + x] = temp;	
		}
}

int matrixMultTransSecond(matrix* A, matrix* B, matrix* out){
	if ((A->n != B->n)||(A->m != out->m)||(B->m != out->n)){
		printf("(%d, %d) x (%d, %d)T != (%d, %d)",A->m,A->n,B->n,B->m,out->m,out->n);
		perror("matrixMult error: sizes incorrect");
		return -1;
	}
	int am = A->m;
       	int an = A->n;
       	int bn = B->m; 
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(ceil(float(bn) / 32.0f), ceil(float(am) / 32.0f));
	matmulttrans2<<<numBlocks, threadsPerBlock>>>(A->array,B->array,out->array,am,an,bn);
	return 0;
}

__global__ void matmulttrans2(float* A, float* B, float* C, int m, int q, int n){ //basic implementation
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float temp = 0;
	if((x < n) && (y < m)){
		for(int i = 0; i < q; i++){
			temp += A[y*q + i] * B[x*q + i];	
		}
	C[y*n + x] = temp;
	}
}

int matrixMultTransFirstNoDelete(matrix* A, matrix* B, matrix* out){
	if ((A->m != B->m)||(A->n != out->m)||(B->n != out->n)){
		printf("(%d, %d)T x (%d, %d) != (%d, %d)",A->m,A->n,B->n,B->m,out->m,out->n);
		perror("matrixMult error: sizes incorrect");
		return -1;
	}
	int am = A->n;
       	int an = A->m;
       	int bn = B->n; 

	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(ceil(float(bn) / 32.0f), ceil(float(am) / 32.0f));
	
	matmulttrans1D<<<numBlocks, threadsPerBlock>>>(A->array,B->array,out->array,am,an,bn);
	
	return 0;
}

__global__ void matmulttrans1D(float* A, float* B, float* C, int m, int q, int n){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float temp = 0;
	if((x < n) && (y < m)){
		for(int i = 0; i < q; i++){
			temp += A[i*m + y] * B[i*n + x];	
		}
	C[y*n + x] += temp;
	}
}

int matrixMultNoDelete(matrix* A, matrix* B, matrix* out){
	
	if ((A->n != B->m)||(A->m != out->m)||(B->n != out->n)){
		printf("(%d, %d) x (%d, %d) != (%d, %d)",A->m,A->n,B->n,B->m,out->m,out->n);
		perror("matrixMult error: sizes incorrect");
		return -1;
	}
	
	int am = A->m;
       	int an = A->n;
       	int bn = B->n; 
	
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(ceil(float(bn) / 32.0f), ceil(float(am) / 32.0f));
	
	matmultD<<<numBlocks, threadsPerBlock>>>(A->array,B->array,out->array,am,an,bn);
	
	return 0;
}

__global__ void matmultD(float* A, float* B, float* C, int m, int q, int n){ //basic implementation
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float temp = 0;
	if((x < n) && (y < m)){
		for(int i = 0; i < q; i++){
			temp += A[y*q + i] * B[i*n + x];	
		}
	C[y*n + x] = temp;
	
	}
}

int matrixMultTransSecondNoDelete(matrix* A, matrix* B, matrix* out){
	if ((A->n != B->n)||(A->m != out->m)||(B->m != out->n)){
		printf("(%d, %d) x (%d, %d)T != (%d, %d)",A->m,A->n,B->n,B->m,out->m,out->n);
		perror("matrixMult error: sizes incorrect");
		return -1;
	}

	int am = A->m;
       	int an = A->n;
       	int bn = B->m; 
	
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(ceil(float(bn) / 32.0f), ceil(float(am) / 32.0f));
	
	matmulttrans2D<<<numBlocks, threadsPerBlock>>>(A->array,B->array,out->array,am,an,bn);
	
	return 0;
}

__global__ void matmulttrans2D(float* A, float* B, float* C, int m, int q, int n){ //basic implementation
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float temp = 0;
	if((x < n) && (y < m)){
		for(int i = 0; i < q; i++){
			temp += A[y*q + i] * B[x*q + i];	
		}
	C[y*n + x] += temp;
	}
}


int matrixAdd(matrix* dst, matrix* src){ //in place,a s opposed to matrix multiply???
	
	if ((dst->n != src->n) || (dst->m != src->m)){
		perror("matrixAdd wrong dimension");
		return -1;
	}
	int m = dst->m; int n = dst->n;
	
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(ceil(float(n) / 32.0f), ceil(float(m) / 32.0f));

	matadd<<<numBlocks, threadsPerBlock>>>(dst->array,src->array,m,n);
	
	return 0;
}

__global__ void matadd(float* A, float* B, int m, int n){ //basic implementation
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if((x < n) && (y < m)){
		A[y*n + x] += B[y*n + x];	
	}
}

int matrixHamProd(matrix* dst, matrix* src){ //in place,a s opposed to matrix multiply???
	
	if ((dst->n != src->n) || (dst->m != src->m)){
		perror("matrixAdd wrong dimension");
		return -1;
	}
	int m = dst->m; int n = dst->n;
	
	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks(ceil(float(n) / 32.0f), ceil(float(m) / 32.0f));

	matham<<<numBlocks, threadsPerBlock>>>(dst->array,src->array,m,n);
	
	return 0;
}

__global__ void matham(float* A, float* B, int m, int n){ //basic implementation
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if((x < n) && (y < m)){
		A[y*n + x] *= B[y*n +x];	
	}
}

void matrixPrint(matrix* mat) {
	cudaDeviceSynchronize();	
	int m = mat->m; int n = mat->n;
	float* temp = (float*)malloc(sizeof(float)*m*n);
	
	cudaMemcpy(temp, mat->array, sizeof(float)*m*n,cudaMemcpyDeviceToHost);
	printf("m: %d n: %d \n", m,n);
    	for (int i = 0; i < m; i++) {
        	for (int j = 0; j < n; j++) {
        	  	  printf(" %f ", temp[i*n + j]);
        	}
        	printf("\n");
    	}
	free(temp);
	
}



int matrixNonZeros(matrix* mat) {
	cudaDeviceSynchronize();	
	int m = mat->m; int n = mat->n;
	float* temp = (float*)malloc(sizeof(float)*m*n);
	
	cudaMemcpy(temp, mat->array, sizeof(float)*m*n,cudaMemcpyDeviceToHost);
	int nonzeros = 0;
    	for (int i = 0; i < m; i++) {
        	for (int j = 0; j < n; j++) {
        		if(temp[i*n + j]){
				nonzeros++;
			}	
		}
    	}
	free(temp);
	printf("%d non zero \n", nonzeros);
	return nonzeros;
}
int matrixEqual(matrix* A, matrix* B) {
	cudaDeviceSynchronize();	
	int m = A->m; int n = A->n;
	float* temp1 = (float*)malloc(sizeof(float)*m*n);
	float* temp2 = (float*)malloc(sizeof(float)*m*n);
	
	cudaMemcpy(temp1, A->array, sizeof(float)*m*n,cudaMemcpyDeviceToHost);
	cudaMemcpy(temp2, B->array, sizeof(float)*m*n,cudaMemcpyDeviceToHost);
	int nonequals = 0;
    	for (int i = 0; i < m; i++) {
        	for (int j = 0; j < n; j++) {
        		if(temp1[i*n + j] - temp2[i*n + j]  >  .01f ){
				printf("i: %d, j: %d, ,%f != %f \n", i,j,temp1[i * n + j], temp2[i * n + j]);
				nonequals++;
			}	
		}
    	}
	free(temp1);
	free(temp2);
	printf("%d non equal \n", nonequals);
	return nonequals;
}
void matrixFree(matrix* mat) {
	
    cudaFree(mat->array); // Free the array of rows
	
}


