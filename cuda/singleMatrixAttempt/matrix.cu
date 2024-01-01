#include "matrix.cuh"
using namespace std;
matrix::matrix(int rows, int cols){
	m = rows;
	n = cols;
	cudaMalloc(&array, m*n*sizeof(float));
};
matrix::~matrix(){
	cudaFree(array);
}
ostream& operator<<(ostream& os, matrix const &mat){
	int m = mat.m; int n = mat.n;
	float* temp = new float[m*n];
	cudaMemcpy(temp, mat.array, m*n*sizeof(float), cudaMemcpyDeviceToHost);
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			os << temp[   j*m   +   i] << ' ';
		}
		os << '\n';
	}
	return os;
}




