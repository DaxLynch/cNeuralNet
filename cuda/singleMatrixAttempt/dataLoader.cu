#include "dataLoader.cuh"
using namespace std;
data::data(const std::string &name, int length): dataMatrix(784, length){
	dataLength = length;

	ifstream file;
	file.open(name);	
	file.seekg(16);
	
	char *charBuff = new char[length*784];
	file.read(charBuff, length*784);
	unsigned char *cudaCharBuff;
	cudaMalloc(&cudaCharBuff,sizeof(char) * length * 784);
	cudaMemcpy(cudaCharBuff, charBuff, length * 784, cudaMemcpyHostToDevice);
	
	dim3 threadsPerBlock(1024);
	dim3 blocks(ceil((float)length/1024.0f));
	castIntToFloat<<<blocks,threadsPerBlock>>>(dataMatrix.array, cudaCharBuff, length);
	cudaDeviceSynchronize();	
	cudaFree(cudaCharBuff);
	delete[] charBuff;
}


ostream& operator<<(ostream& os, const data& dat){
	return os << dat.dataMatrix;
}

void data::displayChar(unsigned char input){
	char ret[] = {' ', '-', '*', '#', '%', '@'};
	cout << ret[input/43];
}

void data::print(int entry){
	float temp[28*28];
	cudaMemcpy(temp, dataMatrix.array + entry*784, 28*28*sizeof(float), cudaMemcpyDeviceToHost);
	for(int i = 0; i < 28; i++){
		for(int j = 0; j < 28; j++){
			displayChar((unsigned char)(temp[i*28 + j]*255.0f  + 33.0f));
		}
		cout << endl;
	}
	cout << endl;
}


__global__ void castIntToFloat(float *dst, unsigned char *src, int length){
	int x = threadIdx.x + blockIdx.x * 1024.0f;
	if ( x < length){
		for(int i = 0; i < 28*28; i++){
			dst[x*784 + i] = ((float)src[x*784 + i] - 33.0f)/255.0f;
		}
	}
}
