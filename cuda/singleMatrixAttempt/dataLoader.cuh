#pragma once
#include "matrix.cuh"
#include <string>
#include <iostream>
#include <fstream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

class data{
public:
	data(const std::string &name, int length);
	data();

	matrix dataMatrix;
	int dataLength;

	friend ostream& operator<<(ostream& os, const data& dat);
	void displayChar(unsigned char input);
	void print(int entry);
private:
		
};

__global__ void castIntToFloat(float *src, unsigned char *dst, int length);
