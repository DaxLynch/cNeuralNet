#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
using namespace std;
class matrix{
public:
	matrix() = default;
	~matrix();
	matrix(int m, int n);
	int m = -1;
	int n = -1;
	float* array = NULL;
	friend ostream& operator<<(ostream& os, matrix const &mat);
private:


};
