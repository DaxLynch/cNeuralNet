#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <wchar.h>
#include <locale.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdarg.h>
#include "matrix.cuh"
#include "dataLoader.cuh"
#include "network.cuh"
#include "statistics.cuh"
#include "evaluate.cuh"
#define CUDA_SAFE_CALL(call)                                   \
do { \
    call; cudaDeviceSynchronize; \
    cudaError_t err = cudaGetLastError(); \
    if (cudaSuccess != err) {                                         \
        wprintf (L"Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#include "evaluate.cu"
#include "backprop.cu"
#include "statistics.cu"
#include "matrix.cu"
#include "dataLoader.cu"
#include "network.cu"
