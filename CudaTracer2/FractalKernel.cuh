#ifndef H_FRACTALKERNEL
#define H_FRACTALKERNEL

#include <windows.h> 
#include <iostream>
#include <memory>

#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "device_launch_parameters.h"

constexpr float EPSILON = 1e-3f;
constexpr float INF = 3.402823466e+38F;

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


#endif