/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#if defined(_MSC_VER) || defined(__MINGW32__) //__MINGW32__ should goes before __GNUC__
  #define JL_SIZE_T_SPECIFIER    "%Iu"
  #define JL_SSIZE_T_SPECIFIER   "%Id"
  #define JL_PTRDIFF_T_SPECIFIER "%Id"
#elif defined(__GNUC__)
  #define JL_SIZE_T_SPECIFIER    "%zu"
  #define JL_SSIZE_T_SPECIFIER   "%zd"
  #define JL_PTRDIFF_T_SPECIFIER "%zd"
#else
  // TODO figure out which to use.
  #if NUMBITS == 32
    #define JL_SIZE_T_SPECIFIER    something_unsigned
    #define JL_SSIZE_T_SPECIFIER   something_signed
    #define JL_PTRDIFF_T_SPECIFIER something_signed
  #else
    #define JL_SIZE_T_SPECIFIER    something_bigger_unsigned
    #define JL_SSIZE_T_SPECIFIER   something_bigger_signed
    #define JL_PTRDIFF_T_SPECIFIER something-bigger_signed
  #endif
#endif

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

__device__ unsigned int bitreverse(unsigned int number) {
	number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
	number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
	number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
	return number;
}

/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 */
__global__ void bitreverse(void *data) {
	unsigned int *idata = (unsigned int*) data;
	idata[threadIdx.x] = bitreverse(idata[threadIdx.x]);
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {
	printf("Program begin...");
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	printf("\nDevice count: %i", deviceCount);
	for (int dev = 0; dev < deviceCount; dev++) {
		cudaDeviceProp deviceProp;
		CUDA_CHECK_RETURN(cudaGetDeviceProperties(&deviceProp, dev));
		if (dev == 0) {
			if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
				printf("\nNo CUDA GPU has been detected");
				return -1;
			} else if (deviceCount == 1) {
				//@@ WbLog is a provided logging API (similar to Log4J).
				//@@ The logging function wbLog takes a level which is either
				//@@ OFF, FATAL, ERROR, WARN, INFO, DEBUG, or TRACE and a
				//@@ message to be printed.
				printf("\nThere is 1 device supporting CUDA");
			} else {
				printf("\nThere are %i devices supporting CUDA", deviceCount);
			}
		}
		printf("\nDevice %i name: %s", dev, deviceProp.name);
		printf("\nComputational Capabilities: %i.%i",deviceProp.major,deviceProp.minor);
		printf("\nMaximum global memory size: "JL_SIZE_T_SPECIFIER, deviceProp.totalGlobalMem);
		printf("\nMaximum constant memory size: "JL_SIZE_T_SPECIFIER, deviceProp.totalConstMem);
		printf("\nMaximum shared memory size per block: %i",deviceProp.sharedMemPerBlock);
		printf("\nMaximum threads per block: %i",deviceProp.maxThreadsPerBlock);
		printf("\nMaximum block dimensions: %ix%ix%i", deviceProp.maxThreadsDim[0],
													deviceProp.maxThreadsDim[1],
													deviceProp.maxThreadsDim[2]);
		printf("\nMaximum grid dimensions: %ix%ix%i", deviceProp.maxGridSize[0],
												   deviceProp.maxGridSize[1],
												   deviceProp.maxGridSize[2]);
		printf("\nWarp size: %i",deviceProp.warpSize);
	}
	printf("\nProgram end...");
	return 0;
}
