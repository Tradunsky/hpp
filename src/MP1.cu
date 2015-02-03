#include <stdio.h>
#include <time.h>
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
		printf("\nError %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

void generateVector(float* vector, int len);
void printVector(char* message, float* vector, int len);

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
	printf("\nthreadIdx.x=%i, gridDim.x=%i, blockIdx.x=%i, blockIdx.x*blockDim.x=%i", threadIdx.x, gridDim.x, blockIdx.x, blockIdx.x*blockDim.x);
//	int indx = threadIdx.x+blockIdx.x*len;
	int indx = threadIdx.x+blockIdx.x*blockDim.x;
//	int indx = threadIdx.x;
  printf("\nA[%i]", indx);
  if (indx<len)
	  out[indx] = in1[indx] + in2[indx];
}

int main2(int argc, char **argv) {
  srand(time(NULL));
//  wbArg_t args;
  int inputLength = 5;
  int dataSize;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

//  args = wbArg_read(argc, argv);
  dataSize = sizeof(float) * inputLength;
//  wbTime_start(Generic, "Importing data and creating memory on host");
  printf("\nImporting data and creating memory on host");
//  hostInput1 = ( float * )wbImport(wbArg_getInputFile(args, 0), &inputLength);
//  hostInput2 = ( float * )wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostInput1 = (float*)malloc(dataSize);
  hostInput2 = (float*)malloc(dataSize);
  hostOutput = (float*)malloc(dataSize);
  generateVector(hostInput1, inputLength);
  printVector("Vector 1", hostInput1, inputLength);
  generateVector(hostInput2, inputLength);
  printVector("Vector 2", hostInput2, inputLength);
//  wbTime_stop(Generic, "Importing data and creating memory on host");
  printf("\nImporting data and creating memory on host");
//  wbLog(TRACE, "The input length is ", inputLength);
  printf("\nThe input length is %i", inputLength);

//  wbTime_start(GPU, "Allocating GPU memory.");
  printf("\nAllocating GPU memory.");
  //@@ Allocate GPU memory here
  CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceInput1, dataSize));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceInput2, dataSize));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceOutput, dataSize));
//  wbTime_stop(GPU, "Allocating GPU memory.");
  printf("\nAllocating GPU memory.");

//  wbTime_start(GPU, "Copying input memory to the GPU.");
  printf("\nCopying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  CUDA_CHECK_RETURN(cudaMemcpy(deviceInput1, hostInput1, dataSize, cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN(cudaMemcpy(deviceInput2, hostInput2, dataSize, cudaMemcpyHostToDevice));
//  wbTime_stop(GPU, "Copying input memory to the GPU.");
  printf("\nCopying input memory to the GPU.");
  //@@ Initialize the grid and block dimensions here
  cudaDeviceProp deviceProp;
  CUDA_CHECK_RETURN(cudaGetDeviceProperties(&deviceProp, 0));
	  int warpSize = deviceProp.warpSize;
	  int maxGridSize = deviceProp.maxGridSize[0];
	  int warpCount = (inputLength / warpSize) + (((inputLength % warpSize) == 0) ? 0 : 1);
	  int warpPerBlock = max(1, min(4, warpCount));
      int threadCount = min( maxGridSize, max(1, warpCount/warpPerBlock) );
      int blockCount = (inputLength+threadCount-1)/threadCount;
//  wbTime_start(Compute, "Performing CUDA computation");
  printf("\nPerforming CUDA computation");
  //@@ Launch the GPU Kernel here
  vecAdd<<<blockCount, threadCount>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  CUDA_CHECK_RETURN(cudaDeviceSynchronize());
//  wbTime_stop(Compute, "Performing CUDA computation");
  printf("\nPerforming CUDA computation");

//  wbTime_start(Copy, "Copying output memory to the CPU");
  printf("\nCopying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  CUDA_CHECK_RETURN(cudaMemcpy(hostOutput, deviceOutput, dataSize, cudaMemcpyDeviceToHost));
//  wbTime_stop(Copy, "Copying output memory to the CPU");
  printf("\nCopying output memory to the CPU");

  printVector("Result vector", hostOutput, inputLength);

//  wbTime_start(GPU, "Freeing GPU Memory");
  printf("\nFreeing GPU Memory");
  //@@ Free the GPU memory here
  CUDA_CHECK_RETURN(cudaFree((float*) deviceInput1));
  CUDA_CHECK_RETURN(cudaFree((float*) deviceInput2));
  CUDA_CHECK_RETURN(cudaFree((float*) deviceOutput));
  CUDA_CHECK_RETURN(cudaDeviceReset());
//  wbTime_stop(GPU, "Freeing GPU Memory");
  printf("\nFreeing GPU Memory");

//  wbSolution(args, hostOutput, inputLength);

  printf("\nFreeing CPU Memory");
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  printf("\nFreeing CPU Memory");
  return 0;
}

void generateVector(float* vector, int len){
	for(int i=0; i<len; i++){
		vector[i] = rand()%100;
	}
}

void printVector(char* message, float* vector, int len){
	printf("\n%s: ", message);
	for(int i=0; i<len; i++){
		printf("%10.2f, ", vector[i]);
	}
}
