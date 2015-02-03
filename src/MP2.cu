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

void generateMatrix(float*& matrixPtr, int numRows, int numColumns);
void printMatrix(float* matrixPtr, int numRows, int numColumns);
void printMatrix(char* message, float* matrixPtr, int numRows, int numColumns);

// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows, int numBColumns,
                               int numCRows, int numCColumns) {
	int col = threadIdx.x+blockIdx.x*blockDim.x;
	int row = threadIdx.y+blockIdx.y*blockDim.y;

	if ((row<numARows) && (col<numBColumns)){
		float c = 0.0;
		printf("\n");
		for(int i =0; i<numBRows; i++){
			c += A[row*numBRows+i] * B[col+i*numBColumns];
			printf("A[%i]*B[%i]=%10.2f + ",row*numBRows+i,col+i*numBColumns, c);
		}
		C[row*numBColumns+col] = c;
		printf("\nA[%i]=%10.2f", row*numBColumns+col, C[row*numBColumns+col]);
	}
  //@@ Insert code to implement matrix multiplication here
//	С[i,k] = A[i,k]
}

int main(int argc, char **argv) {
	srand(time(NULL));
//  wbArg_t args;
  unsigned int TILE_WIDTH = 16;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)
  int sizeA;
  int sizeB;
  int sizeC;

//  args = wbArg_read(argc, argv);

//  wbTime_start(Generic, "Importing data and creating memory on host");
//  hostA =
//      ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
//  hostB =
//      ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  printf("\nCreating memory on host and generating data");
  numARows = 3;
  numAColumns = 4;
  numBRows = numAColumns;
  numBColumns = numARows;
  generateMatrix(hostA, numARows, numAColumns);
  printMatrix(
		  "A",
		  hostA, numARows, numAColumns);
  generateMatrix(hostB, numBRows, numBColumns);
  printMatrix(
		  "B",
		  hostB, numBRows, numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  sizeA = numARows*numAColumns*sizeof(float);
  sizeB = numBRows*numBColumns*sizeof(float);
  sizeC = numCRows*numCColumns*sizeof(float);
  //@@ Allocate the hostC matrix
//  generateMatrix(hostC, numCRows, numCColumns);
  hostC = (float*) malloc(sizeC);
  printf("\nCreated memory on host and generated data");

  printf("\nThe dimensions of A are %ix%i", numARows, numAColumns);
  printf("\nThe dimensions of B are %ix%i", numBRows, numBColumns);

  printf("\nAllocating GPU memory.");
  //@@ Allocate GPU memory here
  CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceA, sizeA));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceB, sizeB));
  CUDA_CHECK_RETURN(cudaMalloc((void **)&deviceC, sizeC));
  printf("\nAllocating GPU memory.");

  printf("\nCopying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  CUDA_CHECK_RETURN(cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN(cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice));
  printf("\nCopying input memory to the GPU.");
  //@@ Initialize the grid and block dimensions here
  //n = countAColumn, m = countARows;
  //k = countBColumn, n = countBRows;
  //k = countCColumn, m = countCRows
//  unsigned int blockX = (numARows-1)/numCRows+1;
//  unsigned int blockY = (numBRows-1)/numCRows+1;
  unsigned int blockX = (numBColumns + TILE_WIDTH - 1) / TILE_WIDTH;
  unsigned int blockY = (numARows + TILE_WIDTH - 1) / TILE_WIDTH;
  dim3 blockCount(blockX, blockY);
  dim3 threadCount(TILE_WIDTH, TILE_WIDTH);
  printf("\nPerforming CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<blockCount, threadCount>>>(deviceA,
		  deviceB,
		  deviceC,
		  numARows,
		  numAColumns,
		  numBRows,
		  numBColumns,
		  numCRows,
		  numCColumns);
  cudaDeviceSynchronize();
  printf("\nPerforming CUDA computation");

  printf("\nCopying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  CUDA_CHECK_RETURN(cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost));
  printf("\nCopying output memory to the CPU");

  printf("\nFreeing GPU Memory");
  //@@ Free the GPU memory here
  CUDA_CHECK_RETURN(cudaFree((float*) deviceA));
  CUDA_CHECK_RETURN(cudaFree((float*) deviceB));
  CUDA_CHECK_RETURN(cudaFree((float*) deviceC));
  CUDA_CHECK_RETURN(cudaDeviceReset());
  printf("\nFreeing GPU Memory");

//  wbSolution(args, hostC, numCRows, numCColumns);

  printMatrix("\nResult hostC:\n", hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}

void generateMatrix(float*& matrixPtr, int numRows, int numColumns){
	matrixPtr = (float*) malloc(numRows*numColumns*sizeof(float));
	for(int i=0; i<numRows; i++){
		for(int j=0; j<numColumns; j++){
			matrixPtr[i*numColumns+j] = rand()%100;
			//printf("\nmatrix[%i]=%10.2f", i*numColumns+j, matrixPtr[i*numColumns+j]);
		}
	}
}

void printMatrix(float* matrixPtr, int numRows, int numColumns){
	for(int i=0; i<numRows; i++){
		for(int j=0; j<numColumns; j++){
			printf("\nmatrix[%i]=%10.2f", i*numColumns+j, matrixPtr[i*numColumns+j]);
		}
	}
}

void printMatrix(char* message, float* matrix, int numRows, int numColumns){
	printf("\n%s:\n", message);
	for(int i=0; i<numRows; i++){
		for(int j=0; j<numColumns; j++){
			printf(" %10.2f, ", matrix[i*numColumns+j]);
		}
		printf("\n");
	}
}
