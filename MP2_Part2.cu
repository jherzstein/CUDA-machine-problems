#include "cuda_runtime.h"
#include <stdio.h>

const int MATRIX_WIDTH[]={450, 500, 1350, 1150};
const int MATRIX_LENGTH[]={400, 450, 1200, 1350};
#define TILE_WIDTH 9 
#define TILE_LENGTH 16
int numElementsMatrix = sizeof(MATRIX_WIDTH) / sizeof(int);
int numElementsTileWidth = sizeof(TILE_WIDTH) / sizeof(int);

// test flag
int test_flag = 0;

// Number of datapoints for each MATRIX_WIDTH and TILE_WIDTH
int data_points = 5;

// Function to measure time
float getElapsedTime(cudaEvent_t start, cudaEvent_t end) {
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, end);
  return (float)elapsedTime;
}

// Tile Matrix Multiplication
__global__ void tiledMatrixMultiplication(float *A, float *B, float *C, float size, int M, int K, int N) {
  __shared__ float tile1[TILE_WIDTH][TILE_LENGTH];
  __shared__ float tile2[TILE_WIDTH][TILE_LENGTH];
  
  int row = blockIdx.y * TILE_LENGTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  
  int value = 0; int location;
  
  
  for (int i = 0; i< K / TILE_LENGTH; ++i) {
    
    location = (i * TILE_LENGTH + threadIdx.y )* N + col; 
    //if (location >= size*size) 
    //  tile2[threadIdx.y][threadIdx.x] = 0;
    //else
      tile2[threadIdx.y][threadIdx.x] = B[location];
    
    
    
    location = row * K + i * TILE_WIDTH + threadIdx.x;
    //if (location >= size*size)
    //  tile1[threadIdx.y][threadIdx.x] = 0;
    //else
      tile1[threadIdx.y][threadIdx.x] = A[location]; 
    
    __syncthreads(); 
    
    for (int j = 0; j < TILE_WIDTH; j++)
      value += tile1[threadIdx.y][j] * tile2[j][threadIdx.x];
    
    __syncthreads(); 
    
    
  }
  
  //if (row < size && col < size) {
    int temp;
    temp = row * size + col;
    C[temp] = temp;
    //}
}

// Matrix Multiplication for host (CPU)
void hostMultiplication (float *A, float *B, float *C, int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      float temp = 0;
      for (int k = 0; k < size; k++) {
        temp += A[i*size + k] * B[k*size + j];
      }
      C[i*size + j] = temp;
    }
  }
}

void GPUDeviceProperties(int device) {
  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, device);
  printf("Device %d: %s \n", device, properties.name);
  printf("Question 1: \n");
  printf("-> Streaming Multiprocessors (SM): %d \n", properties.multiProcessorCount);
  printf("-> Max threads per multiprocessor: %d \n", properties.maxThreadsPerMultiProcessor);
  printf("-> Total Threads: %d \n", properties.maxThreadsPerMultiProcessor*properties.multiProcessorCount);

  cudaFuncAttributes attributes;
  cudaFuncGetAttributes(&attributes, tiledMatrixMultiplication);
  printf("Question 2: \n");
  printf("-> Number of Registers: %d \n", attributes.numRegs);
  printf("-> Shared Memory Size: %d \n", attributes.sharedSizeBytes);
  printf("-> Number of Blocks per Streaming Multiprocessor: %d \n", properties.maxBlocksPerMultiProcessor);
  printf("-> Max threads per block: %d \n", attributes.maxThreadsPerBlock);
}

int main() {
  // Properties for questions 1 and 2
  GPUDeviceProperties(0);

  // Initialize CUDA events for timing
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaDeviceSynchronize();
  
  printf("Printing as CSV: \n");
  printf("N, MATRIX_WIDTH, TILE_WIDTH, HM (ms), TM (ms), Results \n");

  for(int i=0; i<numElementsMatrix/2; i++) { // Looping through Matrix sizes
    for(int n=0; n<numElementsTileWidth; n++) { // Looping through Block widths 
      // Setting average times
      float totalTileMultiplicationTime = 0;
      float totalHostMultiplicationTime = 0;
      float totalDeviceToHostTime = 0;
      float totalHostToDeviceTime = 0;
      for(int m=0; m<data_points; m++) { // Looping for extra data
	// Check if device multiplication and host multiplication are equal
	size_t size1 = MATRIX_WIDTH[i] * MATRIX_LENGTH[i] * sizeof(float);
	size_t size2 = MATRIX_WIDTH[i+1] * MATRIX_LENGTH[i+1] * sizeof(float);
	size_t size3 = MATRIX_WIDTH[i+1] * MATRIX_LENGTH[i] * sizeof(float);
	float* hostMatrixM = (float*)malloc(size1);
	float* hostMatrixN = (float*)malloc(size2);
	float* hostMatrixP = (float*)malloc(size3);
	float* hostMatrixP1 = (float*)malloc(size3);

	// Loading random values into host matrices
	for (int j = 0; j < MATRIX_WIDTH[i]; j++) {
	  for (int k = 0; k < MATRIX_LENGTH[i]; k++) {
	    float value = rand() % 10;
	    *(hostMatrixM + j * MATRIX_WIDTH[i] + k) = value;
	  }
	}
	for (int j = 0; j < MATRIX_WIDTH[i+1]; j++) {
	  for (int k = 0; k < MATRIX_LENGTH[i+1]; k++) {
	    float value = rand() % 10;
	    *(hostMatrixN + j * MATRIX_WIDTH[i+1] + k) = value;
	  }
	}
	
	// Allocate memory for matrices on device
	float* deviceMatrixM;
	float* deviceMatrixN;
	float* deviceMatrixP;
	cudaMalloc(&deviceMatrixM, size1);
	cudaMalloc(&deviceMatrixN, size2);
	cudaMalloc(&deviceMatrixP, size3);

	// Transfer matrices from host to device
	cudaEventRecord(start);
	cudaDeviceSynchronize();
	cudaMemcpy(deviceMatrixM, hostMatrixM, size1, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMatrixN, hostMatrixN, size2, cudaMemcpyHostToDevice);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float hostToDeviceTime = getElapsedTime(start, end);
	totalHostToDeviceTime += hostToDeviceTime;
	//printf("Host to Device Transfer Time: %.6f ms\n", hostToDeviceTime);
	
	// Transfer matrices back from device to host
	cudaEventRecord(start);
	cudaDeviceSynchronize();
	cudaMemcpy(hostMatrixM, deviceMatrixM, size1, cudaMemcpyDeviceToHost);
	cudaMemcpy(hostMatrixN, deviceMatrixN, size2, cudaMemcpyDeviceToHost);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float deviceToHostTime = getElapsedTime(start, end);
	totalDeviceToHostTime += deviceToHostTime;
	//printf("Device to Host Transfer Time: %.6f ms\n", deviceToHostTime);

	int numBlocksWidth = MATRIX_WIDTH[i]/TILE_WIDTH;
	if(MATRIX_WIDTH[i] % TILE_WIDTH) numBlocksWidth++;
	int numBlocksLength = MATRIX_LENGTH[i]/TILE_LENGTH;
	if(MATRIX_LENGTH[i] % TILE_LENGTH) numBlocksLength++;
	dim3 dimBlock(numBlocksWidth, numBlocksLength);
	//dim3 dimGrid(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid(TILE_WIDTH,TILE_LENGTH);
	//dim3 dimBlock(MATRIX_WIDTH[i], MATRIX_WIDTH[i], 1);

	cudaEventRecord(start);
	tiledMatrixMultiplication <<<dimBlock, dimGrid>>> (deviceMatrixN, deviceMatrixM, deviceMatrixP, MATRIX_WIDTH[i], MATRIX_LENGTH[i], MATRIX_WIDTH[i], MATRIX_WIDTH[i+1]);
	//matrixMultiplication <<<1, 1>>> (deviceMatrixA, deviceMatrixB, deviceMatrixC, MATRIX_WIDTH[i]);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaMemcpy(hostMatrixP1, deviceMatrixP, size3, cudaMemcpyDeviceToHost);
	float deviceMultiplicationTime = getElapsedTime(start, end);
	totalTileMultiplicationTime += deviceMultiplicationTime;

	cudaEventRecord(start);
	cudaDeviceSynchronize();
	hostMultiplication(hostMatrixN,hostMatrixM,hostMatrixP,MATRIX_WIDTH[i]);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float hostMultiplicationTime = getElapsedTime(start, end);
	totalHostMultiplicationTime += hostMultiplicationTime;

	for (int x = 0; x < MATRIX_WIDTH[i]; x++) {
	  for (int y = 0; y < MATRIX_WIDTH[i]; y++) {
	    if (hostMatrixP[x * MATRIX_WIDTH[i] + y] != hostMatrixP1[ x * MATRIX_WIDTH[i] + y]) {
	      test_flag=1;
	    }
	  }
	}

	printf("%d, %d, %d, %.6f, %.6f", m, MATRIX_WIDTH[i], TILE_WIDTH, hostMultiplicationTime, deviceMultiplicationTime);

	if (test_flag==0) printf(", Passed!\n");
	else printf(", Failed!\n");

	cudaFree(deviceMatrixM);
	cudaFree(deviceMatrixN);
	cudaFree(deviceMatrixP);
	free(hostMatrixM);
	free(hostMatrixN);
	free(hostMatrixP);
	free(hostMatrixP1);
	test_flag=0;
      }
      
      printf("average, %d, %d, %.6f, %.6f \n", MATRIX_WIDTH[i], TILE_WIDTH, totalHostMultiplicationTime, totalTileMultiplicationTime);
    }
  }
  return 0;
}
