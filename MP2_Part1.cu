#include "cuda_runtime.h"
#include <stdio.h>

const int MATRIX_WIDTH[]={100, 250, 500, 1000, 1500};
//const int TILE_WIDTH[] = {2};
#define TILE_WIDTH 5
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
__global__ void tiledMatrixMultiplication(float *A, float *B, float *C, float size) {
  __shared__ float tile1[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tile2[TILE_WIDTH][TILE_WIDTH];
  
  int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
  
  float temp = 0; int loc;
  
  
  for (int i = 0; i< gridDim.x; ++i) {
    
    loc = (i * TILE_WIDTH + threadIdx.y)* size + col; 
    //if (loc >= size*size) 
    //  tile2[threadIdx.y][threadIdx.x] = 0;
    //else
      tile2[threadIdx.y][threadIdx.x] = B[loc];
    
    
    
    loc = row * size + i * TILE_WIDTH + threadIdx.x;
    //if (loc >= size*size)
    //  tile1[threadIdx.y][threadIdx.x] = 0;
    //else
      tile1[threadIdx.y][threadIdx.x] = A[loc]; 
    
    __syncthreads(); 
    
    
    
    
    for (int j = 0; j < TILE_WIDTH; j++)
      temp += tile1[threadIdx.y][j] * tile2[j][threadIdx.x];
    
    __syncthreads(); 
    
    
  }
  
  //if (row < size && col < size) {
    int temp2;
    temp2 = row * size + col;
    C[temp2] = temp;
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
  // Properties for questions 1 and 2 using CUDA device 0
  GPUDeviceProperties(0);

  // Initialize CUDA events for timing
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaDeviceSynchronize();
  
  printf("Printing as CSV: \n");
  printf("N, MATRIX_WIDTH, TILE_WIDTH, HM (ms), TM (ms), Results \n");

  for(int i=0; i<numElementsMatrix; i++) { // Looping through Matrix sizes
    //for(int n=0; n<numElementsTileWidth; n++) { // Looping through Tile widths 
      // Setting average times
      float totalTileMultiplicationTime = 0;
      float totalHostMultiplicationTime = 0;
      float totalDeviceToHostTime = 0;
      float totalHostToDeviceTime = 0;
      for(int m=0; m<data_points; m++) { // Looping for extra data
	// Check if device multiplication and host multiplication are equal
	size_t size = MATRIX_WIDTH[i] * MATRIX_WIDTH[i] * sizeof(float);
	float* hostMatrixM = (float*)malloc(size);
	float* hostMatrixN = (float*)malloc(size);
	float* hostMatrixP = (float*)malloc(size);
	float* hostMatrixP1 = (float*)malloc(size);

	// Loading random values into host matrices
	for (int j = 0; j < MATRIX_WIDTH[i]; j++) {
	  for (int k = 0; k < MATRIX_WIDTH[i]; k++) {
	    float value1 = rand() % 10;
	    float value2 = rand() % 10;
	    *(hostMatrixM + j * MATRIX_WIDTH[i] + k) = value1;
	    *(hostMatrixN + j * MATRIX_WIDTH[i] + k) = value2;
	  }
	}
	
	// Allocate memory for matrices on device
	float* deviceMatrixM;
	float* deviceMatrixN;
	float* deviceMatrixP;
	cudaMalloc(&deviceMatrixM, size);
	cudaMalloc(&deviceMatrixN, size);
	cudaMalloc(&deviceMatrixP, size);

	// Transfer matrices from host to device
	cudaEventRecord(start);
	cudaDeviceSynchronize();
	cudaMemcpy(deviceMatrixN, hostMatrixN, size, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMatrixM, hostMatrixM, size, cudaMemcpyHostToDevice);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float hostToDeviceTime = getElapsedTime(start, end);
	totalHostToDeviceTime += hostToDeviceTime;
	//printf("Host to Device Transfer Time: %.6f ms\n", hostToDeviceTime);
	
	// Transfer matrices back from device to host
	//cudaEventRecord(start);
	//cudaDeviceSynchronize();
	//cudaMemcpy(hostMatrixN, deviceMatrixN, size, cudaMemcpyDeviceToHost);
	//cudaMemcpy(hostMatrixM, deviceMatrixM, size, cudaMemcpyDeviceToHost);
	//cudaEventRecord(end);
	//cudaEventSynchronize(end);
	//float deviceToHostTime = getElapsedTime(start, end);
	//totalDeviceToHostTime += deviceToHostTime;
	//printf("Device to Host Transfer Time: %.6f ms\n", deviceToHostTime);

	//int numBlocks = MATRIX_WIDTH[i]/TILE_WIDTH[n];
	//if(MATRIX_WIDTH[i] % TILE_WIDTH[n]) numBlocks++;
	int numBlocks = MATRIX_WIDTH[i]/TILE_WIDTH;
	if(MATRIX_WIDTH[i] % TILE_WIDTH) numBlocks++;
	dim3 dimBlock(numBlocks, numBlocks);
	//dim3 dimGrid(TILE_WIDTH[n],TILE_WIDTH[n]);
	dim3 dimGrid(TILE_WIDTH, TILE_WIDTH);
	//dim3 dimBlock(MATRIX_WIDTH[i], MATRIX_WIDTH[i], 1);

	cudaEventRecord(start);
	tiledMatrixMultiplication <<<dimBlock, dimGrid>>> (deviceMatrixN, deviceMatrixM, deviceMatrixP, MATRIX_WIDTH[i]);
	//matrixMultiplication <<<1, 1>>> (deviceMatrixA, deviceMatrixB, deviceMatrixC, MATRIX_WIDTH[i]);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaMemcpy(hostMatrixP1, deviceMatrixP, size, cudaMemcpyDeviceToHost);
	float deviceMultiplicationTime = getElapsedTime(start, end);
	totalTileMultiplicationTime += deviceMultiplicationTime;

	cudaEventRecord(start);
	cudaDeviceSynchronize();
	hostMultiplication(hostMatrixN,hostMatrixM,hostMatrixP,MATRIX_WIDTH[i]);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float hostMultiplicationTime = getElapsedTime(start, end);
	totalHostMultiplicationTime += hostMultiplicationTime;

	double tolerance=1e-6;
	for (int x = 0; x < MATRIX_WIDTH[i]; x++) {
	  for (int y = 0; y < MATRIX_WIDTH[i]; y++) {
	    if (abs(hostMatrixP[x * MATRIX_WIDTH[i] + y] - hostMatrixP1[ x * MATRIX_WIDTH[i] + y]) > tolerance) {
	      test_flag=1;
	    }
	  }
	}

	//printf("%d, %d, %d, %.6f, %.6f", m, MATRIX_WIDTH[i], TILE_WIDTH[n], hostMultiplicationTime, deviceMultiplicationTime);
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
      
      //printf("average, %d, %d, %.6f, %.6f \n", MATRIX_WIDTH[i], TILE_WIDTH[n], totalHostMultiplicationTime/data_points, totalTileMultiplicationTime/data_points);
      printf("average, %d, %d, %.6f, %.6f \n", MATRIX_WIDTH[i], TILE_WIDTH, totalHostMultiplicationTime/data_points, totalTileMultiplicationTime/data_points);
    }
  //}
  return 0;
}
