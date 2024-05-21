// Jordan Herzstein (ID:20215379)
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Matrix sizes and Threads Per Block
const int MATRIX_WIDTH[]={1, 100, 250, 500, 1000, 1500};
const int BLOCK_WIDTH[]={1, 2, 5, 10, 25, 32};
//const int MATRIX_WIDTH[]={100};
//const int BLOCK_WIDTH[]={1};
// part 2 tpb 1 and diff mat sizes

int numElementsMatrix = sizeof(MATRIX_WIDTH) / sizeof(int);
int numElementsBlockWidth = sizeof(BLOCK_WIDTH) / sizeof(int);

// test flag
int test_flag = 0;

// Number of datapoints for each MATRIX_WIDTH and BLOCK_WIDTH
int data_points = 5;

// Function to measure time
float getElapsedTime(cudaEvent_t start, cudaEvent_t end) {
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, end);
    return (float)elapsedTime;
}

// Matrix Multiplication for device (GPU)
__global__ void matrixMultiplication(float *A, float *B, float *C, int size) {
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  int col = blockIdx.x*blockDim.x + threadIdx.x;

  if (row < size && col < size) {
    int temp = row * size + col;
    float value = 0;
    for (int k = 0; k < size; k++){
      value += A[row * size + k] * B[k * size + col];
    }
    C[temp] = value;
  }
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


int main() {
  // Initialize CUDA events for timing
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaDeviceSynchronize();
  
  printf("Printing as CSV: \n");
  printf("N, MATRIX_WIDTH, BLOCK_WIDTH, H2D (ms), D2H (ms), HM (ms), DM (ms), Results \n");
  
  for(int i=0; i<numElementsMatrix; i++) { // Looping through Matrix sizes
    for(int n=0; n<numElementsBlockWidth; n++) { // Looping through Block widths 
      // Setting average times
      float totalDeviceMultiplicationTime = 0;
      float totalHostMultiplicationTime = 0;
      float totalDeviceToHostTime = 0;
      float totalHostToDeviceTime = 0;
      for(int m=0; m<data_points; m++) { // Looping for extra data
	// printf("Matrix Size: %d \n", MATRIX_WIDTH[i]*MATRIX_WIDTH[i]);
	// printf("BLOCK_WIDTH: %d \n", BLOCK_WIDTH[n]);
	// Allocate memory for matrices on host
	size_t size = MATRIX_WIDTH[i] * MATRIX_WIDTH[i] * sizeof(float);
	float* hostMatrixA = (float*)malloc(size);
	float* hostMatrixB = (float*)malloc(size);
	float* hostMatrixC = (float*)malloc(size);
	float* hostMatrixC1 = (float*)malloc(size);

	//loading random values into host matrices
	for (int j = 0; j < MATRIX_WIDTH[i]; j++) {
	  for (int k = 0; k < MATRIX_WIDTH[i]; k++) {
	    float value1 = rand() % 10;
	    float value2 = rand() % 10;
	    *(hostMatrixA + j * MATRIX_WIDTH[i] + k) = value1;
	    *(hostMatrixB + j * MATRIX_WIDTH[i] + k) = value2;
	  }
	}
	
	// Allocate memory for matrices on device
	float* deviceMatrixA;
	float* deviceMatrixB;
	float* deviceMatrixC;
	cudaMalloc(&deviceMatrixA, size);
	cudaMalloc(&deviceMatrixB, size);
	cudaMalloc(&deviceMatrixC, size);
	
	// Transfer matrices from host to device
	cudaEventRecord(start);
	cudaDeviceSynchronize();
	cudaMemcpy(deviceMatrixA, hostMatrixA, size, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMatrixB, hostMatrixB, size, cudaMemcpyHostToDevice);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float hostToDeviceTime = getElapsedTime(start, end);
	totalHostToDeviceTime += hostToDeviceTime;
	//printf("Host to Device Transfer Time: %.6f ms\n", hostToDeviceTime);
	
	// Transfer matrices back from device to host
	cudaEventRecord(start);
	cudaDeviceSynchronize();
	cudaMemcpy(hostMatrixA, deviceMatrixA, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(hostMatrixB, deviceMatrixB, size, cudaMemcpyDeviceToHost);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float deviceToHostTime = getElapsedTime(start, end);
	totalDeviceToHostTime += deviceToHostTime;
	//printf("Device to Host Transfer Time: %.6f ms\n", deviceToHostTime);
	
	
	int numBlocks = MATRIX_WIDTH[i]/BLOCK_WIDTH[n];
	if(MATRIX_WIDTH[i] % BLOCK_WIDTH[n]) numBlocks++;
	dim3 dimBlock(numBlocks, numBlocks);
	//dim3 dimGrid(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(BLOCK_WIDTH[n],BLOCK_WIDTH[n]);
	//dim3 dimBlock(MATRIX_WIDTH[i], MATRIX_WIDTH[i], 1);

	// do block size of 1 and different mat sizes
	cudaEventRecord(start);
	matrixMultiplication <<<dimBlock, dimGrid>>> (deviceMatrixA, deviceMatrixB, deviceMatrixC, MATRIX_WIDTH[i]);
	//matrixMultiplication <<<1, 1>>> (deviceMatrixA, deviceMatrixB, deviceMatrixC, MATRIX_WIDTH[i]);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	cudaMemcpy(hostMatrixC1, deviceMatrixC, size, cudaMemcpyDeviceToHost);
	float deviceMultiplicationTime = getElapsedTime(start, end);
	totalDeviceMultiplicationTime += deviceMultiplicationTime;
	//printf("Matrix Multiplication Time for GPU: %.6f ms\n", matrixMultiplicationTime);

	cudaEventRecord(start);
	cudaDeviceSynchronize();
	hostMultiplication(hostMatrixA,hostMatrixB,hostMatrixC,MATRIX_WIDTH[i]);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float hostMultiplicationTime = getElapsedTime(start, end);
	totalHostMultiplicationTime += hostMultiplicationTime;
	//printf("host multiplication time: %.6f ms \n", hostMultiplicationTime);
	
	// Check if device multiplication and host multiplication are equal
	for (int x = 0; x < MATRIX_WIDTH[i]; x++) {
	  for (int y = 0; y < MATRIX_WIDTH[i]; y++) {
	    if (hostMatrixC[x * MATRIX_WIDTH[i] + y] != hostMatrixC1[ x * MATRIX_WIDTH[i] + y]) {
	      test_flag=1;
	    }
	  }
	}
	
	// Print Results
	printf("%d, %d, %d, %.6f, %.6f, %.6f, %.6f", m, MATRIX_WIDTH[i], BLOCK_WIDTH[n], hostToDeviceTime, deviceToHostTime, hostMultiplicationTime, deviceMultiplicationTime);

	if (test_flag==0) printf(", Passed!\n");
	else printf(", Failed!\n");
	
	// Clean up
	cudaFree(deviceMatrixA);
	cudaFree(deviceMatrixB);
	cudaFree(deviceMatrixC);
	free(hostMatrixA);
	free(hostMatrixB);
	free(hostMatrixC);
	free(hostMatrixC1);
	test_flag=0;
	// cudaEventDestroy(start);
	// cudaEventDestroy(end);
      }
      // Print Averages
      printf("average, %d, %d, %.6f, %.6f, %.6f, %.6f\n", MATRIX_WIDTH[i], BLOCK_WIDTH[n], totalHostToDeviceTime/data_points, totalDeviceToHostTime/data_points, totalHostMultiplicationTime/data_points, totalDeviceMultiplicationTime/data_points);
      
    }
  }
  return 0;
}
