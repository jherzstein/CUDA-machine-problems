// Jordan Herzstein (ID: 20215379)
#include <stdio.h>
#include <cuda_runtime.h>

void GPUDeviceProperties(int device) {
  cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, device);
  printf("Device %d: %s \n", device, properties.name);
  printf("-> Clock Rate: %.2f GHz \n", properties.clockRate / 1e3);
  printf("-> Streaming Multiprocessors (SM): %d \n", properties.multiProcessorCount);
  printf("-> Warp Size: %d \n", properties.warpSize);
  printf("-> Global memory: %.2f GB \n", properties.totalGlobalMem / 1e9);
  printf("-> Constant memory: %.2f MB \n", properties.totalConstMem / 1e6);
  printf("-> Shared memory per block: %.2f KB \n", properties.sharedMemPerBlock / 1e3);
  printf("-> Registers available per block: %d \n", properties.regsPerBlock);
  printf("-> Max threads per block: %d \n", properties.maxThreadsPerBlock);
  printf("-> Max Size of Threads: \n ");
  for (int i = 0; i < 3; i++)
    printf("\t-> Max Size of dimension %d of block: %d \n", i, properties.maxThreadsDim[i]);
  printf("-> Max Grid Sizes: \n");
  for (int i = 0; i < 3; i++)
    printf("\t-> Max Size of dimension %d of grid: %d \n", i, properties.maxGridSize[i]);

  int cores = 0;
  int mpc = properties.multiProcessorCount;

  switch (properties.major) {
    case 2:
      if (properties.minor == 1)
	cores = mpc * 48;
      else
	cores = mpc * 32;
      break;
    case 3:
      cores = mpc * 192;
      break;
    case 5:
      cores = mpc * 128;
      break;
    case 6:
      if (properties.minor == 1 || properties.minor == 2)
	cores = mpc * 128;
      else if (properties.minor == 0)
	cores = mpc * 64;
      else
	printf("Unable to find device type.\n");
      break;
    case 7:
      if (properties.minor == 0 || properties.minor == 5)
	cores = mpc * 64;
      else
	printf("Unable to find device type.\n");
      break; 
    case 8:
      if (properties.minor == 0)
	cores = mpc * 64;
      else if (properties.minor == 6 || properties.minor == 9)
	cores = mpc * 128;
      else
	printf("Unable to find device type.\n");
      break;
    case 9:
      if (properties.minor == 0)
	cores = mpc * 128;
      else
	printf("Unable to find device type.\n");
    default:
      cores = -1;
      printf("Unable to find device type.\n");
  }
  printf("-> Number of device cores: %d \n", cores);


}

int main() {
  int countDevices;
  cudaGetDeviceCount(&countDevices);
  if (countDevices > 0) { 
    printf("Number of detected Devices: %d \n", countDevices);
    for (int i = 0; i < countDevices; i++) {
      GPUDeviceProperties(i);
    } 
  } else {
    printf("No CUDA devices detected. \n");
  }
}
