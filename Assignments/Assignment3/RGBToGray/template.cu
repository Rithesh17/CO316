#include"wb.h"
#include<cuda.h>
#include<cuda_runtime_api.h>

//@@ define error checking macro here.
#define errCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      printErrorLog(ERROR, "Failed to run stmt ", #stmt);                         \
      printErrorLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ INSERT CODE HERE

void __global__ RGBToGray(float* devIn, float* devOut, int imgWd, int imgHt)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if(idx < imgHt && idy < imgWd)
  {
    int id = idx * imgWd + idy;
    
    devOut[id] = 0.21*devIn[3*id] + 0.71*devIn[3*id+1] + 0.07*devIn[3*id+2];
  }
}

#define THREAD_NUM 16

int main(int argc, char *argv[]) {

  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceInputImageData;
  float *deviceOutputImageData;

  /* parse the input arguments */
  //@@ Insert code here

  if(argc != 9)
  {
    printf("Usage:  ./TiledMatrixMultiplication_Template -e <expected.pbm> -i <input.ppm> -o <output.pbm> -t matrix");
    exit(0);
  }

  wbArg_t args = {argc, argv};

  inputImageFile = wbArg_getInputFile(args, 3);

  inputImage = wbImport(inputImageFile);

  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  // For this lab the value is always 3
  imageChannels = wbImage_getChannels(inputImage);

  // Since the image is monochromatic, it only contains one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * imageChannels * sizeof(float),
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE

  dim3 blockSize(THREAD_NUM, THREAD_NUM, 1);
  dim3 gridSize((int)ceil(imageWidth/(float)blockSize.x), (int)ceil(imageHeight/(float)blockSize.y), 1);

  RGBToGray<<<gridSize, blockSize>>>(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight);
  
  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(args, 5, outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
