#include"wb.h"
#include<cuda.h>
#include<cuda_runtime_api.h>

#define BLUR_SIZE 5

//@@ INSERT CODE HERE

__global__ void imgBlur(float* imgIn, float* imgOut, int imageWidth, int imageHeight)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  if(idx<imageWidth && idy<imageHeight)
 {
  float sum = imgIn[idx*imageWidth+idy];
  
  if(idx>0 && idy>0)
    sum += imgIn[(idx-1)*imageWidth+(idy-1)];

  if(idx>0)
    sum += imgIn[(idx-1)*imageWidth+idy];

  if(idx<imageWidth-1)
    sum += imgIn[(idx+1)*imageWidth+idy];

  if(idx<imageWidth-1 && idy<imageHeight-1)
    sum += imgIn[(idx+1)*imageWidth+idy+1];

  if(idx<imageWidth && idy>0)
    sum += imgIn[(idx+1)*imageWidth+idy-1];

  if(idy>0)
    sum += imgIn[idx*imageWidth+idy-1];

  if(idy<imageHeight)
    sum += imgIn[idx*imageWidth+idy+1];

  if(idx>0 && idy<imageHeight)
    sum += imgIn[(idx-1)*imageWidth+idy+1];

  imgOut[idx*imageWidth+idy] = sum / (float)(BLUR_SIZE*BLUR_SIZE);

 }   
}

int main(int argc, char *argv[]) {

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
    printf("Usage: ./ImageBlur_Template ­-e <expected.ppm> -­i <input.ppm> -­o <output.ppm> -t image\n");
    exit(0);
  }

  wbArg_t args = {argc, argv};

  inputImageFile = wbArg_getInputFile(args, 3);

  inputImage = wbImport(inputImageFile);

  // The input image is in grayscale, so the number of channels
  // is 1
  imageWidth  = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);

  // Since the image is monochromatic, it only contains only one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Compute, "Doing the computation on the GPU");

  dim3 blockSize(15, 15, 1);
  dim3 gridSize((int)ceil(imageWidth/(float)blockSize.x), (int)ceil(imageHeight/(float)blockSize.y), 1);

  imgBlur<<<gridSize, blockSize>>>(deviceInputImageData, deviceOutputImageData, imageWidth, imageHeight);

  wbTime_stop(Compute, "Doing the computation on the GPU");

  ///////////////////////////////////////////////////////
  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float),
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbImage_save(outputImage, "convoluted.ppm");

  //wbSolution(args, 5, outputImage);

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
