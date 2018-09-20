#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define min(X, Y) (((X) < (Y)) ? (X) : (Y))
#define max(X, Y) (((X) > (Y)) ? (X) : (Y))

#define Mask_width 5
#define maskRadius Mask_width / 2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

//@@ INSERT CODE HERE
__global__ void convolution(float *I, float *K, float *P, int channels, int width, int height)
{
	for (int i = 0; i <= height; i++)
	{
		for (int j = 0; j <= width; j++)
		{
			for (k = 0; k <= channels; k++)
			{
				float accum = 0;
				for (int y = -maskRadius; y <= maskRadius; y++)
				{
					for (x = -maskRadius; x <= maskRadius; x++)
					{
						int xOffset = j + x;
						int yOffset = i + y;
						if (xOffset >= 0 && xOffset < width && yOffset >= 0 && yOffset < height)
						{
							float imagePixel = I[(yOffset * width + xOffset) * channels + k];
							float maskValue = K[(y+maskRadius)*maskWidth+x+maskRadius];
							accum += imagePixel * maskValue;
						}
					}
				}
				P[(i * width + j)*channels + k] = clamp(accum, 0, 1);
			}
		}
	}
}

int main(int argc, char *argv[]) 
{
  wbArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  arg = wbArg_read(argc, argv); /* parse the input arguments */

	if (arg.argc != 11)
	{
		printf("Usage: ./Convolution_Template ­e <expected.ppm> ­i <input0.ppm> , <input1.raw> ­o <output.ppm> ­t image\n");
		exit(0);
	}

  inputImageFile = wbArg_getInputFile(arg, 3);
  inputMaskFile  = wbArg_getInputFile(arg, 5);

  inputImage   = wbImport(inputImageFile);
  hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
  assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

  imageWidth    = wbImage_getWidth(inputImage);
  imageHeight   = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ INSERT CODE HERE

	cudaMalloc((void **)&deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
  	cudaMalloc((void **)&deviceOutputImageData, imageWidth * imageHeight * sizeof(float));
	cudaMalloc((void **)&deviceMaskData, 5 * 5 * sizeof(float));

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ INSERT CODE HERE

	cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMaskData, hostMaskData, 5 * 5 * sizeof(float), cudaMemcpyHostToDevice);

  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE


  	dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
  	dim3 gridSize((int)ceil(imageWidth/(float)blockSize.x), (int)ceil(imageHeight/(float)blockSize.y), 1);
  	convolution<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceMaskData, deviceOutputImageData, imageChannels, imageWidth, imageHeight);

  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ INSERT CODE HERE

  	cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(arg, outputImage);

  //@@ Insert code here

  free(hostMaskData);
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
