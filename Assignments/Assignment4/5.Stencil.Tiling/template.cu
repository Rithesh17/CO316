#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// #define value(arry, i, j, k) arry[(( i )*width + (j)) * depth + (k)]
float value(float *arry, int i, int j, int k)
{
	return arry[(( i )*width + (j)) * depth + (k)];
}

// #define in(i, j, k) value(input_array, i, j, k)
float in(float *input_array, int i, int j, int k)
{
	return value(input_array, i, j, k);
}

// #define out(i, j, k) value(output_array, i, j, k)
float out(float *input_array, int i, int j, int k)
{
	return value(output_array, i, j, k);
}

#define Max(a, b) ((a) < (b) ? (b) : (a))
#define Min(a, b) ((a) > (b) ? (b) : (a))
#define Clamp(a, start, end) Max(Min(a, end), start)

#define TILE_WIDTH 16

__global__ void stencil(float *output, float *input, int width, int height, int depth) 
{
  //@@ INSERT CODE HERE

	for (int i = 1; i < height; i++) 
	{
		for (int j = 1; j < width; j++) 
		{
			for (int k = 1; k < depth; k++) 
			{
				float res = in(input, i, j, k + 1) + in(input, j, k - 1) + in(input, i, j + 1, k) + in(input, i, j - 1, k) + in(input, i + 1, j, k) + in(input, i - 1, j, k) - 6 * in(input, i, j, k);
				out(output, i, j, k) = Clamp(res, 0, 255);
			}
		}
	}
}

static void launch_stencil(float *deviceOutputData, float *deviceInputData, int imageWidth, int imageHeight, int imageDepth) 
{
  //@@ INSERT CODE HERE

	dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
  	dim3 gridSize((int)ceil(imageWidth/(float)blockSize.x), (int)ceil(imageHeight/(float)blockSize.y), 1);

  	stencil<<<gridSize, blockSize>>>(deviceOutputData, deviceInputData, imageWidthimageWidth, imageHeight, imageDepth);
}

int main(int argc, char *argv[]) 
{
  wbArg_t arg;
  int width;
  int height;
  int depth;
  char *inputFile;
  wbImage_t input;
  wbImage_t output;
  float *hostInputData;
  float *hostOutputData;
  float *deviceInputData;
  float *deviceOutputData;

  arg = wbArg_read(argc, argv);

	 if(argc != 9)
  	{
   		printf("Usage:  ./Stencil_Tiling -e <expected.pbm> -i <input.ppm> -o <output.pbm> -t stencil");
    		exit(0);
  	}

  inputFile = wbArg_getInputFile(arg, 3);

  input = wbImport(inputFile);

  width  = wbImage_getWidth(input);
  height = wbImage_getHeight(input);
  depth  = wbImage_getChannels(input);

  output = wbImage_new(width, height, depth);

  hostInputData  = wbImage_getData(input);
  hostOutputData = wbImage_getData(output);

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputData, width * height * depth * sizeof(float));
  cudaMalloc((void **)&deviceOutputData, width * height * depth * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputData, hostInputData, width * height * depth * sizeof(float),cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  launch_stencil(deviceOutputData, deviceInputData, width, height, depth);
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputData, deviceOutputData, width * height * depth * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbSolution(arg, output);

  cudaFree(deviceInputData);
  cudaFree(deviceOutputData);

  wbImage_delete(output);
  wbImage_delete(input);

  return 0;
}
