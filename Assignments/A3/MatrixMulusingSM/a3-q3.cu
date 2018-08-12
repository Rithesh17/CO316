// Tiled dense matrix multiplication routine using shared memory

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

int num;

void checkCUDAError(const char *msg);

// Matrix multiplication
__global__ void matrix_multiply(float *a, float *b, float *c, size_t width)
{
  	// create shorthand names for threadIdx & blockIdx
  	int tx = threadIdx.x, ty = threadIdx.y;
	int bx = blockIdx.x, by = blockIdx.y;

  	// allocate 2D tiles in __shared__ memory
  	__shared__ float s_a[TILE_WIDTH][TILE_WIDTH];
  	__shared__ float s_b[TILE_WIDTH][TILE_WIDTH];

  	// calculate the row & column index of the element
  	int row = by * blockDim.y + ty;
  	int col = bx * blockDim.x + tx;

  	float result = 0;

  	// loop over the tiles of the input in phases
  	for(int i = 0; i < (width - 1)/TILE_WIDTH + 1; ++i)
  	{
    		// collaboratively load tiles into __shared__ 
		if (row < width && i*TILE_WIDTH + tx < width)
		{	
    			s_a[ty][tx] = a[row*width + i*TILE_WIDTH + tx)];
		}
		else
		{
			s_a[ty][tx] = 0.0;
		}
		if (col < width && i*TILE_WIDTH + ty < width)
		{
    			s_b[ty][tx] = b[(i*TILE_WIDTH + ty)*width + col];
		}
		else
			s_b[ty][tx] = 0.0;
		}

    		// wait until all data is loaded before allowing any thread in this block to continue
    		__syncthreads();

    		// do dot product between row of s_a and column of s_b
    		for(int k = 0; k < TILE_WIDTH; ++k)
    		{
      			result += s_a[ty][k] * s_b[k][tx];
    		}

    		// wait until all threads are finished with the data before allowing any thread in this block to continue
    		__syncthreads();
	}
	if (row < width && col < width)
	{
		c[row*num + col] = result;
	}
}

// To read input files
float* readData(char* filename)
{
	FILE* handle = fopen(filename, "r");
  
  	if (handle == NULL)
  	{
    		printf("Error opening file: %s\n", filename);
    		exit(0);
  	}
  
  	int i;
  
  	fscanf(handle, "%d", &num);
  
  	float *data = = (float *)malloc(sizeof(float) * num * num);
  
  	for (i = 0; i < num; i++)
    	{
		fscanf(handle, "%f", &data[i]);
	}
  
  	// printf("%f %f %f\n", data[0], data[1], data[2]);

  	return data;
}


// Main
int main(int argc, char *argv[]) 
{
	float *h_A = NULL;
	float *h_B = NULL;
  	int i;

  	// parse the input arguments
	if(argc != 11)
  	{
    		printf("\nUsage: ./TiledMatrixMultiplication_Template -e <expected.raw> -i <input0.raw> , <input1.raw> -o <output.raw> -t matrix\n\n");
    		return 0;
  	}
  
  	char* input0_filename = argv[4];
  	char* input1_filename = argv[6];
  	char* output_filename = argv[8];
  
  	// Import host input data
  
  	h_A = readData(input0_filename);
 	h_B = readData(input1_filename);

	// Host output declaration and memory allocation
	float *h_C = (float *)malloc(sizeof(float) * num * num);

	// allocate storage for the device
  	float *d_a = 0, *d_b = 0, *d_c = 0;
  	cudaMalloc((void**)&d_a, sizeof(float) * num * num);
  	cudaMalloc((void**)&d_b, sizeof(float) * num * num);
	cudaMalloc((void**)&d_c, sizeof(float) * num * num);
	checkCUDAError("CUDA malloc");

 	// copy host input to the device
  	cudaMemcpy(d_a, h_a, sizeof(float) * num * num, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sizeof(float) * num * num, cudaMemcpyHostToDevice);
	checkCUDAError("CUDA memcpy");
	
	// kernel launch
    	dim3 dimBlock(TILE_WIDTH,TILE_WIDTH, 1);
	dim3 dimGrid((num - 1) / TILE_WIDTH + 1, (num - 1) / TILE_WIDTH + 1, 1);
 	matrix_multiply<<<dimGrid,dimBlock>>>(d_a, d_b, d_c, num);

	// Copy result from device to host
	cudaMemcpy(h_C, d_C, sizeof(float) * num * num, cudaMemcpyDevicetoHost);
	checkCUDAError("CUDA memcpy results");

  	//Cross-verification
  
  	float* verifyData = readData(output_filename);
  
  	if(num * num != sizeof(verifyData)/sizeof(float))
    		printf("Size not matching: Output size: %d\tExpected size: %d\n", num * num, sizeof(verifyData)/sizeof(float));
  	else
    	for(i=0; i<num * num; i++)
    	{
      		if((float)verifyData[i] != (float)h_C[i])
        	printf("Data not matching: Location: %d\tOutput: %f\tExpected: %f\n", i+1, h_C[i], verifyData[i]);
    	}

	// deallocate device memory
  	cudaFree(d_a);
  	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
