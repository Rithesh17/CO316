// dataset generator for matrix

#include<iostream>
#include<cstdio>
#include<cstdlib>

using namespace std;

static char *base_dir;

static void compute(float **output, float **input0, float **input1, int num)
{
  	for (int i = 0; i < num; i++) 
	{
		for (int j = 0; j < num; j++)
		{
			float out = 0;
			for (int k = 0; k < num; k++)
			{
				out += input0[i][k] + input1[k][j];
			}
			output[i][j] = out;	
  		}
	}
}

static float **generate_data(int n) 
{
	float** data = new float*[n];
	for(int i = 0; i < n; i++)
   	{
 		data[i] = new float[n];
	}
	
  	for (int i = 0; i < n; i++) 
	{
    		for (int j = 0; j < n; j++)
		{
			data[i][j] = ((float)(rand() % 20) - 5) / 5.0f;
		}
  	}
  	return data;
}

static void write_data(char *file_name, float **data, int num) 
{
  	FILE *handle = fopen(file_name, "w");
  	fprintf(handle, "%d", num);
  	for (int i = 0; i < num; i++) 
	{
    		for (int j = 0; j < num; j++)
		{
			fprintf(handle, "\n%.2f", data[i][j]);
		}
  	}
  	fflush(handle);
  	fclose(handle);
}

static void create_dataset(int datasetNum, int dim) 
{

  const char *dir_name="base_dir";

  char *input0_file_name = "input0.raw";
  char *input1_file_name = "input1.raw";
  char *output_file_name = "output.raw";

  float **input0_data = generate_data(dim);
  float **input1_data = generate_data(dim);
  float	**output_data = new float*[dim];
	for(int i = 0; i < dim; i++)
   	{
 		output_data[i] = new float[dim];
	}

  compute(output_data, input0_data, input1_data, dim);

  write_data(input0_file_name, input0_data, dim);
  write_data(input1_file_name, input1_data, dim);
  write_data(output_file_name, output_data, dim);

  delete [] input0_data;
  delete [] input1_data;
  delete [] output_data;
}

int main(void) 
{

/*  create_dataset(0, 16);
  create_dataset(1, 64);
  create_dataset(2, 93);
  create_dataset(3, 112);
  create_dataset(4, 1120);
  create_dataset(5, 9921);
  create_dataset(6, 1233);
  create_dataset(7, 1033);
  create_dataset(8, 4098);
*/  create_dataset(1, 512);
  return 0;
}
