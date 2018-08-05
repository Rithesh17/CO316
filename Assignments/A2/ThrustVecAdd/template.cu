#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

float* readData(char* filename)
{
  FILE* handle = fopen(filename, "r");
  
  if(handle == NULL)
  {
    printf("Error opening file: %s\n", filename);
    exit(0);
  }
  
  int num, i;
  
  fscanf(handle, "%d\n", &num);
  
  float data[num];
  
  for(i=0; i<num; i++)
    fscanf(handle, "%f\n", &data[i]);
  
  return data;
}



int main(int argc, char *argv[]) {

  float *hostInput1 = nullptr;
  float *hostInput2 = nullptr;
  float *hostOutput = nullptr;
  int i;

  /* parse the input arguments */
  //@@ Insert code here

  if(argc != 11)
  {
    printf("Usage: ./ThrustVectorAdd_Template -e <expected.raw> -i <input0.raw> , <input1.raw> -o <output.raw> -t vector");
    return 0;
  }
  
  char* input0_filename = argv[4];
  char* input1_filename = argv[6];
  char* output_filename = argv[8];
  
  // Import host input data
  //@@ Read data from the raw files here
  //@@ Insert code here
  
  hostInput1 = readData(input0);
  hostInput2 = readData(input1);

  // Declare and allocate host output
  //@@ Insert code here
  
  int num = sizeof(hostInput1)/sizeof(float);
  
  thrust::host_vector<float> hostOutput(num);

  // Declare and allocate thrust device input and output vectors
  //@@ Insert code here
  
  thrust::device_vector<float> devInput1(num);
  thrust::device_vector<float> devInput2(num);
  thrust::device_vector<float> devOutput(num);

  // Copy to device
  //@@ Insert code here

  thrust::copy(hostInput1.begin(), hostInput1.end(), devInput1.begin());
  thrust::copy(hostInput2.begin(), hostInput2.end(), devInput2.begin());
  
  // Execute vector addition
  //@@ Insert Code here

  thrust::transform(devInput1.begin(), devInput1.end(), devInput2.begin(), devOutput.begin());
  
  /////////////////////////////////////////////////////////

  // Copy data back to host
  //@@ Insert code here

  thrust::copy(devOutput.begin(), devOutput.end(), hostOutput.begin());
  
  //Cross-verification
  
  float* verifyData = readData(output_filename);
  
  if(num != sizeof(verifyData)/sizeof(float))
    printf("Size not matching: Output size: %d\tExpected size: %d\n", num, sizeof(verifyData)/sizeof(float));
  else
    for(i=0; i<num; i++)
    {
      if((float)verifyData[i] != (float)hostOutput[i])
        printf("Data not matching: Location: %d\tOutput: %f\tExpected: %f\n", i+1, hostOutput[i], verifyData[i]);
    }
  
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  return 0;
}
