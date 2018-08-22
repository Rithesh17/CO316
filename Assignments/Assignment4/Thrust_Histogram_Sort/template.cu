#include <wb.h>

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

int main(int argc, char *argv[])
{
  wbArg_t args;
  int inputLength, num_bins;
  unsigned int *hostInput = NULL;
  
  if(argc != 3)
  {
    printf("\nUsage: ./a.out <input.raw> <output.raw>\n\n");
  }

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0), \
                                       &inputLength, "Integer");
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  thrust::device_vector<unsigned int> deviceInput(inputLength);

  // Copy the input to the GPU
  wbTime_start(GPU, "Allocating GPU memory");
  //@@ Insert code here
  thrust::copy(hostInput, hostInput + inputLength, deviceInput.begin());
  thrust::sort(thrust::device, deviceInput.begin(), deviceInput.end());

  wbTime_stop(GPU, "Allocating GPU memory");

  // Determine the number of bins (num_bins) and create space on the host
  //@@ insert code here
  num_bins = deviceInput.back() + 1;
  hostBins = (unsigned int *)malloc(num_bins * sizeof(unsigned int));

  // Allocate a device vector for the appropriate number of bins
  //@@ insert code here
  thrust::device_vector<unsigned int> deviceUB(num_bins);
  thrust::device_vector<unsigned int> deviceResult(num_bins);
  unsigned int hostResult[num_bins];

  // Create a cumulative histogram. Use thrust::counting_iterator and
  // thrust::upper_bound
  //@@ Insert code here
  thrust::counting_iterator<unsigned int> first(0);

  int i;
  for(i = 0; i < num_bins; i++)
    deviceUB[i] = thrust::upper_bound(thrust::device, deviceInput.begin(), \
      deviceInput.end(), first[i]);

  // Use thrust::adjacent_difference to turn the culumative histogram
  // into a histogram.
  //@@ insert code here.
  thrust::adjacent_difference(thrust::device, deviceUB.begin(), \
    deviceUB.end(), deviceResult.begin());

  // Copy the histogram to the host
  //@@ insert code here
  thrust::copy(deviceResult.begin(), deviceResult.end(), hostResult);

  // Check the solution is correct
  wbSolution(args, hostResult, num_bins);

  return 0;
}
