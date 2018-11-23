#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

long int* get_graph_dim(char* filename)
{
  FILE* file;

  file = fopen(filename, "r");
  if (file == NULL)
  {
    printf("Unable to read the CSR file: %s.", filename);
    exit(1);
  }

  long int* dim = (long int*)malloc(2*sizeof(long int));
  fscanf(file, "%ld %ld", &dim[0], &dim[1]);

  return dim;
}

long int* read_csr(char* filename, long int v, long int e)
{
  FILE* file;

  file = fopen(filename, "r");
  if (file == NULL)
  {
    printf("Unable to read the CSR file: %s.", filename);
    exit(1);
  }

  long int* csr = (long int*)malloc((v+1+(2*e))*sizeof(long int));
  fscanf(file, "%ld %ld", &v, &e);

  long int i;
  for(i=0; i<v+1; i++)
    fscanf(file, "%ld", &csr[i]);

  for(; i<v+1+2*e; i++)
    fscanf(file, "%ld", &csr[i]);

  return csr;
}

__global__ void between_centre(long float* bc, int* R, int* C, long int V, long int E)
{
  long int idx = threadIdx.x;
  long int s = blockIdx.x;

  __shared__ long int d[V], sigma[V];
  long float dep[V];

  __shared__ long int P[V][V];
  __shared__ long int p_top[V];
  //Initialize d and sigma
  for(int k=idx; k<V; k+=blockDim.x)
  {
    if(k == s)
    {
      d[k] = 0;
      sigma[k] = 1;
    }
    else
    {
      d[k] = INT_MAX;
      sigma[k] = 0;
    }

    p_top[k] = 0;

  }

  __shared__ long int Q[V];
  __shared__ long int Q2[V];
  __shared__ long int Q_len;
  __shared__ long int Q2_len;

  __shared__ long int S[V];
  __shared__ long int s_top;

  if(idx == 0)
  {
    Q[0] = s;
    Q_len = 1;
    Q2_len = 0;
    s_top = 0;
  }
  __syncthreads();

  while(1)
  {
    for(int k=idx; k<Q_len; k+=blockDim.x)
    {
      int v = Q[k];

      atomicAdd(&s_top, 1);
      S[s_top] = v;

      for(int r=R[v]; r<R[v+1]; r++)
      {
        long int w = C[r];

        if(atomicCAS(&d[w],INT_MAX,d[v]+1) == INT_MAX)
        {
          int t = atomicAdd(&Q2_len,1);
          Q2[t] = w;
        }

        if(d[w] == (d[v]+1))
        {
          atomicAdd(&sigma[w],sigma[v]);
          atomicAdd(&p_top[w], 1);
          atomicAdd(&P[w][p_top[w]-1], v);
        }
      }
    }
    __syncthreads();

    if(Q2_len == 0)
      break;

    else
    {
      for(int k=idx; k<Q2_len; k+=blockDim.x)
        Q[k] = Q2[k];

      __syncthreads();

      if(idx == 0)
      {
        Q_len = Q2_len;
        Q2_len = 0;
      }
      __syncthreads();
    }
  }

  while(s_top!=0)
  {
    atomicAdd(&s_top, -1);
    long int w = S[s_top];

    for(int k = 0; k < P[w][p_top[w]-1]; k++)
      dep[k] += (double)(sigma[k] * (1 + dep[w]) / sigma[w]);

    if(w!=s)
      atomicAdd(&bc[w], dep[w]);

    __syncthreads();
  }
}

int main()
{
  long int* dim;
  dim = get_graph_dim("01.txt");

  long int v = dim[0], e = dim[1];
  long int* csr;

  csr = read_csr("01.txt", v, e);

  long int *r, *c;
  long float *bc;
  r = (long int*)malloc((v+1)*sizeof(long int));
  c = (long int*)malloc(2*e*sizeof(long int));
  bc = (long float*)malloc(v*sizeof(long float));

  memcpy(r, csr, (v+1)*sizeof(long int));
  memcpy(c, csr+v+1, 2*e*sizeof(long int));

  free(csr);

  long int *d_v, *d_e;
  long int *d_r, *d_c;
  long float *d_bc;

  cudaMalloc((void**)&d_v, sizeof(long int));
  cudaMalloc((void**)&d_e, sizeof(long int));
  cudaMalloc((void**)&d_bc, v * sizeof(long float));
  cudaMalloc((void**)&d_r, (v+1) * sizeof(long int));
  cudaMalloc((void**)&d_c, 2*e * sizeof(long int));

  cudaMemcpy(d_v, &v, sizeof(long int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_e, &e, sizeof(long int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_r, r, (v+1)*sizeof(long int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, 2*e*sizeof(long int), cudaMemcpyHostToDevice);

  dim3 dimGrid(v);
  dim3 dimBlock(1024);

  cudaEvent_t start, stop;
  float timer;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  between_centre<<<dimGrid, dimBlock>>>(d_bc, d_r, d_c, d_v, d_e);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timer, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaMemcpy(bc, d_bc, v*sizeof(long float), cudaMemcpyDeviceToHost);

  for(int k = 0; k < v; k++)
    printf("%.2f ", bc[k]);

  printf("\nElapsed Time: %lf\n", timer);

  cudaFree(d_v);
  cudaFree(d_e);
  cudaFree(d_bc);
  cudaFree(d_r);
  cudaFree(d_c);

  free(r);
  free(c);
}
