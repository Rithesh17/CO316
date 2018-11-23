#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define V 9
#define E 14

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

__global__ void between_centre(double* bc, long int* R, long int* C, long int *vert, long int *edge)
{
  //long int V = *vert;
  //long int E = *edge;

  long int idx = threadIdx.x;
  long int s = blockIdx.x;

  __shared__ long int d[V], sigma[V];
  double dep[V];

  __shared__ long int P[V][V];
  __shared__ long int p_top[V];

  long int k, r;

  //Initialize d and sigma
  // for(k=idx; k<V; k+=blockDim.x)
  // {
  //   if(k == s)
  //   {
  //     d[k] = 0;
  //     sigma[k] = 1;
  //   }
  //   else
  //   {
  //     d[k] = INT_MAX;
  //     sigma[k] = 0;
  //   }
  //
  //   p_top[k] = 0;
  //
  // }
  //
  // __shared__ long int Q[V];
  // __shared__ long int Q2[V];
  // __shared__ long int Q_len;
  // __shared__ long int Q2_len;
  //
  // __shared__ long int S[V];
  // __shared__ long int s_top;
  //
  // if(idx == 0)
  // {
  //   Q[0] = s;
  //   Q_len = 1;
  //   Q2_len = 0;
  //   s_top = 0;
  // }
  // __syncthreads();
  //
  // while(1)
  // {
  //
  //   for(k=idx; k<Q_len; k+=blockDim.x)
  //   {
  //     int v = Q[k];
  //
  //     atomicAdd((int*)&s_top, 1);
  //     S[s_top] = v;
  //
  //     for(r=R[v]; r<R[v+1]; r++)
  //     {
  //       long int w = C[r];
  //
  //       if(atomicCAS((int*)&d[w],INT_MAX,(int)d[v]+1) == INT_MAX)
  //       {
  //         int t = atomicAdd((int*)&Q2_len,1);
  //         Q2[t] = w;
  //       }
  //
  //       if(d[w] == (d[v]+1))
  //       {
  //         atomicAdd((int*)&sigma[w],sigma[v]);
  //         atomicAdd((int*)&p_top[w], 1);
  //         atomicAdd((int*)&P[w][p_top[w]-1], v);
  //       }
  //     }
  //   }
  //   __syncthreads();
  //
  //   if(Q2_len == 0)
  //     break;
  //
  //   else
  //   {
  //     for(k=idx; k<Q2_len; k+=blockDim.x)
  //       Q[k] = Q2[k];
  //
  //     __syncthreads();
  //
  //     if(idx == 0)
  //     {
  //       Q_len = Q2_len;
  //       Q2_len = 0;
  //     }
  //     __syncthreads();
  //   }
  // }
  //
  // while(s_top!=0)
  // {
  //   atomicAdd((int*)&s_top, -1);
  //   long int w = S[s_top];
  //
  //   for(k = 0; k < P[w][p_top[w]-1]; k++)
  //     dep[k] += (double)(sigma[k] * (1 + dep[w]) / sigma[w]);
  //
  //   if(w!=s)
  //     atomicAdd((float*)&bc[w], (float)dep[w]);
  //
  //   __syncthreads();
  // }
}

int main()
{
  long int* dim;
  dim = get_graph_dim("01.txt");

  printf("Hello!\n");

  long int v = dim[0], e = dim[1];
  long int* csr;

  csr = read_csr("01.txt", v, e);

  printf("Holla!!\n");

  long int *r, *c;
  double *bc;
  r = (long int*)malloc((v+1)*sizeof(long int));
  c = (long int*)malloc(2*e*sizeof(long int));
  bc = (double*)malloc(v*sizeof(double));

  memcpy(r, csr, (v+1)*sizeof(long int));
  memcpy(c, csr+v+1, 2*e*sizeof(long int));

  free(csr);

  long int *d_v, *d_e;
  long int *d_r, *d_c;
  double *d_bc;

  printf("Sui!\n");

  cudaMalloc((void**)&d_v, sizeof(long int));
  cudaMalloc((void**)&d_e, sizeof(long int));
  cudaMalloc((void**)&d_bc, v * sizeof(double));
  cudaMalloc((void**)&d_r, (v+1) * sizeof(long int));
  cudaMalloc((void**)&d_c, 2*e * sizeof(long int));

  cudaMemcpy(d_v, &v, sizeof(long int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_e, &e, sizeof(long int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_r, r, (v+1)*sizeof(long int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, 2*e*sizeof(long int), cudaMemcpyHostToDevice);

  printf("Namaskara!\n");

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

  cudaMemcpy(bc, d_bc, v*sizeof(double), cudaMemcpyDeviceToHost);

  long int k;

  for(k = 0; k < v; k++)
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
