%%cu
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define V 9
#define E 14

int* get_graph_dim(char* filename)
{
  FILE* file;

  file = fopen(filename, "r");
  if (file == NULL)
  {
    printf("Unable to read the CSR file: %s.", filename);
    exit(1);
  }

  int* dim = (int*)malloc(2*sizeof(int));
  fscanf(file, "%ld %ld", &dim[0], &dim[1]);

  return dim;
}

int* read_csr(char* filename, int v, int e)
{
  FILE* file;

  file = fopen(filename, "r");
  if (file == NULL)
  {
    printf("Unable to read the CSR file: %s.", filename);
    exit(1);
  }

  int* csr = (int*)malloc((v+1+(2*e))*sizeof(int));
  fscanf(file, "%ld %ld", &v, &e);

  int i;
  for(i=0; i<v+1; i++)
    fscanf(file, "%ld", &csr[i]);

  for(; i<v+1+2*e; i++)
    fscanf(file, "%ld", &csr[i]);

  return csr;
}

__global__ void between_centre(double* bc, int* R, int* C, int root)
{
      int i;
      int idx = threadIdx.x;
    __syncthreads();


      __shared__ int d[V], sigma[V];
      double dep[V];

      for(i = 0; i < V; i++)
        dep[i] = 0;

      for(i = idx; i < V; i += blockDim.x)
      {
        if(i == root)
        {
            d[root] = 0;
            sigma[root] = 1;
        }
        else
        {
            d[i] = -1;
            sigma[i] = 0;
        }
      }
      __syncthreads();

      __shared__ int Q[V], Q2[V];
      __shared__ int Q_len, Q2_len;

      __shared__ int S[V];
      __shared__ int S_len;

      if(idx == 0)
      {
          Q[idx] = root;
          Q_len = 1;
          Q2_len = 0;
          S_len = 0;
      }
      __syncthreads();

      int x2, x1 = 1, x3 = blockDim.x, x4;

      while(1)
      {
          x4 = x1;
          for(i = idx; i < Q_len; i += blockDim.x)
          {
              int v = Q[i];

              int t = atomicAdd(&S_len, 1);
              S[t] = v;

              __syncthreads();

              x1 = x1 * 2;

              int r;
              for(r = R[v]; r < R[v + 1]; r++)
              {
                  int w = C[r];

                  if(atomicCAS(&d[w], -1, d[v]+1) == -1)
                  {
                      t = atomicAdd(&Q2_len, 1);
                      Q2[t] = w;
                  }
                  __syncthreads();


                  if(d[w] == d[v] + 1)
                  {
                      atomicAdd(&sigma[w], sigma[v]);

                  }
                  __syncthreads();
              }

          }
          __syncthreads();

          x1 = x1 * 3;
          x2 = Q2_len + 1;

          if(Q2_len == 0)
            break;

          x1 = x1 * 5;

          for(i = idx; i < Q2_len; i += blockDim.x)
            Q[i] = Q2[i];

          if(idx == 0)
          {
              Q_len = Q2_len;
              Q2_len = 0;
          }

          __syncthreads();

      }

      if(idx==0)
        for(i=0; i<V; i++)
          bc[i] = d[i];

      int P[V][V], p_top[V];

      if(idx == 0)
      {
          for(i = 0; i < V; i++)
          {
              int r;
              for(r = R[i]; r < R[i+1]; ++r)
              {
                  int w = C[r];

                  if(d[w] == d[i] + 1)
                  {
                      p_top[w] = p_top[w] + 1;
                      P[w][p_top[w]-1] = i;
                  }
              }
          }
      }
        __syncthreads();

      if(idx == 0)
        for(i=0; i<S_len; i++)
          bc[i] = 0;

      if(idx == 0)
      {
          for(i = S_len-1; i >= 0; i--)
          {
              int w = S[i];

              int r;
              for(r = 0; r < p_top[w]; r++)
                  dep[P[w][r]] += (double)(sigma[P[w][r]] * (1 + dep[w]) / sigma[w]);

              if(w!=root)
                  //atomicAdd(&bc[w], dep[w]);
                  bc[w] += dep[w];
          }
          __syncthreads();

      }
}

void bc_calc(int* r, int* c, double* bc, int i)
{
  int j;
  for(j=0; j<V; j++)
    bc[j] = 0;

  int *d_r, *d_c;
  double *d_bc;

  cudaMalloc((void**)&d_r, (V+1) * sizeof(int));
  cudaMalloc((void**)&d_c, 2*E * sizeof(int));

  cudaMemcpy(d_r, r, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, 2*E*sizeof(int), cudaMemcpyHostToDevice);

  dim3 dimGrid(1);
  dim3 dimBlock(9);

  cudaMalloc((void**)&d_bc, V * sizeof(double));

  cudaMemcpy(d_bc, bc, V*sizeof(double), cudaMemcpyHostToDevice);

  between_centre<<<dimGrid, dimBlock>>>( d_bc, d_r, d_c, i);

  cudaMemcpy(bc, d_bc, V*sizeof(double), cudaMemcpyDeviceToHost);

  for(j=0; j<V; j++)
    printf("%f ", bc[j]);
  printf("\n");

  cudaFree(d_bc);
/*
  for(i=0; i<V; i++)
  {
      for(j=0; j<V; j++)
        bc[j] = 0;

      cudaMalloc((void**)&d_bc, V * sizeof(double));
      cudaMemcpy(d_bc, bc, V*sizeof(double), cudaMemcpyHostToDevice);

      between_centre<<<dimGrid, dimBlock>>>( d_bc, d_r, d_c, i);
      bc_add<<<1, dimBlock>>>(d_tot_bc, d_bc);


      cudaMemcpy(bc, d_bc, V*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(tot_bc, d_tot_bc, V*sizeof(double), cudaMemcpyDeviceToHost);

      for(j=0; j<V; j++)
        printf("%f ", bc[j]);
      printf("\n");

      cudaFree(d_bc);
      cudaFree(d_r);
    cudaFree(d_c);
  }

  */

}

int main()
{
  //int* dim;
  //dim = get_graph_dim("01.txt");
  int dim[2] = {9, 14};

  int v = dim[0], e = dim[1];
//  int* csr;

//  csr = read_csr("01.txt", v, e);


  int csr[] = {0, 3, 5, 8, 12, 16, 20, 24, 27, 28, 1, 2, 3, 0, 2, 0, 1, 3, 0, 2, 4, 5, 3, 5, 6, 7, 3, 4, 6, 7, 4, 5, 7, 8, 4, 5, 6, 6};

  int *r, *c;
  double bc[V];
  double tot_bc[V];

  r = (int*)malloc((v+1)*sizeof(int));
  c = (int*)malloc(2*e*sizeof(int));
  //tot_bc = (double*)malloc(v*sizeof(double));

  memcpy(r, csr, (v+1)*sizeof(int));
  memcpy(c, csr+v+1, 2*e*sizeof(int));

  //memcpy(tot_bc, 0, V*sizeof(double));

  //free(csr);

  cudaEvent_t start, stop;
  float timer;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  int j;
  for(j=0; j<V; j++)
  {
    bc[j] = 0;
    tot_bc[j] = 0;
  }

  int *d_r, *d_c;
  double *d_bc;

  cudaMalloc((void**)&d_r, (V+1) * sizeof(int));
  cudaMalloc((void**)&d_c, 2*E * sizeof(int));

  cudaMemcpy(d_r, r, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, 2*E*sizeof(int), cudaMemcpyHostToDevice);

  dim3 dimGrid(1);
  dim3 dimBlock(9);

  cudaMalloc((void**)&d_bc, V * sizeof(double));

  cudaMemcpy(d_bc, bc, V*sizeof(double), cudaMemcpyHostToDevice);

  between_centre<<<dimGrid, dimBlock>>>( d_bc, d_r, d_c, 8);

  cudaMemcpy(bc, d_bc, V*sizeof(double), cudaMemcpyDeviceToHost);

  for(j=0; j<V; j++)
    printf("%f ", bc[j]);
  printf("\n");

  cudaFree(d_bc);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timer, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  int k;
/*
  for(k = 0; k < V; k++)
      printf("%f ", tot_bc[k]);
*/
  printf("\nElapsed Time: %lf\n", timer);

  free(r);
  free(c);
}
