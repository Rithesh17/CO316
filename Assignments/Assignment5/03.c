#include<stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include<omp.h>

#define MAX_THREADS 56
#define MAX(a, b) (a>b)?a:b

unsigned long int n;

void daxpy_uniprocess(double* X, double* Y, int a)
{
  unsigned long int i;
  for(i=0; i<n; i++)
    X[i] = a*X[i]+Y[i];
}

void daxpy_multi_process(double* X, double* Y, int a, unsigned long int num_threads)
{
  omp_set_num_threads(MAX(num_threads, MAX_THREADS));
  int thread_id;
  unsigned long int i, index;

  #pragma omp parallel
  {
    thread_id = omp_get_thread_num();
    for(i=0; i<n/num_threads; i++)
    {
      index = i*num_threads+thread_id;
      X[index] = a*X[index]+Y[index];
    }
  }
}

int main()
{
  n = 1<<20;
  double X[n], Y[n];
  unsigned long int i;

  for(i=0; i<n; i++)
  {
    X[i] = rand() + rand() / RAND_MAX;
    Y[i] = rand() + rand() / RAND_MAX;
  }

  int a = 30;

  printf("%ld, %d\n", n, a);

  int start, end;

  printf("Time taken by uniprocessor: ");
  start = clock();
  daxpy_uniprocess(X, Y, a);
  end = clock();
  printf("Start time: %ld  End time: %ld  Time elapsed: %ld clocks\n\n", start, end, end - start);

  for(i=2; i<=80; i++)
  {
    printf("Time taken by %d threads: ", i);
    start = clock();
    daxpy_uniprocess(X, Y, a);
    end = clock();
    printf("Start time: %ld  End time: %ld  Time elapsed: %ld clocks\n\n", start, end, end - start);
  }
  printf("\n");
}
