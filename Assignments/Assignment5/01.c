#include<stdio.h>
#include<omp.h>

int main()
{
  omp_set_num_threads(10);
  #pragma omp parallel
  {
    printf("Hello World\n%d\n", omp_get_max_threads());
  }
}
