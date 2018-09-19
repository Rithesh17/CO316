#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MAX_THREADS 500
#define MAX_POINTS 1000
#define R 1
#define random() rand()/RAND_MAX
#define random_r(s) rand_r(s)/RAND_MAX

inline double dist(double* p1, double* p2)
{ return sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]));}

double pi_uniprocess()
{
  double centre[2] = {R/2.0, R/2.0};
  int i, in_circle = 0;

  for(i=0; i<MAX_POINTS; i++)
  {
    double point[2] = {random()*R, random()*R};
    if(dist(centre, point)<=R)
      in_circle += 1;
  }

  return in_circle * 4 * R * R / MAX_POINTS;
}

double pi_multiprocess(int num_threads)
{
  double centre[2] = {R/2.0, R/2.0};
  int in_circle = 0;

  #pragma omp parallel for private(in_circle)
  {
    int i, thread_id = omp_get_thread_num(), num_threads = omp_get_num_threads;
    int partial_in_circle = 0;

    for(i=thread_id; i<MAX_POINTS; i+=num_threads)
    {
      double point[2] = {random()*R, random()*R};
      if(dist(centre, point)<=R)
        partial_in_circle += 1;
    }

    #pragma omp atomic
      in_circle += partial_in_circle;
  }

    return in_circle * 4 * R * R / MAX_POINTS;
}

double pi_multiprocess_safe(int num_threads)
{
  double centre[2] = {R/2.0, R/2.0};
  int in_circle = 0;

  #pragma omp parallel for private(in_circle)
  {
    int i, thread_id = omp_get_thread_num(), num_threads = omp_get_num_threads;
    int partial_in_circle = 0;

    int state = time(NULL) ^ pid ^ thread_id;

    for(i=thread_id; i<MAX_POINTS; i+=num_threads)
    {
      double point[2] = {random_r(&state)*R, random_r(&state)*R};
      if(dist(centre, point)<=R)
        partial_in_circle += 1;
    }

    #pragma omp atomic
      in_circle += partial_in_circle;
  }

    return in_circle * 4 * R * R / MAX_POINTS;
}

int main()
{
  double start, end;

  start = omp_get_wtime();
  double pi = pi_uniprocess();
  end = omp_get_wtime();

  printf("Pi value from uniprocessor: %lf\tTime elapsed: %lfs\n\n", pi, end - start);
  int i;

  for(i=2; i<MAX_THREADS; i++)
  {
    start = omp_get_wtime();
    pi = pi_multiprocess(i);
    end = omp_get_wtime();

    printf("Pi value from %d threads: %lf\tTime elapsed: %lfs\n", i, pi, end - start);
  }

  printf("\nUsing thred-safe random number generator:\n");

  for(i=2; i<MAX_THREADS; i++)
  {
    start = omp_get_wtime();
    pi = pi_multiprocess_safe(i);
    end = omp_get_wtime();

    printf("Pi value from %d threads: %lf\tTime elapsed: %lfs\n", i, pi, end - start);
  }
}
