#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>

#define MAX_THREADS 500
#define MAX_POINTS 10000
#define R 1
#define random() rand()/RAND_MAX
#define random_r(s) rand_r(s)/RAND_MAX

inline double dist(double* p1, double* p2)
{ return sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]));}

double pi_uniprocess()
{
  double centre[2] = {(float)R, (float)R};
  int i, in_circle = 0;

  for(i=0; i<MAX_POINTS; i++)
  {
    double point[2] = {(float)rand()*2*R/RAND_MAX, (float)rand()*2*R/RAND_MAX};
    if(dist(centre, point)<=R)
      in_circle += 1;
  }

//  printf("Points inside circle: %d\n", in_circle);

  return (float)in_circle * 4.0 / MAX_POINTS;
}

double pi_multiprocess(int num_threads)
{
  double centre[2] = {R, R};
  int in_circle = 0;

  #pragma omp parallel
  {
    int i, thread_id = omp_get_thread_num(), num_threads = omp_get_num_threads();
    int partial_in_circle = 0;

    for(i=thread_id; i<MAX_POINTS; i+=num_threads)
    {
      double point[2] = {(float)rand()*2*R/RAND_MAX, (float)rand()*2*R/RAND_MAX};
      if(dist(centre, point)<=R)
        partial_in_circle += 1;
    }

    #pragma omp atomic
      in_circle += partial_in_circle;
  }

    return (float)in_circle * 4 / MAX_POINTS;
}

double pi_multiprocess_safe(int num_threads)
{
  double centre[2] = {R, R};
  int in_circle = 0;

  #pragma omp parallel
  {
    int i, thread_id = omp_get_thread_num(), num_threads = omp_get_num_threads();
    int partial_in_circle = 0;

    int state = rand() ^ thread_id;

    for(i=thread_id; i<MAX_POINTS; i+=num_threads)
    {
      double point[2] = {(float)rand_r(&state)*2*R/RAND_MAX, (float)rand_r(&state)*2*R/RAND_MAX};
      if(dist(centre, point)<=R)
        partial_in_circle += 1;
    }

    #pragma omp atomic
      in_circle += partial_in_circle;
  }

    return (float)in_circle * 4 / MAX_POINTS;
}

int main()
{
  double start, end;

  start = omp_get_wtime();
  double pi = pi_uniprocess();
  end = omp_get_wtime();

  printf("Pi value from uniprocessor: %lf\tTime elapsed: %lfs\n\n", pi, end - start);
  int i;

  int acc_thread, time_thread;
  double acc = RAND_MAX, least_time = end - start;

  for(i=2; i<MAX_THREADS; i++)
  {
    start = omp_get_wtime();
    pi = pi_multiprocess(i);
    end = omp_get_wtime();

    printf("Pi value from %d threads: %lf\tTime elapsed: %lfs\n", i, pi, end - start);

    if(abs(pi - 3.1415926535) < acc)
    {
      acc = abs(pi - 31415926535);
      acc_thread = i;
    }

    if(end - start < least_time)
    {
      least_time = end - start;
      time_thread = i;
    }
  }

  printf("-------------------------------------------------------------------------\nLeast time consumed: %lfs\tThreads: %d\n", acc, acc_thread, least_time, time_thread);

  printf("\nUsing thread-safe random number generator:\n");

  acc = RAND_MAX;

  for(i=2; i<=MAX_THREADS; i++)
  {
    start = omp_get_wtime();
    pi = pi_multiprocess_safe(i);
    end = omp_get_wtime();

    printf("Pi value from %d threads: %lf\tTime elapsed: %lfs\n", i, pi, end - start);

    if(abs(pi - 3.1415926535) < acc)
    {
      acc = abs(pi - 31415926535);
      acc_thread = i;
    }

    if(end - start < least_time)
    {
      least_time = end - start;
      time_thread = i;
    }

  }

  printf("---------------------------------------------------------------------------\nLeast time consumed: %lfs\tThreads: %d\n", acc, acc_thread, least_time, time_thread);

}
