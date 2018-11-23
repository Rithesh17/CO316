#include <iostream>
#include <vector>
#include <queue>
#include <list>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

void calculate_bc(std::vector<long double>& c_b, long int* r, long int* c, long int v, long int e)
{
  long int s, j;
  for(s=0; s<v; s++)
  {
    std::vector<long int> S, sd (v), d (v), P[v];
    std::queue<long int> Q;

    std::fill(d.begin(), d.end(), -1);

    sd[s] = 1;
    d[s] = 0;

    Q.push(s);

    while(!Q.empty())
    {
      long int x = Q.front();
      Q.pop();

      S.push_back(x);

      for(j=r[x]; j<r[x+1]; j++)
      {
        long int w = c[j];
        // std::cout<<w<<" "<<d[w]<<" "<<sd[w]<<", ";

        if(d[w] < 0)
        {
          Q.push(w);
          d[w] = d[x] + 1;
        }
        // std::cout<<w<<" "<<d[w]<<" "<<sd[w]<<", ";

        if(d[w] == d[x] + 1)
        {
          sd[w] += sd[x];
          P[w].push_back(x);
        }
        // std::cout<<w<<" "<<d[w]<<" "<<sd[w]<<", ";
      }
      // std::cout<<std::endl;
    }

    // for(int k = 0; k<v; k++)
    // {
    //   std::cout<<"d: "<<d[k]<<", sd: "<<sd[k]<<", P: ";
    //   for(std::vector<long int>::iterator it = P[k].begin(); it < P[k].end(); it++)
    //     std::cout<<*it<<" ";
    //   std::cout<<std::endl;
    // }
    // std::cout<<std::endl;

    std::vector<long double> dep (v);

    // for(long int w=0; w<v; w++)
    while(!S.empty())
    {
      long int w = S.back();
      S.pop_back();

      for(std::vector<long int>::iterator it = P[w].begin(); it < P[w].end(); it++)
        dep[*it] += (sd[*it] * (1 + dep[w]) / sd[w]);

      if(w != s)
        c_b[w] += dep[w];
    }

    // for(std::vector<long double>::iterator it = c_b.begin(); it < c_b.end(); it++)
    //   std::cout<<*it<<" ";
    // std::cout<<std::endl<<std::endl;

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
  r = (long int*)malloc((v+1)*sizeof(long int));
  c = (long int*)malloc(2*e*sizeof(long int));

  memcpy(r, csr, (v+1)*sizeof(long int));
  memcpy(c, csr+v+1, 2*e*sizeof(long int));

  free(csr);

  std::vector<long double> c_b(v);

  clock_t t_start = clock();
  calculate_bc(c_b, r, c, v, e);
  clock_t t_end = clock();

  for(std::vector<long double>::iterator it = c_b.begin(); it < c_b.end(); it++)
    *it = *it / 2;

  std::cout<<"BC Values: ";
  for(std::vector<long double>::iterator it = c_b.begin(); it < c_b.end(); it++)
    std::cout<<*it<<" ";

  std::cout<<std::endl<<"Time of execution: "<<(double)(t_end - t_start)/CLOCKS_PER_SEC<<std::endl;
}
