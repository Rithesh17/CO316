// 8. Pack and Unpack

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

struct dd
{
   	char c;
    	int i[2];
    	float f[4];
} inbuf;

int main(int argc, char* argv[])
{
    	int rank, n;
    	
	MPI_Init(&argc, &argv);
    	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    	MPI_Comm_size(MPI_COMM_WORLD, &n);
    
    	char outbuf[100];
    	int position;

    	if (rank == 0)
    	{
        	inbuf.c = 'a';
        	inbuf.i[0] = 1;
        	inbuf.i[1] = 2;
        	inbuf.f[0] = 1.1;
        	inbuf.f[1] = 1.2;
        	inbuf.f[2] = 1.3;
        	inbuf.f[3] = 1.4;
        	printf("\n%d filling inbuf as follows:\n", rank);
        	printf(" c: %c\n i: {%d, %d}\n f: {%f, %f, %f, %f}\n", inbuf.c, inbuf.i[0], inbuf.i[1], inbuf.f[0], inbuf.f[1], inbuf.f[2], inbuf.f[3]);
      	
		position = 0;
        	MPI_Pack(&inbuf.c, 1, MPI_CHAR, outbuf, 100, &position, MPI_COMM_WORLD);
        	MPI_Pack(inbuf.i, 2, MPI_INT, outbuf, 100, &position, MPI_COMM_WORLD);
        	MPI_Pack(inbuf.f, 4, MPI_FLOAT, outbuf, 100, &position, MPI_COMM_WORLD);
		
		for (int i = 1; i < n; i++)
		{
			MPI_Send(&outbuf, 100, MPI_PACKED, i, 0, MPI_COMM_WORLD);
		}
    	}
	else
	{
		MPI_Recv(&outbuf, 100, MPI_PACKED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		position = 0;
    		MPI_Unpack(outbuf, 100, &position, &inbuf.c, 1, MPI_CHAR, MPI_COMM_WORLD);
    		MPI_Unpack(outbuf, 100, &position, inbuf.i, 2, MPI_INT, MPI_COMM_WORLD);
    		MPI_Unpack(outbuf, 100, &position, inbuf.f, 4, MPI_FLOAT, MPI_COMM_WORLD);
    
    		printf("\n%d received outbuf and unpacked inbuf as follows:\n c: %c\n i: {%d, %d}\n f: {%f, %f, %f, %f}\n", rank, inbuf.c, inbuf.i[0], inbuf.i[1], inbuf.f[0], inbuf.f[1], inbuf.f[2], inbuf.f[3]);
	}

    	MPI_Finalize();
    	return 0;
}
