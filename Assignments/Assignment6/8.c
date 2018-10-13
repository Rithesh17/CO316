// 8. Pack and Unpack

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

struct dd
{
   	char c;
    	int i[2];
    	float f[4];
} s, inbuf1, inbuf2;

int main(int argc, char* argv[])
{
    	int rank, n;
    	
	MPI_Init(&argc, &argv);
    	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    	MPI_Comm_size(MPI_COMM_WORLD, &n);
    
    	char outbuf1[100], outbuf2[100];
    	int position;

    	if (rank == 0)
    	{
        	inbuf1.c = 'a';
        	inbuf1.i[0] = 1;
        	inbuf1.i[1] = 2;
        	inbuf1.f[0] = 1.1;
        	inbuf1.f[1] = 1.2;
        	inbuf1.f[2] = 1.3;
        	inbuf1.f[3] = 1.4;
        	printf("\n%d filling inbuf1 as follows:\n", rank);
        	printf(" c: %c\n i: {%d, %d}\n f: {%f, %f, %f, %f}\n", inbuf1.c, inbuf1.i[0], inbuf1.i[1], inbuf1.f[0], inbuf1.f[1], inbuf1.f[2], inbuf1.f[3]);
      	
		position = 0;
        	MPI_Pack(&inbuf1.c, 1, MPI_CHAR, outbuf1, 100, &position, MPI_COMM_WORLD);
        	MPI_Pack(inbuf1.i, 2, MPI_INT, outbuf1, 100, &position, MPI_COMM_WORLD);
        	MPI_Pack(inbuf1.f, 4, MPI_FLOAT, outbuf1, 100, &position, MPI_COMM_WORLD);
		MPI_Send(&outbuf1, 100, MPI_PACKED, 1, 0, MPI_COMM_WORLD);
    	}
	else
	{
		MPI_Recv(&outbuf1, 100, MPI_PACKED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		position = 0;
    		MPI_Unpack(outbuf1, 100, &position, &inbuf1.c, 1, MPI_CHAR, MPI_COMM_WORLD);
    		MPI_Unpack(outbuf1, 100, &position, inbuf1.i, 2, MPI_INT, MPI_COMM_WORLD);
    		MPI_Unpack(outbuf1, 100, &position, inbuf1.f, 4, MPI_FLOAT, MPI_COMM_WORLD);
    
    		printf("\n%d received outbuf1 and unpacked inbuf1 as follows:\n c: %c\n i: {%d, %d}\n f: {%f, %f, %f, %f}\n", rank, inbuf1.c, inbuf1.i[0], inbuf1.i[1], inbuf1.f[0], inbuf1.f[1], inbuf1.f[2], inbuf1.f[3]);
	}

    	MPI_Finalize();
    	return 0;
}
