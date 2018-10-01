#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
	  MPI_Init(NULL, NULL);
	  int N=10;
	  int size;
	  MPI_Comm_size(MPI_COMM_WORLD, &size);
	  int rank;
	  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	  int my_init,my_chunk;
	  if(rank < (N%size){
	  	my_chunk= N+(size-1)/size;
		my_init=rank*my_chunk;
 	  }else{
		  my_chunk=N/size;
		  my_init=rank*(N/size)+(N%size);
	  }
	  printf("%d: init  %d chunk %d  end %d\n",rank,my_init, my_chunk, (my_init+my_chunk));
	  for(int i=my_init; i< (my_init+my_chunk); i++)
		  printf("%d: iter %d\n",rank,i);
	  MPI_Finalize();
}
